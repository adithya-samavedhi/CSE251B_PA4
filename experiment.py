import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import torch.nn as nn

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None # Save your best model in this field and use this in test method.
        
        self.__inference_max_len = config_data['generation']['max_length']

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.device =   torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

        self.__criterion =  nn.CrossEntropyLoss().to(self.device)
        self.__optimizer =  optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load Experiment Data if available
#         self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        import warnings
        warnings.filterwarnings(action="ignore", category=UserWarning)
        start_epoch = self.__current_epoch
        for epoch in tqdm(range(start_epoch, self.__epochs)):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            print(f"Epoch {epoch+1} loss is {train_loss} perplexity is {np.exp(train_loss)}")
            val_loss = self.__val()
            print(f"Epoch {epoch+1} Validation loss is {val_loss} perplexity is {np.exp(val_loss)}")
            
#             self.__record_stats(train_loss, val_loss)
            # self.__log_epoch_stats(start_time)
            self.__save_model()

    def __train(self):
        vocab_size = len(self.__vocab)
        self.__model.train()
        training_loss = []

        # Iterate over the data, implement the training function
        for i, (images, captions, _) in tqdm(enumerate(self.__train_loader)):
            self.__optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs =  images.to(self.device)
            labels=captions
            labels =   labels.to(self.device)
            outputs =  self.__model.forward(inputs, labels)

            
#             loss =  self.__criterion(outputs.view(-1, vocab_size), labels.contiguous().view(-1))
            loss = self.__criterion(torch.flatten(outputs, start_dim=0, end_dim=1),
                                    torch.flatten(labels, start_dim=0, end_dim=1))
    
            loss.backward()
            self.__optimizer.step()
            training_loss.append(loss.item())

        
        return np.mean(training_loss)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        vocab_size = len(self.__vocab)
        self.__model.train()
        validation_loss = []

        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs =  images.to(device)
            labels =   captions.to(device)
            
          
            outputs =  self.__model.forward(inputs, labels)

            
            loss =  self.__criterion(outputs.view(-1, vocab_size), labels.contiguous().view(-1))
            validation_loss.append(loss.item())

        
        return np.mean(validation_loss)
        


    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        vocab_size = len(self.__vocab)
        state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
        self.__model.load_state_dict(state_dict['model'])
        self.__optimizer.load_state_dict(state_dict['optimizer'])
        
        self.__model.eval()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0
        device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_loss = []
        
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                inputs = images.to(device)
                captions = captions.to(device)
                ground_captions = [[i['caption'] for i in self.__coco_test.imgToAnns[idx]] for idx in img_ids]
                
                inputs =  images.to(device)
                labels =   captions.to(device)



                outputs =  self.__model.forward(inputs, labels).to(device) 


                loss =  self.__criterion(outputs.view(-1, vocab_size), labels.contiguous().view(-1))
            
                generate_captions = self.__model.generate_caption(inputs,deterministic=False,temperature=0.1)
                
                
                test_loss.append(loss.item())

                bleu1_val += self.calc_bleu1(ground_captions, generate_captions)
                bleu4_val += self.calc_bleu4(ground_captions, generate_captions)
                
                
        final_test_loss = np.mean(test_loss)
        bleu1_val = bleu1_val / len(self.__test_loader)
        bleu4_val = bleu4_val / len(self.__test_loader)


        result_str = "Test Performance: Loss: {}, Perplexity: {}, Bleu1: {}, Bleu4: {}".format(final_test_loss,
                                                                                               bleu1,
                                                                                               bleu4)
        self.__log(result_str)

        return final_test_loss, bleu1, bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
