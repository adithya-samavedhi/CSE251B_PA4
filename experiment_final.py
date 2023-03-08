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
import nltk
import string
import matplotlib.pyplot as plt
import os

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
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.__criterion =  nn.CrossEntropyLoss().to(self.__device)
        self.__optimizer =  optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])
        self.generation_config = config_data['generation']
        
        self.__max_length = config_data['generation']['max_length']
        self.__stochastic = not config_data['generation']['deterministic']
        self.__temperature = config_data['generation']['temperature']
        self.__is_early_stop = config_data["early_stop"]["is_early_stop"]
        self.__early_stop_epoch = config_data["early_stop"]['early_stopping_rounds']

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            print("Loaded saved model")
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        #Early Stop
        if self.__is_early_stop:
            patience = self.__early_stop_epoch
            best_loss = 1e9
            best_iter = 0
            
        start_epoch = self.__current_epoch
        for epoch in tqdm(range(start_epoch, self.__epochs)):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            print(f"Epoch {epoch+1} loss is {train_loss} perplexity is {np.exp(train_loss)}")
            val_loss = self.__val()
            print(f"Epoch {epoch+1} Validation loss is {val_loss} perplexity is {np.exp(val_loss)}")
            
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            
            #Check for Early Stopping
            if self.__is_early_stop:
                best_loss, best_iter, patience = self.early_stopping(epoch, self.__early_stop_epoch, best_loss,
                                                                    best_iter, val_loss, patience)
                print(f"Patience = {patience}")
                if patience==0:
                    print(f"Training stopped early at epoch:{epoch}, best_loss = {best_loss}, best_iteration={best_iter}")
                    break    
            else:
                self.__save_model()

    def __train(self):
        vocab_size = len(self.__vocab)
        self.__model.train()
        training_loss = []

        print(self.__device)
        # Iterate over the data, implement the training function
        for i, (images, captions, _) in tqdm(enumerate(self.__train_loader)):
            # both inputs and labels have to reside in the same device as the model's
            images = images.to(self.__device)
            captions = captions.to(self.__device)
            outputs =  self.__model.forward(images, captions[:,:-1].cuda())
            
            loss = self.__criterion(torch.flatten(outputs, start_dim=0, end_dim=1),
                                    torch.flatten(captions, start_dim=0, end_dim=1))
    
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            training_loss.append(loss.item())

        return np.mean(training_loss)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        vocab_size = len(self.__vocab)
        self.__model.eval()
        validation_loss = []
        bleu1_val = 0
        bleu4_val = 0

        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__val_loader):
            # both inputs and labels have to reside in the same device as the model's
            images =  images.to(self.__device)
            captions =   captions.to(self.__device)

            outputs =  self.__model.forward(images, captions[:,:-1])

            loss = self.__criterion(torch.flatten(outputs, start_dim=0, end_dim=1),
                                    torch.flatten(captions, start_dim=0, end_dim=1))
            validation_loss.append(loss.item())
            
            if i%100 == 0:
                    print(i)
                    print(captions[0,:].shape)
                    print( self.vec_to_words(captions[0,:]) )
                    print( self.vec_to_words(pred_captions.argmax(dim=2)[0,:]) )
                    print( self.vec_to_words(generate_captions[0,:]) )

        return np.mean(validation_loss)
        


    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        print("Testing")
        vocab_size = len(self.__vocab)
        state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
        self.__model.load_state_dict(state_dict['model'])
        self.__optimizer.load_state_dict(state_dict['optimizer'])
        
        self.__model.eval()
        bleu1_val = 0
        bleu4_val = 0
        test_loss = 0
        
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__train_loader):
                images = images.to(self.__device)
                captions = captions.to(self.__device)
                
                ground_captions = [[i['caption'] for i in self.__coco_test.imgToAnns[idx]] for idx in img_ids]

                
                outputs =  self.__model.forward(images, captions[:,:-1])

                loss = self.__criterion(torch.flatten(outputs, start_dim=0, end_dim=1),
                                    torch.flatten(captions, start_dim=0, end_dim=1))
                test_loss += loss.item()
    
                generate_captions = self.__model.generate_final(images, max_length=self.__max_length,
                                                            stochastic=self.__stochastic, temp=self.__temperature)
                bleu1_val += self.calc_bleu1(ground_captions, generate_captions)
                bleu4_val += self.calc_bleu4(ground_captions, generate_captions)
                
                if iter % 100 ==0:
                    print(loss.item(),bleu1_val/(iter+1),bleu4_val/(iter+1))
                    print(captions[0,:].shape)
                    self.plot_images(images,captions,generate_captions)
                    print( self.vec_to_words(captions[0,:]) )
#                     print( self.vec_to_words(pred_captions.argmax(dim=2)[0,:]) )
                    print( self.vec_to_words(generate_captions[0,:]) )
                    #print(f" generated captions: {generate_captions[:3]}")
                    #print(f" ground captions: {ground_captions[:3]}")
                    
                
        test_loss = test_loss / len(self.__test_loader)
        bleu1_val = bleu1_val / len(self.__test_loader)
        bleu4_val = bleu4_val / len(self.__test_loader)
        
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss, bleu1_val, bleu4_val)

        self.__log(result_str)

        return test_loss, bleu1_val, bleu4_val

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
        
    def calc_bleu1(self, captions, generate_captions):
        reference_captions = []
        predicted_captions = []
        for vec in captions:
            reference_captions.append(self.sequences_to_words(vec))
        for vec in generate_captions:
            predicted_captions.append(self.vec_to_words(vec))
        bleu_value = 0
        for i in range(len(reference_captions)):
            bleu_value += bleu1( reference_captions[i] , predicted_captions[i])
        bleu_value /= len(reference_captions)
        return bleu_value
    
    def calc_bleu4(self, captions, generate_captions):
        reference_captions = []
        predicted_captions = []
        for vec in captions:
            reference_captions.append(self.sequences_to_words(vec))
        for vec in generate_captions:
            predicted_captions.append(self.vec_to_words(vec))
        bleu_value = 0
        for i in range(len(reference_captions)):
            bleu_value += bleu4( reference_captions[i] , predicted_captions[i])
        
        bleu_value /= len(reference_captions)
        return bleu_value
    
    def sequences_to_words(self, captions):
        words = []
        for i in captions:
            caption = nltk.word_tokenize(i)
            tmp = []
            for word in caption:
                if word not in string.punctuation:
                    tmp.append(word.lower())
            words.append(tmp)
        return words
    
    def vec_to_words(self, captions):
        words = []
        for i in captions:
            if i > 3:
                word = self.__vocab.idx2word[i.item()].lower()
                if word not in string.punctuation:
                    words.append(self.__vocab.idx2word[i.item()].lower())
            if i == 2:
                break
        return words
    
    def bleu1(self, reference_captions, predicted_caption):
        return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)


    def bleu4(self, reference_captions, predicted_caption):
        return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)
    
    def early_stopping(self, iter_num, early_stopping_rounds, best_loss, best_iter, loss, patience):
        """
        Implements the early stopping functionality with a loss monitor. If the patience is exhausted it interupts the training process and 
        returns the best model and its corresponding loss and accuracy score on validation data.
        Parameters
        ----------
        iter_num: Current epoch number.
        early_stopping_rounds: User specified hyperparameters that indicates the patience period upper limit.
        best_loss: Best validation set loss observed till the current iteration.
        best_iter: Iteration number of best validation loss (best_loss)
        loss: Current iteration loss on validation data.
        patience: Current patience level. If best_loss is not beaten then patience will be decremented by 1.
        Returns
        -------
        best_loss: Updated best loss after iteration iter_num.
        best_iter: Best iteration till iter_num.
        patience: Updated patience value.
        """
        if loss>=best_loss:
            patience-=1
        else:
            self.__save_model()
            patience = early_stopping_rounds
            best_loss = loss
            best_iter = iter_num

        return best_loss, best_iter, patience
    
    def plot_images(self,images,captions,generate_captions):
        images = images[:10]
        captions = captions[:10]
        generate_captions = generate_captions[:10]
        
        for i,image in enumerate(images):
            plt.imshow(image.permute(1, 2, 0).cpu().numpy())
            plt.savefig(self.__experiment_dir+f'/image_{i}.png')
            caption = self.vec_to_words(caption)
            write_to_file_in_dir(self.__experiment_dir,f"image_{str(i)}_ground_caption.txt",caption)
            
#             for j,caption in enumerate(captions):
#                 caption = self.vec_to_words(caption)
#                 write_to_file_in_dir(self.__experiment_dir,f"image_{str(i)}_ground_caption_{j}.txt",caption)

            for j,caption in enumerate(generate_captions):
                caption = self.vec_to_words(caption)
                write_to_file_in_dir(self.__experiment_dir,f"image_{str(i)}_generated_caption_{j}.txt",caption)
            
        