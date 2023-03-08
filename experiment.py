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
        self.__test_caption = config_data['dataset']['test_annotation_file_path']
        self.__train_caption = config_data['dataset']['training_annotation_file_path']
        self.__coco_test = COCO(self.__test_caption)
        self.__coco_train = COCO(self.__train_caption)
        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__early_stop = config_data['experiment']['early_stop']
        self.__patience = config_data['experiment']['patience']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__val_bleu1 = []
        self.__val_bleu4 = []
        self.__best_model = (None, None)  # Save your best model in this field and use this in test method.
        self.__best_bleu = 0  # Use this criterion to update best model and perform early stopping
        np.random.seed(config_data['experiment']['random_seed'])
 
        # Generation hyperparameters
        self.__max_length = config_data['generation']['max_length']
        self.__stochastic = not config_data['generation']['deterministic']
        self.__temperature = config_data['generation']['temperature']
 
        # Init Model
        encoder, decoder = get_model(config_data, self.__vocab)
        self.__encoder = encoder
        self.__decoder = decoder
 
        self.__criterion = torch.nn.CrossEntropyLoss().cuda()
        self.__optimizer = torch.optim.Adam(itertools.chain(self.__encoder.parameters(), self.__decoder.parameters()),
                                            lr=config_data["experiment"]["learning_rate"])
 
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
 
            state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model.pt'))
            self.__encoder.load_state_dict(state_dict['encoder_state'])
            self.__decoder.load_state_dict(state_dict['decoder_state'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__encoder = self.__encoder.cuda().float()
            self.__decoder = self.__decoder.cuda().float()
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
            
            self.__record_stats(train_loss, val_loss)
            # self.__log_epoch_stats(start_time)
            self.__save_model()

    def __train(self):
        self.__encoder.eval()
        self.__decoder.train()
        training_loss = 0
        # index, image, target, image_id
        for i, (images, captions, img_ids, length) in enumerate(tqdm(self.__train_loader)):
            captions = captions.cuda()
            encoder_output = self.__encoder(images.cuda())
            pred_captions = self.__decoder(encoder_output, captions[:, :-1], length)
            loss = self.__criterion(torch.flatten(pred_captions, start_dim=0, end_dim=1),
                                    torch.flatten(captions, start_dim=0, end_dim=1))
            
            training_loss += loss.sum().item()
            
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
 
        training_loss = training_loss / len(self.__train_loader)
 
        return training_loss
 
    def __val(self):
        self.__encoder.eval()
        self.__decoder.eval()
        val_loss = 0
        bleu1_val = 0
        bleu4_val = 0
        
        with torch.no_grad():
            for i, (images, captions, img_ids, length) in enumerate(tqdm(self.__val_loader)):
                captions = captions.cuda()
                encoder_output = self.__encoder(images.cuda())
                ground_captions = [[i['caption'] for i in self.__coco_train.imgToAnns[idx]] for idx in img_ids]
                pred_captions = self.__decoder(encoder_output, captions[:,:-1], length)
                loss = self.__criterion(torch.flatten(pred_captions, start_dim=0, end_dim=1),
                                        torch.flatten(captions, start_dim=0, end_dim=1))
                generate_captions = self.__decoder.generate(encoder_output, max_length=self.__max_length,
                                                            stochastic=self.__stochastic, temp=self.__temperature)
                
                if i%100 == 0:
                    print(i)
                    print(captions[0,:].shape)
                    print( self.vec_to_words(captions[0,:]) )
                    print( self.vec_to_words(pred_captions.argmax(dim=2)[0,:]) )
                    print( self.vec_to_words(generate_captions[0,:]) )
                    
                
                
                val_loss += loss.sum().item()
                bleu1_val += self.calc_bleu1(ground_captions, generate_captions)
                bleu4_val += self.calc_bleu4(ground_captions, generate_captions)
 
        val_loss = val_loss / len(self.__val_loader)
        bleu1_val = bleu1_val / len(self.__val_loader)
        bleu4_val = bleu4_val / len(self.__val_loader)
 
        return val_loss, bleu1_val, bleu4_val
 
    def test(self):
        self.__encoder.eval()
        self.__decoder.eval()
        print('Testing')
        
        test_loss = 0
        bleu1_val = 0
        bleu4_val = 0
        with torch.no_grad():
            for iter, (images, captions, img_ids, length) in enumerate(self.__test_loader):
                captions = captions.cuda()
                ground_captions = [[i['caption'] for i in self.__coco_test.imgToAnns[idx]] for idx in img_ids]
                encoder_output = self.__encoder(images.cuda())
                pred_captions = self.__decoder(encoder_output, captions[:,:-1], length)
                generate_captions = self.__decoder.generate(encoder_output, max_length=self.__max_length,
                                                            stochastic=self.__stochastic, temp=self.__temperature)
                loss = self.__criterion(torch.flatten(pred_captions, start_dim=0, end_dim=1),
                                        torch.flatten(captions, start_dim=0, end_dim=1))
                test_loss += loss.sum().item()
 
                bleu1_val += self.calc_bleu1(ground_captions, generate_captions)
                bleu4_val += self.calc_bleu4(ground_captions, generate_captions)
 
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
