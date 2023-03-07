import torch.nn as nn
import pickle
import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms, models

#ToDO Fill in the __ values
class TransferLearningResNet34(nn.Module):

    def __init__(self, n_class, hidden_dim, vocab, embedding_size):

        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_size = embedding_size
        self.hidden_dim = hidden_dim
        
        print(self.vocab_size)

        resnet34 = models.resnet34(pretrained=True)
        self.resnet34 = nn.Sequential(*list(resnet34.children())[:-1])
        for param in self.resnet34.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(512, self.embed_size)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x, captions=None):
        x = self.resnet34(x)  # shape (batch_size, 512, 1, 1)
        x = x.squeeze()  # shape (batch_size, 512)
        x = self.linear(x)  # shape (batch_size, 256)

        embeddings = self.embedding(captions)
        
        image_features = x.unsqueeze(1)
        embeddings = torch.cat((image_features, embeddings[:, :-1,:]), dim=1)
        
        hiddens, c = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        return outputs
    
        ## Greedy search 
    def generate_caption(self, inputs, deterministic=False, temperature=0.1):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        x = self.resnet34(inputs)  # shape (batch_size, 512, 1, 1)
        x = x.squeeze()  # shape (batch_size, 512)
        x = self.linear(x)
        image_features = x.unsqueeze(1)
        
        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
#         hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
    
        while True:
            lstm_out, hidden = self.lstm(image_features) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.fc(lstm_out)  # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
            
            output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted
            
            if ("<end>" == self.vocab.idx2word[max_indice.cpu().numpy()[0].item()]):
                # We predicted the <end> word, so there is no further prediction to do
                break
            
            ## Prepare to embed the last predicted word to be the new input of the lstm
            inputs = self.embedding(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            
        return output




    def save(self, filepath="./model.pkl"):
        """
        Saves the Network model in the given filepath.
        Parameters
        ----------
        filepath: filepath of the model to be saved
        Returns
        -------
        None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath="./model.pkl"):
        """
        Loads a pre-trained Network model from the given filepath.
        Parameters
        ----------
        filepath: filepath of the model to be loaded
        Returns
        -------
        model: Loaded Network model
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)