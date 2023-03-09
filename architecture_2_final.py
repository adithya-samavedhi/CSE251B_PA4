import torch.nn as nn
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torchvision import transforms, models

#ToDO Fill in the __ values
class Architecture2(nn.Module):

    def __init__(self, n_class, hidden_dim, vocab, embedding_size, num_layers, model_type):

        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_size = embedding_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type


        resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        num_features = resnet50.fc.in_features
        
        for param in self.resnet50.parameters():
            param.requires_grad = False
            
        self.linear = nn.Linear(num_features, self.embed_size)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        if self.model_type=="RNN":
            self.decoder_unit = nn.RNN(input_size=self.embed_size*2, hidden_size=self.hidden_dim, 
                                       num_layers=self.num_layers, batch_first=True)
        else:
            self.decoder_unit = nn.LSTM(input_size=self.embed_size*2, hidden_size=self.hidden_dim,
                                        num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.bn = nn.BatchNorm1d(self.embed_size)
    
    def forward(self, input, captions=None):
        """
        x: (batch, 3, 256, 256)
        captions: (batch, max_seq_length, vocab_size)
        
        Returns:
        outputs: (batch, max_seq_length, vocab_size)
        """
        seq_len = captions.size(1)
        outputs = torch.zeros((input.size(0), seq_len, self.vocab_size)).to(self.device)
        
        #Encoder
        x = self.resnet50(input)  # shape (batch_size, 512, 1, 1)
        x = x.view(input.size(0),-1)  # shape (batch_size, 512)
        x = self.bn(self.linear(x))  # shape (batch_size, 256)
        x = x.view(input.size(0), 1, -1)
        
        #Concatenation operation
        x = x.unsqueeze(1).expand(-1, captions.shape[1], -1)
        x_cat = torch.cat((x, captions), dim=2)
        
        hiddens, c = self.decoder_unit(x)
        outputs[:,0,:] = self.fc(hiddens).squeeze()

        embeddings = self.embedding(captions)
        for t in range(seq_len-1):
            embed_token = self.embedding(outputs[:, t:t+1])
            x_cat = torch.cat((x, embed_token), dim=2)
            hiddens, c = self.decoder_unit(x_cat, c)
            outputs[:, t+1, :] = self.fc(hiddens).squeeze()
            
        return outputs
    
    def generate_final(self, input, max_length=100, stochastic = False, temp=0.1):
        outputs = torch.zeros((input.size(0), max_length), dtype=torch.long).to(self.device)
        
        def sampling(output, t):
            if stochastic:
                p = F.softmax(output/temp, dim=-1).squeeze()
                m = Categorical(p)
                outputs[:,t] = m.sample()
            else:
                outputs[:,t] = torch.argmax(output, dim=2).view(-1)
        
        #Encoder
        x = self.resnet50(input)  # shape (batch_size, 512, 1, 1)
        x = x.view(input.size(0),-1)  # shape (batch_size, 512)
        x = self.bn(self.linear(x))  # shape (batch_size, 256)
        x = x.view(input.size(0), 1, -1)
        
        x_cat = torch.cat((x, self.embedding(outputs[:, 0]).unsqueeze(1)), dim=2)
        
        #Decoder
        hiddens, c = self.decoder_unit(x_cat)
        output = self.fc(hiddens)
        sampling(output, 0)

        for t in range(max_length-1):
            embed_token = self.embedding(outputs[:, t:t+1])
            x_cat = torch.cat((x, embed_token), dim=2)
            hiddens, c = self.decoder_unit(embed_token, c)
            output = self.fc(hiddens)
            sampling(output, t+1)

        return outputs