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


    def forward(self, x, captions=None):
        x = self.resnet50(x)  # shape (batch_size, 512, 1, 1)
        x = x.reshape(x.size(0), -1)  # shape (batch_size, 512)
        x = self.bn(self.linear(x))  # shape (batch_size, 256)

        #x = x.view((x.shape[0], 1, x.shape[1]))
        captions = torch.cat((torch.zeros(captions.shape[0],1,dtype= torch.long).cuda(),captions),dim=1)
        captions = self.embedding(captions)
        
        #Concatenation operation
        x = x.unsqueeze(1).expand(-1, captions.shape[1], -1)
        x = torch.cat((x, captions), dim=2)
       
        hiddens, c = self.decoder_unit(x)
        outputs = self.fc(hiddens)
        return outputs
    
    def generate_final(self, x, max_length=100, stochastic = False, temp=0.1):
        x = self.resnet50(x)  # shape (batch_size, 512, 1, 1)
        x = x.reshape(x.size(0), -1)  # shape (batch_size, 512)
        x = self.bn(self.linear(x))  # shape (batch_size, 256)

        pred = torch.zeros((x.size(0), max_length), dtype=torch.long).cuda()
        x = x.unsqueeze(1).expand(-1, 1, -1)

        x_cat = torch.cat((x, self.embedding(pred[:, 0]).unsqueeze(1)), dim=2)
        states = None

        for t in range(max_length):
            hiddens, states = self.decoder_unit(x_cat, states) # output dimension?
            outputs = self.fc(hiddens)

            if stochastic:
                outputs = F.softmax(outputs/temp, dim=-1).reshape(outputs.size(0),-1)
                # batch_size * vocab_size
                outputs = Categorical(outputs) 
                pred[:,t] = outputs.sample()
#                 pred[:, t] = torch.multinomial(outputs.data, 1).view(-1)
            else:
                #deterministic
                pred[:, t] = torch.argmax(outputs, dim=2).view(-1)

            x_cat = torch.cat((x, self.embedding(pred[:, t]).unsqueeze(1)), dim=2)

        return pred