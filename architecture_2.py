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
        
        print(self.vocab_size)

        resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(2048, self.embed_size)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        if self.model_type=="RNN":
            self.decoder_unit = nn.RNN(input_size=self.embed_size*2, hidden_size=self.hidden_dim, 
                                       num_layers=self.num_layers, batch_first=True)
        else:
            self.decoder_unit = nn.LSTM(input_size=self.embed_size*2, hidden_size=self.hidden_dim,
                                        num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x, captions=None):
        x = self.resnet50(x)  # shape (batch_size, 512, 1, 1)
        x = x.squeeze()  # shape (batch_size, 512)
        x = self.linear(x)  # shape (batch_size, 256)

#         print(f"captions: {captions}")
        captions = torch.cat((torch.zeros(captions.shape[0],1).cuda(),captions),dim=1)
        embeddings = self.embedding(captions)
        image_features =  x.unsqueeze(1).expand(-1, embeddings.shape[1], -1)
        embeddings = torch.cat((image_features, embeddings), dim=2)
       
        hiddens, c = self.decoder_unit(embeddings)
        outputs = self.fc(hiddens)
        return outputs
    
        ## Greedy search 
    def generate_caption(self, img, generation_config):
        temperature = generation_config['temperature']
        max_length = generation_config['max_length']
        deterministic = generation_config['deterministic']
 
        batch_size = img.shape[0]
    
        x = self.resnet50(img)  # shape (batch_size, 512, 1, 1)
        x = x.squeeze()  # shape (batch_size, 512)
        x = self.linear(x)
        
        # TODO: opertions on text_lists is not very efficient, consider using index instead of string
        text_lists = [['kid'] for _ in range(batch_size)]
 
        while (not all([text_list[-1] == '<end>' for text_list in text_lists])) and all([len(text_list) <= max_length for text_list in text_lists]):
            text = torch.tensor([[self.vocab(text_list[-1])]
                                 for text_list in text_lists], dtype=torch.long).to(img.device)
            
            embeddings = self.embedding(text)  # batch_size, seq_len, embedding_size
            image_features =  x.unsqueeze(1).expand(-1, embeddings.shape[1], -1)
            
            print(f"image_features: {image_features}")
            
#             decode, (hidden, cell) = self.decoder(text_embedded, (hidden, cell))  # seq_len, batch_size, hidden_size
#             decode = decode.permute(1, 0, 2)  # batch_size, seq_len, hidden_size
#             out = self.output(decode).permute(0, 2, 1)  # batch_size, vocab_size, seq_len
            embeddings = torch.cat((image_features, embeddings), dim=2)
            hiddens, c = self.decoder_unit(embeddings)
            out = self.fc(hiddens)
            print(f"out: {out.squeeze_()}")
            
 
            if deterministic:
                print(f"softmax shape: {F.softmax(out.squeeze_(), dim=1).shape}")
                print(f"argmax shape: {torch.argmax(F.softmax(out.squeeze_(), dim=1), dim=1)}")
                _, text_ids = F.softmax(out.squeeze_(), dim=1).max(dim=1)
#                 print(f"text_ids: {text_ids}")
            else:
#                 print(f"softmax shape: {F.softmax(out.squeeze_() / temperature, dim=1).shape}")
                text_ids = Categorical(F.softmax(out.squeeze_() / temperature, dim=1)).sample()
                
            for text_list, text_id in zip(text_lists, text_ids):
                if text_list[-1] != '<end>':
                    text_list.append(self.vocab.idx2word[int(text_id.item())])
                    print(f"predicted_word: {self.vocab.idx2word[int(text_id.item())]}")

        text_lists = [[text for text in text_list if text != '<pad>' and text !=
                       '<start>' and text != '<end>' and text != '<unk>'] for text_list in text_lists]
 
        return text_lists




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