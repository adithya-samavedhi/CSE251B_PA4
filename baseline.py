import torch.nn as nn
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, models

#ToDO Fill in the __ values
class TransferLearningResNet34(nn.Module):

    def __init__(self, n_class, hidden_dim, vocab_size):
        # super().__init__()
        # self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        # self.n_class = n_class
        # resnet50 = models.resnet50(pretrained=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        # self.linear = nn.Linear(self.hidden_dim, 512)
        # self.embedding = nn.Embedding(self.vocab_size, 300)
        # self.lstm = nn.LSTM(300, 512, 2)
        # self.output_linear = nn.Linear(512, self.vocab_size)
        # self.softmax = nn.Softmax(dim=1)

        super().__init__()
        self.vocab_size = vocab_size
        resnet34 = models.resnet34(pretrained=True)
        self.resnet34 = nn.Sequential(*list(resnet34.children())[:-1])
        self.linear = nn.Linear(512, 256)
        self.embedding = nn.Embedding(self.vocab_size, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x, captions=None):
        x = self.resnet34(x)  # shape (batch_size, 512, 1, 1)
        x = x.squeeze()  # shape (batch_size, 512)
        x = self.linear(x)  # shape (batch_size, 256)

    # Decode the image to generate captions
        if captions is not None:
            # Embed the captions
            captions = self.embedding(captions)  # shape (batch_size, max_caption_length, 256)

            # Concatenate the encoded image with the caption embeddings
            x = x.unsqueeze(1)  # shape (batch_size, 1, 256)
            x = x.repeat(1, captions.size(1), 1)  # shape (batch_size, max_caption_length, 256)
            # inputs = torch.cat([x, captions], dim=2)  # shape (batch_size, max_caption_length, 512)
            inputs = captions

            # Pass the concatenated inputs through the LSTM
            hiddens, _ = self.lstm(inputs)  # shape (batch_size, max_caption_length, 256)

            # Project the LSTM outputs to the vocabulary space
            outputs = self.fc(hiddens)  # shape (batch_size, max_caption_length, vocab_size)
            outputs = self.softmax(outputs)  # shape (batch_size, max_caption_length, vocab_size)

            return outputs

        else:
            return x
        
        # x = self.resnet50(x)
        # x = self.linear(x)
        # x = self.embedding(x)
        # x = self.lstm(x)
        # x = self.output_linear(x)
        # x = self.softmax(x)

        # return x




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