import torch.nn as nn
import pickle
from torchvision import transforms, models

#ToDO Fill in the __ values
class TransferLearningResNet34(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        resnet50 = models.resnet50(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        self.linear = nn.Linear(hidden_dim, 512)
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, 512, 2)
        self.output_linear = nn.Linear(512, vocab_size)
        self.softmax = nn.Softmax(dim=1)

#TODO Complete the forward pass
    def forward(self, x):
        # x1 = self.bnd1(self.relu(self.conv1(x)))
        # # Complete the forward function for the rest of the encoder
        # x2 = self.bnd2(self.relu(self.conv2(x1)))
        # x3 = self.bnd3(self.relu(self.conv3(x2)))
        # x4 = self.bnd4(self.relu(self.conv4(x3)))
        # x5 = self.bnd5(self.relu(self.conv5(x4)))

        
        x = self.resnet50(x)
        x = self.linear(x)
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.output_linear(x)
        x = self.softmax(x)

        return x




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