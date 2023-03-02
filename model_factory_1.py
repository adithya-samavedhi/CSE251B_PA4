# Build and return the model here based on the configuration.
class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size=200, num_layers=1):
        super(CaptioningModel, self).__init__()
        
        resnet50 = models.resnet50(pretrained=True)
        num_features = resnet50.fc.in_features
        
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        for param in resnet50.parameters():
            param.requires_grad = False
            
        self.linear = nn.Linear(num_features, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.embed = nn.Embedding(vocab_size, embed_size)
                
        
    def forward(self, images):
        features = self.resnet50(images)
        features = features.reshape(features.size(0),-1)
        features = self.linear(features)
        features = features.view(features.size(0), -1)
        
        inputs = features.unsqueeze(1)
        hiddens = None
        outputs = []
        for i in range(max_caption_length):
            hiddens, lstm_states = self.lstm(inputs, hiddens)

            outputs_t = self.linear(hiddens.squeeze(1))

            _, predicted = outputs_t.max(1)
            outputs.append(predicted)

            inputs = self.embed(predicted).unsqueeze(1)
                
            outputs = torch.stack(outputs, dim=1)
            
        outputs = torch.stack(outputs, dim=1)
            
        return outputs

def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want

    model = CaptioningModel(embedding_size, hidden_size, num_layers=2)

    return model
