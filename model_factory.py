# Build and return the model here based on the configuration.
from baseline import TransferLearningResNet34
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    model = TransferLearningResNet34(4, hidden_size,vocab, embedding_size)

    return model
