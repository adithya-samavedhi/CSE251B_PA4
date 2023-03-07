# Build and return the model here based on the configuration.
from baseline import TransferLearningResNet50
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    num_layers = config_data['model']['num_layers']

    # You may add more parameters if you want
    model = TransferLearningResNet50(4, hidden_size,vocab, embedding_size, num_layers, model_type)

    return model
