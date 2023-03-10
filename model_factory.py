# Build and return the model here based on the configuration.
from baseline import TransferLearningResNet50
from architecture_2_final import Architecture2
from architecture_1_final import Architecture1

def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    num_layers = config_data['model']['num_layers']
    architecture = config_data['model']['architecture']

    print(architecture, model_type)
    # You may add more parameters if you want
    
    if architecture=="architecture 1":
        model = Architecture1(len(vocab), hidden_size, embedding_size, num_layers, model_type)
    else:
        model = Architecture2( hidden_size, vocab, embedding_size, num_layers, model_type)

    return model
