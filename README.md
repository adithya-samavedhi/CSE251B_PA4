# Image Captioning

## Usage

* Define the configuration for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- main.py: Main driver class
- experiment_final.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments. Implemented the training and testing functions.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace
- architecture_1_final.py: Contains the architecture for the architecture 1 mentioned in the assignment. Also contains the code for generation used during testing.
- architecture_2_final.py: Contains the architecture for the architecture 2 mentioned in the assignment. Also contains the code for generation used during testing.
- default.json: Contains the config of the hyperparameters used in our experiments.
- get_datasets.ipynb: Contains the code to create the dataset.
- coco_dataset.py: Contains the helper class to convert our data to coco format.
- test_ids.csv, train_ids.csv, val_ids.csv: Contains the image ids for test, train and val datasets respectively.

