'''
This module is implemented to run the whole
model pipeline on local, if desired. It
includes options to train the model from zero
or load it from the repository.
'''

import joblib
import torch
from transformers import AutoModelForSequenceClassification
from src import ROOT_PATH, PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH
from src.models.train_model import stemming, preprocess_and_tokenize_data, training, DataLoader
from src.models.test_model import score_function
from src.data.get_and_save_data import get_data_from_local

TRAIN_ALL_MODEL = False
MODELS_DIR = ROOT_PATH / "models"

if __name__ == '__main__':

    # Read the train and test datasets
    train_data = get_data_from_local(PROCESSED_TRAIN_DATA_PATH / "train_data.csv")
    test_data = get_data_from_local(PROCESSED_TEST_DATA_PATH / "test_data.csv")

    # Set this flag based on whether stemming is applied or not
    USE_STEMMING = True
    train_data = stemming(train_data, USE_STEMMING)
    test_data = stemming(test_data, USE_STEMMING)

    # Preprocess and tokenize data
    dataset_train = preprocess_and_tokenize_data(train_data, USE_STEMMING)
    dataset_test = preprocess_and_tokenize_data(test_data, USE_STEMMING)

    # Empty cache
    torch.cuda.empty_cache()

    # DataLoader
    train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=4)
    eval_dataloader = DataLoader(dataset=dataset_test, batch_size=4)

    if TRAIN_ALL_MODEL:
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        training(train_dataloader, model)
    else:
        model = joblib.load(MODELS_DIR / 'transfer-learning.joblib', mmap_mode='r')

    score_function(eval_dataloader, model)
