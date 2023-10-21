from test_model import *
from transformers import AutoModelForSequenceClassification
import joblib
from src import PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH

TRAIN_ALL_MODEL = False
MODELS_DIR = ROOT_PATH / "models"

if __name__ == '__main__':
    # Read the train and test datasets
    train_data = read_data(PROCESSED_TRAIN_DATA_PATH / "train_data.csv")
    test_data = read_data(PROCESSED_TEST_DATA_PATH / "test_data.csv")

    # Set this flag based on whether stemming is applied or not
    use_stemming = True
    train_data = stemming(train_data, use_stemming)
    test_data = stemming(test_data, use_stemming)

    # Preprocess and tokenize data
    dataset_train = preprocess_and_tokenize_data(train_data, use_stemming)
    dataset_test = preprocess_and_tokenize_data(test_data, use_stemming)

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
