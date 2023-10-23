"""
This module is created to check the model
performance.
"""

import pytest
import torch
import os
import json
from src.models.test_model import prediction, score_function
from src.data.get_and_save_data import get_data_from_local
from src.models.train_model import stemming, preprocess_and_tokenize_data
from torch.utils.data import DataLoader
from src import PROCESSED_TEST_DATA_PATH, ROOT_PATH


def mock_dataloader():
    """
    Creates a mock DataLoader for the evaluation performed
    on test_score_function_output.
    """

    USE_STEMMING = True
    test_data = get_data_from_local(PROCESSED_TEST_DATA_PATH / "test_data.csv")
    test_data = stemming(test_data, USE_STEMMING)
    dataset_test = preprocess_and_tokenize_data(test_data, USE_STEMMING)
    eval_dataloader = DataLoader(dataset=dataset_test, batch_size=4)

    return eval_dataloader


def mock_load_model():
    """
    Creates a mock model for the evaluation performed
    on test_score_function_output.
    """

    MODELS_DIR = ROOT_PATH / "model"
    model = torch.load(
        MODELS_DIR / 'transfer-learning.pt', map_location=torch.device('cuda')
    )

    return model

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full evaluation test requires a GPU."
)
def test_score_function_output():
    """
    Tests if score_function creates the json with
    the accuracy metric of the model loaded.
    """

    mock_eval_dataloader = mock_dataloader()
    mock_model = mock_load_model()

    score_function(mock_eval_dataloader, mock_model)

    path_to_metrics = ROOT_PATH / 'metrics' / 'scores.json'

    assert os.path.isfile(path_to_metrics)

def test_accuracy_metric():
    """
    Tests if json with model metrics contains
    the accuracy metric and if this is above
    75%.
    """

    # Load the JSON file containing the accuracy metric
    with open(ROOT_PATH / 'metrics' / 'scores.json', "r", encoding="utf-8") as scores_file:
        scores = json.load(scores_file)

    # Check if the "accuracy" key exists and the value is above 0.75
    assert "accuracy" in scores
    assert scores["accuracy"]["accuracy"] > 0.75
