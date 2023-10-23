"""
This module is created to check the model
performance.
"""

import os
import json
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from src.models.test_model import prediction, score_function
from src.data.get_and_save_data import get_data_from_local
from src.models.train_model import stemming, preprocess_and_tokenize_data
from src import PROCESSED_TEST_DATA_PATH, ROOT_PATH


def mock_dataloader(source: pd.DataFrame):
    """
    Creates a mock DataLoader for the evaluation performed
    on test_score_function_output and test_prediction_minimum_functionality,
    test_prediction_invariance and test_prediction_directional.

    Args:
        source: which dataframe take
    """

    use_stemming = True
    test_data = source
    test_data = stemming(test_data, use_stemming)
    dataset_test = preprocess_and_tokenize_data(test_data, use_stemming)
    eval_dataloader = DataLoader(dataset=dataset_test, batch_size=4)

    return eval_dataloader


def mock_load_model(device: str):
    """
    Creates a mock model for the evaluation performed
    on test_score_function_output and test_prediction_minimum_functionality,
    test_prediction_invariance and test_prediction_directional

    Args:
        device: where device to set the model (cpu or gpu)
    """

    models_path = ROOT_PATH / "model"
    model = torch.load(
        models_path / 'transfer-learning.pt', map_location=torch.device(device)
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

    data = get_data_from_local(PROCESSED_TEST_DATA_PATH / "test_data.csv")
    mock_eval_dataloader = mock_dataloader(data)
    mock_model = mock_load_model('gpu')

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


def test_prediction_minimum_functionality():
    """
    This tests if the output of prediction function
    is a tensor list as expected and if it behaviours
    well given an input.
    """

    data = [
        ['beauti top uniqu ordinari bought usual medium found '
         'fit tight across chest although babi year nurs could '
         'bought would size', 1]
    ]

    test_data = pd.DataFrame(
        data, index=None, columns=['Stemmed Review Text', 'Top Product']
    )

    eval_dataloader = mock_dataloader(test_data)

    model = mock_load_model('cpu')

    predict = prediction(eval_dataloader, model)

    assert isinstance(predict, list)
    isinstance(predict[0], torch.Tensor)
    assert len(predict) == 1
    assert predict[0] == 1


@pytest.mark.parametrize(
    "sentence1, sentence2, result",
    [
        ('hate shirt', 'hate skirt', 0),
        ('Maria love dress', 'Marwa love dress', 1)
    ]
)
def test_prediction_invariance(sentence1, sentence2, result):
    """
    Tests if the model is invariant to non-important words
    Args:
        sentence1: sentence with some sentyment
        sentence2: sentence1 with a non-important word changed
        result: which outputs is exepcted
    """

    data = [
        [sentence1, result],
        [sentence2, result]
    ]

    test_data = pd.DataFrame(
        data, index=None, columns=['Stemmed Review Text', 'Top Product']
    )

    eval_dataloader = mock_dataloader(test_data)

    model = mock_load_model('cpu')

    predict = prediction(eval_dataloader, model)

    assert predict[0][0] == result
    assert predict[0][0] == predict[0][1]


def test_prediction_directional():
    """
    Tests if our model receives direction of sentences.
    By changing an important sentyment word, it should
    return different outputs.
    """

    data = [
        ['hate skirt', 0],
        ['love skirt', 1]
    ]

    test_data = pd.DataFrame(
        data, index=None, columns=['Stemmed Review Text', 'Top Product']
    )

    eval_dataloader = mock_dataloader(test_data)

    model = mock_load_model('cpu')

    predict = prediction(eval_dataloader, model)

    assert predict[0][0] == 0
    assert predict[0][1] == 1
    assert predict[0][0] != predict[0][1]
