"""
This module is created to test functions on
src/models/train_model.py
"""

import pandas as pd
import torch
import pytest
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from src.models.train_model import (
    stemming,
    tokenize_dataset,
    tokenize_dataset_stem,
    training,
    preprocess_and_tokenize_data
)
from src.data.get_and_save_data import get_data_from_local
from src import PROCESSED_TRAIN_DATA_PATH


def test_stemming_output():
    """
    This tests if stemming function is returning the
    right output depending on the stem boolean variable.
    """
    tokenized = 'trousers cool'
    stemmed = 'trouser cool'

    data = [[tokenized, stemmed]]

    dataframe = pd.DataFrame(
        data, index=[1],
        columns=['Review Text', 'Stemmed Review Text']
    )

    dataframe_stemmed = stemming(dataframe.copy(), stem=True)
    dataframe_not_stemmed = stemming(dataframe.copy(), stem=False)

    assert 'Review Text' not in dataframe_stemmed.columns
    assert (dataframe_stemmed['Stemmed Review Text'] == stemmed).all()
    assert 'Stemmed Review Text' not in dataframe_not_stemmed.columns
    assert (dataframe_not_stemmed['Review Text'] == tokenized).all()


def test_tokenize_dataset_output():
    """
    This tests if tokenize_dataset function
    returns a non-empty tokenized dataset with
    the expected columns it should have.
    """
    # Create a simple dataset for testing
    data = Dataset.from_dict({"Review Text": ["trousers cool"]})

    # Call the function to tokenize the dataset
    tokenized_dataset = tokenize_dataset(data)

    # Ensure the tokenized dataset has the expected keys
    expected_columns = ["input_ids", "attention_mask", "token_type_ids"]
    for column in expected_columns:
        assert column in tokenized_dataset
    assert len(tokenized_dataset) > 0


def test_tokenize_dataset_stem_output():
    """
    This tests if tokenize_dataset_stem function
    returns a non-empty tokenized dataset with
    the expected columns it should have.
    """
    # Create a simple dataset for testing
    data = Dataset.from_dict({"Stemmed Review Text": ["trouser cool"]})

    # Call the function to tokenize the dataset
    tokenized_dataset = tokenize_dataset_stem(data)

    # Ensure the tokenized dataset has the expected keys
    expected_columns = ["input_ids", "attention_mask", "token_type_ids"]
    for column in expected_columns:
        assert column in tokenized_dataset
    assert len(tokenized_dataset) > 0


def mock_dataloader():
    """
    This creates a mock dataloader for being used
    on test_training function.
    """

    train_data = get_data_from_local(PROCESSED_TRAIN_DATA_PATH / "train_data.csv")
    use_stemming = True
    train_data = stemming(train_data, use_stemming)
    dataset_train = preprocess_and_tokenize_data(train_data, use_stemming)
    train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=4)

    return train_dataloader


def mock_model_pretrained():
    """
    This created a mock model to use it
    on test_training function
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )

    return model


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full evaluation test requires a GPU."
)
def test_training():
    """
    This tests if the model training is ensured
    by using cpu either gpu
    """

    mock_train_dataloader = mock_dataloader()
    mock_model = mock_model_pretrained()

    assert training(mock_train_dataloader, mock_model, which_device='cpu')
    assert training(mock_train_dataloader, mock_model, which_device='gpu')
