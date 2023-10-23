"""
This module is created to test functions on
src/models/train_model.py
"""

import pandas as pd
from datasets import Dataset
from src.models.train_model import stemming, tokenize_dataset, tokenize_dataset_stem

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
