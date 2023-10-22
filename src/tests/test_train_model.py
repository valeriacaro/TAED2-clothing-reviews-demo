"""
This module is created to test functions on
src/models/train_model.py
"""
import pandas as pd
import pytest
from src.models.train_model import stemming

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
