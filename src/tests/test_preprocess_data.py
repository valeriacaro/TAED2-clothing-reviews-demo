"""
This module is created to test functions stored
on src.data.preprocess_data
"""
import pandas as pd
from src.data.preprocess_data import dropping, binarization, clean_df

def test_dropping_output():

    data = {
        'Index': 1,
        'Review': 'good shirt',
        'Age': '44',
        'Rating': '5',
        'Top Product': '1'
    }

    dataframe = pd.DataFrame(
        data, index=['Index'],
        columns=['Review', 'Age', 'Rating', 'Top Product']
    )

    dataframe = dropping(dataframe)

    assert 'Age' not in dataframe.columns
    assert 'Rating' not in dataframe.columns


def test_binarization_output():

    pass

def test_clean_df_output():

    pass
