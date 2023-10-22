"""
This module is created to test functions stored
on src.data.preprocess_data
"""
import pandas as pd
import pytest
from src.data.preprocess_data import dropping, binarization, clean_df

def test_dropping_output():
    """
    Checks if given a random dataframe
    with some columns to delete by dropping,
    but not with all on the list of the function,
    just keeps important columns.
    """

    data = {
        'Index': 1,
        'Review': 'good shirt',
        'Age': 44,
        'Rating': 5,
        'Top Product': 1
    }

    dataframe = pd.DataFrame(
        data, index=['Index'],
        columns=['Review', 'Age', 'Rating', 'Top Product']
    )

    dataframe = dropping(dataframe)

    assert 'Age' not in dataframe.columns
    assert 'Rating' not in dataframe.columns


def test_binarization_output(Rate, Top):
    """
    Checks if given a dataframe with
    a Rating column, binarization returns
    the same dataframe without that column,
    but with Top Product with the respective
    mapping.
    """

    data = {
        'Index': 1,
        'Rating': Rate
    }
    dataframe = pd.DataFrame(
        data, index=['Index'], columns=['Rating']
    )

    dataframe = binarization(dataframe)

    assert 'Top Product' in dataframe.columns
    assert (dataframe['Top Product'] == Top).all()


@pytest.mark.parametrize(
    "Rating, Top_Product",
    [
        (5, 1),
        (3, 0),
        (4, 0),
        (2, 0),
        (1, 0)
    ]
)
def test_binarization_output_parametrize(Rating, Top_Product):
    test_binarization_output(Rating, Top_Product)


def test_clean_df_output():

    pass
