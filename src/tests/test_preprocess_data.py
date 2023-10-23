"""
This module is created to test functions stored
on src/data/preprocess_data
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
    # Create a sample dataframe
    data = [['good shirt', 44, 5, 1]]

    dataframe = pd.DataFrame(
        data, index=[1],
        columns=['Review Text', 'Age', 'Rating', 'Top Product']
    )
    # Call dropping function
    dataframe = dropping(dataframe)
    # Only Review Text and Top Product columns should exist
    assert 'Age' not in dataframe.columns
    assert 'Rating' not in dataframe.columns
    assert 'Review Text' in dataframe.columns
    assert 'Top Product' in dataframe.columns


@pytest.mark.parametrize(
    "rate, top",
    [
        (5, 1),
        (4, 0),
        (3, 0),
        (2, 0),
        (1, 0)
    ]
)
def test_binarization_output(rate, top):
    """
    Checks if given a dataframe with
    a Rating column, binarization returns
    the same dataframe without that column,
    but with Top Product with the respective
    mapping.
    """
    # Create a sample dataframe
    data = [[rate]]

    dataframe = pd.DataFrame(
        data, index=[1], columns=['Rating']
    )
    # Call the binarization function
    dataframe = binarization(dataframe)
    # Check if the Top Product column has the expected value
    assert 'Top Product' in dataframe.columns
    assert (dataframe['Top Product'] == top).all()


def test_clean_df_output():
    """
    Checks if the whole integration of
    dropping and binarization returns the
    expected output.
    """
    # Create a sample dataframe size (2, 3)
    data = [
        ['good shirt', 5, 44],
        [None, 4, 33]
    ]

    dataframe = pd.DataFrame(
        data=data, index=[1, 2],
        columns=['Review Text', 'Rating', 'Age']
    )
    # Call the clean_df function
    dataframe = clean_df(dataframe)
    # Line with NaN value should be deleted
    # Rating and Age column should be deleted
    assert dataframe.shape == (1, 2)
    assert 'Rating' not in dataframe.columns
    assert 'Age' not in dataframe.columns
    # Review Text and Top Product should exist
    assert 'Review Text' in dataframe.columns
    assert 'Top Product' in dataframe.columns
    # The value of Top Product should be as expected
    assert (dataframe['Top Product'] == 1).all()
