"""
This module is created to test functions on
src/data/get_and_save_data.py
"""

import os
import pytest
import pandas as pd
from src.data.get_and_save_data import (
    get_data_from_local,
    get_data_from_source,
    save_data_to_local
)
from src import (
    RAW_DATA_PATH,
    PROCESSED_TEST_DATA_PATH,
    PROCESSED_TRAIN_DATA_PATH
)


@pytest.mark.parametrize(
    "sample_csv",
    [
        RAW_DATA_PATH / 'raw_data.csv',
        PROCESSED_TEST_DATA_PATH / 'test_data.csv',
        PROCESSED_TRAIN_DATA_PATH / 'train_data.csv'
    ]
)
def test_get_data_from_local_output(sample_csv):
    """
    Tests if when reading a csv file, the
    expected get_data_from_local output returns
    a non-empty dataframe
    Args:
        sample_csv: path to the csv to read
    """
    # Call the function to read the sample data
    dataframe = get_data_from_local(sample_csv)

    # Perform assertions
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) > 0


def test_get_data_from_source():
    """
    Tests if when downloading the dataset from
    kaggle it is well saved on the repository
    checking if the folder exists.
    """
    get_data_from_source()

    assert os.path.isfile((RAW_DATA_PATH / 'raw_data.csv').as_posix())


@pytest.mark.parametrize(
    "sample_csv",
    [
        RAW_DATA_PATH / 'raw_data.csv',
        PROCESSED_TEST_DATA_PATH / 'test_data.csv',
        PROCESSED_TRAIN_DATA_PATH / 'train_data.csv'
    ]
)
def test_save_data_to_local(sample_csv):
    """
    Tests if when we save a dataframe as a csv
    on our repository, the path to it exists.
    Args:
        sample_csv: path to the dataframe csv
    """

    dataframe = get_data_from_local(sample_csv)

    save_data_to_local(sample_csv, dataframe)

    assert os.path.isfile(sample_csv)
