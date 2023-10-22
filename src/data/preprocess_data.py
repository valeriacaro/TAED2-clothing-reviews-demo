"""
This module is created to unify all functions
related to preprocess data before performing
text processing on it. This includes removing
some columns or mapping values.
"""

import pandas as pd
from src import ROOT_PATH, RAW_DATA_PATH
from src.data.get_and_save_data import (get_data_from_local,
                                        get_data_from_source,
                                        save_data_to_local)

GET_DATA_FROM_SOURCE = True


def dropping(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
        Erases all non-necessary columns from the DataFrame.

        Args:
            dataframe (DataFrame): the input DataFrame to be changed.

        Returns:
            dataframe: The changed DataFrame.
    """

    dataframe.drop(
        ["Unnamed: 0", "Title", "Positive Feedback Count", "Division Name",
         "Department Name", "Class Name", "Age",
         "Clothing ID", "Rating"], axis=1, errors='ignore', inplace=True
    )
    dataframe.drop_duplicates(inplace=True)

    return dataframe


def binarization(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
        Takes the Ratings variable and binarizes its values to 1 or 0.

        Args:
            dataframe (DataFrame): Input DataFrame to be modified.

        Returns:
            dataframe: Modified DataFrame.
    """

    dataframe.loc[dataframe['Rating'] <= 4, 'Recommended IND'] = 0
    dataframe.loc[dataframe['Rating'] > 4, 'Recommended IND'] = 1
    dataframe.rename(
        columns={'Recommended IND': 'Top Product'}, inplace=True
    )

    return dataframe


def clean_df(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
        Transforms a whole dataframe into a simplified version
        that only includes variables that will be used and
        a clear target column

        Args:
            dataframe (DataFrame): Input DataFrame to be modified.

        Returns:
            dataframe: Modified DataFrame.
    """

    dataframe = binarization(dataframe)
    dataframe = dropping(dataframe)
    dataframe = dataframe.dropna(subset=['Review Text'])
    dataframe = dataframe.dropna(subset=['Top Product'])

    return dataframe


if __name__ == '__main__':

    if GET_DATA_FROM_SOURCE:
        get_data_from_source()

    df = get_data_from_local(RAW_DATA_PATH)

    df = clean_df(df)

    INTERIM_DATA_PATH = ROOT_PATH / 'data' / 'interim' / 'interim_data.csv'
    save_data_to_local(
        INTERIM_DATA_PATH,
        df
    )
