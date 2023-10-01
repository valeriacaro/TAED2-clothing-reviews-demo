import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from download_data import *
from data_from_folder import *
from save_data import *
from stemming import *



def dropping(dataframe) -> pd.DataFrame:
    """
        Erases all non-necessary columns from the DataFrame.

        Args:
            dataframe (DataFrame): the input DataFrame to be changed.

        Returns:
            dataframe: The changed DataFrame.
    """
    dataframe.drop(
        ["Title", "Positive Feedback Count", "Division Name", "Department Name", "Class Name", "Age",
         "Clothing ID", "Rating"], axis=1, inplace=True)
    dataframe.drop_duplicates(inplace=True)
    return dataframe


def binarization(dataframe) -> pd.DataFrame:
    """
        Takes the Ratings variable and binarizes its values to 1 or 0.

        Args:
            dataframe (DataFrame): Input DataFrame to be modified.

        Returns:
            dataframe: Modified DataFrame.
    """
    dataframe.loc[dataframe['Rating'] <= 4, 'Recommended IND'] = 0
    dataframe.rename(columns={'Recommended IND': 'Top Product'}, inplace=True)
    return dataframe


if __name__ == '__main__':
    download_df()
    df = create_df()
    df = binarization(df)
    df = dropping(df)
    df = tokenization(df)
    df = stemmed_text(df)
    new_dataset(df)


