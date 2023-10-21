'''
This module is created to unify all functions related
to processing text in order to improve the results of
our model. This includes tokenization and getting
stemmed text.
'''

import string
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from src.data.get_and_save_data import get_data_from_local, save_data_to_local
from src import ROOT_PATH


def tokenization(dataframe) -> pd.DataFrame:

    """
            Tokenizes the text and lower-cases it.

            Args:
                dataframe (DataFrame): Input DataFrame to be modified.

            Returns:
                dataframe: Modified DataFrame.
    """

    nltk.download("punkt")
    nltk.download("stopwords")

    custom_stopwords = ['\'', '\'\'']

    english_sw = stopwords.words('english') + list(string.punctuation) + custom_stopwords
    dataframe['Review Text'] = dataframe['Review Text'].apply(
        lambda x: " ".join([w.lower() for w in word_tokenize(
            re.sub(r'[^\w\s]', '', str(x))) if w.lower() not in english_sw])
    )

    dataframe['Review Text'] = dataframe['Review Text'].apply(
        lambda text: " ".join([w for w in text.split() if w != "'s"])
    )

    return dataframe


def get_stemmed_text(dataframe) -> pd.DataFrame:

    """
                Stems the Review Text from the dataframe.

                Args:
                    dataframe (DataFrame): Input DataFrame to be modified.

                Returns:
                    dataframe: Modified DataFrame.
    """

    stem = SnowballStemmer('english')
    dataframe['Stemmed Review Text'] = dataframe['Review Text'].apply(
        lambda text: " ".join([stem.stem(w) for w in text.split()])
    )
    return dataframe


def process_df(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
        Transforms a dataframe with a text column into a
        processed version that includes two versions of
        the text: a tokenized one, and a tokenized plus
        stemmed one

        Args:
            dataframe (DataFrame): Input DataFrame to be modified.

        Returns:
            dataframe: Modified DataFrame.
    """

    dataframe = tokenization(dataframe)
    dataframe = get_stemmed_text(dataframe)

    return dataframe


if __name__ == '__main__':

    INTERIM_DATA_PATH = ROOT_PATH / 'data' / 'interim' / 'interim_data.csv'
    df = get_data_from_local(INTERIM_DATA_PATH)

    df = process_df(df)

    PROCESSED_DATA_PATH = ROOT_PATH / 'data' / 'processed' / 'processed_data.csv'
    save_data_to_local(
        PROCESSED_DATA_PATH,
        df
    )
