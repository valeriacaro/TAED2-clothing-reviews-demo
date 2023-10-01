import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from get_and_save_data import *


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

    english_sw = set(stopwords.words('english') + list(string.punctuation))
    dataframe['Review Text'] = dataframe['Review Text'].apply(
        lambda x: " ".join([w.lower() for w in word_tokenize(str(x)) if w.lower() not in english_sw]))
    dataframe['Review Text'] = dataframe['Review Text'].apply(lambda text: " ".join([w for w in text if w not in "'s"]))

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
    dataframe['Stemmed Review Text'] = dataframe['Review Text'].apply(lambda text: " ".join([stem.stem(w) for w in text]))
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

    path_to_interim_data = "./data/interim/interim_data.csv"
    df = get_data_from_local(path_to_interim_data)

    df = process_df(df)

    path_to_processed_data = "./data/processed/processed_data.csv"
    save_data_to_local(
        path_to_processed_data,
        df
    )