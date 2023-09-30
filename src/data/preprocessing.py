import pandas
import pandas as pd
import altair as alt
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize


def create_df() -> pd.DataFrame:
    """
        Reads data from csv and creates a DataFrame from it.

        Args:
            None.

        Returns:
            DataFrame: The DataFrame with the csv file's data.
    """
    dataframe = pd.read_csv(...)
    return dataframe


def dropping(dataframe) -> pd.DataFrame:
    """
        Erases all non-necessary columns from the DataFrame.

        Args:
            dataframe (DataFrame): the input DataFrame to be changed.

        Returns:
            dataframe: The changed DataFrame.
    """
    dataframe.drop(
        ["Unnamed:0", "Title", "Positive Feedback Count", "Division Name", "Department Name", "Class Name", "Age",
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
        lambda x: [w.lower() for w in word_tokenize(str(x)) if w.lower() not in english_sw])
    return dataframe

if __name__ == '__main__':
    df = create_df()
    df = binarization(df)
    df = dropping(df)
    df = tokenization(df)
