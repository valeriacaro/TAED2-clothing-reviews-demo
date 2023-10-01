import pandas as pd
import altair as alt
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import kaggle

def download_df():
    """
    Downloads kaggle dataset and saves it in the repository.

    Args:
        None

    Returns:
        None
    """
    download_dir = "/Users/claudialen/Dropbox/UNI/7eQuadri/TAED2/clothing-reviews/data/external"
    username = "nicapotato"
    dataset_name = "womens-ecommerce-clothing-reviews"

    # Download the dataset in CSV format
    kaggle.api.dataset_download_files(f"{username}/{dataset_name}", path=download_dir, unzip=True)


def create_df() -> pd.DataFrame:
    """
        Reads data from csv and creates a DataFrame from it.

        Args:
            None.

        Returns:
            DataFrame: The DataFrame with the csv file's data.
    """
    dataframe = pd.read_csv("/Users/claudialen/Dropbox/UNI/7eQuadri/TAED2/clothing-reviews/data/external/Womens Clothing E-Commerce Reviews.csv")
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

def stemmed_text(dataframe) -> pd.DataFrame:
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

def new_dataset(dataframe):
    """
    Saves processed dataset in csv file.

    Args:
        dataframe (DataFrame): Input DataFrame to be saved.

    Returns:
        None.
    """
    csv_file_path = "/Users/claudialen/Dropbox/UNI/7eQuadri/TAED2/clothing-reviews/data/interim/preprocessed_data.csv"
    dataframe.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    download_df()
    df = create_df()
    df = binarization(df)
    df = dropping(df)
    df = tokenization(df)
    df = stemmed_text(df)
    new_dataset(df)