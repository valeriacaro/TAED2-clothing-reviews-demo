from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd

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