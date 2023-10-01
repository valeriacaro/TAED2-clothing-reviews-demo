import pandas as pd
from get_filename import *

def create_df() -> pd.DataFrame:
    """
        Reads data from csv and creates a DataFrame from it.

        Args:
            None.

        Returns:
            DataFrame: The DataFrame with the csv file's data.
    """
    filename = get_filename()
    dataframe = pd.read_csv(filename)
    return dataframe

if __name__ == '__main__':
    df = create_df()