import pandas as pd
from get_filename import *
def new_dataset(dataframe):
    """
    Saves processed dataset in csv file.

    Args:
        dataframe (DataFrame): Input DataFrame to be saved.

    Returns:
        None.
    """
    csv_file_path = get_filename()
    dataframe.to_csv(csv_file_path, index=False)