import kaggle
import json
import os
import pandas as pd


def get_data_from_source() -> None:

    """
    Downloads kaggle dataset and saves it in a given path.

    Args:
        None

    Returns:
        None
    """

    # path where to save data csv
    download_dir = "../../data/raw"
    # path where to find data to get kaggle dataset
    connection_source = "../../data/external/external_connection.json"

    #with open(connection_source, 'r') as file:
        #source_data = json.load(file)

    # save json data
    username = "nicapotato"
        #source_data)["username"]
    dataset_name = "womens-ecommerce-clothing-reviews"
    #source_data["dataset_name"]

    # download the dataset in CSV format
    kaggle.api.dataset_download_files(f"{username}/{dataset_name}", path=download_dir, unzip=True)

    original_file_path = os.path.join(download_dir, r'Womens Clothing E-Commerce Reviews.csv')
    desired_file_name = os.path.join(download_dir, 'raw_data.csv')

    # Rename the file
    os.rename(original_file_path, desired_file_name)


def get_data_from_local(path_to_data: str) -> pd.DataFrame:

    """
        Reads data from csv and creates a DataFrame from it.

        Args:
            path_to_data: Path where data we want can be found

        Returns:
            DataFrame: The DataFrame with the csv file's data.
    """

    dataframe = pd.read_csv(path_to_data)

    return dataframe


def save_data_to_local(
        path_to_save: str,
        dataframe: pd.DataFrame
) -> None:

    """
        Saves a dataframe on a certain path

        Args:
            path_to_save: Path where we want data to be stored
            dataframe: pd.DataFrame with data

        Returns:
            None
    """

    dataframe.to_csv(path_to_save, index=False)


if __name__ == '__main__':

    pass