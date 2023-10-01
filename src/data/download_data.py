import sys
import kaggle

def download_df():
    """
    Downloads kaggle dataset and saves it in a given path.

    Args:
        None

    Returns:
        None
    """

    download_dir = sys.argv[1]

    username = "nicapotato"
    dataset_name = "womens-ecommerce-clothing-reviews"

    # Download the dataset in CSV format
    kaggle.api.dataset_download_files(f"{username}/{dataset_name}", path=download_dir, unzip=True)
