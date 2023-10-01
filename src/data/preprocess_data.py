from get_and_save_data import *

GET_DATA_FROM_SOURCE = True


def dropping(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
        Erases all non-necessary columns from the DataFrame.

        Args:
            dataframe (DataFrame): the input DataFrame to be changed.

        Returns:
            dataframe: The changed DataFrame.
    """

    dataframe.drop(
        ["Title", "Positive Feedback Count", "Division Name",
         "Department Name", "Class Name", "Age",
         "Clothing ID", "Rating"], axis=1, inplace=True
    )
    dataframe.drop_duplicates(inplace=True)

    return dataframe


def binarization(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
        Takes the Ratings variable and binarizes its values to 1 or 0.

        Args:
            dataframe (DataFrame): Input DataFrame to be modified.

        Returns:
            dataframe: Modified DataFrame.
    """

    dataframe.loc[dataframe['Rating'] <= 4, 'Recommended IND'] = 0
    dataframe.rename(
        columns={'Recommended IND': 'Top Product'}, inplace=True
    )

    return dataframe


def clean_df(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
        Transforms a whole dataframe into a simplified version
        that only includes variables that will be used and
        a clear target column

        Args:
            dataframe (DataFrame): Input DataFrame to be modified.

        Returns:
            dataframe: Modified DataFrame.
    """

    dataframe = binarization(dataframe)
    dataframe = dropping(dataframe)

    return dataframe


if __name__ == '__main__':

    if GET_DATA_FROM_SOURCE:
        get_data_from_source()

    path_to_raw_data = "./data/raw/raw_data.csv"
    df = get_data_from_local(path_to_raw_data)

    df = clean_df(df)

    path_to_interim_data = "./data/interim/interim_data.csv"
    save_data_to_local(
        path_to_interim_data,
        df
    )

    path_to_raw_data = "./data/raw/raw_data.csv"
    df = get_data_from_local(path_to_raw_data)

    df = clean_df(df)

    path_to_interim_data = "./data/interim/interim_data.csv"
    save_data_to_local(
        path_to_interim_data,
        df
    )
