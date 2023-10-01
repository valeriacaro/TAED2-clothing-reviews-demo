from download_data import *
from data_from_folder import *
from save_data import *
from stemming import *
from cleaning_data import *


if __name__ == '__main__':
    download_df()
    df = create_df()
    df = binarization(df)
    df = dropping(df)
    df = tokenization(df)
    df = stemmed_text(df)
    new_dataset(df)


