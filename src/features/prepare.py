"""
This module is created to take data from data/raw
and process it. It excludes non-useful columns
and processes text in order to remove stop words,
delete punctuation and stem it.
"""

import os
import yaml
from src.data.process_data import process_df
from src.data.preprocess_data import clean_df, get_data_from_local, save_data_to_local
from src import ROOT_PATH, PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH

if __name__ == '__main__':

    params_path = ROOT_PATH / "params.yaml"

    data = get_data_from_local(ROOT_PATH / 'data' / 'raw' / 'raw_data.csv')

    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["prepare"]
        except yaml.YAMLError as exc:
            print(exc)

    data = clean_df(data)
    data = process_df(data)

    train_data = data.sample(frac=params["train_size"], random_state=params["random_state"])
    test_data = data.drop(train_data.index)

    IS_EXIST = os.path.exists(PROCESSED_TRAIN_DATA_PATH)
    if not IS_EXIST:
        os.makedirs(PROCESSED_TRAIN_DATA_PATH)

    IS_EXIST = os.path.exists(PROCESSED_TEST_DATA_PATH)
    if not IS_EXIST:
        os.makedirs(PROCESSED_TEST_DATA_PATH)

    save_data_to_local(PROCESSED_TRAIN_DATA_PATH / 'train_data.csv', train_data)
    save_data_to_local(PROCESSED_TEST_DATA_PATH / 'test_data.csv', test_data)
