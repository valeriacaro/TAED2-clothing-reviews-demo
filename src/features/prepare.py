import yaml
from src.data.process_data import *
from src.data.preprocess_data import *
from src import *

if __name__ == '__main__':

    params_path = ROOT_PATH / "params.yaml"

    data = get_data_from_local(ROOT_PATH / 'data' / 'raw' / 'raw_data.csv')

    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["prepare"]
        except yaml.YAMLError as exc:
            print(exc)

    data = clean_df(data)
    data = process_df(data)

    train_data = data.sample(frac=params["train_size"], random_state=params["random_state"])
    test_data = data.drop(train_data.index)

    isExist = os.path.exists(PROCESSED_TRAIN_DATA_PATH)
    if not isExist:
        os.makedirs(PROCESSED_TRAIN_DATA_PATH)

    isExist = os.path.exists(PROCESSED_TEST_DATA_PATH)
    if not isExist:
        os.makedirs(PROCESSED_TEST_DATA_PATH)

    save_data_to_local(PROCESSED_TRAIN_DATA_PATH / 'train_data.csv', train_data)
    save_data_to_local(PROCESSED_TEST_DATA_PATH / 'test_data.csv', test_data)
