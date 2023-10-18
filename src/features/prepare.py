import yaml
from src.data.process_data import *
from src.data.preprocess_data import *
from src import *

if __name__ == '__main__':

    params_path = ROOT_PATH / "dvc.yaml"

    data = load_data_from_remote(RAW_DATA_PATH)

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

    save_data_to_local(PROCESSED_DATA_PATH, train_data)
    save_data_to_local(PROCESSED_DATA_PATH, test_data)
