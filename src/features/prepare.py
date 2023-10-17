from pathlib import Path
import yaml
from src.data.process_data import *
from src.data.preprocess_data import *

params_path = Path("../../dvc.yaml")

path_to_raw_data = "./data/raw/raw_data.csv"

data = get_data_from_local(path_to_raw_data)

with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["prepare"]
    except yaml.YAMLError as exc:
        print(exc)

data = clean_df(data)
data = process_df(data)

train_data = data.sample(frac=0.8, random_state=2023)
test_data = data.drop(train_data.index)


path_to_processed = "./data/processed"

train_data_path = path_to_processed / "train_data_processed.csv"
test_data_path = path_to_processed / "test_data_processed.csv"


save_data_to_local(train_data_path, train_data)
save_data_to_local(test_data_path, test_data)