from pathlib import Path
import yaml
import sklearn.impute import SimpleImputer
from src.models.predict_model import *
from src.data.process_data import *
from src.data.preprocess_data import *
from sklearn.model_selection import train_test_split

paramspath = Path("params.yaml")

path_to_raw_data = "./data/raw/raw_data.csv"

data = get_data_from_local(path_to_raw_data)

with open(paramspath, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["prepare"]
    except yaml.YAMLError as exc:
        print(exc)

data = clean_df(data)
data = process_df(data)

# Set this flag based on whether stemming is applied or not

# Split the data into training (70%), validation (15%), and test (15%) sets
# AIXO D'AQUI ES DE LO QUE NO ESTIC SEGURA DE QUE SIGUI AIXI IGUAL QUE LO NOSTRE
train_data = data.sample(frac=0.8, random_state=2023)
test_data = data.drop(train_data.index)


path_to_processed = "./data/processed"

train_data_path = path_to_processed / "train_data_processed.csv"
test_data_path = path_to_processed / "test_data_processed.csv"


save_data_to_local(train_data_path, train_data)
save_data_to_local(test_data_path, test_data)
