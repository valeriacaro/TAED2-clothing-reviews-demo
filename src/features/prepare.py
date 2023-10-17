from pathlib import Path
import yaml
import pandas as pd
import sklearn.impute import SimpleImputer
from src.models.predict_model import *
from src.data.get_and_save_data import *
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
use_stemming = True
x, y = stemming(df, use_stemming)

word_vocab, pad_index, unk_index = create_word_vocab(x)
label_vocab = create_label_vocab(y)

# Convert the data to indices
x_idx, y_idx = convert_data_to_indices(x, y, word_vocab, label_vocab)

# Split the data into training (70%), validation (15%), and test (15%) sets
# AIXO D'AQUI ES DE LO QUE NO ESTIC SEGURA DE QUE SIGUI AIXI IGUAL QUE LO NOSTRE
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=params["train_size"],test_size=params["test_size"], random_state=params["random_state"])

# Handle Missing Values with Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(x_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(x_val))

# Imputation removed column names so we put them back
imputed_X_train.columns = x_train.columns
imputed_X_valid.columns = x_val.columns
x_train = imputed_X_train
x_val = imputed_X_valid

path_to_processed = "./data/processed"

x_train_path = path_to_processed / "x_train_processed.csv"
y_train_path = path_to_processed / "y_train_processed.csv"
x_val_path = path_to_processed / "x_val_processed.csv"
y_val_path = path_to_processed / "y_val_processed.csv"

save_data_to_local(x_train_path, x_train)
save_data_to_local(y_train_path, y_train)
save_data_to_local(x_val_path, x_val)
save_data_to_local(y_val_path, y_val)