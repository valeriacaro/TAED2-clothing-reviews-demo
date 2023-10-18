import yaml
from pathlib import Path
from src.data.get_and_save_data import *

if __name__ == '__main__':

    params_path = Path("../../params.yaml")

    get_data_from_source()

    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["extract"]
        except yaml.YAMLError as exc:
            print(exc)