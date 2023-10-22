"""
This module is created as the initialization
heart of our project. Also is used to declare
some global variables.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = Path(Path(__file__).resolve().parent.parent)

RAW_DATA_PATH = ROOT_PATH / "data" / "raw"
PROCESSED_TRAIN_DATA_PATH = ROOT_PATH / "data" / "processed" / "train"
PROCESSED_TEST_DATA_PATH = ROOT_PATH / "data" / "processed" / "test"
