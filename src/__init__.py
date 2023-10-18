from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = Path(Path(__file__).resolve().parent.parent)

RAW_DATA_PATH = ROOT_PATH / "data" / "raw"
PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed"
