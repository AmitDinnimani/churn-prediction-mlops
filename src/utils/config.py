import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

LOG_FILE_PATH = ROOT_PATH / "logs/app.log"
DATA_PATH = ROOT_PATH / os.getenv("DATA_DIR", "data/raw/churn.csv")
