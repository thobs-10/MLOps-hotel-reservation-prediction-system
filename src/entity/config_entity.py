import os
from dataclasses import dataclass

from dotenv import load_dotenv

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(root_dir, ".env"))


@dataclass
class DataIngestionConfig:
    raw_data_path: str = "data/raw/train.csv"


@dataclass
class DataPreprocessingConfig:
    processed_data_path: str = "data/processed_data/"


@dataclass
class ModelTrainingConfig:
    model_artifact_dir: str = "src/models/artifacts/"
