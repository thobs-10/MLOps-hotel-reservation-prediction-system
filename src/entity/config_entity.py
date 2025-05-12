import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(root_dir, ".env"))


@dataclass
class DataIngestionConfig:
    raw_data_path: Optional[str] = os.getenv("RAW_DATA_PATH")


@dataclass
class DataPreprocessingConfig:
    processed_data_path: Optional[str] = os.getenv("PROCESSED_DATA_PATH")


@dataclass
class FeatureEngineeringConfig:
    feature_engineering_dir: Optional[str] = os.getenv("FEATURES_PATH")


@dataclass
class ModelTrainingConfig:
    model_artifact_dir: Optional[str] = os.getenv("MODEL_ARTIFACTS_PATH")
