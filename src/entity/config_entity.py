import os
from dataclasses import dataclass

from dotenv import load_dotenv

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(root_dir, ".env"))


@dataclass
class DataIngestionConfig:
    raw_data_path: str = "data/raw/Hotel Reservations.csv"


@dataclass
class DataPreprocessingConfig:
    processed_data_path: str = "data/processed_data/"


@dataclass
class FeatureEngineeringConfig:
    feature_engineering_dir: str = "data/feature_store/"


@dataclass
class ModelTrainingConfig:
    model_artifact_dir: str = "src/models/artifacts/"
    trained_models_dir: str = "src/models/trained_models/"
    model_registry_path: str = "src/models/model_registry/"


@dataclass
class FeatureStoreConfig:
    feature_store_repo_path: str = "/users/thobelasixpence/Documents/mlops-zoomcamp-project-2024/MLOPs-hotel-reservation-prediction-system/feature_store"
