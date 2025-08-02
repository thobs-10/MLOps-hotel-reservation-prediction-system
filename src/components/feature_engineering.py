import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from zenml.steps import step

from src.entity.config_entity import DataPreprocessingConfig, FeatureEngineeringConfig
from src.entity.constants import FeatureEngineeringConstants
from src.utils.feature_utils import (
    prepare_feast_data,
    save_to_feast_store,
    validate_feast_data,
)
from src.utils.main_utils import (
    create_feature_importance_selector,
    generate_label_encoder,
)


@step(enable_cache=True)
def load_processed_data() -> pd.DataFrame:
    """
    Load processed data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        file_path: str = DataPreprocessingConfig.processed_data_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Loading processed data from {file_path}")
        df = pd.read_parquet(os.path.join(file_path, "cleaned_data.parquet"))
        return df
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise e


@step
def generate_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate new features from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to generate new features from.
    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    df["total_nights_stayed"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["is_repeat_guest"] = df["repeated_guest"].apply(lambda x: 1 if x == "Yes" else 0)
    return df


@step
def encode_categorical_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode categorical columns in the DataFrame using the provided LabelEncoder.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    label_encoder = generate_label_encoder()
    fe_constants = FeatureEngineeringConstants()
    categorical_columns: List[str] = fe_constants.categorical_columns
    if not categorical_columns:
        raise ValueError("No categorical columns provided for encoding.")
    if label_encoder is None:
        raise ValueError("LabelEncoder instance is not provided.")
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot encode categorical columns.")
    if not all(col in df.columns for col in categorical_columns):
        raise ValueError(
            "Some categorical columns are not present in the DataFrame: "
            f"{set(categorical_columns) - set(df.columns)}"
        )
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df, label_encoder


@step
def separate_data(
    df: pd.DataFrame,
    label_encoder: Optional[LabelEncoder] = None,
) -> Tuple[pd.DataFrame, np.ndarray[Any, Any]]:
    """
    Separate the DataFrame into features (X) and target variable (y).
    Args:
        df (pd.DataFrame): DataFrame containing the data.
    Returns:
        Tuple[pd.DataFrame, np.ndarray[Any, Any]]: Features DataFrame (X) and target variable array (y).
    """
    fe_constants = FeatureEngineeringConstants()
    target_column: str = fe_constants.target_column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    X = df.drop(columns=[target_column], axis=1)
    y = df[target_column]
    y_encoded = label_encoder.fit_transform(y) if label_encoder else y
    return X, y_encoded


@step
def get_important_features(
    X: pd.DataFrame,
    y: np.ndarray[Any, Any],
    threshold: float = 0.05,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Get features with importance scores above the threshold.

    Args:
        X: Features DataFrame
        y: Target variable numpy array
        threshold: Minimum importance score (default: 0.05)

    Returns:
        DataFrame containing only the important features and a list of their names.
    """
    X_copy = X.copy()
    X.drop(columns=["booking_date"], inplace=True, errors="ignore")
    selector = create_feature_importance_selector(X, y)
    if not isinstance(selector, ExtraTreesClassifier):
        raise ValueError("Selector must be an instance of ExtraTreesClassifier.")
    if X.empty:
        raise ValueError("Features DataFrame is empty. Cannot get important features.")
    if not hasattr(selector, "feature_importances_"):
        raise ValueError("Selector does not have feature_importances_ attribute.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a numeric value (int or float).")
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1.")
    importances = selector.feature_importances_
    mask = importances > threshold
    feature_names = X.columns[mask]
    return X[feature_names], feature_names.tolist(), X_copy


@step
def save_feature_engineered_data(
    df: pd.DataFrame,
    X_copy: pd.DataFrame,
    y: np.ndarray[Any, Any],
) -> None:
    """
    Save the feature engineered DataFrame and target variable for both Feast and local backup.

    This function saves data in two locations:
    1. For Feast consumption (as specified in hotel_features.py)
    2. As local backup in the feature store directory

    Args:
        df (pd.DataFrame): Feature engineered DataFrame.
        y (np.ndarray[Any, Any]): Target variable array.
    """
    output_path: str = FeatureEngineeringConfig.feature_engineering_dir
    if not output_path:
        raise ValueError("Output path is not set in the configuration file.")

    try:
        # Create directories
        os.makedirs(output_path, exist_ok=True)
        # os.makedirs("data/feature_store", exist_ok=True)

        # Prepare data for Feast (combine features and target)
        feast_df = prepare_feast_data(df, X_copy, y)
        combined_df = feast_df.copy()

        # Validate Feast data format
        if validate_feast_data(feast_df):
            logger.info("✅ Feast data validation passed")
        else:
            logger.error("❌ Feast data validation failed")
            raise ValueError("Feast data validation failed. Check the data format.")

        # 1. Save for Feast consumption (as specified in hotel_features.py)
        feast_path = "data/feature_store/feast_hotel_features.parquet"
        # feast_df.drop(columns=["booking_id"], inplace=True)
        feast_df.to_parquet(feast_path, index=False)
        logger.info(f"✅ Feast-ready data saved to {feast_path}")

        # 2. Save local backup in feature store directory
        backup_features_path = os.path.join(
            output_path, "feast_hotel_features_backup.parquet"
        )
        backup_target_path = os.path.join(output_path, "hotel_target_backup.csv")

        df.to_parquet(backup_features_path, index=False)
        y_df = pd.DataFrame(y, columns=["booking_status"])
        y_df.to_csv(backup_target_path, index=False)
        # y.to_csv(backup_target_path, index=False)
        logger.info(f"💾 Backup features saved to {backup_features_path}")
        logger.info(f"💾 Backup target saved to {backup_target_path}")

        # 3. Save additional formats for analysis
        combined_data_path = os.path.join(
            output_path, "hotel_features_with_target.parquet"
        )
        # combined_df = df.copy()
        # combined_df["booking_status"] = y
        combined_df.to_parquet(combined_data_path, index=False)
        logger.info(f"📊 Combined data saved to {combined_data_path}")

        save_to_feast_store()

        logger.info("🎉 All feature engineered data saved successfully!")

    except Exception as e:
        logger.error(f"❌ Error saving feature engineered data: {e}")
        raise e
