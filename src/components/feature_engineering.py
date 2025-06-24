import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import ExtraTreesClassifier

from src.entity.config_entity import DataPreprocessingConfig, FeatureEngineeringConfig
from src.entity.constants import FeatureEngineeringConstants
from src.utils.main_utils import (
    create_feature_importance_selector,
    fit_pca,
    generate_label_encoder,
)


def load_procesed_data() -> pd.DataFrame:
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
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise e


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


def encode_categorical_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode categorical columns in the DataFrame using the provided LabelEncoder.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    label_encoder = generate_label_encoder()
    categorical_columns: List[str] = FeatureEngineeringConstants.categorical_columns
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
    return df


def separate_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the DataFrame into features (X) and target variable (y).
    Args:
        df (pd.DataFrame): DataFrame containing the data.
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame (X) and target variable Series (y).
    """
    target_column: str = FeatureEngineeringConstants.target_column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    X = df.drop(columns=[target_column], axis=1)
    y = df[target_column]
    return X, y


def get_important_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.05,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Get features with importance scores above the threshold.

    Args:
        X: Features DataFrame
        y: Target variable Series
        threshold: Minimum importance score (default: 0.05)

    Returns:
        DataFrame containing only the important features
    """
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
    return X[feature_names], feature_names.tolist()


def get_pca_feature_importance(
    pca, columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Get the most important original feature for each principal component.
    Returns a DataFrame and a list of most important feature names.
    """
    pca, X_pca = fit_pca(pca, columns)
    n_pcs = pca.components_.shape[0]
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    most_important_names = [columns[most_important[i]] for i in range(n_pcs)]
    dic = {"PC{}".format(i + 1): most_important_names[i] for i in range(n_pcs)}
    df = pd.DataFrame(
        sorted(dic.items()), columns=["Principal Component", "Most Important Feature"]
    )
    return df, most_important_names


def select_pca_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Select columns from X based on feature_names.
    """
    return X.loc[:, feature_names]


def save_feature_engineered_data(
    df: pd.DataFrame,
    y: pd.Series,
) -> None:
    """
    Save the feature engineered DataFrame and target variable to a CSV file.
    Args:
        df (pd.DataFrame): Feature engineered DataFrame.
        y (pd.Series): Target variable Series.
    """
    output_path: str = FeatureEngineeringConfig.feature_engineering_dir
    if not output_path:
        raise ValueError("Output path is not set in the configuration file.")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        y.to_csv(output_path.replace(".csv", "_target.csv"), index=False)
        logger.info(f"Feature engineered data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving feature engineered data: {e}")
        raise e
