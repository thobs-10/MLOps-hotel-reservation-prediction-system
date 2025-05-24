import os

import pandas as pd
from loguru import logger
from zenml.steps import step

from src.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig
from src.entity.constants import irrelevant_columns
from src.utils.main_utils import get_categorical_columns, get_numerical_columns


@step(enable_cache=True)
def load_raw_data() -> pd.DataFrame:
    """
    Load data from a CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        file_path: str = DataIngestionConfig.raw_data_path
        logger.info(f"loading data from {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise e


@step
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to remove duplicates from.
    Returns:
        pd.DataFrame: DataFrame without duplicates.
    """
    df = df.drop_duplicates()
    logger.debug(f"DataFrame shape after removing duplicates: {df.shape}")
    return df


@step
def handling_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to handle missing values for.
    Returns:
        pd.DataFrame: DataFrame with handled missing values.
    """
    logger.info("Performing data cleaning")
    features_with_null_values = [
        features
        for features in df.columns
        if df[features].isnull().sum() >= 1
        and df[features].isnull().sum() < df.shape[0]
    ]
    for feature in features_with_null_values:
        if pd.api.types.is_numeric_dtype(df[feature]):
            df[feature] = df[feature].fillna(df[feature].mean())
        else:
            df[feature] = df[feature].fillna(df[feature].mode()[0])
    # Handle columns with all null values
    columns_with_all_nulls = [col for col in df.columns if df[col].isnull().all()]
    if columns_with_all_nulls:
        logger.exception(f"Columns with all null values: {columns_with_all_nulls}. ")
        raise ValueError(
            f"Columns with all null values: {columns_with_all_nulls}. "
            "Please check the data source."
        )
    logger.debug("Handled missing values successfully")
    return df


@step
def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove irrelevant columns from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to remove irrelevant columns from.
    Returns:
        pd.DataFrame: DataFrame without irrelevant columns.
    """
    if irrelevant_columns not in df.columns:
        logger.error(f"Irrelevant columns {irrelevant_columns} not found in DataFrame.")
        raise ValueError(
            f"Irrelevant columns {irrelevant_columns} not found in DataFrame."
        )
    df.drop(columns=irrelevant_columns, inplace=True)
    return df


@step
def handle_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle data types of the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to handle data types for.
    Returns:
        pd.DataFrame: DataFrame with handled data types.
    """
    cat_cols = get_categorical_columns(df)
    for col in [col for col in cat_cols if df[col].dtype == "object"]:
        df[col] = df[col].astype("category")

    num_cols = get_numerical_columns(df)
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_column = [
        col for col in df.columns if col not in num_cols and col not in cat_cols
    ]
    if invalid_column:
        logger.error(f"Invalid columns found: {invalid_column}.")
        raise ValueError(f"Invalid columns found: {invalid_column}.")
    return df


@step
def save_cleaned_data(df: pd.DataFrame) -> None:
    """
    Save the cleaned DataFrame to a CSV file.
    Args:
        df (pd.DataFrame): DataFrame to save.
    """
    try:
        output_path = DataPreprocessingConfig.processed_data_path
        if not output_path:
            raise ValueError("Output path is not set in the configuration.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving cleaned data: {e}")
        raise e
