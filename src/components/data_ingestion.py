import calendar
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
        if not file_path:
            raise ValueError("File path is not set in the configuration.")
        logger.info(f"loading data from {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise e


def _fix_and_create_booking_date(day, month, year):
    """
    Fix invalid date components and create a valid booking date.

    Args:
        day: Day of the month
        month: Month of the year
        year: Year

    Returns:
        pd.Timestamp or pd.NaT: Valid booking date or NaT if unfixable
    """
    try:
        # Convert to integers, handle NaN
        if pd.isna(day) or pd.isna(month) or pd.isna(year):
            return pd.NaT

        day, month, year = int(day), int(month), int(year)

        # Fix invalid years (use reasonable default if invalid)
        if year < 2017 or year > 2019:
            year = 2018  # Default to reasonable year

        # Fix invalid months
        if month < 1:
            month = 1
        elif month > 12:
            month = 12

        # Fix invalid days based on the month
        if day < 1:
            day = 1
        else:
            # Get maximum days in the month
            max_days = calendar.monthrange(year, month)[1]
            if day > max_days:
                day = max_days

        return pd.to_datetime(f"{day:02d}-{month:02d}-{year}", format="%d-%m-%Y")

    except Exception as e:
        logger.error(
            f"Error creating date for day={day}, month={month}, year={year}: {e}"
        )
        return pd.NaT


@step
def create_booking_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a booking_date column with intelligent error handling and date fixing.

    This function validates date components and intelligently fixes invalid dates:
    - Fixes impossible dates (Feb 30 -> Feb 28/29, Apr 31 -> Apr 30)
    - Handles out-of-range values (day > 31 -> last day of month)
    - Preserves all data rows (no data loss)

    Args:
        df (pd.DataFrame): DataFrame containing arrival_date, arrival_month, arrival_year columns

    Returns:
        pd.DataFrame: DataFrame with new booking_date column

    Raises:
        ValueError: If required date columns are missing
    """
    logger.info("🚀 Creating booking_date column with intelligent date fixing...")

    # Validate required columns
    required_cols = ["arrival_date", "arrival_month", "arrival_year"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    initial_rows = len(df)
    missing_stats = {col: df[col].isnull().sum() for col in required_cols}

    for col, missing_count in missing_stats.items():
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in {col}")

    invalid_days = ((df["arrival_date"] > 31) | (df["arrival_date"] < 1)).sum()
    invalid_months = ((df["arrival_month"] > 12) | (df["arrival_month"] < 1)).sum()
    invalid_years = ((df["arrival_year"] < 2018) | (df["arrival_year"] > 2019)).sum()
    if invalid_days > 0:
        logger.warning(f"Found {invalid_days} invalid day values - will be fixed")
    if invalid_months > 0:
        logger.warning(f"Found {invalid_months} invalid month values - will be fixed")
    if invalid_years > 0:
        logger.warning(f"Found {invalid_years} invalid year values - will be fixed")

    logger.info("📅 Applying intelligent date fixing...")
    df["booking_date"] = df.apply(
        lambda row: _fix_and_create_booking_date(
            row["arrival_date"], row["arrival_month"], row["arrival_year"]
        ),
        axis=1,
    )
    valid_dates = df["booking_date"].notna().sum()
    invalid_dates = df["booking_date"].isna().sum()

    logger.info("✅ Booking date creation completed!")
    logger.info(
        f"📊 Results: {initial_rows} total rows, {valid_dates} valid dates, {invalid_dates} invalid dates"
    )
    if valid_dates > 0:
        date_range = f"{df['booking_date'].min()} to {df['booking_date'].max()}"
        logger.info(f"📅 Date range: {date_range}")

    return df


@step
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to remove duplicates from.
    Returns:
        pd.DataFrame: DataFrame without duplicates.
    """
    initial_shape = df.shape
    cleaned_df = df.drop_duplicates()
    duplicates_removed = initial_shape[0] - cleaned_df.shape[0]
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")
    logger.debug(f"DataFrame shape after removing duplicates: {cleaned_df.shape}")
    return cleaned_df


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
    logger.info(f"Removed irrelevant columns: {irrelevant_columns}")
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
        if invalid_column[0] == "booking_date":
            df[invalid_column[0]] = pd.to_datetime(
                df[invalid_column[0]], errors="coerce"
            )
            invalid_column.remove(invalid_column[0])
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
        df.to_parquet(os.path.join(output_path, "cleaned_data.parquet"), index=False)
        logger.info(f"Cleaned data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving cleaned data: {e}")
        raise e
