from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, Int64, UnixTimestamp
from loguru import logger


def save_to_feast_store() -> None:
    """
    Save the feature engineered DataFrame to Feast.
    """

    # Define the source for the feature view
    # Only specify event_timestamp as the timestamp field, treat booking_date as a regular feature
    source = FileSource(
        path="/Users/thobelasixpence/Documents/mlops-zoomcamp-project-2024/MLOps-hotel-reservation-prediction-system/data/feature_store/feast_hotel_features.parquet",
        timestamp_field="event_timestamp",  # Use timestamp_field instead of event_timestamp_column
    )

    # Define the entity
    booking_entity = Entity(
        name="booking_id",
        value_type=ValueType.INT64,
        description="Unique identifier for each booking",
    )

    # Define the feature view with explicit schema
    hotel_feature_view = FeatureView(
        name="hotel_features",
        entities=[booking_entity],  # Only use booking_entity
        ttl=timedelta(days=7),  # Set proper TTL
        schema=[
            Field(name="lead_time", dtype=Int64),
            Field(name="avg_price_per_room", dtype=Float32),
            Field(name="no_of_special_requests", dtype=Int64),
            Field(name="arrival_date", dtype=Int64),
            Field(name="arrival_month", dtype=Int64),
            Field(name="booking_status", dtype=Int64),
            # Treat booking_date as a timestamp feature, not entity timestamp
            Field(name="booking_date", dtype=UnixTimestamp),
        ],
        source=source,
    )

    # Initialize the Feast store
    fs = FeatureStore(
        repo_path="/Users/thobelasixpence/Documents/mlops-zoomcamp-project-2024/MLOps-hotel-reservation-prediction-system/feature_store",
    )

    hotel_feature_service = FeatureService(
        name="hotel_prediction_v1",
        features=[hotel_feature_view],
    )

    # Apply the feature view to the store
    fs.apply([booking_entity, hotel_feature_view, hotel_feature_service])

    # Materialize the features
    try:
        fs.materialize_incremental(end_date=datetime.now() - timedelta(minutes=30))
        logger.info("✅ Feature view applied and materialized in Feast store.")
    except Exception as e:
        logger.warning(
            f"⚠️ Materialization warning (this is often normal for first run): {e}"
        )
        logger.info("✅ Feature view applied to Feast store (materialization skipped).")


def validate_feast_data(feast_df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame is properly formatted for Feast consumption.

    Args:
        feast_df: DataFrame to validate

    Returns:
        bool: True if valid, raises exception if not
    """
    required_columns = [
        "booking_id",
        "event_timestamp",
        "booking_date",
        # "created",
        "lead_time",
        "avg_price_per_room",
        "no_of_special_requests",
        "arrival_date",
        "arrival_month",
        "booking_status",
    ]

    # Check required columns
    missing_cols = set(required_columns) - set(feast_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for Feast: {missing_cols}")

    # Check data types
    expected_types = {
        "booking_id": ["int64"],
        "event_timestamp": ["datetime64[ns]"],
        "booking_date": ["datetime64[ns]"],
        "lead_time": ["int64"],
        "avg_price_per_room": ["float32", "float64"],
        "no_of_special_requests": ["int64"],
        "arrival_date": ["int64"],
        "arrival_month": ["int64"],
        "booking_status": ["int64"],
    }

    for col, valid_types in expected_types.items():
        if col in feast_df.columns:
            if str(feast_df[col].dtype) not in valid_types:
                logger.warning(
                    f"⚠️ Column '{col}' has dtype {feast_df[col].dtype}, expected one of {valid_types}"
                )

    # Check for null values in critical columns
    critical_cols = ["booking_id", "event_timestamp", "booking_date"]
    for col in critical_cols:
        if feast_df[col].isnull().any():
            raise ValueError(
                f"Column '{col}' contains null values, which is not allowed for Feast"
            )

    logger.info("✅ Feast data validation passed")
    return True


def prepare_feast_data(
    df: pd.DataFrame, X_copy: pd.DataFrame, y: np.ndarray[Any, Any]
) -> pd.DataFrame:
    """
    Prepare data in the format expected by Feast based on hotel_features.py schema.

    Args:
        df: Feature engineered DataFrame
        y: Target variable Series

    Returns:
        pd.DataFrame: Data formatted for Feast consumption
    """
    from datetime import datetime

    # Create a copy to avoid modifying original data
    feast_df = df.copy()

    # Add target variable
    feast_df["booking_status"] = y

    # Add required Feast columns
    current_time = datetime.now()
    feast_df["booking_id"] = range(len(feast_df))  # Generate booking IDs
    feast_df["event_timestamp"] = current_time
    # get the booking date from X_copy to feast_df
    feast_df["booking_date"] = X_copy["booking_date"].astype("datetime64[ns]")
    # feast_df["created"] = current_time

    # Ensure we have all the columns expected by Feast schema
    expected_feast_columns = [
        "booking_id",
        "event_timestamp",
        "booking_date",
        # "created",
        "lead_time",
        "avg_price_per_room",
        "no_of_special_requests",
        "arrival_date",
        "arrival_month",
        "booking_status",
    ]

    # Check for missing columns and handle them
    for col in expected_feast_columns:
        if col not in feast_df.columns:
            if col in ["booking_id", "event_timestamp", "booking_date"]:
                continue  # Already handled above
            else:
                logger.warning(
                    f"⚠️ Missing column '{col}' for Feast schema, filling with default values"
                )
                if col == "market_segment_type":
                    feast_df[col] = "Online"  # Default market segment
                else:
                    feast_df[col] = 0  # Default numeric value

    # Convert market_segment_type back to string if it was encoded
    if "market_segment_type" in feast_df.columns:
        if feast_df["market_segment_type"].dtype in ["int64", "int32"]:
            # Map encoded values back to strings (you may need to adjust this mapping)
            segment_mapping = {0: "Online", 1: "Offline", 2: "Corporate", 3: "Aviation"}
            feast_df["market_segment_type"] = feast_df["market_segment_type"].map(
                lambda x: segment_mapping.get(x, "Online")
            )

    # Select only the columns needed for Feast
    feast_df = feast_df[expected_feast_columns]

    # Ensure proper data types for Feast
    feast_df["booking_id"] = feast_df["booking_id"].astype("int64")
    feast_df["lead_time"] = feast_df["lead_time"].astype("int64")
    feast_df["event_timestamp"] = pd.to_datetime(
        feast_df["event_timestamp"], errors="coerce"
    )
    feast_df["booking_date"] = pd.to_datetime(feast_df["booking_date"], errors="coerce")
    feast_df["avg_price_per_room"] = feast_df["avg_price_per_room"].astype("float32")
    feast_df["no_of_special_requests"] = feast_df["no_of_special_requests"].astype(
        "int64"
    )
    feast_df["arrival_date"] = feast_df["arrival_date"].astype("int64")
    feast_df["arrival_month"] = feast_df["arrival_month"].astype("int64")

    # Ensure booking_status is properly encoded as int64
    if "booking_status" in feast_df.columns:
        if feast_df["booking_status"].dtype == "object":
            # If it's still string, encode it
            feast_df["booking_status"] = (
                feast_df["booking_status"]
                .map({"Not_Canceled": 0, "Canceled": 1})
                .fillna(0)
                .astype("int64")
            )
        else:
            feast_df["booking_status"] = feast_df["booking_status"].astype("int64")

    logger.info(
        f"📋 Prepared Feast data with {len(feast_df)} rows and columns: {list(feast_df.columns)}"
    )

    return feast_df
