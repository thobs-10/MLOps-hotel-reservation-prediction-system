from typing import List

from pydantic import BaseModel


class FeatureEngineeringConstants(BaseModel):
    """
    Configuration for feature engineering.
    """

    categorical_columns: List[str] = [
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
    ]
    numerical_columns: List[str] = [
        "lead_time",
        "no_of_weekend_nights",
        "no_of_week_nights",
    ]
    target_column: str = "booking_status"
    new_features: List[str] = ["total_nights_stayed", "is_repeat_guest"]
    label_encoder_columns: List[str] = [
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
    ]
    pca_components: int = 2


# class DataPreprocessingConstants(BaseModel):
#     """Irrelevant columns for data preprocessing."""

irrelevant_columns: str = "Booking_ID"
