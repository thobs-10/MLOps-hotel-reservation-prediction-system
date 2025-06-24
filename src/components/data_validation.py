import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd
from zenml.steps import step


def no_duplicates(df: pd.DataFrame) -> bool:
    """Check that there are no duplicate rows in the DataFrame."""
    return not df.duplicated().any()


# Define the schema for the cleaned DataFrame
def define_cleaned_schema() -> DataFrameSchema:
    """
    Define the schema for the cleaned DataFrame.
    Returns:
        DataFrameSchema: Schema for the cleaned DataFrame.
    """
    return DataFrameSchema(
        columns={
            "lead_time": Column(pa.Int, Check.ge(0), nullable=False),
            "avg_price_per_room": Column(pa.Float, Check.ge(0), nullable=False),
            "no_of_special_requests": Column(pa.Int, Check.ge(0), nullable=False),
            "arrival_date": Column(pa.Int, Check.in_range(1, 32), nullable=False),
            "arrival_month": Column(pa.Int, Check.in_range(1, 13), nullable=False),
            "arrival_year": Column(pa.Int, Check.in_range(2000, 2100), nullable=False),
            "market_segment_type": Column(pa.Int, nullable=False),
            "room_type_reserved": Column(pa.Int, nullable=False),
            "type_of_meal_plan": Column(pa.Int, nullable=False),
            "total_nights_stayed": Column(pa.Int, Check.ge(0), nullable=False),
            "no_of_week_nights": Column(pa.Int, Check.ge(0), nullable=False),
            "no_of_weekend_nights": Column(pa.Int, Check.ge(0), nullable=False),
            "repeated_guest": Column(pa.Int, Check.isin([0, 1]), nullable=False),
            "required_car_parking_space": Column(
                pa.Int, Check.isin([0, 1]), nullable=False
            ),
            "booking_status": Column(
                pa.String, Check.isin(["Canceled", "Not_Canceled"]), nullable=False
            ),
        },
        checks=Check(no_duplicates, error="Duplicate rows found in the DataFrame."),
    )


@step(enable_cache=False)
def validate_cleaned_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the cleaned DataFrame using pandera schema.
    Raises an error if validation fails.
    """
    cleaned_schema = define_cleaned_schema()
    return cleaned_schema.validate(df)
