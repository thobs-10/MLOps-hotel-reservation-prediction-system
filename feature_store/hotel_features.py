from datetime import timedelta
import feast
from feast.types import Float32, Int64, String, UnixTimestamp
import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String
from loguru import logger

# project for the feature repo
project = Project(
    name="hotel_booking_features",
    description="A project for hotel booking features for ML models",
)

# 1. Define an entity for the hotel booking.
booking = Entity(name="booking", join_keys=["booking_id"])

# 2. Define your data source
hotel_stats_source = FileSource(
    name="hotel_booking_data_source",
    path="data/feature_store/feast_hotel_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",  # Optional, but good for tracking data freshness
)

# 3. Define your core FeatureView for hotel booking statistics
hotel_stats_fv = FeatureView(
    name="hotel_booking_stats",
    entities=[booking],  # Associate with the 'booking' entity
    ttl=timedelta(days=7),  # Increased TTL for historical data availability.
    schema=[
        Field(
            name="event_timestamp",
            dtype=UnixTimestamp,
            description="Timestamp of the event",
        ),
        Field(
            name="lead_time",
            dtype=Int64,
            description="Number of days between booking and arrival",
        ),
        Field(
            name="arrival_month",
            dtype=Int64,
            description="Month of the year for arrival",
        ),
        Field(
            name="arrival_date",
            dtype=Int64,
            description="Day of the month for arrival",
        ),
        Field(
            name="avg_price_per_room",
            dtype=Float32,
            description="Average price paid per room per night",
        ),
        Field(
            name="no_of_special_requests",
            dtype=Int64,
            description="Number of special requests made by the customer",
        ),
    ],
    online=True,  # features available for online serving
    source=hotel_stats_source,
    tags={
        "team": "booking_ml",
        "model_use": "cancellation_prediction",
    },
)

# 4. Define a RequestSource for real-time input features
hotel_booking_request_source = RequestSource(
    name="hotel_booking_input_request",
    schema=[
        Field(name="lead_time_current", dtype=Int64),
        Field(name="avg_price_per_room_current", dtype=Float32),
        Field(name="no_of_special_requests_current", dtype=Int64),
        Field(name="current_promotion_discount", dtype=Float32),
    ],
)


# 5. Define an On-Demand Feature View (ODFV) for transformations
@on_demand_feature_view(
    sources=[hotel_stats_fv, hotel_booking_request_source],
    schema=[
        Field(
            name="price_per_night_per_request",
            dtype=Float64,
            description="Average price per room per night per special request",
        ),
        Field(
            name="discounted_price_per_room",
            dtype=Float64,
            description="Calculated price after applying current discount",
        ),
    ],
)
def transformed_hotel_features(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["price_per_night_per_request"] = inputs["avg_price_per_room"] / (
        inputs["no_of_special_requests"] + 0.001
    )  # epsilon to avoid division by zero
    df["discounted_price_per_room"] = inputs["avg_price_per_room_current"] * (
        1 - inputs["current_promotion_discount"]
    )
    return df


# 6. Define Feature Services to group features for specific models
# Feature Service for a basic booking prediction model
hotel_prediction_v1 = FeatureService(
    name="hotel_prediction_v1",
    features=[
        hotel_stats_fv[
            [
                "event_timestamp",
                "lead_time",
                "arrival_month",
                "arrival_date",
                "avg_price_per_room",
                "no_of_special_requests",
            ]
        ],
        transformed_hotel_features,
    ],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path="data/feature_logs")
    ),
)

# Another Feature Service, perhaps for a more advanced model or a different version
hotel_prediction_v2 = FeatureService(
    name="hotel_prediction_v2",
    features=[
        hotel_stats_fv,
        transformed_hotel_features,
    ],
)

# 7. Define a PushSource for real-time updates
hotel_stats_push_source = PushSource(
    name="hotel_stats_push_source",
    batch_source=hotel_stats_source,
)

# 8. Define a FeatureView that leverages the PushSource for ultra-fresh features
hotel_stats_fresh_fv = FeatureView(
    name="hotel_booking_stats_fresh",
    entities=[booking],
    ttl=timedelta(seconds=3600),  # Shorter TTL for very fresh data (e.g., 1 hour)
    schema=[
        Field(name="lead_time", dtype=Int64),
        Field(name="arrival_month", dtype=Int64),
        Field(name="arrival_date", dtype=Int64),
        Field(name="avg_price_per_room", dtype=Float32),
        Field(name="no_of_special_requests", dtype=Int64),
    ],
    online=True,
    source=hotel_stats_push_source,
    tags={"team": "booking_ml", "data_freshness": "realtime"},
)


# 9. On-demand feature view using the fresh feature view
@on_demand_feature_view(
    sources=[hotel_stats_fresh_fv, hotel_booking_request_source],
    schema=[
        Field(name="fresh_price_per_night_per_request", dtype=Float64),
        Field(name="fresh_discounted_price_per_room", dtype=Float64),
    ],
)
def transformed_hotel_features_fresh(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["fresh_price_per_night_per_request"] = inputs["avg_price_per_room"] / (
        inputs["no_of_special_requests"] + 0.001
    )
    df["fresh_discounted_price_per_room"] = inputs["avg_price_per_room_current"] * (
        1 - inputs["current_promotion_discount"]
    )
    return df


# 10. Feature Service for models requiring the freshest features
hotel_prediction_v3 = FeatureService(
    name="hotel_prediction_v3",
    features=[hotel_stats_fresh_fv, transformed_hotel_features_fresh],
)
