from loguru import logger
from zenml.pipelines import pipeline

from src.components.data_ingestion import (
    create_booking_date,
    handle_data_types,
    handling_null_values,
    load_raw_data,
    remove_duplicates,
    remove_irrelevant_columns,
    save_cleaned_data,
)


@pipeline(enable_cache=False)
def data_ingestion_pipeline() -> None:
    """
    Pipeline for data ingestion and preprocessing.
    """
    logger.info("Starting data ingestion pipeline")
    raw_data = load_raw_data()
    cleaned_date_data = create_booking_date(raw_data)
    removed_duplicates_data = remove_duplicates(cleaned_date_data)
    no_null_data = handling_null_values(removed_duplicates_data)
    no_irrelevant_data = remove_irrelevant_columns(no_null_data)
    cleaned_data = handle_data_types(no_irrelevant_data)
    save_cleaned_data(cleaned_data)


if __name__ == "__main__":
    data_ingestion_pipeline()
