from loguru import logger
from zenml.pipelines import pipeline

from src.components.data_ingestion import (
    handle_data_types,
    handling_null_values,
    load_raw_data,
    remove_duplicates,
    remove_irrelevant_columns,
    save_cleaned_data,
)


@pipeline(enable_cache=True)
def data_ingestion_pipeline() -> None:
    """
    Pipeline for data ingestion and preprocessing.
    """
    logger.info("Starting data ingestion pipeline")
    raw_data = load_raw_data()
    no_duplicates = remove_duplicates(raw_data)
    cleaned_data = handling_null_values(no_duplicates)
    cleaned_data = remove_irrelevant_columns(cleaned_data)
    cleaned_data = handle_data_types(cleaned_data)
    save_cleaned_data(cleaned_data)


if __name__ == "__main__":
    data_ingestion_pipeline()
