from loguru import logger

from src.pipeline.data_ingestion_pipeline import data_ingestion_pipeline
from src.pipeline.feature_engineering_pipeline import feature_engineering_pipeline
from src.pipeline.training_model_pipeline import training_pipeline


def run_pipelines() -> None:
    """
    Run all pipelines in sequence.
    """
    logger.info("Starting data ingestion pipeline")
    data_ingestion_pipeline()

    logger.info("Starting feature engineering pipeline")
    feature_engineering_pipeline()

    logger.info("Starting training model pipeline")
    training_pipeline()

    logger.info("All pipelines executed successfully")


if __name__ == "__main__":
    run_pipelines()
    logger.info("Pipelines execution completed")
