from loguru import logger
from zenml.pipelines import pipeline

from src.components.data_validation_improved import (
    generate_validation_report,
    validate_cleaned_data,
)
from src.components.feature_engineering import (
    encode_categorical_columns,
    generate_new_features,
    get_important_features,
    load_processed_data,
    save_feature_engineered_data,
    separate_data,
)


@pipeline(enable_cache=False)
def feature_engineering_pipeline() -> None:
    """
    Pipeline for feature engineering.
    """
    logger.info("Starting feature engineering pipeline")
    processed_data = load_processed_data()
    try:
        # Use auto-clean to handle duplicates and other common issues
        validated_df = validate_cleaned_data(
            processed_data, strict=False, auto_clean=False
        )
        report = generate_validation_report(validated_df)
        logger.info(f"Validation report: \n{report}")
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise
    processed_data = generate_new_features(validated_df)
    processed_data, label_encoder = encode_categorical_columns(processed_data)
    X, y = separate_data(processed_data, label_encoder)
    important_features_df, feature_names, df_copy = get_important_features(X, y)
    logger.info(f"Important features identified: {feature_names}")
    save_feature_engineered_data(important_features_df, df_copy, y)


if __name__ == "__main__":
    feature_engineering_pipeline()
