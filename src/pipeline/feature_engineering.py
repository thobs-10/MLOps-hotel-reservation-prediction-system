from src.components.data_validation import validate_cleaned_data
from src.components.feature_engineering import (
    load_processed_data,
    generate_new_features,
    encode_categorical_columns,
    separate_data,
    get_important_features,
    get_pca_feature_importance,
    select_pca_features,
    save_feature_engineered_data,
)
from zenml.pipelines import pipeline
from loguru import logger


@pipeline(enable_cache=True)
def feature_engineering_pipeline() -> None:
    """
    Pipeline for feature engineering.
    """
    logger.info("Starting feature engineering pipeline")
    processed_data = load_processed_data()
    validated_df = validate_cleaned_data(processed_data)
    processed_data = generate_new_features(validated_df)
    processed_data = encode_categorical_columns(processed_data)
    X, y = separate_data(processed_data)
    important_features, feature_names = get_important_features(X, y)
    df, most_important_col_names = get_pca_feature_importance(X, feature_names)
    selected_features = select_pca_features(df, most_important_col_names)
    save_feature_engineered_data(X[selected_features], y)
