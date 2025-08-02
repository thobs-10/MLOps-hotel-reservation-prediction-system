from loguru import logger
from zenml.pipelines import pipeline

from src.components.model_training import (
    feature_scaling,
    load_training_features,
    save_model_artifacts,
    split_data,
    train_models,
)
from src.components.model_tuning import perform_hyperparameter_tuning
from src.components.model_validation import validate_model


@pipeline(enable_cache=False)
def training_pipeline() -> None:
    """
    Pipeline for training machine learning models.
    """
    logger.info("Starting training pipeline")
    features_df, target_df = load_training_features()
    # features_df, target_df = load_data_from_feature_store()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        features_df,
        target_df,
    )
    X_train_scaled, X_test_scaled, X_valid_scaled, scaler = feature_scaling(
        X_train,
        X_test,
        X_valid,
    )
    best_model_name, best_model_score, best_model = train_models(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )
    model_run_id, best_model = perform_hyperparameter_tuning(
        X_train_scaled,
        X_test_scaled,
        y_test,
        y_train,
        best_model,
    )
    validate_model(
        model_run_id,
        best_model,
        X_test_scaled,
        y_test,
    )
    save_model_artifacts(
        scaler,
        best_model_name,
        best_model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        X_valid_scaled,
        y_valid,
    )


if __name__ == "__main__":
    training_pipeline()
    logger.info("Training pipeline completed successfully")
