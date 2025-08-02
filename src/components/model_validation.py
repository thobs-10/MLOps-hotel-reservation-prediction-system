from typing import Any

import mlflow
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from zenml.steps import step


@step
def validate_model(
    model_run_id: str,
    best_model: Any,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    deployment_threshold: float = 0.85,
) -> None:
    """
    Validate the trained model using the test dataset.

    Args:
        run_id (str): The MLflow run ID for tracking.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target labels.
        deployment_threshold (float): Minimum accuracy threshold for deployment.
    """
    logger.info("Validating model performance on test set")

    if X_test_scaled.empty or y_test.empty:
        raise ValueError("Test data is empty. Cannot validate model.")

    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"📊 Test set accuracy: {test_accuracy:.4f}")
    logger.info(
        f"📋 Test set classification report:\n{classification_report(y_test, y_pred)}"
    )
    if test_accuracy < deployment_threshold:
        raise ValueError(
            f"Model accuracy {test_accuracy:.4f} below threshold {deployment_threshold}"
        )
    else:
        logger.info(
            f"✅ Model meets deployment threshold: {test_accuracy:.4f} >= {deployment_threshold}"
        )
        _register_model(model_run_id, model_name="best_hotel_reservation_model")
        logger.info(f"Model registered successfully with run ID: {model_run_id}")


def _register_model(
    model_run_id: str,
    model_name: str = "best_model",
) -> None:
    """
    Register the model with MLflow.
    Args:
        model_run_id (str): The MLflow model_run_id ID.
        model_name (str): The name of the model to register.
    """
    try:
        mlflow.register_model(
            "runs:/{}/model".format(model_run_id),
            model_name,
        )
        logger.info(f"Model registered as {model_name} with run ID {model_run_id}")
    except Exception as e:
        logger.error(f"Error registering model {model_name}: {e}")
