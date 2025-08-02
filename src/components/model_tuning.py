import multiprocessing as mp
from typing import Any, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from zenml.steps import step

from src.entity.constants import randomcv_models

tracking_uri = mlflow.get_tracking_uri()
mlflow_client = MlflowClient(tracking_uri=tracking_uri)


@step
def perform_hyperparameter_tuning(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series,
    best_model: Any,
    threshold: float = 0.850,
) -> Tuple[str, Any]:
    """
    Perform hyperparameter tuning for the model.
    This function is a placeholder and should be implemented with actual logic.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target variable.
    Returns:
        None
    """
    mlflow.set_experiment("Hyperparameter Tuning Experiment")
    mlflow.set_tag("model-tuning", "v1.0.0")
    logger.info("Performing hyperparameter tuning...")
    if best_model is None:
        raise ValueError("No model provided for hyperparameter tuning.")
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    model_tuple = [
        item for item in randomcv_models if item[0] == best_model.__class__.__name__
    ]
    if not model_tuple:
        raise ValueError(
            f"Model {best_model.__class__.__name__} not found in randomcv_models."
        )
    model_name, model_class, search_space = model_tuple[0]
    randomized_cv_model = RandomizedSearchCV(
        estimator=model_class,
        param_distributions=search_space,
        n_iter=10,
        cv=k_fold,
        verbose=1,
        n_jobs=mp.cpu_count(),
        random_state=42,
        scoring="accuracy",
    )

    logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
    logger.info("Fitting the model with hyperparameter tuning...")
    randomized_cv_model.fit(X_train_scaled, y_train)
    best_model = randomized_cv_model.best_estimator_
    best_model.set_params(**randomized_cv_model.best_params_)
    y_pred = best_model.predict(X_test_scaled)
    signature = infer_signature(X_test_scaled, y_pred)
    mlflow.log_metric("accuracy", float(accuracy_score(y_test, y_pred)))
    if float(accuracy_score(y_test, y_pred)) < threshold:
        logger.warning(
            f"Model accuracy {accuracy_score(y_test, y_pred)} is below the threshold of {threshold}."
        )
    try:
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="best_hotel_reservation_model",
        )
    except Exception as e:
        logger.error(f"Error logging model: {e}")
        raise

    return mlflow.active_run().info.run_id, best_model
