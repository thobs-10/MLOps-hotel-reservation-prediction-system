import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from joblib import Memory
from loguru import logger
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from zenml.steps import step

from src.entity.config_entity import (
    FeatureEngineeringConfig,
    FeatureStoreConfig,
    ModelTrainingConfig,
)
from src.entity.constants import models

mlflow.set_tracking_uri("http://localhost:8085")
mlflow.set_experiment("hotel_reservation_prediction_experiments")

memory = Memory(location="cachedir", verbose=0)


@step
def load_data_from_feature_store() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load features from the feature store using the defined feature service.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature DataFrame and target Series
    """

    try:
        logger.info("Loading features from feature store")

        # First, let's load the actual data to create a proper entity spine
        feast_data = pd.read_parquet("data/feature_store/feast_hotel_features.parquet")

        # Create entity spine from actual data to ensure we get matching records
        entity_spine = feast_data[["booking_id", "event_timestamp"]].copy()

        logger.info(f"Created entity spine with {len(entity_spine)} records")
        logger.info(f"Entity spine columns: {list(entity_spine.columns)}")
        logger.info(f"Entity spine sample:\n{entity_spine.head()}")

        fs = FeatureStore(repo_path=FeatureStoreConfig.feature_store_repo_path)

        # Get historical features using the proper entity spine
        feature_vector = fs.get_historical_features(
            entity_df=entity_spine,
            features=fs.get_feature_service("hotel_prediction_v1"),
        ).to_df()

        logger.info(f"Retrieved feature vector with shape: {feature_vector.shape}")
        logger.info(f"Feature vector columns: {list(feature_vector.columns)}")
        logger.info(f"Feature vector sample:\n{feature_vector.head()}")

        # Check if we got any data
        if feature_vector.empty:
            logger.error(
                "Feature vector is empty! This suggests an issue with the entity matching."
            )
            # Fallback to direct data loading
            logger.info("Falling back to direct data loading...")
            return load_training_features()

        # Extract target variable
        target = feature_vector.pop("booking_status")

        # Drop entity and timestamp columns for training
        feature_vector = feature_vector.drop(
            columns=["booking_id", "event_timestamp", "booking_date"], errors="ignore"
        )

        logger.info(f"Final feature vector shape: {feature_vector.shape}")
        logger.info(f"Target shape: {target.shape}")

        return feature_vector, target

    except Exception as e:
        logger.error(f"Error loading features from feature store: {e}")
        logger.info("Falling back to direct data loading...")
        # Fallback to the direct loading method
        return load_training_features()


@step(enable_cache=True)
def load_training_features() -> Tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Load features from the feature store using the defined feature service.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature DataFrame and target Series
    """
    try:
        logger.info("Loading features from feature store")
        features_df = pd.read_parquet(
            os.path.join(
                FeatureEngineeringConfig.feature_engineering_dir,
                "feast_hotel_features.parquet",
            )
        )
        target_df = pd.read_csv(
            os.path.join(
                FeatureEngineeringConfig.feature_engineering_dir,
                "hotel_target_backup.csv",
            )
        )
        features_for_training = features_df.drop(
            columns=["booking_id", "event_timestamp", "booking_status", "booking_date"],
            errors="ignore",
        )

        # Convert target DataFrame to Series
        if isinstance(target_df, pd.DataFrame):
            target_series = (
                target_df.squeeze()
            )  # Convert single-column DataFrame to Series
            if not isinstance(target_series, pd.Series):
                # If squeeze() returns a scalar, create a Series from the DataFrame
                target_series = pd.Series(
                    target_df.iloc[:, 0].values, name=target_df.columns[0]
                )
        else:
            target_series = (
                pd.Series(target_df)
                if not isinstance(target_df, pd.Series)
                else target_df
            )

        logger.info(f"Loaded features shape: {features_for_training.shape}")
        logger.info(f"Target length: {len(target_series)}")
        return features_for_training, target_series

    except Exception as e:
        logger.error(f"Error loading features from feature store: {e}")
        raise


@step
def split_data(
    features_df: pd.DataFrame,
    target: pd.Series,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """split data into training and testing sets"""
    # Convert target to a Series
    # target_series = target_df.squeeze()
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_df,
        target,
        train_size=0.7,
        random_state=42,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.3,
        random_state=42,
    )
    logger.info(
        f"Training set shape: {X_train.shape}, Validation set shape: {X_valid.shape}"
    )
    logger.info(f"Test set shape: {X_test.shape}, Test target shape: {y_test.shape}")
    logger.info(
        f"Validation target shape: {y_valid.shape}, Training target shape: {y_train.shape}"
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


@step
@memory.cache
def feature_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Any,
]:
    """
    Apply feature scaling to the training and validation datasets.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Validation features.
        X_valid (pd.DataFrame): Test features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and validation features.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_valid_scaled = scaler.transform(X_valid)

    logger.info("Feature scaling applied")
    return (
        pd.DataFrame(X_train_scaled, columns=X_train.columns),
        pd.DataFrame(X_test_scaled, columns=X_test.columns),
        pd.DataFrame(X_valid_scaled, columns=X_valid.columns),
        scaler,
    )


def _get_best_model(results: dict) -> Tuple[str, float, Any]:
    if not results:
        return "None", 0.0, None

    best_model_name = max(results, key=lambda k: results[k][0])
    best_model_metrics = results[best_model_name][0]
    best_model = results[best_model_name][2]

    return best_model_name, best_model_metrics, best_model


def _train_and_save(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model: Any,
    model_name: str,
) -> Tuple[dict, str, Any]:
    """Train a single model and save it with Mlflow.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training features.
        X_test_scaled (pd.DataFrame): Scaled validation features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Validation target variable.
        model: The model to train.
        model_name (str): Name of the model.

    Returns:
        Tuple[dict, str]: Metrics dictionary and model path.
    """
    with mlflow.start_run() as run:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy_score_value = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy_score", float(accuracy_score_value))
        mlflow.log_param("model_name", type(model).__name__)
        signature = infer_signature(X_test_scaled, predictions)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        metrics = {
            "accuracy": float(accuracy_score_value),
            "model_name": model_name,
            "run_id": run.info.run_id,
        }
        model_path = f"runs:/{run.info.run_id}/model"
        logger.info(
            f"Trained {type(model).__name__} with accuracy: {accuracy_score_value}"
        )
        mlflow.end_run()
        return metrics, model_path, model


def _save_trained_model(best_model_name: str, best_model: Any) -> None:
    """Save the trained model to a file.

    Args:
        best_model_name (str): Name of the best model.
        best_model (Any): The trained model instance.
    """
    import joblib

    trained_model_artifact_dir = ModelTrainingConfig.model_artifact_dir
    os.makedirs(trained_model_artifact_dir, exist_ok=True)
    model_path = f"{trained_model_artifact_dir}/{best_model_name}.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")


@step
def train_models(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[str, float, Any]:
    """Train multiple models in parallel and track with Mlflow.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training features.
        X_test_scaled (pd.DataFrame): Scaled validation features.
        y_train (pd.Series): Training target variable.

    Returns:
        Tuple[str, float, dict]: Best model name, best model score, and all model scores.
    """
    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_model = {
            executor.submit(
                _train_and_save,
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                model,
                model_name,
            ): model_name
            for model_name, model in models.items()
        }

        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                metrics, model_path, model = future.result()
                results[model_name] = (metrics["accuracy"], model_path, model)
            except Exception as e:
                logger.error(f"Error training model {model_name}: {e}")

    best_model_name, best_metrics, best_model = _get_best_model(results)
    if best_model_name:
        logger.info(f"Best model: {best_model_name} with score: {best_metrics}")
        _save_trained_model(best_model_name, best_model)
        return best_model_name, best_metrics, best_model
    else:
        logger.warning("No models were trained successfully.")
        return "None", 0.0, None


@step
def save_model_artifacts(
    scaler: Any,
    best_model_name: str,
    best_model: Any,
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    X_valid_scaled: pd.DataFrame,
    y_valid: pd.Series,
) -> None:
    """Save the data and model artifacts.

    Args:
        scaler (Any): Scaler used for feature scaling.
        best_model_name (str): Name of the best model.
        best_model (Any): The trained model instance.
        X_train_scaled (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target variable.
        X_test_scaled (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target variable.
        X_valid_scaled (pd.DataFrame): Scaled validation features.
        y_valid (pd.Series): Validation target variable.
        standard_scaler (Optional[Any]): StandardScaler instance if used.
    """
    import joblib

    model_artefact_dir = ModelTrainingConfig.model_artifact_dir
    model_registry_path = ModelTrainingConfig.model_registry_path
    # Save the best model
    os.makedirs(model_registry_path, exist_ok=True)
    best_model_path = os.path.join(model_registry_path, f"{best_model_name}.pkl")
    best_model_pipeline = make_pipeline(scaler, best_model)
    joblib.dump(best_model_pipeline, best_model_path)
    # Save training, validation and test data
    os.makedirs(model_artefact_dir, exist_ok=True)
    X_train_scaled.to_csv(os.path.join(model_artefact_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(model_artefact_dir, "X_test.csv"), index=False)
    X_valid_scaled.to_csv(os.path.join(model_artefact_dir, "X_valid.csv"), index=False)

    y_train.to_csv(os.path.join(model_artefact_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(model_artefact_dir, "y_test.csv"), index=False)
    y_valid.to_csv(os.path.join(model_artefact_dir, "y_valid.csv"), index=False)
    logger.info(f"Model saved to {best_model_path}")
