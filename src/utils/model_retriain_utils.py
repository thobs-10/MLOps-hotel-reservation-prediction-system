import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

DRIFT_LOG_PATH = os.getenv("DRIFT_LOG_PATH", "src/logs/drift_logs.csv")


def calculate_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: Optional[list] = None,
) -> float:
    """
    Calculate data drift using Kolmogorov-Smirnov test for numerical features.
    Returns average p-value (lower means more drift).
    """
    if feature_columns is None:
        feature_columns = [
            col
            for col in reference_df.columns
            if reference_df[col].dtype in ["float64", "int64"]
        ]
    p_values = []
    for col in feature_columns:
        if col in current_df.columns:
            stat, p = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
            p_values.append(p)
    if not p_values:
        return 1.0  # No drift if no features
    drift_score = 1 - sum(p_values) / len(p_values)  # Lower p-value = more drift
    return drift_score


def calculate_model_drift(
    reference_preds: pd.Series, current_preds: pd.Series
) -> float:
    """
    Calculate model drift as the difference in prediction distributions.
    Returns absolute difference in mean predictions.
    """
    return abs(reference_preds.mean() - current_preds.mean())


def calculate_prediction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate accuracy score.
    """
    return float(accuracy_score(y_true, y_pred))


def log_data_drift(
    drift_score: float, threshold: float, details: Optional[Dict] = None
) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "data_drift",
        "drift_score": drift_score,
        "threshold": threshold,
        "details": details or {},
    }
    _append_log(entry)


def log_model_drift(
    model_score: float, threshold: float, details: Optional[Dict] = None
) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "model_drift",
        "model_score": model_score,
        "threshold": threshold,
        "details": details or {},
    }
    _append_log(entry)


def log_prediction_accuracy(
    accuracy: float, threshold: float, details: Optional[Dict] = None
) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "prediction_accuracy",
        "accuracy": accuracy,
        "threshold": threshold,
        "details": details or {},
    }
    _append_log(entry)


# --- Example usage ---
def monitor_drift_and_accuracy(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_preds: pd.Series,
    current_preds: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Tuple[float, float]:
    """
    Calculate and log drift and accuracy metrics.
    """
    data_drift_score = calculate_data_drift(reference_df, current_df)
    model_drift_score = calculate_model_drift(reference_preds, current_preds)
    accuracy = calculate_prediction_accuracy(y_true, y_pred)
    log_data_drift(data_drift_score, threshold=0.2, details={"method": "ks_2samp"})
    log_model_drift(model_drift_score, threshold=0.1, details={"method": "mean_diff"})
    log_prediction_accuracy(
        accuracy, threshold=0.8, details={"method": "accuracy_score"}
    )
    return data_drift_score, accuracy


def _append_log(entry: dict) -> None:
    """
    Append a log entry to the drift log CSV file.
    """
    try:
        df = pd.DataFrame([entry])
        if not os.path.exists(DRIFT_LOG_PATH):
            df.to_csv(DRIFT_LOG_PATH, index=False)
        else:
            df.to_csv(DRIFT_LOG_PATH, mode="a", header=False, index=False)
    except Exception as e:
        logger.error(f"Failed to log drift/accuracy: {e}")
