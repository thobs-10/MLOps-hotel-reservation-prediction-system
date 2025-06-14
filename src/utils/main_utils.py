from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get categorical columns from the dataframe
    """
    categorical_columns: List[str] = df.select_dtypes(
        include=["object"]
    ).columns.tolist()
    return categorical_columns


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get numerical columns from the dataframe
    """
    numerical_columns: List[str] = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    return numerical_columns


def generate_label_encoder() -> LabelEncoder:
    """
    Generate a LabelEncoder instance for encoding categorical variables.
    Returns:
        LabelEncoder: Instance of LabelEncoder.
    """
    label_encoder = LabelEncoder()
    return label_encoder


def create_feature_importance_selector(
    X: pd.DataFrame, y: pd.Series
) -> ExtraTreesClassifier:
    """
    Create a feature importance selector using ExtraTreesClassifier.
    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable Series.
    Returns:
        ExtraTreesClassifier: Fitted ExtraTreesClassifier instance.
    """
    if X.empty or y.empty:
        raise ValueError("Features DataFrame or target variable Series is empty.")
    selector = ExtraTreesClassifier(n_estimators=100)
    selector.fit(X, y)
    return selector


def fit_pca(
    X: pd.DataFrame, columns: List[str], n_components: int = 4
) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA on selected columns of X.
    Returns the fitted PCA object and the transformed X.
    """
    pca = PCA(n_components=n_components)
    X_selected = X[columns]
    X_pca = pca.fit_transform(X_selected)
    return pca, X_pca
