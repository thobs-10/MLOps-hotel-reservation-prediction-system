from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from combo.models.classifier_stacking import Stacking
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier


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
    X: pd.DataFrame, y: np.ndarray
) -> ExtraTreesClassifier:
    """
    Create a feature importance selector using ExtraTreesClassifier.
    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable Series.
    Returns:
        ExtraTreesClassifier: Fitted ExtraTreesClassifier instance.
    """
    if X.empty:
        raise ValueError("Features DataFrame is empty.")
    if not isinstance(y, np.ndarray):
        raise ValueError("Target variable must be a numpy array.")
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


def create_stacking_classifier() -> "Stacking":
    """
    Create a stacking classifier with a set of base estimators.
    Returns:
        Stacking: Instance of Stacking classifier.
    """
    classifiers = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        LogisticRegression(max_iter=1000),
        SVC(probability=True),
        KNeighborsClassifier(),
    ]

    stacked_clf = Stacking(base_estimators=classifiers)
    return stacked_clf


def get_models() -> Dict[
    str,
    Union[
        Stacking,
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        XGBClassifier,
        LogisticRegression,
        SVC,
        KNeighborsClassifier,
    ],
]:
    stacked_clf = create_stacking_classifier()
    models = {
        "stacked_clf": stacked_clf,
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "ada_boost": AdaBoostClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svc": SVC(probability=True),
        "knn": KNeighborsClassifier(),
    }
    return models
