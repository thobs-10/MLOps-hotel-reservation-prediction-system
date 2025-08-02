from dataclasses import dataclass
from typing import List

from combo.models.classifier_stacking import Stacking
from pydantic import BaseModel
from scipy.stats import (
    loguniform,
    randint,
    uniform,
)
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


class FeatureEngineeringConstants(BaseModel):
    """
    Configuration for feature engineering.
    """

    categorical_columns: List[str] = [
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
    ]
    numerical_columns: List[str] = [
        "lead_time",
        "no_of_weekend_nights",
        "no_of_week_nights",
    ]
    target_column: str = "booking_status"
    new_features: List[str] = ["total_nights_stayed", "is_repeat_guest"]
    label_encoder_columns: List[str] = [
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
    ]
    pca_components: int = 5


irrelevant_columns: str = "Booking_ID"


@dataclass
class ModelConfig:
    name: str
    estimator: BaseEstimator
    params: dict
    needs_scaling: bool = True


classifiers = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    LogisticRegression(max_iter=1000),
    SVC(probability=True),
    KNeighborsClassifier(),
]
# stacked_clf = Stacking(base_estimators=classifiers)
models = {
    "stacked_clf": Stacking(base_estimators=classifiers),
    "random_forest": RandomForestClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "ada_boost": AdaBoostClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svc": SVC(probability=True),
    "knn": KNeighborsClassifier(),
}

search_space = {
    "RandomForestClassifier": {
        "n_estimators": randint(100, 500),  # Random integer between 100 and 500
        "max_depth": randint(10, 50),  # Random integer between 10 and 50
        "min_samples_split": randint(2, 10),  # Random integer between 2 and 10
        "min_samples_leaf": randint(1, 5),  # Random integer between 1 and 5
        "criterion": ["gini", "entropy"],  # Categorical (no distribution)
    },
    "DecisionTreeClassifier": {
        "max_depth": randint(10, 50),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "criterion": ["gini", "entropy"],
    },
    "GradientBoostingClassifier": {
        "n_estimators": randint(100, 500),
        "learning_rate": loguniform(
            1e-3, 1e-1
        ),  # Log-uniform distribution between 0.001 and 0.1
        "max_depth": randint(10, 50),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
    },
    "LogisticRegression": {
        "C": loguniform(1e-2, 1e2),  # Log-uniform distribution between 0.01 and 100
        "solver": ["lbfgs", "liblinear", "sag", "saga"],
    },
    "KNeighborsClassifier": {
        "n_neighbors": randint(1, 31),  # Random integer between 1 and 30
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
    },
    "XGBClassifier": {
        "n_estimators": randint(100, 500),
        "learning_rate": loguniform(1e-3, 1e-1),
        "max_depth": randint(10, 50),
        "subsample": uniform(0.5, 0.5),  # Uniform distribution between 0.5 and 1.0
        "colsample_bytree": uniform(0.5, 0.5),
    },
    "CatBoostingClassifier": {
        "iterations": randint(100, 500),
        "learning_rate": loguniform(1e-3, 1e-1),
        "depth": randint(4, 11),
        "l2_leaf_reg": randint(1, 11),
    },
    "SVC": {
        "C": loguniform(1e-1, 1e1),  # Log-uniform distribution between 0.1 and 10
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
    },
    "AdaBoostClassifier": {
        "n_estimators": randint(50, 300),
        "learning_rate": loguniform(
            1e-3, 1e0
        ),  # Log-uniform distribution between 0.001 and 1.0
    },
    "Stacking": {
        "final_estimator": [
            "LogisticRegression",
            "RandomForestClassifier",
            "XGBClassifier",
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "KNeighborsClassifier",
            "SVC",
        ],
        "cv": randint(3, 10),  # Random integer between 3 and 10
        "n_jobs": [-1],  # Use all available cores
    },
}

randomcv_models = [
    ("XGBClassifier", XGBClassifier(), search_space["XGBClassifier"]),
    (
        "RandomForestClassifier",
        RandomForestClassifier(),
        search_space["RandomForestClassifier"],
    ),
    (
        "KNeighborsClassifier",
        KNeighborsClassifier(),
        search_space["KNeighborsClassifier"],
    ),
    # ("Decision_Tree", DecisionTreeClassifier(), search_space["Decision_Tree"]),
    (
        "GradientBoostingClassifier",
        GradientBoostingClassifier(),
        search_space["GradientBoostingClassifier"],
    ),
    (
        "LogisticRegression",
        LogisticRegression(),
        search_space["LogisticRegression"],
    ),
    (
        "SVC",
        SVC(),
        search_space["SVC"],
    ),
    (
        "AdaBoostClassifier",
        AdaBoostClassifier(),
        search_space["AdaBoostClassifier"],
    ),
    (
        "Stacking",
        Stacking(base_estimators=classifiers),
        search_space["Stacking"],
    ),
]
