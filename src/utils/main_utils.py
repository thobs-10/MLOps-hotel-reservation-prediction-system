from typing import List

import pandas as pd


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get categorical columns from the dataframe
    """
    categorical_columns: List[str] = df.select_dtypes(include=["object"]).columns.tolist()
    return categorical_columns


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get numerical columns from the dataframe
    """
    numerical_columns: List[str] = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return numerical_columns
