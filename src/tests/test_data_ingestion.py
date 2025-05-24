import pandas as pd
import pytest
from pytest import MonkeyPatch

from src.components.data_ingestion import (
    load_raw_data,
)


def test_load_data_success(
    mock_raw_data_patch: MonkeyPatch, sample_test_data: pd.DataFrame, mocker
):
    """Test load_data function success"""
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=sample_test_data)
    data = load_raw_data()
    mock_read_csv.assert_called_once_with("data/raw/Hotel Reservations.csv")
    assert not data.empty
    assert data.shape == (4, 2)
    assert list(data.columns) == ["feature1", "feature2"]


def test_load_data_not_found(mock_raw_data_patch: MonkeyPatch, mocker):
    """Test load_data function file not found"""
    mock_read_csv = mocker.patch(
        "pandas.read_csv", side_effect=FileNotFoundError("File not found")
    )
    with pytest.raises(FileNotFoundError):
        load_raw_data()
    mock_read_csv.assert_called_once_with("data/raw/Hotel Reservations.csv")


def test_load_data_empty_file(mock_raw_data_patch: MonkeyPatch, mocker):
    """Test load_data function empty file"""
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=pd.DataFrame())
    data = load_raw_data()
    mock_read_csv.assert_called_once_with("data/raw/Hotel Reservations.csv")
    assert data.empty


def test_load_data_invalid_format(mock_raw_data_patch: MonkeyPatch, mocker):
    """Test load_data function with invalid format"""
    mock_read_csv = mocker.patch(
        "pandas.read_csv", side_effect=pd.errors.ParserError("Invalid format")
    )
    with pytest.raises(pd.errors.ParserError):
        load_raw_data()
    mock_read_csv.assert_called_once_with("data/raw/Hotel Reservations.csv")
