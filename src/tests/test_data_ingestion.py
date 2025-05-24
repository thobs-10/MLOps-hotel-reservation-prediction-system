import pandas as pd
import pytest
from pytest import MonkeyPatch

from src.components.data_ingestion import (
    handling_null_values,
    load_raw_data,
    remove_duplicates,
    remove_irrelevant_columns,
    handle_data_types,
)


def test_load_data_success(
    mock_raw_data_patch: MonkeyPatch, sample_test_data: pd.DataFrame, mocker
):
    """Test load_data function success"""
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=sample_test_data)
    data = load_raw_data()
    mock_read_csv.assert_called_once_with("data/raw/Hotel Reservations.csv")
    assert not data.empty
    assert data.shape == (5, 2)
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


def test_remove_duplicates_success(sample_test_data: pd.DataFrame):
    """Test remove_duplicates function success"""
    data = sample_test_data.copy()
    assert data.shape == (5, 2)
    cleaned_data = remove_duplicates(data)
    assert cleaned_data.shape == (4, 2)


def test_remove_duplicates_no_duplicates(sample_test_data: pd.DataFrame):
    """Test remove_duplicates function with no duplicates"""
    data = sample_test_data.copy()
    data = data.drop_duplicates()
    assert data.shape == (4, 2)
    cleaned_data = remove_duplicates(data)
    assert cleaned_data.shape == (4, 2)
    assert cleaned_data.equals(data)


def test_remove_ducplicates_all_duplicates(sample_test_data: pd.DataFrame):
    """Test remove_duplicates function with all duplicates"""
    data = sample_test_data.copy()
    data = pd.concat([data, data])  # Duplicate all rows
    assert data.shape == (10, 2)
    cleaned_data = remove_duplicates(data)
    assert cleaned_data.shape == (4, 2)
    assert cleaned_data.equals(data.drop_duplicates())


def test_handling_null_values_success(sample_test_data: pd.DataFrame):
    """Test handling_null_values function success"""
    data = sample_test_data.copy()
    assert data["feature1"].isnull().sum() == 1
    cleaned_data = handling_null_values(data)
    assert cleaned_data["feature1"].isnull().sum() == 0


def test_handling_null_values_no_nulls(sample_test_data: pd.DataFrame):
    """Test handling_null_values function with no nulls"""
    data = sample_test_data.copy()
    data["feature1"] = data["feature1"].fillna(0)  # Fill nulls
    assert data["feature1"].isnull().sum() == 0
    cleaned_data = handling_null_values(data)
    assert cleaned_data.equals(data)  # No changes expected


def test_handling_null_values_all_nulls():
    """Test handling_null_values function with all nulls"""
    data = pd.DataFrame(
        {
            "feature1": [None, None, None, None, None],
            "feature2": [None, None, None, None, None],
        }
    )
    data["feature2"] = pd.Series([None, None, None, None, None])
    assert data["feature2"].isnull().sum() == 5
    with pytest.raises(ValueError):
        handling_null_values(data)  # Expecting an exception for all nulls


def test_remove_irrelevant_columns_success(sample_test_data: pd.DataFrame):
    """Test remove_irrelevant_columns function success"""
    data = sample_test_data.copy()
    data["Booking_ID"] = [1, 2, 3, 4, 5]
    assert "Booking_ID" in data.columns
    cleaned_data = remove_irrelevant_columns(data)
    assert "Booking_ID" not in cleaned_data.columns
    assert cleaned_data.shape == (5, 2)  # Only original columns should remain


def test_remove_irrelevant_columns_no_irrelevant_columns(
    sample_test_data: pd.DataFrame,
):
    """Test remove_irrelevant_columns function with no irrelevant columns"""
    data = sample_test_data.copy()
    assert "Booking_ID" not in data.columns
    with pytest.raises(ValueError):
        remove_irrelevant_columns(data)


def test_handle_data_types_success(sample_test_data: pd.DataFrame):
    """Test handle_data_types function success"""
    data = sample_test_data.copy()
    data["feature1"] = data["feature1"].astype("float64")  # Change type to float
    assert data["feature1"].dtype == "float64"
    data["feature2"] = data["feature1"].astype("object")  # Change type to object
    assert data["feature2"].dtype == "object"
    cleaned_data = handle_data_types(data)
    assert cleaned_data["feature1"].dtype == "float64"
    assert cleaned_data["feature2"].dtype == "category"


def test_handle_data_types_invalid_type(sample_test_data: pd.DataFrame):
    """Test handle_data_types function with type not in numerical or categorical"""
    data = sample_test_data.copy()
    data["feature2"] = data["feature2"].astype("bool")  # Change type to str
    assert data["feature2"].dtype == "bool"
    with pytest.raises(ValueError):
        handle_data_types(data)  # Expecting an exception for invalid type
