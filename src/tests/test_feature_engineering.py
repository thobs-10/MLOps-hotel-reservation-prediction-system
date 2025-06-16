import pandas as pd
import pytest
from pytest import MonkeyPatch

from src.components.feature_engineering import (
    load_processed_data,
    generate_new_features,
    encode_categorical_columns,
    separate_data,
    get_important_features,
    get_pca_feature_importance,
    select_pca_features,
)


def test_load_processed_data_success(
    mock_processed_data_patch: MonkeyPatch,
    sample_processed_data: pd.DataFrame,
    mocker,
):
    """Test load_processed_data function success"""
    mocker.patch("os.path.exists", return_value=True)
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=sample_processed_data)
    data = load_processed_data()
    mock_read_csv.assert_called_once_with("data/processed_data/")
    assert not data.empty
    assert data.shape == (5, 4)
    assert list(data.columns) == ["feature1", "feature2", "feature3", "feature4"]


def test_load_processed_data_file_not_found(
    mock_processed_data_patch: MonkeyPatch, mocker
):
    """Test load_processed_data function file not found"""
    mocker.patch("os.path.exists", return_value=False)
    with pytest.raises(FileNotFoundError):
        load_processed_data()
    mocker.patch("pandas.read_csv", side_effect=FileNotFoundError("File not found"))
    mock_read_csv = mocker.patch("pandas.read_csv")


def test_generate_new_features_success(sample_processed_data: pd.DataFrame):
    """Test generate_new_features function success"""
    data = sample_processed_data.copy()
    data["no_of_weekend_nights"] = [1, 2, 1, 0, 3]
    data["no_of_week_nights"] = [2, 3, 1, 4, 2]
    data["repeated_guest"] = ["Yes", "No", "Yes", "No", "Yes"]

    new_data = generate_new_features(data)

    assert "total_nights_stayed" in new_data.columns
    assert "is_repeat_guest" in new_data.columns
    assert new_data["total_nights_stayed"].equals(pd.Series([3, 5, 2, 4, 5]))
    assert new_data["is_repeat_guest"].equals(pd.Series([1, 0, 1, 0, 1]))


def test_generate_new_features_missing_columns(sample_processed_data: pd.DataFrame):
    """Test generate_new_features function with missing columns"""
    data = sample_processed_data.copy()

    with pytest.raises(KeyError):
        generate_new_features(data)


def test_encode_categorical_columns_success(sample_processed_data: pd.DataFrame):
    """Test encode_categorical_columns function success"""
    data = sample_processed_data.copy()
    data["type_of_meal_plan"] = ["A", "B", "A", "C", "B"]
    data["room_type_reserved"] = ["cat1", "cat2", "cat1", "cat3", "cat2"]
    data["market_segment_type"] = ["low", "medium", "high", "medium", "low"]

    encoded_data = encode_categorical_columns(data)

    assert "type_of_meal_plan" in encoded_data.columns
    assert encoded_data["type_of_meal_plan"].dtype == "int64"
    assert set(encoded_data["type_of_meal_plan"].unique()) == {0, 1, 2}

    assert "room_type_reserved" in encoded_data.columns
    assert encoded_data["room_type_reserved"].dtype == "int64"
    assert set(encoded_data["room_type_reserved"].unique()) == {0, 1, 2}

    assert "market_segment_type" in encoded_data.columns
    assert encoded_data["market_segment_type"].dtype == "int64"
    assert set(encoded_data["market_segment_type"].unique()) == {0, 1, 2}


def test_encode_categorical_columns_missing_columns(
    sample_processed_data: pd.DataFrame,
):
    """Test encode_categorical_columns function with missing columns"""
    data = sample_processed_data.copy()
    data["type_of_meal_plan"] = ["A", "B", "A", "C", "B"]

    with pytest.raises(ValueError):
        encode_categorical_columns(data)


def test_encode_categorical_columns_empty_dataframe(
    sample_processed_data: pd.DataFrame,
):
    """Test encode_categorical_columns function with empty DataFrame"""
    data = pd.DataFrame()

    with pytest.raises(ValueError):
        encode_categorical_columns(data)
    assert sample_processed_data.equals(sample_processed_data)


def test_encode_categorical_columns_no_categorical_columns(
    sample_processed_data: pd.DataFrame,
):
    """Test encode_categorical_columns function with no categorical columns"""
    data = sample_processed_data.copy()
    # data.drop(columns=["feature1", "feature2", "feature3", "feature4"], inplace=True)

    with pytest.raises(ValueError):
        encode_categorical_columns(data)
    assert sample_processed_data.equals(sample_processed_data)


def test_encode_categorical_columns_label_encoder_not_provided(
    sample_processed_data: pd.DataFrame,
):
    """Test encode_categorical_columns function with LabelEncoder not provided"""
    data = sample_processed_data.copy()
    data["type_of_meal_plan"] = ["A", "B", "A", "C", "B"]

    with pytest.raises(ValueError):
        encode_categorical_columns(data)


def test_separate_data_success(sample_processed_data: pd.DataFrame):
    """Test separate_data function success"""
    data = sample_processed_data.copy()
    data["booking_status"] = [1, 0, 1, 0, 1]

    X, y = separate_data(data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 4
    assert y.shape[0] == data.shape[0]
    assert "booking_status" not in X.columns


def test_separate_data_missing_target_column(sample_processed_data: pd.DataFrame):
    """Test separate_data function with missing target column"""
    data = sample_processed_data.copy()

    with pytest.raises(ValueError):
        separate_data(data)
    assert sample_processed_data.equals(sample_processed_data)


def test_get_important_features_success(sample_processed_data: pd.DataFrame):
    """Test get_important_features function success"""
    data = sample_processed_data.copy()
    data["booking_status"] = [1, 0, 1, 0, 1]

    X, y = separate_data(data)
    important_features, feature_names = get_important_features(X, y)

    assert isinstance(important_features, pd.DataFrame)
    assert isinstance(feature_names, list)
    assert not important_features.empty
    assert len(feature_names) > 0
    assert all(col in X.columns for col in feature_names)


def test_get_important_features_incorrect_threshold(
    sample_processed_data: pd.DataFrame,
):
    """Test get_important_features function with incorrect threshold"""
    data = sample_processed_data.copy()
    data["booking_status"] = [1, 0, 1, 0, 1]

    X, y = separate_data(data)

    with pytest.raises(ValueError):
        get_important_features(X, y, threshold=-0.05)
    assert sample_processed_data.equals(sample_processed_data)


def test_get_important_features_empty_dataframe(
    sample_processed_data: pd.DataFrame,
):
    """Test get_important_features function with empty DataFrame"""
    data = pd.DataFrame()

    with pytest.raises(ValueError):
        get_important_features(data, pd.Series())
    assert sample_processed_data.equals(sample_processed_data)


def test_get_pca_feature_importance_success(sample_processed_data: pd.DataFrame):
    """Test get_pca_feature_importance function success"""
    data = sample_processed_data.copy()
    data["booking_status"] = [1, 0, 1, 0, 1]
    X, _ = separate_data(data)
    columns = ["feature1", "feature2", "feature3", "feature4"]
    pca_df, important_features = get_pca_feature_importance(X, columns)
    assert isinstance(pca_df, pd.DataFrame)
    assert isinstance(important_features, list)


def test_select_pca_features_success(sample_processed_data: pd.DataFrame):
    """Test select_pca_features function success"""
    data = sample_processed_data.copy()
    data["booking_status"] = [1, 0, 1, 0, 1]
    X, _ = separate_data(data)
    # columns = ["feature1", "feature2", "feature3", "feature4"]
    X, feature_list = get_important_features(X, data["booking_status"], threshold=0.05)
    pca_df, most_important_features = get_pca_feature_importance(X, feature_list)
    selected_features = select_pca_features(pca_df, most_important_features)
    assert isinstance(selected_features, pd.DataFrame)
    assert selected_features.shape[1] == 2
