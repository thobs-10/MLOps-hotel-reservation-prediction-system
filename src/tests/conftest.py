from unittest.mock import patch

import pandas as pd
import pytest

# from _pytest.monkeypatch import MonkeyPatch
from pytest import MonkeyPatch
from zenml.steps import BaseStep


@pytest.fixture
def mock_raw_data_patch(monkeypatch: MonkeyPatch) -> None:
    """
    Mock the environment variable for data path.
    """
    monkeypatch.setenv("RAW_DATA_PATH", "data/raw/Hotel Reservations.csv")


@pytest.fixture
def sample_test_data() -> pd.DataFrame:
    """
    Sample test data for testing.
    """
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, None],
            "feature2": ["A", "B", "C", "D"],
        }
    )


@pytest.fixture(autouse=True)
def mock_zenml_runtime():  # type: ignore[no-untyped-def]
    """Completely disable ZenML step execution during tests."""

    def mock_step_call(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        return self.entrypoint(*args, **kwargs)

    with (
        patch.object(BaseStep, "__call__", new=mock_step_call),
        patch("zenml.steps.step_decorator.step", lambda x: x),
    ):  # Bypass @step decorator
        yield
