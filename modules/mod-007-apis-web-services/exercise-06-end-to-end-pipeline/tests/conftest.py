"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    return df


@pytest.fixture
def sample_features(sample_data):
    """Get feature DataFrame."""
    return sample_data.drop("target", axis=1)


@pytest.fixture
def sample_target(sample_data):
    """Get target Series."""
    return sample_data["target"]


@pytest.fixture
def sample_csv_file(tmp_path, sample_data):
    """Create temporary CSV file."""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return str(csv_path)
