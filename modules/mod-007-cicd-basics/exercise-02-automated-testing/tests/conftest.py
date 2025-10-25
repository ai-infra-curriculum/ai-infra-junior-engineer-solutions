"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_data():
    """Sample dataset for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': [100, 200, 300, 400, 500],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_data_with_missing():
    """Sample dataset with missing values."""
    return pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, np.nan, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_data_with_duplicates():
    """Sample dataset with duplicate rows."""
    return pd.DataFrame({
        'feature1': [1, 2, 1, 4, 5],
        'feature2': [10, 20, 10, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_data_with_outliers():
    """Sample dataset with outliers."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 1000],  # 1000 is outlier
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def large_sample_data():
    """Large sample dataset for performance tests."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y

    return df


@pytest.fixture
def train_test_data(large_sample_data):
    """Train/test split for model training tests."""
    X = large_sample_data.drop('target', axis=1)
    y = large_sample_data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_model(train_test_data):
    """Pre-trained model for testing."""
    from src.model_training import train_model

    X_train, _, y_train, _ = train_test_data

    model = train_model(
        X_train, y_train,
        model_type="random_forest",
        hyperparameters={"n_estimators": 10, "max_depth": 3},
        random_state=42
    )

    return model


@pytest.fixture
def sample_text_data():
    """Sample text data for text feature tests."""
    return pd.DataFrame({
        'text': [
            'This is a test',
            'Another example text',
            'Short',
            'A much longer text with many more words',
        ]
    })


@pytest.fixture
def sample_date_data():
    """Sample data with datetime column."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10, freq='D'),
        'value': range(10)
    })


@pytest.fixture
def expected_schema():
    """Expected schema for validation tests."""
    return {
        'feature1': 'int',
        'feature2': 'int',
        'feature3': 'int',
        'target': 'int'
    }


@pytest.fixture
def temp_model_file(tmp_path, trained_model):
    """Temporary file for model saving/loading tests."""
    import joblib

    model_path = tmp_path / "test_model.pkl"
    joblib.dump(trained_model, model_path)

    return model_path


@pytest.fixture(scope="session")
def test_config():
    """Test configuration (session-scoped for efficiency)."""
    return {
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 5,
        "batch_size": 100,
    }


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "property: marks tests as property-based tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
