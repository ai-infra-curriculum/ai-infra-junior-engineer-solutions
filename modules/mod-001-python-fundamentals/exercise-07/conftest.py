"""Pytest configuration and shared fixtures."""

import pytest


# ============================================================================
# Session-Scoped Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary test data directory."""
    data_dir = tmp_path_factory.mktemp("test_data")
    return data_dir


@pytest.fixture(scope="session")
def sample_config():
    """Provide test configuration."""
    return {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "model_type": "resnet50",
    }


# ============================================================================
# Module-Scoped Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def large_dataset():
    """Generate large dataset (expensive operation)."""
    return [{"id": i, "value": i * 2, "label": i % 2} for i in range(10000)]


# ============================================================================
# Function-Scoped Fixtures
# ============================================================================

@pytest.fixture
def sample_predictions():
    """Fixture providing sample predictions."""
    return [1, 0, 1, 1, 0, 1, 0, 0]


@pytest.fixture
def sample_labels():
    """Fixture providing sample labels."""
    return [1, 0, 1, 0, 0, 1, 0, 1]


@pytest.fixture
def sample_dataset():
    """Fixture providing small sample dataset."""
    return [
        {"id": 1, "features": [1.0, 2.0], "label": 0},
        {"id": 2, "features": [2.0, 3.0], "label": 1},
        {"id": 3, "features": [3.0, 4.0], "label": 0},
    ]


@pytest.fixture
def temp_model_file(tmp_path):
    """Fixture providing temporary model file."""
    model_file = tmp_path / "model.txt"
    model_file.write_text("model_weights_v1.0")
    return model_file


# ============================================================================
# Parametrized Fixtures
# ============================================================================

@pytest.fixture(params=[16, 32, 64])
def batch_sizes(request):
    """Fixture providing different batch sizes."""
    return request.param


@pytest.fixture(params=["mean", "median"])
def fill_strategies(request):
    """Fixture providing different fill strategies."""
    return request.param


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )
