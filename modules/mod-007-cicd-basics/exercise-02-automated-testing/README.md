# Exercise 02: Automated Testing for ML Code

## Overview

Learn comprehensive testing strategies specifically for machine learning systems. This exercise covers unit testing, integration testing, data validation, model testing, and continuous testing practices for ML applications.

## Learning Objectives

- Write unit tests for ML code
- Test ML models (training, inference, performance)
- Validate data quality and schemas
- Implement property-based testing
- Use fixtures and mocking effectively
- Measure and improve test coverage
- Set up continuous testing pipelines
- Test ML-specific concerns (reproducibility, fairness, drift)

## Prerequisites

- Python 3.11+ installed
- Understanding of pytest
- Basic ML knowledge
- Completed Exercise 01 (Git Workflows)

## Project Structure

```
exercise-02-automated-testing/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data transformation functions
│   ├── feature_engineering.py     # Feature creation
│   ├── model_training.py          # Training logic
│   ├── model_inference.py         # Prediction logic
│   ├── model_evaluation.py        # Metrics and evaluation
│   └── data_validation.py         # Data quality checks
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── unit/
│   │   ├── test_data_preprocessing.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_model_training.py
│   │   └── test_model_inference.py
│   ├── integration/
│   │   ├── test_training_pipeline.py
│   │   ├── test_inference_pipeline.py
│   │   └── test_end_to_end.py
│   ├── property/
│   │   ├── test_data_properties.py
│   │   └── test_model_properties.py
│   └── performance/
│       ├── test_training_performance.py
│       └── test_inference_performance.py
├── fixtures/
│   ├── sample_data.csv
│   ├── sample_model.pkl
│   └── test_config.yaml
├── data/
│   └── README.md
├── docs/
│   ├── TESTING_STRATEGY.md
│   ├── TEST_PATTERNS.md
│   └── COVERAGE_GUIDE.md
├── pytest.ini
├── requirements.txt
├── requirements-test.txt
└── README.md
```

## Types of ML Tests

### 1. Unit Tests
Test individual functions and components in isolation.

**Examples:**
- Data preprocessing functions
- Feature engineering logic
- Model helper functions
- Utility functions

### 2. Integration Tests
Test multiple components working together.

**Examples:**
- Complete training pipeline
- End-to-end prediction pipeline
- Data loading → preprocessing → inference

### 3. Data Tests
Validate data quality and schemas.

**Examples:**
- Schema validation
- Data type checks
- Range and distribution checks
- Missing value detection
- Outlier detection

### 4. Model Tests
Test ML model behavior and properties.

**Examples:**
- Training convergence
- Prediction correctness
- Model reproducibility
- Invariance tests
- Directional expectation tests

### 5. Performance Tests
Ensure code meets performance requirements.

**Examples:**
- Training time benchmarks
- Inference latency
- Memory usage
- Throughput tests

### 6. Property-Based Tests
Test invariants and properties using generated inputs.

**Examples:**
- Output shape invariance
- Data transformation reversibility
- Model prediction bounds

## Testing ML-Specific Concerns

### Reproducibility

Ensure models produce same results with same inputs:

```python
def test_model_reproducibility():
    """Test that model produces same results with same random seed."""
    X_train, y_train = load_data()

    # Train two models with same seed
    model1 = train_model(X_train, y_train, random_state=42)
    model2 = train_model(X_train, y_train, random_state=42)

    X_test = load_test_data()
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)

    assert np.array_equal(pred1, pred2)
```

### Model Invariance

Test that model respects expected invariances:

```python
def test_model_invariance_to_feature_order():
    """Test that shuffling feature order doesn't change predictions."""
    model = load_model()
    X = load_test_data()

    # Original predictions
    pred_original = model.predict(X)

    # Shuffle columns
    shuffled_columns = np.random.permutation(X.columns)
    X_shuffled = X[shuffled_columns]
    pred_shuffled = model.predict(X_shuffled[X.columns])

    np.testing.assert_array_almost_equal(pred_original, pred_shuffled)
```

### Directional Expectations

Test that model responds correctly to changes:

```python
def test_price_increases_with_size():
    """Test that predicted price increases when house size increases."""
    model = load_model()

    # Base example
    base_features = {"size": 1000, "bedrooms": 2, "location": "urban"}
    base_price = model.predict([base_features])[0]

    # Larger house
    large_features = {"size": 2000, "bedrooms": 2, "location": "urban"}
    large_price = model.predict([large_features])[0]

    assert large_price > base_price
```

### Data Drift Detection

Test for changes in data distribution:

```python
def test_no_data_drift():
    """Test that new data distribution matches training data."""
    train_stats = load_training_statistics()
    new_data = load_new_data()

    for column in new_data.columns:
        # Compare means
        mean_diff = abs(new_data[column].mean() - train_stats[column]['mean'])
        assert mean_diff < train_stats[column]['std'] * 3

        # Compare distributions (KS test)
        statistic, p_value = ks_2samp(
            new_data[column],
            train_stats[column]['distribution']
        )
        assert p_value > 0.05
```

## Pytest Best Practices

### Fixtures

```python
# conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    """Provide sample dataset for tests."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })

@pytest.fixture
def trained_model(sample_data):
    """Provide pre-trained model."""
    from src.model_training import train_model
    X = sample_data[['feature1', 'feature2']]
    y = sample_data['target']
    return train_model(X, y)

@pytest.fixture(scope="session")
def test_config():
    """Load test configuration (session-scoped for efficiency)."""
    import yaml
    with open('fixtures/test_config.yaml') as f:
        return yaml.safe_load(f)
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input_value,expected", [
    (0, 0),
    (1, 1),
    (2, 4),
    (3, 9),
    (-1, 1),
])
def test_square_function(input_value, expected):
    """Test square function with multiple inputs."""
    assert square(input_value) == expected
```

### Test Markers

```python
@pytest.mark.slow
def test_long_training():
    """Test that takes a long time."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test."""
    pass

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_gpu_training():
    """Test GPU training."""
    pass
```

Run specific tests:
```bash
pytest -m "not slow"           # Skip slow tests
pytest -m integration          # Run only integration tests
pytest -m "slow and gpu"       # Run slow GPU tests
```

### Mocking

```python
from unittest.mock import Mock, patch

def test_api_call_with_mock():
    """Test function that calls external API."""
    with patch('src.api.requests.get') as mock_get:
        # Configure mock
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'result': 'success'}

        # Test function
        result = fetch_data_from_api()

        # Assertions
        assert result == {'result': 'success'}
        mock_get.assert_called_once()
```

## Code Coverage

### Measure Coverage

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html

# Set coverage threshold
pytest --cov=src --cov-fail-under=80
```

### Coverage Configuration

```ini
# pytest.ini or setup.cfg
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-branch
    --cov-fail-under=80
```

## Continuous Testing

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Run tests
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## Test Organization

### Directory Structure

```
tests/
├── unit/              # Fast, isolated tests
├── integration/       # Component interaction tests
├── e2e/              # End-to-end system tests
├── property/         # Property-based tests
├── performance/      # Benchmark tests
└── conftest.py       # Shared fixtures
```

### Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Fixtures: Descriptive names without `test_` prefix

## Common Testing Patterns

### 1. Arrange-Act-Assert (AAA)

```python
def test_feature_scaling():
    # Arrange
    data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    scaler = StandardScaler()

    # Act
    scaled = scaler.fit_transform(data)

    # Assert
    assert scaled.mean() == pytest.approx(0, abs=1e-10)
    assert scaled.std() == pytest.approx(1, abs=1e-10)
```

### 2. Test Fixtures for Setup/Teardown

```python
@pytest.fixture
def temp_model_file(tmp_path):
    """Create temporary model file."""
    model_path = tmp_path / "model.pkl"
    model = train_simple_model()
    joblib.dump(model, model_path)

    yield model_path

    # Cleanup (happens automatically with tmp_path)
```

### 3. Snapshot Testing

```python
def test_model_predictions_snapshot(snapshot):
    """Test that model predictions match snapshot."""
    model = load_model()
    X_test = load_test_data()
    predictions = model.predict(X_test)

    snapshot.assert_match(predictions.tolist(), 'predictions')
```

## Testing Checklist

### Before Writing Tests

- [ ] Understand what you're testing
- [ ] Identify edge cases
- [ ] Consider what could go wrong
- [ ] Think about test data needs

### Writing Tests

- [ ] One test per behavior
- [ ] Clear test names
- [ ] Arrange-Act-Assert structure
- [ ] Test edge cases
- [ ] Use appropriate assertions
- [ ] Add comments for complex tests

### After Writing Tests

- [ ] All tests pass
- [ ] Tests are fast
- [ ] Tests are independent
- [ ] Coverage is adequate
- [ ] Tests are maintainable

## Example Test Suite Structure

```python
# tests/unit/test_data_preprocessing.py

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import clean_data, handle_missing_values

class TestDataCleaning:
    """Tests for data cleaning functions."""

    def test_remove_duplicates(self, sample_data_with_duplicates):
        """Test that duplicates are removed correctly."""
        cleaned = clean_data(sample_data_with_duplicates)
        assert cleaned.duplicated().sum() == 0

    def test_preserve_valid_rows(self, sample_data):
        """Test that valid rows are preserved."""
        cleaned = clean_data(sample_data)
        assert len(cleaned) == len(sample_data)

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
    def test_missing_value_strategies(self, data_with_missing, strategy):
        """Test different missing value imputation strategies."""
        filled = handle_missing_values(data_with_missing, strategy=strategy)
        assert filled.isnull().sum().sum() == 0

class TestDataValidation:
    """Tests for data validation."""

    def test_schema_validation(self, sample_data):
        """Test that data matches expected schema."""
        from src.data_validation import validate_schema
        is_valid, errors = validate_schema(sample_data)
        assert is_valid
        assert len(errors) == 0

    def test_value_ranges(self, sample_data):
        """Test that values are within expected ranges."""
        assert sample_data['age'].between(0, 120).all()
        assert sample_data['price'].ge(0).all()
```

## Advanced Testing Techniques

### Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_mean_is_between_min_and_max(values):
    """Property: mean should always be between min and max."""
    mean = np.mean(values)
    assert min(values) <= mean <= max(values)
```

### Testing with Different Backends

```python
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "pytorch"])
def test_model_inference_backends(backend, sample_input):
    """Test model inference with different backends."""
    model = load_model(backend=backend)
    output = model.predict(sample_input)
    assert output.shape == (1, 10)
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_data_preprocessing.py

# Run specific test
pytest tests/unit/test_data_preprocessing.py::test_remove_duplicates

# Run tests matching pattern
pytest -k "test_data"

# Run with markers
pytest -m "not slow"

# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Generate coverage report
pytest --cov=src --cov-report=html
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing ML Code](https://madewithml.com/courses/mlops/testing/)
- [Property-Based Testing](https://hypothesis.readthedocs.io/)
- [Test-Driven Development](https://www.obeythetestinggoat.com/)
- [ML Testing Best Practices](https://developers.google.com/machine-learning/testing-debugging)

## Next Steps

After completing this exercise:

1. ✅ Write unit tests for all functions
2. ✅ Add integration tests for pipelines
3. ✅ Implement data validation tests
4. ✅ Set up continuous testing
5. ✅ Achieve >80% code coverage

**Move on to**: Exercise 03 - Docker Image CI/CD

## Summary

Testing ML code requires special attention to:
- Data quality and validation
- Model reproducibility
- Performance characteristics
- Invariance properties
- Continuous monitoring

Comprehensive testing ensures your ML systems are reliable, maintainable, and production-ready.
