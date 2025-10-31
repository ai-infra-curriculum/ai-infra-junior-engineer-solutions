# Exercise 04: Testing with pytest

## Overview

Master testing fundamentals with pytest to ensure code quality and reliability. Learn to write comprehensive test suites, use fixtures, mock external dependencies, and practice Test-Driven Development (TDD).

## Learning Objectives

- ✅ Write unit tests with pytest
- ✅ Use fixtures for test setup
- ✅ Mock external dependencies
- ✅ Write parameterized tests
- ✅ Measure test coverage
- ✅ Practice Test-Driven Development (TDD)
- ✅ Test ML pipeline components

## Topics Covered

### 1. pytest Fundamentals

#### Basic Test Structure

```python
# test_basic.py
def test_addition():
    """Test basic addition."""
    assert 2 + 2 == 4

def test_string_operations():
    """Test string operations."""
    text = "hello"
    assert text.upper() == "HELLO"
    assert len(text) == 5
    assert "ell" in text

def test_list_operations():
    """Test list operations."""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert max(numbers) == 5
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest test_basic.py

# Run specific test
pytest test_basic.py::test_addition

# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run tests in parallel
pytest -n 4
```

#### Test Discovery

pytest automatically discovers tests:
- Files: `test_*.py` or `*_test.py`
- Functions: `test_*`
- Classes: `Test*`
- Methods: `test_*`

### 2. Assertions

#### Basic Assertions

```python
# Equality
assert value == expected

# Inequality
assert value != other

# Boolean
assert condition is True
assert condition is False

# Membership
assert item in collection
assert item not in collection

# Type checking
assert isinstance(obj, MyClass)

# None checking
assert value is None
assert value is not None
```

#### pytest Assertion Introspection

```python
def test_detailed_assertion():
    """pytest provides detailed assertion information."""
    expected = {"name": "Model", "version": "1.0.0"}
    actual = {"name": "Model", "version": "2.0.0"}

    # pytest will show exactly what differs
    assert expected == actual
```

#### Approximate Comparisons

```python
from pytest import approx

def test_float_comparison():
    """Test floating point numbers."""
    result = 0.1 + 0.2
    assert result == approx(0.3)

    # With tolerance
    assert 10.5 == approx(10, rel=0.1)  # 10% tolerance
    assert 10.05 == approx(10, abs=0.1)  # Absolute tolerance
```

#### Exception Testing

```python
import pytest

def test_exception():
    """Test that exception is raised."""
    with pytest.raises(ValueError):
        int("not a number")

    # Check exception message
    with pytest.raises(ValueError, match="invalid literal"):
        int("not a number")

    # Get exception info
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("Custom message")

    assert "Custom" in str(exc_info.value)
```

### 3. Fixtures

#### Basic Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    """Test using fixture."""
    assert sum(sample_data) == 15

def test_length(sample_data):
    """Another test using same fixture."""
    assert len(sample_data) == 5
```

#### Fixture Scope

```python
# Function scope (default) - run for each test
@pytest.fixture(scope="function")
def function_fixture():
    return "function"

# Class scope - run once per test class
@pytest.fixture(scope="class")
def class_fixture():
    return "class"

# Module scope - run once per module
@pytest.fixture(scope="module")
def module_fixture():
    return "module"

# Session scope - run once per session
@pytest.fixture(scope="session")
def session_fixture():
    return "session"
```

#### Fixture Setup and Teardown

```python
@pytest.fixture
def database_connection():
    """Fixture with setup and teardown."""
    # Setup
    conn = create_connection()
    print("\nDatabase connected")

    yield conn  # Provide fixture value

    # Teardown
    conn.close()
    print("\nDatabase closed")

def test_query(database_connection):
    """Test using connection fixture."""
    result = database_connection.execute("SELECT 1")
    assert result is not None
```

#### Fixture Dependencies

```python
@pytest.fixture
def user():
    """Create test user."""
    return {"id": 1, "name": "Alice"}

@pytest.fixture
def user_with_posts(user):
    """Create user with posts (depends on user fixture)."""
    user["posts"] = [
        {"id": 1, "title": "Post 1"},
        {"id": 2, "title": "Post 2"}
    ]
    return user

def test_user_posts(user_with_posts):
    """Test user with posts."""
    assert len(user_with_posts["posts"]) == 2
```

#### Parametrized Fixtures

```python
@pytest.fixture(params=["cuda", "cpu", "mps"])
def device(request):
    """Fixture that runs tests on different devices."""
    return request.param

def test_model_on_device(device):
    """Test runs 3 times, once for each device."""
    model = create_model(device=device)
    assert model.device == device
```

#### conftest.py

```python
# conftest.py - shared fixtures across multiple test files
import pytest

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def ml_model():
    """Provide ML model for testing."""
    model = load_model("test_model.pkl")
    return model
```

### 4. Mocking

#### Basic Mocking with unittest.mock

```python
from unittest.mock import Mock, MagicMock, patch

def test_mock_function():
    """Test with mock function."""
    mock_func = Mock(return_value=42)

    result = mock_func()

    assert result == 42
    mock_func.assert_called_once()

def test_mock_with_side_effect():
    """Test mock with side effect."""
    mock_func = Mock(side_effect=[1, 2, 3])

    assert mock_func() == 1
    assert mock_func() == 2
    assert mock_func() == 3
```

#### Patching

```python
# Code to test (in mymodule.py)
import requests

def get_user(user_id):
    """Fetch user from API."""
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()

# Test with patch
from unittest.mock import patch

@patch('mymodule.requests.get')
def test_get_user(mock_get):
    """Test get_user with mocked requests."""
    # Configure mock
    mock_get.return_value.json.return_value = {"id": 1, "name": "Alice"}

    # Call function
    user = get_user(1)

    # Assertions
    assert user["name"] == "Alice"
    mock_get.assert_called_once_with("https://api.example.com/users/1")
```

#### Context Manager Patching

```python
def test_with_context_manager():
    """Test using patch as context manager."""
    with patch('mymodule.requests.get') as mock_get:
        mock_get.return_value.json.return_value = {"id": 1}

        user = get_user(1)
        assert user["id"] == 1
```

#### Patch Object

```python
class ModelService:
    def predict(self, data):
        # Expensive operation
        return expensive_prediction(data)

def test_model_service():
    """Test with patched method."""
    service = ModelService()

    with patch.object(service, 'predict', return_value=0.95):
        result = service.predict([1, 2, 3])
        assert result == 0.95
```

#### pytest-mock

```python
# Install: pip install pytest-mock

def test_with_mocker(mocker):
    """Test using pytest-mock."""
    mock_get = mocker.patch('mymodule.requests.get')
    mock_get.return_value.json.return_value = {"id": 1}

    user = get_user(1)
    assert user["id"] == 1
```

### 5. Parameterized Tests

#### Basic Parametrization

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
    (5, 25),
])
def test_square(input, expected):
    """Test square function with multiple inputs."""
    assert input ** 2 == expected
```

#### Multiple Parameters

```python
@pytest.mark.parametrize("x,y,expected", [
    (1, 2, 3),
    (10, 20, 30),
    (-1, 1, 0),
    (0, 0, 0),
])
def test_addition(x, y, expected):
    """Test addition with multiple cases."""
    assert x + y == expected
```

#### Parametrize with IDs

```python
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
], ids=["two", "three", "four"])
def test_with_ids(input, expected):
    """Tests will show with custom IDs."""
    assert input ** 2 == expected
```

#### Nested Parametrization

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_model_training(device, batch_size):
    """Test runs 6 times (2 devices × 3 batch sizes)."""
    model = train_model(device=device, batch_size=batch_size)
    assert model is not None
```

### 6. Test Organization

#### Test Classes

```python
class TestModelOperations:
    """Group related tests in a class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for all tests in class."""
        self.model = create_model()

    def test_prediction(self):
        """Test model prediction."""
        result = self.model.predict([1, 2, 3])
        assert result is not None

    def test_training(self):
        """Test model training."""
        self.model.train(data)
        assert self.model.is_trained
```

#### Test Markers

```python
import pytest

# Mark slow tests
@pytest.mark.slow
def test_slow_operation():
    """This test takes a long time."""
    time.sleep(5)

# Mark integration tests
@pytest.mark.integration
def test_api_integration():
    """Integration test."""
    response = requests.get(API_URL)
    assert response.status_code == 200

# Skip tests
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    """This will be skipped."""
    pass

# Conditional skip
@pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
def test_unix_feature():
    """Skipped on Windows."""
    pass

# Expected failure
@pytest.mark.xfail
def test_known_bug():
    """This test is expected to fail."""
    assert False
```

#### Running Marked Tests

```bash
# Run only slow tests
pytest -m slow

# Run everything except slow tests
pytest -m "not slow"

# Run integration tests
pytest -m integration

# Combine markers
pytest -m "slow and integration"
```

### 7. Test Coverage

#### Measuring Coverage

```bash
# Install coverage
pip install pytest-cov

# Run tests with coverage
pytest --cov=mymodule

# Generate HTML report
pytest --cov=mymodule --cov-report=html

# Show missing lines
pytest --cov=mymodule --cov-report=term-missing

# Set minimum coverage threshold
pytest --cov=mymodule --cov-fail-under=80
```

#### Coverage Configuration

```ini
# pyproject.toml or setup.cfg
[tool:pytest]
addopts = --cov=mymodule --cov-report=html --cov-report=term

[coverage:run]
source = mymodule
omit =
    */tests/*
    */venv/*
    */__init__.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

### 8. Test-Driven Development (TDD)

#### TDD Workflow

1. **Red**: Write a failing test
2. **Green**: Write minimal code to pass
3. **Refactor**: Improve code while keeping tests green

#### Example TDD Cycle

```python
# Step 1: RED - Write failing test
def test_calculate_accuracy():
    """Test accuracy calculation."""
    predictions = [1, 0, 1, 1, 0]
    labels = [1, 0, 1, 0, 0]

    accuracy = calculate_accuracy(predictions, labels)

    assert accuracy == 0.8  # 4 out of 5 correct

# Step 2: GREEN - Minimal implementation
def calculate_accuracy(predictions, labels):
    """Calculate prediction accuracy."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)

# Step 3: REFACTOR - Add validation
def calculate_accuracy(predictions, labels):
    """
    Calculate prediction accuracy.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Accuracy score between 0 and 1

    Raises:
        ValueError: If lengths don't match
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return 0.0

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)

# Add more tests
def test_calculate_accuracy_empty():
    """Test with empty inputs."""
    assert calculate_accuracy([], []) == 0.0

def test_calculate_accuracy_mismatched_lengths():
    """Test mismatched lengths."""
    with pytest.raises(ValueError, match="same length"):
        calculate_accuracy([1, 2], [1])
```

---

## Project: Tested ML Pipeline Components

Build a complete ML pipeline with comprehensive test coverage.

### Requirements

**Components to Test:**
1. Data loader and preprocessor
2. Feature engineering
3. Model trainer
4. Model evaluator
5. Prediction service

**Testing Requirements:**
- Unit tests for each component
- Integration tests for pipeline
- Mocked external dependencies
- Parameterized tests for multiple scenarios
- Minimum 90% code coverage
- TDD approach for new features

### Implementation

See `solutions/ml_pipeline.py` and `tests/test_ml_pipeline.py` for complete implementation.

### Example Usage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_pipeline --cov-report=html

# Run specific test category
pytest -m unit
pytest -m integration

# Run tests in parallel
pytest -n 4
```

---

## Practice Problems

### Problem 1: Test String Utilities

```python
def test_reverse_string():
    """Test string reversal."""
    # Your implementation here
    pass

def test_is_palindrome():
    """Test palindrome checker."""
    # Your implementation here
    pass

@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("WORLD", "WORLD"),
    ("MiXeD", "MIXED"),
])
def test_uppercase(input, expected):
    """Test uppercase conversion."""
    # Your implementation here
    pass
```

### Problem 2: Test Math Operations

```python
@pytest.mark.parametrize("a,b,expected", [
    (10, 5, 15),
    (-1, 1, 0),
    (0, 0, 0),
])
def test_add(a, b, expected):
    """Test addition."""
    pass

def test_divide_by_zero():
    """Test division by zero raises error."""
    pass
```

### Problem 3: Test with Fixtures

```python
@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame."""
    # Your implementation here
    pass

def test_dataframe_shape(sample_dataframe):
    """Test DataFrame shape."""
    pass

def test_dataframe_columns(sample_dataframe):
    """Test DataFrame columns."""
    pass
```

### Problem 4: Test with Mocking

```python
@patch('requests.get')
def test_fetch_data(mock_get):
    """Test data fetching with mocked requests."""
    # Your implementation here
    pass
```

### Problem 5: TDD Exercise

Write tests first, then implement:

```python
# 1. Write test
def test_calculate_f1_score():
    """Test F1 score calculation."""
    precision = 0.8
    recall = 0.9

    f1 = calculate_f1_score(precision, recall)

    assert f1 == approx(0.847, rel=0.01)

# 2. Implement function
def calculate_f1_score(precision, recall):
    # Your implementation here
    pass
```

---

## Best Practices

### 1. Test Organization

```
tests/
├── unit/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_model.py
├── integration/
│   ├── test_pipeline.py
│   └── test_api.py
├── conftest.py
└── __init__.py
```

### 2. Test Naming

```python
# Good - descriptive names
def test_model_raises_error_on_invalid_input():
    pass

def test_preprocessor_scales_features_to_zero_one_range():
    pass

# Bad - vague names
def test_model():
    pass

def test_1():
    pass
```

### 3. Arrange-Act-Assert Pattern

```python
def test_model_prediction():
    # Arrange - set up test data
    model = MLModel()
    input_data = [1, 2, 3, 4, 5]

    # Act - perform operation
    prediction = model.predict(input_data)

    # Assert - check result
    assert prediction is not None
    assert 0 <= prediction <= 1
```

### 4. One Assertion Per Test (Generally)

```python
# Good - focused test
def test_model_returns_probability():
    """Test model returns probability between 0 and 1."""
    prediction = model.predict(data)
    assert 0 <= prediction <= 1

def test_model_returns_float():
    """Test model returns float type."""
    prediction = model.predict(data)
    assert isinstance(prediction, float)

# Acceptable - related assertions
def test_dataframe_structure():
    """Test DataFrame has correct structure."""
    df = create_dataframe()
    assert len(df) == 100
    assert list(df.columns) == ["feature1", "feature2", "label"]
```

### 5. Use Fixtures for Setup

```python
# Good - reusable fixture
@pytest.fixture
def trained_model():
    model = MLModel()
    model.train(training_data)
    return model

def test_prediction(trained_model):
    result = trained_model.predict(test_data)
    assert result is not None

# Bad - repeated setup
def test_prediction():
    model = MLModel()
    model.train(training_data)
    result = model.predict(test_data)
    assert result is not None
```

---

## Common Pitfalls

### 1. Testing Implementation Details

```python
# Bad - tests implementation
def test_model_uses_specific_algorithm():
    model = MLModel()
    assert model.algorithm == "random_forest"

# Good - tests behavior
def test_model_predicts_correctly():
    model = MLModel()
    prediction = model.predict(test_data)
    assert accuracy(prediction, labels) > 0.9
```

### 2. Flaky Tests

```python
# Bad - depends on timing
def test_cache_expires():
    cache.set("key", "value")
    time.sleep(1)  # Flaky - timing dependent
    assert cache.get("key") is None

# Good - control time explicitly
def test_cache_expires(mocker):
    mock_time = mocker.patch('time.time')
    mock_time.return_value = 100

    cache.set("key", "value", ttl=10)

    mock_time.return_value = 111
    assert cache.get("key") is None
```

### 3. Test Interdependence

```python
# Bad - tests depend on each other
def test_create_user():
    global user
    user = create_user("Alice")

def test_update_user():
    user.update(name="Bob")  # Depends on previous test

# Good - independent tests
@pytest.fixture
def user():
    return create_user("Alice")

def test_update_user(user):
    user.update(name="Bob")
    assert user.name == "Bob"
```

---

## Validation

Run the test suite:

```bash
pytest tests/ -v --cov=ml_pipeline --cov-report=term-missing
```

Expected output:
```
==================== test session starts ====================
collected 45 items

tests/test_ml_pipeline.py::test_data_loader ✓
tests/test_ml_pipeline.py::test_preprocessor ✓
tests/test_ml_pipeline.py::test_feature_engineering ✓
...

---------- coverage: platform linux, python 3.11 -----------
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
ml_pipeline.py           150      5    96%   23, 45-48
-----------------------------------------------------
TOTAL                    150      5    96%

==================== 45 passed in 2.34s =====================

🎉 Exercise 04 Complete!
```

---

## Resources

### Documentation
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

### Books
- "Python Testing with pytest" by Brian Okken
- "Test-Driven Development with Python" by Harry Percival

### Articles
- [Effective Python Testing With Pytest](https://realpython.com/pytest-python-testing/)
- [Getting Started With Testing in Python](https://realpython.com/python-testing/)

---

## Next Steps

After completing this exercise:

1. **Exercise 05: Data Processing** - Process data with pandas and numpy
2. Apply TDD to all new features
3. Maintain high test coverage (>80%)
4. Write tests for edge cases
5. Use continuous integration (CI) for automated testing

---

**Write tests, write better code! 🧪**
