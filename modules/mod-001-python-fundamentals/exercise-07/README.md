# Exercise 07: Unit Testing ML Utility Functions with Pytest - Complete Solution

## Overview

This solution demonstrates comprehensive unit testing practices for ML utility functions using pytest. It covers all essential testing patterns: fixtures, parametrization, mocking, async testing, coverage measurement, and test organization for production-ready ML infrastructure.

## Solution Structure

```
exercise-07/
├── README.md                      # This file
├── IMPLEMENTATION_GUIDE.md        # Step-by-step implementation guide
├── TESTING_GUIDELINES.md          # Testing best practices
├── conftest.py                    # Pytest configuration and shared fixtures
├── pytest.ini                     # Pytest settings
├── .coveragerc                    # Coverage configuration
├── src/                           # Source code to test
│   ├── __init__.py
│   ├── preprocessing.py           # Data preprocessing functions
│   ├── metrics.py                 # ML metrics calculations
│   ├── validation.py              # Input validation functions
│   └── async_utils.py             # Async utility functions
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_basics.py             # Basic pytest examples
│   ├── test_fixtures.py           # Fixture usage
│   ├── test_parametrized.py       # Parametrized tests
│   ├── test_preprocessing.py      # Data processing tests
│   ├── test_metrics.py            # Metrics calculation tests
│   ├── test_mocking.py            # Mocking external dependencies
│   ├── test_error_handling.py     # Error handling tests
│   ├── test_async.py              # Async function tests
│   ├── test_coverage_example.py   # Coverage demonstration
│   ├── test_integration.py        # Integration tests
│   ├── test_markers.py            # Custom markers
│   └── test_validation.py         # Meta tests
└── docs/
    └── ANSWERS.md                 # Reflection question answers
```

## Key Features

### 1. Comprehensive Test Coverage

**Unit Tests**
- Test individual functions in isolation
- Cover happy paths, edge cases, and error conditions
- Use parametrized tests for multiple scenarios
- Examples: `test_basics.py`, `test_preprocessing.py`

**Integration Tests**
- Test complete workflows and component interactions
- Validate data pipelines end-to-end
- Example: `test_integration.py`

**Async Tests**
- Test asynchronous functions with `pytest-asyncio`
- Validate concurrent operations
- Example: `test_async.py`

### 2. Pytest Fixtures

**Function-Scoped Fixtures**
```python
@pytest.fixture
def sample_predictions():
    """Fresh predictions for each test."""
    return [1, 0, 1, 1, 0, 1, 0, 0]
```

**Module-Scoped Fixtures**
```python
@pytest.fixture(scope="module")
def large_dataset():
    """Expensive setup shared across module."""
    return [{"id": i, "value": i * 2} for i in range(10000)]
```

**Session-Scoped Fixtures**
```python
@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Shared temporary directory for all tests."""
    return tmp_path_factory.mktemp("test_data")
```

### 3. Parametrized Testing

**Single Parameter**
```python
@pytest.mark.parametrize("value,expected", [
    (5, 0.5),   # Middle value
    (0, 0.0),   # Min value
    (10, 1.0),  # Max value
])
def test_normalize(value, expected):
    result = normalize_value(value, 0, 10)
    assert abs(result - expected) < 1e-6
```

**Multiple Parameters (Cartesian Product)**
```python
@pytest.mark.parametrize("batch_size", [16, 32, 64])
@pytest.mark.parametrize("num_samples", [100, 500, 1000])
def test_batch_combinations(batch_size, num_samples):
    # Tests 3 × 3 = 9 combinations
    num_batches = (num_samples + batch_size - 1) // batch_size
    assert num_batches > 0
```

### 4. Mocking External Dependencies

**File I/O Mocking**
```python
def test_load_config_mock():
    mock_config = {"model": "resnet", "lr": 0.001}
    mock_data = json.dumps(mock_config)

    with patch("builtins.open", mock_open(read_data=mock_data)):
        config = load_model_config("config.json")
        assert config == mock_config
```

**API Mocking**
```python
def test_download_model_mock():
    mock_response = Mock()
    mock_response.json.return_value = {"model": "downloaded"}

    with patch("requests.get", return_value=mock_response):
        result = download_model("https://example.com/model")
        assert result == {"model": "downloaded"}
```

### 5. Error Handling Tests

**Testing Exceptions**
```python
def test_accuracy_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        calculate_accuracy([1, 0], [1])

def test_type_error():
    with pytest.raises(TypeError, match="must be integer"):
        validate_batch_size("32")
```

**Parametrized Error Tests**
```python
@pytest.mark.parametrize("invalid_value", [-1, 0, 2048, "32", 32.5, None])
def test_invalid_values(self, invalid_value):
    with pytest.raises((TypeError, ValueError)):
        validate_batch_size(invalid_value)
```

### 6. Async Testing

**Basic Async Tests**
```python
@pytest.mark.asyncio
async def test_async_single_sample():
    result = await async_process_sample(1)
    assert result["id"] == 1
    assert result["processed"] is True
```

**Async Fixtures**
```python
@pytest.fixture
async def async_sample_data():
    await asyncio.sleep(0.01)
    return {"samples": [1, 2, 3]}

@pytest.mark.asyncio
async def test_with_async_fixture(async_sample_data):
    assert len(async_sample_data["samples"]) == 3
```

### 7. Test Organization with Markers

**Custom Markers**
```python
# conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")

# tests/test_markers.py
@pytest.mark.slow
def test_slow_operation():
    time.sleep(2)
    assert True

@pytest.mark.gpu
@pytest.mark.skipif(not os.path.exists("/dev/nvidia0"),
                   reason="GPU not available")
def test_gpu_operation():
    pass
```

**Running Specific Tests**
```bash
pytest -m slow                    # Run only slow tests
pytest -m "not slow"              # Skip slow tests
pytest -m integration             # Run integration tests
pytest -k "test_batch"            # Run tests matching pattern
```

### 8. Coverage Measurement

**Running with Coverage**
```bash
# Terminal output
pytest tests/ --cov=src --cov-report=term-missing

# HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Fail if coverage below threshold
pytest tests/ --cov=src --cov-fail-under=80
```

**Coverage Configuration** (`.coveragerc`)
```ini
[run]
source = src
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[report]
precision = 2
show_missing = True
skip_covered = False
```

## Running the Tests

### Basic Usage

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Show print statements
pytest -s

# Run specific file
pytest tests/test_basics.py

# Run specific test
pytest tests/test_basics.py::test_accuracy_perfect

# Run tests matching pattern
pytest -k "accuracy"
```

### Advanced Usage

```bash
# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Show slowest tests
pytest --durations=10

# Dry run (collect tests without running)
pytest --collect-only
```

### Coverage and Reports

```bash
# Coverage with terminal report
pytest --cov=src --cov-report=term-missing

# Coverage with HTML report
pytest --cov=src --cov-report=html

# Coverage with XML report (for CI)
pytest --cov=src --cov-report=xml

# JUnit XML report (for CI)
pytest --junitxml=test-results.xml
```

### Test Selection

```bash
# Run only slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run integration tests
pytest -m integration

# Multiple markers (OR)
pytest -m "slow or integration"

# Multiple markers (AND)
pytest -m "slow and integration"
```

## Test Suite Statistics

**Total Tests**: 60+
- Unit tests: 45+
- Integration tests: 8+
- Async tests: 7+

**Coverage**: >90%
- Source functions: 100% coverage
- All branches tested
- Edge cases covered

**Performance**:
- Fast tests: <1s total
- With slow tests: ~5s total
- Parallel execution: ~2s total

## Key Testing Patterns

### 1. Arrange-Act-Assert (AAA)

```python
def test_calculate_accuracy():
    # Arrange: Set up test data
    predictions = [1, 0, 1, 1]
    labels = [1, 0, 1, 1]

    # Act: Execute function
    result = calculate_accuracy(predictions, labels)

    # Assert: Verify result
    assert result == 1.0
```

### 2. Test Classes for Organization

```python
class TestRemoveOutliers:
    """Test suite for outlier removal."""

    def test_no_outliers(self):
        data = [1, 2, 3, 4, 5]
        result = remove_outliers(data)
        assert len(result) == len(data)

    def test_with_outliers(self):
        data = [1, 2, 3, 4, 5, 100]
        result = remove_outliers(data)
        assert 100 not in result
```

### 3. Fixture-Based Setup

```python
@pytest.fixture
def data_pipeline():
    """Create pipeline for tests."""
    pipeline = DataPipeline()
    yield pipeline
    pipeline.cleanup()  # Teardown

def test_pipeline(data_pipeline):
    data_pipeline.load([1, 2, 3])
    assert len(data_pipeline.data) == 3
```

### 4. Parametrize for Scenarios

```python
@pytest.mark.parametrize("input_data,expected", [
    ([1, 2, 3], 2.0),           # Normal case
    ([1], 1.0),                 # Single value
    ([], 0.0),                  # Empty
    ([1, 1, 1], 1.0),           # All same
])
def test_calculate_mean(input_data, expected):
    result = calculate_mean(input_data)
    assert abs(result - expected) < 1e-6
```

## Testing Best Practices

### ✓ DO

- **Test behavior, not implementation**
- **Write tests first (TDD) when possible**
- **Use descriptive test names**
- **Test one thing per test**
- **Mock external dependencies**
- **Use fixtures for common setup**
- **Parametrize similar tests**
- **Aim for 80%+ coverage**
- **Run tests before committing**
- **Keep tests fast (<5s total)**

### ✗ DON'T

- **Don't test framework code**
- **Don't mock what you're testing**
- **Don't rely on test execution order**
- **Don't use real external services**
- **Don't skip failing tests**
- **Don't test private methods directly**
- **Don't duplicate production code in tests**
- **Don't ignore warnings**

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio

      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          files: ./coverage.xml
```

## ML-Specific Testing Considerations

### 1. Testing Numerical Functions

```python
def test_numerical_function():
    result = complex_calculation(data)
    expected = known_result
    # Use tolerance for floating point
    assert abs(result - expected) < 1e-6
```

### 2. Testing with Random Data

```python
@pytest.fixture
def deterministic_random():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed()  # Reset

def test_with_random(deterministic_random):
    data = np.random.randn(100)
    result = process(data)
    # Results should be reproducible
```

### 3. Testing Model Predictions

```python
def test_model_predictions(mock_model):
    """Test prediction shape and range."""
    inputs = np.random.randn(10, 5)
    predictions = mock_model.predict(inputs)

    assert predictions.shape == (10, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))
```

## Common Pitfalls and Solutions

### Pitfall 1: Tests Depend on External State

```python
# ✗ BAD: Depends on external file
def test_load():
    data = load_file("/path/to/file.csv")
    assert len(data) > 0

# ✓ GOOD: Use tmp_path fixture
def test_load(tmp_path):
    test_file = tmp_path / "test.csv"
    test_file.write_text("col1,col2\n1,2\n")
    data = load_file(str(test_file))
    assert len(data) == 1
```

### Pitfall 2: Tests Are Too Slow

```python
# ✗ BAD: Tests real API every time
def test_api():
    response = requests.get("https://real-api.com/data")
    assert response.status_code == 200

# ✓ GOOD: Mock the API
def test_api(mock_requests):
    mock_requests.get.return_value.status_code = 200
    response = requests.get("https://real-api.com/data")
    assert response.status_code == 200
```

### Pitfall 3: Not Testing Edge Cases

```python
# ✗ BAD: Only tests happy path
def test_divide():
    assert divide(10, 2) == 5

# ✓ GOOD: Tests edge cases
@pytest.mark.parametrize("a,b,expected", [
    (10, 2, 5),           # Happy path
    (0, 5, 0),            # Zero numerator
    (10, 1, 10),          # Divide by 1
])
def test_divide(a, b, expected):
    assert divide(a, b) == expected

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)
```

## Key Takeaways

1. **Pytest is powerful**: Fixtures, parametrization, and markers enable efficient testing
2. **Test early, test often**: Catch bugs before production
3. **Coverage ≠ quality**: 100% coverage doesn't guarantee correctness
4. **Mock external dependencies**: Tests should be isolated and fast
5. **Organize tests logically**: Use classes, modules, and markers
6. **Async testing works**: Use `pytest-asyncio` for async code
7. **CI integration is essential**: Automate testing on every commit

## Next Steps

- **Set up CI/CD**: Add automated testing to your projects
- **Practice TDD**: Write tests before implementation
- **Explore advanced patterns**: Property-based testing with Hypothesis
- **Learn mocking deeply**: Master `unittest.mock` and `pytest-mock`
- **Measure test quality**: Use mutation testing to validate tests

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Test-Driven Development](https://testdriven.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)

---

**Congratulations!** You've completed Module 001: Python Fundamentals. You now have a solid foundation in Python for ML infrastructure, including testing practices that ensure code reliability and maintainability.
