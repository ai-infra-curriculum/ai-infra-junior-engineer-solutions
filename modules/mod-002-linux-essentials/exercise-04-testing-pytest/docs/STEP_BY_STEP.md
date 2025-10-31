# Step-by-Step Implementation Guide: Testing with Pytest

## Overview

Master testing ML infrastructure code with pytest. Learn unit testing, fixtures, mocking, parametrization, test organization, and continuous testing practices for production Python code.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Write unit tests with pytest
✅ Use fixtures for test setup
✅ Mock external dependencies
✅ Parametrize tests for multiple scenarios
✅ Test ML code effectively
✅ Measure code coverage
✅ Integrate tests into CI/CD

---

## Setup

```bash
# Install pytest
pip install pytest pytest-cov pytest-mock

# Project structure
project/
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
└── tests/
    ├── __init__.py
    ├── test_model.py
    └── test_utils.py
```

---

## Phase 1: Basic Testing

### Simple Test

```python
# tests/test_utils.py
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::test_add

# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x
```

---

## Phase 2: Test Organization

### Test Class

```python
class TestDataProcessor:
    """Group related tests"""

    def test_load_data(self):
        processor = DataProcessor()
        data = processor.load('data.csv')
        assert len(data) > 0

    def test_preprocess(self):
        processor = DataProcessor()
        data = [[1, 2], [3, 4]]
        result = processor.preprocess(data)
        assert result.shape == (2, 2)

    def test_validate(self):
        processor = DataProcessor()
        assert processor.validate({'col1': [1, 2]}) == True
        assert processor.validate({}) == False
```

---

## Phase 3: Fixtures

### Basic Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15

def test_len(sample_data):
    assert len(sample_data) == 5
```

### Fixture Scopes

```python
@pytest.fixture(scope="function")  # Default: per test
def temp_file():
    f = open('temp.txt', 'w')
    yield f
    f.close()
    os.remove('temp.txt')

@pytest.fixture(scope="class")  # Per test class
def database():
    db = Database()
    db.connect()
    yield db
    db.disconnect()

@pytest.fixture(scope="module")  # Per test file
def ml_model():
    model = load_model('model.pth')
    yield model

@pytest.fixture(scope="session")  # Once per test session
def test_config():
    return {'batch_size': 32, 'epochs': 10}
```

### Setup and Teardown

```python
@pytest.fixture
def model_checkpoint(tmp_path):
    """Create temporary model checkpoint"""
    checkpoint_path = tmp_path / "model.pth"

    # Setup
    model = create_model()
    torch.save(model.state_dict(), checkpoint_path)

    yield checkpoint_path

    # Teardown (after test)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
```

---

## Phase 4: Parametrization

### Parametrize Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
    (5, 25),
])
def test_square(input, expected):
    assert input ** 2 == expected

# Multiple parameters
@pytest.mark.parametrize("model_name,expected_layers", [
    ("resnet18", 18),
    ("resnet34", 34),
    ("resnet50", 50),
])
def test_model_layers(model_name, expected_layers):
    model = create_model(model_name)
    assert count_layers(model) == expected_layers
```

---

## Phase 5: Mocking

### Mock External Calls

```python
from unittest.mock import Mock, patch, MagicMock

# Mock API call
def test_api_call(mocker):
    # Mock requests.get
    mock_response = Mock()
    mock_response.json.return_value = {'status': 'success'}
    mock_response.status_code = 200

    mocker.patch('requests.get', return_value=mock_response)

    result = call_api('http://example.com')
    assert result['status'] == 'success'

# Mock file operations
def test_load_config(mocker):
    mock_open = mocker.mock_open(read_data='{"key": "value"}')
    mocker.patch('builtins.open', mock_open)

    config = load_config('config.json')
    assert config['key'] == 'value'

# Mock database
def test_save_model(mocker):
    mock_db = mocker.Mock()
    mocker.patch('database.connect', return_value=mock_db)

    save_model(model, 'model_id')

    mock_db.save.assert_called_once_with('model_id', model)
```

---

## Phase 6: Testing ML Code

### Test Data Pipeline

```python
def test_data_loader():
    """Test data loading"""
    loader = DataLoader('data.csv', batch_size=32)

    batch = next(iter(loader))
    assert batch.shape == (32, 10)  # 32 samples, 10 features

def test_preprocessing():
    """Test data preprocessing"""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    preprocessor = Preprocessor()

    result = preprocessor.fit_transform(data)

    # Check output shape
    assert result.shape == data.shape

    # Check normalization
    assert np.allclose(result.mean(), 0, atol=1e-7)
    assert np.allclose(result.std(), 1, atol=1e-7)
```

### Test Model

```python
def test_model_forward():
    """Test model forward pass"""
    model = MyModel(input_dim=10, output_dim=2)
    model.eval()

    x = torch.randn(32, 10)  # Batch of 32
    output = model(x)

    assert output.shape == (32, 2)
    assert torch.isfinite(output).all()

def test_model_training_step():
    """Test single training step"""
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Create batch
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))

    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Verify loss is finite
    assert torch.isfinite(loss)

    # Verify gradients exist
    for param in model.parameters():
        assert param.grad is not None
```

### Test Model Saving/Loading

```python
def test_model_save_load(tmp_path):
    """Test model serialization"""
    # Create and save model
    model = MyModel()
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)

    # Load model
    loaded_model = MyModel()
    loaded_model.load_state_dict(torch.load(path))

    # Compare outputs
    x = torch.randn(1, 10)
    with torch.no_grad():
        output1 = model(x)
        output2 = loaded_model(x)

    assert torch.allclose(output1, output2)
```

---

## Phase 7: Code Coverage

```bash
# Run tests with coverage
pytest --cov=src tests/

# Generate HTML report
pytest --cov=src --cov-report=html tests/

# View report
open htmlcov/index.html

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=80 tests/
```

### Coverage Config

```ini
# .coveragerc
[run]
source = src
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

---

## Phase 8: Test Markers

```python
import pytest

# Mark slow tests
@pytest.mark.slow
def test_long_running():
    # Takes 10+ seconds
    pass

# Mark GPU tests
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_gpu_training():
    model = model.cuda()
    # ...

# Run marked tests
# pytest -m slow
# pytest -m "not slow"
# pytest -m gpu
```

---

## Phase 9: CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Best Practices

✅ Test one thing per test
✅ Use descriptive test names
✅ Keep tests independent
✅ Use fixtures for setup
✅ Mock external dependencies
✅ Test edge cases
✅ Aim for >80% coverage
✅ Run tests in CI/CD
✅ Keep tests fast
✅ Test public APIs, not internals

---

## Common Patterns

### Test Exception Handling

```python
import pytest

def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_file_not_found():
    with pytest.raises(FileNotFoundError, match="data.csv"):
        load_data("data.csv")
```

### Test Warnings

```python
def test_deprecated_function():
    with pytest.warns(DeprecationWarning):
        old_function()
```

---

**Testing with pytest mastered!** ✅
