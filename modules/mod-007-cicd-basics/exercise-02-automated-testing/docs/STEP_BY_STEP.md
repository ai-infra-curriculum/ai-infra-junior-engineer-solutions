# Step-by-Step Implementation Guide: Automated Testing in CI/CD

## Overview

Implement comprehensive automated testing in CI/CD pipelines! Learn unit tests, integration tests, coverage reporting, test parallelization, and testing best practices for ML code.

**Time**: 2 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Integrate pytest into CI/CD
✅ Achieve high code coverage
✅ Run parallel tests
✅ Test ML models and data pipelines
✅ Generate test reports
✅ Implement test fixtures and mocks
✅ Configure test environments

---

## CI Test Workflow

```.github/workflows/test.yml
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

      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: test-${{ runner.os }}-${{ hashFiles('requirements-test.txt') }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Run unit tests
        run: pytest tests/unit -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Testing ML Code

### Model Tests

```python
# tests/test_model.py
import pytest
import torch
from src.model import MLModel

def test_model_forward():
    model = MLModel(input_dim=10, output_dim=2)
    x = torch.randn(32, 10)
    output = model(x)
    assert output.shape == (32, 2)

def test_model_training_step():
    model = MLModel()
    optimizer = torch.optim.Adam(model.parameters())
    x, y = torch.randn(32, 10), torch.randint(0, 2, (32,))

    model.train()
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
```

### Data Pipeline Tests

```python
# tests/test_pipeline.py
def test_data_preprocessing():
    pipeline = DataPipeline()
    raw_data = load_test_data()

    processed = pipeline.preprocess(raw_data)

    assert processed.shape == (100, 20)
    assert not processed.isna().any().any()

def test_data_validation():
    validator = DataValidator()
    valid_data = create_valid_dataset()
    invalid_data = create_invalid_dataset()

    assert validator.validate(valid_data)
    assert not validator.validate(invalid_data)
```

---

## Best Practices

✅ Aim for >80% code coverage
✅ Test edge cases and failure modes
✅ Use fixtures for test data
✅ Mock external dependencies
✅ Run tests in parallel (`pytest -n auto`)
✅ Separate unit and integration tests
✅ Test ML model behaviors, not just outputs
✅ Include performance regression tests

---

**Automated Testing mastered!** ✅
