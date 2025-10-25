# Step-by-Step Guide: Automated Testing in CI/CD

## Overview
Build a comprehensive automated testing pipeline using pytest, coverage reporting, and GitHub Actions integration for continuous quality assurance.

## Phase 1: Setup Testing Environment (10 minutes)

### Install Testing Dependencies
```bash
# Create project structure
mkdir -p automated-testing/tests
cd automated-testing

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install testing tools
pip install pytest pytest-cov pytest-mock coverage
pip freeze > requirements.txt
```

### Create Sample Application
Create `src/calculator.py`:
```python
class Calculator:
    def add(self, a: float, b: float) -> float:
        return a + b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def multiply(self, a: float, b: float) -> float:
        return a * b
```

**Validation**: Run `python -c "from src.calculator import Calculator; print(Calculator().add(2, 3))"`

## Phase 2: Write Comprehensive Tests (15 minutes)

### Create Unit Tests
Create `tests/test_calculator.py`:
```python
import pytest
from src.calculator import Calculator

class TestCalculator:
    @pytest.fixture
    def calc(self):
        return Calculator()

    def test_add_positive_numbers(self, calc):
        assert calc.add(2, 3) == 5

    def test_add_negative_numbers(self, calc):
        assert calc.add(-1, -1) == -2

    def test_divide_normal(self, calc):
        assert calc.divide(10, 2) == 5

    def test_divide_by_zero_raises_error(self, calc):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(10, 0)

    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 6),
        (0, 5, 0),
        (-2, 3, -6),
    ])
    def test_multiply_parametrized(self, calc, a, b, expected):
        assert calc.multiply(a, b) == expected
```

### Run Tests Locally
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
```

**Validation**: Confirm 100% code coverage and all tests passing.

## Phase 3: Configure pytest (10 minutes)

### Create pytest Configuration
Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80

markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### Create Coverage Configuration
Create `.coveragerc`:
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

[html]
directory = htmlcov
```

**Validation**: Run `pytest` - configuration should apply automatically.

## Phase 4: GitHub Actions Integration (15 minutes)

### Create CI Workflow
Create `.github/workflows/test.yml`:
```yaml
name: Automated Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### Add Pre-commit Hooks
Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

**Validation**: Push to GitHub and verify workflow runs successfully.

## Phase 5: Advanced Testing Patterns (15 minutes)

### Add Mocking Tests
Create `tests/test_advanced.py`:
```python
from unittest.mock import Mock, patch
import pytest

class TestMocking:
    def test_mock_external_api(self):
        # Mock external dependency
        mock_api = Mock()
        mock_api.get_data.return_value = {"status": "success"}

        result = mock_api.get_data()
        assert result["status"] == "success"
        mock_api.get_data.assert_called_once()

    @patch('src.calculator.Calculator.add')
    def test_patch_method(self, mock_add):
        mock_add.return_value = 100
        calc = Calculator()
        result = calc.add(1, 2)
        assert result == 100
```

### Create Test Reports
```bash
# Install reporting plugin
pip install pytest-html

# Generate HTML report
pytest tests/ --html=report.html --self-contained-html

# Run with junit XML (for Jenkins/CI)
pytest tests/ --junitxml=junit.xml
```

**Validation**: Open `report.html` in browser to view test results.

## Phase 6: Continuous Improvement (10 minutes)

### Add Quality Gates
Create `scripts/quality-check.sh`:
```bash
#!/bin/bash
set -e

echo "Running quality checks..."

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term --cov-fail-under=80

# Check code formatting (optional)
if command -v black &> /dev/null; then
    black --check src/ tests/
fi

# Check linting (optional)
if command -v flake8 &> /dev/null; then
    flake8 src/ tests/ --max-line-length=100
fi

echo "All quality checks passed!"
```

```bash
chmod +x scripts/quality-check.sh
./scripts/quality-check.sh
```

### Monitor Test Performance
```bash
# Install pytest-benchmark
pip install pytest-benchmark

# Add benchmark test
# In tests/test_performance.py:
def test_calculator_performance(benchmark, calc):
    result = benchmark(calc.add, 100, 200)
    assert result == 300
```

**Validation**: Run `pytest tests/test_performance.py` to see benchmark results.

## Summary

You've built a production-ready automated testing pipeline with:
- **Comprehensive test suite** using pytest with fixtures and parametrization
- **Code coverage tracking** with 80%+ threshold enforcement
- **GitHub Actions CI** running tests on multiple Python versions
- **Advanced patterns** including mocking, patching, and benchmarking
- **Quality gates** preventing low-quality code from merging
- **Multiple report formats** (HTML, XML, terminal) for different audiences

This testing infrastructure ensures code quality through automated validation on every commit and pull request.
