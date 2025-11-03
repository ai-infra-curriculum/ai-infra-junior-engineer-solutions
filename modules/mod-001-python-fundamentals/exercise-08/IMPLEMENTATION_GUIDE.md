# Implementation Guide: Python Packaging and Distribution

This guide provides step-by-step instructions for implementing the `ml-infra-utils` package from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Implementing Core Utilities](#implementing-core-utilities)
4. [Writing Tests](#writing-tests)
5. [Configuration Files](#configuration-files)
6. [Building and Testing](#building-and-testing)
7. [Publishing](#publishing)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

```bash
# Python 3.8 or higher
python --version

# Install packaging tools
pip install --upgrade pip setuptools wheel build twine pytest pytest-cov
```

### Knowledge Requirements

- Python basics (functions, classes, modules)
- Understanding of decorators
- Basic testing with pytest
- Git version control

---

## Project Setup

### Step 1: Create Directory Structure

```bash
# Create main directory
mkdir ml-infra-utils
cd ml-infra-utils

# Create subdirectories
mkdir -p src/ml_infra_utils/{preprocessing,metrics,decorators,logging}
mkdir -p tests
mkdir -p examples
mkdir -p docs

# Create __init__.py files
touch src/ml_infra_utils/__init__.py
touch src/ml_infra_utils/preprocessing/__init__.py
touch src/ml_infra_utils/metrics/__init__.py
touch src/ml_infra_utils/decorators/__init__.py
touch src/ml_infra_utils/logging/__init__.py
touch tests/__init__.py
```

### Step 2: Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial project structure"
```

---

## Implementing Core Utilities

### Step 3: Preprocessing Module

#### 3.1: Create `src/ml_infra_utils/preprocessing/normalization.py`

Implement three normalization functions:
- `normalize()`: Min-max scaling to [0, 1]
- `standardize()`: Z-score normalization (mean=0, std=1)
- `clip_outliers()`: Percentile-based outlier clipping

**Key implementation details:**
- Use type hints for all parameters and return values
- Include comprehensive docstrings with examples
- Handle edge cases (empty data, identical values)
- Raise informative exceptions with clear messages

```python
def normalize(data: List[float]) -> List[float]:
    """Normalize data to [0, 1] range using min-max scaling."""
    if not data:
        raise ValueError("Cannot normalize empty data")

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        raise ValueError("All values are identical, cannot normalize")

    range_val = max_val - min_val
    return [(x - min_val) / range_val for x in data]
```

#### 3.2: Create `src/ml_infra_utils/preprocessing/validation.py`

Implement data validation functions:
- `check_missing_values()`: Detect None and NaN
- `validate_range()`: Check values within bounds
- `check_data_types()`: Validate type consistency

#### 3.3: Update `src/ml_infra_utils/preprocessing/__init__.py`

Export all public functions:

```python
from ml_infra_utils.preprocessing.normalization import (
    normalize,
    standardize,
    clip_outliers,
)
from ml_infra_utils.preprocessing.validation import (
    check_missing_values,
    validate_range,
    check_data_types,
)

__all__ = [
    "normalize",
    "standardize",
    "clip_outliers",
    "check_missing_values",
    "validate_range",
    "check_data_types",
]
```

### Step 4: Metrics Module

#### 4.1: Create `src/ml_infra_utils/metrics/classification.py`

Implement classification metrics:
- `accuracy()`: Correct predictions / total predictions
- `precision()`: True positives / (true positives + false positives)
- `recall()`: True positives / (true positives + false negatives)
- `f1_score()`: Harmonic mean of precision and recall

**Key implementation details:**
- Support multi-class classification with `positive_class` parameter
- Handle edge cases (no predictions, no actual positives)
- Include length validation

```python
def accuracy(predictions: List[int], labels: List[int]) -> float:
    """Calculate accuracy score."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)
```

### Step 5: Decorators Module

#### 5.1: Create `src/ml_infra_utils/decorators/timing.py`

Implement timing decorators:
- `@timer`: Simple execution time measurement
- `@timer_with_units()`: Configurable time units

**Key implementation details:**
- Use `functools.wraps` to preserve function metadata
- Support multiple time units (seconds, milliseconds, microseconds)
- Print timing information to stdout

```python
import functools
import time

def timer(func: Callable) -> Callable:
    """Decorator to measure and print function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper
```

#### 5.2: Create `src/ml_infra_utils/decorators/retry.py`

Implement retry decorators:
- `@retry()`: Exponential backoff retry
- `@retry_on_condition()`: Condition-based retry

**Key implementation details:**
- Support configurable max attempts
- Implement exponential backoff (delay *= backoff)
- Allow specific exception types
- Print retry status

### Step 6: Logging Module

#### 6.1: Create `src/ml_infra_utils/logging/structured.py`

Implement structured logging:
- `StructuredLogger`: Main logger class
- `JsonFormatter`: Custom JSON formatter
- `log_ml_event()`: ML-specific helper

**Key implementation details:**
- Output JSON-formatted logs
- Include timestamp, level, message, context
- Support additional fields via **kwargs

```python
import json
import logging
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "context"):
            log_data["context"] = record.context

        return json.dumps(log_data)
```

### Step 7: Main Package Init

#### 7.1: Create `src/ml_infra_utils/__init__.py`

Define package metadata and exports:

```python
"""ML Infrastructure Utilities Package."""

__version__ = "0.1.0"
__author__ = "ML Infrastructure Team"
__email__ = "ml-team@example.com"

# Import commonly used functions
from ml_infra_utils.preprocessing.normalization import normalize, standardize
from ml_infra_utils.metrics.classification import accuracy, precision, recall, f1_score
from ml_infra_utils.decorators.timing import timer
from ml_infra_utils.decorators.retry import retry
from ml_infra_utils.logging.structured import StructuredLogger

__all__ = [
    "normalize",
    "standardize",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "timer",
    "retry",
    "StructuredLogger",
    "__version__",
]
```

---

## Writing Tests

### Step 8: Test Suite

#### 8.1: Create `tests/test_preprocessing.py`

Test all preprocessing functions:
- Normal operation
- Edge cases (empty data, identical values)
- Error conditions (invalid input)
- Boundary values

```python
import pytest
from ml_infra_utils.preprocessing import normalize

def test_normalize_basic():
    """Test basic normalization."""
    data = [1, 2, 3, 4, 5]
    result = normalize(data)
    assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

def test_normalize_empty_raises():
    """Test normalization with empty data raises ValueError."""
    with pytest.raises(ValueError, match="Cannot normalize empty data"):
        normalize([])
```

#### 8.2: Create `tests/test_metrics.py`

Test all classification metrics:
- Perfect predictions
- Partial correctness
- Zero scores
- Multi-class scenarios
- Length mismatches

#### 8.3: Create `tests/test_decorators.py`

Test decorator functionality:
- Timer outputs
- Retry behavior
- Exponential backoff
- Exception handling
- Condition-based retry

**Testing tips:**
- Use `capsys` fixture to capture printed output
- Test both success and failure paths
- Use `pytest.approx()` for float comparisons
- Mock time-dependent operations where appropriate

#### 8.4: Create `tests/test_logging.py`

Test logging functionality:
- JSON output format
- Log levels
- Context inclusion
- ML event logging

### Step 9: Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=ml_infra_utils --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py
```

---

## Configuration Files

### Step 10: Create pyproject.toml (Modern Method - Recommended)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ml-infra-utils"
version = "0.1.0"
description = "Reusable ML infrastructure utilities"
readme = "README.md"
requires-python = ">=3.8"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=ml_infra_utils --cov-report=html"
```

### Step 11: Create setup.py (Legacy Method)

For backward compatibility:

```python
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from package
about = {}
with open("src/ml_infra_utils/__init__.py", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            about["__version__"] = line.split('"')[1]
            break

setup(
    name="ml-infra-utils",
    version=about["__version__"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
```

### Step 12: Create Supporting Files

#### 12.1: README.md
- Package overview
- Installation instructions
- Quick start examples
- API documentation

#### 12.2: LICENSE
- Choose appropriate license (MIT recommended)
- Include copyright year and holder

#### 12.3: CHANGELOG.md
- Follow Keep a Changelog format
- Document all changes by version

#### 12.4: .gitignore
- Exclude __pycache__, *.pyc
- Exclude dist/, build/, *.egg-info/
- Exclude virtual environments
- Exclude IDE files

---

## Building and Testing

### Step 13: Build Package

```bash
# Install build tool
pip install build

# Build both sdist and wheel
python -m build

# Output:
# dist/ml-infra-utils-0.1.0.tar.gz  (source distribution)
# dist/ml_infra_utils-0.1.0-py3-none-any.whl  (wheel)
```

### Step 14: Verify Package

```bash
# Check package metadata
twine check dist/*

# List wheel contents
unzip -l dist/ml_infra_utils-0.1.0-py3-none-any.whl

# Extract and inspect sdist
tar -tzf dist/ml-infra-utils-0.1.0.tar.gz
```

### Step 15: Test Installation

```bash
# Create test environment
python -m venv test-env
source test-env/bin/activate

# Install from wheel
pip install dist/ml_infra_utils-0.1.0-py3-none-any.whl

# Test imports
python -c "from ml_infra_utils import normalize, accuracy, timer; print('✅ Import successful')"

# Test functionality
python -c "from ml_infra_utils import normalize; print(normalize([1,2,3,4,5]))"

# Deactivate and clean up
deactivate
rm -rf test-env
```

---

## Publishing

### Step 16: Local Package Index (Option 1: devpi)

```bash
# Install devpi
pip install devpi-server devpi-client

# Initialize and start server
devpi-init
devpi-server --start --port 3141

# Create user and index
devpi use http://localhost:3141
devpi user -c testuser password=testpass
devpi login testuser --password=testpass
devpi index -c dev bases=root/pypi
devpi use testuser/dev

# Upload package
devpi upload dist/*

# Install from devpi
pip install --index-url http://localhost:3141/testuser/dev/+simple/ ml-infra-utils
```

### Step 17: Simple HTTP Server (Option 2: Quick Testing)

```bash
# Create package directory
mkdir -p ~/pypi-packages/ml-infra-utils
cp dist/* ~/pypi-packages/ml-infra-utils/

# Start server
cd ~/pypi-packages
python -m http.server 8080

# In another terminal, install
pip install --index-url http://localhost:8080/simple/ ml-infra-utils
```

### Step 18: PyPI (Option 3: Public Distribution)

```bash
# Create PyPI account at https://pypi.org

# Create API token at https://pypi.org/manage/account/token/

# Configure credentials in ~/.pypirc
cat > ~/.pypirc << EOF
[pypi]
username = __token__
password = pypi-your-api-token-here
EOF

# Upload to PyPI
twine upload dist/*

# Install from PyPI
pip install ml-infra-utils
```

---

## Troubleshooting

### Common Issues

#### Issue 1: ModuleNotFoundError after installation

**Cause**: Package not installed in current environment

**Solution**:
```bash
# Check which Python is being used
which python
python -c "import sys; print(sys.executable)"

# Check if package is installed
pip list | grep ml-infra-utils

# Reinstall if needed
pip install -e .
```

#### Issue 2: Package doesn't include source files

**Cause**: Incorrect package discovery in setup.py/pyproject.toml

**Solution**:
```python
# In setup.py, ensure correct package_dir and packages
setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)

# In pyproject.toml
[tool.setuptools.packages.find]
where = ["src"]
```

#### Issue 3: Tests can't import package

**Cause**: Package not installed or wrong Python path

**Solution**:
```bash
# Install package in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Issue 4: Build fails with "No module named setuptools"

**Cause**: setuptools not installed or outdated

**Solution**:
```bash
pip install --upgrade pip setuptools wheel
```

#### Issue 5: Wheel is platform-specific instead of universal

**Cause**: C extensions or platform-specific code

**Solution**:
- For pure Python packages, wheel should be `py3-none-any`
- Check for accidentally included compiled files
- Ensure no C extensions in package

---

## Best Practices

### Versioning Strategy

1. **Pre-1.0 Development**
   - Start with 0.1.0
   - Increment MINOR for new features
   - Increment PATCH for bug fixes
   - Breaking changes allowed in MINOR

2. **Post-1.0 Stable**
   - MAJOR: Breaking changes
   - MINOR: New features (backward compatible)
   - PATCH: Bug fixes only

### Dependency Management

1. **Libraries vs Applications**
   - Libraries: Use version ranges (`>=1.0,<2.0`)
   - Applications: Pin exact versions (`==1.2.3`)

2. **Optional Dependencies**
   - Group related dependencies (`dev`, `docs`, `test`)
   - Users install only what they need

### Testing Strategy

1. **Coverage Goals**
   - Aim for >90% code coverage
   - Focus on critical paths
   - Test edge cases

2. **Test Organization**
   - One test file per source file
   - Group related tests in classes
   - Use descriptive test names

### Documentation

1. **Docstrings**
   - Every public function/class
   - Include parameters, returns, raises
   - Add usage examples

2. **README**
   - Quick start guide
   - Installation instructions
   - Basic examples

3. **CHANGELOG**
   - Document all changes
   - Group by type (Added, Changed, Fixed)
   - Link to releases

---

## Next Steps

After completing this implementation:

1. **Enhance Package**
   - Add more utilities
   - Implement CLI tools
   - Create comprehensive documentation site

2. **Automation**
   - Set up CI/CD (GitHub Actions)
   - Automate testing and publishing
   - Add pre-commit hooks

3. **Distribution**
   - Publish to PyPI
   - Create releases with tags
   - Announce to users

---

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PEP 517](https://peps.python.org/pep-0517/) - Build System Interface
- [PEP 518](https://peps.python.org/pep-0518/) - pyproject.toml
- [PEP 621](https://peps.python.org/pep-0621/) - Project Metadata

---

**Guide Version**: 1.0.0
**Last Updated**: 2025-10-30
**Estimated Time**: 2-3 hours
