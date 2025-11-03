# Exercise 07: Testing - Implementation Guide

## Overview

This guide walks you through implementing a comprehensive test suite for ML utility functions using pytest. Follow the steps in order to build understanding progressively.

## Prerequisites

```bash
# Install required packages
pip install pytest pytest-cov pytest-asyncio pytest-mock

# Verify installation
pytest --version
```

## Implementation Steps

### Part 1: Basic Testing (20 minutes)

#### Step 1: Create Your First Test File

**File**: `tests/test_basics.py`

1. Create a simple function to test
2. Write basic assertions
3. Test happy path and edge cases
4. Run tests with `pytest tests/test_basics.py -v`

**Key Concepts**:
- Test function naming (`test_*`)
- Assert statements
- pytest.raises() for exceptions
- Test discovery

### Part 2: Fixtures (15 minutes)

#### Step 2: Use Fixtures for Test Data

**File**: `tests/test_fixtures.py`, `conftest.py`

1. Create function-scoped fixtures
2. Use `tmp_path` for temporary files
3. Create module-scoped fixtures for expensive setup
4. Share fixtures via `conftest.py`

**Key Concepts**:
- @pytest.fixture decorator
- Fixture scopes (function, module, session)
- Built-in fixtures (tmp_path, tmp_path_factory)
- Fixture reuse via conftest.py

### Part 3: Parametrized Tests (15 minutes)

#### Step 3: Test Multiple Scenarios

**File**: `tests/test_parametrized.py`

1. Use `@pytest.mark.parametrize` for single parameter
2. Test multiple parameters (Cartesian product)
3. Create parametrized fixtures
4. Name test cases for clarity

**Key Concepts**:
- @pytest.mark.parametrize decorator
- Multiple parametrize decorators
- Parametrized fixtures
- Test case IDs

### Part 4: Testing ML Functions (25 minutes)

#### Step 4: Test Data Preprocessing

**Files**: `src/preprocessing.py`, `tests/test_preprocessing.py`

1. Implement data processing functions
2. Test with various data types and shapes
3. Test edge cases (empty, None, outliers)
4. Organize tests in classes

**Key Concepts**:
- Testing numerical operations
- Tolerance for floating point comparisons
- Test organization with classes
- Edge case identification

#### Step 5: Test ML Metrics

**Files**: `src/metrics.py`, `tests/test_metrics.py`

1. Implement metric calculations (accuracy, precision, recall)
2. Test with known values
3. Test edge cases (empty, perfect scores, zero)
4. Parametrize for multiple metrics

### Part 5: Mocking (20 minutes)

#### Step 6: Mock External Dependencies

**File**: `tests/test_mocking.py`

1. Mock file I/O with `mock_open`
2. Mock HTTP requests with `Mock`
3. Use `@patch` decorator
4. Create mock classes

**Key Concepts**:
- unittest.mock module
- mock_open for file operations
- @patch decorator
- Mock objects and return_value

### Part 6: Error Handling Tests (15 minutes)

#### Step 7: Test Exception Handling

**Files**: `src/validation.py`, `tests/test_error_handling.py`

1. Test that exceptions are raised
2. Verify exception messages
3. Test different exception types
4. Parametrize invalid inputs

**Key Concepts**:
- pytest.raises context manager
- Exception message matching
- Multiple exception types
- Parametrized error tests

### Part 7: Async Testing (15 minutes)

#### Step 8: Test Async Functions

**Files**: `src/async_utils.py`, `tests/test_async.py`

1. Mark tests with `@pytest.mark.asyncio`
2. Test async functions with await
3. Create async fixtures
4. Test concurrent operations

**Key Concepts**:
- @pytest.mark.asyncio decorator
- Async fixtures
- await in tests
- Testing asyncio.gather

### Part 8: Coverage (10 minutes)

#### Step 9: Measure Test Coverage

**Files**: `.coveragerc`, `tests/test_coverage_example.py`

1. Run tests with coverage: `pytest --cov=src`
2. Generate HTML report: `pytest --cov=src --cov-report=html`
3. Identify untested branches
4. Write tests for 100% coverage

**Key Concepts**:
- pytest-cov plugin
- Coverage reports
- Branch coverage
- Coverage configuration

### Part 9: Integration Tests (15 minutes)

#### Step 10: Test Complete Workflows

**File**: `tests/test_integration.py`

1. Create a data pipeline class
2. Test end-to-end workflow
3. Test failure scenarios
4. Validate state transitions

**Key Concepts**:
- Integration vs unit tests
- Testing workflows
- State validation
- Error propagation

### Part 10: Test Organization (10 minutes)

#### Step 11: Organize Test Suite

**Files**: `conftest.py`, `pytest.ini`, `tests/test_markers.py`

1. Define custom markers
2. Create session-scoped fixtures
3. Configure pytest.ini
4. Run specific test subsets

**Key Concepts**:
- pytest.ini configuration
- Custom markers
- Test selection (-m, -k)
- Test organization patterns

## Running Tests

### Basic Commands

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
```

### With Coverage

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# HTML report
pytest --cov=src --cov-report=html

# Fail if coverage below 80%
pytest --cov=src --cov-fail-under=80
```

### Test Selection

```bash
# Run slow tests only
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run tests matching pattern
pytest -k "accuracy or precision"

# Run last failed tests
pytest --lf
```

### Advanced Options

```bash
# Parallel execution
pytest -n auto

# Stop on first failure
pytest -x

# Show slowest tests
pytest --durations=10

# Collect tests without running
pytest --collect-only
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .

# Or use pytest-pythonpath
pip install pytest-pythonpath
# Add to pytest.ini:
# [pytest]
# python_paths = .
```

### Issue 2: Fixture Not Found

**Problem**: `fixture 'my_fixture' not found`

**Solution**:
- Ensure fixture is in `conftest.py` or same file
- Check fixture name spelling
- Verify conftest.py is in correct location
- Use `pytest --fixtures` to list available fixtures

### Issue 3: Async Tests Not Running

**Problem**: `RuntimeWarning: coroutine 'test_async' was never awaited`

**Solution**:
```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Mark test with decorator
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### Issue 4: Coverage Not Working

**Problem**: Coverage shows 0% or incorrect values

**Solution**:
```bash
# Specify source explicitly
pytest --cov=src --cov-report=term

# Check .coveragerc exists and is correct
# Ensure tests import from src, not absolute paths
```

## Implementation Checklist

- [ ] Part 1: Basic tests with assertions
- [ ] Part 2: Fixtures for test data
- [ ] Part 3: Parametrized tests
- [ ] Part 4: ML function tests (preprocessing, metrics)
- [ ] Part 5: Mocking external dependencies
- [ ] Part 6: Error handling tests
- [ ] Part 7: Async function tests
- [ ] Part 8: Coverage measurement (>80%)
- [ ] Part 9: Integration tests
- [ ] Part 10: Test organization with markers
- [ ] All tests pass: `pytest -v`
- [ ] Coverage meets threshold: `pytest --cov=src --cov-fail-under=80`
- [ ] Documentation complete: ANSWERS.md

## Testing Workflow

### 1. Before Writing Code (TDD Approach)

```python
# Write test first
def test_new_feature():
    result = new_feature(input_data)
    assert result == expected_output

# Run test (should fail)
pytest tests/test_new_feature.py

# Implement feature
def new_feature(data):
    # Implementation

# Run test (should pass)
pytest tests/test_new_feature.py
```

### 2. After Writing Code

```python
# Write comprehensive tests
class TestNewFeature:
    def test_happy_path(self):
        pass

    def test_edge_cases(self):
        pass

    def test_error_handling(self):
        pass

# Measure coverage
pytest --cov=src/new_feature.py --cov-report=term-missing

# Add tests for uncovered lines
```

### 3. Before Committing

```bash
# Run all tests
pytest

# Check coverage
pytest --cov=src --cov-fail-under=80

# Run linter
flake8 src/ tests/

# Format code
black src/ tests/
```

## Best Practices Summary

### Test Structure (AAA Pattern)

```python
def test_function():
    # Arrange: Set up test data
    data = create_test_data()

    # Act: Execute function
    result = function_under_test(data)

    # Assert: Verify result
    assert result == expected_value
```

### Test Naming

```python
# ✓ Good: Descriptive, indicates scenario
def test_calculate_accuracy_with_perfect_predictions():
    pass

def test_calculate_accuracy_with_empty_input_returns_zero():
    pass

# ✗ Bad: Vague, unclear purpose
def test_accuracy():
    pass

def test_1():
    pass
```

### Test Independence

```python
# ✓ Good: Each test is independent
def test_a():
    data = [1, 2, 3]
    result = process(data)
    assert result == 6

def test_b():
    data = [4, 5, 6]
    result = process(data)
    assert result == 15

# ✗ Bad: Tests depend on execution order
shared_data = []

def test_first():
    shared_data.append(1)
    assert len(shared_data) == 1

def test_second():  # Fails if run alone
    assert len(shared_data) == 1
```

## Time Estimates

| Part | Task | Estimated Time |
|------|------|----------------|
| 1 | Basic testing | 20 min |
| 2 | Fixtures | 15 min |
| 3 | Parametrized tests | 15 min |
| 4 | ML function tests | 25 min |
| 5 | Mocking | 20 min |
| 6 | Error handling | 15 min |
| 7 | Async testing | 15 min |
| 8 | Coverage | 10 min |
| 9 | Integration tests | 15 min |
| 10 | Organization | 10 min |
| **Total** | | **2-3 hours** |

## Success Criteria

✓ All tests pass (`pytest -v`)
✓ Coverage >80% (`pytest --cov=src --cov-fail-under=80`)
✓ No warnings (`pytest -W error`)
✓ Fast execution (<5s for unit tests)
✓ Tests are independent (can run in any order)
✓ Clear test names and documentation
✓ Proper use of fixtures and parametrization
✓ Mocking for external dependencies
✓ Integration tests for workflows
✓ Organized with custom markers

## Next Steps

After completing this exercise:

1. **Apply to real projects**: Add tests to your ML code
2. **Set up CI/CD**: Automate testing with GitHub Actions
3. **Explore advanced topics**:
   - Property-based testing (Hypothesis)
   - Mutation testing (mutmut)
   - Performance testing (pytest-benchmark)
4. **Move to Module 002**: Linux Essentials

---

**Remember**: Good tests give you confidence to refactor, catch regressions early, and serve as documentation for how code should behave.
