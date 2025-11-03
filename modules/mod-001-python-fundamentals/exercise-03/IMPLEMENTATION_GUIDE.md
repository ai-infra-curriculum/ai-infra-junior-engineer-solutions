# Implementation Guide - Exercise 03: Functions & Modules

Step-by-step guide for implementing this exercise from scratch.

## Prerequisites

- Python 3.11+ installed
- Completed Exercise 01 (Environment Setup) and Exercise 02 (Data Structures)
- Text editor or IDE with Python support
- pytest installed (`pip install pytest`)

## Time Estimate

100-120 minutes total

## Implementation Steps

### Part 1: Function Basics (30 minutes)

**Step 1: Create function_basics.py**
- Implement `calculate_accuracy()` with type hints
- Implement `normalize_features()` with multiple methods
- Implement `split_data()` with flexible parameters
- Add comprehensive docstrings
- Include example usage in `main()`
- Run: `python scripts/function_basics.py`

**Key concepts:**
- Type hints: `List[int]`, `Optional[int]`, `Tuple[List, List, List]`
- Default arguments: `method: str = "minmax"`
- Input validation: `if len(predictions) != len(labels)`
- Docstrings with Args, Returns, Raises, Examples

### Part 2: Flexible Functions (25 minutes)

**Step 2: Create flexible_functions.py**
- Implement `log_metrics(*args, **kwargs)` for flexible logging
- Implement `create_model(model_type, *layers, **config)` mixing arg types
- Implement `batch_process(data, func, *args, **kwargs)` forwarding args
- Implement `augment_image(**transforms)` for arbitrary options
- Run: `python scripts/flexible_functions.py`

**Key concepts:**
- *args for variable positional arguments
- **kwargs for variable keyword arguments
- Combining positional, keyword, *args, and **kwargs
- Type hints with `Any` for flexibility

### Part 3: Decorators (30 minutes)

**Step 3: Create decorators.py**
- Implement `timing_decorator` to measure execution time
- Implement `log_calls` to log function invocations
- Implement `retry(max_attempts, delay)` with parameters
- Implement `cache_results` for memoization
- Implement `validate_inputs(**validators)` for input checking
- Run: `python scripts/decorators.py`

**Key concepts:**
- `@functools.wraps(func)` to preserve metadata
- Decorator factories (decorators with parameters)
- Nested decorators and execution order
- Caching with dictionary

### Part 4: Reusable Modules (40 minutes)

**Step 4: Create ml_utils package structure**
```bash
mkdir -p ml_utils
touch ml_utils/__init__.py
```

**Step 5: Create ml_utils/metrics.py**
- Implement classification metrics:
  - `accuracy()`, `precision()`, `recall()`, `f1_score()`
  - `confusion_matrix()`, `classification_report()`
- Implement regression metrics:
  - `mean_squared_error()`, `root_mean_squared_error()`
  - `mean_absolute_error()`, `r_squared()`
  - `regression_report()`

**Step 6: Create ml_utils/preprocessing.py**
- Implement normalization:
  - `normalize_minmax()`, `normalize_zscore()`
- Implement outlier handling:
  - `remove_outliers(method="iqr"|"zscore")`
- Implement missing value handling:
  - `fill_missing_values(strategy="mean"|"median"|...)`
- Implement encoding:
  - `one_hot_encode()`, `label_encode()`
- Implement data splitting:
  - `train_test_split()`, `stratified_split()`

**Step 7: Update ml_utils/__init__.py**
```python
from . import metrics
from . import preprocessing

__version__ = "0.1.0"
__all__ = ["metrics", "preprocessing"]
```

**Step 8: Create test_ml_utils.py**
- Import and test metrics module
- Import and test preprocessing module
- Run: `python scripts/test_ml_utils.py`

### Part 5: Functional Programming (20 minutes)

**Step 9: Create functional_patterns.py**
- Demonstrate `map()` for transformations
- Demonstrate `filter()` for selection
- Demonstrate `reduce()` for aggregation
- Demonstrate lambda functions
- Demonstrate function composition/pipelines
- Run: `python scripts/functional_patterns.py`

**Key concepts:**
- Lambda syntax: `lambda x: x ** 2`
- map() returns iterator (use `list()` to materialize)
- reduce() requires import from functools
- Combining map/filter/reduce for pipelines

### Part 6: Testing (15 minutes)

**Step 10: Create pytest tests**
- Create `tests/test_metrics.py`
  - Test all metrics functions
  - Test edge cases (empty lists, mismatched lengths)
  - Test error handling
- Create `tests/test_preprocessing.py`
  - Test all preprocessing functions
  - Test each normalization method
  - Test data splitting with seeds

**Run tests:**
```bash
pytest tests/ -v
pytest tests/ --cov=ml_utils --cov-report=term-missing
```

### Part 7: Validation (10 minutes)

**Step 11: Create validate_module.py**
- Validate type hints exist
- Validate decorators work
- Validate module imports
- Validate function correctness
- Validate error handling
- Validate edge cases
- Run: `python scripts/validate_module.py`

## Quick Validation

```bash
# Run all implementation scripts
python scripts/function_basics.py
python scripts/flexible_functions.py
python scripts/decorators.py
python scripts/functional_patterns.py
python scripts/test_ml_utils.py

# Validate everything
python scripts/validate_module.py

# Run pytest
pytest tests/ -v
```

## Key Concepts Checklist

- [ ] Type hints on all functions
- [ ] Comprehensive docstrings (Args, Returns, Raises, Examples)
- [ ] Input validation with clear error messages
- [ ] Default arguments for flexibility
- [ ] *args for variable positional arguments
- [ ] **kwargs for variable keyword arguments
- [ ] functools.wraps in decorators
- [ ] Decorator factories (parameterized decorators)
- [ ] Module organization with __init__.py
- [ ] map/filter/reduce for functional patterns
- [ ] Lambda functions for simple operations
- [ ] pytest for testing
- [ ] Edge case handling (empty, None, mismatched lengths)

## Common Issues

**Issue:** `ModuleNotFoundError: No module named 'ml_utils'`
**Solution:** Add parent directory to Python path:
```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Issue:** Decorator loses function metadata (`__name__`, `__doc__`)
**Solution:** Use `@functools.wraps(func)`:
```python
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

**Issue:** Type hints with circular imports
**Solution:** Use string annotations:
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_utils import metrics
```

**Issue:** Cache in decorator grows unbounded
**Solution:** Use `functools.lru_cache` with maxsize:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(x):
    return x ** 2
```

## Code Style Guidelines

1. **Function naming:** lowercase_with_underscores
2. **Type hints:** Always include on parameters and return values
3. **Docstrings:** Google or NumPy style, include examples
4. **Line length:** Max 88 characters (Black formatter)
5. **Imports:** Standard library, third-party, local (separated by blank lines)
6. **Error messages:** Clear, specific, actionable

## Performance Tips

1. **Type hints:** No runtime cost, only for static analysis
2. **Decorators:** Small overhead (~5%), acceptable for most cases
3. **map/filter:** 10-20% faster than equivalent loops
4. **Caching:** Can provide 100-1000x speedup for repeated calls
5. **Validation:** Do it early (fail fast principle)

## Next Steps

After completing this exercise:

1. **Exercise 04: File Handling** - Read/write ML data files
2. **Exercise 05: Decorators** - Advanced decorator patterns
3. **Exercise 06: Async Programming** - Asynchronous ML workflows
4. **Apply to real projects:**
   - Create your own utility libraries
   - Build decorator collections for common patterns
   - Practice functional programming in data pipelines

## Resources

- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 3107 - Function Annotations](https://peps.python.org/pep-3107/)
- [Python Decorators Primer](https://realpython.com/primer-on-python-decorators/)
- [Python Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)
- [pytest Documentation](https://docs.pytest.org/)

---

**Last Updated:** 2025-10-30
