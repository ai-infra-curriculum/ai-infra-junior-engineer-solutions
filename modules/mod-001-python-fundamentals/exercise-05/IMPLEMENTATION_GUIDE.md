# Implementation Guide - Exercise 05: Error Handling

Step-by-step guide for implementing robust error handling for ML workflows.

## Prerequisites

- Python 3.11+ installed
- Completed Exercises 01-04
- Understanding of ML workflow failure points
- Basic knowledge of logging

## Time Estimate

90-120 minutes total

## Implementation Steps

### Part 1: Exception Basics (20 minutes)

**Step 1: Create exception_basics.py**
- Implement `demonstrate_exceptions()` showcasing common types
- Cover: ValueError, TypeError, KeyError, IndexError
- Cover: FileNotFoundError, ZeroDivisionError, AttributeError
- Run: `python scripts/exception_basics.py`

**Key concepts:**
- Exception types and when they occur
- Reading exception messages
- Understanding stack traces
- ML-specific error scenarios

### Part 2: Exception Handling Patterns (25 minutes)

**Step 2: Create exception_handling.py**
- Implement `safe_divide()` with try-except-else-finally
- Implement `load_model_with_fallback()` for nested try-except
- Implement `process_batch_safe()` for batch error handling
- Set up logging configuration
- Run: `python scripts/exception_handling.py`

**Key concepts:**
- try: Code that might raise exception
- except: Handle specific exceptions
- else: Runs if no exception occurred
- finally: Always runs (cleanup)
- Multiple except blocks for different errors

### Part 3: Custom Exceptions (20 minutes)

**Step 3: Create custom_exceptions.py**
- Create base `MLException` class
- Implement `ModelNotFoundError` with model path
- Implement `InvalidDataError` with data info
- Implement `GPUOutOfMemoryError` with batch/model size
- Implement `TrainingFailedError` with epoch and reason
- Implement `ConfigurationError` with param validation
- Create validation functions that raise custom exceptions
- Run: `python scripts/custom_exceptions.py`

**Key concepts:**
- Inheriting from Exception or custom base
- Adding attributes to exceptions
- Custom __init__ methods
- Exception hierarchies
- When to create custom vs. use built-ins

### Part 4: Retry Logic (30 minutes)

**Step 4: Create retry_logic.py**
- Implement `retry_with_backoff()` decorator
- Handle exponential backoff calculation
- Add configurable exception types
- Implement `download_model()` with retry decorator
- Implement `load_from_storage()` with retry
- Create `ResilientDataLoader` class with fallback sources
- Implement `load_data()` with source iteration
- Implement `_load_from_source()` with retry
- Run: `python scripts/retry_logic.py`

**Key concepts:**
- Decorator pattern for retry logic
- Exponential backoff: delay *= backoff_factor
- Transient vs. permanent failures
- Fallback to alternative sources
- @functools.wraps to preserve metadata
- Logging retry attempts

### Part 5: Context Managers (25 minutes)

**Step 5: Create context_managers.py**
- Implement `GPUContext` class
  - `__init__`: Store device ID
  - `__enter__`: Allocate GPU resources
  - `__exit__`: Release GPU, handle errors
- Implement `ModelCheckpoint` class
  - `__enter__`: Start checkpoint operation
  - `__exit__`: Finalize or cleanup on error
- Implement `TimerContext` class
  - `__enter__`: Record start time
  - `__exit__`: Calculate and log elapsed time
- Create `train_with_context()` demonstration
- Run: `python scripts/context_managers.py`

**Key concepts:**
- `__enter__` method runs at `with` start
- `__exit__(exc_type, exc_val, exc_tb)` always runs
- Return True from __exit__ to suppress exception
- Return False to propagate exception
- Nested context managers
- Resource cleanup guarantees

### Part 6: Robust ML Pipeline (35 minutes)

**Step 6: Create ml_pipeline_robust.py**
- Create `PipelineStatus` enum (SUCCESS, PARTIAL_FAILURE, etc.)
- Create `PipelineResult` dataclass
- Implement `RobustMLPipeline` class
  - `__init__`: Store config, initialize error lists
  - `run()`: Execute pipeline with error handling
  - `_load_data_safe()`: Load with error handling
  - `_validate_data_safe()`: Validate with error handling
  - `_preprocess_safe()`: Preprocess with error handling
  - `_train_safe()`: Train with error handling
  - `_evaluate_safe()`: Evaluate with error handling
  - `_create_failure_result()`: Build failure result
- Set up comprehensive logging
- Run: `python scripts/ml_pipeline_robust.py`

**Key concepts:**
- Pipeline stages with independent error handling
- Collecting errors and warnings throughout execution
- Structured results with metadata
- Graceful degradation (continue on warnings)
- Fail-fast on critical errors
- Comprehensive logging at each stage

### Part 7: Validation (15 minutes)

**Step 7: Create validate_error_handling.py**
- Test basic exception handling
- Test custom exceptions work correctly
- Test retry logic functions
- Test context managers
- Test pipeline error handling
- Run: `python scripts/validate_error_handling.py`

### Part 8: Testing (20 minutes)

**Step 8: Create pytest tests**
- Create `tests/test_custom_exceptions.py`
  - Test each custom exception type
  - Test exception attributes
  - Test exception messages
- Create `tests/test_retry_logic.py`
  - Test retry decorator functionality
  - Test backoff calculation
  - Test max retries respected
  - Test exception propagation
- Create `tests/test_pipeline.py`
  - Test pipeline success case
  - Test pipeline failure handling
  - Test partial failures
  - Test result structure

**Run tests:**
```bash
pytest tests/ -v
pytest tests/ --cov=scripts --cov-report=term-missing
```

## Quick Validation

```bash
# Run all implementation scripts
python scripts/exception_basics.py
python scripts/exception_handling.py
python scripts/custom_exceptions.py
python scripts/retry_logic.py
python scripts/context_managers.py
python scripts/ml_pipeline_robust.py

# Validate everything
python scripts/validate_error_handling.py

# Run pytest
pytest tests/ -v
```

## Key Concepts Checklist

- [ ] Understanding common exception types
- [ ] try-except-else-finally blocks
- [ ] Catching specific vs. broad exceptions
- [ ] Multiple except blocks
- [ ] Custom exception classes with attributes
- [ ] Exception inheritance hierarchies
- [ ] Retry decorator with exponential backoff
- [ ] Transient vs. permanent failure handling
- [ ] Context manager __enter__ and __exit__
- [ ] Resource cleanup in __exit__
- [ ] Multi-stage pipeline error handling
- [ ] Error and warning collection
- [ ] Structured error results
- [ ] Comprehensive logging
- [ ] Testing error handling code

## Common Issues

**Issue:** Exception not caught
**Solution:** Catch specific exceptions before general ones:
```python
try:
    operation()
except ValueError:  # Specific first
    handle_value_error()
except Exception:   # General last
    handle_general()
```

**Issue:** Losing exception context
**Solution:** Use `raise` without arguments to preserve traceback:
```python
try:
    risky_operation()
except ValueError as e:
    logger.error(f"Operation failed: {e}")
    raise  # Preserves original traceback
```

**Issue:** Retry logic retrying permanent failures
**Solution:** Only retry specific transient exceptions:
```python
@retry_with_backoff(exceptions=(ConnectionError, TimeoutError))
def download():
    # Only retries connection/timeout errors
    pass
```

**Issue:** Context manager not cleaning up
**Solution:** Ensure __exit__ always runs cleanup:
```python
def __exit__(self, exc_type, exc_val, exc_tb):
    self.cleanup()  # Always cleanup
    return False     # Propagate exceptions
```

**Issue:** Silent failures
**Solution:** Always log errors before handling:
```python
try:
    operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    # Then handle or re-raise
```

## Best Practices

1. **Catch specific exceptions:**
   ```python
   try:
       value = int(input)
   except ValueError:  # Specific
       handle_invalid_input()
   ```

2. **Always log errors:**
   ```python
   except FileNotFoundError as e:
       logger.error(f"File not found: {e}")
   ```

3. **Use finally for cleanup:**
   ```python
   try:
       f = open(file)
       process(f)
   finally:
       f.close()
   ```

4. **Don't swallow exceptions:**
   ```python
   except Exception as e:
       logger.error(f"Error: {e}")
       raise  # Don't hide the error
   ```

5. **Create custom exceptions for domain logic:**
   ```python
   class DataValidationError(Exception):
       pass

   if not valid:
       raise DataValidationError("Invalid data")
   ```

6. **Use context managers for resources:**
   ```python
   with open(file) as f:
       data = f.read()
   # Automatic cleanup
   ```

7. **Retry only transient failures:**
   ```python
   @retry_with_backoff(exceptions=(ConnectionError,))
   def download():
       pass
   ```

8. **Provide actionable error messages:**
   ```python
   raise ValueError(
       f"batch_size must be positive, got {batch_size}"
   )
   ```

## Error Handling Strategies

### Strategy 1: Fail Fast
```python
def process(data, config):
    # Validate inputs immediately
    if not data:
        raise ValueError("Empty data")

    if config["batch_size"] <= 0:
        raise ValueError("Invalid batch_size")

    # Continue processing...
```

### Strategy 2: Graceful Degradation
```python
def load_model(primary_path, backup_path):
    try:
        return load_from(primary_path)
    except FileNotFoundError:
        logger.warning("Using backup model")
        return load_from(backup_path)
```

### Strategy 3: Retry with Backoff
```python
@retry_with_backoff(max_retries=3)
def download_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
```

### Strategy 4: Collect and Report
```python
errors = []
for item in batch:
    try:
        process(item)
    except ValueError as e:
        errors.append(f"Item {item}: {e}")

if errors:
    logger.error(f"Processing errors: {errors}")
```

## Performance Considerations

1. **Exception handling overhead:** ~10-50x slower than normal flow
   - Only use for exceptional cases, not control flow
   - Don't use exceptions for expected conditions

2. **Retry logic:** Can significantly increase latency
   - Set reasonable max_retries (3-5 typical)
   - Use exponential backoff to prevent overwhelming services

3. **Logging:** Can be expensive with many errors
   - Use appropriate log levels
   - Consider sampling in high-throughput scenarios

4. **Context managers:** Minimal overhead
   - Always prefer over manual cleanup

## Next Steps

After completing this exercise:

1. **Exercise 06: Async Programming** - Error handling in async code
2. **Exercise 07: Testing** - Testing error scenarios
3. **Apply to projects:**
   - Build production ML pipelines with error handling
   - Implement monitoring and alerting
   - Create fault-tolerant data processing

## Resources

- [Python Exceptions Tutorial](https://docs.python.org/3/tutorial/errors.html)
- [Exception Hierarchy](https://docs.python.org/3/library/exceptions.html)
- [Logging Best Practices](https://docs.python.org/3/howto/logging.html)
- [Context Managers](https://docs.python.org/3/reference/datamodel.html#context-managers)
- [Tenacity Library](https://github.com/jd/tenacity)

---

**Last Updated:** 2025-10-30
