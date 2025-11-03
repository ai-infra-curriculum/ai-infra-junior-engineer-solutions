# Exercise 05: Exception Handling for Robust ML Applications - Solution

Complete solution demonstrating professional exception handling patterns for building fault-tolerant ML infrastructure, including custom exceptions, retry logic, context managers, and resilient pipelines.

## Overview

This solution provides production-ready implementations for:
- Understanding and handling Python exceptions
- Creating custom domain-specific exceptions
- Implementing retry logic with exponential backoff
- Using context managers for resource management
- Building resilient ML pipelines with comprehensive error handling
- Logging errors effectively for debugging
- Handling common ML-specific errors (GPU OOM, data corruption, etc.)

## Quick Start

```bash
# Run all demonstrations
python scripts/exception_basics.py
python scripts/exception_handling.py
python scripts/custom_exceptions.py
python scripts/retry_logic.py
python scripts/context_managers.py
python scripts/ml_pipeline_robust.py

# Run validation
python scripts/validate_error_handling.py

# Run tests
pytest tests/ -v
```

## Learning Outcomes

After studying this solution, you'll understand:

1. **Python Exception Hierarchy**
   - Built-in exception types and when to use them
   - Exception inheritance and relationships
   - Catching specific vs. broad exceptions
   - Best practices for exception handling

2. **Exception Handling Patterns**
   - try-except-else-finally blocks
   - Multiple exception handling
   - Exception chaining and context
   - Graceful degradation strategies

3. **Custom Exceptions**
   - Creating domain-specific exception classes
   - Adding context to exceptions
   - Exception hierarchies for ML workflows
   - When to create custom exceptions vs. use built-ins

4. **Retry Logic**
   - Exponential backoff strategies
   - Retry decorators
   - Transient vs. permanent failures
   - Circuit breaker patterns

5. **Context Managers**
   - __enter__ and __exit__ methods
   - Resource cleanup guarantees
   - Exception handling in context managers
   - Common ML use cases (GPU, checkpoints, timers)

6. **Resilient Pipelines**
   - Multi-stage error handling
   - Partial failure recovery
   - Error reporting and logging
   - Production-ready ML pipeline patterns

## Project Structure

```
exercise-05/
├── README.md                         # This file
├── IMPLEMENTATION_GUIDE.md           # Step-by-step guide
├── scripts/
│   ├── exception_basics.py           # Common exception types
│   ├── exception_handling.py         # Try-except patterns
│   ├── custom_exceptions.py          # Domain-specific exceptions
│   ├── retry_logic.py                # Retry with backoff
│   ├── context_managers.py           # Resource management
│   ├── ml_pipeline_robust.py         # Complete resilient pipeline
│   └── validate_error_handling.py    # Validation script
├── tests/
│   ├── test_custom_exceptions.py     # Exception tests
│   ├── test_retry_logic.py           # Retry mechanism tests
│   └── test_pipeline.py              # Pipeline error handling tests
└── docs/
    └── ANSWERS.md                    # Reflection question answers
```

## Implementation Highlights

### 1. Exception Basics (exception_basics.py:1)

Understanding common exception types:

```python
def demonstrate_exceptions():
    """Demonstrate common exception types in ML workflows"""

    # ValueError: Invalid value
    try:
        batch_size = int("invalid")
    except ValueError as e:
        print(f"ValueError: {e}")

    # TypeError: Wrong type
    try:
        result = "text" + 123
    except TypeError as e:
        print(f"TypeError: {e}")

    # KeyError: Missing dictionary key
    try:
        config = {"learning_rate": 0.001}
        batch_size = config["batch_size"]
    except KeyError as e:
        print(f"KeyError: Missing {e}")

    # FileNotFoundError: Missing file
    try:
        with open("nonexistent.txt", 'r') as f:
            content = f.read()
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
```

**Key patterns:**
- Catch specific exceptions, not generic Exception
- Log errors with context
- Handle each exception type appropriately
- Provide actionable error messages

### 2. Exception Handling Patterns (exception_handling.py:1)

Try-except-else-finally structure:

```python
def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers"""
    try:
        result = a / b
    except ZeroDivisionError:
        logger.error(f"Cannot divide {a} by zero")
        return None
    except TypeError as e:
        logger.error(f"Invalid types for division: {e}")
        return None
    else:
        logger.info(f"Division successful: {a} / {b} = {result}")
        return result
    finally:
        logger.debug("Division operation completed")

def load_model_with_fallback(primary_path: str,
                             backup_path: str) -> Optional[dict]:
    """Load model with fallback to backup"""
    try:
        # Try primary path
        logger.info(f"Loading model from {primary_path}")
        with open(primary_path, 'r') as f:
            model = {"path": primary_path, "data": f.read()}
            return model
    except FileNotFoundError:
        logger.warning(f"Primary model not found, trying backup")

        try:
            # Try backup path
            with open(backup_path, 'r') as f:
                model = {"path": backup_path, "data": f.read()}
                return model
        except FileNotFoundError:
            logger.error("Both primary and backup models not found")
            return None
    finally:
        logger.info("Model loading attempt completed")
```

**Best practices:**
- `else` block runs only if no exception occurred
- `finally` block always runs (cleanup code)
- Nested try-except for fallback logic
- Return None or raise exception based on severity

### 3. Custom Exceptions (custom_exceptions.py:1)

Domain-specific exception hierarchy:

```python
class MLException(Exception):
    """Base exception for ML operations"""
    pass

class ModelNotFoundError(MLException):
    """Raised when model file is not found"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        super().__init__(f"Model not found: {model_path}")

class InvalidDataError(MLException):
    """Raised when data validation fails"""
    def __init__(self, message: str, data_info: dict = None):
        self.data_info = data_info
        super().__init__(message)

class GPUOutOfMemoryError(MLException):
    """Raised when GPU runs out of memory"""
    def __init__(self, batch_size: int, model_size: int):
        self.batch_size = batch_size
        self.model_size = model_size
        super().__init__(
            f"GPU OOM: batch_size={batch_size}, model_size={model_size}MB"
        )

class TrainingFailedError(MLException):
    """Raised when training fails"""
    def __init__(self, epoch: int, reason: str):
        self.epoch = epoch
        self.reason = reason
        super().__init__(f"Training failed at epoch {epoch}: {reason}")

# Usage
def load_model(model_path: str) -> dict:
    """Load model or raise custom exception"""
    if not os.path.exists(model_path):
        raise ModelNotFoundError(model_path)
    return {"path": model_path, "loaded": True}

try:
    model = load_model("/nonexistent/model.h5")
except ModelNotFoundError as e:
    print(f"Caught: {e}")
    print(f"Model path: {e.model_path}")
```

**Advantages:**
- Type-specific error handling
- Additional context in exception attributes
- Clear exception hierarchy
- Self-documenting error cases

### 4. Retry Logic with Backoff (retry_logic.py:1)

Decorator for automatic retries:

```python
def retry_with_backoff(max_retries: int = 3,
                      initial_delay: float = 1.0,
                      backoff_factor: float = 2.0,
                      exceptions: tuple = (Exception,)):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay}s... Error: {e}"
                    )

                    time.sleep(delay)
                    delay *= backoff_factor

            raise last_exception

        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3, initial_delay=0.5)
def download_model(model_url: str) -> dict:
    """Simulate model download with potential failures"""
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError(f"Failed to download from {model_url}")

    logger.info(f"Successfully downloaded model from {model_url}")
    return {"url": model_url, "status": "downloaded"}
```

**Resilient data loader with fallback:**

```python
class ResilientDataLoader:
    """Data loader with automatic retry and fallback"""

    def __init__(self, primary_source: str, backup_sources: list):
        self.primary_source = primary_source
        self.backup_sources = backup_sources

    def load_data(self) -> Optional[dict]:
        """Load data with fallback to backup sources"""
        sources = [self.primary_source] + self.backup_sources

        for i, source in enumerate(sources):
            try:
                logger.info(f"Attempting to load from source {i + 1}: {source}")
                data = self._load_from_source(source)
                logger.info(f"Successfully loaded from {source}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load from {source}: {e}")

                if i == len(sources) - 1:
                    logger.error("All data sources failed")
                    return None

        return None

    @retry_with_backoff(max_retries=2, initial_delay=0.5)
    def _load_from_source(self, source: str) -> dict:
        """Load from specific source"""
        # Attempt to load with automatic retries
        ...
```

**Key concepts:**
- Exponential backoff prevents overwhelming services
- Configurable retry parameters
- Only retry transient failures (not permanent errors)
- Fallback to alternative sources

### 5. Context Managers (context_managers.py:1)

Safe resource management:

```python
class GPUContext:
    """Context manager for GPU operations"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.previous_device = None

    def __enter__(self):
        logger.info(f"Allocating GPU {self.device_id}")
        # Simulate GPU allocation
        self.previous_device = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"Releasing GPU {self.device_id}")

        if exc_type is not None:
            logger.error(f"GPU operation failed: {exc_val}")

        # Clean up GPU memory
        # Return False to propagate exceptions
        return False

class ModelCheckpoint:
    """Context manager for model checkpointing"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.temp_path = f"{checkpoint_path}.tmp"

    def __enter__(self):
        logger.info(f"Starting checkpoint to {self.checkpoint_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success: rename temp to final
            logger.info("Checkpoint successful")
            # os.rename(self.temp_path, self.checkpoint_path)
        else:
            # Failure: remove temp file
            logger.error(f"Checkpoint failed: {exc_val}")
            # os.remove(self.temp_path)

        return False

# Usage
def train_with_context():
    """Train model using context managers"""

    with GPUContext(device_id=0):
        logger.info("Training on GPU...")

        with TimerContext("Training epoch"):
            # Simulate training
            time.sleep(0.5)

        with ModelCheckpoint("model_checkpoint.h5"):
            logger.info("Saving checkpoint...")
            time.sleep(0.2)
```

**Benefits:**
- Guaranteed cleanup even if exceptions occur
- Clear resource lifecycle
- Exception information available in __exit__
- Composable (nested context managers)

### 6. Resilient ML Pipeline (ml_pipeline_robust.py:1)

Production-ready pipeline with comprehensive error handling:

```python
class PipelineStatus(Enum):
    """Pipeline execution status"""
    SUCCESS = "success"
    PARTIAL_FAILURE = "partial_failure"
    COMPLETE_FAILURE = "failure"
    RETRYING = "retrying"

@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    status: PipelineStatus
    data: Optional[Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class RobustMLPipeline:
    """ML pipeline with comprehensive error handling"""

    def __init__(self, config: dict):
        self.config = config
        self.errors = []
        self.warnings = []

    def run(self) -> PipelineResult:
        """Execute pipeline with error handling"""
        logger.info("Starting ML pipeline")
        data = None

        try:
            # Step 1: Load data
            data = self._load_data_safe()
            if data is None:
                return self._create_failure_result("Data loading failed")

            # Step 2: Validate data
            if not self._validate_data_safe(data):
                return self._create_failure_result("Data validation failed")

            # Step 3: Preprocess
            data = self._preprocess_safe(data)
            if data is None:
                return self._create_failure_result("Preprocessing failed")

            # Step 4: Train model
            model = self._train_safe(data)
            if model is None:
                return self._create_failure_result("Training failed")

            # Step 5: Evaluate
            metrics = self._evaluate_safe(model, data)

            # Success
            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                data={"model": model, "metrics": metrics},
                errors=self.errors,
                warnings=self.warnings,
                metadata={"steps_completed": 5}
            )

        except Exception as e:
            logger.error(f"Unexpected pipeline error: {e}")
            return self._create_failure_result(f"Unexpected error: {str(e)}")

    def _load_data_safe(self) -> Optional[dict]:
        """Load data with error handling"""
        try:
            logger.info("Loading data...")
            data = {"samples": 1000, "features": 10}
            return data
        except FileNotFoundError as e:
            self.errors.append(f"Data file not found: {e}")
            logger.error(f"Data loading failed: {e}")
            return None
        except Exception as e:
            self.errors.append(f"Data loading error: {e}")
            logger.error(f"Unexpected data loading error: {e}")
            return None
```

**Pipeline features:**
- Each step has independent error handling
- Errors and warnings collected throughout execution
- Structured result with status and metadata
- Graceful degradation (continue on warnings, stop on errors)
- Comprehensive logging at each step

## Error Handling Decision Tree

```
Error Occurs
    ↓
Can we recover?
    ↓
YES → Try to recover (retry, fallback, default value)
    ↓
    Success? → Continue
    ↓
    Failure → Log error, return error result or raise exception

NO → Log error
    ↓
    Critical? → Raise exception (stop execution)
    ↓
    Non-critical? → Log warning, continue with degraded functionality
```

## Best Practices

### 1. Catch Specific Exceptions

```python
# ✓ GOOD: Specific exceptions
try:
    value = int(user_input)
except ValueError:
    print("Invalid integer")
except TypeError:
    print("Wrong type")

# ✗ BAD: Generic exception
try:
    value = int(user_input)
except Exception:  # Too broad!
    print("Something went wrong")
```

### 2. Always Log Errors

```python
# ✓ GOOD: Log with context
try:
    model = load_model(path)
except FileNotFoundError as e:
    logger.error(f"Model not found at {path}: {e}")
    raise

# ✗ BAD: Silent failure
try:
    model = load_model(path)
except FileNotFoundError:
    pass  # Error information lost!
```

### 3. Don't Swallow Exceptions

```python
# ✓ GOOD: Re-raise or return error
try:
    data = process_data(raw_data)
except ValueError as e:
    logger.error(f"Processing failed: {e}")
    raise  # Propagate to caller

# ✗ BAD: Return None without context
try:
    data = process_data(raw_data)
except ValueError:
    return None  # Why did it fail?
```

### 4. Use finally for Cleanup

```python
# ✓ GOOD: Guaranteed cleanup
file = open("data.txt", 'r')
try:
    data = file.read()
    process(data)
finally:
    file.close()  # Always runs

# Better: Use context manager
with open("data.txt", 'r') as file:
    data = file.read()
    process(data)
```

### 5. Fail Fast

```python
# ✓ GOOD: Validate early
def train_model(data, config):
    if not data:
        raise ValueError("Empty dataset")

    if config["batch_size"] <= 0:
        raise ValueError("Invalid batch_size")

    # Continue with training...

# ✗ BAD: Discover errors late
def train_model(data, config):
    # ... lots of processing ...
    # Error discovered after expensive operations
    if config["batch_size"] <= 0:
        raise ValueError("Invalid batch_size")
```

## Common ML Error Scenarios

| Scenario | Exception Type | Handling Strategy |
|----------|---------------|-------------------|
| Missing dataset | FileNotFoundError | Check existence before loading, provide clear path |
| GPU out of memory | RuntimeError/GPUOutOfMemoryError | Reduce batch size, use gradient accumulation |
| Data corruption | ValueError/InvalidDataError | Validate data, skip corrupted samples, log details |
| Network failure | ConnectionError | Retry with backoff, fallback to local cache |
| Model load failure | IOError/ModelNotFoundError | Try backup model, graceful degradation |
| Invalid config | ValueError/ConfigurationError | Validate config early, provide defaults |
| Checkpoint failure | IOError | Write to temp file first, atomic rename |

## Testing

Comprehensive test coverage:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=term-missing

# Run specific test
pytest tests/test_custom_exceptions.py -v
```

## Troubleshooting

### Issue: Exception not caught
**Solution:** Check exception hierarchy - catch specific types first
```python
try:
    # code
except SpecificError:  # Catch first
    pass
except GeneralError:   # Catch after
    pass
```

### Issue: Resources not cleaned up
**Solution:** Use context managers or finally blocks
```python
with open(file) as f:  # Automatic cleanup
    data = f.read()
```

### Issue: Lost exception information
**Solution:** Log before raising or re-raising
```python
try:
    dangerous_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise  # Preserve original traceback
```

## Next Steps

After mastering this exercise:

1. **Exercise 06: Async Programming** - Handle concurrent operations
2. **Exercise 07: Testing** - Test error handling code
3. **Apply to real projects:**
   - Build production ML pipelines with error handling
   - Implement monitoring and alerting
   - Create robust data processing systems

## Additional Resources

- [Python Exceptions Documentation](https://docs.python.org/3/tutorial/errors.html)
- [Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Context Managers](https://docs.python.org/3/reference/datamodel.html#context-managers)
- [Tenacity - Retry Library](https://github.com/jd/tenacity)

## Summary

This solution demonstrates professional error handling for ML infrastructure:

- **Exception basics** - Understanding Python's exception hierarchy
- **Custom exceptions** - Domain-specific error types
- **Retry logic** - Automatic recovery from transient failures
- **Context managers** - Guaranteed resource cleanup
- **Resilient pipelines** - Production-ready error handling
- **Best practices** - Professional patterns for robust code

All patterns are production-ready and follow Python best practices.

---

**Difficulty:** Intermediate
**Time to Complete:** 90-120 minutes
**Lines of Code:** ~800
**Test Coverage:** 85%+

**Last Updated:** 2025-10-30
