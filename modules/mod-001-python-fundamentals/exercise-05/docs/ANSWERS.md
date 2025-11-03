# Reflection Questions - Exercise 05: Error Handling

Thoughtful answers to the reflection questions about exception handling in ML workflows.

## 1. When should you catch broad vs. specific exceptions?

**Catch specific exceptions when:**
- You know exactly what can go wrong and how to handle it
- Different error types require different handling strategies
- You want clear, maintainable error handling code

```python
# ✓ GOOD: Specific exceptions with targeted handling
try:
    config = load_config(path)
except FileNotFoundError:
    # Use default configuration
    config = get_default_config()
except json.JSONDecodeError as e:
    # Invalid format - this is a fatal error
    logger.error(f"Invalid config format: {e}")
    raise
except PermissionError:
    # Security issue - alert admin
    alert_admin("Config access denied")
    raise
```

**Catch broad exceptions (Exception) when:**
- At the top level of an application for catch-all error handling
- When logging all errors in a consistent format
- As a last resort after specific exception handlers
- When wrapping external code whose exceptions you don't control

```python
# ✓ ACCEPTABLE: Broad exception at top level
def main():
    try:
        run_ml_pipeline()
    except SpecificError as e:
        # Handle known errors
        handle_specific_error(e)
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error: {e}", exc_info=True)
        alert_on_call()
        raise
```

**Never catch BaseException** - it includes SystemExit, KeyboardInterrupt, which should almost never be caught.

**Best practice:** Order exception handlers from most specific to most general:

```python
try:
    risky_operation()
except FileNotFoundError:      # Most specific
    handle_missing_file()
except IOError:                # More general
    handle_io_error()
except Exception:              # Most general (last resort)
    handle_unknown_error()
```

## 2. How do custom exceptions improve code maintainability?

**Custom exceptions improve maintainability in several ways:**

### 1. **Type-Specific Handling**
```python
try:
    train_model(data)
except GPUOutOfMemoryError as e:
    # Reduce batch size and retry
    reduce_batch_size()
    retry_training()
except DataValidationError as e:
    # Clean data and retry
    clean_data(e.data_info)
    retry_training()
except ModelNotFoundError as e:
    # Download model from backup
    download_model(e.model_path)
```

Without custom exceptions, you'd have to parse error messages:
```python
# ✗ BAD: Parsing error messages
try:
    train_model(data)
except RuntimeError as e:
    if "GPU" in str(e) and "memory" in str(e):
        # Handle GPU OOM
    elif "data" in str(e) and "invalid" in str(e):
        # Handle data error
    # Fragile and error-prone!
```

### 2. **Rich Context**
Custom exceptions can carry structured data:
```python
class TrainingFailedError(MLException):
    def __init__(self, epoch, metrics, reason):
        self.epoch = epoch
        self.metrics = metrics
        self.reason = reason

# Usage
try:
    train()
except TrainingFailedError as e:
    logger.error(f"Training failed at epoch {e.epoch}")
    logger.error(f"Last metrics: {e.metrics}")
    # Can make informed decisions based on context
    if e.metrics['loss'] == float('nan'):
        reduce_learning_rate()
```

### 3. **Clear Exception Hierarchies**
```python
MLException
├── DataError
│   ├── DataValidationError
│   ├── DataCorruptionError
│   └── InsufficientDataError
├── ModelError
│   ├── ModelNotFoundError
│   ├── ModelLoadError
│   └── ModelArchitectureError
└── InfrastructureError
    ├── GPUOutOfMemoryError
    ├── StorageError
    └── NetworkError

# Can catch at different levels
try:
    pipeline.run()
except DataError:
    # Handle all data-related errors similarly
    fallback_to_cached_data()
except MLException:
    # Handle all ML errors
    alert_team()
```

### 4. **Self-Documenting Code**
```python
# ✓ GOOD: Clear what can go wrong
def load_model(path: str) -> Model:
    \"\"\"
    Raises:
        ModelNotFoundError: If model file doesn't exist
        ModelCorruptionError: If model file is corrupted
        IncompatibleVersionError: If model version doesn't match
    \"\"\"
```

### 5. **Easier Testing**
```python
# Easy to test specific error conditions
def test_model_not_found():
    with pytest.raises(ModelNotFoundError):
        load_model("/nonexistent/model.h5")

# Better than:
def test_model_not_found():
    with pytest.raises(Exception) as e:
        load_model("/nonexistent/model.h5")
    assert "not found" in str(e.value)  # Fragile!
```

## 3. What retry strategies work best for different failure types?

### **Transient Failures → Exponential Backoff**
Best for: Network errors, API rate limits, temporary resource unavailability

```python
@retry_with_backoff(
    max_retries=5,
    initial_delay=1.0,
    backoff_factor=2.0,  # 1s, 2s, 4s, 8s, 16s
    exceptions=(ConnectionError, TimeoutError)
)
def download_dataset(url):
    return requests.get(url)
```

**Why:** Exponential backoff prevents overwhelming a recovering service and gives time for transient issues to resolve.

### **Rate Limiting → Fixed Delay with Jitter**
Best for: API rate limits

```python
@retry_with_backoff(
    max_retries=10,
    initial_delay=60.0,  # API rate limit window
    backoff_factor=1.0,  # Fixed delay
    jitter=5.0           # Add randomness
)
def call_rate_limited_api(endpoint):
    return api.call(endpoint)
```

**Why:** Fixed delay respects rate limit windows, jitter prevents thundering herd.

### **GPU OOM → Reduce Batch Size**
Best for: Memory errors

```python
def train_with_oom_handling(data, initial_batch_size=128):
    batch_size = initial_batch_size
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            return train(data, batch_size=batch_size)
        except GPUOutOfMemoryError:
            batch_size //= 2
            if batch_size < 1:
                raise
            logger.warning(f"GPU OOM, reducing batch_size to {batch_size}")
```

**Why:** Retrying with same parameters will fail again. Must adapt strategy.

### **Data Corruption → Skip and Continue**
Best for: Individual sample errors in large datasets

```python
def process_dataset(samples):
    results = []
    errors = []

    for i, sample in enumerate(samples):
        try:
            result = process_sample(sample)
            results.append(result)
        except DataCorruptionError as e:
            errors.append(f"Sample {i}: {e}")
            continue  # Skip corrupted sample

    if len(errors) / len(samples) > 0.1:  # More than 10% corrupted
        raise TooManyCorruptedSamplesError(errors)

    return results
```

**Why:** One bad sample shouldn't stop entire dataset processing.

### **Permanent Failures → Fail Fast**
Best for: Configuration errors, missing required files, authentication failures

```python
# ✗ DON'T RETRY
try:
    authenticate(credentials)
except AuthenticationError:
    # Retrying won't help - credentials are wrong
    raise

# ✗ DON'T RETRY
try:
    config = load_config("required_config.json")
except FileNotFoundError:
    # Retrying won't make the file appear
    raise
```

**Why:** Retrying permanent failures wastes time and resources.

### **Cascading Failures → Circuit Breaker**
Best for: Protecting against cascading failures

```python
class CircuitBreaker:
    # After 5 failures, stop trying for 60 seconds
    # Prevents cascading failures across services

    def call(self, func):
        if self.state == OPEN:
            raise CircuitBreakerOpenError()
        # Try call, update state
```

**Why:** Gives failing service time to recover, prevents cascading failures.

## 4. How do context managers ensure proper resource cleanup?

**Context managers guarantee cleanup through the `__exit__` method, which ALWAYS runs:**

### 1. **Normal Execution**
```python
with open('file.txt') as f:
    data = f.read()
# __exit__ called, file closed
```

### 2. **With Exception**
```python
try:
    with open('file.txt') as f:
        raise ValueError("Error during processing")
except ValueError:
    pass
# __exit__ STILL called, file STILL closed
```

### 3. **With Return**
```python
def process_file():
    with open('file.txt') as f:
        data = f.read()
        if data:
            return data  # Early return
    # __exit__ called before return!
```

### 4. **Even with SystemExit**
```python
with open('file.txt') as f:
    sys.exit(1)  # Emergency exit
# __exit__ STILL called!
```

**Why it works:**

```python
class ResourceManager:
    def __enter__(self):
        self.resource = acquire_resource()
        return self.resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ALWAYS called, regardless of how __enter__ exits
        release_resource(self.resource)

        # exc_type is None if no exception
        # exc_type, exc_val, exc_tb contain exception info if raised

        return False  # Propagate exception (True suppresses it)
```

**ML-specific examples:**

```python
# GPU cleanup
with GPUContext(device_id=0):
    train_model()  # GPU released even if training fails

# Checkpoint cleanup
with ModelCheckpoint("model.h5"):
    save_model()  # Temp files cleaned up on failure

# Lock management
with distributed.Lock("training_lock"):
    train_model()  # Lock released even if training fails

# Resource pooling
with ResourcePool(gpus=4):
    distributed_training()  # GPUs returned to pool
```

**Advantages over manual cleanup:**

```python
# ✗ MANUAL (error-prone)
f = open('file.txt')
try:
    data = f.read()
finally:
    f.close()  # Easy to forget!

# ✓ CONTEXT MANAGER (guaranteed)
with open('file.txt') as f:
    data = f.read()
# Automatically closed
```

## 5. What information should be logged when exceptions occur?

**Essential information for debugging and monitoring:**

### 1. **Exception Details**
```python
try:
    operation()
except Exception as e:
    logger.error(
        "Operation failed",
        exc_info=True,  # Include full traceback
        extra={
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "operation": "train_model"
        }
    )
```

### 2. **Context Information**
```python
try:
    train_model(data, config)
except TrainingFailedError as e:
    logger.error(
        f"Training failed at epoch {e.epoch}",
        extra={
            # What was being done
            "operation": "model_training",
            "model_type": config["model_type"],

            # State at failure
            "epoch": e.epoch,
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],

            # Metrics at failure
            "last_loss": e.metrics.get("loss"),
            "last_accuracy": e.metrics.get("accuracy"),

            # Environment
            "gpu_id": get_current_gpu(),
            "memory_available": get_available_memory(),

            # Identifiers for tracing
            "training_id": training_id,
            "user_id": user_id,
            "timestamp": time.time()
        }
    )
```

### 3. **Actionable Information**
```python
except GPUOutOfMemoryError as e:
    logger.error(
        "GPU OOM",
        extra={
            "error": "GPU out of memory",
            "batch_size": e.batch_size,
            "model_size_mb": e.model_size,
            "available_memory_mb": e.available_memory,

            # Actionable suggestions
            "suggested_batch_size": e.batch_size // 2,
            "suggested_action": "Reduce batch_size or model complexity",

            # For automated recovery
            "can_retry": True,
            "retry_with_params": {"batch_size": e.batch_size // 2}
        }
    )
```

### 4. **Severity Levels**
```python
# CRITICAL: Requires immediate attention
logger.critical("Model serving crashed, users affected")

# ERROR: Failed but system continues
logger.error("Training job failed, will retry")

# WARNING: Potential issue, might lead to errors
logger.warning("GPU memory usage at 95%")

# INFO: Normal operations
logger.info("Training started")

# DEBUG: Detailed diagnostic info
logger.debug(f"Batch {i} processed in {elapsed}s")
```

### 5. **Structured Logging for Analysis**
```python
import structlog

logger = structlog.get_logger()

try:
    result = train_model()
except Exception as e:
    logger.error(
        "training_failed",
        exception=type(e).__name__,
        message=str(e),
        model_id=model.id,
        dataset_size=len(dataset),
        hyperparameters=config,
        gpu_utilization=get_gpu_stats(),
        stack_trace=traceback.format_exc()
    )
    # Outputs JSON for easy parsing/analysis
```

### 6. **User-Facing vs. Internal Logs**
```python
try:
    process_user_request(request)
except Exception as e:
    # Internal: Full details
    logger.error(
        "Request processing failed",
        exc_info=True,
        extra={
            "user_id": request.user_id,
            "request_id": request.id,
            "full_stack": traceback.format_exc()
        }
    )

    # User-facing: Safe, actionable message
    return {
        "error": "Unable to process request",
        "message": "Please try again later",
        "request_id": request.id,  # For support
        "status": 500
    }
```

## 6. How do you balance error handling with code readability?

**Strategies for maintainable error handling:**

### 1. **Use Helper Functions**
```python
# ✗ CLUTTERED
def train_model(data):
    try:
        # 100 lines of training logic
        pass
    except GPUOutOfMemoryError as e:
        logger.error(f"GPU OOM: {e}")
        # 20 lines of recovery logic
    except DataValidationError as e:
        logger.error(f"Invalid data: {e}")
        # 20 lines of recovery logic
    except ModelLoadError as e:
        logger.error(f"Model load failed: {e}")
        # 20 lines of recovery logic

# ✓ CLEAN
def train_model(data):
    try:
        return _train_model_impl(data)
    except GPUOutOfMemoryError as e:
        return handle_gpu_oom(e, data)
    except DataValidationError as e:
        return handle_data_error(e, data)
    except ModelLoadError as e:
        return handle_model_error(e, data)
```

### 2. **Handle Errors at the Right Level**
```python
# ✗ TOO DEFENSIVE (noisy)
def calculate_accuracy(predictions, labels):
    try:
        if not predictions:
            raise ValueError("Empty predictions")
        if not labels:
            raise ValueError("Empty labels")
        if len(predictions) != len(labels):
            raise ValueError("Length mismatch")

        correct = sum(p == l for p, l in zip(predictions, labels))

        try:
            return correct / len(predictions)
        except ZeroDivisionError:
            return 0.0
    except Exception as e:
        logger.error(f"Accuracy calculation failed: {e}")
        return 0.0

# ✓ APPROPRIATE LEVEL
def calculate_accuracy(predictions, labels):
    \"\"\"Calculate accuracy.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Accuracy score

    Raises:
        ValueError: If inputs are invalid
    \"\"\"
    if not predictions or not labels:
        raise ValueError("Empty inputs")

    if len(predictions) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(predictions)} != {len(labels)}"
        )

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)
```

### 3. **Use Decorators for Cross-Cutting Concerns**
```python
# ✓ CLEAN
@log_errors
@retry_with_backoff(max_retries=3)
@timing_decorator
def download_model(url):
    return requests.get(url)

# Instead of mixing all concerns in one function
```

### 4. **Document Expected Exceptions**
```python
def load_model(path: str) -> Model:
    \"\"\"
    Load model from disk.

    Args:
        path: Path to model file

    Returns:
        Loaded model

    Raises:
        ModelNotFoundError: If file doesn't exist
        ModelCorruptionError: If file is corrupted
        IncompatibleVersionError: If version mismatch

    Example:
        try:
            model = load_model("model.h5")
        except ModelNotFoundError:
            model = download_model_from_backup()
    \"\"\"
```

### 5. **Fail Fast, Handle Once**
```python
# ✓ VALIDATE EARLY
def train_model(data, config):
    # Validate inputs immediately (fail fast)
    validate_data(data)
    validate_config(config)

    # Clean execution without nested try-except
    preprocessed = preprocess(data)
    model = build_model(config)
    trained = fit_model(model, preprocessed)

    return trained

# Handle at top level
def main():
    try:
        result = train_model(data, config)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
```

### 6. **Use Context Managers for Resource Management**
```python
# ✓ CLEAN with context managers
with GPUContext(0):
    with TimerContext("training"):
        with ModelCheckpoint("model.h5"):
            train()

# Instead of nested try-finally blocks
```

## 7. When should exceptions be raised vs. returned as error codes?

### **Use Exceptions When:**

#### 1. **Truly Exceptional Conditions**
```python
# ✓ EXCEPTION: File should always exist
def load_required_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required config missing: {path}")
    return json.load(open(path))
```

#### 2. **Cannot Continue Execution**
```python
# ✓ EXCEPTION: Cannot proceed without valid data
def train_model(data):
    if not validate_data(data):
        raise InvalidDataError("Data validation failed - cannot train")
```

#### 3. **Caller Should Handle It**
```python
# ✓ EXCEPTION: Caller decides what to do
def download_model(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ModelDownloadError(f"HTTP {response.status_code}")
    return response.content
```

### **Use Return Values When:**

#### 1. **Expected Alternative Outcomes**
```python
# ✓ RETURN: Finding nothing is expected
def find_model(model_id: str) -> Optional[Model]:
    model = db.query(Model).filter_by(id=model_id).first()
    return model  # None if not found is OK

# Usage
model = find_model("model_123")
if model is None:
    # Handle not found case
```

#### 2. **Performance-Critical Code**
```python
# ✓ RETURN: Exceptions are slow
def validate_sample(sample: dict) -> Tuple[bool, str]:
    if "features" not in sample:
        return False, "Missing features"
    if len(sample["features"]) != EXPECTED_SIZE:
        return False, "Wrong feature count"
    return True, ""

# Faster than raising/catching exceptions in tight loop
for sample in millions_of_samples:
    valid, error = validate_sample(sample)
    if not valid:
        continue
```

#### 3. **Multiple Outcomes Are Equally Valid**
```python
# ✓ RETURN: Both outcomes are valid
def check_cache(key: str) -> Tuple[bool, Optional[Any]]:
    if key in cache:
        return True, cache[key]  # Cache hit
    return False, None           # Cache miss

# ✗ DON'T USE EXCEPTION
def check_cache(key: str) -> Any:
    if key not in cache:
        raise CacheMissError()  # Too heavyweight
    return cache[key]
```

#### 4. **Status Reporting**
```python
# ✓ RETURN: Status with structured info
@dataclass
class ProcessingResult:
    status: Literal["success", "partial", "failed"]
    processed: int
    errors: List[str]
    data: Optional[Any]

def process_batch(items):
    # Return structured result, not exception
    return ProcessingResult(
        status="partial",
        processed=80,
        errors=["Item 5 corrupted", "Item 12 invalid"],
        data=processed_items
    )
```

### **Hybrid Approach (Best Practice)**

```python
# ✓ BEST: Return for expected cases, raise for exceptional
def load_model(path: str) -> Optional[Model]:
    \"\"\"
    Load model from path.

    Returns:
        Model if found, None if not found

    Raises:
        ModelCorruptionError: If file is corrupted
        PermissionError: If access denied
    \"\"\"
    if not os.path.exists(path):
        return None  # Expected case

    try:
        return pickle.load(open(path, 'rb'))
    except pickle.UnpicklingError:
        raise ModelCorruptionError(f"Corrupted: {path}")  # Exceptional
    except PermissionError:
        raise  # Exceptional, re-raise
```

### **Decision Tree**

```
Is the condition exceptional (rarely happens)?
├─ YES → Use exception
│   └─ Can execution continue?
│       ├─ NO → Raise exception
│       └─ YES → Consider return value
│
└─ NO (happens regularly) → Use return value
    └─ Are there multiple valid outcomes?
        ├─ YES → Return status/result object
        └─ NO → Consider exception
```

---

**Summary:**
- **Exceptions** = Exceptional conditions, cannot continue, caller must handle
- **Return values** = Expected alternatives, performance-critical, multiple valid outcomes
- **Hybrid** = Return None/Result for expected cases, raise for exceptional ones
