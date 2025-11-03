# Exercise 03: Reflection Question Answers

Comprehensive answers to all reflection questions about functions, modules, and functional programming.

## Question 1: When should you use type hints in production code?

### Answer: Always Use Type Hints in Production Code

Type hints should be used in **all production code** for the following reasons:

**Benefits:**

1. **IDE Support and Autocomplete**
   - IDEs can provide better code completion
   - Function signatures are self-documenting
   - Catches errors before runtime

2. **Static Type Checking**
   - Use mypy to catch type errors before deployment
   - Prevents common bugs like passing wrong types
   - Reduces need for runtime type checking

3. **Documentation**
   - Self-documents function interfaces
   - Makes code easier to understand
   - Reduces need for comments

4. **Maintainability**
   - Easier to refactor with confidence
   - Clear contracts between functions
   - Onboarding new developers is faster

**Example:**
```python
# ✗ BAD: No type hints
def train_model(data, epochs, lr):
    # What types are these? What does it return?
    pass

# ✓ GOOD: Clear type hints
def train_model(data: List[float],
                epochs: int,
                lr: float) -> Dict[str, float]:
    """
    Train ML model.

    Args:
        data: Training data
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        Dictionary with training metrics
    """
    pass
```

**When to Skip Type Hints:**
- Quick prototypes/experiments (but add them before committing)
- One-off scripts
- Internal helper functions where types are obvious

**Best Practices:**
- Use `Optional[T]` for nullable parameters
- Use `Union[T1, T2]` for multiple possible types
- Use `Any` sparingly (loses type safety)
- Use `from __future__ import annotations` for forward references

**Performance Note:** Type hints have **zero runtime cost** - they're only used by static analyzers.

---

## Question 2: How do decorators help make code more maintainable?

### Answer: Decorators Provide Cross-Cutting Concerns Without Code Duplication

Decorators improve maintainability by:

**1. Separation of Concerns**

Instead of mixing logging/timing/validation with business logic:

```python
# ✗ WITHOUT decorators: mixed concerns
def train_model(epochs):
    # Logging code
    logger.info(f"Starting training with {epochs} epochs")
    start_time = time.time()

    # Business logic
    accuracy = 0.92  # Training happens here

    # More logging code
    end_time = time.time()
    logger.info(f"Training took {end_time - start_time}s")
    return accuracy

# ✓ WITH decorators: separated concerns
@timing_decorator
@log_calls
def train_model(epochs):
    accuracy = 0.92  # Only business logic
    return accuracy
```

**2. DRY Principle (Don't Repeat Yourself)**

Apply common patterns once via decorator, use everywhere:

```python
# Define once
def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
        return wrapper
    return decorator

# Use everywhere
@retry(max_attempts=3)
def load_model(path): pass

@retry(max_attempts=5)
def fetch_data(url): pass

@retry(max_attempts=3)
def save_checkpoint(path): pass
```

**3. Composability**

Stack decorators to combine behaviors:

```python
@rate_limit(calls_per_second=10)
@retry(max_attempts=3)
@timing_decorator
@log_calls
def call_external_api(endpoint):
    pass
```

**4. Testability**

Decorators can be tested independently:

```python
def test_timing_decorator():
    @timing_decorator
    def slow_func():
        time.sleep(0.1)

    # Capture logs and verify timing was logged
    slow_func()
```

**5. Centralized Updates**

Fix bugs or add features in one place:

```python
# Update retry logic once, affects all decorated functions
def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TemporaryError as e:  # ← Added specific exception
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(2 ** attempt)  # ← Added exponential backoff
        return wrapper
    return decorator
```

---

## Question 3: What are the benefits of organizing code into modules?

### Answer: Modules Provide Organization, Reusability, and Namespace Management

**Benefits of Modularization:**

**1. Code Organization**

```
ml_utils/
├── __init__.py          # Package initialization
├── metrics.py           # All metrics in one place
├── preprocessing.py     # All preprocessing in one place
└── visualization.py     # All visualization in one place
```

Clear structure makes code easy to find and navigate.

**2. Reusability**

Write once, use everywhere:

```python
# Define metrics module once
# ml_utils/metrics.py
def accuracy(preds, labels):
    return sum(p == l for p, l in zip(preds, labels)) / len(preds)

# Use in multiple projects
from ml_utils import metrics
acc = metrics.accuracy(predictions, labels)
```

**3. Namespace Management**

Avoid naming conflicts:

```python
# Without modules: conflict!
def load():  # Which load?
    pass

# With modules: clear
import ml_utils.data_loader as data_loader
import ml_utils.model_loader as model_loader

data = data_loader.load("data.csv")
model = model_loader.load("model.h5")
```

**4. Encapsulation**

Hide implementation details:

```python
# ml_utils/__init__.py
from . import metrics  # Public API
from . import preprocessing  # Public API
from . import _internal_helpers  # Private (convention: leading _)

__all__ = ["metrics", "preprocessing"]  # Explicit exports
```

**5. Testing**

Test modules independently:

```python
# tests/test_metrics.py
from ml_utils import metrics

def test_accuracy():
    assert metrics.accuracy([1, 0], [1, 0]) == 1.0
```

**6. Lazy Loading**

Import only what you need:

```python
# Load only metrics (not entire package)
from ml_utils import metrics

# Even more specific
from ml_utils.metrics import accuracy
```

**7. Versioning**

```python
# ml_utils/__init__.py
__version__ = "1.2.3"
```

**Package Structure Best Practices:**

```python
my_package/
├── __init__.py           # Package root
├── core/                 # Core functionality
│   ├── __init__.py
│   ├── models.py
│   └── utils.py
├── preprocessing/        # Data preprocessing
│   ├── __init__.py
│   ├── normalization.py
│   └── augmentation.py
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_models.py
│   └── test_preprocessing.py
└── examples/             # Usage examples
    └── basic_usage.py
```

---

## Question 4: When should you use *args and **kwargs?

### Answer: Use *args/**kwargs for Flexible, Extensible Interfaces

**When to Use *args:**

1. **Variable number of positional arguments:**

```python
def log_metrics(*metrics):
    """Log any number of metric values."""
    for i, metric in enumerate(metrics):
        print(f"Metric {i}: {metric}")

log_metrics(0.92, 0.88, 0.95)  # Works with any number
```

2. **Forwarding arguments:**

```python
def wrapper_function(data, *args):
    """Wrap another function and forward extra args."""
    return process_data(data, *args)
```

3. **Collecting remaining positional arguments:**

```python
def create_model(model_type, *layer_sizes, dropout=0.0):
    """
    Create model with flexible layer specification.

    Args:
        model_type: Type of model
        *layer_sizes: Variable number of layer sizes
        dropout: Dropout rate
    """
    return {"type": model_type, "layers": list(layer_sizes), "dropout": dropout}

model = create_model("cnn", 64, 128, 256, dropout=0.5)
```

**When to Use **kwargs:**

1. **Optional keyword arguments:**

```python
def train_model(data, **training_params):
    """Train with flexible parameters."""
    epochs = training_params.get("epochs", 10)
    lr = training_params.get("learning_rate", 0.001)
    batch_size = training_params.get("batch_size", 32)
    # ...

train_model(data, epochs=50, learning_rate=0.01, optimizer="adam")
```

2. **Configuration passing:**

```python
def create_pipeline(**config):
    """Create pipeline from configuration."""
    return Pipeline(
        normalize=config.get("normalize", True),
        augment=config.get("augment", False),
        batch_size=config.get("batch_size", 32),
        **config  # Pass remaining config
    )
```

3. **Forwarding keyword arguments:**

```python
def wrapper(data, **kwargs):
    """Wrap function and forward kwargs."""
    return process(data, **kwargs)
```

**Combining *args and **kwargs:**

```python
def flexible_function(required_arg, *args, optional_kwarg="default", **kwargs):
    """
    Demonstrates all argument types.

    Args:
        required_arg: Required positional argument
        *args: Additional positional arguments
        optional_kwarg: Optional keyword argument with default
        **kwargs: Additional keyword arguments
    """
    print(f"Required: {required_arg}")
    print(f"Args: {args}")
    print(f"Optional kwarg: {optional_kwarg}")
    print(f"Kwargs: {kwargs}")

# Usage examples:
flexible_function(1)
flexible_function(1, 2, 3)
flexible_function(1, optional_kwarg="custom")
flexible_function(1, 2, 3, optional_kwarg="custom", extra="value")
```

**When NOT to Use *args/**kwargs:**

1. **When you know the exact parameters:**

```python
# ✗ BAD: Unclear interface
def calculate_metrics(**kwargs):
    accuracy = kwargs["accuracy"]  # What if it's missing?
    precision = kwargs["precision"]
    recall = kwargs["recall"]

# ✓ GOOD: Clear interface
def calculate_metrics(accuracy: float, precision: float, recall: float):
    pass
```

2. **When type safety is important:**

```python
# ✗ BAD: Loses type checking
def train(data, **params):
    epochs = params["epochs"]  # Could be any type!

# ✓ GOOD: Type-safe
def train(data, epochs: int, learning_rate: float):
    pass
```

**Best Practice:** Use specific parameters for core functionality, *args/**kwargs for extensions.

---

## Question 5: How does functional programming improve code readability?

### Answer: Functional Patterns Provide Declarative, Composable Code

**Functional Programming Benefits:**

**1. Declarative vs. Imperative**

Functional code says *what* to do, not *how*:

```python
# ✗ IMPERATIVE: How to filter and transform
result = []
for x in data:
    if x > 0:
        squared = x ** 2
        result.append(squared)

# ✓ FUNCTIONAL: What to do
result = list(map(lambda x: x ** 2, filter(lambda x: x > 0, data)))

# Or with comprehension (even clearer):
result = [x ** 2 for x in data if x > 0]
```

**2. No Side Effects**

Pure functions are easier to understand:

```python
# ✗ IMPURE: Modifies external state
total = 0
def add_to_total(x):
    global total
    total += x  # Side effect!
    return total

# ✓ PURE: No side effects
def add(x, y):
    return x + y  # Only depends on inputs
```

**3. Composability**

Build complex operations from simple ones:

```python
# Simple functions
is_positive = lambda x: x > 0
square = lambda x: x ** 2
is_even = lambda x: x % 2 == 0

# Compose into pipeline
data = [-2, -1, 0, 1, 2, 3, 4, 5]

result = list(
    map(square,                    # 3. Square
        filter(is_positive,        # 2. Keep positive
               filter(is_even,     # 1. Keep even
                      data))))

# Result: [4, 16] (from 2→4, 4→16)
```

**4. Higher-Order Functions**

Functions as first-class citizens:

```python
def apply_to_batch(batch, transform_func):
    """Apply any transformation to batch."""
    return [transform_func(item) for item in batch]

# Use with different transformations
normalized = apply_to_batch(batch, lambda x: x / 255.0)
squared = apply_to_batch(batch, lambda x: x ** 2)
```

**5. Pipelines for Data Processing**

Clear flow of data transformations:

```python
# ML preprocessing pipeline
result = (
    data
    | filter(is_valid)           # Remove invalid samples
    | map(normalize)             # Normalize features
    | map(augment)               # Apply augmentation
    | batch(32)                  # Create batches
    | shuffle()                  # Shuffle batches
)
```

**Real-World Example:**

```python
# Process model metrics: filter high accuracy, extract names, sort
models = [
    {"name": "m1", "acc": 0.85, "loss": 0.25},
    {"name": "m2", "acc": 0.92, "loss": 0.15},
    {"name": "m3", "acc": 0.88, "loss": 0.20},
    {"name": "m4", "acc": 0.95, "loss": 0.10},
]

# FUNCTIONAL: Clear pipeline
best_models = sorted(
    map(lambda m: m["name"],
        filter(lambda m: m["acc"] > 0.87, models))
)
# Result: ['m2', 'm3', 'm4']

# vs. IMPERATIVE: More verbose
best_models = []
for model in models:
    if model["acc"] > 0.87:
        best_models.append(model["name"])
best_models.sort()
```

**When NOT to Use Functional Patterns:**

1. Complex logic (use explicit loops)
2. Need to debug step-by-step
3. Performance-critical code (measure first!)

**Balance:** Use functional patterns for simple transformations, explicit loops for complex logic.

---

## Question 6: What testing strategies should you use for utility functions?

### Answer: Comprehensive Testing Strategy for Utility Functions

**Testing Pyramid for Utilities:**

1. **Unit Tests** (Most important)
2. **Property-Based Tests**
3. **Integration Tests**
4. **Edge Case Tests**

### 1. Unit Tests

Test each function independently:

```python
def test_accuracy_perfect():
    """Test accuracy with perfect predictions."""
    preds = [1, 0, 1, 0]
    labels = [1, 0, 1, 0]
    assert accuracy(preds, labels) == 1.0

def test_accuracy_half():
    """Test accuracy with 50% correct."""
    preds = [1, 0, 1, 0]
    labels = [0, 1, 1, 0]
    assert accuracy(preds, labels) == 0.5
```

### 2. Edge Cases

Test boundary conditions:

```python
def test_accuracy_empty():
    """Test accuracy with empty lists."""
    assert accuracy([], []) == 0.0

def test_normalize_single_value():
    """Test normalization with single value."""
    assert normalize_minmax([5.0]) == [0.0]

def test_normalize_same_values():
    """Test normalization when all values are same."""
    result = normalize_minmax([3.0, 3.0, 3.0])
    assert all(x == 0.0 for x in result)
```

### 3. Error Handling

Test that errors are raised appropriately:

```python
def test_accuracy_mismatched_lengths():
    """Test that mismatched lengths raise ValueError."""
    with pytest.raises(ValueError):
        accuracy([1, 0, 1], [1, 0])

def test_invalid_normalization_method():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError):
        normalize_features([1, 2, 3], method="unknown")
```

### 4. Parameterized Tests

Test multiple inputs efficiently:

```python
@pytest.mark.parametrize("preds,labels,expected", [
    ([1, 0, 1, 0], [1, 0, 1, 0], 1.0),
    ([1, 0, 1, 0], [0, 1, 1, 0], 0.5),
    ([1, 1, 1, 1], [0, 0, 0, 0], 0.0),
])
def test_accuracy_various_inputs(preds, labels, expected):
    assert accuracy(preds, labels) == expected
```

### 5. Property-Based Tests

Test properties that should always hold:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0, max_value=100)))
def test_normalize_output_range(data):
    """Normalized values should be in [0, 1]."""
    if len(data) > 0:
        normalized = normalize_minmax(data)
        assert all(0 <= x <= 1 for x in normalized)

@given(st.lists(st.integers(min_value=0, max_value=1), min_size=1))
def test_accuracy_range(labels):
    """Accuracy should always be in [0, 1]."""
    # Test with perfect predictions
    acc = accuracy(labels, labels)
    assert 0 <= acc <= 1
```

### 6. Reproducibility Tests

Test that functions with randomness are reproducible with seeds:

```python
def test_train_test_split_reproducible():
    """Test that split with same seed gives same results."""
    data = list(range(100))
    train1, test1 = train_test_split(data, test_size=0.2, random_seed=42)
    train2, test2 = train_test_split(data, test_size=0.2, random_seed=42)
    assert train1 == train2
    assert test1 == test2
```

### 7. Coverage Targets

- **Unit tests:** 90%+ code coverage
- **Edge cases:** All boundary conditions covered
- **Error paths:** All `raise` statements tested
- **Integration:** Key workflows tested end-to-end

### Complete Test Example:

```python
class TestAccuracy:
    """Comprehensive tests for accuracy function."""

    def test_perfect_accuracy(self):
        """Perfect predictions should give 1.0."""
        assert accuracy([1, 0], [1, 0]) == 1.0

    def test_zero_accuracy(self):
        """All wrong predictions should give 0.0."""
        assert accuracy([1, 1], [0, 0]) == 0.0

    def test_half_accuracy(self):
        """Half correct should give 0.5."""
        assert accuracy([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5

    def test_empty_lists(self):
        """Empty lists should give 0.0."""
        assert accuracy([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            accuracy([1], [1, 0])

    @pytest.mark.parametrize("preds,labels,expected", [
        ([1], [1], 1.0),
        ([0], [1], 0.0),
        ([1, 1], [1, 1], 1.0),
    ])
    def test_various_inputs(self, preds, labels, expected):
        """Test various input combinations."""
        assert accuracy(preds, labels) == expected
```

---

## Question 7: How do you balance flexibility vs. simplicity in function design?

### Answer: Progressive Enhancement - Start Simple, Add Flexibility as Needed

**Design Principles:**

### 1. Start with Simplest Interface

```python
# ✓ SIMPLE: Start here
def train_model(data, epochs):
    """Train model with essential parameters only."""
    return train(data, epochs)
```

### 2. Add Defaults for Common Options

```python
# ✓ SLIGHTLY MORE FLEXIBLE: Add common options with defaults
def train_model(data, epochs, learning_rate=0.001, batch_size=32):
    """Train model with sensible defaults."""
    return train(data, epochs, learning_rate, batch_size)
```

### 3. Use **kwargs for Advanced Options

```python
# ✓ FLEXIBLE: Advanced users can customize
def train_model(data, epochs, learning_rate=0.001, batch_size=32, **advanced_options):
    """
    Train model with optional advanced configuration.

    Args:
        data: Training data
        epochs: Number of epochs
        learning_rate: Learning rate (default: 0.001)
        batch_size: Batch size (default: 32)
        **advanced_options: Advanced options:
            - optimizer: Optimizer type ("adam", "sgd")
            - weight_decay: L2 regularization
            - warmup_steps: Learning rate warmup
    """
    optimizer = advanced_options.get("optimizer", "adam")
    # ...
```

### Decision Matrix:

| Scenario | Approach | Example |
|----------|----------|---------|
| 1-2 parameters | Required args | `def add(x, y)` |
| 3-5 parameters | Required + defaults | `def train(data, epochs=10, lr=0.001)` |
| 5+ parameters | Config object | `def train(config: TrainingConfig)` |
| Variable args | *args or List | `def log_metrics(*metrics)` |
| Many options | **kwargs | `def create_model(**config)` |

### Real-World Examples:

**✓ GOOD: Progressive API**

```python
# Simple for beginners
train_model(data, epochs=10)

# Intermediate users tune key parameters
train_model(data, epochs=50, learning_rate=0.01, batch_size=64)

# Advanced users customize everything
train_model(
    data,
    epochs=100,
    learning_rate=0.001,
    batch_size=32,
    optimizer="adam",
    weight_decay=0.0001,
    warmup_steps=1000,
    gradient_clip=1.0
)
```

**✗ BAD: Too Flexible (Unclear)**

```python
# What parameters are valid? What do they do?
train_model(**config)
```

**✗ BAD: Too Rigid (Not Extensible)**

```python
# Can't customize anything!
def train_model(data):
    epochs = 10  # Hardcoded!
    lr = 0.001   # Can't change!
    # ...
```

### Guidelines:

1. **Required parameters:** Essential inputs with no sensible defaults
2. **Optional with defaults:** Common parameters users might tune
3. ****kwargs:** Advanced/rare options
4. **Config objects:** When you have 10+ parameters

### Documentation is Key:

```python
def train_model(data, epochs=10, **kwargs):
    """
    Train ML model.

    Args:
        data: Training data
        epochs: Number of epochs (default: 10)
        **kwargs: Advanced options:
            learning_rate (float): Learning rate (default: 0.001)
            batch_size (int): Batch size (default: 32)
            optimizer (str): Optimizer ("adam", "sgd", default: "adam")
            weight_decay (float): L2 regularization (default: 0.0)

    Examples:
        # Simple usage:
        >>> train_model(data, epochs=50)

        # Advanced usage:
        >>> train_model(data, epochs=100, learning_rate=0.01,
        ...            optimizer="sgd", weight_decay=0.0001)
    """
```

---

## Summary

Key takeaways:

1. **Type hints:** Always use in production for IDE support and static checking
2. **Decorators:** Separate cross-cutting concerns, promote DRY, enhance maintainability
3. **Modules:** Organize code, provide reusability, manage namespaces
4. **\*args/\*\*kwargs:** Use for flexibility, but prefer explicit parameters when possible
5. **Functional programming:** Declarative, composable, clear data flow
6. **Testing:** Comprehensive unit tests, edge cases, error handling, property-based tests
7. **Flexibility vs. simplicity:** Start simple, add flexibility progressively with defaults and **kwargs

**Golden Rule:** Design for the common case, but allow for customization.

---

**Last Updated:** 2025-10-30
