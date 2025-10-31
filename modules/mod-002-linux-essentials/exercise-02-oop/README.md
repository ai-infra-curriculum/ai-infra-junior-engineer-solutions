# Exercise 02: Object-Oriented Programming (OOP)

## Overview

Master object-oriented programming principles to build scalable, maintainable AI infrastructure systems. Learn classes, inheritance, design patterns, and best practices for production Python code.

## Learning Objectives

- ✅ Create and use classes with proper encapsulation
- ✅ Implement inheritance and polymorphism
- ✅ Use magic methods (dunder methods) effectively
- ✅ Apply decorators and properties
- ✅ Understand and apply common design patterns
- ✅ Write maintainable object-oriented code

## Topics Covered

### 1. Classes and Objects

```python
class MLModel:
    """Machine learning model class."""

    # Class variable (shared by all instances)
    model_type = "classifier"

    def __init__(self, name: str, version: str):
        """
        Initialize model.

        Args:
            name: Model name
            version: Model version
        """
        # Instance variables (unique to each instance)
        self.name = name
        self.version = version
        self.is_trained = False

    def train(self, data):
        """Train the model."""
        print(f"Training {self.name} v{self.version}")
        # Training logic here
        self.is_trained = True

    def predict(self, input_data):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        # Prediction logic here
        return []

# Creating instances
model1 = MLModel("ImageClassifier", "1.0")
model2 = MLModel("TextClassifier", "2.0")

# Using methods
model1.train(data)
predictions = model1.predict(input_data)
```

### 2. Encapsulation (Private/Protected Members)

```python
class ModelManager:
    """Manage ML models with encapsulation."""

    def __init__(self):
        self._models = {}  # Protected (convention: single underscore)
        self.__api_key = None  # Private (name mangling: double underscore)

    def add_model(self, name: str, model):
        """Public method to add model."""
        self._models[name] = model

    def _validate_model(self, model):
        """Protected method (internal use)."""
        return hasattr(model, 'predict')

    def __authenticate(self):
        """Private method (strongly discouraged external access)."""
        # Authentication logic
        pass

# Usage
manager = ModelManager()
manager.add_model("classifier", model)  # OK
manager._models  # Accessible but discouraged
manager.__authenticate()  # AttributeError (name mangled to _ModelManager__authenticate)
```

### 3. Inheritance

```python
# Base class
class Model:
    """Base class for all models."""

    def __init__(self, name: str):
        self.name = name

    def train(self, data):
        """Train model - to be overridden."""
        raise NotImplementedError("Subclasses must implement train()")

    def evaluate(self, test_data):
        """Evaluate model - common to all models."""
        print(f"Evaluating {self.name}")
        # Common evaluation logic
        return {"accuracy": 0.95}


# Derived class
class NeuralNetwork(Model):
    """Neural network model."""

    def __init__(self, name: str, layers: int):
        super().__init__(name)  # Call parent constructor
        self.layers = layers

    def train(self, data):
        """Override train method."""
        print(f"Training neural network with {self.layers} layers")
        # NN-specific training logic


class RandomForest(Model):
    """Random forest model."""

    def __init__(self, name: str, n_trees: int):
        super().__init__(name)
        self.n_trees = n_trees

    def train(self, data):
        """Override train method."""
        print(f"Training random forest with {self.n_trees} trees")
        # RF-specific training logic


# Multiple inheritance
class Logger:
    """Mixin for logging functionality."""

    def log(self, message: str):
        print(f"[LOG] {message}")


class MonitoredModel(Model, Logger):
    """Model with logging capabilities."""

    def train(self, data):
        self.log("Starting training")
        super().train(data)
        self.log("Training complete")
```

### 4. Polymorphism

```python
def train_model(model: Model, data):
    """
    Train any model - polymorphism in action.

    Works with any Model subclass.
    """
    model.train(data)
    results = model.evaluate(test_data)
    return results

# Works with different model types
nn = NeuralNetwork("CNN", layers=10)
rf = RandomForest("Classifier", n_trees=100)

train_model(nn, training_data)  # Uses NeuralNetwork.train()
train_model(rf, training_data)  # Uses RandomForest.train()
```

### 5. Magic Methods (Dunder Methods)

```python
class Dataset:
    """Dataset with magic methods."""

    def __init__(self, data: list):
        self._data = data

    def __len__(self):
        """Called by len()."""
        return len(self._data)

    def __getitem__(self, index):
        """Called by dataset[index]."""
        return self._data[index]

    def __setitem__(self, index, value):
        """Called by dataset[index] = value."""
        self._data[index] = value

    def __str__(self):
        """Called by str() and print()."""
        return f"Dataset with {len(self)} items"

    def __repr__(self):
        """Called by repr() - should be unambiguous."""
        return f"Dataset({self._data})"

    def __eq__(self, other):
        """Called by ==."""
        if not isinstance(other, Dataset):
            return False
        return self._data == other._data

    def __add__(self, other):
        """Called by +."""
        return Dataset(self._data + other._data)

    def __iter__(self):
        """Make dataset iterable."""
        return iter(self._data)

    def __contains__(self, item):
        """Called by 'in' operator."""
        return item in self._data

    def __call__(self, transform=None):
        """Make dataset callable."""
        if transform:
            return Dataset([transform(x) for x in self._data])
        return self


# Usage
dataset = Dataset([1, 2, 3, 4, 5])
print(len(dataset))  # 5
print(dataset[0])  # 1
print(3 in dataset)  # True
for item in dataset:  # Iterable
    print(item)
```

### 6. Properties and Decorators

```python
class Model:
    """Model with properties."""

    def __init__(self, name: str):
        self._name = name
        self._accuracy = 0.0

    @property
    def name(self):
        """Get model name."""
        return self._name

    @property
    def accuracy(self):
        """Get accuracy."""
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value: float):
        """Set accuracy with validation."""
        if not 0 <= value <= 1:
            raise ValueError("Accuracy must be between 0 and 1")
        self._accuracy = value

    @property
    def is_trained(self):
        """Computed property."""
        return self._accuracy > 0

    # Class method
    @classmethod
    def from_config(cls, config: dict):
        """Create model from configuration."""
        return cls(config['name'])

    # Static method
    @staticmethod
    def validate_data(data):
        """Validate data - doesn't need self or cls."""
        return len(data) > 0


# Usage
model = Model("Classifier")
model.accuracy = 0.95  # Uses setter
print(model.accuracy)  # Uses getter
print(model.is_trained)  # True (computed)

# Class method
model2 = Model.from_config({"name": "Regressor"})

# Static method
Model.validate_data(data)
```

### 7. Abstract Base Classes

```python
from abc import ABC, abstractmethod

class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, path: str):
        """Load data - must be implemented by subclasses."""
        pass

    @abstractmethod
    def save(self, path: str, data):
        """Save data - must be implemented by subclasses."""
        pass

    def validate(self, data):
        """Concrete method - shared by all subclasses."""
        return data is not None


class CSVLoader(DataLoader):
    """CSV data loader."""

    def load(self, path: str):
        """Load CSV file."""
        import pandas as pd
        return pd.read_csv(path)

    def save(self, path: str, data):
        """Save to CSV."""
        data.to_csv(path, index=False)


class JSONLoader(DataLoader):
    """JSON data loader."""

    def load(self, path: str):
        """Load JSON file."""
        import json
        with open(path) as f:
            return json.load(f)

    def save(self, path: str, data):
        """Save to JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(data, f)


# Cannot instantiate abstract class
# loader = DataLoader()  # TypeError

# Can instantiate concrete classes
csv_loader = CSVLoader()
json_loader = JSONLoader()
```

---

## Design Patterns

### 1. Singleton Pattern

```python
class ModelRegistry:
    """Singleton pattern - only one instance exists."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def register(self, name: str, model):
        """Register a model."""
        self._models[name] = model

    def get(self, name: str):
        """Get registered model."""
        return self._models.get(name)


# Always returns same instance
registry1 = ModelRegistry()
registry2 = ModelRegistry()
assert registry1 is registry2  # True
```

### 2. Factory Pattern

```python
class ModelFactory:
    """Factory pattern for creating models."""

    @staticmethod
    def create_model(model_type: str, **kwargs):
        """Create model based on type."""
        if model_type == "neural_network":
            return NeuralNetwork(**kwargs)
        elif model_type == "random_forest":
            return RandomForest(**kwargs)
        elif model_type == "svm":
            return SVM(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Usage
model = ModelFactory.create_model(
    "neural_network",
    name="CNN",
    layers=10
)
```

### 3. Observer Pattern

```python
class Subject:
    """Subject being observed."""

    def __init__(self):
        self._observers = []

    def attach(self, observer):
        """Attach an observer."""
        self._observers.append(observer)

    def notify(self, event):
        """Notify all observers."""
        for observer in self._observers:
            observer.update(event)


class TrainingMonitor:
    """Observer for training events."""

    def update(self, event):
        """Handle training event."""
        print(f"Training event: {event}")


class MetricsLogger:
    """Observer for logging metrics."""

    def update(self, event):
        """Log metrics."""
        if 'metrics' in event:
            print(f"Metrics: {event['metrics']}")


# Usage
trainer = Subject()
trainer.attach(TrainingMonitor())
trainer.attach(MetricsLogger())

trainer.notify({"event": "epoch_complete", "metrics": {"loss": 0.5}})
```

### 4. Strategy Pattern

```python
from abc import ABC, abstractmethod

class TrainingStrategy(ABC):
    """Abstract training strategy."""

    @abstractmethod
    def train(self, model, data):
        """Train model using this strategy."""
        pass


class StandardTraining(TrainingStrategy):
    """Standard training strategy."""

    def train(self, model, data):
        print("Training with standard approach")
        # Standard training logic


class DistributedTraining(TrainingStrategy):
    """Distributed training strategy."""

    def train(self, model, data):
        print("Training with distributed approach")
        # Distributed training logic


class Trainer:
    """Trainer that uses different strategies."""

    def __init__(self, strategy: TrainingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: TrainingStrategy):
        """Change strategy at runtime."""
        self._strategy = strategy

    def train_model(self, model, data):
        """Train using current strategy."""
        return self._strategy.train(model, data)


# Usage
trainer = Trainer(StandardTraining())
trainer.train_model(model, data)

# Switch strategy
trainer.set_strategy(DistributedTraining())
trainer.train_model(model, data)
```

---

## Project: ML Model Manager with Caching

Build a complete ML model management system with caching, monitoring, and multiple model support.

### Requirements

**Features:**
1. Model registration and retrieval
2. LRU caching for predictions
3. Model versioning
4. Performance monitoring
5. Model comparison
6. Serialization/deserialization

**Classes to Implement:**
- `BaseModel` (abstract base class)
- `ModelCache` (LRU cache implementation)
- `ModelRegistry` (singleton for model management)
- `ModelMetrics` (track performance)
- `ModelComparator` (compare models)

### Implementation

See `solutions/model_manager.py` for complete implementation.

### Example Usage

```python
from model_manager import ModelRegistry, ModelCache, BaseModel

# Create and register models
registry = ModelRegistry()
model = MyMLModel("classifier", "1.0")
registry.register("my_model", model)

# Use caching
cache = ModelCache(max_size=100)
cached_model = cache.wrap_model(model)

# Make predictions (cached)
result1 = cached_model.predict(data)  # Cache miss
result2 = cached_model.predict(data)  # Cache hit (faster)

# Get metrics
metrics = model.get_metrics()
print(f"Avg latency: {metrics.avg_latency_ms}ms")
print(f"Cache hit rate: {cache.hit_rate}%")
```

---

## Best Practices

### 1. SOLID Principles

**S - Single Responsibility:**
```python
# Good - each class has one responsibility
class DataLoader:
    def load(self, path): pass

class DataValidator:
    def validate(self, data): pass

class DataTransformer:
    def transform(self, data): pass

# Bad - too many responsibilities
class DataProcessor:
    def load(self, path): pass
    def validate(self, data): pass
    def transform(self, data): pass
    def save(self, path, data): pass
```

**O - Open/Closed (Open for extension, closed for modification):**
```python
# Good - extend with inheritance
class Model(ABC):
    @abstractmethod
    def predict(self, data): pass

class NewModel(Model):
    def predict(self, data):
        # New implementation
        pass
```

**L - Liskov Substitution:**
```python
# Subclasses should be substitutable for base classes
def process_model(model: Model):
    return model.predict(data)

# Works with any Model subclass
process_model(NeuralNetwork())
process_model(RandomForest())
```

**I - Interface Segregation:**
```python
# Good - separate interfaces
class Trainable(ABC):
    @abstractmethod
    def train(self, data): pass

class Evaluable(ABC):
    @abstractmethod
    def evaluate(self, data): pass

# Classes implement only what they need
class SimpleModel(Trainable):
    def train(self, data): pass
```

**D - Dependency Inversion:**
```python
# Depend on abstractions, not concretions
class ModelTrainer:
    def __init__(self, model: Model):  # Depends on abstraction
        self.model = model

    def train(self, data):
        self.model.train(data)
```

### 2. Composition over Inheritance

```python
# Prefer composition
class ModelWithLogging:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def predict(self, data):
        self.logger.log("Predicting...")
        return self.model.predict(data)

# Over inheritance
class LoggingModel(Model, Logger):  # Multiple inheritance can be complex
    pass
```

### 3. Type Hints

```python
from typing import List, Dict, Optional, Union

class Model:
    def __init__(self, name: str, version: str):
        self.name: str = name
        self.version: str = version

    def predict(
        self,
        inputs: List[float]
    ) -> Dict[str, Union[int, float]]:
        """Type hints improve code clarity."""
        return {"prediction": 1, "confidence": 0.95}
```

---

## Common Patterns in ML Infrastructure

### 1. Builder Pattern for Model Configuration

```python
class ModelBuilder:
    """Builder pattern for complex model construction."""

    def __init__(self):
        self._config = {}

    def set_architecture(self, arch: str):
        self._config['architecture'] = arch
        return self

    def set_layers(self, layers: int):
        self._config['layers'] = layers
        return self

    def set_optimizer(self, optimizer: str):
        self._config['optimizer'] = optimizer
        return self

    def build(self):
        """Build the model."""
        return Model(self._config)


# Usage
model = (ModelBuilder()
         .set_architecture('resnet')
         .set_layers(50)
         .set_optimizer('adam')
         .build())
```

### 2. Context Manager for Resource Management

```python
class GPUMemoryManager:
    """Context manager for GPU memory."""

    def __enter__(self):
        """Acquire resources."""
        import torch
        torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources."""
        import torch
        torch.cuda.empty_cache()
        return False  # Don't suppress exceptions


# Usage
with GPUMemoryManager():
    # GPU-intensive operations
    model.train(data)
```

---

## Testing OOP Code

```python
import pytest

class TestModel:
    """Test model class."""

    def test_initialization(self):
        """Test model initialization."""
        model = Model("test", "1.0")
        assert model.name == "test"
        assert model.version == "1.0"

    def test_inheritance(self):
        """Test inheritance."""
        nn = NeuralNetwork("nn", layers=5)
        assert isinstance(nn, Model)
        assert nn.layers == 5

    def test_polymorphism(self):
        """Test polymorphic behavior."""
        models = [
            NeuralNetwork("nn", layers=5),
            RandomForest("rf", n_trees=100)
        ]
        for model in models:
            model.train(data)  # Works for all models

    @pytest.fixture
    def model(self):
        """Fixture for model instance."""
        return Model("test", "1.0")

    def test_with_fixture(self, model):
        """Test using fixture."""
        assert model.name == "test"
```

---

## Validation

Run the test suite:

```bash
pytest tests/test_oop.py -v
```

Expected output:
```
✅ Class instantiation tests passed
✅ Inheritance tests passed
✅ Polymorphism tests passed
✅ Magic methods tests passed
✅ Design patterns tests passed
✅ Model manager project tests passed

🎉 Exercise 02 Complete!
```

---

## Resources

### Documentation
- [Python Classes](https://docs.python.org/3/tutorial/classes.html)
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)

### Books
- "Fluent Python" - Luciano Ramalho
- "Python Design Patterns" - Brandon Rhodes
- "Clean Code in Python" - Mariano Anaya

### Design Patterns
- [Refactoring Guru](https://refactoring.guru/design-patterns/python)
- [Python Patterns](https://python-patterns.guide/)

---

## Next Steps

After completing this exercise:

1. **Exercise 03: File I/O & Error Handling**
2. Practice OOP with real projects
3. Refactor old code to use OOP principles
4. Study more design patterns

---

**Time to build scalable systems with OOP! 🏗️**
