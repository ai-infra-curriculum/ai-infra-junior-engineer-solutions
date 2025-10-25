# Step-by-Step Implementation Guide: Object-Oriented Programming

## Overview

Master OOP principles to build scalable AI infrastructure systems: classes, inheritance, design patterns, and production-ready code.

**Time**: 4-6 hours | **Difficulty**: Intermediate

---

## Phase 1: Classes and Objects (1 hour)

### Step 1: Create Basic Class

```python
# models/ml_model.py
class MLModel:
    """Machine learning model class."""

    model_type = "classifier"  # Class variable

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.is_trained = False
        self.accuracy = 0.0

    def train(self, X, y):
        """Train the model."""
        print(f"Training {self.name} v{self.version}...")
        # Simulate training
        self.is_trained = True
        self.accuracy = 0.95
        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return ["prediction"] * len(X)

    def __repr__(self):
        return f"MLModel(name={self.name}, version={self.version}, accuracy={self.accuracy})"

# Test
if __name__ == "__main__":
    model = MLModel("ImageClassifier", "1.0")
    model.train([[1, 2], [3, 4]], [0, 1])
    predictions = model.predict([[5, 6]])
    print(model)
```

---

## Phase 2: Inheritance and Polymorphism (1.5 hours)

### Step 2: Implement Inheritance

```python
# models/specialized_models.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False

    @abstractmethod
    def train(self, X, y):
        """Train the model (must be implemented by subclasses)."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions (must be implemented by subclasses)."""
        pass

class Classifier(BaseModel):
    """Classification model."""

    def __init__(self, name: str, num_classes: int):
        super().__init__(name)
        self.num_classes = num_classes

    def train(self, X, y):
        print(f"Training classifier {self.name} with {self.num_classes} classes")
        self.is_trained = True

    def predict(self, X):
        import random
        return [random.randint(0, self.num_classes - 1) for _ in X]

class Regressor(BaseModel):
    """Regression model."""

    def train(self, X, y):
        print(f"Training regressor {self.name}")
        self.is_trained = True

    def predict(self, X):
        import random
        return [random.random() for _ in X]

# Polymorphism
def evaluate_model(model: BaseModel, X, y):
    """Evaluate any model (polymorphism)."""
    model.train(X, y)
    predictions = model.predict(X)
    print(f"Model {model.name}: {len(predictions)} predictions")

# Test
classifier = Classifier("ImageNet", num_classes=1000)
regressor = Regressor("HousePrices")

evaluate_model(classifier, [[1]], [0])
evaluate_model(regressor, [[1]], [100.5])
```

---

## Phase 3: Design Patterns (1.5 hours)

### Step 3: Singleton Pattern

```python
# patterns/singleton.py
class ModelRegistry:
    """Singleton registry for models."""

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
        """Get a registered model."""
        return self._models.get(name)

# Test singleton
registry1 = ModelRegistry()
registry1.register("model1", "Model Instance 1")

registry2 = ModelRegistry()
print(registry2.get("model1"))  # Accesses same instance
print(registry1 is registry2)  # True
```

### Step 4: Factory Pattern

```python
# patterns/factory.py
class ModelFactory:
    """Factory for creating models."""

    @staticmethod
    def create_model(model_type: str, **kwargs):
        """Create model based on type."""
        if model_type == "classifier":
            return Classifier(**kwargs)
        elif model_type == "regressor":
            return Regressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Usage
model = ModelFactory.create_model("classifier", name="BERT", num_classes=2)
```

### Step 5: Builder Pattern

```python
# patterns/builder.py
class ModelBuilder:
    """Builder for complex model configuration."""

    def __init__(self):
        self._config = {}

    def set_name(self, name: str):
        self._config['name'] = name
        return self

    def set_optimizer(self, optimizer: str):
        self._config['optimizer'] = optimizer
        return self

    def set_learning_rate(self, lr: float):
        self._config['learning_rate'] = lr
        return self

    def build(self):
        return self._config

# Usage
config = (ModelBuilder()
          .set_name("ResNet50")
          .set_optimizer("Adam")
          .set_learning_rate(0.001)
          .build())
```

---

## Phase 4: Properties and Decorators (1 hour)

### Step 6: Properties

```python
# advanced/properties.py
class Model:
    """Model with properties."""

    def __init__(self, name: str):
        self._name = name
        self._accuracy = 0.0

    @property
    def name(self):
        """Get model name."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set model name with validation."""
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value

    @property
    def accuracy(self):
        """Get accuracy as percentage."""
        return f"{self._accuracy * 100:.2f}%"

    @accuracy.setter
    def accuracy(self, value: float):
        """Set accuracy with validation."""
        if not 0 <= value <= 1:
            raise ValueError("Accuracy must be between 0 and 1")
        self._accuracy = value

# Usage
model = Model("BERT")
model.accuracy = 0.95
print(model.accuracy)  # "95.00%"
```

### Step 7: Custom Decorators

```python
# advanced/decorators.py
import time
from functools import wraps

def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

def validate_trained(func):
    """Decorator to validate model is trained."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return func(self, *args, **kwargs)
    return wrapper

class MLModel:
    def __init__(self):
        self.is_trained = False

    @timer
    def train(self, X, y):
        time.sleep(0.1)  # Simulate training
        self.is_trained = True

    @validate_trained
    @timer
    def predict(self, X):
        return [1] * len(X)

# Test
model = MLModel()
model.train([[1]], [0])
model.predict([[2]])
```

---

## Phase 5: Magic Methods (1 hour)

### Step 8: Implement Magic Methods

```python
# advanced/magic_methods.py
class ModelEnsemble:
    """Ensemble of models with magic methods."""

    def __init__(self, models=None):
        self.models = models or []

    def __len__(self):
        """len(ensemble)"""
        return len(self.models)

    def __getitem__(self, index):
        """ensemble[index]"""
        return self.models[index]

    def __setitem__(self, index, model):
        """ensemble[index] = model"""
        self.models[index] = model

    def __contains__(self, model):
        """model in ensemble"""
        return model in self.models

    def __add__(self, other):
        """ensemble1 + ensemble2"""
        return ModelEnsemble(self.models + other.models)

    def __repr__(self):
        return f"ModelEnsemble({len(self)} models)"

    def __str__(self):
        return f"Ensemble with {len(self)} models"

# Test
ensemble = ModelEnsemble([model1, model2])
print(len(ensemble))  # 2
print(ensemble[0])  # model1
print(model1 in ensemble)  # True
```

---

## Summary

**What You Built**:
- ✅ Classes with proper encapsulation
- ✅ Inheritance hierarchy (BaseModel → Classifier/Regressor)
- ✅ Design patterns (Singleton, Factory, Builder)
- ✅ Properties with validation
- ✅ Custom decorators (@timer, @validate_trained)
- ✅ Magic methods for Pythonic classes

**Key OOP Concepts**:
```python
# Inheritance
class Derived(Base):
    def __init__(self):
        super().__init__()

# Properties
@property
def value(self):
    return self._value

# Decorators
@staticmethod
def method():
    pass

# Magic methods
def __len__(self):
    return len(self.items)
```

**Next Steps**:
- Exercise 03: File I/O and Error Handling
- Study SOLID principles
- Practice design patterns
- Read "Design Patterns" by Gang of Four
