#!/usr/bin/env python3
"""
ML Model Manager with Caching

A complete object-oriented system for managing ML models with LRU caching,
versioning, monitoring, and comparison capabilities.

This implementation demonstrates:
- Class design and OOP principles
- Encapsulation and abstraction
- Inheritance and polymorphism
- Magic methods
- Design patterns (Singleton, Factory, Observer)
- Context managers
- Property decorators
- Type hints and documentation
"""

from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple
import time
import pickle
import hashlib
from collections import OrderedDict
from pathlib import Path


# ===========================
# Abstract Base Classes
# ===========================

class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    Defines the interface that all models must implement.
    """

    def __init__(self, name: str, version: str):
        """
        Initialize base model.

        Args:
            name: Model name
            version: Model version string
        """
        self._name = name
        self._version = version
        self._is_trained = False
        self._metrics: Dict[str, float] = {}
        self._created_at = datetime.now()

    @property
    def name(self) -> str:
        """Get model name."""
        return self._name

    @property
    def version(self) -> str:
        """Get model version."""
        return self._version

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    @property
    def metrics(self) -> Dict[str, float]:
        """Get model metrics."""
        return self._metrics.copy()

    @abstractmethod
    def train(self, data: Any) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def __str__(self) -> str:
        """String representation."""
        status = "trained" if self._is_trained else "untrained"
        return f"{self._name} v{self._version} ({status})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"BaseModel(name='{self._name}', version='{self._version}', "
            f"is_trained={self._is_trained})"
        )


# ===========================
# Concrete Model Classes
# ===========================

class NeuralNetwork(BaseModel):
    """Neural network model implementation."""

    def __init__(self, name: str, version: str, layers: List[int]):
        """
        Initialize neural network.

        Args:
            name: Model name
            version: Model version
            layers: List of layer sizes
        """
        super().__init__(name, version)
        self.layers = layers
        self._weights = None

    def train(self, data: Any) -> None:
        """Train the neural network."""
        print(f"Training {self.name} with layers {self.layers}")
        time.sleep(0.1)  # Simulate training
        self._is_trained = True
        self._metrics = {
            "accuracy": 0.95,
            "loss": 0.05,
            "training_time": 120.5
        }

    def predict(self, input_data: Any) -> Any:
        """Make predictions."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        # Simulate prediction
        return [0.8, 0.2]

    def save(self, path: str) -> None:
        """Save model to disk."""
        model_data = {
            "name": self._name,
            "version": self._version,
            "layers": self.layers,
            "is_trained": self._is_trained,
            "metrics": self._metrics,
            "weights": self._weights
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self._name = model_data["name"]
        self._version = model_data["version"]
        self.layers = model_data["layers"]
        self._is_trained = model_data["is_trained"]
        self._metrics = model_data["metrics"]
        self._weights = model_data["weights"]


class RandomForest(BaseModel):
    """Random forest model implementation."""

    def __init__(self, name: str, version: str, n_trees: int = 100):
        """
        Initialize random forest.

        Args:
            name: Model name
            version: Model version
            n_trees: Number of trees
        """
        super().__init__(name, version)
        self.n_trees = n_trees
        self._trees = None

    def train(self, data: Any) -> None:
        """Train the random forest."""
        print(f"Training {self.name} with {self.n_trees} trees")
        time.sleep(0.1)  # Simulate training
        self._is_trained = True
        self._metrics = {
            "accuracy": 0.92,
            "f1_score": 0.91,
            "training_time": 45.2
        }

    def predict(self, input_data: Any) -> Any:
        """Make predictions."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        # Simulate prediction
        return 1

    def save(self, path: str) -> None:
        """Save model to disk."""
        model_data = {
            "name": self._name,
            "version": self._version,
            "n_trees": self.n_trees,
            "is_trained": self._is_trained,
            "metrics": self._metrics,
            "trees": self._trees
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self._name = model_data["name"]
        self._version = model_data["version"]
        self.n_trees = model_data["n_trees"]
        self._is_trained = model_data["is_trained"]
        self._metrics = model_data["metrics"]
        self._trees = model_data["trees"]


# ===========================
# LRU Cache Implementation
# ===========================

class LRUCache:
    """
    Least Recently Used cache for model predictions.

    Demonstrates custom data structure and magic methods.
    """

    def __init__(self, capacity: int):
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items to cache
        """
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        """Get cache size."""
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self.cache

    def __str__(self) -> str:
        """String representation."""
        return (
            f"LRUCache(capacity={self.capacity}, size={len(self.cache)}, "
            f"hit_rate={self.hit_rate:.2%})"
        )


# ===========================
# Model Registry (Singleton)
# ===========================

class ModelRegistry:
    """
    Singleton registry for managing all models.

    Demonstrates Singleton design pattern.
    """

    _instance: Optional['ModelRegistry'] = None

    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry (only once)."""
        if self._initialized:
            return

        self._models: Dict[str, BaseModel] = {}
        self._cache = LRUCache(capacity=100)
        self._observers: List['ModelObserver'] = []
        self._initialized = True

    def register(self, model: BaseModel) -> None:
        """
        Register a model.

        Args:
            model: Model to register
        """
        model_id = f"{model.name}:{model.version}"
        self._models[model_id] = model
        self._notify_observers("register", model)
        print(f"✓ Registered model: {model_id}")

    def get_model(self, name: str, version: str) -> Optional[BaseModel]:
        """
        Get model by name and version.

        Args:
            name: Model name
            version: Model version

        Returns:
            Model instance or None
        """
        model_id = f"{name}:{version}"
        return self._models.get(model_id)

    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())

    def predict_with_cache(
        self,
        name: str,
        version: str,
        input_data: Any
    ) -> Any:
        """
        Make prediction with caching.

        Args:
            name: Model name
            version: Model version
            input_data: Input data for prediction

        Returns:
            Prediction result
        """
        model = self.get_model(name, version)
        if not model:
            raise ValueError(f"Model {name}:{version} not found")

        # Create cache key from input data
        cache_key = self._create_cache_key(name, version, input_data)

        # Check cache
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            print(f"✓ Cache hit! (hit_rate={self._cache.hit_rate:.2%})")
            return cached_result

        # Make prediction
        result = model.predict(input_data)

        # Cache result
        self._cache.put(cache_key, result)
        print(f"✓ Prediction cached (hit_rate={self._cache.hit_rate:.2%})")

        return result

    def add_observer(self, observer: 'ModelObserver') -> None:
        """Add an observer."""
        self._observers.append(observer)

    def _notify_observers(self, event: str, model: BaseModel) -> None:
        """Notify all observers of an event."""
        for observer in self._observers:
            observer.update(event, model)

    def _create_cache_key(
        self,
        name: str,
        version: str,
        input_data: Any
    ) -> str:
        """Create cache key from input data."""
        data_str = str(input_data)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"{name}:{version}:{data_hash}"

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "capacity": self._cache.capacity,
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "hit_rate": self._cache.hit_rate
        }

    def __len__(self) -> int:
        """Get number of registered models."""
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        """Check if model is registered."""
        return model_id in self._models

    def __iter__(self):
        """Iterate over models."""
        return iter(self._models.values())


# ===========================
# Observer Pattern
# ===========================

class ModelObserver(ABC):
    """Abstract observer for model events."""

    @abstractmethod
    def update(self, event: str, model: BaseModel) -> None:
        """Handle model event."""
        pass


class MetricsLogger(ModelObserver):
    """Observer that logs model metrics."""

    def __init__(self):
        self.log: List[Dict[str, Any]] = []

    def update(self, event: str, model: BaseModel) -> None:
        """Log model event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "model": str(model),
            "metrics": model.metrics if model.is_trained else {}
        }
        self.log.append(entry)
        print(f"[MetricsLogger] {event}: {model.name} v{model.version}")


# ===========================
# Factory Pattern
# ===========================

class ModelFactory:
    """
    Factory for creating models.

    Demonstrates Factory design pattern.
    """

    _model_types = {
        "neural_network": NeuralNetwork,
        "random_forest": RandomForest
    }

    @classmethod
    def create_model(
        cls,
        model_type: str,
        name: str,
        version: str,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            name: Model name
            version: Model version
            **kwargs: Additional model-specific parameters

        Returns:
            Model instance

        Raises:
            ValueError: If model type is unknown
        """
        if model_type not in cls._model_types:
            available = ", ".join(cls._model_types.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {available}"
            )

        model_class = cls._model_types[model_type]
        return model_class(name, version, **kwargs)

    @classmethod
    def register_model_type(
        cls,
        type_name: str,
        model_class: type
    ) -> None:
        """Register a new model type."""
        cls._model_types[type_name] = model_class


# ===========================
# Model Metrics Tracker
# ===========================

class ModelMetrics:
    """Track and compare model metrics."""

    def __init__(self, model: BaseModel):
        """Initialize metrics tracker."""
        self.model = model
        self.history: List[Dict[str, Any]] = []

    def record_metrics(self, metrics: Dict[str, float]) -> None:
        """Record model metrics."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy()
        }
        self.history.append(entry)

    def get_latest_metrics(self) -> Optional[Dict[str, float]]:
        """Get most recent metrics."""
        if not self.history:
            return None
        return self.history[-1]["metrics"]

    def get_metric_trend(self, metric_name: str) -> List[float]:
        """Get historical trend for a metric."""
        return [
            entry["metrics"].get(metric_name, 0.0)
            for entry in self.history
        ]

    def __str__(self) -> str:
        """String representation."""
        latest = self.get_latest_metrics()
        if not latest:
            return f"ModelMetrics({self.model.name}: No metrics)"

        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in latest.items()
        )
        return f"ModelMetrics({self.model.name}: {metrics_str})"


# ===========================
# Context Manager
# ===========================

class ModelContext:
    """
    Context manager for model operations.

    Ensures proper resource management.
    """

    def __init__(self, model: BaseModel, operation: str):
        """
        Initialize context manager.

        Args:
            model: Model to manage
            operation: Operation description
        """
        self.model = model
        self.operation = operation
        self.start_time = None

    def __enter__(self) -> BaseModel:
        """Enter context."""
        print(f"Starting {self.operation} for {self.model.name}")
        self.start_time = time.time()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        duration = time.time() - self.start_time

        if exc_type is not None:
            print(
                f"✗ {self.operation} failed for {self.model.name} "
                f"after {duration:.2f}s: {exc_val}"
            )
            return False

        print(
            f"✓ {self.operation} completed for {self.model.name} "
            f"in {duration:.2f}s"
        )
        return True


# ===========================
# Model Comparator
# ===========================

class ModelComparator:
    """Compare multiple models."""

    @staticmethod
    def compare_metrics(
        models: List[BaseModel],
        metric_name: str
    ) -> List[Tuple[str, float]]:
        """
        Compare models by a specific metric.

        Args:
            models: List of models to compare
            metric_name: Metric to compare

        Returns:
            List of (model_name, metric_value) tuples, sorted by metric
        """
        results = []
        for model in models:
            if model.is_trained and metric_name in model.metrics:
                model_id = f"{model.name} v{model.version}"
                metric_value = model.metrics[metric_name]
                results.append((model_id, metric_value))

        # Sort by metric value (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def best_model(
        models: List[BaseModel],
        metric_name: str = "accuracy"
    ) -> Optional[BaseModel]:
        """
        Find best model by metric.

        Args:
            models: List of models
            metric_name: Metric to use for comparison

        Returns:
            Best model or None
        """
        trained_models = [m for m in models if m.is_trained]
        if not trained_models:
            return None

        return max(
            trained_models,
            key=lambda m: m.metrics.get(metric_name, 0.0)
        )


# ===========================
# Demo/Testing Function
# ===========================

def demo():
    """Demonstrate the ML Model Manager."""
    print("=" * 60)
    print("ML Model Manager Demo")
    print("=" * 60)

    # Create registry (Singleton)
    registry = ModelRegistry()

    # Add observer
    logger = MetricsLogger()
    registry.add_observer(logger)

    # Create models using Factory
    print("\n1. Creating models with Factory...")
    nn_model = ModelFactory.create_model(
        "neural_network",
        name="ImageClassifier",
        version="1.0.0",
        layers=[784, 128, 64, 10]
    )

    rf_model = ModelFactory.create_model(
        "random_forest",
        name="TextClassifier",
        version="1.0.0",
        n_trees=150
    )

    # Register models
    print("\n2. Registering models...")
    registry.register(nn_model)
    registry.register(rf_model)

    # Train models with context manager
    print("\n3. Training models...")
    with ModelContext(nn_model, "training") as model:
        model.train("training_data")

    with ModelContext(rf_model, "training") as model:
        model.train("training_data")

    # Make predictions with caching
    print("\n4. Making predictions with caching...")
    test_input = [1.0, 2.0, 3.0]

    # First prediction (cache miss)
    result1 = registry.predict_with_cache(
        "ImageClassifier", "1.0.0", test_input
    )
    print(f"Result: {result1}")

    # Second prediction (cache hit)
    result2 = registry.predict_with_cache(
        "ImageClassifier", "1.0.0", test_input
    )
    print(f"Result: {result2}")

    # Cache statistics
    print("\n5. Cache statistics:")
    stats = registry.cache_stats
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Compare models
    print("\n6. Comparing models...")
    models = [nn_model, rf_model]
    comparison = ModelComparator.compare_metrics(models, "accuracy")
    for model_name, accuracy in comparison:
        print(f"   {model_name}: {accuracy:.4f}")

    best = ModelComparator.best_model(models, "accuracy")
    print(f"\n   Best model: {best}")

    # Display observer log
    print("\n7. Observer log:")
    for entry in logger.log:
        print(f"   [{entry['timestamp']}] {entry['event']}: {entry['model']}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
