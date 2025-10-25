#!/usr/bin/env python3
"""
Design Patterns in Python

Comprehensive examples of common design patterns used in ML infrastructure.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading


# ===========================
# 1. Singleton Pattern
# ===========================

class ConfigManager:
    """
    Singleton configuration manager.

    Ensures only one instance exists across the application.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Create or return existing instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration (only once)."""
        if not hasattr(self, '_initialized'):
            self.config: Dict[str, Any] = {}
            self._initialized = True

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)


def demo_singleton():
    """Demonstrate Singleton pattern."""
    print("\n" + "=" * 60)
    print("Singleton Pattern Demo")
    print("=" * 60)

    # Create two instances
    config1 = ConfigManager()
    config2 = ConfigManager()

    # They are the same object
    print(f"config1 is config2: {config1 is config2}")

    # Setting value in one affects the other
    config1.set("api_key", "secret123")
    print(f"config2.get('api_key'): {config2.get('api_key')}")


# ===========================
# 2. Factory Pattern
# ===========================

class DataLoader(ABC):
    """Abstract data loader."""

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load data from path."""
        pass


class CSVLoader(DataLoader):
    """Load CSV data."""

    def load(self, path: str) -> List[List[str]]:
        """Load CSV file."""
        print(f"Loading CSV from {path}")
        return [["col1", "col2"], ["val1", "val2"]]


class JSONLoader(DataLoader):
    """Load JSON data."""

    def load(self, path: str) -> Dict:
        """Load JSON file."""
        print(f"Loading JSON from {path}")
        return {"key": "value"}


class ParquetLoader(DataLoader):
    """Load Parquet data."""

    def load(self, path: str) -> Any:
        """Load Parquet file."""
        print(f"Loading Parquet from {path}")
        return {"data": "parquet_data"}


class DataLoaderFactory:
    """Factory for creating data loaders."""

    _loaders = {
        "csv": CSVLoader,
        "json": JSONLoader,
        "parquet": ParquetLoader
    }

    @classmethod
    def create_loader(cls, file_type: str) -> DataLoader:
        """
        Create appropriate data loader.

        Args:
            file_type: Type of file (csv, json, parquet)

        Returns:
            DataLoader instance
        """
        loader_class = cls._loaders.get(file_type.lower())
        if not loader_class:
            raise ValueError(f"Unknown file type: {file_type}")
        return loader_class()

    @classmethod
    def register_loader(cls, file_type: str, loader_class: type) -> None:
        """Register new loader type."""
        cls._loaders[file_type.lower()] = loader_class


def demo_factory():
    """Demonstrate Factory pattern."""
    print("\n" + "=" * 60)
    print("Factory Pattern Demo")
    print("=" * 60)

    # Create different loaders
    for file_type in ["csv", "json", "parquet"]:
        loader = DataLoaderFactory.create_loader(file_type)
        data = loader.load(f"data.{file_type}")
        print(f"Loaded: {data}")


# ===========================
# 3. Observer Pattern
# ===========================

class TrainingObserver(ABC):
    """Abstract observer for training events."""

    @abstractmethod
    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Handle training update."""
        pass


class ConsoleLogger(TrainingObserver):
    """Log training progress to console."""

    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log to console."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch}: {metrics_str}")


class FileLogger(TrainingObserver):
    """Log training progress to file."""

    def __init__(self, filename: str):
        self.filename = filename
        self.logs: List[str] = []

    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log to file."""
        log_entry = f"[{datetime.now()}] Epoch {epoch}: {metrics}"
        self.logs.append(log_entry)
        print(f"Logged to {self.filename}: Epoch {epoch}")


class MetricsTracker(TrainingObserver):
    """Track metrics history."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Track metrics."""
        self.history.append({
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now()
        })


class Trainer:
    """Training class that notifies observers."""

    def __init__(self):
        self._observers: List[TrainingObserver] = []

    def attach(self, observer: TrainingObserver) -> None:
        """Attach an observer."""
        self._observers.append(observer)

    def detach(self, observer: TrainingObserver) -> None:
        """Detach an observer."""
        self._observers.remove(observer)

    def _notify(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Notify all observers."""
        for observer in self._observers:
            observer.update(epoch, metrics)

    def train(self, epochs: int = 3) -> None:
        """Train model and notify observers."""
        for epoch in range(1, epochs + 1):
            # Simulate training
            metrics = {
                "loss": 1.0 / epoch,
                "accuracy": 0.5 + (epoch * 0.1)
            }
            self._notify(epoch, metrics)


def demo_observer():
    """Demonstrate Observer pattern."""
    print("\n" + "=" * 60)
    print("Observer Pattern Demo")
    print("=" * 60)

    trainer = Trainer()

    # Attach observers
    trainer.attach(ConsoleLogger())
    trainer.attach(FileLogger("training.log"))
    tracker = MetricsTracker()
    trainer.attach(tracker)

    # Train
    trainer.train(epochs=3)

    print(f"\nTracked {len(tracker.history)} epochs")


# ===========================
# 4. Strategy Pattern
# ===========================

class OptimizationStrategy(ABC):
    """Abstract optimization strategy."""

    @abstractmethod
    def optimize(self, parameters: List[float], gradient: List[float]) -> List[float]:
        """Apply optimization step."""
        pass


class SGD(OptimizationStrategy):
    """Stochastic Gradient Descent."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def optimize(self, parameters: List[float], gradient: List[float]) -> List[float]:
        """Apply SGD update."""
        print(f"Applying SGD (lr={self.learning_rate})")
        return [p - self.learning_rate * g for p, g in zip(parameters, gradient)]


class Adam(OptimizationStrategy):
    """Adam optimizer."""

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def optimize(self, parameters: List[float], gradient: List[float]) -> List[float]:
        """Apply Adam update."""
        print(f"Applying Adam (lr={self.learning_rate})")
        # Simplified Adam update
        return [p - self.learning_rate * g for p, g in zip(parameters, gradient)]


class Optimizer:
    """Optimizer that uses a strategy."""

    def __init__(self, strategy: OptimizationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OptimizationStrategy) -> None:
        """Change optimization strategy."""
        self._strategy = strategy

    def step(self, parameters: List[float], gradient: List[float]) -> List[float]:
        """Perform optimization step."""
        return self._strategy.optimize(parameters, gradient)


def demo_strategy():
    """Demonstrate Strategy pattern."""
    print("\n" + "=" * 60)
    print("Strategy Pattern Demo")
    print("=" * 60)

    parameters = [1.0, 2.0, 3.0]
    gradient = [0.1, 0.2, 0.3]

    # Use SGD strategy
    optimizer = Optimizer(SGD(learning_rate=0.01))
    new_params = optimizer.step(parameters, gradient)
    print(f"New parameters: {new_params}")

    # Switch to Adam strategy
    optimizer.set_strategy(Adam(learning_rate=0.001))
    new_params = optimizer.step(parameters, gradient)
    print(f"New parameters: {new_params}")


# ===========================
# 5. Builder Pattern
# ===========================

class ModelConfig:
    """Model configuration."""

    def __init__(self):
        self.name: Optional[str] = None
        self.layers: List[int] = []
        self.activation: str = "relu"
        self.optimizer: str = "adam"
        self.learning_rate: float = 0.001
        self.batch_size: int = 32
        self.epochs: int = 10

    def __str__(self) -> str:
        return (
            f"ModelConfig(\n"
            f"  name={self.name},\n"
            f"  layers={self.layers},\n"
            f"  activation={self.activation},\n"
            f"  optimizer={self.optimizer},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  batch_size={self.batch_size},\n"
            f"  epochs={self.epochs}\n"
            f")"
        )


class ModelConfigBuilder:
    """Builder for ModelConfig."""

    def __init__(self):
        self._config = ModelConfig()

    def set_name(self, name: str) -> 'ModelConfigBuilder':
        """Set model name."""
        self._config.name = name
        return self

    def add_layer(self, size: int) -> 'ModelConfigBuilder':
        """Add a layer."""
        self._config.layers.append(size)
        return self

    def set_activation(self, activation: str) -> 'ModelConfigBuilder':
        """Set activation function."""
        self._config.activation = activation
        return self

    def set_optimizer(self, optimizer: str) -> 'ModelConfigBuilder':
        """Set optimizer."""
        self._config.optimizer = optimizer
        return self

    def set_learning_rate(self, lr: float) -> 'ModelConfigBuilder':
        """Set learning rate."""
        self._config.learning_rate = lr
        return self

    def set_batch_size(self, batch_size: int) -> 'ModelConfigBuilder':
        """Set batch size."""
        self._config.batch_size = batch_size
        return self

    def set_epochs(self, epochs: int) -> 'ModelConfigBuilder':
        """Set number of epochs."""
        self._config.epochs = epochs
        return self

    def build(self) -> ModelConfig:
        """Build and return configuration."""
        return self._config


def demo_builder():
    """Demonstrate Builder pattern."""
    print("\n" + "=" * 60)
    print("Builder Pattern Demo")
    print("=" * 60)

    # Build configuration with fluent interface
    config = (
        ModelConfigBuilder()
        .set_name("ImageClassifier")
        .add_layer(784)
        .add_layer(128)
        .add_layer(64)
        .add_layer(10)
        .set_activation("relu")
        .set_optimizer("adam")
        .set_learning_rate(0.001)
        .set_batch_size(64)
        .set_epochs(20)
        .build()
    )

    print(config)


# ===========================
# 6. Decorator Pattern
# ===========================

class Model(ABC):
    """Abstract model interface."""

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make prediction."""
        pass


class SimpleModel(Model):
    """Simple model implementation."""

    def predict(self, input_data: Any) -> Any:
        """Make prediction."""
        return f"prediction for {input_data}"


class ModelDecorator(Model):
    """Base decorator for models."""

    def __init__(self, model: Model):
        self._model = model

    def predict(self, input_data: Any) -> Any:
        """Delegate to wrapped model."""
        return self._model.predict(input_data)


class LoggingDecorator(ModelDecorator):
    """Add logging to model predictions."""

    def predict(self, input_data: Any) -> Any:
        """Log before and after prediction."""
        print(f"[LOG] Making prediction for: {input_data}")
        result = self._model.predict(input_data)
        print(f"[LOG] Prediction result: {result}")
        return result


class CachingDecorator(ModelDecorator):
    """Add caching to model predictions."""

    def __init__(self, model: Model):
        super().__init__(model)
        self._cache: Dict[str, Any] = {}

    def predict(self, input_data: Any) -> Any:
        """Check cache before prediction."""
        cache_key = str(input_data)

        if cache_key in self._cache:
            print(f"[CACHE] Cache hit for: {input_data}")
            return self._cache[cache_key]

        print(f"[CACHE] Cache miss for: {input_data}")
        result = self._model.predict(input_data)
        self._cache[cache_key] = result
        return result


class TimingDecorator(ModelDecorator):
    """Add timing to model predictions."""

    def predict(self, input_data: Any) -> Any:
        """Time the prediction."""
        import time
        start = time.time()
        result = self._model.predict(input_data)
        duration = time.time() - start
        print(f"[TIMING] Prediction took {duration:.4f}s")
        return result


def demo_decorator():
    """Demonstrate Decorator pattern."""
    print("\n" + "=" * 60)
    print("Decorator Pattern Demo")
    print("=" * 60)

    # Create base model
    model = SimpleModel()

    # Wrap with decorators
    model = LoggingDecorator(model)
    model = CachingDecorator(model)
    model = TimingDecorator(model)

    # Make predictions
    print("\nFirst prediction:")
    model.predict("input1")

    print("\nSecond prediction (same input - should hit cache):")
    model.predict("input1")

    print("\nThird prediction (different input):")
    model.predict("input2")


# ===========================
# 7. Adapter Pattern
# ===========================

class ThirdPartyModel:
    """Third-party model with different interface."""

    def make_inference(self, data: str) -> str:
        """Make inference (different method name)."""
        return f"third_party_result: {data}"


class ModelInterface(ABC):
    """Standard model interface."""

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make prediction."""
        pass


class ThirdPartyModelAdapter(ModelInterface):
    """Adapter for third-party model."""

    def __init__(self, third_party_model: ThirdPartyModel):
        self._model = third_party_model

    def predict(self, input_data: Any) -> Any:
        """Adapt to standard interface."""
        # Convert input format if needed
        adapted_input = str(input_data)

        # Call third-party method
        result = self._model.make_inference(adapted_input)

        # Convert output format if needed
        return result


def demo_adapter():
    """Demonstrate Adapter pattern."""
    print("\n" + "=" * 60)
    print("Adapter Pattern Demo")
    print("=" * 60)

    # Create third-party model
    third_party = ThirdPartyModel()

    # Wrap with adapter
    adapted_model = ThirdPartyModelAdapter(third_party)

    # Use standard interface
    result = adapted_model.predict("test_input")
    print(f"Result: {result}")


# ===========================
# Main Demo
# ===========================

def main():
    """Run all design pattern demos."""
    print("\n" + "=" * 70)
    print("Design Patterns in Python for ML Infrastructure")
    print("=" * 70)

    demo_singleton()
    demo_factory()
    demo_observer()
    demo_strategy()
    demo_builder()
    demo_decorator()
    demo_adapter()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
