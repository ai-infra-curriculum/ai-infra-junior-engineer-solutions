#!/usr/bin/env python3
"""
Test suite for OOP exercise.

Tests all aspects of the ML Model Manager implementation.
"""

import pytest
import sys
from pathlib import Path

# Add solutions directory to path
solutions_dir = Path(__file__).parent.parent / "solutions"
sys.path.insert(0, str(solutions_dir))

from model_manager import (
    BaseModel,
    NeuralNetwork,
    RandomForest,
    LRUCache,
    ModelRegistry,
    ModelFactory,
    MetricsLogger,
    ModelMetrics,
    ModelContext,
    ModelComparator
)


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def neural_network():
    """Create a neural network model."""
    return NeuralNetwork("TestNN", "1.0.0", layers=[10, 5, 2])


@pytest.fixture
def random_forest():
    """Create a random forest model."""
    return RandomForest("TestRF", "1.0.0", n_trees=50)


@pytest.fixture
def trained_neural_network(neural_network):
    """Create a trained neural network."""
    neural_network.train("dummy_data")
    return neural_network


@pytest.fixture
def trained_random_forest(random_forest):
    """Create a trained random forest."""
    random_forest.train("dummy_data")
    return random_forest


@pytest.fixture
def lru_cache():
    """Create an LRU cache."""
    return LRUCache(capacity=3)


@pytest.fixture
def model_registry():
    """Create a fresh model registry."""
    # Reset singleton
    ModelRegistry._instance = None
    return ModelRegistry()


# ===========================
# Base Model Tests
# ===========================

class TestBaseModel:
    """Test BaseModel abstract class."""

    def test_model_initialization(self, neural_network):
        """Test model is initialized correctly."""
        assert neural_network.name == "TestNN"
        assert neural_network.version == "1.0.0"
        assert not neural_network.is_trained
        assert neural_network.metrics == {}

    def test_model_properties(self, neural_network):
        """Test model properties are read-only."""
        assert neural_network.name == "TestNN"
        assert neural_network.version == "1.0.0"

    def test_model_string_representation(self, neural_network):
        """Test __str__ method."""
        assert "TestNN" in str(neural_network)
        assert "1.0.0" in str(neural_network)
        assert "untrained" in str(neural_network)

    def test_model_repr(self, neural_network):
        """Test __repr__ method."""
        repr_str = repr(neural_network)
        assert "BaseModel" in repr_str
        assert "TestNN" in repr_str


# ===========================
# Concrete Model Tests
# ===========================

class TestNeuralNetwork:
    """Test NeuralNetwork class."""

    def test_initialization(self):
        """Test neural network initialization."""
        nn = NeuralNetwork("MyNN", "2.0.0", layers=[100, 50, 10])
        assert nn.name == "MyNN"
        assert nn.version == "2.0.0"
        assert nn.layers == [100, 50, 10]
        assert not nn.is_trained

    def test_training(self, neural_network):
        """Test model training."""
        assert not neural_network.is_trained

        neural_network.train("training_data")

        assert neural_network.is_trained
        assert "accuracy" in neural_network.metrics
        assert "loss" in neural_network.metrics

    def test_prediction_before_training(self, neural_network):
        """Test prediction fails before training."""
        with pytest.raises(RuntimeError, match="must be trained"):
            neural_network.predict([1, 2, 3])

    def test_prediction_after_training(self, trained_neural_network):
        """Test prediction succeeds after training."""
        result = trained_neural_network.predict([1, 2, 3])
        assert result is not None
        assert isinstance(result, list)

    def test_save_and_load(self, trained_neural_network, tmp_path):
        """Test model save and load."""
        # Save model
        model_path = tmp_path / "model.pkl"
        trained_neural_network.save(str(model_path))
        assert model_path.exists()

        # Load into new model
        new_model = NeuralNetwork("Temp", "0.0.0", layers=[1])
        new_model.load(str(model_path))

        assert new_model.name == trained_neural_network.name
        assert new_model.version == trained_neural_network.version
        assert new_model.is_trained == trained_neural_network.is_trained
        assert new_model.layers == trained_neural_network.layers


class TestRandomForest:
    """Test RandomForest class."""

    def test_initialization(self):
        """Test random forest initialization."""
        rf = RandomForest("MyRF", "1.0.0", n_trees=200)
        assert rf.name == "MyRF"
        assert rf.n_trees == 200

    def test_training(self, random_forest):
        """Test model training."""
        random_forest.train("training_data")

        assert random_forest.is_trained
        assert "accuracy" in random_forest.metrics
        assert "f1_score" in random_forest.metrics

    def test_prediction(self, trained_random_forest):
        """Test prediction."""
        result = trained_random_forest.predict([1, 2, 3])
        assert result is not None


# ===========================
# LRU Cache Tests
# ===========================

class TestLRUCache:
    """Test LRU cache implementation."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = LRUCache(capacity=5)
        assert len(cache) == 0
        assert cache.capacity == 5

    def test_put_and_get(self, lru_cache):
        """Test basic put and get operations."""
        lru_cache.put("key1", "value1")
        assert len(lru_cache) == 1
        assert lru_cache.get("key1") == "value1"

    def test_cache_miss(self, lru_cache):
        """Test cache miss."""
        result = lru_cache.get("nonexistent")
        assert result is None
        assert lru_cache.misses == 1

    def test_cache_hit(self, lru_cache):
        """Test cache hit."""
        lru_cache.put("key1", "value1")
        lru_cache.get("key1")
        assert lru_cache.hits == 1

    def test_eviction(self, lru_cache):
        """Test LRU eviction."""
        # Fill cache to capacity (3)
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2")
        lru_cache.put("key3", "value3")
        assert len(lru_cache) == 3

        # Add one more item, should evict key1
        lru_cache.put("key4", "value4")
        assert len(lru_cache) == 3
        assert "key1" not in lru_cache
        assert "key4" in lru_cache

    def test_lru_order(self, lru_cache):
        """Test LRU ordering."""
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2")
        lru_cache.put("key3", "value3")

        # Access key1, making it most recently used
        lru_cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        lru_cache.put("key4", "value4")
        assert "key2" not in lru_cache
        assert "key1" in lru_cache

    def test_hit_rate(self, lru_cache):
        """Test hit rate calculation."""
        lru_cache.put("key1", "value1")

        lru_cache.get("key1")  # hit
        lru_cache.get("key2")  # miss
        lru_cache.get("key1")  # hit

        assert lru_cache.hits == 2
        assert lru_cache.misses == 1
        assert lru_cache.hit_rate == 2 / 3

    def test_clear(self, lru_cache):
        """Test cache clear."""
        lru_cache.put("key1", "value1")
        lru_cache.put("key2", "value2")
        lru_cache.get("key1")

        lru_cache.clear()

        assert len(lru_cache) == 0
        assert lru_cache.hits == 0
        assert lru_cache.misses == 0

    def test_contains(self, lru_cache):
        """Test __contains__ magic method."""
        lru_cache.put("key1", "value1")
        assert "key1" in lru_cache
        assert "key2" not in lru_cache

    def test_str(self, lru_cache):
        """Test __str__ method."""
        lru_cache.put("key1", "value1")
        cache_str = str(lru_cache)
        assert "LRUCache" in cache_str
        assert "capacity=3" in cache_str


# ===========================
# Model Registry Tests
# ===========================

class TestModelRegistry:
    """Test ModelRegistry singleton."""

    def test_singleton_pattern(self):
        """Test only one instance is created."""
        # Reset singleton
        ModelRegistry._instance = None

        registry1 = ModelRegistry()
        registry2 = ModelRegistry()

        assert registry1 is registry2

    def test_register_model(self, model_registry, neural_network):
        """Test model registration."""
        model_registry.register(neural_network)

        assert "TestNN:1.0.0" in model_registry.list_models()
        assert len(model_registry) == 1

    def test_get_model(self, model_registry, neural_network):
        """Test retrieving model."""
        model_registry.register(neural_network)

        retrieved = model_registry.get_model("TestNN", "1.0.0")

        assert retrieved is neural_network

    def test_get_nonexistent_model(self, model_registry):
        """Test retrieving nonexistent model."""
        result = model_registry.get_model("NonExistent", "1.0.0")
        assert result is None

    def test_predict_with_cache(
        self,
        model_registry,
        trained_neural_network
    ):
        """Test prediction with caching."""
        model_registry.register(trained_neural_network)

        input_data = [1.0, 2.0, 3.0]

        # First prediction (cache miss)
        result1 = model_registry.predict_with_cache(
            "TestNN", "1.0.0", input_data
        )

        # Second prediction (cache hit)
        result2 = model_registry.predict_with_cache(
            "TestNN", "1.0.0", input_data
        )

        assert result1 == result2
        assert model_registry.cache_stats["hits"] == 1

    def test_predict_unregistered_model(self, model_registry):
        """Test prediction for unregistered model."""
        with pytest.raises(ValueError, match="not found"):
            model_registry.predict_with_cache(
                "NonExistent", "1.0.0", [1, 2, 3]
            )

    def test_cache_stats(self, model_registry, trained_neural_network):
        """Test cache statistics."""
        model_registry.register(trained_neural_network)

        stats = model_registry.cache_stats

        assert "size" in stats
        assert "capacity" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_contains(self, model_registry, neural_network):
        """Test __contains__ method."""
        model_registry.register(neural_network)
        assert "TestNN:1.0.0" in model_registry

    def test_iteration(self, model_registry, neural_network, random_forest):
        """Test __iter__ method."""
        model_registry.register(neural_network)
        model_registry.register(random_forest)

        models = list(model_registry)
        assert len(models) == 2
        assert neural_network in models
        assert random_forest in models


# ===========================
# Observer Pattern Tests
# ===========================

class TestObserver:
    """Test Observer pattern."""

    def test_observer_notification(self, model_registry, neural_network):
        """Test observer receives notifications."""
        logger = MetricsLogger()
        model_registry.add_observer(logger)

        model_registry.register(neural_network)

        assert len(logger.log) == 1
        assert logger.log[0]["event"] == "register"
        assert "TestNN" in logger.log[0]["model"]


# ===========================
# Factory Pattern Tests
# ===========================

class TestModelFactory:
    """Test Factory pattern."""

    def test_create_neural_network(self):
        """Test creating neural network via factory."""
        model = ModelFactory.create_model(
            "neural_network",
            name="FactoryNN",
            version="1.0.0",
            layers=[100, 50, 10]
        )

        assert isinstance(model, NeuralNetwork)
        assert model.name == "FactoryNN"
        assert model.layers == [100, 50, 10]

    def test_create_random_forest(self):
        """Test creating random forest via factory."""
        model = ModelFactory.create_model(
            "random_forest",
            name="FactoryRF",
            version="1.0.0",
            n_trees=200
        )

        assert isinstance(model, RandomForest)
        assert model.name == "FactoryRF"
        assert model.n_trees == 200

    def test_create_unknown_model(self):
        """Test creating unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model(
                "unknown_type",
                name="Test",
                version="1.0.0"
            )

    def test_register_new_model_type(self):
        """Test registering new model type."""
        # Create custom model class
        class CustomModel(BaseModel):
            def train(self, data):
                pass
            def predict(self, input_data):
                pass
            def save(self, path):
                pass
            def load(self, path):
                pass

        # Register it
        ModelFactory.register_model_type("custom", CustomModel)

        # Create it
        model = ModelFactory.create_model(
            "custom",
            name="CustomTest",
            version="1.0.0"
        )

        assert isinstance(model, CustomModel)


# ===========================
# Model Metrics Tests
# ===========================

class TestModelMetrics:
    """Test ModelMetrics class."""

    def test_initialization(self, neural_network):
        """Test metrics tracker initialization."""
        tracker = ModelMetrics(neural_network)
        assert tracker.model is neural_network
        assert len(tracker.history) == 0

    def test_record_metrics(self, neural_network):
        """Test recording metrics."""
        tracker = ModelMetrics(neural_network)

        metrics = {"accuracy": 0.95, "loss": 0.05}
        tracker.record_metrics(metrics)

        assert len(tracker.history) == 1
        assert tracker.history[0]["metrics"] == metrics

    def test_get_latest_metrics(self, neural_network):
        """Test getting latest metrics."""
        tracker = ModelMetrics(neural_network)

        tracker.record_metrics({"accuracy": 0.90})
        tracker.record_metrics({"accuracy": 0.95})

        latest = tracker.get_latest_metrics()
        assert latest["accuracy"] == 0.95

    def test_get_metric_trend(self, neural_network):
        """Test getting metric trend."""
        tracker = ModelMetrics(neural_network)

        tracker.record_metrics({"accuracy": 0.90})
        tracker.record_metrics({"accuracy": 0.92})
        tracker.record_metrics({"accuracy": 0.95})

        trend = tracker.get_metric_trend("accuracy")
        assert trend == [0.90, 0.92, 0.95]

    def test_str_with_metrics(self, trained_neural_network):
        """Test __str__ with metrics."""
        tracker = ModelMetrics(trained_neural_network)
        tracker.record_metrics(trained_neural_network.metrics)

        metrics_str = str(tracker)
        assert "ModelMetrics" in metrics_str
        assert "TestNN" in metrics_str


# ===========================
# Context Manager Tests
# ===========================

class TestModelContext:
    """Test ModelContext context manager."""

    def test_context_manager_success(self, neural_network):
        """Test successful operation."""
        with ModelContext(neural_network, "training") as model:
            model.train("data")

        assert neural_network.is_trained

    def test_context_manager_failure(self, neural_network):
        """Test failed operation."""
        try:
            with ModelContext(neural_network, "prediction"):
                neural_network.predict([1, 2, 3])
        except RuntimeError:
            pass  # Expected to fail (not trained)

    def test_context_manager_timing(self, neural_network):
        """Test timing measurement."""
        import time

        context = ModelContext(neural_network, "test")

        with context as model:
            time.sleep(0.01)

        # Should have recorded some time
        assert context.start_time is not None


# ===========================
# Model Comparator Tests
# ===========================

class TestModelComparator:
    """Test ModelComparator class."""

    def test_compare_metrics(
        self,
        trained_neural_network,
        trained_random_forest
    ):
        """Test comparing models by metric."""
        models = [trained_neural_network, trained_random_forest]

        comparison = ModelComparator.compare_metrics(models, "accuracy")

        assert len(comparison) == 2
        # Should be sorted by accuracy (descending)
        assert comparison[0][1] >= comparison[1][1]

    def test_best_model(
        self,
        trained_neural_network,
        trained_random_forest
    ):
        """Test finding best model."""
        models = [trained_neural_network, trained_random_forest]

        best = ModelComparator.best_model(models, "accuracy")

        assert best is not None
        # Should be neural network (accuracy 0.95 > 0.92)
        assert best is trained_neural_network

    def test_best_model_no_trained_models(self, neural_network):
        """Test best model with no trained models."""
        models = [neural_network]

        best = ModelComparator.best_model(models)

        assert best is None


# ===========================
# Integration Tests
# ===========================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, model_registry):
        """Test complete ML workflow."""
        # 1. Create models with factory
        nn = ModelFactory.create_model(
            "neural_network",
            name="IntegrationNN",
            version="1.0.0",
            layers=[10, 5, 2]
        )

        # 2. Register model
        model_registry.register(nn)

        # 3. Train model
        with ModelContext(nn, "training") as model:
            model.train("training_data")

        # 4. Make predictions with caching
        input_data = [1.0, 2.0, 3.0]
        result = model_registry.predict_with_cache(
            "IntegrationNN", "1.0.0", input_data
        )

        assert result is not None

    def test_multiple_model_comparison(self):
        """Test comparing multiple models."""
        # Create and train multiple models
        models = []
        for i in range(3):
            model = ModelFactory.create_model(
                "neural_network",
                name=f"Model{i}",
                version="1.0.0",
                layers=[10, 5, 2]
            )
            model.train("data")
            models.append(model)

        # Compare models
        best = ModelComparator.best_model(models, "accuracy")
        assert best is not None
        assert best in models


# ===========================
# Run Tests
# ===========================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
