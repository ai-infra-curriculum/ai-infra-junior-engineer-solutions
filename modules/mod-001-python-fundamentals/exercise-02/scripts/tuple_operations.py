#!/usr/bin/env python3
"""
Tuple operations for immutable ML data.

Demonstrates tuple usage for fixed data, named tuples, and multiple return values.
"""

from collections import namedtuple
from typing import Tuple


def main():
    """Demonstrate tuple operations for immutable data."""
    print("=" * 60)
    print("Tuple Operations for Immutable Data")
    print("=" * 60)
    print()

    # Model metadata (immutable)
    model_metadata = ("ResNet50", "1.0.0", "2024-10-30", 0.92)
    model_name, version, date, accuracy = model_metadata  # Unpacking

    print(f"Model: {model_name}")
    print(f"Version: {version}")
    print(f"Released: {date}")
    print(f"Accuracy: {accuracy}")
    print()

    # Tuple of tuples: training history
    training_history = (
        (1, 0.85, 0.45),   # (epoch, accuracy, loss)
        (2, 0.88, 0.32),
        (3, 0.91, 0.25),
        (4, 0.92, 0.20),
        (5, 0.93, 0.18)
    )

    print("Training History:")
    for epoch, acc, loss in training_history:
        print(f"  Epoch {epoch}: Accuracy={acc:.2f}, Loss={loss:.2f}")
    print()

    # Find best epoch
    best_epoch = max(training_history, key=lambda x: x[1])
    print(f"Best epoch: {best_epoch[0]} with accuracy {best_epoch[1]:.2f}")
    print()

    # Named tuples for better readability
    print("=" * 60)
    print("Named Tuples")
    print("=" * 60)
    print()

    ModelConfig = namedtuple('ModelConfig',
                            ['name', 'layers', 'params', 'memory_mb'])

    resnet_config = ModelConfig('ResNet50', 50, 25_500_000, 98)
    vgg_config = ModelConfig('VGG16', 16, 138_000_000, 528)

    print(f"{resnet_config.name}:")
    print(f"  Layers: {resnet_config.layers}")
    print(f"  Parameters: {resnet_config.params:,}")
    print(f"  Memory: {resnet_config.memory_mb} MB")
    print()

    print(f"{vgg_config.name}:")
    print(f"  Layers: {vgg_config.layers}")
    print(f"  Parameters: {vgg_config.params:,}")
    print(f"  Memory: {vgg_config.memory_mb} MB")
    print()

    # Compare memory
    if resnet_config.memory_mb < vgg_config.memory_mb:
        savings = vgg_config.memory_mb - resnet_config.memory_mb
        print(f"{resnet_config.name} saves {savings} MB of memory!")
    print()

    # Tuple as dictionary key (immutable)
    print("=" * 60)
    print("Tuples as Dictionary Keys")
    print("=" * 60)
    print()

    model_performance = {
        ('ResNet50', 'ImageNet'): 0.92,
        ('VGG16', 'ImageNet'): 0.88,
        ('ResNet50', 'CIFAR10'): 0.95,
        ('MobileNet', 'ImageNet'): 0.87,
    }

    print("Model performance by (model, dataset):")
    for (model, dataset), acc in model_performance.items():
        print(f"  {model} on {dataset}: {acc:.2f}")
    print()

    key = ('ResNet50', 'ImageNet')
    accuracy = model_performance[key]
    print(f"Looking up {key}: {accuracy:.2f}")
    print()

    # Return multiple values from function
    print("=" * 60)
    print("Multiple Return Values")
    print("=" * 60)
    print()

    def train_model(epochs: int) -> Tuple[float, float, int, int]:
        """
        Simulate training, return multiple metrics.

        Returns:
            Tuple of (accuracy, loss, time, params)
        """
        final_accuracy = 0.92
        final_loss = 0.15
        training_time = 3600  # seconds
        num_params = 25_500_000

        return final_accuracy, final_loss, training_time, num_params

    # Unpack return values
    acc, loss, time, params = train_model(50)
    print("Training complete:")
    print(f"  Accuracy: {acc:.2f}")
    print(f"  Loss: {loss:.2f}")
    print(f"  Time: {time}s ({time/60:.1f} minutes)")
    print(f"  Parameters: {params:,}")
    print()

    # Immutability demonstration
    print("=" * 60)
    print("Tuple Immutability")
    print("=" * 60)
    print()

    config = ("model_v1", 100, 0.001)
    print(f"Original config: {config}")

    try:
        config[0] = "model_v2"  # This will fail
        print("  ✗ Modified tuple (shouldn't happen)")
    except TypeError as e:
        print(f"  ✓ Tuple is immutable: {type(e).__name__}")

    # To "modify", create a new tuple
    new_config = ("model_v2",) + config[1:]
    print(f"New config: {new_config}")

    print()
    print("✓ Tuple operations demonstration complete")


if __name__ == "__main__":
    main()
