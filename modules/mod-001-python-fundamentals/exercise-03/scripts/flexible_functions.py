#!/usr/bin/env python3
"""
Flexible Functions with *args and **kwargs

Demonstrates variable positional and keyword arguments for flexible
function interfaces in ML applications.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import time


def log_metrics(*args, **kwargs) -> None:
    """
    Log any number of metrics with flexible arguments.

    Accepts both positional metrics (unnamed) and keyword metrics (named).
    This pattern is useful for logging functions where you don't know
    ahead of time which metrics will be logged.

    Args:
        *args: Positional metrics to log
        **kwargs: Named metrics to log

    Examples:
        >>> log_metrics(0.92, 0.15, 0.89)
        >>> log_metrics(accuracy=0.92, loss=0.15, f1=0.89)
        >>> log_metrics(0.92, loss=0.15, f1=0.89)
    """
    print("=" * 60)
    print("Metrics Log")
    print("=" * 60)

    if args:
        print("\nPositional metrics:")
        for i, value in enumerate(args):
            print(f"  Metric {i+1}: {value}")

    if kwargs:
        print("\nNamed metrics:")
        for name, value in kwargs.items():
            print(f"  {name}: {value}")

    print("=" * 60)


def create_model(model_type: str,
                *layers: int,
                activation: str = "relu",
                dropout: float = 0.0,
                **config: Any) -> Dict[str, Any]:
    """
    Create a model configuration with flexible layer specification.

    Demonstrates mixing required positional args, *args, keyword args,
    and **kwargs for maximum flexibility.

    Args:
        model_type: Type of model (e.g., "cnn", "mlp", "rnn")
        *layers: Variable number of layer sizes
        activation: Activation function (default: "relu")
        dropout: Dropout rate (default: 0.0)
        **config: Additional configuration parameters

    Returns:
        Model configuration dictionary

    Examples:
        >>> model = create_model("mlp", 128, 64, 32, dropout=0.3)
        >>> model = create_model("cnn", 64, 128, 256, activation="relu",
        ...                      dropout=0.5, batch_norm=True)
    """
    model_config = {
        "type": model_type,
        "layers": list(layers),  # Convert tuple to list
        "activation": activation,
        "dropout": dropout,
    }

    # Merge any additional config
    model_config.update(config)

    return model_config


def batch_process(data: List[Any],
                 processor_func: Callable,
                 batch_size: int = 32,
                 *processor_args,
                 **processor_kwargs) -> List[Any]:
    """
    Process data in batches using a processor function.

    Forwards additional arguments to the processor function, allowing
    flexible batch processing pipelines.

    Args:
        data: Data to process
        processor_func: Function to apply to each batch
        batch_size: Size of each batch (default: 32)
        *processor_args: Additional positional args for processor
        **processor_kwargs: Additional keyword args for processor

    Returns:
        List of processed results

    Examples:
        >>> def process(batch, multiplier=1):
        ...     return sum(batch) * multiplier
        >>> results = batch_process(range(10), process, 3, multiplier=2)
    """
    results = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        # Call processor with batch and all additional arguments
        result = processor_func(batch, *processor_args, **processor_kwargs)
        results.append(result)

    return results


def augment_image(image: Any,
                 flip_horizontal: bool = False,
                 flip_vertical: bool = False,
                 rotate: Optional[int] = None,
                 brightness: float = 1.0,
                 **transforms: Any) -> Dict[str, Any]:
    """
    Apply image augmentations with flexible transformation options.

    Supports common transformations plus arbitrary custom transforms
    via **kwargs.

    Args:
        image: Input image
        flip_horizontal: Whether to flip horizontally
        flip_vertical: Whether to flip vertically
        rotate: Rotation angle in degrees (None = no rotation)
        brightness: Brightness adjustment factor (1.0 = no change)
        **transforms: Additional custom transformations

    Returns:
        Dictionary with augmentation info

    Examples:
        >>> aug = augment_image("img.jpg", flip_horizontal=True, rotate=90,
        ...                     contrast=1.5, saturation=0.8)
    """
    augmentations = {
        "original": image,
        "transforms_applied": []
    }

    if flip_horizontal:
        augmentations["transforms_applied"].append("flip_h")

    if flip_vertical:
        augmentations["transforms_applied"].append("flip_v")

    if rotate is not None:
        augmentations["transforms_applied"].append(f"rotate_{rotate}")

    if brightness != 1.0:
        augmentations["transforms_applied"].append(f"brightness_{brightness}")

    # Apply any custom transforms
    for transform_name, transform_value in transforms.items():
        augmentations["transforms_applied"].append(
            f"{transform_name}_{transform_value}"
        )

    return augmentations


def merge_configs(*configs: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries with overrides.

    Later configs override earlier ones. Final overrides have highest priority.

    Args:
        *configs: Variable number of config dictionaries to merge
        **overrides: Final overrides (highest priority)

    Returns:
        Merged configuration dictionary

    Examples:
        >>> default = {"lr": 0.001, "batch_size": 32}
        >>> custom = {"batch_size": 64, "epochs": 100}
        >>> merged = merge_configs(default, custom, lr=0.01)
        >>> merged["batch_size"]
        64
    """
    merged = {}

    # Merge all positional config dicts
    for config in configs:
        merged.update(config)

    # Apply final overrides
    merged.update(overrides)

    return merged


def train_model(model_config: Dict[str, Any],
               *callbacks,
               epochs: int = 10,
               verbose: bool = True,
               **training_params) -> Dict[str, Any]:
    """
    Simulate model training with flexible callbacks and parameters.

    Args:
        model_config: Model configuration dictionary
        *callbacks: Variable number of callback functions
        epochs: Number of training epochs
        verbose: Whether to print training progress
        **training_params: Additional training parameters

    Returns:
        Training results dictionary
    """
    if verbose:
        print(f"\nTraining {model_config['type']} model...")
        print(f"Epochs: {epochs}")
        if training_params:
            print(f"Additional params: {training_params}")

    results = {
        "model_type": model_config["type"],
        "epochs_trained": epochs,
        "final_accuracy": 0.92,
        "final_loss": 0.15,
        "callbacks_executed": len(callbacks)
    }

    # Execute callbacks
    for callback in callbacks:
        if callable(callback):
            callback(results)

    return results


def create_pipeline(*steps: str, cache: bool = False,
                   parallel: bool = False, **step_configs: Dict) -> Dict[str, Any]:
    """
    Create a data processing pipeline with flexible steps.

    Args:
        *steps: Pipeline step names
        cache: Whether to cache intermediate results
        parallel: Whether to execute steps in parallel
        **step_configs: Per-step configuration dictionaries

    Returns:
        Pipeline configuration dictionary

    Examples:
        >>> pipeline = create_pipeline("normalize", "augment", "batch",
        ...                           cache=True,
        ...                           normalize={"method": "minmax"},
        ...                           batch={"size": 32})
    """
    pipeline = {
        "steps": list(steps),
        "cache_enabled": cache,
        "parallel_execution": parallel,
        "step_configs": step_configs
    }

    return pipeline


def main():
    """Demonstrate flexible function patterns."""
    print("=" * 70)
    print("Flexible Functions with *args and **kwargs")
    print("=" * 70)
    print()

    # Example 1: Flexible logging
    print("Example 1: Flexible Metric Logging")
    print("-" * 70)
    print("\na) Positional metrics only:")
    log_metrics(0.92, 0.15, 0.89)
    print()

    print("b) Named metrics only:")
    log_metrics(accuracy=0.92, loss=0.15, f1_score=0.89)
    print()

    print("c) Mixed positional and named:")
    log_metrics(0.92, loss=0.15, f1_score=0.89, learning_rate=0.001)
    print()

    # Example 2: Model creation with variable layers
    print("\nExample 2: Flexible Model Creation")
    print("-" * 70)
    model1 = create_model("mlp", 128, 64, 32, dropout=0.3)
    print(f"Model 1 (simple):")
    print(f"  {model1}")

    model2 = create_model("cnn", 64, 128, 256, activation="relu",
                         dropout=0.5, batch_norm=True, pool_size=2)
    print(f"\nModel 2 (with extra config):")
    print(f"  {model2}")

    model3 = create_model("rnn", 256, 128, activation="tanh",
                         bidirectional=True, num_layers=2)
    print(f"\nModel 3 (RNN with custom params):")
    print(f"  {model3}")
    print()

    # Example 3: Batch processing with variable args
    print("\nExample 3: Flexible Batch Processing")
    print("-" * 70)

    def simple_processor(batch, multiplier=1, offset=0):
        """Simple batch processor for demonstration."""
        return sum(batch) * multiplier + offset

    data = list(range(1, 21))
    print(f"Data: {data}")
    print(f"Batch size: 5")

    results1 = batch_process(data, simple_processor, batch_size=5)
    print(f"\nResults (default params): {results1}")

    results2 = batch_process(data, simple_processor, batch_size=5,
                            multiplier=2)
    print(f"Results (multiplier=2): {results2}")

    results3 = batch_process(data, simple_processor, batch_size=5,
                            multiplier=2, offset=10)
    print(f"Results (multiplier=2, offset=10): {results3}")
    print()

    # Example 4: Image augmentation with flexible transforms
    print("\nExample 4: Flexible Image Augmentation")
    print("-" * 70)
    aug_result1 = augment_image(
        "image_001.jpg",
        flip_horizontal=True,
        rotate=90,
        brightness=1.2
    )
    print(f"Image: {aug_result1['original']}")
    print(f"Transforms: {aug_result1['transforms_applied']}")

    aug_result2 = augment_image(
        "image_002.jpg",
        flip_horizontal=True,
        rotate=90,
        brightness=1.2,
        contrast=1.5,
        saturation=0.8,
        blur=2.0
    )
    print(f"\nImage: {aug_result2['original']}")
    print(f"Transforms: {aug_result2['transforms_applied']}")
    print()

    # Example 5: Config merging
    print("\nExample 5: Flexible Config Merging")
    print("-" * 70)
    default_config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
    print(f"Default config: {default_config}")

    user_config = {
        "batch_size": 64,
        "epochs": 50,
        "optimizer": "adam"
    }
    print(f"User config: {user_config}")

    merged = merge_configs(default_config, user_config,
                          learning_rate=0.01, weight_decay=0.0001)
    print(f"\nMerged config (with overrides):")
    for key, value in merged.items():
        print(f"  {key}: {value}")
    print()

    # Example 6: Training with callbacks
    print("\nExample 6: Training with Flexible Callbacks")
    print("-" * 70)

    def checkpoint_callback(results):
        print(f"  [Checkpoint] Saving model at epoch {results['epochs_trained']}")

    def logging_callback(results):
        print(f"  [Logger] Accuracy: {results['final_accuracy']:.2%}")

    def metrics_callback(results):
        print(f"  [Metrics] Loss: {results['final_loss']:.4f}")

    model_cfg = {"type": "resnet50", "layers": [64, 128, 256]}
    results = train_model(
        model_cfg,
        checkpoint_callback,
        logging_callback,
        metrics_callback,
        epochs=5,
        optimizer="adam",
        learning_rate=0.001
    )
    print(f"\nTraining complete: {results}")
    print()

    # Example 7: Pipeline creation
    print("\nExample 7: Flexible Pipeline Creation")
    print("-" * 70)
    pipeline1 = create_pipeline(
        "normalize", "augment", "batch",
        cache=True,
        normalize={"method": "minmax"},
        augment={"flip": True, "rotate": True},
        batch={"size": 32}
    )
    print("Pipeline 1:")
    print(f"  Steps: {pipeline1['steps']}")
    print(f"  Cache: {pipeline1['cache_enabled']}")
    print(f"  Step configs: {pipeline1['step_configs']}")

    pipeline2 = create_pipeline(
        "load", "preprocess", "validate", "save",
        parallel=True,
        load={"format": "json"},
        validate={"schema": "v2.0"}
    )
    print("\nPipeline 2:")
    print(f"  Steps: {pipeline2['steps']}")
    print(f"  Parallel: {pipeline2['parallel_execution']}")
    print(f"  Step configs: {pipeline2['step_configs']}")
    print()

    print("=" * 70)
    print("✓ All flexible function demonstrations completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
