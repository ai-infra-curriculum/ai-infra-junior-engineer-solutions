#!/usr/bin/env python3
"""
List comprehensions for data processing.

Demonstrates efficient data processing using list comprehensions.
"""

from typing import List, Tuple


def main():
    """Demonstrate list comprehensions for ML data processing."""
    print("=" * 60)
    print("List Comprehensions for Data Processing")
    print("=" * 60)
    print()

    # Sample data: model training losses
    losses = [2.5, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02, 1.01]

    # Square all losses
    squared_losses = [loss ** 2 for loss in losses]
    print(f"Original losses: {losses}")
    print(f"Squared losses: {squared_losses[:5]}...")
    print()

    # Filter losses below threshold
    low_losses = [loss for loss in losses if loss < 1.5]
    print(f"Losses below 1.5: {low_losses}")
    print()

    # Transform and filter
    normalized_losses = [(loss - min(losses)) / (max(losses) - min(losses))
                         for loss in losses if loss < 2.0]
    print(f"Normalized losses (< 2.0): {[f'{x:.3f}' for x in normalized_losses]}")
    print()

    # Nested list comprehension: create batches
    data_points = list(range(1, 21))  # 20 data points
    batch_size = 5
    batches = [data_points[i:i+batch_size]
               for i in range(0, len(data_points), batch_size)]
    print(f"Data batches (size {batch_size}):")
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}: {batch}")
    print()

    # Process image dimensions
    image_sizes = [(224, 224), (256, 256), (512, 512), (1024, 1024)]
    total_pixels = [width * height for width, height in image_sizes]
    print("Image sizes and total pixels:")
    for size, pixels in zip(image_sizes, total_pixels):
        print(f"  {size}: {pixels:,} pixels")
    print()

    # Create one-hot encoding
    classes = ["cat", "dog", "bird", "fish"]
    target_class = "dog"
    one_hot = [1 if cls == target_class else 0 for cls in classes]
    print(f"One-hot encoding for '{target_class}': {one_hot}")
    print()

    # Parse model filenames
    model_files = [
        "model_v1_acc_0.85.h5",
        "model_v2_acc_0.92.h5",
        "model_v3_acc_0.88.h5"
    ]

    # Extract accuracies
    accuracies = [float(f.split("_acc_")[1].replace(".h5", ""))
                  for f in model_files]
    print("Model accuracies:")
    for file, acc in zip(model_files, accuracies):
        print(f"  {file}: {acc:.2f}")

    # Find best model
    best_idx = accuracies.index(max(accuracies))
    best_model = model_files[best_idx]
    print(f"\nBest model: {best_model} (acc: {max(accuracies)})")
    print()

    # Conditional list building
    training_config = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "use_gpu": True,
        "augmentation": True
    }

    enabled_features = [key for key, value in training_config.items()
                        if isinstance(value, bool) and value]
    print(f"Enabled features: {enabled_features}")
    print()

    # Extended tasks
    print("=" * 60)
    print("Extended Tasks")
    print("=" * 60)
    print()

    # Task 1: Filter models with accuracy > 0.90
    high_acc_models = [(f, acc) for f, acc in zip(model_files, accuracies)
                       if acc > 0.90]
    print("Models with accuracy > 0.90:")
    for model, acc in high_acc_models:
        print(f"  {model}: {acc:.2f}")
    print()

    # Task 2: Overlapping batches (stride < batch_size)
    data = list(range(1, 11))
    batch_size, stride = 4, 2
    overlapping_batches = [data[i:i+batch_size]
                          for i in range(0, len(data)-batch_size+1, stride)]
    print(f"Overlapping batches (size={batch_size}, stride={stride}):")
    for i, batch in enumerate(overlapping_batches, 1):
        print(f"  Batch {i}: {batch}")
    print()

    # Task 3: List of tuples sorted by accuracy
    model_acc_pairs = sorted(
        [(f, acc) for f, acc in zip(model_files, accuracies)],
        key=lambda x: x[1],
        reverse=True
    )
    print("Models sorted by accuracy (descending):")
    for model, acc in model_acc_pairs:
        print(f"  {model}: {acc:.2f}")
    print()

    # Task 4: Flatten 2D list
    features_2d = [
        ["age", "income"],
        ["location", "occupation"],
        ["clicks", "views", "conversions"]
    ]
    flattened = [feature for sublist in features_2d for feature in sublist]
    print(f"Original 2D: {features_2d}")
    print(f"Flattened: {flattened}")

    print()
    print("✓ List comprehensions demonstration complete")


if __name__ == "__main__":
    main()
