#!/usr/bin/env python3
"""
Functional Programming Patterns

Demonstrates map, filter, reduce, lambda functions, and function composition
for clean, concise ML data processing.
"""

from typing import List, Callable, Any, Dict
from functools import reduce


# Lambda functions for common operations
square = lambda x: x ** 2
is_even = lambda x: x % 2 == 0
is_positive = lambda x: x > 0
normalize = lambda x, min_val, max_val: (x - min_val) / (max_val - min_val) if max_val != min_val else 0


def example_map():
    """Demonstrate map() for transformations."""
    print("=" * 70)
    print("Map Examples (Transformations)")
    print("=" * 70)
    print()

    # Example 1: Square all numbers
    print("1. Square all numbers:")
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(square, numbers))
    print(f"   Original: {numbers}")
    print(f"   Squared:  {squared}")
    print()

    # Example 2: Convert probabilities to predictions
    print("2. Convert probabilities to class predictions:")
    probabilities = [0.2, 0.8, 0.6, 0.3, 0.9]
    predictions = list(map(lambda p: 1 if p > 0.5 else 0, probabilities))
    print(f"   Probabilities: {probabilities}")
    print(f"   Predictions:   {predictions}")
    print()

    # Example 3: Parse filenames
    print("3. Extract version numbers from filenames:")
    files = ["model_v1.h5", "model_v2.h5", "model_v3.h5"]
    versions = list(map(lambda f: f.split("_v")[1].split(".")[0], files))
    print(f"   Files:    {files}")
    print(f"   Versions: {versions}")
    print()

    # Example 4: Normalize features
    print("4. Normalize features to 0-1 range:")
    features = [10, 20, 30, 40, 50]
    min_val, max_val = min(features), max(features)
    normalized = list(map(lambda x: normalize(x, min_val, max_val), features))
    print(f"   Original:   {features}")
    print(f"   Normalized: {[f'{x:.2f}' for x in normalized]}")
    print()

    # Example 5: Apply function to dict values
    print("5. Scale all metric values by 100:")
    metrics = {"accuracy": 0.92, "precision": 0.88, "recall": 0.95}
    scaled = dict(map(lambda item: (item[0], item[1] * 100), metrics.items()))
    print(f"   Original: {metrics}")
    print(f"   Scaled:   {scaled}")
    print()


def example_filter():
    """Demonstrate filter() for selection."""
    print("=" * 70)
    print("Filter Examples (Selection)")
    print("=" * 70)
    print()

    # Example 1: Filter even numbers
    print("1. Filter even numbers:")
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    evens = list(filter(is_even, numbers))
    print(f"   Numbers: {numbers}")
    print(f"   Evens:   {evens}")
    print()

    # Example 2: Filter high-accuracy models
    print("2. Filter models with accuracy > 0.90:")
    models = [
        {"name": "model1", "accuracy": 0.85},
        {"name": "model2", "accuracy": 0.92},
        {"name": "model3", "accuracy": 0.88},
        {"name": "model4", "accuracy": 0.95},
    ]
    high_accuracy = list(filter(lambda m: m["accuracy"] > 0.90, models))
    print(f"   All models: {[m['name'] for m in models]}")
    print(f"   High accuracy: {[m['name'] for m in high_accuracy]}")
    print()

    # Example 3: Filter completed experiments
    print("3. Filter completed experiments:")
    experiments = [
        ("exp1", "completed"),
        ("exp2", "running"),
        ("exp3", "completed"),
        ("exp4", "failed"),
        ("exp5", "completed"),
    ]
    completed = list(filter(lambda e: e[1] == "completed", experiments))
    print(f"   All: {[e[0] for e in experiments]}")
    print(f"   Completed: {[e[0] for e in completed]}")
    print()

    # Example 4: Filter valid samples
    print("4. Filter samples with positive labels:")
    samples = [
        {"id": 1, "label": 1, "confidence": 0.9},
        {"id": 2, "label": 0, "confidence": 0.8},
        {"id": 3, "label": 1, "confidence": 0.95},
        {"id": 4, "label": -1, "confidence": 0.3},  # Invalid
        {"id": 5, "label": 1, "confidence": 0.85},
    ]
    valid = list(filter(lambda s: s["label"] > 0, samples))
    print(f"   All samples: {len(samples)}")
    print(f"   Valid samples: {len(valid)}")
    print(f"   Valid IDs: {[s['id'] for s in valid]}")
    print()

    # Example 5: Filter outliers
    print("5. Remove outliers (values outside 1-100):")
    data = [5, 150, 23, -10, 67, 89, 12, 200, 45]
    filtered = list(filter(lambda x: 1 <= x <= 100, data))
    print(f"   Original: {data}")
    print(f"   Filtered: {filtered}")
    print()


def example_reduce():
    """Demonstrate reduce() for aggregation."""
    print("=" * 70)
    print("Reduce Examples (Aggregation)")
    print("=" * 70)
    print()

    # Example 1: Sum all numbers
    print("1. Sum all numbers:")
    numbers = [1, 2, 3, 4, 5]
    total = reduce(lambda acc, x: acc + x, numbers, 0)
    print(f"   Numbers: {numbers}")
    print(f"   Sum:     {total}")
    print()

    # Example 2: Find maximum
    print("2. Find maximum accuracy:")
    accuracies = [0.85, 0.92, 0.88, 0.95, 0.90]
    max_acc = reduce(lambda acc, x: max(acc, x), accuracies, 0)
    print(f"   Accuracies: {accuracies}")
    print(f"   Maximum:    {max_acc}")
    print()

    # Example 3: Merge dictionaries
    print("3. Merge configuration dictionaries:")
    configs = [
        {"learning_rate": 0.001},
        {"batch_size": 32},
        {"epochs": 100},
        {"optimizer": "adam"},
    ]
    merged = reduce(lambda acc, d: {**acc, **d}, configs, {})
    print(f"   Configs: {configs}")
    print(f"   Merged:  {merged}")
    print()

    # Example 4: Product of all numbers
    print("4. Calculate product:")
    numbers = [2, 3, 4, 5]
    product = reduce(lambda acc, x: acc * x, numbers, 1)
    print(f"   Numbers: {numbers}")
    print(f"   Product: {product}")
    print()

    # Example 5: Flatten nested lists
    print("5. Flatten nested lists:")
    nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    flattened = reduce(lambda acc, lst: acc + lst, nested, [])
    print(f"   Nested:    {nested}")
    print(f"   Flattened: {flattened}")
    print()

    # Example 6: Count occurrences
    print("6. Count class occurrences:")
    labels = ["cat", "dog", "cat", "bird", "dog", "cat", "dog", "dog"]
    counts = reduce(
        lambda acc, label: {**acc, label: acc.get(label, 0) + 1},
        labels,
        {}
    )
    print(f"   Labels: {labels}")
    print(f"   Counts: {counts}")
    print()


def example_composition():
    """Demonstrate function composition."""
    print("=" * 70)
    print("Function Composition (Pipelines)")
    print("=" * 70)
    print()

    # Example 1: Filter, transform, aggregate
    print("1. Pipeline: filter evens → square → sum:")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = reduce(
        lambda acc, x: acc + x,
        map(square, filter(is_even, data)),
        0
    )
    print(f"   Data:   {data}")
    print(f"   Evens:  {list(filter(is_even, data))}")
    print(f"   Squared: {list(map(square, filter(is_even, data)))}")
    print(f"   Sum:    {result}")
    print()

    # Example 2: Process model metrics
    print("2. Filter high-accuracy models → extract names:")
    metrics = [
        {"model": "m1", "acc": 0.85, "loss": 0.25},
        {"model": "m2", "acc": 0.92, "loss": 0.15},
        {"model": "m3", "acc": 0.88, "loss": 0.20},
        {"model": "m4", "acc": 0.95, "loss": 0.10},
        {"model": "m5", "acc": 0.82, "loss": 0.30},
    ]
    best_models = list(map(
        lambda m: m["model"],
        filter(lambda m: m["acc"] > 0.87, metrics)
    ))
    print(f"   All models: {[m['model'] for m in metrics]}")
    print(f"   Best models (acc > 0.87): {best_models}")
    print()

    # Example 3: Data preprocessing pipeline
    print("3. Data preprocessing pipeline:")
    raw_data = [-5, 10, -2, 15, 20, -8, 25, 30]
    print(f"   Raw data: {raw_data}")

    # Step 1: Filter positive values
    positive = list(filter(is_positive, raw_data))
    print(f"   Step 1 (filter positive): {positive}")

    # Step 2: Normalize to 0-1
    min_val, max_val = min(positive), max(positive)
    normalized = list(map(lambda x: normalize(x, min_val, max_val), positive))
    print(f"   Step 2 (normalize): {[f'{x:.2f}' for x in normalized]}")

    # Step 3: Calculate mean
    mean = reduce(lambda acc, x: acc + x, normalized, 0) / len(normalized)
    print(f"   Step 3 (mean): {mean:.4f}")
    print()

    # Example 4: Complex transformation
    print("4. Extract and process file metadata:")
    files = [
        "model_v1_acc_92.h5",
        "model_v2_acc_88.h5",
        "model_v3_acc_95.h5",
        "model_v4_acc_85.h5",
    ]

    # Extract accuracy, filter > 90, get file names
    file_data = list(map(
        lambda f: {
            "name": f,
            "accuracy": int(f.split("_acc_")[1].split(".")[0]) / 100
        },
        files
    ))
    print(f"   Files: {files}")
    print(f"   Parsed: {file_data}")

    high_acc_files = list(map(
        lambda d: d["name"],
        filter(lambda d: d["accuracy"] > 0.90, file_data)
    ))
    print(f"   High accuracy (>90%): {high_acc_files}")
    print()


def example_advanced():
    """Demonstrate advanced functional patterns."""
    print("=" * 70)
    print("Advanced Functional Patterns")
    print("=" * 70)
    print()

    # Example 1: Chaining transformations
    print("1. Chaining multiple transformations:")
    dataset = [
        {"id": 1, "features": [1, 2, 3], "label": 1},
        {"id": 2, "features": [4, 5, 6], "label": 0},
        {"id": 3, "features": [7, 8, 9], "label": 1},
        {"id": 4, "features": [10, 11, 12], "label": 1},
    ]

    # Chain: filter label==1 → extract features → flatten → sum
    result = reduce(
        lambda acc, x: acc + x,
        reduce(
            lambda acc, lst: acc + lst,
            map(
                lambda d: d["features"],
                filter(lambda d: d["label"] == 1, dataset)
            ),
            []
        ),
        0
    )
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Filtered (label==1): {len(list(filter(lambda d: d['label'] == 1, dataset)))}")
    print(f"   Sum of all features: {result}")
    print()

    # Example 2: Partial application
    print("2. Partial function application:")

    def multiply(x, y):
        return x * y

    # Create specialized functions
    double = lambda x: multiply(x, 2)
    triple = lambda x: multiply(x, 3)

    values = [1, 2, 3, 4, 5]
    doubled = list(map(double, values))
    tripled = list(map(triple, values))

    print(f"   Original: {values}")
    print(f"   Doubled:  {doubled}")
    print(f"   Tripled:  {tripled}")
    print()

    # Example 3: Custom aggregations
    print("3. Custom aggregations:")
    experiments = [
        {"name": "exp1", "metrics": {"acc": 0.85, "loss": 0.25}},
        {"name": "exp2", "metrics": {"acc": 0.92, "loss": 0.15}},
        {"name": "exp3", "metrics": {"acc": 0.88, "loss": 0.20}},
    ]

    # Calculate average accuracy
    avg_acc = reduce(
        lambda acc, x: acc + x,
        map(lambda e: e["metrics"]["acc"], experiments),
        0
    ) / len(experiments)

    print(f"   Experiments: {[e['name'] for e in experiments]}")
    print(f"   Average accuracy: {avg_acc:.4f}")

    # Find best experiment
    best = reduce(
        lambda best, exp: exp if exp["metrics"]["acc"] > best["metrics"]["acc"] else best,
        experiments
    )
    print(f"   Best experiment: {best['name']} (acc={best['metrics']['acc']})")
    print()


def main():
    """Run all functional programming demonstrations."""
    example_map()
    print()

    example_filter()
    print()

    example_reduce()
    print()

    example_composition()
    print()

    example_advanced()
    print()

    print("=" * 70)
    print("✓ All functional programming patterns demonstrated")
    print("=" * 70)


if __name__ == "__main__":
    main()
