#!/usr/bin/env python3
"""
Dictionary comprehensions for ML metrics processing.

Demonstrates efficient dictionary operations using comprehensions.
"""

from typing import Dict


def main():
    """Demonstrate dict comprehensions for ML metrics."""
    print("=" * 60)
    print("Dict Comprehensions for Metrics Processing")
    print("=" * 60)
    print()

    # Create metric dictionary from lists
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    metric_values = [0.92, 0.89, 0.94, 0.91]

    metrics = {name: value for name, value in zip(metric_names, metric_values)}
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.2f}")
    print()

    # Filter metrics above threshold
    high_metrics = {k: v for k, v in metrics.items() if v > 0.90}
    print("High metrics (>0.90):")
    for name, value in high_metrics.items():
        print(f"  {name}: {value:.2f}")
    print()

    # Transform values
    metrics_percentage = {k: f"{v*100:.1f}%" for k, v in metrics.items()}
    print("Metrics as percentages:")
    for name, pct in metrics_percentage.items():
        print(f"  {name}: {pct}")
    print()

    # Nested dictionary: experiment results
    print("=" * 60)
    print("Experiment Results Analysis")
    print("=" * 60)
    print()

    experiments = {
        "exp_001": {
            "model": "resnet50",
            "accuracy": 0.92,
            "loss": 0.15,
            "epoch": 50,
            "status": "completed"
        },
        "exp_002": {
            "model": "vgg16",
            "accuracy": 0.88,
            "loss": 0.22,
            "epoch": 45,
            "status": "completed"
        },
        "exp_003": {
            "model": "mobilenet",
            "accuracy": 0.85,
            "loss": 0.28,
            "epoch": 30,
            "status": "failed"
        }
    }

    print("All experiments:")
    for exp_id, data in experiments.items():
        print(f"  {exp_id}: {data['model']} - acc={data['accuracy']:.2f}, status={data['status']}")
    print()

    # Find best experiment by accuracy
    completed_exps = {k: v for k, v in experiments.items()
                      if v["status"] == "completed"}
    best_exp_id = max(completed_exps, key=lambda k: completed_exps[k]["accuracy"])
    best_exp = completed_exps[best_exp_id]
    print(f"Best experiment: {best_exp_id}")
    print(f"  Model: {best_exp['model']}")
    print(f"  Accuracy: {best_exp['accuracy']:.2f}")
    print()

    # Extract specific field from all experiments
    accuracies = {exp_id: data["accuracy"]
                  for exp_id, data in experiments.items()
                  if data["status"] == "completed"}
    print("Completed experiment accuracies:")
    for exp_id, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {exp_id}: {acc:.2f}")
    print()

    # Group experiments by model
    by_model: Dict[str, list] = {}
    for exp_id, data in experiments.items():
        model = data["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(exp_id)

    print("Experiments grouped by model:")
    for model, exp_ids in by_model.items():
        print(f"  {model}: {exp_ids}")
    print()

    # Create summary statistics
    completed_count = sum(1 for v in experiments.values() if v["status"] == "completed")
    avg_accuracy = sum(v["accuracy"] for v in experiments.values()
                      if v["status"] == "completed") / completed_count

    summary = {
        "total_experiments": len(experiments),
        "completed": completed_count,
        "failed": sum(1 for v in experiments.values() if v["status"] == "failed"),
        "avg_accuracy": avg_accuracy
    }

    print("Summary statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Invert dictionary (swap keys and values)
    model_to_exp = {v["model"]: k for k, v in experiments.items()}
    print("Model to experiment ID mapping:")
    for model, exp_id in model_to_exp.items():
        print(f"  {model}: {exp_id}")

    print()
    print("✓ Dict comprehensions demonstration complete")


if __name__ == "__main__":
    main()
