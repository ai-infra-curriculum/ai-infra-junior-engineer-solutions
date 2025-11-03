#!/usr/bin/env python3
"""
Test ML Utils Module

Demonstrates usage of the ml_utils package for metrics and preprocessing.
"""

import sys
from pathlib import Path

# Add parent directory to path to import ml_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_utils import metrics, preprocessing


def test_metrics():
    """Test metrics module functionality."""
    print("=" * 70)
    print("Testing Metrics Module")
    print("=" * 70)
    print()

    # Test data: binary classification
    predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    labels = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

    print("Binary Classification Metrics:")
    print(f"  Predictions: {predictions}")
    print(f"  Labels:      {labels}")
    print()

    # Calculate individual metrics
    acc = metrics.accuracy(predictions, labels)
    prec = metrics.precision(predictions, labels)
    rec = metrics.recall(predictions, labels)
    f1 = metrics.f1_score(predictions, labels)

    print(f"  Accuracy:  {acc:.2%} ({acc:.4f})")
    print(f"  Precision: {prec:.2%} ({prec:.4f})")
    print(f"  Recall:    {rec:.2%} ({rec:.4f})")
    print(f"  F1 Score:  {f1:.2%} ({f1:.4f})")
    print()

    # Generate classification report
    print("Classification Report:")
    report = metrics.classification_report(
        predictions, labels,
        class_names=["negative", "positive"]
    )

    for class_name, class_metrics in report.items():
        if class_name != "accuracy":
            print(f"  {class_name}:")
            if isinstance(class_metrics, dict):
                for metric_name, metric_value in class_metrics.items():
                    print(f"    {metric_name:12s}: {metric_value:.4f}")
    print(f"  Overall Accuracy: {report['accuracy']:.2%}")
    print()

    # Confusion matrix
    print("Confusion Matrix:")
    cm = metrics.confusion_matrix(predictions, labels, num_classes=2)
    print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print()

    # Regression metrics
    print("-" * 70)
    print("Regression Metrics:")
    print()

    preds_reg = [2.5, 3.1, 4.2, 5.8, 6.3, 7.1, 8.5, 9.2]
    labels_reg = [2.0, 3.0, 4.0, 6.0, 6.5, 7.0, 8.0, 9.0]

    print(f"  Predictions: {preds_reg}")
    print(f"  Labels:      {labels_reg}")
    print()

    # Calculate regression metrics
    mse = metrics.mean_squared_error(preds_reg, labels_reg)
    rmse = metrics.root_mean_squared_error(preds_reg, labels_reg)
    mae = metrics.mean_absolute_error(preds_reg, labels_reg)
    r2 = metrics.r_squared(preds_reg, labels_reg)

    print(f"  MSE:        {mse:.4f}")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"  R² Score:   {r2:.4f}")
    print()

    # Regression report
    print("Regression Report:")
    reg_report = metrics.regression_report(preds_reg, labels_reg)
    for metric_name, metric_value in reg_report.items():
        print(f"  {metric_name:12s}: {metric_value:.4f}")
    print()


def test_preprocessing():
    """Test preprocessing module functionality."""
    print("=" * 70)
    print("Testing Preprocessing Module")
    print("=" * 70)
    print()

    # Test 1: Min-max normalization
    print("1. Min-Max Normalization:")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = preprocessing.normalize_minmax(data)
    print(f"   Original:   {data}")
    print(f"   Normalized: {[f'{x:.2f}' for x in normalized]}")
    print()

    # Test 2: Z-score normalization
    print("2. Z-Score Normalization:")
    normalized_z = preprocessing.normalize_zscore(data)
    print(f"   Original:   {data}")
    print(f"   Z-score:    {[f'{x:.2f}' for x in normalized_z]}")
    print()

    # Test 3: Outlier removal
    print("3. Outlier Removal (IQR method):")
    data_with_outliers = [1, 2, 3, 4, 5, 100, 2, 3, 4, 5]
    cleaned = preprocessing.remove_outliers(data_with_outliers, method="iqr")
    print(f"   With outliers: {data_with_outliers}")
    print(f"   Cleaned:       {cleaned}")
    print(f"   Removed: {len(data_with_outliers) - len(cleaned)} values")
    print()

    # Test 4: Missing value handling
    print("4. Missing Value Handling:")
    data_with_missing = [1.0, 2.0, None, 4.0, None, 6.0, 7.0]
    print(f"   With missing: {data_with_missing}")

    filled_mean = preprocessing.fill_missing_values(data_with_missing, "mean")
    print(f"   Filled (mean):    {[f'{x:.2f}' for x in filled_mean]}")

    filled_median = preprocessing.fill_missing_values(data_with_missing, "median")
    print(f"   Filled (median):  {[f'{x:.2f}' for x in filled_median]}")

    filled_forward = preprocessing.fill_missing_values(data_with_missing, "forward")
    print(f"   Filled (forward): {[f'{x:.2f}' for x in filled_forward]}")
    print()

    # Test 5: One-hot encoding
    print("5. One-Hot Encoding:")
    labels = [0, 1, 2, 1, 0, 2]
    one_hot = preprocessing.one_hot_encode(labels, num_classes=3)
    print(f"   Labels:   {labels}")
    print(f"   One-hot:")
    for label, encoding in zip(labels, one_hot):
        print(f"     {label} → {encoding}")
    print()

    # Test 6: Label encoding
    print("6. Label Encoding:")
    string_labels = ["cat", "dog", "cat", "bird", "dog", "cat"]
    encoded, mapping = preprocessing.label_encode(string_labels)
    print(f"   String labels: {string_labels}")
    print(f"   Encoded:       {encoded}")
    print(f"   Mapping:       {mapping}")
    print()

    # Test 7: Train-test split
    print("7. Train-Test Split:")
    dataset = list(range(20))
    train, test = preprocessing.train_test_split(dataset, test_size=0.2,
                                                 shuffle=True, random_seed=42)
    print(f"   Dataset size:   {len(dataset)}")
    print(f"   Train size:     {len(train)} ({len(train)/len(dataset):.0%})")
    print(f"   Test size:      {len(test)} ({len(test)/len(dataset):.0%})")
    print(f"   Train samples:  {train[:5]} ...")
    print(f"   Test samples:   {test[:3]} ...")
    print()

    # Test 8: Stratified split
    print("8. Stratified Split (maintains class distribution):")
    labeled_data = [
        ("sample1", 0), ("sample2", 1), ("sample3", 0), ("sample4", 1),
        ("sample5", 0), ("sample6", 1), ("sample7", 0), ("sample8", 1),
    ]
    train_strat, test_strat = preprocessing.stratified_split(
        labeled_data, test_size=0.25, random_seed=42
    )
    print(f"   Total samples: {len(labeled_data)}")
    print(f"   Train size: {len(train_strat)}")
    print(f"   Test size:  {len(test_strat)}")

    # Check class distribution
    train_class_0 = sum(1 for _, label in train_strat if label == 0)
    train_class_1 = sum(1 for _, label in train_strat if label == 1)
    test_class_0 = sum(1 for _, label in test_strat if label == 0)
    test_class_1 = sum(1 for _, label in test_strat if label == 1)

    print(f"   Train: class 0={train_class_0}, class 1={train_class_1}")
    print(f"   Test:  class 0={test_class_0}, class 1={test_class_1}")
    print()

    # Test 9: Value clipping
    print("9. Value Clipping:")
    data_to_clip = [1, 5, 10, 15, 20, 25]
    clipped = preprocessing.clip_values(data_to_clip, min_value=5, max_value=20)
    print(f"   Original: {data_to_clip}")
    print(f"   Clipped (5-20): {clipped}")
    print()

    # Test 10: Batch normalization
    print("10. Batch Normalization:")
    batches = [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [100.0, 200.0, 300.0, 400.0]
    ]
    normalized_batches = preprocessing.batch_normalize(batches)
    print(f"    Batch 1: {batches[0]}")
    print(f"    Normalized: {[f'{x:.2f}' for x in normalized_batches[0]]}")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 70)
    print("ML Utils Package - Usage Examples")
    print("=" * 70)
    print()

    # Test metrics module
    test_metrics()

    print("\n")

    # Test preprocessing module
    test_preprocessing()

    print("=" * 70)
    print("✓ All tests completed successfully")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
