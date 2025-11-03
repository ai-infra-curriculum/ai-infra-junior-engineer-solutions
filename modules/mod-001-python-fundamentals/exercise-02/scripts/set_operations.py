#!/usr/bin/env python3
"""
Set operations for dataset management and validation.

Demonstrates set operations for preventing data leakage and dataset analysis.
"""

from typing import Set


def main():
    """Demonstrate set operations for dataset management."""
    print("=" * 60)
    print("Set Operations for Dataset Management")
    print("=" * 60)
    print()

    # Training and validation dataset IDs
    train_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    val_ids = {9, 10, 11, 12, 13}
    test_ids = {13, 14, 15, 16, 17}

    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    print(f"Test samples: {len(test_ids)}")
    print()

    # Find overlapping samples (data leakage check)
    print("=" * 60)
    print("Data Leakage Detection")
    print("=" * 60)
    print()

    train_val_overlap = train_ids & val_ids  # Intersection
    print(f"Train-Val overlap: {train_val_overlap}")
    if train_val_overlap:
        print("  ⚠️  WARNING: Data leakage detected!")

    train_test_overlap = train_ids & test_ids
    print(f"Train-Test overlap: {train_test_overlap}")
    if not train_test_overlap:
        print("  ✓ No train-test leakage")

    val_test_overlap = val_ids & test_ids
    print(f"Val-Test overlap: {val_test_overlap}")
    if val_test_overlap:
        print("  ⚠️  WARNING: Val-test leakage!")
    print()

    # Union: all unique samples
    all_samples = train_ids | val_ids | test_ids
    print(f"Total unique samples: {len(all_samples)}")
    print(f"All sample IDs: {sorted(all_samples)}")
    print()

    # Difference: samples only in training
    train_only = train_ids - val_ids - test_ids
    print(f"Samples only in training: {train_only}")
    print()

    # Symmetric difference: samples in either but not both
    train_val_exclusive = train_ids ^ val_ids
    print(f"Exclusive to train or val (not both): {train_val_exclusive}")
    print()

    # Check if sets are disjoint (no overlap)
    is_clean_split = train_ids.isdisjoint(test_ids)
    print(f"Clean train-test split (disjoint): {is_clean_split}")
    print()

    # Add and remove samples
    print("=" * 60)
    print("Dataset Operations")
    print("=" * 60)
    print()

    new_samples = {18, 19, 20}
    test_ids_updated = test_ids | new_samples  # Union
    print(f"Original test set: {sorted(test_ids)}")
    print(f"Updated test set: {sorted(test_ids_updated)}")
    print()

    # Remove outliers
    outliers = {5, 10}
    train_cleaned = train_ids - outliers
    print(f"Training before removing outliers: {sorted(train_ids)}")
    print(f"Training after removing outliers: {sorted(train_cleaned)}")
    print()

    # Subset check
    small_set = {1, 2, 3}
    is_subset = small_set.issubset(train_ids)
    print(f"Is {small_set} subset of training? {is_subset}")

    is_superset = train_ids.issuperset(small_set)
    print(f"Is training superset of {small_set}? {is_superset}")
    print()

    # Validate clean splits
    print("=" * 60)
    print("Split Validation")
    print("=" * 60)
    print()

    def validate_splits(train: Set, val: Set, test: Set) -> bool:
        """Validate that dataset splits have no overlap."""
        issues = []

        if not train.isdisjoint(val):
            issues.append(f"Train-Val overlap: {train & val}")
        if not train.isdisjoint(test):
            issues.append(f"Train-Test overlap: {train & test}")
        if not val.isdisjoint(test):
            issues.append(f"Val-Test overlap: {val & test}")

        if issues:
            print("Validation FAILED:")
            for issue in issues:
                print(f"  ✗ {issue}")
            return False
        else:
            print("Validation PASSED:")
            print("  ✓ No overlaps detected")
            print("  ✓ Clean splits")
            return True

    # Clean splits
    clean_train = {1, 2, 3, 4, 5}
    clean_val = {6, 7, 8}
    clean_test = {9, 10, 11}

    validate_splits(clean_train, clean_val, clean_test)
    print()

    # Set operations for deduplication
    print("=" * 60)
    print("Deduplication")
    print("=" * 60)
    print()

    samples_with_duplicates = [1, 2, 3, 2, 4, 5, 3, 6, 1]
    unique_samples = set(samples_with_duplicates)

    print(f"Original samples (with duplicates): {samples_with_duplicates}")
    print(f"Unique samples: {sorted(unique_samples)}")
    print(f"Duplicates removed: {len(samples_with_duplicates) - len(unique_samples)}")

    print()
    print("✓ Set operations demonstration complete")


if __name__ == "__main__":
    main()
