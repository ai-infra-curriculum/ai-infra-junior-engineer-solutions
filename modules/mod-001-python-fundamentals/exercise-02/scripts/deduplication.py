#!/usr/bin/env python3
"""
Deduplication utilities for ML datasets.

Demonstrates techniques for finding and removing duplicate data.
"""

import os
from typing import List, Set, Dict
from collections import defaultdict, Counter


def remove_duplicate_samples(sample_ids: List[int]) -> List[int]:
    """
    Remove duplicates while preserving order.

    Args:
        sample_ids: List of sample IDs (may contain duplicates)

    Returns:
        List of unique sample IDs in original order
    """
    seen: Set[int] = set()
    unique_samples: List[int] = []

    for sample_id in sample_ids:
        if sample_id not in seen:
            seen.add(sample_id)
            unique_samples.append(sample_id)

    return unique_samples


def find_duplicate_files(filepaths: List[str]) -> Dict[str, List[str]]:
    """
    Find duplicate files by basename.

    Args:
        filepaths: List of file paths

    Returns:
        Dictionary mapping basename -> list of full paths
    """
    basename_map: Dict[str, List[str]] = defaultdict(list)

    for filepath in filepaths:
        basename = os.path.basename(filepath)
        basename_map[basename].append(filepath)

    # Return only duplicates
    duplicates = {k: v for k, v in basename_map.items() if len(v) > 1}
    return duplicates


def validate_unique_classes(dataset: List[tuple]) -> bool:
    """
    Validate that all sample IDs are unique.

    Args:
        dataset: List of (sample_id, class, filepath) tuples

    Returns:
        True if all sample IDs are unique
    """
    sample_ids = [item[0] for item in dataset]
    unique_ids = set(sample_ids)

    if len(sample_ids) != len(unique_ids):
        duplicates = len(sample_ids) - len(unique_ids)
        print(f"Warning: {duplicates} duplicate sample IDs found")
        return False
    return True


def find_class_label_inconsistencies(dataset: List[tuple]) -> Dict[int, List[str]]:
    """
    Find samples with multiple class labels.

    Args:
        dataset: List of (sample_id, class, filepath) tuples

    Returns:
        Dictionary mapping sample_id -> list of inconsistent labels
    """
    sample_to_classes: Dict[int, Set[str]] = defaultdict(set)

    for sample_id, class_label, _ in dataset:
        sample_to_classes[sample_id].add(class_label)

    # Return only inconsistencies (multiple labels for same ID)
    inconsistencies = {
        sid: list(classes)
        for sid, classes in sample_to_classes.items()
        if len(classes) > 1
    }

    return inconsistencies


def main():
    """Demonstrate deduplication techniques."""
    print("=" * 60)
    print("Deduplication for ML Datasets")
    print("=" * 60)
    print()

    # Example: dataset with duplicates
    raw_dataset = [
        (1, "cat", "img1.jpg"),
        (2, "dog", "img2.jpg"),
        (3, "cat", "img3.jpg"),
        (1, "cat", "img1_copy.jpg"),  # Duplicate ID
        (4, "bird", "img4.jpg"),
        (2, "dog", "img2_v2.jpg"),    # Duplicate ID
        (5, "cat", "img5.jpg"),
        (6, "dog", "img6.jpg"),
    ]

    print(f"Original dataset size: {len(raw_dataset)}")
    print()

    # Extract unique IDs
    sample_ids = [item[0] for item in raw_dataset]
    unique_ids = list(set(sample_ids))
    print(f"Unique sample IDs: {len(unique_ids)}")
    print(f"Total IDs: {len(sample_ids)}")
    print(f"Duplicates: {len(sample_ids) - len(unique_ids)}")
    print()

    # Remove duplicates preserving first occurrence
    cleaned_ids = remove_duplicate_samples(sample_ids)
    print(f"Cleaned IDs (preserving order): {cleaned_ids}")
    print()

    # Find duplicate entries
    seen_ids: Set[int] = set()
    duplicates = []
    for item in raw_dataset:
        if item[0] in seen_ids:
            duplicates.append(item)
        seen_ids.add(item[0])

    print(f"Duplicate entries ({len(duplicates)}):")
    for dup in duplicates:
        print(f"  ID {dup[0]}: {dup[1]} - {dup[2]}")
    print()

    # Validate uniqueness
    is_valid = validate_unique_classes(raw_dataset)
    print(f"Dataset has unique IDs: {is_valid}")
    print()

    # Find duplicate filenames
    print("=" * 60)
    print("Duplicate Filename Detection")
    print("=" * 60)
    print()

    image_files = [
        "/data/train/img001.jpg",
        "/data/train/img002.jpg",
        "/data/val/img001.jpg",      # Same basename
        "/data/test/img002.jpg",     # Same basename
        "/data/train/img003.jpg",
        "/data/val/img003.jpg",      # Same basename
    ]

    duplicate_files = find_duplicate_files(image_files)
    print(f"Duplicate filenames found: {len(duplicate_files)}")
    for basename, paths in duplicate_files.items():
        print(f"\n  {basename}:")
        for path in paths:
            print(f"    - {path}")
    print()

    # Check for class label inconsistencies
    print("=" * 60)
    print("Class Label Consistency")
    print("=" * 60)
    print()

    inconsistent_dataset = [
        (1, "cat", "img1.jpg"),
        (2, "dog", "img2.jpg"),
        (1, "dog", "img1_v2.jpg"),  # Inconsistent! ID 1 has 2 labels
        (3, "bird", "img3.jpg"),
        (2, "cat", "img2_v2.jpg"),  # Inconsistent! ID 2 has 2 labels
    ]

    inconsistencies = find_class_label_inconsistencies(inconsistent_dataset)
    if inconsistencies:
        print(f"⚠️  Found {len(inconsistencies)} samples with inconsistent labels:")
        for sample_id, labels in inconsistencies.items():
            print(f"  Sample {sample_id}: {labels}")
    else:
        print("✓ All samples have consistent labels")
    print()

    # Class distribution analysis
    print("=" * 60)
    print("Class Distribution")
    print("=" * 60)
    print()

    class_labels = [item[1] for item in raw_dataset]
    unique_classes = set(class_labels)
    print(f"Unique classes: {unique_classes}")
    print(f"Number of classes: {len(unique_classes)}")
    print()

    # Count occurrences
    class_counts = Counter(class_labels)
    print("Class distribution:")
    for cls, count in class_counts.most_common():
        pct = (count / len(class_labels)) * 100
        print(f"  {cls}: {count} samples ({pct:.1f}%)")

    print()
    print("✓ Deduplication demonstration complete")


if __name__ == "__main__":
    main()
