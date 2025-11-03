#!/usr/bin/env python3
"""
Comprehensive ML dataset manager using all data structures.

Demonstrates professional dataset management combining lists, dicts, sets, and tuples.
"""

import random
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class MLDatasetManager:
    """Comprehensive dataset manager using all data structures."""

    def __init__(self):
        """Initialize dataset manager."""
        # Dictionary: sample_id -> sample data
        self.samples: Dict[int, Dict] = {}

        # Sets: track dataset splits
        self.train_ids: Set[int] = set()
        self.val_ids: Set[int] = set()
        self.test_ids: Set[int] = set()

        # List: maintain order for class labels
        self.class_names: List[str] = []

        # Dictionary: class -> list of sample IDs
        self.class_to_samples: Dict[str, List[int]] = defaultdict(list)

    def add_sample(self, sample_id: int, filepath: str,
                   class_label: str, metadata: Optional[Dict] = None):
        """
        Add a sample to the dataset.

        Args:
            sample_id: Unique sample identifier
            filepath: Path to data file
            class_label: Class label
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If sample_id already exists
        """
        if sample_id in self.samples:
            raise ValueError(f"Sample {sample_id} already exists")

        self.samples[sample_id] = {
            'filepath': filepath,
            'class': class_label,
            'metadata': metadata or {}
        }

        # Update class tracking
        if class_label not in self.class_names:
            self.class_names.append(class_label)

        self.class_to_samples[class_label].append(sample_id)

    def remove_sample(self, sample_id: int):
        """
        Remove a sample from the dataset.

        Args:
            sample_id: Sample to remove
        """
        if sample_id in self.samples:
            class_label = self.samples[sample_id]['class']
            self.class_to_samples[class_label].remove(sample_id)
            del self.samples[sample_id]

            # Remove from splits
            self.train_ids.discard(sample_id)
            self.val_ids.discard(sample_id)
            self.test_ids.discard(sample_id)

    def split_dataset(self, train_ratio: float = 0.7,
                     val_ratio: float = 0.15, seed: int = 42):
        """
        Split dataset into train/val/test.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed for reproducibility
        """
        random.seed(seed)

        all_ids = list(self.samples.keys())
        random.shuffle(all_ids)

        n = len(all_ids)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        self.train_ids = set(all_ids[:train_end])
        self.val_ids = set(all_ids[train_end:val_end])
        self.test_ids = set(all_ids[val_end:])

    def stratified_split(self, train_ratio: float = 0.7,
                        val_ratio: float = 0.15, seed: int = 42):
        """
        Split dataset maintaining class distribution.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed
        """
        random.seed(seed)

        self.train_ids = set()
        self.val_ids = set()
        self.test_ids = set()

        # Split each class separately
        for class_label, sample_ids in self.class_to_samples.items():
            ids_copy = sample_ids.copy()
            random.shuffle(ids_copy)

            n = len(ids_copy)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            self.train_ids.update(ids_copy[:train_end])
            self.val_ids.update(ids_copy[train_end:val_end])
            self.test_ids.update(ids_copy[val_end:])

    def validate_splits(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset splits have no overlap.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check overlaps
        train_val = self.train_ids & self.val_ids
        if train_val:
            issues.append(f"Train-Val overlap: {len(train_val)} samples")

        train_test = self.train_ids & self.test_ids
        if train_test:
            issues.append(f"Train-Test overlap: {len(train_test)} samples")

        val_test = self.val_ids & self.test_ids
        if val_test:
            issues.append(f"Val-Test overlap: {len(val_test)} samples")

        # Check all samples assigned
        all_split_ids = self.train_ids | self.val_ids | self.test_ids
        if len(all_split_ids) != len(self.samples):
            unassigned = len(self.samples) - len(all_split_ids)
            issues.append(f"{unassigned} samples not assigned to any split")

        return len(issues) == 0, issues

    def get_class_distribution(self, split: str = 'train') -> Dict[str, int]:
        """
        Get class distribution for a split.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            Dictionary mapping class -> count

        Raises:
            ValueError: If split name is invalid
        """
        if split == 'train':
            split_ids = self.train_ids
        elif split == 'val':
            split_ids = self.val_ids
        elif split == 'test':
            split_ids = self.test_ids
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        distribution = {}
        for class_name in self.class_names:
            class_samples = set(self.class_to_samples[class_name])
            count = len(class_samples & split_ids)
            distribution[class_name] = count

        return distribution

    def get_sample_batch(self, split: str, batch_size: int,
                        shuffle: bool = True) -> List[Dict]:
        """
        Get a batch of samples from a split.

        Args:
            split: One of 'train', 'val', 'test'
            batch_size: Number of samples to return
            shuffle: Whether to shuffle before selecting

        Returns:
            List of sample dictionaries
        """
        if split == 'train':
            split_ids = list(self.train_ids)
        elif split == 'val':
            split_ids = list(self.val_ids)
        elif split == 'test':
            split_ids = list(self.test_ids)
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        if shuffle:
            random.shuffle(split_ids)

        batch_ids = split_ids[:batch_size]
        return [self.samples[sid] for sid in batch_ids]

    def get_summary(self) -> Dict:
        """
        Get dataset summary statistics.

        Returns:
            Dictionary with summary information
        """
        return {
            'total_samples': len(self.samples),
            'num_classes': len(self.class_names),
            'classes': self.class_names,
            'train_samples': len(self.train_ids),
            'val_samples': len(self.val_ids),
            'test_samples': len(self.test_ids),
            'class_distribution': {
                cls: len(samples)
                for cls, samples in self.class_to_samples.items()
            }
        }

    def get_imbalance_ratio(self) -> float:
        """
        Calculate class imbalance ratio.

        Returns:
            Ratio of max to min class size (1.0 = balanced)
        """
        class_sizes = [len(samples) for samples in self.class_to_samples.values()]
        if not class_sizes or min(class_sizes) == 0:
            return float('inf')
        return max(class_sizes) / min(class_sizes)


def main():
    """Demonstrate comprehensive dataset manager."""
    print("=" * 60)
    print("ML Dataset Manager - Comprehensive Demo")
    print("=" * 60)
    print()

    manager = MLDatasetManager()

    # Add samples
    samples_data = [
        (1, "/data/cat_001.jpg", "cat", {"size": (224, 224), "augmented": False}),
        (2, "/data/dog_001.jpg", "dog", {"size": (256, 256), "augmented": False}),
        (3, "/data/cat_002.jpg", "cat", {"size": (224, 224), "augmented": True}),
        (4, "/data/bird_001.jpg", "bird", {"size": (512, 512), "augmented": False}),
        (5, "/data/dog_002.jpg", "dog", {"size": (256, 256), "augmented": False}),
        (6, "/data/cat_003.jpg", "cat", {"size": (224, 224), "augmented": False}),
        (7, "/data/bird_002.jpg", "bird", {"size": (512, 512), "augmented": True}),
        (8, "/data/dog_003.jpg", "dog", {"size": (256, 256), "augmented": False}),
        (9, "/data/cat_004.jpg", "cat", {"size": (224, 224), "augmented": False}),
        (10, "/data/bird_003.jpg", "bird", {"size": (512, 512), "augmented": False}),
    ]

    for sample_id, filepath, class_label, metadata in samples_data:
        manager.add_sample(sample_id, filepath, class_label, metadata)

    print(f"Added {len(manager.samples)} samples")
    print()

    # Split dataset (stratified)
    print("Stratified split (70/15/15)...")
    manager.stratified_split(train_ratio=0.6, val_ratio=0.2, seed=42)

    # Validate splits
    is_valid, issues = manager.validate_splits()
    print(f"Splits valid: {is_valid}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    print()

    # Get summary
    summary = manager.get_summary()
    print("Dataset Summary:")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Number of classes: {summary['num_classes']}")
    print(f"  Classes: {summary['classes']}")
    print(f"  Train samples: {summary['train_samples']}")
    print(f"  Val samples: {summary['val_samples']}")
    print(f"  Test samples: {summary['test_samples']}")
    print()

    print("Overall class distribution:")
    for cls, count in summary['class_distribution'].items():
        pct = (count / summary['total_samples']) * 100
        print(f"  {cls}: {count} samples ({pct:.1f}%)")
    print()

    # Check class distributions per split
    print("Class Distribution per Split:")
    for split in ['train', 'val', 'test']:
        dist = manager.get_class_distribution(split)
        print(f"  {split.upper()}:")
        for cls, count in dist.items():
            print(f"    {cls}: {count} samples")
    print()

    # Check imbalance
    imbalance = manager.get_imbalance_ratio()
    print(f"Class imbalance ratio: {imbalance:.2f}x")
    if imbalance > 2.0:
        print("  ⚠️  Dataset is imbalanced!")
    else:
        print("  ✓ Dataset is reasonably balanced")
    print()

    # Get a batch
    batch = manager.get_sample_batch('train', batch_size=3, shuffle=True)
    print(f"Sample batch from training ({len(batch)} samples):")
    for i, sample in enumerate(batch, 1):
        print(f"  {i}. {sample['filepath']} - {sample['class']}")

    print()
    print("✓ Dataset manager demonstration complete")


if __name__ == "__main__":
    main()
