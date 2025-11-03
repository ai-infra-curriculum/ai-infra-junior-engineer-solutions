#!/usr/bin/env python3
"""
Batch processing class for ML training data.

Implements efficient batch creation with stratified sampling support.
"""

import random
from typing import List, Dict, Any, Optional
from collections import defaultdict


class DataBatchProcessor:
    """Process data in batches for ML training."""

    def __init__(self, data: List[Any], batch_size: int, shuffle: bool = True):
        """
        Initialize batch processor.

        Args:
            data: List of data samples
            batch_size: Size of each batch
            shuffle: Whether to shuffle data before batching
        """
        self.data = data.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_batches(self) -> List[List[Any]]:
        """
        Generate batches from data.

        Returns:
            List of batches
        """
        if self.shuffle:
            random.shuffle(self.data)

        batches = []
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i + self.batch_size]
            batches.append(batch)

        return batches

    def get_batch_statistics(self) -> Dict[str, int]:
        """
        Calculate batch statistics.

        Returns:
            Dictionary with batch statistics
        """
        num_batches = (len(self.data) + self.batch_size - 1) // self.batch_size
        last_batch_size = len(self.data) % self.batch_size or self.batch_size

        return {
            "total_samples": len(self.data),
            "batch_size": self.batch_size,
            "num_batches": num_batches,
            "last_batch_size": last_batch_size
        }

    def get_batch_indices(self) -> List[List[int]]:
        """
        Get batch indices instead of data.

        Returns:
            List of batches containing indices
        """
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)

        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            batches.append(batch)

        return batches

    def drop_last(self) -> List[List[Any]]:
        """
        Get batches, dropping the last incomplete batch.

        Returns:
            List of complete batches only
        """
        batches = self.get_batches()

        # Drop last if incomplete
        if len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        return batches


class StratifiedBatchProcessor:
    """Batch processor with stratified sampling."""

    def __init__(self, data: List[tuple], batch_size: int, shuffle: bool = True):
        """
        Initialize stratified batch processor.

        Args:
            data: List of (sample, label) tuples
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group data by class
        self.class_data: Dict[Any, List[tuple]] = defaultdict(list)
        for sample in data:
            _, label = sample
            self.class_data[label].append(sample)

    def get_stratified_batches(self) -> List[List[tuple]]:
        """
        Create batches with balanced class distribution.

        Returns:
            List of stratified batches
        """
        num_classes = len(self.class_data)
        samples_per_class = self.batch_size // num_classes

        # Prepare class iterators
        class_iters = {}
        for label, samples in self.class_data.items():
            if self.shuffle:
                random.shuffle(samples)
            class_iters[label] = iter(samples)

        batches = []
        batch = []

        try:
            while True:
                # Take samples from each class
                for label in sorted(class_iters.keys()):
                    for _ in range(samples_per_class):
                        batch.append(next(class_iters[label]))

                # Fill remaining spots
                while len(batch) < self.batch_size:
                    for label in sorted(class_iters.keys()):
                        if len(batch) >= self.batch_size:
                            break
                        try:
                            batch.append(next(class_iters[label]))
                        except StopIteration:
                            continue

                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []

        except StopIteration:
            # Add remaining samples as last batch
            if batch:
                batches.append(batch)

        return batches


def main():
    """Demonstrate batch processing."""
    print("=" * 60)
    print("Batch Processing for ML Training")
    print("=" * 60)
    print()

    # Example 1: Basic batching
    sample_ids = list(range(1, 101))  # 100 samples

    processor = DataBatchProcessor(sample_ids, batch_size=16)
    batches = processor.get_batches()

    print(f"Generated {len(batches)} batches")
    print(f"First batch: {batches[0]}")
    print(f"Last batch size: {len(batches[-1])}")
    print()

    stats = processor.get_batch_statistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Example 2: Batch indices
    print("=" * 60)
    print("Batch Indices")
    print("=" * 60)
    print()

    processor2 = DataBatchProcessor(list(range(20)), batch_size=5, shuffle=False)
    indices = processor2.get_batch_indices()
    print(f"Batch indices: {indices}")
    print()

    # Example 3: Drop last incomplete batch
    print("=" * 60)
    print("Drop Last Incomplete Batch")
    print("=" * 60)
    print()

    processor3 = DataBatchProcessor(list(range(25)), batch_size=8, shuffle=False)
    all_batches = processor3.get_batches()
    complete_batches = processor3.drop_last()

    print(f"All batches: {len(all_batches)} (last has {len(all_batches[-1])} samples)")
    print(f"Complete batches: {len(complete_batches)}")
    print()

    # Example 4: Stratified batching
    print("=" * 60)
    print("Stratified Batching")
    print("=" * 60)
    print()

    # Create imbalanced dataset
    data = []
    for i in range(30):
        data.append((f"sample_{i}", "cat"))
    for i in range(30, 55):
        data.append((f"sample_{i}", "dog"))
    for i in range(55, 70):
        data.append((f"sample_{i}", "bird"))

    random.seed(42)
    strat_processor = StratifiedBatchProcessor(data, batch_size=12, shuffle=True)
    strat_batches = strat_processor.get_stratified_batches()

    print(f"Created {len(strat_batches)} stratified batches")
    print("\nFirst batch class distribution:")
    first_batch = strat_batches[0]
    class_counts = {}
    for _, label in first_batch:
        class_counts[label] = class_counts.get(label, 0) + 1
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} samples")

    print()
    print("✓ Batch processing demonstration complete")


if __name__ == "__main__":
    main()
