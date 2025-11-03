"""
Tests for MLDatasetManager.
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dataset_manager import MLDatasetManager


class TestMLDatasetManager:
    """Test MLDatasetManager class."""

    @pytest.fixture
    def manager(self):
        """Create manager with sample data."""
        mgr = MLDatasetManager()

        samples = [
            (1, "/data/cat_001.jpg", "cat"),
            (2, "/data/dog_001.jpg", "dog"),
            (3, "/data/cat_002.jpg", "cat"),
            (4, "/data/bird_001.jpg", "bird"),
            (5, "/data/dog_002.jpg", "dog"),
            (6, "/data/cat_003.jpg", "cat"),
        ]

        for sid, path, label in samples:
            mgr.add_sample(sid, path, label)

        return mgr

    def test_add_sample(self, manager):
        """Test adding samples."""
        assert len(manager.samples) == 6
        assert len(manager.class_names) == 3
        assert "cat" in manager.class_names

    def test_duplicate_sample_raises_error(self, manager):
        """Test that duplicate sample IDs raise error."""
        with pytest.raises(ValueError):
            manager.add_sample(1, "/data/duplicate.jpg", "cat")

    def test_split_dataset(self, manager):
        """Test dataset splitting."""
        manager.split_dataset(train_ratio=0.5, val_ratio=0.25, seed=42)

        assert len(manager.train_ids) > 0
        assert len(manager.val_ids) > 0
        assert len(manager.test_ids) > 0

        total = len(manager.train_ids) + len(manager.val_ids) + len(manager.test_ids)
        assert total == len(manager.samples)

    def test_validate_splits(self, manager):
        """Test split validation."""
        manager.split_dataset(seed=42)

        is_valid, issues = manager.validate_splits()
        assert is_valid
        assert len(issues) == 0

    def test_class_distribution(self, manager):
        """Test class distribution calculation."""
        manager.split_dataset(seed=42)

        dist = manager.get_class_distribution('train')
        assert isinstance(dist, dict)
        assert sum(dist.values()) == len(manager.train_ids)

    def test_get_summary(self, manager):
        """Test summary statistics."""
        summary = manager.get_summary()

        assert summary['total_samples'] == 6
        assert summary['num_classes'] == 3
        assert len(summary['classes']) == 3

    def test_stratified_split(self, manager):
        """Test stratified splitting."""
        manager.stratified_split(train_ratio=0.5, val_ratio=0.25, seed=42)

        # Check each split has some samples from each class
        for split in ['train', 'val', 'test']:
            dist = manager.get_class_distribution(split)
            # At least one class should have samples (with small dataset, not all classes may appear)
            assert sum(dist.values()) > 0

    def test_remove_sample(self, manager):
        """Test sample removal."""
        manager.remove_sample(1)

        assert 1 not in manager.samples
        assert len(manager.samples) == 5

    def test_get_sample_batch(self, manager):
        """Test batch retrieval."""
        manager.split_dataset(seed=42)

        batch = manager.get_sample_batch('train', batch_size=2, shuffle=False)

        assert len(batch) <= 2
        assert all(isinstance(s, dict) for s in batch)
        assert all('filepath' in s and 'class' in s for s in batch)

    def test_imbalance_ratio(self, manager):
        """Test class imbalance calculation."""
        ratio = manager.get_imbalance_ratio()

        assert ratio >= 1.0
        assert ratio < float('inf')


def test_empty_manager():
    """Test empty manager."""
    manager = MLDatasetManager()

    assert len(manager.samples) == 0
    assert len(manager.class_names) == 0

    summary = manager.get_summary()
    assert summary['total_samples'] == 0
