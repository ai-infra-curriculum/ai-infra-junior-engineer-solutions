"""
Tests for data loading utilities.
"""

import pytest
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.utils.data_loader import load_dataset, SentimentDataset


class TestSentimentDataset:
    """Test SentimentDataset class."""

    @pytest.fixture
    def tokenizer(self):
        """Load tokenizer for testing."""
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        texts = [
            "This is a great product!",
            "Terrible experience, would not recommend.",
            "Average quality, nothing special.",
        ]
        labels = [1, 0, 0]
        return texts, labels

    def test_dataset_length(self, sample_data, tokenizer):
        """Test dataset returns correct length."""
        texts, labels = sample_data
        dataset = SentimentDataset(texts, labels, tokenizer)
        assert len(dataset) == 3

    def test_dataset_item_structure(self, sample_data, tokenizer):
        """Test dataset item has correct structure."""
        texts, labels = sample_data
        dataset = SentimentDataset(texts, labels, tokenizer)

        item = dataset[0]

        # Check keys
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

        # Check types
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_dataset_max_length(self, sample_data, tokenizer):
        """Test dataset respects max_length parameter."""
        texts, labels = sample_data
        max_length = 64

        dataset = SentimentDataset(texts, labels, tokenizer, max_length=max_length)
        item = dataset[0]

        assert item["input_ids"].shape[0] == max_length
        assert item["attention_mask"].shape[0] == max_length

    def test_dataset_labels(self, sample_data, tokenizer):
        """Test dataset returns correct labels."""
        texts, labels = sample_data
        dataset = SentimentDataset(texts, labels, tokenizer)

        for idx, expected_label in enumerate(labels):
            item = dataset[idx]
            assert item["labels"].item() == expected_label

    def test_dataset_with_empty_text(self, tokenizer):
        """Test dataset handles empty text."""
        texts = ["", "Valid text"]
        labels = [0, 1]

        dataset = SentimentDataset(texts, labels, tokenizer)

        # Should not raise error
        item = dataset[0]
        assert item["input_ids"].shape[0] > 0


class TestLoadDataset:
    """Test load_dataset function."""

    @pytest.fixture
    def temp_csv(self, tmp_path):
        """Create temporary CSV file for testing."""
        df = pd.DataFrame({
            "text": [
                "Great product!",
                "Bad quality",
                "Excellent service",
                "Disappointed",
                "Highly recommend",
            ],
            "label": [1, 0, 1, 0, 1],
        })

        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_load_dataset_returns_dataframes(self, temp_csv):
        """Test load_dataset returns DataFrames."""
        train_df, test_df = load_dataset(temp_csv, test_size=0.2)

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_load_dataset_split_ratio(self, temp_csv):
        """Test load_dataset respects test_size parameter."""
        train_df, test_df = load_dataset(temp_csv, test_size=0.4)

        total_samples = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total_samples

        # Allow small tolerance due to rounding
        assert abs(test_ratio - 0.4) < 0.1

    def test_load_dataset_columns(self, temp_csv):
        """Test load_dataset preserves required columns."""
        train_df, test_df = load_dataset(temp_csv)

        assert "text" in train_df.columns
        assert "label" in train_df.columns
        assert "text" in test_df.columns
        assert "label" in test_df.columns

    def test_load_dataset_missing_file(self):
        """Test load_dataset raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_file.csv")

    def test_load_dataset_reproducibility(self, temp_csv):
        """Test load_dataset produces same split with same seed."""
        train_df1, test_df1 = load_dataset(temp_csv, random_state=42)
        train_df2, test_df2 = load_dataset(temp_csv, random_state=42)

        pd.testing.assert_frame_equal(train_df1, train_df2)
        pd.testing.assert_frame_equal(test_df1, test_df2)
