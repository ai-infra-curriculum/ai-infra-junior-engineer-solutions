"""
Tests for metrics computation utilities.
"""

import pytest
import numpy as np
from pathlib import Path

from src.utils.metrics import compute_metrics, plot_confusion_matrix


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        predictions = [0, 0, 1, 1, 0, 1]
        labels = [0, 0, 1, 1, 0, 1]

        metrics = compute_metrics(predictions, labels)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        predictions = [1, 1, 0, 0]
        labels = [0, 0, 1, 1]

        metrics = compute_metrics(predictions, labels)

        assert metrics["accuracy"] == 0.0

    def test_mixed_predictions(self):
        """Test metrics with mixed predictions."""
        predictions = [0, 1, 1, 0, 1]
        labels = [0, 1, 0, 0, 1]

        metrics = compute_metrics(predictions, labels)

        # 4 correct out of 5
        assert metrics["accuracy"] == 0.8

        # Check metrics exist and are between 0 and 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_all_positive_predictions(self):
        """Test metrics when all predictions are positive."""
        predictions = [1, 1, 1, 1]
        labels = [0, 1, 0, 1]

        metrics = compute_metrics(predictions, labels)

        # Should handle division by zero gracefully
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_empty_lists(self):
        """Test metrics with empty lists."""
        predictions = []
        labels = []

        # Should handle gracefully or raise appropriate error
        try:
            metrics = compute_metrics(predictions, labels)
            # If it succeeds, check returned values
            assert isinstance(metrics, dict)
        except (ValueError, ZeroDivisionError):
            # Expected behavior for empty input
            pass


class TestPlotConfusionMatrix:
    """Test plot_confusion_matrix function."""

    def test_plot_saves_file(self, tmp_path):
        """Test confusion matrix plot saves to file."""
        predictions = [0, 1, 1, 0, 1, 0]
        labels = [0, 1, 0, 0, 1, 0]
        output_path = tmp_path / "confusion_matrix.png"

        plot_confusion_matrix(labels, predictions, output_path)

        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # File is not empty

    def test_plot_with_custom_class_names(self, tmp_path):
        """Test confusion matrix with custom class names."""
        predictions = [0, 1, 1, 0]
        labels = [0, 1, 0, 0]
        output_path = tmp_path / "confusion_matrix_custom.png"
        class_names = ["Bad", "Good"]

        # Should not raise error
        plot_confusion_matrix(labels, predictions, output_path, class_names=class_names)

        assert output_path.exists()

    def test_plot_creates_parent_directory(self, tmp_path):
        """Test confusion matrix creates parent directories."""
        output_path = tmp_path / "subdir" / "confusion_matrix.png"
        predictions = [0, 1, 0, 1]
        labels = [0, 1, 0, 1]

        plot_confusion_matrix(labels, predictions, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_plot_with_perfect_predictions(self, tmp_path):
        """Test confusion matrix with perfect predictions."""
        predictions = [0, 0, 1, 1]
        labels = [0, 0, 1, 1]
        output_path = tmp_path / "perfect_cm.png"

        plot_confusion_matrix(labels, predictions, output_path)

        assert output_path.exists()

    def test_plot_with_imbalanced_data(self, tmp_path):
        """Test confusion matrix with imbalanced data."""
        predictions = [1] * 10 + [0] * 2
        labels = [1] * 10 + [0] * 2
        output_path = tmp_path / "imbalanced_cm.png"

        plot_confusion_matrix(labels, predictions, output_path)

        assert output_path.exists()
