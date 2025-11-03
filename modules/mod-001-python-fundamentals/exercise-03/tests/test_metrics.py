"""
Tests for ml_utils.metrics module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_utils import metrics


class TestClassificationMetrics:
    """Test classification metrics."""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        preds = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        assert metrics.accuracy(preds, labels) == 1.0

    def test_accuracy_half(self):
        """Test accuracy with 50% correct."""
        preds = [1, 0, 1, 0]
        labels = [0, 1, 1, 0]
        assert metrics.accuracy(preds, labels) == 0.5

    def test_accuracy_empty(self):
        """Test accuracy with empty lists."""
        assert metrics.accuracy([], []) == 0.0

    def test_accuracy_mismatched_lengths(self):
        """Test accuracy raises error for mismatched lengths."""
        with pytest.raises(ValueError):
            metrics.accuracy([1, 0], [1])

    def test_precision(self):
        """Test precision calculation."""
        preds = [1, 1, 0, 1]
        labels = [1, 0, 0, 1]
        # TP=2, FP=1 → precision = 2/3
        prec = metrics.precision(preds, labels)
        assert abs(prec - 2/3) < 0.01

    def test_recall(self):
        """Test recall calculation."""
        preds = [1, 1, 0, 1]
        labels = [1, 0, 0, 1]
        # TP=2, FN=0 → recall = 2/2 = 1.0
        rec = metrics.recall(preds, labels)
        assert rec == 1.0

    def test_f1_score(self):
        """Test F1 score calculation."""
        preds = [1, 1, 0, 1]
        labels = [1, 0, 0, 1]
        f1 = metrics.f1_score(preds, labels)
        # F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 0.8
        assert abs(f1 - 0.8) < 0.01

    def test_confusion_matrix(self):
        """Test confusion matrix."""
        preds = [0, 1, 0, 1]
        labels = [0, 1, 1, 1]
        cm = metrics.confusion_matrix(preds, labels, num_classes=2)
        assert cm[0][0] == 1  # TN
        assert cm[0][1] == 1  # FP
        assert cm[1][0] == 0  # FN
        assert cm[1][1] == 2  # TP

    def test_classification_report(self):
        """Test classification report."""
        preds = [1, 0, 1, 1]
        labels = [1, 0, 0, 1]
        report = metrics.classification_report(preds, labels,
                                               ["neg", "pos"])
        assert "accuracy" in report
        assert "neg" in report
        assert "pos" in report
        assert 0 <= report["accuracy"] <= 1


class TestRegressionMetrics:
    """Test regression metrics."""

    def test_mse(self):
        """Test mean squared error."""
        preds = [1.0, 2.0, 3.0]
        labels = [1.0, 2.0, 3.0]
        mse = metrics.mean_squared_error(preds, labels)
        assert mse == 0.0

    def test_mse_with_errors(self):
        """Test MSE with actual errors."""
        preds = [1.0, 2.0, 3.0]
        labels = [2.0, 3.0, 4.0]
        # Errors: [-1, -1, -1], squared: [1, 1, 1], mean: 1.0
        mse = metrics.mean_squared_error(preds, labels)
        assert mse == 1.0

    def test_rmse(self):
        """Test root mean squared error."""
        preds = [1.0, 2.0, 3.0]
        labels = [2.0, 3.0, 4.0]
        rmse = metrics.root_mean_squared_error(preds, labels)
        assert rmse == 1.0

    def test_mae(self):
        """Test mean absolute error."""
        preds = [1.0, 2.0, 3.0]
        labels = [2.0, 3.0, 4.0]
        mae = metrics.mean_absolute_error(preds, labels)
        assert mae == 1.0

    def test_r_squared_perfect(self):
        """Test R² with perfect predictions."""
        preds = [1.0, 2.0, 3.0, 4.0]
        labels = [1.0, 2.0, 3.0, 4.0]
        r2 = metrics.r_squared(preds, labels)
        assert r2 == 1.0

    def test_r_squared_baseline(self):
        """Test R² with baseline predictions."""
        labels = [1.0, 2.0, 3.0, 4.0]
        mean_val = sum(labels) / len(labels)
        preds = [mean_val] * len(labels)
        r2 = metrics.r_squared(preds, labels)
        assert abs(r2 - 0.0) < 0.01

    def test_regression_report(self):
        """Test regression report."""
        preds = [1.0, 2.0, 3.0]
        labels = [1.1, 2.1, 2.9]
        report = metrics.regression_report(preds, labels)
        assert "mse" in report
        assert "rmse" in report
        assert "mae" in report
        assert "r_squared" in report


def test_empty_predictions():
    """Test that empty predictions are handled."""
    assert metrics.accuracy([], []) == 0.0
    assert metrics.mean_squared_error([], []) == 0.0


def test_mismatched_lengths():
    """Test that mismatched lengths raise errors."""
    with pytest.raises(ValueError):
        metrics.accuracy([1], [1, 0])

    with pytest.raises(ValueError):
        metrics.mean_squared_error([1.0], [1.0, 2.0])
