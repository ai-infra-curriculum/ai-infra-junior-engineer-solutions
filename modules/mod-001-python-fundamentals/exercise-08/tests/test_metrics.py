"""Tests for classification metrics."""

import pytest
from ml_infra_utils.metrics import accuracy, precision, recall, f1_score


class TestAccuracy:
    """Tests for accuracy metric."""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        assert accuracy(predictions, labels) == 1.0

    def test_accuracy_50_percent(self):
        """Test accuracy with 50% correct."""
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 0, 1]
        assert accuracy(predictions, labels) == 0.5

    def test_accuracy_zero(self):
        """Test accuracy with all incorrect."""
        predictions = [1, 1, 1, 1]
        labels = [0, 0, 0, 0]
        assert accuracy(predictions, labels) == 0.0

    def test_accuracy_empty(self):
        """Test accuracy with empty lists."""
        assert accuracy([], []) == 0.0

    def test_accuracy_length_mismatch_raises(self):
        """Test accuracy with mismatched lengths."""
        with pytest.raises(ValueError, match="must have same length"):
            accuracy([1, 0], [1, 0, 1])


class TestPrecision:
    """Tests for precision metric."""

    def test_precision_perfect(self):
        """Test precision with perfect predictions."""
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        assert precision(predictions, labels) == 1.0

    def test_precision_with_false_positives(self):
        """Test precision with false positives."""
        predictions = [1, 1, 1, 0]
        labels = [1, 0, 1, 0]
        # 2 true positives, 1 false positive
        assert precision(predictions, labels) == pytest.approx(2 / 3)

    def test_precision_no_predictions(self):
        """Test precision when no positive predictions."""
        predictions = [0, 0, 0, 0]
        labels = [1, 0, 1, 0]
        assert precision(predictions, labels) == 0.0

    def test_precision_length_mismatch_raises(self):
        """Test precision with mismatched lengths."""
        with pytest.raises(ValueError, match="must have same length"):
            precision([1, 0], [1, 0, 1])


class TestRecall:
    """Tests for recall metric."""

    def test_recall_perfect(self):
        """Test recall with perfect predictions."""
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        assert recall(predictions, labels) == 1.0

    def test_recall_with_false_negatives(self):
        """Test recall with false negatives."""
        predictions = [1, 0, 0, 0]
        labels = [1, 0, 1, 0]
        # 1 true positive, 1 false negative
        assert recall(predictions, labels) == 0.5

    def test_recall_no_actual_positives(self):
        """Test recall when no actual positives."""
        predictions = [1, 1, 1, 1]
        labels = [0, 0, 0, 0]
        assert recall(predictions, labels) == 0.0

    def test_recall_length_mismatch_raises(self):
        """Test recall with mismatched lengths."""
        with pytest.raises(ValueError, match="must have same length"):
            recall([1, 0], [1, 0, 1])


class TestF1Score:
    """Tests for F1 score metric."""

    def test_f1_perfect(self):
        """Test F1 score with perfect predictions."""
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        assert f1_score(predictions, labels) == 1.0

    def test_f1_balanced(self):
        """Test F1 score with balanced errors."""
        predictions = [1, 1, 0, 0]
        labels = [1, 0, 1, 0]
        # Precision = 0.5 (1 TP, 1 FP)
        # Recall = 0.5 (1 TP, 1 FN)
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert f1_score(predictions, labels) == 0.5

    def test_f1_zero(self):
        """Test F1 score when zero."""
        predictions = [0, 0, 0, 0]
        labels = [1, 1, 1, 1]
        assert f1_score(predictions, labels) == 0.0

    def test_f1_high_precision_low_recall(self):
        """Test F1 score with high precision but low recall."""
        predictions = [1, 0, 0, 0, 0]
        labels = [1, 1, 1, 0, 0]
        # Precision = 1.0 (1 TP, 0 FP)
        # Recall = 1/3 (1 TP, 2 FN)
        # F1 = 2 * (1.0 * 1/3) / (1.0 + 1/3) = 0.5
        assert f1_score(predictions, labels) == 0.5


class TestMultiClass:
    """Tests for multi-class classification metrics."""

    def test_precision_multiclass(self):
        """Test precision with multiple classes."""
        predictions = [0, 1, 2, 1, 2]
        labels = [0, 1, 2, 0, 2]
        # For class 1: 1 TP, 1 FP -> precision = 0.5
        assert precision(predictions, labels, positive_class=1) == 0.5

    def test_recall_multiclass(self):
        """Test recall with multiple classes."""
        predictions = [0, 1, 2, 1, 2]
        labels = [0, 1, 2, 0, 2]
        # For class 2: 2 TP, 0 FN -> recall = 1.0
        assert recall(predictions, labels, positive_class=2) == 1.0

    def test_f1_multiclass(self):
        """Test F1 score with multiple classes."""
        predictions = [0, 1, 2, 1, 2]
        labels = [0, 1, 2, 0, 2]
        # For class 0: 1 TP, 0 FP, 1 FN
        # Precision = 1.0, Recall = 0.5, F1 = 2/3
        assert f1_score(predictions, labels, positive_class=0) == pytest.approx(2 / 3)
