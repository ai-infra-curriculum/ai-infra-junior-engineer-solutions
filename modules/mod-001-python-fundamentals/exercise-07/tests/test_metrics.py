"""Tests for ML metrics calculations."""

import pytest
from src.metrics import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score
)


# ============================================================================
# Accuracy Tests
# ============================================================================

class TestCalculateAccuracy:
    """Test suite for accuracy calculation."""

    def test_accuracy_perfect(self):
        """Test perfect accuracy."""
        preds = [1, 0, 1, 1]
        labels = [1, 0, 1, 1]
        assert calculate_accuracy(preds, labels) == 1.0

    def test_accuracy_zero(self):
        """Test zero accuracy."""
        preds = [1, 1, 1, 1]
        labels = [0, 0, 0, 0]
        assert calculate_accuracy(preds, labels) == 0.0

    def test_accuracy_half(self):
        """Test 50% accuracy."""
        preds = [1, 0, 1, 0]
        labels = [1, 1, 0, 0]
        assert calculate_accuracy(preds, labels) == 0.5

    def test_accuracy_empty(self):
        """Test empty inputs."""
        assert calculate_accuracy([], []) == 0.0

    def test_accuracy_length_mismatch(self):
        """Test mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            calculate_accuracy([1, 0], [1])

    @pytest.mark.parametrize("preds,labels,expected", [
        ([1, 1, 1], [1, 1, 1], 1.0),        # Perfect
        ([0, 0, 0], [1, 1, 1], 0.0),        # Zero
        ([1, 0, 1, 0], [1, 0, 0, 1], 0.5),  # Half
        ([1, 0], [1, 0], 1.0),              # Small
    ])
    def test_accuracy_parametrized(self, preds, labels, expected):
        """Test various accuracy scenarios."""
        result = calculate_accuracy(preds, labels)
        assert abs(result - expected) < 1e-6

    def test_with_fixtures(self, sample_predictions, sample_labels):
        """Test using fixtures."""
        accuracy = calculate_accuracy(sample_predictions, sample_labels)
        assert 0.0 <= accuracy <= 1.0


# ============================================================================
# Precision Tests
# ============================================================================

class TestCalculatePrecision:
    """Test suite for precision calculation."""

    def test_precision_perfect(self):
        """Test perfect precision."""
        preds = [1, 1, 1]
        labels = [1, 1, 1]
        assert calculate_precision(preds, labels) == 1.0

    def test_precision_no_true_positives(self):
        """Test when no true positives."""
        preds = [1, 1, 1]
        labels = [0, 0, 0]
        assert calculate_precision(preds, labels) == 0.0

    def test_precision_no_predictions(self):
        """Test when no positive predictions."""
        preds = [0, 0, 0]
        labels = [1, 1, 1]
        assert calculate_precision(preds, labels) == 0.0

    def test_precision_mixed(self):
        """Test mixed predictions."""
        preds = [1, 1, 0, 1]
        labels = [1, 0, 0, 1]
        # TP=2, FP=1, Precision = 2/3
        expected = 2.0 / 3.0
        result = calculate_precision(preds, labels)
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize("preds,labels,expected", [
        ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),  # Perfect
        ([1, 1, 0, 0], [1, 0, 1, 0], 0.5),  # Half
        ([0, 0, 0, 0], [1, 1, 1, 1], 0.0),  # No predictions
    ])
    def test_precision_parametrized(self, preds, labels, expected):
        """Test various precision scenarios."""
        result = calculate_precision(preds, labels)
        assert abs(result - expected) < 1e-6


# ============================================================================
# Recall Tests
# ============================================================================

class TestCalculateRecall:
    """Test suite for recall calculation."""

    def test_recall_perfect(self):
        """Test perfect recall."""
        preds = [1, 1, 1]
        labels = [1, 1, 1]
        assert calculate_recall(preds, labels) == 1.0

    def test_recall_no_true_positives(self):
        """Test when no true positives."""
        preds = [0, 0, 0]
        labels = [1, 1, 1]
        assert calculate_recall(preds, labels) == 0.0

    def test_recall_no_actual_positives(self):
        """Test when no actual positives."""
        preds = [1, 1, 1]
        labels = [0, 0, 0]
        assert calculate_recall(preds, labels) == 0.0

    def test_recall_mixed(self):
        """Test mixed predictions."""
        preds = [1, 0, 0, 1]
        labels = [1, 1, 0, 1]
        # TP=2, FN=1, Recall = 2/3
        expected = 2.0 / 3.0
        result = calculate_recall(preds, labels)
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize("preds,labels,expected", [
        ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),  # Perfect
        ([1, 0, 1, 0], [1, 1, 0, 0], 0.5),  # Half
        ([0, 0, 0, 0], [1, 1, 1, 1], 0.0),  # No predictions
    ])
    def test_recall_parametrized(self, preds, labels, expected):
        """Test various recall scenarios."""
        result = calculate_recall(preds, labels)
        assert abs(result - expected) < 1e-6


# ============================================================================
# F1 Score Tests
# ============================================================================

class TestCalculateF1Score:
    """Test suite for F1 score calculation."""

    def test_f1_perfect(self):
        """Test perfect F1 score."""
        preds = [1, 1, 1]
        labels = [1, 1, 1]
        assert calculate_f1_score(preds, labels) == 1.0

    def test_f1_zero(self):
        """Test zero F1 score."""
        preds = [1, 1, 1]
        labels = [0, 0, 0]
        assert calculate_f1_score(preds, labels) == 0.0

    def test_f1_balanced(self):
        """Test balanced precision and recall."""
        # Precision = 2/3, Recall = 2/3, F1 = 2/3
        preds = [1, 1, 0, 1]
        labels = [1, 0, 1, 1]
        expected = 2.0 / 3.0
        result = calculate_f1_score(preds, labels)
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize("preds,labels,expected", [
        ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),  # Perfect
        ([1, 1, 0, 0], [1, 1, 1, 1], 2/3),  # Unbalanced
        ([0, 0, 0, 0], [1, 1, 1, 1], 0.0),  # No predictions
    ])
    def test_f1_parametrized(self, preds, labels, expected):
        """Test various F1 scenarios."""
        result = calculate_f1_score(preds, labels)
        assert abs(result - expected) < 1e-6


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestMetricsEdgeCases:
    """Test edge cases for all metrics."""

    @pytest.mark.parametrize("metric_func", [
        calculate_accuracy,
        calculate_precision,
        calculate_recall,
        calculate_f1_score,
    ])
    def test_empty_input(self, metric_func):
        """Test all metrics with empty input."""
        if metric_func == calculate_accuracy:
            result = metric_func([], [])
            assert result == 0.0
        else:
            # Precision, recall, F1 return 0 for empty
            result = metric_func([], [])
            assert result == 0.0

    @pytest.mark.parametrize("metric_func", [
        calculate_accuracy,
        calculate_precision,
        calculate_recall,
        calculate_f1_score,
    ])
    def test_length_mismatch(self, metric_func):
        """Test all metrics with mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            metric_func([1, 0], [1])

    def test_all_metrics_consistency(self):
        """Test that all metrics are consistent."""
        preds = [1, 1, 0, 1]
        labels = [1, 0, 1, 1]

        accuracy = calculate_accuracy(preds, labels)
        precision = calculate_precision(preds, labels)
        recall = calculate_recall(preds, labels)
        f1 = calculate_f1_score(preds, labels)

        # All should be between 0 and 1
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0

        # F1 should be harmonic mean of precision and recall
        if precision + recall > 0:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            assert abs(f1 - expected_f1) < 1e-6
