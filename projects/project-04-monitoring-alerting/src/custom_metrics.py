"""Custom ML metrics for production monitoring.

This module implements ML-specific metrics that go beyond standard
application monitoring:

- Data drift detection (distribution shift in features)
- Model performance degradation tracking
- Prediction confidence analysis
- Data quality monitoring

References:
- scipy.stats: https://docs.scipy.org/doc/scipy/reference/stats.html
- Evidently AI: https://evidentlyai.com/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Prometheus exporters are looked up lazily from instrumentation.py so this
# module remains importable in test contexts without a Prometheus registry.
try:
    from .instrumentation import (
        data_drift_score,
        missing_features_total,
        model_accuracy,
    )
except Exception:  # pragma: no cover - exercised only when instrumentation absent
    data_drift_score = None
    missing_features_total = None
    model_accuracy = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftDetectionResult:
    """One feature's drift-detection outcome."""

    feature_name: str
    statistic: float
    p_value: Optional[float]
    is_drift: bool
    test_method: str
    timestamp: datetime


@dataclass(frozen=True)
class ModelPerformanceMetrics:
    """Snapshot of model classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_count: int
    timestamp: datetime


# ---------------------------------------------------------------------------
# Data Drift Detection
# ---------------------------------------------------------------------------


# PSI bins above 0.25 traditionally indicate "significant" distribution shift.
_PSI_DRIFT_THRESHOLD = 0.25
# JS divergence above 0.1 is a conservative "noticeable shift" boundary.
_JS_DRIFT_THRESHOLD = 0.1
_PSI_EPSILON = 1e-10


class DataDriftDetector:
    """Detect distribution shifts in input data using statistical tests.

    Implements:
        - Kolmogorov-Smirnov (``ks``) — continuous features.
        - Population Stability Index (``psi``) — binned distributions.
        - Jensen-Shannon divergence (``js``) — probability distributions.
    """

    SUPPORTED_METHODS = ("ks", "psi", "js")

    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.05,
        method: str = "ks",
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Choose from {self.SUPPORTED_METHODS}."
            )
        reference_data = np.asarray(reference_data)
        if reference_data.ndim != 2:
            raise ValueError("reference_data must be 2D (n_samples, n_features).")
        if len(feature_names) != reference_data.shape[1]:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) must match "
                f"number of features ({reference_data.shape[1]})."
            )
        self.reference_data = reference_data
        self.feature_names = list(feature_names)
        self.threshold = threshold
        self.method = method
        self.n_features = reference_data.shape[1]

    # -- statistical tests -------------------------------------------------

    def kolmogorov_smirnov_test(
        self, reference: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float]:
        """Return (statistic, p_value) for the two-sample KS test."""
        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)

    def population_stability_index(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """Population Stability Index between two 1D distributions."""
        ref_hist, bin_edges = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bin_edges)
        ref_pct = ref_hist / max(len(reference), 1)
        cur_pct = cur_hist / max(len(current), 1)
        ref_pct = np.where(ref_pct == 0, _PSI_EPSILON, ref_pct)
        cur_pct = np.where(cur_pct == 0, _PSI_EPSILON, cur_pct)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def jensen_shannon_divergence(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 50
    ) -> float:
        """Jensen-Shannon distance between two 1D distributions."""
        ref_hist, bin_edges = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        ref_sum = np.sum(ref_hist) or 1.0
        cur_sum = np.sum(cur_hist) or 1.0
        ref_prob = ref_hist / ref_sum
        cur_prob = cur_hist / cur_sum
        return float(jensenshannon(ref_prob, cur_prob))

    # -- top-level drift sweep --------------------------------------------

    def detect_drift(self, current_data: np.ndarray) -> List[DriftDetectionResult]:
        """Run the configured drift test across every feature."""
        current_data = np.asarray(current_data)
        if current_data.shape[1] != self.n_features:
            raise ValueError(
                f"current_data has {current_data.shape[1]} features; "
                f"expected {self.n_features}."
            )

        results: List[DriftDetectionResult] = []
        now = datetime.now()
        for index, feature_name in enumerate(self.feature_names):
            reference_feature = self.reference_data[:, index]
            current_feature = current_data[:, index]

            if self.method == "ks":
                statistic, p_value = self.kolmogorov_smirnov_test(
                    reference_feature, current_feature
                )
                is_drift = p_value < self.threshold
            elif self.method == "psi":
                statistic = self.population_stability_index(
                    reference_feature, current_feature
                )
                p_value = None
                is_drift = statistic > _PSI_DRIFT_THRESHOLD
            else:  # method == "js"
                statistic = self.jensen_shannon_divergence(
                    reference_feature, current_feature
                )
                p_value = None
                is_drift = statistic > _JS_DRIFT_THRESHOLD

            result = DriftDetectionResult(
                feature_name=feature_name,
                statistic=statistic,
                p_value=p_value,
                is_drift=is_drift,
                test_method=self.method,
                timestamp=now,
            )
            results.append(result)

            if is_drift:
                logger.warning(
                    "Drift detected in %s: statistic=%.4f p_value=%s method=%s",
                    feature_name,
                    statistic,
                    f"{p_value:.4f}" if p_value is not None else "n/a",
                    self.method,
                )
        return results

    def export_drift_metrics(self, drift_results: List[DriftDetectionResult]) -> None:
        """Push drift statistics into Prometheus (if available) and logs."""
        for result in drift_results:
            if data_drift_score is not None:
                data_drift_score.labels(feature_name=result.feature_name).set(
                    result.statistic
                )
            logger.info(
                "Drift detection result",
                extra={
                    "feature_name": result.feature_name,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "is_drift": result.is_drift,
                    "method": result.test_method,
                },
            )


# ---------------------------------------------------------------------------
# Model Performance Monitor
# ---------------------------------------------------------------------------


@dataclass
class ModelPerformanceMonitor:
    """Track model classification metrics across a rolling stream."""

    model_name: str
    min_samples: int = 100
    _predictions: List[int] = field(default_factory=list)
    _ground_truth: List[int] = field(default_factory=list)
    _prediction_timestamps: List[datetime] = field(default_factory=list)
    _pending: Dict[str, int] = field(default_factory=dict)

    def log_prediction(
        self,
        prediction: int,
        ground_truth: Optional[int] = None,
        prediction_id: Optional[str] = None,
    ) -> None:
        """Record a prediction (with optional immediate label)."""
        self._predictions.append(int(prediction))
        self._prediction_timestamps.append(datetime.now())
        if ground_truth is not None:
            self._ground_truth.append(int(ground_truth))
        elif prediction_id is not None:
            # Buffer the prediction until a label arrives via add_ground_truth.
            self._pending[prediction_id] = int(prediction)

    def add_ground_truth(self, prediction_id: str, ground_truth: int) -> None:
        """Attach a label to a previously buffered prediction."""
        if prediction_id not in self._pending:
            logger.warning(
                "Ground truth received for unknown prediction_id=%s", prediction_id
            )
            return
        del self._pending[prediction_id]
        # Pair the label with the most recent prediction lacking a label.
        self._ground_truth.append(int(ground_truth))

    def calculate_metrics(self) -> Optional[ModelPerformanceMetrics]:
        """Compute classification metrics over the buffered samples."""
        if len(self._ground_truth) < self.min_samples:
            logger.debug(
                "Not enough samples for metrics calculation (%d / %d)",
                len(self._ground_truth),
                self.min_samples,
            )
            return None

        # Lazy import keeps sklearn out of the import path for tests that
        # don't exercise this code.
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        labeled_predictions = self._predictions[: len(self._ground_truth)]
        accuracy = float(accuracy_score(self._ground_truth, labeled_predictions))
        precision = float(
            precision_score(
                self._ground_truth,
                labeled_predictions,
                average="weighted",
                zero_division=0,
            )
        )
        recall = float(
            recall_score(
                self._ground_truth,
                labeled_predictions,
                average="weighted",
                zero_division=0,
            )
        )
        f1 = float(
            f1_score(
                self._ground_truth,
                labeled_predictions,
                average="weighted",
                zero_division=0,
            )
        )

        metrics = ModelPerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sample_count=len(self._ground_truth),
            timestamp=datetime.now(),
        )

        if model_accuracy is not None:
            model_accuracy.labels(model_name=self.model_name).set(accuracy)

        logger.info(
            "Model performance metrics: accuracy=%.4f precision=%.4f "
            "recall=%.4f f1=%.4f samples=%d",
            accuracy,
            precision,
            recall,
            f1,
            metrics.sample_count,
        )
        return metrics

    def check_degradation(
        self, baseline_accuracy: float, threshold: float = 0.1
    ) -> bool:
        """Return True when measured accuracy has dropped beyond ``threshold``."""
        if len(self._ground_truth) < self.min_samples:
            return False
        from sklearn.metrics import accuracy_score

        labeled_predictions = self._predictions[: len(self._ground_truth)]
        current_accuracy = float(accuracy_score(self._ground_truth, labeled_predictions))
        degradation = baseline_accuracy - current_accuracy
        if degradation > threshold:
            logger.error(
                "Performance degradation detected: baseline=%.4f current=%.4f "
                "degradation=%.4f",
                baseline_accuracy,
                current_accuracy,
                degradation,
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Confidence Analyzer
# ---------------------------------------------------------------------------


class ConfidenceAnalyzer:
    """Track prediction confidence distribution across a sliding window."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self._confidences: List[float] = []
        self._correctness: List[bool] = []

    def log_confidence(
        self, confidence: float, is_correct: Optional[bool] = None
    ) -> None:
        self._confidences.append(float(confidence))
        if is_correct is not None:
            self._correctness.append(bool(is_correct))
        if len(self._confidences) > self.window_size:
            self._confidences = self._confidences[-self.window_size :]
            self._correctness = self._correctness[-self.window_size :]

    def get_statistics(self) -> Dict[str, float]:
        if not self._confidences:
            return {}
        arr = np.asarray(self._confidences)
        result: Dict[str, float] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "count": float(len(self._confidences)),
        }
        if self._correctness:
            result["calibration_score"] = self._calculate_calibration()
        return result

    def _calculate_calibration(self, n_bins: int = 10) -> float:
        """Mean absolute calibration error across ``n_bins`` confidence buckets."""
        confidences = np.asarray(self._confidences[: len(self._correctness)])
        correctness = np.asarray(self._correctness, dtype=float)
        if len(confidences) == 0:
            return 0.0
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges[1:-1])
        weighted_error = 0.0
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            count = int(mask.sum())
            if count == 0:
                continue
            avg_conf = float(confidences[mask].mean())
            avg_acc = float(correctness[mask].mean())
            weighted_error += (count / len(confidences)) * abs(avg_conf - avg_acc)
        return float(weighted_error)


# ---------------------------------------------------------------------------
# Data Quality Monitor
# ---------------------------------------------------------------------------


_TYPE_MAP: Dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}


class DataQualityMonitor:
    """Validate inbound request payloads against an expected schema."""

    def __init__(
        self,
        expected_schema: Dict[str, str],
        value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.expected_schema = dict(expected_schema)
        self.value_ranges = dict(value_ranges or {})
        self.issue_counts: Dict[str, Dict[str, int]] = {
            "missing": {},
            "out_of_range": {},
            "type_error": {},
        }
        self.schema_mismatch_count = 0

    def validate_request(self, data: Dict) -> Dict[str, List[str]]:
        issues: Dict[str, List[str]] = {
            "missing": [],
            "type_error": [],
            "out_of_range": [],
        }
        for feature_name in self.expected_schema:
            if feature_name not in data:
                issues["missing"].append(feature_name)
                self.issue_counts["missing"][feature_name] = (
                    self.issue_counts["missing"].get(feature_name, 0) + 1
                )
                if missing_features_total is not None:
                    missing_features_total.labels(feature_name=feature_name).inc()
        for feature_name, value in data.items():
            expected_type_name = self.expected_schema.get(feature_name)
            if not expected_type_name:
                self.schema_mismatch_count += 1
                continue
            expected_type = _TYPE_MAP.get(expected_type_name)
            if expected_type is not None and not isinstance(value, expected_type):
                issues["type_error"].append(feature_name)
                self.issue_counts["type_error"][feature_name] = (
                    self.issue_counts["type_error"].get(feature_name, 0) + 1
                )
            value_range = self.value_ranges.get(feature_name)
            if (
                value_range is not None
                and isinstance(value, (int, float))
                and not (value_range[0] <= value <= value_range[1])
            ):
                issues["out_of_range"].append(feature_name)
                self.issue_counts["out_of_range"][feature_name] = (
                    self.issue_counts["out_of_range"].get(feature_name, 0) + 1
                )
        return issues


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------


def _demo() -> None:  # pragma: no cover - manual entrypoint
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    reference_data = rng.normal(0.0, 1.0, size=(1000, 3))
    detector = DataDriftDetector(
        reference_data=reference_data,
        feature_names=["feature_1", "feature_2", "feature_3"],
        threshold=0.05,
        method="ks",
    )
    drifted_data = rng.normal(0.5, 1.0, size=(1000, 3))
    for result in detector.detect_drift(drifted_data):
        print(
            f"{result.feature_name}: drift={result.is_drift} "
            f"statistic={result.statistic:.4f}"
        )


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    _demo()
