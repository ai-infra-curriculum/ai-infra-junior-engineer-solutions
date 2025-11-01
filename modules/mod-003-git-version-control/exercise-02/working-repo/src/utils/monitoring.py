"""
Performance Monitoring Module

Tracks model inference performance metrics.
"""

import time
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    inference_times: List[float] = field(default_factory=list)
    request_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_inference(self, duration: float):
        """Record inference duration."""
        self.inference_times.append(duration)

    def record_request(self, endpoint: str):
        """Record API request."""
        self.request_counts[endpoint] += 1

    def record_error(self, error_type: str):
        """Record error occurrence."""
        self.error_counts[error_type] += 1

    def get_average_inference_time(self) -> float:
        """Calculate average inference time."""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def get_p95_inference_time(self) -> float:
        """Calculate 95th percentile inference time."""
        if not self.inference_times:
            return 0.0
        sorted_times = sorted(self.inference_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]

    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics."""
        return {
            "average_inference_ms": self.get_average_inference_time() * 1000,
            "p95_inference_ms": self.get_p95_inference_time() * 1000,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "requests_by_endpoint": dict(self.request_counts),
            "errors_by_type": dict(self.error_counts)
        }


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = PerformanceMetrics()


# Global monitor instance
monitor = PerformanceMonitor()
