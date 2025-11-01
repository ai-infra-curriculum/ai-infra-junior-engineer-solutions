#!/bin/bash

#######################################################################
# Exercise 03: Setup Repository - Non-Interactive
#######################################################################
# This script sets up the working repository with all branches
# without interactive pauses (for automated execution).
#######################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_DIR="$(dirname "$SCRIPT_DIR")"
SOLUTIONS_DIR="$(dirname "$(dirname "$EXERCISE_DIR")")"

SOURCE_REPO="$SOLUTIONS_DIR/mod-003-git-version-control/exercise-02/working-repo"
TARGET_REPO="$EXERCISE_DIR/working-repo"

echo "Setting up Exercise 03 repository..."

# Clean up existing repo
if [ -d "$TARGET_REPO" ]; then
    rm -rf "$TARGET_REPO"
fi

# Copy from Exercise 02
cp -r "$SOURCE_REPO" "$TARGET_REPO"
cd "$TARGET_REPO"

echo "✓ Repository copied from Exercise 02"

#######################################################################
# Feature 1: Batch Inference
#######################################################################

echo "Creating feature/batch-inference..."
git switch -c feature/batch-inference

mkdir -p src/utils

cat > src/utils/batch_processor.py << 'EOF'
"""
Batch Inference Processor

Handles batch processing of images for efficient inference.
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path
import torch
from PIL import Image

from ..models.classifier import ImageClassifier
from ..preprocessing.image import ImagePreprocessor


class BatchProcessor:
    """Process multiple images in batches for efficient inference."""

    def __init__(
        self,
        classifier: ImageClassifier,
        preprocessor: ImagePreprocessor,
        batch_size: int = 32
    ):
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.batch_size = batch_size

    async def process_batch(
        self,
        image_paths: List[Path]
    ) -> List[Dict[str, Any]]:
        """Process a batch of images."""
        results = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_results = await self._process_single_batch(batch_paths)
            results.extend(batch_results)
        return results

    async def _process_single_batch(
        self,
        batch_paths: List[Path]
    ) -> List[Dict[str, Any]]:
        """Process a single batch of images."""
        images = []
        for path in batch_paths:
            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed = self.preprocessor.preprocess(image)
            images.append(processed)

        batch_tensor = torch.cat(images, dim=0)
        with torch.no_grad():
            predictions = self.classifier.predict_batch(batch_tensor)

        results = []
        for i, path in enumerate(batch_paths):
            results.append({
                "image_path": str(path),
                "predictions": predictions[i],
                "top_class": predictions[i][0]["class"],
                "confidence": predictions[i][0]["confidence"]
            })

        return results
EOF

git add src/utils/batch_processor.py
git commit -m "feat: add batch inference processing

Implement efficient batch processing for multiple images:
- BatchProcessor class with configurable batch size
- Async batch processing support
- Memory-efficient batching

This enables processing multiple images efficiently."

echo "✓ Created feature/batch-inference"

#######################################################################
# Feature 2: Model Caching
#######################################################################

echo "Creating feature/model-caching..."
git switch master
git switch -c feature/model-caching

cat > src/utils/cache.py << 'EOF'
"""
Model Cache Manager

Implements caching for ML inference results to reduce redundant predictions.
"""

import hashlib
import time
from typing import Any, Dict, Optional


class ModelCache:
    """Cache inference results with TTL and LRU eviction."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_entries: int = 10000
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key."""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                return None
            entry["last_access"] = time.time()
            self.hits += 1
            return entry["value"]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cache value for key."""
        if len(self._cache) >= self.max_entries:
            self._evict_lru()
        self._cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "last_access": time.time()
        }

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]["last_access"]
        )
        del self._cache[lru_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "entries": len(self._cache),
        }

    @staticmethod
    def compute_key(data: bytes) -> str:
        """Compute cache key from data."""
        return hashlib.sha256(data).hexdigest()
EOF

git add src/utils/cache.py
git commit -m "feat: add model inference caching

Implement caching layer for inference results:
- ModelCache class with TTL and LRU eviction
- Content-based cache keys (SHA-256)
- Cache statistics tracking

Reduces redundant predictions for identical images."

echo "✓ Created feature/model-caching"

#######################################################################
# Feature 3: Prometheus Metrics
#######################################################################

echo "Creating feature/prometheus-metrics..."
git switch master
git switch -c feature/prometheus-metrics

cat > src/utils/prometheus_metrics.py << 'EOF'
"""
Prometheus Metrics Exporter

Exports ML inference metrics in Prometheus format.
"""

import time
from typing import Dict, Any
from collections import defaultdict


class PrometheusMetrics:
    """Track and export metrics in Prometheus format."""

    def __init__(self):
        self.request_count = 0
        self.error_count = defaultdict(int)
        self.prediction_count = defaultdict(int)
        self.latency_buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        self.latency_counts = defaultdict(int)
        self.latency_sum = 0.0
        self.active_requests = 0
        self.start_time = time.time()

    def record_request(self) -> None:
        """Record an inference request."""
        self.request_count += 1
        self.active_requests += 1

    def record_response(self, latency: float, prediction: str = None) -> None:
        """Record inference response."""
        self.active_requests -= 1
        self.latency_sum += latency
        for bucket in self.latency_buckets:
            if latency <= bucket:
                self.latency_counts[bucket] += 1
        if prediction:
            self.prediction_count[prediction] += 1

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        lines.append("# HELP inference_requests_total Total inference requests")
        lines.append("# TYPE inference_requests_total counter")
        lines.append(f"inference_requests_total {self.request_count}")

        lines.append("# HELP inference_requests_active Active inference requests")
        lines.append("# TYPE inference_requests_active gauge")
        lines.append(f"inference_requests_active {self.active_requests}")

        lines.append("# HELP inference_latency_seconds Inference latency")
        lines.append("# TYPE inference_latency_seconds histogram")
        cumulative = 0
        for bucket in self.latency_buckets:
            cumulative += self.latency_counts[bucket]
            lines.append(f'inference_latency_seconds_bucket{{le="{bucket}"}} {cumulative}')
        lines.append(f'inference_latency_seconds_bucket{{le="+Inf"}} {self.request_count}')
        lines.append(f"inference_latency_seconds_sum {self.latency_sum}")
        lines.append(f"inference_latency_seconds_count {self.request_count}")

        return "\n".join(lines) + "\n"

metrics = PrometheusMetrics()
EOF

mkdir -p configs/monitoring
cat > configs/monitoring/prometheus.yml << 'EOF'
# Prometheus configuration for ML inference API

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-inference-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
EOF

git add src/utils/prometheus_metrics.py configs/monitoring/
git commit -m "feat: add Prometheus metrics export

Implement Prometheus metrics for monitoring:
- PrometheusMetrics class tracking key metrics
- Request counters and active request gauge
- Latency histograms with configurable buckets
- Prometheus scrape configuration

Enables production monitoring and alerting."

echo "✓ Created feature/prometheus-metrics"

#######################################################################
# Bug Fix Branch
#######################################################################

echo "Creating fix/null-pointer-in-preprocessing..."
git switch master
git switch -c fix/null-pointer-in-preprocessing

cat > src/preprocessing/validation.py << 'EOF'
"""
Input validation for image preprocessing.

Fixes null pointer issues when processing invalid images.
"""

from PIL import Image
from typing import Optional
import io


def validate_image(image_data: bytes) -> Optional[Image.Image]:
    """
    Validate and load image data safely.

    Args:
        image_data: Raw image bytes

    Returns:
        PIL Image or None if invalid
    """
    if not image_data:
        return None

    try:
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        image = Image.open(io.BytesIO(image_data))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except Exception as e:
        print(f"Invalid image data: {e}")
        return None
EOF

git add src/preprocessing/validation.py
git commit -m "fix: add null pointer validation in preprocessing

Add robust image validation to prevent null pointer errors:
- Validate image data before processing
- Safely handle corrupted/invalid images
- Convert non-RGB images to RGB

Fixes production issue with invalid uploaded images."

echo "✓ Created fix/null-pointer-in-preprocessing"

# Merge the bug fix
git switch master
git merge fix/null-pointer-in-preprocessing --no-edit
git branch -d fix/null-pointer-in-preprocessing

echo "✓ Merged and deleted fix branch"

#######################################################################
# Summary
#######################################################################

git switch master

echo ""
echo "✓ Repository setup complete!"
echo ""
echo "Branches created:"
git branch
echo ""
echo "Commit history:"
git log --oneline --graph --all -10
echo ""
echo "Repository ready for Exercise 03"
