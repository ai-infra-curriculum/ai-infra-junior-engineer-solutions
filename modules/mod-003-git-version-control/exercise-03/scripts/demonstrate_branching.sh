#!/bin/bash

#######################################################################
# Exercise 03: Branching for Feature Development - Demonstration Script
#######################################################################
#
# This script demonstrates Git branching concepts for ML infrastructure:
# - Creating and switching branches
# - Branch naming conventions
# - Parallel feature development
# - Comparing branches
# - Deleting branches
# - Stashing uncommitted changes
# - Complete feature workflow
#
# Usage: ./demonstrate_branching.sh
#
# The script will:
# 1. Copy the repository from Exercise 02
# 2. Create multiple feature branches
# 3. Develop features in parallel
# 4. Demonstrate branch management
# 5. Show comparison techniques
# 6. Clean up merged branches
#
#######################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_DIR="$(dirname "$SCRIPT_DIR")"
SOLUTIONS_DIR="$(dirname "$(dirname "$EXERCISE_DIR")")"

# Source and target directories
SOURCE_REPO="$SOLUTIONS_DIR/mod-003-git-version-control/exercise-02/working-repo"
TARGET_REPO="$EXERCISE_DIR/working-repo"

#######################################################################
# Helper Functions
#######################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_section() {
    echo -e "\n${CYAN}--- $1 ---${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_command() {
    echo -e "${MAGENTA}$ $1${NC}"
}

pause_for_review() {
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read -r
}

#######################################################################
# Part 0: Setup
#######################################################################

setup_repository() {
    print_header "Part 0: Setup - Copying Repository"

    print_section "Copying from Exercise 02"

    if [ -d "$TARGET_REPO" ]; then
        print_info "Removing existing repository..."
        rm -rf "$TARGET_REPO"
    fi

    print_command "cp -r $SOURCE_REPO $TARGET_REPO"
    cp -r "$SOURCE_REPO" "$TARGET_REPO"

    cd "$TARGET_REPO"

    print_success "Repository copied successfully"
    print_info "Current directory: $(pwd)"

    print_section "Initial Branch Status"
    print_command "git branch -a"
    git branch -a

    print_command "git log --oneline -5"
    git log --oneline -5

    pause_for_review
}

#######################################################################
# Part 1: Creating Branches
#######################################################################

demonstrate_branch_creation() {
    print_header "Part 1: Creating Branches"

    print_section "Creating Feature Branch: batch-inference"
    print_info "Branch naming convention: feature/<feature-name>"

    print_command "git branch feature/batch-inference"
    git branch feature/batch-inference

    print_command "git branch"
    git branch

    print_success "Created feature/batch-inference branch"

    print_section "Creating and Switching to Branch (One Command)"
    print_info "Modern approach: git switch -c <branch>"

    print_command "git switch -c feature/model-caching"
    git switch -c feature/model-caching

    print_command "git branch"
    git branch

    print_success "Created and switched to feature/model-caching"

    print_section "Alternative: Using checkout -b"
    print_info "Traditional approach: git checkout -b <branch>"

    print_command "git checkout -b feature/prometheus-metrics"
    git checkout -b feature/prometheus-metrics

    print_command "git branch"
    git branch

    print_success "Created and switched to feature/prometheus-metrics"

    print_section "Creating Experimental Branch"
    print_info "For experimental features: experiment/<feature>"

    print_command "git switch -c experiment/onnx-runtime"
    git switch -c experiment/onnx-runtime

    print_command "git branch"
    git branch

    print_success "Created experimental branch"

    print_section "Current Branch Status"
    print_command "git branch -v"
    git branch -v

    pause_for_review
}

#######################################################################
# Part 2: Parallel Feature Development
#######################################################################

develop_batch_inference() {
    print_header "Part 2.1: Developing Feature - Batch Inference"

    print_section "Switching to feature/batch-inference"
    print_command "git switch feature/batch-inference"
    git switch feature/batch-inference

    print_info "Current branch: $(git branch --show-current)"

    print_section "Creating batch inference utility"

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
    """
    Process multiple images in batches for efficient inference.

    Features:
    - Configurable batch size
    - Async processing
    - Memory-efficient batching
    - Progress tracking
    """

    def __init__(
        self,
        classifier: ImageClassifier,
        preprocessor: ImagePreprocessor,
        batch_size: int = 32
    ):
        """
        Initialize batch processor.

        Args:
            classifier: Trained image classifier
            preprocessor: Image preprocessing pipeline
            batch_size: Number of images per batch
        """
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.batch_size = batch_size

    async def process_batch(
        self,
        image_paths: List[Path]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of prediction results
        """
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
        # Load and preprocess images
        images = []
        for path in batch_paths:
            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed = self.preprocessor.preprocess(image)
            images.append(processed)

        # Stack into batch tensor
        batch_tensor = torch.cat(images, dim=0)

        # Run inference
        with torch.no_grad():
            predictions = self.classifier.predict_batch(batch_tensor)

        # Format results
        results = []
        for i, path in enumerate(batch_paths):
            results.append({
                "image_path": str(path),
                "predictions": predictions[i],
                "top_class": predictions[i][0]["class"],
                "confidence": predictions[i][0]["confidence"]
            })

        return results

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            "batch_size": self.batch_size,
            "device": str(self.classifier.device)
        }
EOF

    print_success "Created src/utils/batch_processor.py"

    print_section "Adding batch prediction to classifier"

    cat >> src/models/classifier.py << 'EOF'

    def predict_batch(
        self,
        batch_tensor: torch.Tensor,
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Predict on a batch of images.

        Args:
            batch_tensor: Batch of preprocessed images [B, C, H, W]
            top_k: Number of top predictions to return

        Returns:
            List of predictions for each image in batch
        """
        batch_tensor = batch_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        batch_results = []
        for probs in probabilities:
            top_probs, top_indices = torch.topk(probs, top_k)

            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                predictions.append({
                    "class_id": int(idx),
                    "class": self.class_names.get(int(idx), f"class_{idx}"),
                    "confidence": float(prob)
                })

            batch_results.append(predictions)

        return batch_results
EOF

    print_success "Added batch prediction method"

    print_section "Adding batch endpoint to API"

    cat >> src/api/app.py << 'EOF'


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Classify multiple uploaded images in batch.

    Efficient batch processing for multiple images.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 images per batch"
        )

    results = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        processed_image = preprocessor.preprocess(image)
        predictions = classifier.predict(processed_image)

        results.append(PredictionResponse(
            success=True,
            predictions=predictions,
            top_prediction=predictions[0]["class"],
            confidence=predictions[0]["confidence"]
        ))

    return results
EOF

    print_success "Added /predict/batch endpoint"

    print_section "Committing batch inference feature"
    print_command "git add src/utils/batch_processor.py src/models/classifier.py src/api/app.py"
    git add src/utils/batch_processor.py src/models/classifier.py src/api/app.py

    print_command "git commit -m 'feat: add batch inference processing'"
    git commit -m "feat: add batch inference processing

Implement efficient batch processing for multiple images:
- BatchProcessor class with configurable batch size
- Async batch processing support
- Memory-efficient batching
- Batch prediction method in classifier
- REST API batch endpoint (/predict/batch)

This enables processing multiple images efficiently,
reducing per-image overhead and improving throughput."

    print_success "Committed batch inference feature"

    print_command "git log --oneline -1"
    git log --oneline -1

    pause_for_review
}

develop_model_caching() {
    print_header "Part 2.2: Developing Feature - Model Caching"

    print_section "Switching to feature/model-caching"
    print_command "git switch feature/model-caching"
    git switch feature/model-caching

    print_info "Current branch: $(git branch --show-current)"

    print_section "Creating model cache manager"

    cat > src/utils/cache.py << 'EOF'
"""
Model Cache Manager

Implements caching for ML inference results to reduce redundant predictions.
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional
from pathlib import Path
import pickle


class ModelCache:
    """
    Cache inference results with TTL and size limits.

    Features:
    - Time-to-live (TTL) expiration
    - LRU eviction when size limit reached
    - File-based persistence
    - Cache hit/miss statistics
    """

    def __init__(
        self,
        cache_dir: Path = Path("cache"),
        ttl_seconds: int = 3600,
        max_entries: int = 10000
    ):
        """
        Initialize model cache.

        Args:
            cache_dir: Directory for cache storage
            ttl_seconds: Time-to-live for cache entries
            max_entries: Maximum number of cached entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        # Statistics
        self.hits = 0
        self.misses = 0

        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key in self._cache:
            entry = self._cache[key]

            # Check if expired
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                return None

            # Update access time for LRU
            entry["last_access"] = time.time()
            self.hits += 1
            return entry["value"]

        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Set cache value for key.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict if at capacity
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

    def invalidate(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds
        }

    @staticmethod
    def compute_key(data: bytes) -> str:
        """Compute cache key from data."""
        return hashlib.sha256(data).hexdigest()
EOF

    print_success "Created src/utils/cache.py"

    print_section "Integrating cache with API"

    cat > src/api/cache_middleware.py << 'EOF'
"""
Cache middleware for FastAPI.

Caches inference results based on image content.
"""

from fastapi import Request, Response
from typing import Callable
import hashlib

from ..utils.cache import ModelCache


cache = ModelCache()


async def cache_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to cache inference results.

    Caches based on image content hash to avoid redundant predictions.
    """
    # Only cache prediction endpoints
    if not request.url.path.startswith("/predict"):
        return await call_next(request)

    # For POST requests with file uploads
    if request.method == "POST":
        body = await request.body()
        cache_key = ModelCache.compute_key(body)

        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return Response(
                content=cached_result,
                media_type="application/json",
                headers={"X-Cache": "HIT"}
            )

        # Process request
        response = await call_next(request)

        # Cache result
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        cache.set(cache_key, response_body)

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )

    return await call_next(request)
EOF

    print_success "Created cache middleware"

    print_section "Adding cache stats endpoint"

    cat >> src/api/app.py << 'EOF'


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    from .cache_middleware import cache
    return cache.get_stats()


@app.post("/cache/clear")
async def clear_cache():
    """Clear the inference cache."""
    from .cache_middleware import cache
    cache.clear()
    return {"success": True, "message": "Cache cleared"}
EOF

    print_success "Added cache endpoints"

    print_section "Committing model caching feature"
    print_command "git add src/utils/cache.py src/api/cache_middleware.py src/api/app.py"
    git add src/utils/cache.py src/api/cache_middleware.py src/api/app.py

    print_command "git commit -m 'feat: add model inference caching'"
    git commit -m "feat: add model inference caching

Implement caching layer for inference results:
- ModelCache class with TTL and LRU eviction
- Cache middleware for FastAPI
- Content-based cache keys (SHA-256)
- Cache statistics tracking
- Cache management endpoints

Benefits:
- Reduces redundant predictions for identical images
- Improves response time for cached results
- Configurable TTL and size limits"

    print_success "Committed model caching feature"

    print_command "git log --oneline -1"
    git log --oneline -1

    pause_for_review
}

develop_prometheus_metrics() {
    print_header "Part 2.3: Developing Feature - Prometheus Metrics"

    print_section "Switching to feature/prometheus-metrics"
    print_command "git switch feature/prometheus-metrics"
    git switch feature/prometheus-metrics

    print_info "Current branch: $(git branch --show-current)"

    print_section "Creating Prometheus metrics exporter"

    cat > src/utils/prometheus_metrics.py << 'EOF'
"""
Prometheus Metrics Exporter

Exports ML inference metrics in Prometheus format.
"""

from typing import Dict, Any
import time
from collections import defaultdict


class PrometheusMetrics:
    """
    Track and export metrics in Prometheus format.

    Metrics tracked:
    - Inference request count
    - Inference latency histogram
    - Model prediction distribution
    - Cache hit/miss rate
    - Error count by type
    """

    def __init__(self):
        """Initialize metrics tracking."""
        # Counters
        self.request_count = 0
        self.error_count = defaultdict(int)
        self.prediction_count = defaultdict(int)

        # Histograms (latency buckets in seconds)
        self.latency_buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        self.latency_counts = defaultdict(int)
        self.latency_sum = 0.0

        # Gauges
        self.active_requests = 0

        # Start time
        self.start_time = time.time()

    def record_request(self) -> None:
        """Record an inference request."""
        self.request_count += 1
        self.active_requests += 1

    def record_response(self, latency: float, prediction: str = None) -> None:
        """
        Record inference response.

        Args:
            latency: Request latency in seconds
            prediction: Top prediction class
        """
        self.active_requests -= 1
        self.latency_sum += latency

        # Record in histogram buckets
        for bucket in self.latency_buckets:
            if latency <= bucket:
                self.latency_counts[bucket] += 1

        if prediction:
            self.prediction_count[prediction] += 1

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        self.error_count[error_type] += 1
        self.active_requests -= 1

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        lines = []

        # Request counter
        lines.append("# HELP inference_requests_total Total inference requests")
        lines.append("# TYPE inference_requests_total counter")
        lines.append(f"inference_requests_total {self.request_count}")

        # Active requests gauge
        lines.append("# HELP inference_requests_active Active inference requests")
        lines.append("# TYPE inference_requests_active gauge")
        lines.append(f"inference_requests_active {self.active_requests}")

        # Latency histogram
        lines.append("# HELP inference_latency_seconds Inference latency")
        lines.append("# TYPE inference_latency_seconds histogram")
        cumulative = 0
        for bucket in self.latency_buckets:
            cumulative += self.latency_counts[bucket]
            lines.append(f'inference_latency_seconds_bucket{{le="{bucket}"}} {cumulative}')
        lines.append(f'inference_latency_seconds_bucket{{le="+Inf"}} {self.request_count}')
        lines.append(f"inference_latency_seconds_sum {self.latency_sum}")
        lines.append(f"inference_latency_seconds_count {self.request_count}")

        # Prediction distribution
        lines.append("# HELP inference_predictions_total Predictions by class")
        lines.append("# TYPE inference_predictions_total counter")
        for class_name, count in self.prediction_count.items():
            lines.append(f'inference_predictions_total{{class="{class_name}"}} {count}')

        # Error counter
        lines.append("# HELP inference_errors_total Inference errors by type")
        lines.append("# TYPE inference_errors_total counter")
        for error_type, count in self.error_count.items():
            lines.append(f'inference_errors_total{{type="{error_type}"}} {count}')

        # Uptime
        uptime = time.time() - self.start_time
        lines.append("# HELP process_uptime_seconds Process uptime")
        lines.append("# TYPE process_uptime_seconds gauge")
        lines.append(f"process_uptime_seconds {uptime}")

        return "\n".join(lines) + "\n"

    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable metrics summary."""
        avg_latency = (
            self.latency_sum / self.request_count
            if self.request_count > 0
            else 0.0
        )

        return {
            "total_requests": self.request_count,
            "active_requests": self.active_requests,
            "average_latency": avg_latency,
            "total_errors": sum(self.error_count.values()),
            "top_predictions": dict(
                sorted(
                    self.prediction_count.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ),
            "uptime_seconds": time.time() - self.start_time
        }


# Global metrics instance
metrics = PrometheusMetrics()
EOF

    print_success "Created src/utils/prometheus_metrics.py"

    print_section "Adding metrics endpoints"

    cat >> src/api/app.py << 'EOF'


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    from ..utils.prometheus_metrics import metrics
    return Response(
        content=metrics.export_prometheus(),
        media_type="text/plain; version=0.0.4"
    )


@app.get("/metrics/summary")
async def metrics_summary():
    """Human-readable metrics summary."""
    from ..utils.prometheus_metrics import metrics
    return metrics.get_summary()
EOF

    print_success "Added metrics endpoints"

    print_section "Creating Prometheus configuration"

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

    print_success "Created Prometheus configuration"

    print_section "Committing Prometheus metrics feature"
    print_command "git add src/utils/prometheus_metrics.py src/api/app.py configs/monitoring/"
    git add src/utils/prometheus_metrics.py src/api/app.py configs/monitoring/

    print_command "git commit -m 'feat: add Prometheus metrics export'"
    git commit -m "feat: add Prometheus metrics export

Implement Prometheus metrics for monitoring:
- PrometheusMetrics class tracking key metrics
- Request counters and active request gauge
- Latency histograms with configurable buckets
- Prediction distribution tracking
- Error counters by type
- Uptime tracking
- /metrics endpoint (Prometheus format)
- /metrics/summary endpoint (JSON format)
- Prometheus scrape configuration

Enables production monitoring and alerting."

    print_success "Committed Prometheus metrics feature"

    print_command "git log --oneline -1"
    git log --oneline -1

    pause_for_review
}

develop_experimental_onnx() {
    print_header "Part 2.4: Developing Experimental - ONNX Runtime"

    print_section "Switching to experiment/onnx-runtime"
    print_command "git switch experiment/onnx-runtime"
    git switch experiment/onnx-runtime

    print_info "Current branch: $(git branch --show-current)"
    print_info "This is an experimental feature - may not be merged"

    print_section "Creating ONNX runtime support"

    cat > src/models/onnx_classifier.py << 'EOF'
"""
ONNX Runtime Classifier

Experimental support for ONNX model inference.
"""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ONNXClassifier:
    """
    ONNX Runtime classifier for faster inference.

    NOTE: This is experimental and requires onnxruntime package.

    Benefits over PyTorch:
    - Faster inference (optimized execution)
    - Smaller memory footprint
    - Cross-platform deployment
    - Hardware acceleration (CPU/GPU/TensorRT)
    """

    def __init__(self, model_path: Path):
        """
        Initialize ONNX classifier.

        Args:
            model_path: Path to ONNX model file
        """
        if ort is None:
            raise ImportError(
                "onnxruntime not installed. "
                "Install with: pip install onnxruntime"
            )

        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict on a single image.

        Args:
            image: Preprocessed image [1, C, H, W]
            top_k: Number of top predictions

        Returns:
            List of top predictions
        """
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image}
        )[0]

        # Apply softmax
        exp_outputs = np.exp(outputs - np.max(outputs))
        probabilities = exp_outputs / np.sum(exp_outputs)

        # Get top k
        top_indices = np.argsort(probabilities[0])[::-1][:top_k]
        top_probs = probabilities[0][top_indices]

        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                "class_id": int(idx),
                "class": f"class_{idx}",
                "confidence": float(prob)
            })

        return predictions


def convert_pytorch_to_onnx(
    pytorch_model,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224)
):
    """
    Convert PyTorch model to ONNX format.

    Args:
        pytorch_model: PyTorch model
        output_path: Output ONNX file path
        input_shape: Input tensor shape
    """
    import torch

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
EOF

    print_success "Created src/models/onnx_classifier.py"

    print_section "Adding ONNX to requirements"

    cat >> requirements.txt << 'EOF'

# ONNX Runtime (experimental)
onnxruntime==1.16.0  # CPU only
# onnxruntime-gpu==1.16.0  # For GPU support
EOF

    print_success "Added ONNX dependencies"

    print_section "Committing experimental ONNX support"
    print_command "git add src/models/onnx_classifier.py requirements.txt"
    git add src/models/onnx_classifier.py requirements.txt

    print_command "git commit -m 'experiment: add ONNX runtime support'"
    git commit -m "experiment: add ONNX runtime support

Add experimental ONNX Runtime inference:
- ONNXClassifier for faster inference
- PyTorch to ONNX conversion utility
- ONNX runtime dependencies

This is experimental and may not be merged to main.
Requires thorough testing and benchmarking.

Potential benefits:
- 2-3x faster inference
- Lower memory usage
- Better cross-platform support"

    print_success "Committed experimental feature"

    print_command "git log --oneline -1"
    git log --oneline -1

    pause_for_review
}

#######################################################################
# Part 3: Comparing Branches
#######################################################################

demonstrate_branch_comparison() {
    print_header "Part 3: Comparing Branches"

    print_section "Switching back to main branch"
    print_command "git switch main"
    git switch main

    print_section "Viewing All Branches"
    print_command "git branch -v"
    git branch -v

    print_section "Comparing Branches with Log"
    print_info "See commits in feature/batch-inference not in main"

    print_command "git log main..feature/batch-inference --oneline"
    git log main..feature/batch-inference --oneline

    print_section "Comparing Multiple Branches"
    print_info "See all feature branches together"

    print_command "git log --oneline --graph --all --decorate"
    git log --oneline --graph --all --decorate

    print_section "Comparing File Changes"
    print_info "What files changed in feature/model-caching?"

    print_command "git diff main..feature/model-caching --name-only"
    git diff main..feature/model-caching --name-only

    print_section "Detailed Diff of Specific Feature"
    print_info "See actual code changes in batch-inference"

    print_command "git diff main..feature/batch-inference --stat"
    git diff main..feature/batch-inference --stat

    print_section "Comparing Two Feature Branches"
    print_info "Difference between batch-inference and model-caching"

    print_command "git diff feature/batch-inference..feature/model-caching --name-only"
    git diff feature/batch-inference..feature/model-caching --name-only

    print_section "Using git show-branch"
    print_info "Visual comparison of branches"

    print_command "git show-branch main feature/* experiment/*"
    git show-branch main feature/* experiment/* || true

    pause_for_review
}

#######################################################################
# Part 4: Stashing Changes
#######################################################################

demonstrate_stashing() {
    print_header "Part 4: Stashing Uncommitted Changes"

    print_section "Creating uncommitted changes"
    print_command "git switch feature/batch-inference"
    git switch feature/batch-inference

    print_info "Adding some work-in-progress changes..."

    cat >> src/utils/batch_processor.py << 'EOF'


# TODO: Add progress callback
# def set_progress_callback(self, callback):
#     self.progress_callback = callback
EOF

    print_command "git status"
    git status

    print_section "Stashing changes to switch branches"
    print_info "Need to switch branches but don't want to commit yet"

    print_command "git stash save 'WIP: progress callback for batch processing'"
    git stash save "WIP: progress callback for batch processing"

    print_success "Changes stashed"

    print_command "git status"
    git status

    print_section "Switching to another branch"
    print_command "git switch feature/model-caching"
    git switch feature/model-caching

    print_success "Switched successfully - no uncommitted changes"

    print_section "Listing stashes"
    print_command "git stash list"
    git stash list

    print_section "Returning to original branch"
    print_command "git switch feature/batch-inference"
    git switch feature/batch-inference

    print_section "Applying stashed changes"
    print_command "git stash pop"
    git stash pop

    print_success "Stashed changes restored"

    print_command "git status"
    git status

    print_section "Cleaning up uncommitted changes"
    print_command "git restore src/utils/batch_processor.py"
    git restore src/utils/batch_processor.py

    pause_for_review
}

#######################################################################
# Part 5: Bug Fix Branch
#######################################################################

create_bugfix_branch() {
    print_header "Part 5: Creating Bug Fix Branch"

    print_section "Simulating production bug discovery"
    print_info "Critical bug found in production - needs immediate fix"

    print_command "git switch main"
    git switch main

    print_section "Creating hotfix branch from main"
    print_info "Branch naming: fix/<bug-description>"

    print_command "git switch -c fix/null-pointer-in-preprocessing"
    git switch -c fix/null-pointer-in-preprocessing

    print_section "Fixing the bug"

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

        # Verify image can be loaded
        image.verify()

        # Reopen after verify (PIL requires this)
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except Exception as e:
        # Log error but don't crash
        print(f"Invalid image data: {e}")
        return None
EOF

    print_success "Created validation module"

    print_section "Committing bug fix"
    print_command "git add src/preprocessing/validation.py"
    git add src/preprocessing/validation.py

    print_command "git commit -m 'fix: add null pointer validation in preprocessing'"
    git commit -m "fix: add null pointer validation in preprocessing

Add robust image validation to prevent null pointer errors:
- Validate image data before processing
- Safely handle corrupted/invalid images
- Convert non-RGB images to RGB
- Proper error logging without crashing

Fixes production issue #142 - null pointer errors
when processing invalid uploaded images."

    print_success "Bug fix committed"

    print_command "git log --oneline -1"
    git log --oneline -1

    print_info "In real workflow, this would be:"
    print_info "1. Merged to main immediately"
    print_info "2. Deployed to production"
    print_info "3. Cherry-picked to active feature branches"

    pause_for_review
}

#######################################################################
# Part 6: Branch Visualization
#######################################################################

visualize_branches() {
    print_header "Part 6: Branch Visualization"

    print_section "Comprehensive Branch Overview"

    print_command "git branch -a"
    git branch -a

    print_section "Branches with Last Commit"
    print_command "git branch -v"
    git branch -v

    print_section "Branches with Commit Count"
    print_command "git branch -vv"
    git branch -vv || git branch -v

    print_section "Visual History Graph"
    print_command "git log --all --graph --oneline --decorate"
    git log --all --graph --oneline --decorate

    print_section "Detailed Branch Graph"
    print_command "git log --all --graph --pretty=format:'%C(yellow)%h%Creset -%C(cyan)%d%Creset %s %C(green)(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
    git log --all --graph --pretty=format:'%C(yellow)%h%Creset -%C(cyan)%d%Creset %s %C(green)(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit

    pause_for_review
}

#######################################################################
# Part 7: Branch Management
#######################################################################

demonstrate_branch_management() {
    print_header "Part 7: Branch Management and Cleanup"

    print_section "Merged vs Unmerged Branches"
    print_info "Show which branches have been merged to main"

    print_command "git switch main"
    git switch main

    print_command "git branch --merged"
    git branch --merged

    print_command "git branch --no-merged"
    git branch --no-merged

    print_section "Simulating Branch Merge"
    print_info "Let's merge the bug fix to main"

    print_command "git merge fix/null-pointer-in-preprocessing --no-edit"
    git merge fix/null-pointer-in-preprocessing --no-edit

    print_success "Bug fix merged to main"

    print_section "Deleting Merged Branch"
    print_info "Safe to delete after merging"

    print_command "git branch -d fix/null-pointer-in-preprocessing"
    git branch -d fix/null-pointer-in-preprocessing

    print_success "Merged branch deleted"

    print_section "Attempting to Delete Unmerged Branch"
    print_info "Git prevents accidental deletion of unmerged work"

    print_command "git branch -d feature/batch-inference"
    git branch -d feature/batch-inference 2>&1 || true

    print_info "Use -D to force delete (be careful!)"

    print_section "Deleting Experimental Branch"
    print_info "Experimental feature not ready - force delete"

    print_command "git branch -D experiment/onnx-runtime"
    git branch -D experiment/onnx-runtime

    print_success "Experimental branch deleted"

    print_section "Final Branch Status"
    print_command "git branch -v"
    git branch -v

    pause_for_review
}

#######################################################################
# Part 8: Summary
#######################################################################

show_summary() {
    print_header "Part 8: Summary"

    print_section "Branches Created"
    echo "✓ feature/batch-inference - Batch processing for images"
    echo "✓ feature/model-caching - Result caching with TTL"
    echo "✓ feature/prometheus-metrics - Monitoring integration"
    echo "✓ experiment/onnx-runtime - Experimental ONNX support (deleted)"
    echo "✓ fix/null-pointer-in-preprocessing - Bug fix (merged & deleted)"

    print_section "Key Concepts Demonstrated"
    echo "✓ Branch creation (git branch, git switch -c)"
    echo "✓ Branch naming conventions (feature/, fix/, experiment/)"
    echo "✓ Parallel feature development"
    echo "✓ Branch comparison (git log, git diff)"
    echo "✓ Stashing uncommitted changes"
    echo "✓ Branch visualization (git log --graph)"
    echo "✓ Branch cleanup (git branch -d/-D)"
    echo "✓ Merged vs unmerged detection"

    print_section "Current Repository State"
    print_command "git log --oneline --graph --all -10"
    git log --oneline --graph --all -10

    print_section "Remaining Branches (Ready to Merge)"
    print_command "git branch"
    git branch

    print_success "Exercise 03 demonstration complete!"

    print_info "Next steps:"
    echo "  - Exercise 04: Learn merging strategies"
    echo "  - Exercise 05: Resolve merge conflicts"
    echo "  - Exercise 06: Remote collaboration"
}

#######################################################################
# Main Execution
#######################################################################

main() {
    print_header "Exercise 03: Branching for Feature Development"

    echo "This script demonstrates Git branching for ML infrastructure:"
    echo "  - Creating feature branches"
    echo "  - Parallel development workflow"
    echo "  - Branch comparison and visualization"
    echo "  - Branch management and cleanup"
    echo ""
    echo "Duration: ~10-15 minutes"
    echo ""

    pause_for_review

    # Execute all parts
    setup_repository
    demonstrate_branch_creation
    develop_batch_inference
    develop_model_caching
    develop_prometheus_metrics
    develop_experimental_onnx
    demonstrate_branch_comparison
    demonstrate_stashing
    create_bugfix_branch
    visualize_branches
    demonstrate_branch_management
    show_summary

    print_success "All demonstrations complete!"
}

# Run main function
main
