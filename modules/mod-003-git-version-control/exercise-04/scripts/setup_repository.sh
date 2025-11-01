#!/bin/bash

#######################################################################
# Exercise 04: Merging and Conflict Resolution - Setup Script
#######################################################################
# This script sets up the working repository with realistic merge
# scenarios and conflicts for practicing merge techniques.
#######################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_DIR="$(dirname "$SCRIPT_DIR")"
SOLUTIONS_DIR="$(dirname "$(dirname "$EXERCISE_DIR")")"

SOURCE_REPO="$SOLUTIONS_DIR/mod-003-git-version-control/exercise-03/working-repo"
TARGET_REPO="$EXERCISE_DIR/working-repo"

echo "Setting up Exercise 04 repository..."

# Clean up existing repo
if [ -d "$TARGET_REPO" ]; then
    rm -rf "$TARGET_REPO"
fi

# Copy from Exercise 03
cp -r "$SOURCE_REPO" "$TARGET_REPO"
cd "$TARGET_REPO"

# Start from master
git switch master

echo "✓ Repository copied from Exercise 03"

#######################################################################
# Scenario 1: Fast-Forward Merge - Health Check Enhancements
#######################################################################

echo "Creating feature/health-check-enhancements..."
git switch -c feature/health-check-enhancements

cat > src/api/health.py << 'EOF'
"""
Enhanced Health Check Module

Provides detailed health checks for the ML inference API.
"""

from typing import Dict, Any
import time
import psutil


class HealthChecker:
    """Comprehensive health checker for API services."""

    def __init__(self):
        self.start_time = time.time()

    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - self.start_time,
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }

    def check_model_health(self, classifier) -> Dict[str, bool]:
        """Check if model is loaded and ready."""
        return {
            "model_loaded": classifier is not None,
            "model_ready": classifier.is_loaded() if classifier else False,
        }

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all dependencies are available."""
        checks = {}

        # Check PyTorch
        try:
            import torch
            checks["pytorch"] = True
        except ImportError:
            checks["pytorch"] = False

        # Check PIL
        try:
            from PIL import Image
            checks["pillow"] = True
        except ImportError:
            checks["pillow"] = False

        return checks


health_checker = HealthChecker()
EOF

mkdir -p tests/unit
cat > tests/unit/test_health.py << 'EOF'
"""Tests for health check module."""

import pytest
from src.api.health import HealthChecker


def test_health_checker_initialization():
    """Test health checker creates successfully."""
    checker = HealthChecker()
    assert checker is not None
    assert checker.start_time > 0


def test_system_health_check():
    """Test system health check returns required fields."""
    checker = HealthChecker()
    health = checker.check_system_health()

    assert "status" in health
    assert "uptime_seconds" in health
    assert "memory_usage_mb" in health
    assert "cpu_percent" in health
    assert health["status"] == "healthy"
EOF

git add src/api/health.py tests/unit/test_health.py
git commit -m "feat: add enhanced health check system

Add comprehensive health checking:
- System resource monitoring (CPU, memory)
- Model health verification
- Dependency availability checks
- Detailed health status reporting"

echo "✓ Created feature/health-check-enhancements"

#######################################################################
# Scenario 2: Version Info (for no-ff merge demonstration)
#######################################################################

echo "Creating feature/add-version-info..."
git switch master
git switch -c feature/add-version-info

cat > src/version.py << 'EOF'
"""Version information for ML Inference API."""

__version__ = "0.3.0"
__api_version__ = "v1"
__model_version__ = "resnet50-1.0.0"


def get_version_info():
    """Get complete version information."""
    return {
        "version": __version__,
        "api_version": __api_version__,
        "model_version": __model_version__
    }


def get_version_string():
    """Get formatted version string."""
    return f"ML Inference API v{__version__} (API: {__api_version__})"
EOF

git add src/version.py
git commit -m "feat: add version information module

Add version tracking for:
- Application version
- API version
- Model version

Provides version info endpoints and tracking."

echo "✓ Created feature/add-version-info"

#######################################################################
# Scenario 3: Conflicting Config Changes
#######################################################################

echo "Creating conflicting config branches..."

# Branch 1: Feature toggles
git switch master
git switch -c feature/config-feature-toggles

cat >> configs/default.yaml << 'EOF'

# Feature Toggles
features:
  batch_inference: true
  caching: true
  metrics_export: true
  health_checks: true
EOF

git add configs/default.yaml
git commit -m "feat: add feature toggle configuration

Add feature flags for:
- Batch inference
- Result caching
- Metrics export
- Enhanced health checks"

# Branch 2: Performance settings (will conflict)
git switch master
git switch -c feature/config-performance

cat >> configs/default.yaml << 'EOF'

# Performance Settings
performance:
  max_workers: 4
  timeout_seconds: 30
  enable_profiling: false
  queue_size: 1000
EOF

git add configs/default.yaml
git commit -m "feat: add performance configuration

Add performance tuning options:
- Worker pool size
- Request timeouts
- Profiling toggle
- Queue management"

echo "✓ Created conflicting config branches"

#######################################################################
# Scenario 4: Conflicting Python Code (app.py)
#######################################################################

echo "Creating conflicting API changes..."

# Branch 1: Request validation
git switch master
git switch -c feature/request-validation

cat > src/api/validators.py << 'EOF'
"""Request validation utilities."""

from typing import Optional
from fastapi import HTTPException


def validate_file_type(content_type: str) -> None:
    """Validate uploaded file is an image."""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Must be an image."
        )


def validate_file_size(file_size: int, max_size: int = 10485760) -> None:
    """Validate file size is within limits."""
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size} bytes. Max: {max_size} bytes."
        )


def validate_image_data(image_bytes: bytes) -> Optional[str]:
    """Validate image data is valid."""
    if len(image_bytes) == 0:
        return "Empty image data"

    # Check for common image headers
    if not (
        image_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
        image_bytes.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
        image_bytes.startswith(b'GIF87a') or  # GIF
        image_bytes.startswith(b'GIF89a')
    ):
        return "Invalid image format"

    return None
EOF

git add src/api/validators.py
git commit -m "feat: add request validation utilities

Add comprehensive validation for:
- File type checking
- File size limits
- Image format validation
- Detailed error messages"

# Branch 2: Metrics tracking (will conflict with validation)
git switch master
git switch -c feature/api-metrics-tracking

cat > src/api/metrics.py << 'EOF'
"""API metrics tracking."""

import time
from typing import Dict, Any
from collections import defaultdict


class APIMetrics:
    """Track API request metrics."""

    def __init__(self):
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.latencies = defaultdict(list)

    def record_request(self, endpoint: str) -> None:
        """Record API request."""
        self.request_count[endpoint] += 1

    def record_error(self, error_type: str) -> None:
        """Record error occurrence."""
        self.error_count[error_type] += 1

    def record_latency(self, endpoint: str, latency: float) -> None:
        """Record request latency."""
        self.latencies[endpoint].append(latency)

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics statistics."""
        stats = {
            "total_requests": sum(self.request_count.values()),
            "total_errors": sum(self.error_count.values()),
            "requests_by_endpoint": dict(self.request_count),
            "errors_by_type": dict(self.error_count),
        }

        # Calculate average latencies
        avg_latencies = {}
        for endpoint, latencies in self.latencies.items():
            if latencies:
                avg_latencies[endpoint] = sum(latencies) / len(latencies)
        stats["average_latency_by_endpoint"] = avg_latencies

        return stats


api_metrics = APIMetrics()
EOF

git add src/api/metrics.py
git commit -m "feat: add API metrics tracking module

Track API performance metrics:
- Request counts by endpoint
- Error counts by type
- Response latencies
- Summary statistics"

echo "✓ Created conflicting API feature branches"

#######################################################################
# Scenario 5: Squash Merge Example - Logging Improvements
#######################################################################

echo "Creating feature with multiple commits for squash merge..."
git switch master
git switch -c feature/logging-improvements

# Commit 1
cat >> src/utils/logging.py << 'EOF'


# Timestamp formatting improvements
def format_timestamp(timestamp):
    """Format timestamp for logs."""
    return timestamp.isoformat()
EOF

git add src/utils/logging.py
git commit -m "logging: add timestamp formatting function"

# Commit 2
cat >> src/utils/logging.py << 'EOF'


# Log rotation configuration
LOG_ROTATION_SIZE = 10485760  # 10MB
LOG_ROTATION_COUNT = 5
EOF

git add src/utils/logging.py
git commit -m "logging: add log rotation configuration"

# Commit 3
cat >> src/utils/logging.py << 'EOF'


# Structured logging fields
STRUCTURED_FIELDS = ["timestamp", "level", "message", "context"]
EOF

git add src/utils/logging.py
git commit -m "logging: add structured field definitions"

echo "✓ Created feature/logging-improvements (3 commits for squash)"

#######################################################################
# Summary
#######################################################################

git switch master

echo ""
echo "✓ Repository setup complete!"
echo ""
echo "Branches created:"
git branch | grep -v "master" | sort
echo ""
echo "Merge scenarios ready:"
echo "  1. Fast-forward: health-check-enhancements"
echo "  2. No-FF: add-version-info"
echo "  3. Config conflict: feature-toggles vs performance"
echo "  4. Code conflict: request-validation vs metrics-tracking"
echo "  5. Squash merge: logging-improvements (3 commits)"
echo ""
echo "Repository ready for Exercise 04"
