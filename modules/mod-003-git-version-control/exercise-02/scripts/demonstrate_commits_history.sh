#!/bin/bash
###############################################################################
# Git Commits and History Demonstration Script
###############################################################################
#
# Purpose: Demonstrate advanced commit techniques and history navigation
#          for Exercise 02 of the Git Version Control module
#
# This script demonstrates:
# - Conventional commit message format
# - Viewing and searching history
# - Amending commits
# - Reverting changes
# - Complete feature workflow
#
# Usage: ./demonstrate_commits_history.sh
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Source repository (from Exercise 01)
SOURCE_REPO="../../exercise-01/example-repo"
WORK_REPO="../working-repo"

###############################################################################
# Helper Functions
###############################################################################

log_step() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo ""
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_command() {
    echo -e "${YELLOW}$${NC} $1"
}

pause_for_review() {
    echo ""
    echo -e "${MAGENTA}Press Enter to continue...${NC}"
    read -r
}

###############################################################################
# Setup
###############################################################################

setup_repository() {
    log_step "Setup: Copying Repository from Exercise 01"

    # Check if source exists
    if [ ! -d "$SOURCE_REPO" ]; then
        echo -e "${RED}Error: Source repository not found at $SOURCE_REPO${NC}"
        echo "Please complete Exercise 01 first."
        exit 1
    fi

    # Remove old working repo if exists
    if [ -d "$WORK_REPO" ]; then
        log_info "Removing existing working repository..."
        rm -rf "$WORK_REPO"
    fi

    # Copy the repository
    log_info "Copying repository..."
    cp -r "$SOURCE_REPO" "$WORK_REPO"

    cd "$WORK_REPO"

    log_success "Repository ready for Exercise 02"

    echo ""
    echo "Initial commit history:"
    git log --oneline -5
}

###############################################################################
# Part 1: Conventional Commits
###############################################################################

part1_conventional_commits() {
    log_step "Part 1: Writing Effective Commit Messages"

    cd "$WORK_REPO"

    # Task 1.1: Add monitoring module with conventional commit
    log_info "Task 1.1: Adding performance monitoring (feat: commit)"

    cat > src/utils/monitoring.py << 'EOF'
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
EOF

    git add src/utils/monitoring.py

    git commit -m "feat: add performance monitoring for inference tracking

Implement PerformanceMetrics and PerformanceMonitor classes to track:
- Inference latency (average and p95)
- Request counts per endpoint
- Error counts by type
- Service uptime

This enables real-time monitoring and helps identify performance
bottlenecks in the ML inference pipeline.

The monitor uses structured data collection for easy integration
with monitoring systems like Prometheus and Grafana."

    log_success "Committed with conventional 'feat:' prefix"

    echo ""
    echo "Commit created:"
    git log --oneline -1

    pause_for_review

    # Task 1.2: Bug fix with proper documentation
    log_info "Task 1.2: Fixing image preprocessing bug (fix: commit)"

    # First create the bug
    cat > src/preprocessing/simple_preprocess.py << 'EOF'
"""Simple image preprocessing."""

import torch
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image - has a bug!"""
    image = Image.open(io.BytesIO(image_bytes))
    # BUG: Not converting to RGB
    # This will fail with grayscale or RGBA images
    return image
EOF

    git add src/preprocessing/simple_preprocess.py
    git commit -m "refactor: simplify preprocessing

Add simplified preprocessing module"

    # Now fix it
    cat > src/preprocessing/simple_preprocess.py << 'EOF'
"""Simple image preprocessing."""

import torch
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image with RGB conversion."""
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed (handles grayscale and RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image
EOF

    git add src/preprocessing/simple_preprocess.py
    git commit -m "fix: convert images to RGB before preprocessing

The preprocessing pipeline now converts all images to RGB mode
before applying transformations. This fixes inference failures
when processing:
- Grayscale images (mode 'L')
- RGBA images with alpha channel
- Other non-RGB image formats

Without this conversion, the model receives incorrect tensor
dimensions causing inference to fail.

Fixes issue where grayscale medical images caused 500 errors."

    log_success "Committed bug fix with detailed explanation"

    echo ""
    echo "Last 2 commits:"
    git log --oneline -2

    pause_for_review

    # Task 1.3: Configuration feature
    log_info "Task 1.3: Adding rate limiting configuration (feat: commit)"

    cat >> configs/default.yaml << 'EOF'

# Rate Limiting
rate_limiting:
  enabled: true
  requests_per_minute: 100
  burst_size: 20
  strategy: "sliding_window"

# Request Timeout
timeout:
  request_timeout_seconds: 30
  inference_timeout_seconds: 10
EOF

    cat > src/utils/rate_limiter.py << 'EOF'
"""
Rate Limiting Module

Implements request rate limiting for API endpoints.
"""

import time
from collections import deque
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 100,
        burst_size: int = 20
    ):
        """Initialize rate limiter."""
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.requests = deque()

    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_update = now

    def allow_request(self) -> bool:
        """Check if request is allowed."""
        self._refill_tokens()
        if self.tokens >= 1:
            self.tokens -= 1
            self.requests.append(time.time())
            return True
        return False

    def get_wait_time(self) -> float:
        """Get time to wait before next request allowed."""
        self._refill_tokens()
        if self.tokens >= 1:
            return 0.0
        return (1.0 - self.tokens) / (self.requests_per_minute / 60.0)

    def reset(self):
        """Reset rate limiter state."""
        self.tokens = self.burst_size
        self.last_update = time.time()
        self.requests.clear()
EOF

    git add configs/default.yaml src/utils/rate_limiter.py

    git commit -m "feat(api): implement request rate limiting

Add token bucket rate limiter to prevent API abuse:

Configuration:
- Configurable requests per minute limit
- Burst capacity for traffic spikes
- Sliding window strategy

Implementation:
- Token bucket algorithm
- Automatic token refill
- Wait time calculation
- Request tracking

Default settings: 100 requests/min with burst of 20.

This protects the service from abuse while allowing legitimate
traffic spikes."

    log_success "Committed with scope: feat(api)"

    echo ""
    echo "Recent commits:"
    git log --oneline -3
}

###############################################################################
# Part 2: Viewing and Searching History
###############################################################################

part2_viewing_history() {
    log_step "Part 2: Viewing and Searching History"

    cd "$WORK_REPO"

    log_info "Demonstrating various git log commands..."

    echo ""
    echo -e "${CYAN}=== Basic log ===${NC}"
    log_command "git log --oneline -5"
    git log --oneline -5

    pause_for_review

    echo ""
    echo -e "${CYAN}=== Log with statistics ===${NC}"
    log_command "git log --stat --oneline -3"
    git log --stat --oneline -3

    pause_for_review

    echo ""
    echo -e "${CYAN}=== Pretty format ===${NC}"
    log_command "git log --pretty=format:'%h - %an, %ar : %s' -5"
    git log --pretty=format:"%h - %an, %ar : %s" -5

    pause_for_review

    echo ""
    echo -e "${CYAN}=== Search for 'fix' commits ===${NC}"
    log_command "git log --grep='fix' --oneline"
    git log --grep="fix" --oneline || echo "No matching commits"

    pause_for_review

    echo ""
    echo -e "${CYAN}=== Search for code changes ===${NC}"
    log_command "git log -S 'RateLimiter' --oneline"
    git log -S "RateLimiter" --oneline || echo "Not found yet"

    pause_for_review

    echo ""
    echo -e "${CYAN}=== File-specific history ===${NC}"
    log_command "git log --oneline -- configs/default.yaml"
    git log --oneline -- configs/default.yaml

    log_success "Demonstrated various history viewing commands"
}

###############################################################################
# Part 3: Amending Commits
###############################################################################

part3_amending_commits() {
    log_step "Part 3: Amending Commits"

    cd "$WORK_REPO"

    # Task 3.1: Fix commit message
    log_info "Task 3.1: Amending commit message (fixing typo)"

    echo "# API Documentation" > docs/api.md
    git add docs/api.md
    git commit -m "doc: add API documentaton"  # Intentional typo

    echo ""
    echo "Original commit (with typo):"
    git log --oneline -1

    pause_for_review

    log_info "Amending the commit message..."
    git commit --amend -m "docs: add API documentation

Create initial API documentation file for endpoint reference."

    echo ""
    echo "Amended commit (typo fixed):"
    git log --oneline -1

    log_success "Commit message amended"

    pause_for_review

    # Task 3.2: Add forgotten file
    log_info "Task 3.2: Adding forgotten file to last commit"

    cat > docs/api.md << 'EOF'
# ML Inference API Documentation

## Endpoints

### POST /predict
Upload image for classification.

**Request:**
```
POST /predict
Content-Type: multipart/form-data

file: <image-file>
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "class_id": 281,
      "class": "tabby_cat",
      "confidence": 0.94
    }
  ]
}
```

### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
EOF

    git add docs/api.md
    git commit -m "docs: add API endpoint documentation"

    # Forgot to update README
    cat >> README.md << 'EOF'

## API Documentation

See [API Documentation](docs/api.md) for detailed endpoint information.
EOF

    echo ""
    echo "Oops! Forgot to add README update. Adding it now..."

    git add README.md
    git commit --amend --no-edit

    echo ""
    echo "Amended commit now includes both files:"
    git show --stat --oneline HEAD

    log_success "Added forgotten file to commit"

    pause_for_review

    # Task 3.3: Amend with both message and content
    log_info "Task 3.3: Amending both message and content"

    cat > src/api/middleware.py << 'EOF'
"""API Middleware"""

from fastapi import Request
import time

async def logging_middleware(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    print(f"{request.method} {request.url.path} - {duration:.3f}s")
    return response
EOF

    git add src/api/middleware.py
    git commit -m "feat: add middleware"

    echo ""
    echo "Original commit (incomplete):"
    git log --oneline -1

    pause_for_review

    # Enhance it
    cat > src/api/middleware.py << 'EOF'
"""API Middleware"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time

from src.utils.rate_limiter import RateLimiter

rate_limiter = RateLimiter()


async def logging_middleware(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    print(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response


async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to requests."""
    if not rate_limiter.allow_request():
        wait_time = rate_limiter.get_wait_time()
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": wait_time
            }
        )
    return await call_next(request)
EOF

    git add src/api/middleware.py
    git commit --amend -m "feat(api): add request logging and rate limiting middleware

Implement two middleware components:

1. Logging Middleware:
   - Logs all incoming requests
   - Tracks request duration
   - Includes status codes

2. Rate Limiting Middleware:
   - Enforces rate limits per client
   - Returns 429 status when limited
   - Includes retry-after in response

These middleware protect the API and provide observability."

    echo ""
    echo "Amended commit (complete):"
    git log --oneline -1
    git show --stat HEAD

    log_success "Amended with better message and complete implementation"
}

###############################################################################
# Part 4: Reverting Changes
###############################################################################

part4_reverting_changes() {
    log_step "Part 4: Reverting Changes"

    cd "$WORK_REPO"

    # Task 4.1: Revert a single commit
    log_info "Task 4.1: Reverting a bad commit"

    cat > src/utils/cache.py << 'EOF'
"""
Caching Module - EXPERIMENTAL

WARNING: This implementation has issues!
"""

cache = {}  # Global cache - not thread-safe!

def get_from_cache(key):
    return cache.get(key)

def add_to_cache(key, value):
    cache[key] = value
EOF

    git add src/utils/cache.py
    git commit -m "feat: add caching module"

    echo ""
    echo "Created problematic commit:"
    git log --oneline -1

    pause_for_review

    log_info "Reverting the commit (creates new commit that undoes changes)..."

    # Revert with custom message
    GIT_EDITOR=true git revert HEAD --no-edit

    # Amend the revert message to be more descriptive
    git commit --amend -m "Revert \"feat: add caching module\"

The cache implementation has thread-safety issues.
Will reimplement using proper locking mechanisms.

This reverts commit $(git log --format="%H" -n 1 HEAD~1)"

    echo ""
    echo "Revert commit created:"
    git log --oneline -2

    echo ""
    echo "Cache file removed:"
    ls src/utils/cache.py 2>&1 || echo -e "${GREEN}✓ File successfully removed by revert${NC}"

    log_success "Safely reverted problematic commit"

    pause_for_review

    # Task 4.2: Revert multiple commits
    log_info "Task 4.2: Reverting multiple commits"

    echo "config1" > bad_config1.txt
    git add bad_config1.txt
    git commit -m "bad commit 1"

    echo "config2" > bad_config2.txt
    git add bad_config2.txt
    git commit -m "bad commit 2"

    echo "config3" > bad_config3.txt
    git add bad_config3.txt
    git commit -m "bad commit 3"

    echo ""
    echo "Created 3 bad commits:"
    git log --oneline -3

    pause_for_review

    log_info "Reverting all three commits..."
    git revert HEAD HEAD~1 HEAD~2 --no-edit

    echo ""
    echo "Revert commits created:"
    git log --oneline -6

    echo ""
    echo "Bad config files removed:"
    ls bad_config*.txt 2>&1 || echo -e "${GREEN}✓ All files successfully removed${NC}"

    log_success "Reverted multiple commits safely"
}

###############################################################################
# Part 5: Complete Workflow
###############################################################################

part5_complete_workflow() {
    log_step "Part 5: Complete Feature Development Workflow"

    cd "$WORK_REPO"

    log_info "Implementing metrics endpoint feature..."

    # Step 1: Add metrics endpoint
    cat > src/api/metrics.py << 'EOF'
"""
Metrics Endpoint

Exposes performance metrics for monitoring.
"""

from fastapi import APIRouter
from src.utils.monitoring import monitor

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    metrics = monitor.metrics.get_metrics_summary()
    metrics["uptime_seconds"] = monitor.get_uptime()
    return metrics


@router.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics (admin only)."""
    monitor.reset_metrics()
    return {"status": "metrics reset"}
EOF

    git add src/api/metrics.py
    git commit -m "feat(api): add metrics endpoint for monitoring

Expose /metrics endpoint to retrieve:
- Average inference time
- P95 inference latency
- Request counts by endpoint
- Error counts by type
- Service uptime

Includes /metrics/reset for admin use.

This enables integration with Prometheus and Grafana
for infrastructure monitoring."

    log_success "Step 1: Committed metrics endpoint"

    # Step 2: Register in main app
    echo "" >> src/api/app.py
    echo "# Metrics endpoints would be registered here" >> src/api/app.py

    git add src/api/app.py
    git commit -m "feat(api): register metrics router in main application

Include metrics endpoints in FastAPI app routing."

    log_success "Step 2: Committed router registration"

    # Step 3: Add tests
    mkdir -p tests/unit
    cat > tests/unit/test_metrics.py << 'EOF'
"""Tests for metrics endpoint."""

def test_metrics_endpoint():
    """Test metrics endpoint returns data."""
    # Mock test - would use TestClient in real implementation
    assert True


def test_metrics_reset():
    """Test metrics can be reset."""
    assert True
EOF

    git add tests/unit/test_metrics.py
    git commit -m "test: add unit tests for metrics endpoint

Test metrics retrieval and reset functionality."

    log_success "Step 3: Committed tests"

    # Step 4: Update documentation
    cat >> docs/api.md << 'EOF'

### GET /metrics
Get current performance metrics.

**Response:**
```json
{
  "average_inference_ms": 45.2,
  "p95_inference_ms": 78.5,
  "total_requests": 1523,
  "total_errors": 3,
  "uptime_seconds": 86400
}
```

### POST /metrics/reset
Reset all metrics (admin only).
EOF

    git add docs/api.md
    git commit -m "docs: document metrics endpoints

Add documentation for metrics retrieval and reset endpoints."

    log_success "Step 4: Committed documentation"

    echo ""
    echo "Complete feature workflow (4 atomic commits):"
    git log --oneline -4

    log_success "Completed feature with clean commit history"
}

###############################################################################
# Summary and Review
###############################################################################

show_summary() {
    log_step "Summary: Reviewing All Commits"

    cd "$WORK_REPO"

    echo ""
    echo -e "${CYAN}=== All Commits Created ===${NC}"
    git log --oneline --graph --all

    echo ""
    echo ""
    echo -e "${CYAN}=== Commit Statistics ===${NC}"
    echo "Total commits: $(git rev-list --count HEAD)"
    echo "Files tracked: $(git ls-files | wc -l)"

    echo ""
    echo ""
    echo -e "${CYAN}=== Commits by Type ===${NC}"
    echo "feat commits: $(git log --oneline | grep -c '^[a-f0-9]* feat' || echo 0)"
    echo "fix commits: $(git log --oneline | grep -c '^[a-f0-9]* fix' || echo 0)"
    echo "docs commits: $(git log --oneline | grep -c '^[a-f0-9]* docs' || echo 0)"
    echo "test commits: $(git log --oneline | grep -c '^[a-f0-9]* test' || echo 0)"

    echo ""
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Exercise 02 Complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "You have learned:"
    echo "  ✓ Conventional commit messages (feat:, fix:, docs:)"
    echo "  ✓ Viewing history with git log"
    echo "  ✓ Searching for specific changes"
    echo "  ✓ Amending commits"
    echo "  ✓ Reverting changes safely"
    echo "  ✓ Complete feature workflow"
    echo ""
    echo "Repository location: $WORK_REPO"
    echo ""
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log_step "Git Commits and History - Exercise 02"

    echo "This script demonstrates:"
    echo "  1. Conventional commit messages"
    echo "  2. Viewing and searching history"
    echo "  3. Amending commits"
    echo "  4. Reverting changes"
    echo "  5. Complete workflow"
    echo ""
    echo "The script will pause after each section for review."
    echo ""

    pause_for_review

    # Execute all parts
    setup_repository
    part1_conventional_commits
    part2_viewing_history
    part3_amending_commits
    part4_reverting_changes
    part5_complete_workflow
    show_summary
}

main "$@"
