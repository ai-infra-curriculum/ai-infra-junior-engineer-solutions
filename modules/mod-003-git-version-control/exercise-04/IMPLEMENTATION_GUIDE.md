# Exercise 04: Merging and Conflict Resolution - Implementation Guide

## Overview

Master Git merge strategies and conflict resolution for ML project integration. Learn to merge feature branches, resolve conflicts in code and configuration, and choose the right merge strategy for your workflow.

**Estimated Time**: 90-120 minutes
**Difficulty**: Intermediate
**Prerequisites**: Exercise 03 - Branching

## What You'll Learn

- ✅ Fast-forward vs three-way merges
- ✅ Resolving merge conflicts
- ✅ Handling conflicts in Python, YAML, and notebooks
- ✅ Merge strategies (merge, squash, rebase)
- ✅ Using merge tools
- ✅ Aborting and retrying merges
- ✅ Pre-merge validation
- ✅ ML project merge best practices

---

## Part 1: Understanding Merge Types

### Step 1.1: Fast-Forward Merge

**What is Fast-Forward?**
- Linear history (no divergence)
- Simply moves branch pointer forward
- No merge commit created
- Clean, linear history

```bash
# Continue from Exercise 03 repository
cd ml-inference-api

# Ensure you're on main
git switch main

# View current state
git log --oneline --graph -3

# Create simple feature branch
git switch -c feature/add-health-endpoint

# Add health check endpoint
cat > src/api/health.py << 'EOF'
"""
Health Check Endpoint

Provides system health status for monitoring.
"""
from fastapi import APIRouter, Response, status

router = APIRouter()

@router.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "ml-inference-api",
        "version": "0.2.0"
    }

@router.get("/health/ready")
def readiness_check():
    """Readiness probe for Kubernetes."""
    # Check if model is loaded
    return {
        "status": "ready",
        "model_loaded": True
    }

@router.get("/health/live")
def liveness_check():
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}
EOF

git add src/api/health.py
git commit -m "feat(health): add Kubernetes health probes

Add health check endpoints:
- /health: Basic health status
- /health/ready: Readiness probe
- /health/live: Liveness probe

For Kubernetes deployment monitoring."

# Switch back to main and merge
git switch main

# This will be a fast-forward merge
git merge feature/add-health-endpoint

# Expected output:
# Updating abc123..def456
# Fast-forward
#  src/api/health.py | 35 +++++++++++++++++++++++++++++++++++
#  1 file changed, 35 insertions(+)

# View the history - notice LINEAR progression
git log --oneline --graph -5
# Output:
# * def456 (HEAD -> main, feature/add-health-endpoint) feat(health): add Kubernetes health probes
# * abc123 Previous commit...
# * xyz789 Earlier commit...
```

**Key Points:**
- Main branch pointer simply moved forward
- No merge commit created
- History remains linear
- Only possible when branches haven't diverged

### Step 1.2: Force No-Fast-Forward Merge

Sometimes you WANT a merge commit even when fast-forward is possible.

```bash
# Create another simple feature
git switch -c feature/add-version-endpoint

cat > src/api/version.py << 'EOF'
"""Version Information Endpoint"""
from fastapi import APIRouter

router = APIRouter()

__version__ = "0.2.0"
__api_version__ = "v1"

@router.get("/version")
def get_version():
    """Return API version information."""
    return {
        "version": __version__,
        "api_version": __api_version__,
        "service": "ml-inference-api"
    }
EOF

git add src/api/version.py
git commit -m "feat(version): add version info endpoint

Expose version information for API discovery."

# Switch back and merge with --no-ff
git switch main
git merge --no-ff feature/add-version-endpoint -m "Merge feature: version endpoint

Add version information endpoint for API introspection.

Reviewed-by: Tech Lead
PR: #42"

# View graph - notice MERGE COMMIT
git log --oneline --graph -5
# Output:
# *   mno345 (HEAD -> main) Merge feature: version endpoint
# |\
# | * pqr678 (feature/add-version-endpoint) feat(version): add version info endpoint
# |/
# * def456 feat(health): add Kubernetes health probes
```

**When to use --no-ff:**
- ✅ Preserve feature branch context
- ✅ Track when features were integrated
- ✅ Link to pull requests
- ✅ Facilitate rollbacks
- ❌ Don't use for tiny commits (creates noise)

### Step 1.3: Three-Way Merge

When both branches have new commits (diverged history).

```bash
# Create feature branch from main
git switch main
git switch -c feature/add-metrics

# Add metrics module
cat > src/monitoring/metrics.py << 'EOF'
"""Prometheus Metrics Collection"""
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)

# Model metrics
model_inference_time = Histogram(
    'model_inference_seconds',
    'Model inference time'
)

active_requests = Gauge(
    'api_active_requests',
    'Number of active requests'
)
EOF

git add src/monitoring/metrics.py
git commit -m "feat(metrics): add Prometheus metrics

Instrument API with Prometheus metrics:
- Request count by endpoint
- Request duration histogram
- Model inference timing
- Active request gauge"

# Meanwhile, on main branch, someone else merged a change
git switch main

# Add configuration file
cat > configs/monitoring.yaml << 'EOF'
monitoring:
  prometheus:
    enabled: true
    port: 9090
    path: /metrics

  logging:
    level: INFO
    format: json

  tracing:
    enabled: false
EOF

git add configs/monitoring.yaml
git commit -m "config: add monitoring configuration

Configure Prometheus and logging settings."

# Now try to merge metrics feature
git merge feature/add-metrics

# Git opens editor for merge commit message
# Default message shows:
# "Merge branch 'feature/add-metrics'"
#
# Customize it:
# "Merge feature: Prometheus metrics
#
# Integrate metrics collection with monitoring config:
# - Prometheus instrumentation
# - Request and inference metrics
# - Configuration for metrics export"

# View the THREE-WAY MERGE
git log --oneline --graph -6
# Output shows two parent commits:
# *   stu901 (HEAD -> main) Merge feature: Prometheus metrics
# |\
# | * vwx234 (feature/add-metrics) feat(metrics): add Prometheus metrics
# * | yza567 config: add monitoring configuration
# |/
# * mno345 Merge feature: version endpoint
```

**Three-Way Merge Characteristics:**
- Creates merge commit with TWO parents
- Combines changes from both branches
- Shows when branches diverged and reconverged
- Preserves complete history

---

## Part 2: Resolving Conflicts

### Step 2.1: Create a Configuration Conflict

Conflicts occur when both branches modify the same part of a file.

```bash
# Create two branches that edit the same config file

# Branch 1: Add feature toggles
git switch main
git switch -c feature/feature-toggles

cat >> configs/default.yaml << 'EOF'

# Feature Toggles
features:
  batch_inference: true
  prediction_cache: true
  async_processing: true
  rate_limiting: true
EOF

git add configs/default.yaml
git commit -m "config: add feature toggle system

Enable/disable features via configuration:
- Batch inference
- Prediction caching
- Async processing
- Rate limiting"

# Branch 2: Add performance settings (conflicts with branch 1)
git switch main
git switch -c feature/performance-config

cat >> configs/default.yaml << 'EOF'

# Performance Tuning
performance:
  worker_threads: 4
  max_batch_size: 32
  timeout_seconds: 30
  queue_size: 1000
  enable_profiling: false
EOF

git add configs/default.yaml
git commit -m "config: add performance tuning settings

Configure performance parameters:
- Worker thread count
- Batch size limits
- Timeout settings
- Queue configuration"

# Merge first branch - OK
git switch main
git merge feature/feature-toggles --no-edit

# Try to merge second branch - CONFLICT!
git merge feature/performance-config
```

**Expected Output:**
```
Auto-merging configs/default.yaml
CONFLICT (content): Merge conflict in configs/default.yaml
Automatic merge failed; fix conflicts and then commit the result.
```

### Step 2.2: Examine the Conflict

```bash
# Check status
git status
# Output:
# On branch main
# You have unmerged paths.
#   (fix conflicts and run "git commit")
#   (use "git merge --abort" to abort the merge)
#
# Unmerged paths:
#   (use "git add <file>..." to mark resolution)
#         both modified:   configs/default.yaml

# View the conflict markers
cat configs/default.yaml
```

**Conflict Markers Explained:**
```yaml
# ... existing config ...

<<<<<<< HEAD
# Feature Toggles
features:
  batch_inference: true
  prediction_cache: true
  async_processing: true
  rate_limiting: true
=======
# Performance Tuning
performance:
  worker_threads: 4
  max_batch_size: 32
  timeout_seconds: 30
  queue_size: 1000
  enable_profiling: false
>>>>>>> feature/performance-config
```

**Understanding Markers:**
- `<<<<<<< HEAD` - Current branch (main) version
- `=======` - Separator between versions
- `>>>>>>> feature/performance-config` - Incoming branch version

### Step 2.3: Resolve the Conflict

**Resolution Strategy:** Keep BOTH changes (not mutually exclusive).

```bash
# Edit configs/default.yaml to include BOTH sections
cat > configs/default.yaml << 'EOF'
# Default Configuration for ML Inference API

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false

model:
  name: "resnet50"
  version: "1.0.0"
  weights_path: "models/resnet50_weights.pth"
  device: "cpu"

# Feature Toggles
features:
  batch_inference: true
  prediction_cache: true
  async_processing: true
  rate_limiting: true

# Performance Tuning
performance:
  worker_threads: 4
  max_batch_size: 32
  timeout_seconds: 30
  queue_size: 1000
  enable_profiling: false

# Monitoring
monitoring:
  prometheus:
    enabled: true
    port: 9090
    path: /metrics
  logging:
    level: INFO
    format: json
EOF

# Mark as resolved by staging
git add configs/default.yaml

# Check status - should show "All conflicts fixed"
git status
# Output:
# On branch main
# All conflicts fixed but you are still merging.
#   (use "git commit" to conclude merge)

# Complete the merge
git commit -m "Merge feature: performance configuration

Resolved conflicts by integrating both feature toggles
and performance settings into unified configuration.

Both configuration systems are complementary:
- Feature toggles control functionality
- Performance settings tune execution"

# Verify
git log --oneline --graph -5
```

---

## Part 3: Code Conflict Resolution

### Step 3.1: Create Python Code Conflict

```bash
# Branch 1: Add request validation
git switch main
git switch -c feature/request-validation

mkdir -p src/api
cat > src/api/predict.py << 'EOF'
"""Prediction API Endpoint"""
from fastapi import APIRouter, File, UploadFile, HTTPException
import structlog

from src.models.inference import predict_image
from src.utils.validation import validate_image_file

logger = structlog.get_logger()

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict image class with validation."""

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png)"
        )

    # Read and validate image
    image_bytes = await file.read()

    if not validate_image_file(image_bytes):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )

    try:
        prediction = await predict_image(image_bytes)

        logger.info(
            "prediction_success",
            filename=file.filename,
            predicted_class=prediction["class"]
        )

        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
EOF

git add src/api/predict.py
git commit -m "feat(api): add image validation to predict endpoint

Validate images before inference:
- Check MIME type
- Validate image format
- Reject corrupted files
- Better error messages"

# Branch 2: Add metrics tracking (conflicts with validation)
git switch main
git switch -c feature/metrics-integration

cat > src/api/predict.py << 'EOF'
"""Prediction API Endpoint"""
from fastapi import APIRouter, File, UploadFile, HTTPException
import structlog
import time

from src.models.inference import predict_image
from src.monitoring.metrics import (
    request_count,
    request_duration,
    model_inference_time
)

logger = structlog.get_logger()

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict image class with metrics tracking."""

    # Track request
    request_count.labels(
        endpoint="/predict",
        method="POST",
        status="processing"
    ).inc()

    start_time = time.time()

    # Validate file type
    if not file.content_type.startswith("image/"):
        request_count.labels(
            endpoint="/predict",
            method="POST",
            status="error"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        image_bytes = await file.read()

        # Track inference time
        inference_start = time.time()
        prediction = await predict_image(image_bytes)
        inference_time = time.time() - inference_start

        model_inference_time.observe(inference_time)

        # Track success
        duration = time.time() - start_time
        request_duration.labels(endpoint="/predict").observe(duration)
        request_count.labels(
            endpoint="/predict",
            method="POST",
            status="success"
        ).inc()

        logger.info(
            "prediction_success",
            filename=file.filename,
            duration=duration,
            inference_time=inference_time
        )

        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        request_count.labels(
            endpoint="/predict",
            method="POST",
            status="error"
        ).inc()
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
EOF

git add src/api/predict.py
git commit -m "feat(metrics): add metrics tracking to predict endpoint

Instrument prediction endpoint:
- Track request count by status
- Measure request duration
- Track inference timing
- Record success/failure rates"

# Merge first feature
git switch main
git merge feature/request-validation --no-edit

# Merge second - CONFLICT!
git merge feature/metrics-integration
```

### Step 3.2: Resolve Python Code Conflict

```bash
# View the conflict
git status
# Shows: both modified: src/api/predict.py

# View conflict markers
cat src/api/predict.py | head -60
```

**Resolution Strategy:** Combine BOTH features - validation AND metrics.

```bash
# Create unified version with both features
cat > src/api/predict.py << 'EOF'
"""Prediction API Endpoint"""
from fastapi import APIRouter, File, UploadFile, HTTPException
import structlog
import time

from src.models.inference import predict_image
from src.utils.validation import validate_image_file
from src.monitoring.metrics import (
    request_count,
    request_duration,
    model_inference_time
)

logger = structlog.get_logger()

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict image class with validation and metrics.

    Features:
    - Image validation (format and integrity)
    - Metrics tracking (requests, duration, inference time)
    - Comprehensive error handling
    """

    # Track request start
    request_count.labels(
        endpoint="/predict",
        method="POST",
        status="processing"
    ).inc()

    start_time = time.time()

    # Validate file type
    if not file.content_type.startswith("image/"):
        request_count.labels(
            endpoint="/predict",
            method="POST",
            status="error"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png)"
        )

    # Read and validate image content
    image_bytes = await file.read()

    if not validate_image_file(image_bytes):
        request_count.labels(
            endpoint="/predict",
            method="POST",
            status="error"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )

    try:
        # Track inference time
        inference_start = time.time()
        prediction = await predict_image(image_bytes)
        inference_time = time.time() - inference_start

        model_inference_time.observe(inference_time)

        # Track success metrics
        duration = time.time() - start_time
        request_duration.labels(endpoint="/predict").observe(duration)
        request_count.labels(
            endpoint="/predict",
            method="POST",
            status="success"
        ).inc()

        logger.info(
            "prediction_success",
            filename=file.filename,
            predicted_class=prediction["class"],
            duration=duration,
            inference_time=inference_time
        )

        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        request_count.labels(
            endpoint="/predict",
            method="POST",
            status="error"
        ).inc()
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
EOF

# Stage resolved file
git add src/api/predict.py

# Verify resolution
git status
# Should show: All conflicts fixed

# Complete merge
git commit -m "Merge feature: metrics integration

Combined metrics tracking with image validation.

Integrated features:
- Image validation (format, integrity)
- Prometheus metrics (requests, timing)
- Comprehensive error tracking

Both features work together for production-ready endpoint."

# View merge history
git log --oneline --graph -6
```

---

## Part 4: Merge Strategies

### Step 4.1: Squash Merge

Combine multiple commits into one for cleaner history.

```bash
# Create feature with many small commits
git switch -c feature/logging-improvements

# Commit 1
cat > src/utils/logger.py << 'EOF'
"""Logging utilities."""
import structlog

def configure_logging():
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ]
    )
EOF

git add src/utils/logger.py
git commit -m "logging: add timestamp processor"

# Commit 2
cat >> src/utils/logger.py << 'EOF'

def get_logger(name: str):
    """Get configured logger."""
    return structlog.get_logger(name)
EOF

git add src/utils/logger.py
git commit -m "logging: add logger factory function"

# Commit 3
cat >> src/utils/logger.py << 'EOF'

def log_request(endpoint: str, duration: float):
    """Log API request."""
    logger = get_logger(__name__)
    logger.info("api_request", endpoint=endpoint, duration=duration)
EOF

git add src/utils/logger.py
git commit -m "logging: add request logging helper"

# View commits (3 small commits)
git log --oneline -3

# Switch to main and SQUASH merge
git switch main
git merge --squash feature/logging-improvements

# Status shows changes staged but NOT committed
git status
# Output:
# On branch main
# Changes to be committed:
#   (use "git restore --staged <file>..." to unstage)
#         new file:   src/utils/logger.py

# Create single consolidated commit
git commit -m "feat(logging): improve logging infrastructure

Consolidated logging improvements:
- Structured logging with timestamps
- Logger factory with configuration
- Request logging helpers
- ISO timestamp format
- Log level support

Replaces 3 WIP commits with production-ready implementation."

# View history - only ONE commit on main
git log --oneline -3
# Notice: Feature branch commits NOT in main history
```

**When to Use Squash:**
- ✅ Many WIP (work in progress) commits
- ✅ Want clean main branch history
- ✅ Individual commits not meaningful
- ✅ Before merging to production
- ❌ Don't use if individual commits have value
- ❌ Don't use for collaborative branches

### Step 4.2: Rebase Before Merge

Create linear history by replaying commits.

```bash
# Create feature branch
git switch -c feature/add-caching

cat > src/cache/redis_cache.py << 'EOF'
"""Redis caching for predictions."""
import redis
import json

class PredictionCache:
    def __init__(self, host='localhost', port=6379):
        self.client = redis.Redis(host=host, port=port)

    def get(self, key: str):
        """Get cached prediction."""
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key: str, value: dict, ttl: int = 3600):
        """Cache prediction with TTL."""
        self.client.setex(
            key,
            ttl,
            json.dumps(value)
        )
EOF

git add src/cache/redis_cache.py
git commit -m "feat(cache): add Redis caching implementation"

# Meanwhile, main branch advanced
git switch main
echo "# Important update" >> README.md
git add README.md
git commit -m "docs: update README"

# Before merging, rebase feature branch on main
git switch feature/add-caching
git rebase main

# Expected output:
# Successfully rebased and updated refs/heads/feature/add-caching.

# Now merge (will be fast-forward)
git switch main
git merge feature/add-caching

# View LINEAR history
git log --oneline --graph -5
# No merge commit! Clean linear history.
```

**Rebase vs Merge:**

| Strategy | Pros | Cons | Use When |
|----------|------|------|----------|
| **Merge** | Preserves history, shows integration points | Creates merge commits, complex graph | Shared branches, preserving context |
| **Rebase** | Linear history, clean, simple | Rewrites history, dangerous on shared branches | Solo work, before merging |
| **Squash** | Very clean history, hides WIP | Loses individual commits | Many small commits, cleaning up |

---

## Part 5: Advanced Conflict Strategies

### Step 5.1: Abort a Problematic Merge

```bash
# Start a complex merge
git switch -c feature/major-refactor

# Make massive changes that will conflict
cat > src/api/predict.py << 'EOF'
# Completely rewritten implementation
print("This will cause major conflicts!")
EOF

git add src/api/predict.py
git commit -m "refactor: complete rewrite of predict API"

# Try to merge
git switch main
git merge feature/major-refactor

# Output shows MANY conflicts
git status
# Shows multiple files with conflicts

# Decide this merge is too complex - abort it
git merge --abort

# Verify we're back to pre-merge state
git status
# Output: nothing to commit, working tree clean

git diff
# No changes
```

**When to Abort:**
- Too many conflicts to resolve at once
- Realize merge is based on wrong branch
- Need to discuss resolution strategy with team
- Want to try different merge strategy

### Step 5.2: Choose One Side Entirely

Sometimes the correct resolution is to use one version completely.

```bash
# During a conflict, you can choose a side:

# Example: Create conflict
git switch main
echo "version A" > example.txt
git add example.txt
git commit -m "Add version A"

git switch -c feature/version-b
echo "version B" > example.txt
git add example.txt
git commit -m "Add version B"

git switch main
git merge feature/version-b
# CONFLICT!

# Keep OUR version (current branch)
git checkout --ours example.txt
git add example.txt

# OR keep THEIR version (incoming branch)
git checkout --theirs example.txt
git add example.txt

# Complete merge
git commit -m "Merge: kept version B as more recent"
```

**Use Cases for --ours/--theirs:**
- Configuration files (use production config)
- Generated files (use newer version)
- Documentation (choose more complete version)
- Lock files (npm/poetry) - usually use theirs

### Step 5.3: Use Merge Tools

```bash
# Configure merge tool (one-time setup)
git config --global merge.tool vimdiff
# Or use: meld, kdiff3, vscode, p4merge

# Configure to not create .orig backup files
git config --global mergetool.keepBackup false

# When conflict occurs:
git mergetool

# This opens visual merge tool with 3 panes:
# - LOCAL (your changes)
# - BASE (common ancestor)
# - REMOTE (their changes)
# - MERGED (result)

# Make edits, save, exit

# Verify resolution
git diff --cached

# Complete merge
git commit
```

**Popular Merge Tools:**
- **VS Code**: Built-in, user-friendly
- **vimdiff**: Terminal-based, fast
- **meld**: Visual, intuitive
- **kdiff3**: Powerful, auto-merge capable
- **p4merge**: Professional, feature-rich

---

## Part 6: Preventing Merge Problems

### Step 6.1: Pre-Merge Validation

```bash
# Before merging, validate without committing
git switch main

# Test merge without actually merging
git merge --no-commit --no-ff feature/branch-name

# Review what would be merged
git status
git diff --cached

# Check for conflicts
if git status | grep -q "Unmerged paths"; then
    echo "Conflicts detected! Review before merging."
    git diff
fi

# If everything looks good:
git commit -m "Merge feature: description"

# If problems found:
git merge --abort
# Fix issues in feature branch, then retry
```

### Step 6.2: Post-Merge Validation Script

```bash
# Create comprehensive post-merge check
cat > scripts/post-merge-check.sh << 'EOF'
#!/bin/bash
# Post-Merge Validation Script

set -e  # Exit on any error

echo "=== Post-Merge Validation ==="
echo ""

# 1. Check for conflict markers
echo "1. Checking for unresolved conflicts..."
if grep -r "<<<<<<< HEAD" src/ 2>/dev/null; then
    echo "❌ ERROR: Unresolved conflict markers found!"
    exit 1
fi
if grep -r ">>>>>>> " src/ 2>/dev/null; then
    echo "❌ ERROR: Unresolved conflict markers found!"
    exit 1
fi
echo "✅ No conflict markers"

# 2. Check Python syntax
echo ""
echo "2. Checking Python syntax..."
if ! python -m py_compile src/**/*.py 2>/dev/null; then
    echo "❌ ERROR: Python syntax errors found!"
    exit 1
fi
echo "✅ Python syntax valid"

# 3. Check imports
echo ""
echo "3. Checking imports..."
if ! python -c "import sys; sys.path.insert(0, '.'); import src" 2>/dev/null; then
    echo "⚠️  Warning: Import issues detected"
fi

# 4. Run linter
echo ""
echo "4. Running linter..."
if command -v ruff &> /dev/null; then
    ruff check src/ || echo "⚠️  Linting issues found"
fi

# 5. Run tests
echo ""
echo "5. Running tests..."
if [ -d "tests" ]; then
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short || {
            echo "❌ ERROR: Tests failed!"
            exit 1
        }
    else
        echo "⚠️  pytest not installed, skipping tests"
    fi
fi
echo "✅ Tests passed"

# 6. Check YAML configs
echo ""
echo "6. Validating YAML configs..."
if command -v yamllint &> /dev/null; then
    yamllint configs/*.yaml || echo "⚠️  YAML issues found"
fi

echo ""
echo "=== ✅ All validation checks passed ==="
echo ""
echo "Safe to push!"
EOF

chmod +x scripts/post-merge-check.sh

# Run after every merge
git add scripts/post-merge-check.sh
git commit -m "chore: add post-merge validation script"

# Use it
./scripts/post-merge-check.sh
```

### Step 6.3: Keep Branches Updated

```bash
# Regularly sync feature branch with main
git switch feature/long-running-feature

# Pull latest main changes
git fetch origin main

# Rebase on main (or merge main into feature)
git rebase origin/main

# Resolve any conflicts incrementally
# This prevents massive conflicts at merge time

# Alternative: Merge main into feature
git merge main
# Resolve conflicts in feature branch
# Main branch stays clean
```

**Best Practice:**
- Sync feature branches with main DAILY
- Smaller, more frequent conflict resolutions
- Easier to remember context
- Reduces risk of massive conflicts at merge time

---

## Part 7: ML Project Merge Scenarios

### Scenario 1: Merging Notebook Changes

```bash
# Jupyter notebooks are JSON - difficult to merge

# Strategy 1: Clear outputs before committing
jupyter nbconvert --clear-output --inplace notebook.ipynb
git add notebook.ipynb
git commit -m "experiment: add data analysis notebook"

# Strategy 2: Use nbdime for notebook merging
pip install nbdime

# Configure Git to use nbdime
nbdime config-git --enable --global

# Now notebook merges are cell-aware
git merge feature/notebook-updates
# nbdime handles cell-level merging
```

### Scenario 2: Model Weight File Conflicts

```bash
# Problem: Binary files (models) can't be merged

# Solution: Use Git LFS and version strategy
git lfs track "models/*.pth"
git lfs track "models/*.h5"

# Naming convention prevents conflicts
# Instead of: model.pth
# Use: model_v1.0.0_20250101.pth

# In code, reference by version
cat > src/models/loader.py << 'EOF'
"""Model loader with versioning."""
def load_model(version: str = "1.0.0"):
    """Load specific model version."""
    path = f"models/model_v{version}.pth"
    return torch.load(path)
EOF

# Merge strategy: Keep both versions
git merge feature/new-model
# If conflict on models/model.pth:
# 1. Rename both versions
# 2. Update code to load correct version
# 3. Mark resolved
```

### Scenario 3: Data Pipeline Conflicts

```bash
# Multiple data preprocessing approaches

# Strategy: Feature flags
cat > src/data/preprocessing.py << 'EOF'
"""Data preprocessing with feature flags."""
from src.config import settings

def preprocess_data(data):
    """Preprocess with configurable strategies."""

    if settings.USE_NEW_PREPROCESSING:
        # New approach from feature branch
        return new_preprocessing(data)
    else:
        # Stable approach from main
        return legacy_preprocessing(data)

def new_preprocessing(data):
    """New preprocessing strategy."""
    # Implementation from feature branch
    pass

def legacy_preprocessing(data):
    """Legacy preprocessing strategy."""
    # Implementation from main branch
    pass
EOF

# Merge both approaches
git merge feature/new-preprocessing --no-commit

# Update code to include both
# Add feature flag to config
# Test both paths
# Commit merge
```

---

## Verification Checklist

After completing all merge exercises:

- [ ] Performed fast-forward merge successfully
- [ ] Created merge commit with --no-ff
- [ ] Completed three-way merge
- [ ] Resolved configuration file conflict
- [ ] Resolved Python code conflict
- [ ] Used squash merge for multiple commits
- [ ] Aborted and retried a problematic merge
- [ ] Used --ours or --theirs to resolve conflict
- [ ] Configured and tested merge tool
- [ ] Created post-merge validation script
- [ ] Understand when to use each merge strategy
- [ ] No unresolved conflict markers in code
- [ ] All tests passing after merges
- [ ] Clean git history with meaningful commits

---

## Common Issues and Solutions

### Issue 1: "Cannot merge - uncommitted changes"

```bash
# Save work before merging
git stash push -m "WIP: saving work before merge"

# Do merge
git merge feature/branch

# Restore work
git stash pop
```

### Issue 2: "Merge created unexpected results"

```bash
# Undo the merge (if not pushed)
git reset --hard HEAD~1

# Or if already pushed (creates revert commit)
git revert -m 1 HEAD
```

### Issue 3: "Lost track of what I'm merging"

```bash
# Check merge status
git status
# Shows: "You are currently merging branch 'feature/xyz'"

# See what's being merged
cat .git/MERGE_HEAD
# Shows commit hash

# See branch name
git branch --contains $(cat .git/MERGE_HEAD)
```

### Issue 4: "Conflict in file I didn't change"

```bash
# Someone else changed it on main
# View both versions
git show :1:path/to/file  # Base (common ancestor)
git show :2:path/to/file  # Ours (current branch)
git show :3:path/to/file  # Theirs (merging branch)

# Understand changes
git log main..feature/branch -- path/to/file

# Resolve carefully
```

---

## Best Practices

### DO:
- ✅ Merge frequently (prevents large conflicts)
- ✅ Keep feature branches short-lived
- ✅ Test after every merge
- ✅ Write descriptive merge commit messages
- ✅ Review changes before merging
- ✅ Use --no-ff for feature merges
- ✅ Squash WIP commits before merging
- ✅ Validate with automated checks

### DON'T:
- ❌ Merge without reviewing changes
- ❌ Force push after merge (if shared)
- ❌ Leave conflict markers in code
- ❌ Merge broken code
- ❌ Skip tests after merge
- ❌ Merge without updating from main first
- ❌ Ignore merge conflicts (resolve properly!)

---

## Summary

You've mastered Git merging and conflict resolution:

- ✅ Fast-forward, no-ff, and three-way merges
- ✅ Conflict resolution in configs and code
- ✅ Merge strategies (merge, squash, rebase)
- ✅ Aborting and retrying merges
- ✅ Using merge tools effectively
- ✅ Pre and post-merge validation
- ✅ ML project-specific merge scenarios
- ✅ Best practices for team collaboration

**Key Takeaways:**
- Conflicts are normal - resolve them carefully
- Choose merge strategy based on context
- Test thoroughly after every merge
- Clean history matters for maintainability
- Frequent small merges beat rare large merges

**Time to Complete:** ~120 minutes

**Next Exercise:** Exercise 05 - Collaboration Workflows and Pull Requests
