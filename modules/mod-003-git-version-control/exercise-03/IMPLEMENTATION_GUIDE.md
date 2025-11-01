# Exercise 03: Branching for Feature Development - Implementation Guide

## Overview

Master Git branching for parallel feature development in ML projects. Learn to create, manage, and navigate branches for isolated feature work, experiments, and production releases.

**Estimated Time**: 75-90 minutes
**Difficulty**: Beginner to Intermediate
**Prerequisites**: Exercises 01-02 completed

## What You'll Learn

- ✅ Create and switch between branches
- ✅ Branch naming conventions
- ✅ Parallel feature development
- ✅ Isolating experimental work
- ✅ Comparing branches
- ✅ Cleaning up old branches
- ✅ ML project branching strategies

---

## Part 1: Understanding Branches

### Step 1.1: View Current Branches

```bash
# List local branches
git branch

# Output:
# * main

# List all branches (including remote)
git branch -a

# List branches with last commit
git branch -v
```

**What is a Branch?**
- Lightweight movable pointer to a commit
- Allows parallel lines of development
- Enables experimentation without affecting main code
- Essential for team collaboration

### Step 1.2: Create Your First Feature Branch

**Branch Naming Conventions:**
```
feature/short-description      # New features
bugfix/issue-description      # Bug fixes
hotfix/critical-fix           # Urgent production fixes
experiment/idea-name          # Experimental work
release/v1.2.0                # Release branches
```

```bash
# Create and switch to new branch
git checkout -b feature/batch-inference

# OR with modern syntax (Git 2.23+)
git switch -c feature/batch-inference

# Verify you switched
git branch
# Output:
#   main
# * feature/batch-inference
```

---

## Part 2: Working on Feature Branch

### Step 2.1: Implement Batch Inference

```bash
# Create batch inference module
cat > src/api/batch.py << 'EOF'
"""
Batch Inference Module

Handle multiple inference requests efficiently.
"""
from typing import List, Dict
import asyncio
from fastapi import HTTPException

class BatchInferenceHandler:
    """Process multiple inference requests in batches."""

    def __init__(self, batch_size: int = 32, timeout: float = 30.0):
        self.batch_size = batch_size
        self.timeout = timeout

    async def process_batch(
        self,
        requests: List[Dict]
    ) -> List[Dict]:
        """
        Process a batch of inference requests.

        Args:
            requests: List of request dictionaries

        Returns:
            List of prediction results
        """
        if len(requests) > self.batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of {self.batch_size}"
            )

        # Process requests (simplified for demo)
        results = []
        for req in requests:
            # In production, would batch process on GPU
            result = {
                "request_id": req.get("id"),
                "prediction": "class_A",
                "confidence": 0.95
            }
            results.append(result)

        return results
EOF

# Commit on feature branch
git add src/api/batch.py
git commit -m "feat(batch): add batch inference handler

Implement batch processing for multiple inference requests:
- Configurable batch size (default: 32)
- Timeout protection (default: 30s)
- Async processing for better performance
- Validation of batch size limits

Improves throughput for high-volume clients."
```

### Step 2.2: Add Configuration

```bash
# Add batch configuration
cat > configs/batch_config.yaml << 'EOF'
batch_inference:
  max_batch_size: 32
  timeout_seconds: 30.0
  queue_max_size: 1000
  enable_batching: true

  # Performance tuning
  batch_wait_time_ms: 100  # Wait for batch to fill
  max_concurrent_batches: 4
EOF

git add configs/batch_config.yaml
git commit -m "config(batch): add batch inference configuration

Configuration includes:
- Maximum batch size and timeout
- Queue management settings
- Performance tuning parameters
- Feature toggle for enabling/disabling batching"
```

### Step 2.3: Check Status

```bash
# View commits on this branch
git log --oneline main..feature/batch-inference

# Expected output:
# abc1234 config(batch): add batch inference configuration
# def5678 feat(batch): add batch inference handler

# Compare with main branch
git diff main
```

---

## Part 3: Managing Multiple Branches

### Step 3.1: Switch to Main and Create Another Feature

```bash
# Switch back to main
git switch main

# Verify you're on main
git branch
# Output:
# * main
#   feature/batch-inference

# Create another feature branch
git switch -c feature/model-caching

# Implement caching
cat > src/models/cache.py << 'EOF'
"""
Model Caching Module

Cache model predictions to reduce inference load.
"""
from typing import Optional, Dict, Any
import hashlib
import json

class PredictionCache:
    """LRU cache for model predictions."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}

    def _generate_key(self, input_data: Dict) -> str:
        """Generate cache key from input."""
        json_str = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get(self, input_data: Dict) -> Optional[Dict]:
        """Retrieve cached prediction if exists."""
        key = self._generate_key(input_data)
        return self.cache.get(key)

    def set(self, input_data: Dict, prediction: Dict):
        """Store prediction in cache."""
        key = self._generate_key(input_data)

        # Simple LRU: remove oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = prediction
EOF

git add src/models/cache.py
git commit -m "feat(cache): implement prediction caching

Add LRU cache for model predictions:
- SHA-256 based cache keys from input
- Configurable cache size (default: 10k)
- Simple LRU eviction policy
- Reduces redundant inference calls

Can improve response time by 10-100x for repeated requests."
```

### Step 3.2: View All Branches

```bash
# List all branches
git branch

# Output:
#   feature/batch-inference
# * feature/model-caching
#   main

# Visualize branch history
git log --oneline --graph --all

# Expected output (simplified):
# * xyz9012 (HEAD -> feature/model-caching) feat(cache): implement prediction caching
# | * abc1234 (feature/batch-inference) config(batch): add batch inference configuration
# | * def5678 feat(batch): add batch inference handler
# |/
# * mno3456 (main) Previous commits...
```

---

## Part 4: Comparing Branches

### Step 4.1: See What's Different Between Branches

```bash
# Show files changed between main and feature branch
git diff main..feature/model-caching --name-only

# Show actual differences
git diff main..feature/model-caching

# Compare two feature branches
git diff feature/batch-inference..feature/model-caching

# Show commits on one branch but not another
git log main..feature/model-caching --oneline
```

### Step 4.2: Show Branch Information

```bash
# Show last commit on each branch
git branch -v

# Show merged branches
git branch --merged

# Show unmerged branches
git branch --no-merged

# Show branches with tracking info
git branch -vv
```

---

## Part 5: Switching Between Branches

### Step 5.1: Safe Switching

```bash
# Clean working directory required for switching
git status

# If you have uncommitted changes:
# Option 1: Commit them
git add .
git commit -m "wip: work in progress"

# Option 2: Stash them
git stash
git stash list

# Now switch branches
git switch feature/batch-inference

# Apply stash after switching
git stash pop
```

### Step 5.2: Quick Context Switching

```bash
# Switch to previous branch (like cd -)
git switch -

# Creates shortcut for rapid context switching
git switch feature/model-caching
# ... do some work ...
git switch -  # Back to batch-inference
git switch -  # Back to model-caching
```

---

## Part 6: Experimental Branches

### Step 6.1: Create Experiment Branch

```bash
# Switch to main first
git switch main

# Create experiment branch
git switch -c experiment/onnx-runtime

# Implement experimental feature
cat > src/models/onnx_inference.py << 'EOF'
"""
EXPERIMENTAL: ONNX Runtime Inference

Testing ONNX runtime for faster inference.
"""
import onnxruntime as ort
import numpy as np

class ONNXInference:
    """ONNX-based inference engine."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        return self.session.run(
            None,
            {self.input_name: image}
        )[0]
EOF

git add src/models/onnx_inference.py
git commit -m "experiment(onnx): test ONNX runtime integration

EXPERIMENTAL: Testing ONNX runtime as alternative to PyTorch:
- Potential 2-3x inference speedup
- Reduced memory footprint
- Better CPU performance

Status: Proof of concept - needs validation
Performance: TBD - requires benchmarking
Decision: Pending performance tests"
```

### Step 6.2: Branch for Each Experiment

**Best Practice for ML:**
```bash
# Create experiment tracking
git switch -c experiment/quantization-test
# Test INT8 quantization...

git switch main
git switch -c experiment/different-architecture
# Test alternative model...

git switch main
git switch -c experiment/data-augmentation-v2
# Test new augmentation strategy...
```

This allows you to:
- Compare results across branches
- Keep experiments isolated
- Merge successful experiments
- Discard failed experiments without polluting history

---

## Part 7: Branch Cleanup

### Step 7.1: Delete Merged Branches

```bash
# After merging a feature to main
# (Assume we merged batch-inference)

# Delete local branch (safe - only if merged)
git branch -d feature/batch-inference

# Force delete (CAREFUL - deletes even if not merged)
git branch -D experiment/failed-test

# Delete remote branch
git push origin --delete feature/batch-inference
```

### Step 7.2: Clean Up Old Branches

```bash
# List branches by age
git for-each-ref --sort=committerdate refs/heads/ \
  --format='%(committerdate:short) %(refname:short)'

# Delete all merged branches except main
git branch --merged | grep -v "main" | xargs git branch -d

# List stale branches (no commits in 30+ days)
git for-each-ref --sort=-committerdate refs/heads/ \
  --format='%(refname:short) %(committerdate:relative)'
```

---

## Part 8: Branching Strategies for ML Projects

### Strategy 1: GitFlow (Traditional)

```
main                 # Production code
  └─ develop         # Integration branch
      ├─ feature/batch-inference
      ├─ feature/model-caching
      └─ feature/new-model

release/v2.0         # Release preparation
hotfix/critical-bug  # Emergency fixes
```

**Usage:**
```bash
# Start feature
git switch develop
git switch -c feature/new-feature

# Complete feature
git switch develop
git merge feature/new-feature

# Prepare release
git switch -c release/v1.2.0 develop

# Hotfix
git switch -c hotfix/bug-fix main
```

### Strategy 2: GitHub Flow (Simpler)

```
main                    # Always deployable
  ├─ feature/batch
  ├─ feature/caching
  └─ bugfix/normalization
```

**Usage:**
```bash
# Create feature from main
git switch main
git pull
git switch -c feature/new-feature

# Deploy via pull request
# Merge to main = deploy to production
```

### Strategy 3: ML Experiment Flow (Custom)

```
main                           # Stable models
  ├─ experiment/model-v2      # Model experiments
  ├─ experiment/new-arch
  ├─ data/augmentation-v3     # Data experiments
  └─ feature/api-upgrade      # Infrastructure features
```

**Naming Conventions:**
- `experiment/*` - Model/training experiments
- `data/*` - Data processing experiments
- `feature/*` - Infrastructure features
- `model/*` - Production model updates

---

## Practical Exercise

### Complete This Workflow

```bash
# 1. Create metrics export feature
git switch main
git switch -c feature/prometheus-metrics

# 2. Implement metrics endpoint
cat > src/api/metrics.py << 'EOF'
"""Prometheus Metrics Endpoint"""
from fastapi import APIRouter
from prometheus_client import make_asgi_app

router = APIRouter()
metrics_app = make_asgi_app()

@router.get("/health")
def health_check():
    return {"status": "healthy"}
EOF

git add src/api/metrics.py
git commit -m "feat(metrics): add Prometheus metrics endpoint"

# 3. Check your branches
git branch -v

# 4. Compare with main
git diff main --stat

# 5. Switch to another task
git switch main
git switch -c bugfix/memory-leak

# 6. View all branches
git log --oneline --graph --all --decorate
```

---

## Verification Checklist

- [ ] Created at least 3 feature branches
- [ ] Switched between branches successfully
- [ ] Made commits on different branches
- [ ] Compared branches with diff
- [ ] Viewed branch history with log --graph
- [ ] Deleted a branch safely
- [ ] Understand when to use which branch type
- [ ] Can work on multiple features in parallel

---

## Common Issues and Solutions

### Issue 1: "Cannot switch branches - uncommitted changes"

```bash
# Option 1: Commit the changes
git add .
git commit -m "wip: save current work"

# Option 2: Stash the changes
git stash
git switch other-branch
git stash pop

# Option 3: Create temp branch
git switch -c temp-save-work
git add .
git commit -m "temp: saving work"
git switch main
```

---

### Issue 2: "Accidentally committed to wrong branch"

```bash
# If not pushed yet:
# 1. Create correct branch from current state
git branch correct-branch

# 2. Reset current branch
git reset --hard HEAD~1

# 3. Switch to correct branch
git switch correct-branch
```

---

### Issue 3: "Lost track of what's in each branch"

```bash
# Show branches with descriptions
git branch -v

# Show last commit message
git branch --format="%(refname:short) %(subject)"

# Use git log to see branch contents
git log main..feature/branch --oneline

# Visualize all branches
git log --graph --all --decorate --oneline
```

---

### Issue 4: "Don't know which branch I'm on"

```bash
# Current branch
git branch --show-current

# Or look for * in branch list
git branch

# Show in prompt (bash)
export PS1='\w $(__git_ps1 "(%s)") $ '
```

---

## Best Practices

### DO:
- ✅ Create branches from up-to-date main
- ✅ Use descriptive branch names
- ✅ Keep branches focused on single feature
- ✅ Commit regularly on branches
- ✅ Delete merged branches
- ✅ Pull main regularly and rebase/merge
- ✅ Use experiments/* for risky changes

### DON'T:
- ❌ Commit directly to main (use branches!)
- ❌ Create overly long-lived branches
- ❌ Mix multiple features in one branch
- ❌ Leave stale branches around
- ❌ Forget which branch you're on
- ❌ Force push to shared branches

---

## Advanced Tips

### Tip 1: Branch from Specific Commit

```bash
# Create branch from older commit
git switch -c feature/new main~3

# Create branch from specific hash
git switch -c bugfix/old-issue abc1234
```

### Tip 2: Rename Branch

```bash
# Rename current branch
git branch -m new-name

# Rename different branch
git branch -m old-name new-name
```

### Tip 3: Track Remote Branch

```bash
# Create local branch tracking remote
git switch -c feature/new origin/feature/new

# Set upstream for current branch
git branch --set-upstream-to=origin/feature/new
```

---

## Next Steps

1. **Practice Branch Workflows:**
   - Create 5 feature branches
   - Work on them in parallel
   - Practice switching between them

2. **Learn Merging:**
   - Move to Exercise 04: Merging and Conflicts
   - Practice merge strategies
   - Resolve merge conflicts

3. **Set Up Branch Protection:**
   - Protect main branch on GitHub
   - Require pull requests
   - Enable status checks

4. **Automate Branch Management:**
   - Auto-delete merged branches
   - Branch naming validation
   - Stale branch detection

---

## Resources

- [Git Branching Model](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [GitLab Flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html)
- [Branch Naming Conventions](https://dev.to/varbsan/a-simplified-convention-for-naming-branches-and-commits-in-git-il4)

---

## Summary

You've mastered:
- ✅ Creating and switching branches
- ✅ Branch naming conventions
- ✅ Parallel feature development
- ✅ Comparing and visualizing branches
- ✅ Safe branch deletion
- ✅ ML project branching strategies
- ✅ Experiment isolation

**Key Takeaways:**
- Branches enable parallel development
- Use branches to isolate experiments
- Keep branches focused and short-lived
- Delete merged branches regularly
- Choose branching strategy that fits your team

**Time to Complete:** ~90 minutes

**Next Exercise:** Exercise 04 - Merging and Conflict Resolution
