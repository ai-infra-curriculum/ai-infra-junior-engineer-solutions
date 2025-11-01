# Exercise 03: Branching for Feature Development - Solution

## Overview

This solution demonstrates Git branching strategies for parallel ML feature development. It shows how to create, manage, and compare branches while developing multiple features simultaneously - a critical skill for ML infrastructure teams working on complex projects.

## Learning Objectives

This exercise covers:

- ✅ Creating branches for features and bug fixes
- ✅ Branch naming conventions (feature/, fix/, experiment/)
- ✅ Parallel feature development workflow
- ✅ Switching between branches safely
- ✅ Comparing branches (commits and code)
- ✅ Stashing uncommitted changes
- ✅ Branch visualization techniques
- ✅ Deleting merged and unmerged branches
- ✅ Complete branching workflow for ML projects

## Solution Structure

```
exercise-03/
├── working-repo/          # Repository with multiple branches
│   ├── main              # Main development branch
│   ├── feature/batch-inference
│   ├── feature/model-caching
│   ├── feature/prometheus-metrics
│   ├── experiment/onnx-runtime  # Experimental (deleted)
│   └── fix/null-pointer-in-preprocessing  # Bug fix (merged)
├── scripts/
│   └── demonstrate_branching.sh  # Automated demonstration
├── docs/
│   └── QUICK_REFERENCE.md  # Branching command reference
└── README.md             # This file
```

## Key Concepts Demonstrated

### 1. Branch Naming Conventions

Consistent naming makes team collaboration easier:

```bash
# Feature branches
feature/batch-inference
feature/model-caching
feature/prometheus-metrics

# Bug fixes
fix/null-pointer-in-preprocessing
fix/memory-leak-in-cache

# Hotfixes (critical production issues)
hotfix/security-vulnerability
hotfix/crash-on-startup

# Experimental features
experiment/onnx-runtime
experiment/quantized-models

# Release branches
release/v1.0.0
release/v2.0.0-beta
```

**Benefits:**
- Clear categorization of work
- Easy to filter and search
- Automated CI/CD triggers based on prefix
- Team understands branch purpose immediately

### 2. Creating Branches

Multiple ways to create and switch branches:

```bash
# Method 1: Create then switch (traditional)
git branch feature/new-feature
git checkout feature/new-feature

# Method 2: Create and switch (traditional shorthand)
git checkout -b feature/new-feature

# Method 3: Create and switch (modern)
git switch -c feature/new-feature

# Method 4: Create branch from specific commit
git branch feature/new-feature abc123
```

**Recommended:** Use `git switch -c` for clarity (Git 2.23+)

### 3. Parallel Development

Work on multiple features simultaneously:

```bash
# Feature 1: Batch inference
git switch -c feature/batch-inference
# ... develop batch processing ...
git commit -m "feat: add batch inference"

# Feature 2: Model caching (independent)
git switch -c feature/model-caching
# ... develop caching layer ...
git commit -m "feat: add model caching"

# Feature 3: Prometheus metrics (independent)
git switch -c feature/prometheus-metrics
# ... add monitoring ...
git commit -m "feat: add Prometheus metrics"
```

**Benefits:**
- Features develop independently
- No waiting for other features to complete
- Easy to prioritize and release separately
- Can test features in isolation

### 4. Comparing Branches

View differences between branches:

```bash
# See commits in feature branch not in main
git log main..feature/batch-inference

# See what files changed
git diff main..feature/batch-inference --name-only

# See detailed code changes
git diff main..feature/batch-inference

# Compare with statistics
git diff main..feature/batch-inference --stat

# Compare two feature branches
git diff feature/a..feature/b
```

### 5. Stashing Changes

Save uncommitted work when switching branches:

```bash
# Save uncommitted changes
git stash save "WIP: refactoring batch processor"

# List all stashes
git stash list

# Apply most recent stash
git stash pop

# Apply specific stash
git stash apply stash@{1}

# View stash contents
git stash show -p stash@{0}

# Drop a stash
git stash drop stash@{0}

# Clear all stashes
git stash clear
```

**Use case:** Switch branches mid-work without committing

### 6. Branch Visualization

See branch structure graphically:

```bash
# Simple graph
git log --oneline --graph --all

# Detailed graph with colors
git log --all --graph --pretty=format:'%C(yellow)%h%Creset -%C(cyan)%d%Creset %s %C(green)(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit

# Show branch relationships
git show-branch main feature/* fix/*

# Branches with last commit
git branch -v

# Merged vs unmerged
git branch --merged
git branch --no-merged
```

## Quick Start

### Run the Automated Demonstration

```bash
cd scripts/
chmod +x demonstrate_branching.sh
./demonstrate_branching.sh
```

This script:
1. Copies repository from Exercise 02
2. Creates multiple feature branches
3. Develops features in parallel:
   - Batch inference processing
   - Model result caching
   - Prometheus metrics export
   - Experimental ONNX runtime
4. Demonstrates branch comparison
5. Shows stashing workflow
6. Creates and merges bug fix
7. Visualizes branch structure
8. Cleans up branches

The script pauses after each section for review.

### Explore the Repository Manually

```bash
cd working-repo/

# View all branches
git branch -a

# See branch history
git log --oneline --graph --all

# Switch to feature branch
git switch feature/batch-inference

# Compare with main
git diff main..feature/batch-inference --stat

# View commits unique to this branch
git log main..feature/batch-inference --oneline
```

## Features Developed

### 1. Batch Inference Processing

**Branch:** `feature/batch-inference`

**Purpose:** Process multiple images efficiently in batches

**Files Added:**
- `src/utils/batch_processor.py` - BatchProcessor class
- `src/models/classifier.py` - Added `predict_batch()` method
- `src/api/app.py` - Added `/predict/batch` endpoint

**Key Features:**
```python
class BatchProcessor:
    """Process multiple images in batches."""

    def __init__(self, classifier, preprocessor, batch_size=32):
        self.batch_size = batch_size

    async def process_batch(self, image_paths: List[Path]):
        """Process batch of images efficiently."""
        # Batch processing logic
```

**Benefits:**
- 3-5x faster than sequential processing
- Better GPU utilization
- Configurable batch size
- Async processing support

### 2. Model Result Caching

**Branch:** `feature/model-caching`

**Purpose:** Cache inference results to avoid redundant predictions

**Files Added:**
- `src/utils/cache.py` - ModelCache with TTL and LRU
- `src/api/cache_middleware.py` - FastAPI caching middleware
- `src/api/app.py` - Cache management endpoints

**Key Features:**
```python
class ModelCache:
    """Cache with TTL and LRU eviction."""

    def __init__(self, ttl_seconds=3600, max_entries=10000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

    def get(self, key: str) -> Optional[Any]:
        """Get cached value with TTL check."""

    def set(self, key: str, value: Any):
        """Set value with LRU eviction."""
```

**Benefits:**
- Instant responses for cached results
- Configurable TTL (time-to-live)
- LRU eviction when full
- Cache hit/miss statistics

### 3. Prometheus Metrics

**Branch:** `feature/prometheus-metrics`

**Purpose:** Export ML inference metrics for monitoring

**Files Added:**
- `src/utils/prometheus_metrics.py` - PrometheusMetrics class
- `src/api/app.py` - `/metrics` and `/metrics/summary` endpoints
- `configs/monitoring/prometheus.yml` - Prometheus configuration

**Metrics Tracked:**
```
inference_requests_total - Total requests
inference_requests_active - Active requests (gauge)
inference_latency_seconds - Latency histogram
inference_predictions_total - Predictions by class
inference_errors_total - Errors by type
process_uptime_seconds - Service uptime
```

**Benefits:**
- Production-ready monitoring
- Prometheus integration
- Latency percentiles (P50, P95, P99)
- Error tracking

### 4. Experimental ONNX Runtime

**Branch:** `experiment/onnx-runtime` (deleted)

**Purpose:** Test ONNX Runtime for faster inference

**Files Added:**
- `src/models/onnx_classifier.py` - ONNX inference
- `requirements.txt` - Added onnxruntime

**Status:** Experimental - requires more testing

**Potential Benefits:**
- 2-3x faster inference
- Lower memory usage
- Cross-platform deployment

**Why deleted:** Not ready for production, needs benchmarking

### 5. Null Pointer Bug Fix

**Branch:** `fix/null-pointer-in-preprocessing` (merged and deleted)

**Purpose:** Fix crash when processing invalid images

**Files Added:**
- `src/preprocessing/validation.py` - Input validation

**Fix:**
```python
def validate_image(image_data: bytes) -> Optional[Image.Image]:
    """Safely validate and load image."""
    if not image_data:
        return None

    try:
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        # Reopen after verify
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        log_error(f"Invalid image: {e}")
        return None
```

**Result:** Merged to main, branch deleted

## Branch Workflow

### Complete Feature Development Workflow

```bash
# 1. Start from main
git switch main
git pull origin main

# 2. Create feature branch
git switch -c feature/new-feature

# 3. Develop feature
# ... make changes ...
git add .
git commit -m "feat: implement new feature"

# 4. Keep feature branch updated
git switch main
git pull origin main
git switch feature/new-feature
git rebase main  # Or merge main into feature

# 5. Push feature branch
git push -u origin feature/new-feature

# 6. Create pull request (on GitHub/GitLab)
# ... code review ...

# 7. Merge to main (via pull request)
git switch main
git pull origin main

# 8. Delete feature branch
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

### Bug Fix Workflow

```bash
# 1. Create fix branch from main
git switch main
git switch -c fix/bug-description

# 2. Fix the bug
# ... make changes ...
git commit -m "fix: resolve bug description"

# 3. Test the fix
# ... run tests ...

# 4. Merge immediately (for critical bugs)
git switch main
git merge fix/bug-description --no-edit

# 5. Deploy to production
# ... deploy ...

# 6. Clean up
git branch -d fix/bug-description

# 7. Update feature branches with fix
git switch feature/some-feature
git cherry-pick <fix-commit-hash>
```

### Experimental Feature Workflow

```bash
# 1. Create experimental branch
git switch -c experiment/risky-feature

# 2. Experiment freely
# ... try different approaches ...
git commit -m "experiment: try approach A"
git commit -m "experiment: try approach B"

# 3. Decision time
# If successful: rename to feature branch
git branch -m experiment/risky-feature feature/proven-feature

# If failed: delete branch
git switch main
git branch -D experiment/risky-feature
```

## Branch Management

### Listing Branches

```bash
# Local branches
git branch

# Local branches with last commit
git branch -v

# Local branches with tracking info
git branch -vv

# All branches (local + remote)
git branch -a

# Remote branches only
git branch -r

# Merged branches
git branch --merged

# Unmerged branches
git branch --no-merged
```

### Deleting Branches

```bash
# Delete merged branch (safe)
git branch -d feature/completed-feature

# Force delete unmerged branch (careful!)
git branch -D experiment/failed-experiment

# Delete remote branch
git push origin --delete feature/old-feature

# Prune deleted remote branches
git fetch --prune
```

### Renaming Branches

```bash
# Rename current branch
git branch -m new-name

# Rename specific branch
git branch -m old-name new-name

# Update remote after rename
git push origin :old-name new-name
git push origin -u new-name
```

## Best Practices

### 1. Branch Naming

✅ **Good:**
```bash
feature/batch-inference
fix/null-pointer-in-preprocessing
hotfix/security-vulnerability
experiment/onnx-runtime
release/v1.0.0
```

❌ **Bad:**
```bash
johns-branch
new-stuff
temp
feature1
fix-2
```

### 2. Branch Size

✅ **Good:** Small, focused branches
- One feature per branch
- Easy to review (< 500 lines changed)
- Quick to merge (hours to days)

❌ **Bad:** Large, long-lived branches
- Multiple features
- Thousands of lines changed
- Weeks or months of work
- High merge conflict risk

### 3. Keep Branches Updated

```bash
# Regularly sync with main
git switch feature/my-feature
git fetch origin
git rebase origin/main

# Or merge main
git merge origin/main
```

### 4. Clean Up Regularly

```bash
# Delete merged local branches
git branch --merged | grep -v "main" | xargs git branch -d

# Prune remote tracking branches
git fetch --prune

# View stale branches
git branch -vv | grep ': gone]'
```

### 5. Descriptive Commits on Branches

```bash
# Good commit messages even on branches
git commit -m "feat: add batch processing for images

Implement BatchProcessor class that processes images
in configurable batch sizes for improved throughput."

# Not just "WIP" or "updates"
```

## Common Workflows

### Workflow 1: Start New Feature

```bash
# Always start from updated main
git switch main
git pull origin main

# Create feature branch
git switch -c feature/user-authentication

# Develop feature
# ... make changes ...

# Commit regularly
git add .
git commit -m "feat: add user authentication"
```

### Workflow 2: Switch Features Mid-Work

```bash
# Working on feature A
git switch feature/feature-a
# ... make changes but not ready to commit ...

# Need to switch to feature B
git stash save "WIP: refactoring feature A"
git switch feature/feature-b
# ... work on feature B ...

# Return to feature A
git switch feature/feature-a
git stash pop
```

### Workflow 3: Urgent Bug Fix

```bash
# Working on feature when bug reported
git switch feature/my-feature
git stash save "WIP: my feature"

# Create fix branch
git switch main
git switch -c fix/critical-bug

# Fix bug
# ... make changes ...
git commit -m "fix: resolve critical bug"

# Merge immediately
git switch main
git merge fix/critical-bug
git push origin main

# Return to feature
git switch feature/my-feature
git cherry-pick <fix-commit>  # Include fix in feature
git stash pop
```

### Workflow 4: Compare Features Before Merging

```bash
# Review what changed
git diff main..feature/my-feature --stat

# See commit history
git log main..feature/my-feature --oneline

# Visualize branch
git log --graph --oneline main feature/my-feature

# Test feature locally
git switch feature/my-feature
# ... run tests ...

# Merge if good
git switch main
git merge feature/my-feature
```

## Troubleshooting

### Issue: "Cannot switch branch - uncommitted changes"

**Problem:** Trying to switch with uncommitted changes

**Solutions:**
```bash
# Option 1: Stash changes
git stash save "WIP: description"
git switch other-branch
# ... work ...
git switch original-branch
git stash pop

# Option 2: Commit changes
git add .
git commit -m "WIP: work in progress"
git switch other-branch

# Option 3: Discard changes (careful!)
git restore .
git switch other-branch
```

### Issue: "Branch already exists"

**Problem:** Trying to create a branch that already exists

**Solutions:**
```bash
# Check existing branches
git branch -a

# Use different name
git switch -c feature/batch-inference-v2

# Or delete old branch first (if safe)
git branch -D feature/batch-inference
git switch -c feature/batch-inference
```

### Issue: "Lost commits after branch deletion"

**Problem:** Accidentally deleted branch with unmerged commits

**Solution:**
```bash
# Find commit using reflog
git reflog

# Recreate branch at that commit
git branch recovered-branch <commit-hash>

# Or cherry-pick specific commits
git cherry-pick <commit-hash>
```

### Issue: "Too many branches"

**Problem:** Many stale branches cluttering repository

**Solution:**
```bash
# List merged branches
git branch --merged main

# Delete merged branches
git branch --merged main | grep -v "main" | xargs git branch -d

# Delete remote branches
git push origin --delete old-branch-name

# Prune remote tracking
git fetch --prune
```

## Verification Checklist

- [ ] Can create branches with `git branch` and `git switch -c`
- [ ] Can switch between branches safely
- [ ] Understand branch naming conventions
- [ ] Can develop features in parallel
- [ ] Can compare branches (log, diff, stat)
- [ ] Can stash and restore uncommitted changes
- [ ] Can visualize branch structure
- [ ] Can identify merged vs unmerged branches
- [ ] Can delete branches safely
- [ ] Understand complete feature workflow

## Related Documentation

- **QUICK_REFERENCE.md**: Comprehensive branching commands
- **Exercise 02 README**: Commit and history basics
- **Exercise 04**: Merging strategies (next exercise)

## Next Steps

After completing this exercise:

1. **Practice** - Use feature branches in your projects
2. **Experiment** - Try different branching strategies
3. **Learn** - Study merging and conflict resolution (Exercise 04)
4. **Apply** - Use branches for team collaboration

## Summary

This exercise demonstrated:

- ✅ Professional branch naming conventions
- ✅ Creating and switching branches
- ✅ Parallel feature development
- ✅ Branch comparison techniques
- ✅ Stashing uncommitted changes
- ✅ Branch visualization
- ✅ Branch cleanup and management
- ✅ Complete branching workflows

**Key Takeaway:** Branches enable parallel development and experimentation without affecting the main codebase. Master branching to work efficiently on complex ML infrastructure projects.

## Support

For questions or issues:
- Review `docs/QUICK_REFERENCE.md`
- Run the demonstration script
- Check branch structure with `git log --graph`
- Refer to Git official documentation

---

**Exercise 03 Complete** ✓

Ready to move on to Exercise 04: Merging and Conflict Resolution
