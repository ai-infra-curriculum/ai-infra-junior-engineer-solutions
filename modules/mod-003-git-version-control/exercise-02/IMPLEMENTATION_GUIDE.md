# Exercise 02: Working with Commits and History - Implementation Guide

## Overview

This guide teaches advanced commit techniques and effective Git history management for ML projects. You'll learn to create professional commits, navigate history, amend mistakes, and maintain a clean, searchable commit log.

**Estimated Time**: 75-90 minutes
**Difficulty**: Beginner
**Prerequisites**: Exercise 01 completed, basic Git knowledge

## What You'll Learn

- ✅ Conventional commit message format
- ✅ Atomic commits for code review
- ✅ Navigating and searching Git history
- ✅ Amending commits safely
- ✅ Reverting changes without losing history
- ✅ Using `git show`, `git diff`, `git log` effectively

---

## Part 1: Conventional Commit Messages

### Step 1.1: Understanding Conventional Commits

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting (no code change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Example:**
```
feat(monitoring): add performance tracking module

Implement Prometheus metrics collection for:
- Model inference latency
- Request throughput
- Error rates
- GPU utilization

Metrics exposed on /metrics endpoint for scraping.

Closes #123
```

### Step 1.2: Create Performance Monitoring Module

```bash
# Ensure you're in the ml-inference-api directory
cd ml-inference-api

# Create monitoring module
mkdir -p src/utils
cat > src/utils/monitoring.py << 'EOF'
"""
Performance Monitoring Module

Tracks model inference performance metrics using Prometheus.
"""
import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge

# Metrics
inference_requests_total = Counter(
    'ml_inference_requests_total',
    'Total number of inference requests',
    ['model_name', 'status']
)

inference_latency_seconds = Histogram(
    'ml_inference_latency_seconds',
    'Inference latency in seconds',
    ['model_name']
)

model_loaded = Gauge(
    'ml_model_loaded',
    'Whether model is currently loaded',
    ['model_name', 'version']
)

class PerformanceMonitor:
    """Monitor and track ML model performance metrics."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def track_inference(self, duration: float, status: str):
        """Record inference request metrics."""
        inference_requests_total.labels(
            model_name=self.model_name,
            status=status
        ).inc()

        if status == "success":
            inference_latency_seconds.labels(
                model_name=self.model_name
            ).observe(duration)

    def set_model_status(self, loaded: bool, version: str):
        """Update model load status."""
        model_loaded.labels(
            model_name=self.model_name,
            version=version
        ).set(1 if loaded else 0)
EOF
```

### Step 1.3: Commit with Conventional Format

```bash
# Stage the file
git add src/utils/monitoring.py

# Create commit with conventional format
git commit -m "feat(monitoring): add performance tracking module

Implement Prometheus metrics collection for ML inference:
- Track total inference requests with status labels
- Record inference latency histograms
- Monitor model load status

Metrics include:
  * inference_requests_total: Counter for all requests
  * inference_latency_seconds: Histogram of latencies
  * model_loaded: Gauge for model availability

Metrics will be exposed via /metrics endpoint for Prometheus
scraping in future commits.

Related to monitoring requirements in architecture doc."
```

**Verify Commit:**
```bash
# View last commit
git log -1

# View commit with stats
git show HEAD --stat
```

---

## Part 2: Making Atomic Commits

### Step 2.1: Understanding Atomic Commits

**Atomic commit** = One logical change per commit

**Benefits:**
- Easier code review
- Simpler to revert specific changes
- Clearer history
- Better bisecting for bugs

**Example - Breaking Up a Large Change:**

❌ Bad (multiple changes in one commit):
```
"Add logging, fix preprocessing bug, update documentation, refactor API"
```

✅ Good (atomic commits):
```
"fix(preprocessing): correct image normalization range"
"feat(logging): add structured logging with loguru"
"docs: update API endpoint documentation"
"refactor(api): extract validation into separate function"
```

### Step 2.2: Fix Preprocessing Bug (Atomic Commit 1)

```bash
# Create preprocessing module
cat > src/preprocessing/image.py << 'EOF'
"""
Image Preprocessing Module

Handles image preparation for model inference.
"""
import numpy as np
from PIL import Image
from typing import Tuple

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image array (H, W, C) in [0, 255]

    Returns:
        Normalized image in [0, 1] range
    """
    # Bug fix: Previous version divided by 256 instead of 255
    return image.astype(np.float32) / 255.0

def resize_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize image to target dimensions."""
    return image.resize(size, Image.LANCZOS)

def preprocess(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Args:
        image: PIL Image
        target_size: Target (width, height)

    Returns:
        Preprocessed numpy array ready for inference
    """
    # Resize
    resized = resize_image(image, target_size)

    # Convert to array
    arr = np.array(resized)

    # Normalize
    normalized = normalize_image(arr)

    return normalized
EOF

# Commit ONLY this bug fix
git add src/preprocessing/image.py
git commit -m "fix(preprocessing): correct image normalization range

Change normalization divisor from 256 to 255 for correct [0, 1]
range mapping.

Previous implementation:
  normalized = image / 256.0  # Wrong: max value = 255/256 = 0.996

Corrected implementation:
  normalized = image / 255.0  # Correct: max value = 255/255 = 1.0

This bug caused slight dimming of bright pixels and affected
model accuracy by approximately 2% on validation set.

Fixes #142"
```

### Step 2.3: Add Logging (Atomic Commit 2)

```bash
# Create logging configuration
cat > src/utils/logger.py << 'EOF'
"""
Logging Configuration

Structured logging setup for the application.
"""
import sys
from loguru import logger

def setup_logging(log_level: str = "INFO", log_file: str = "logs/api.log"):
    """
    Configure application logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # File handler with JSON format
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="500 MB",
        retention="10 days",
        compression="zip"
    )

    return logger
EOF

# Commit ONLY logging addition
git add src/utils/logger.py
git commit -m "feat(logging): add structured logging with loguru

Implement logging configuration with:
- Colored console output for development
- JSON file logging for production
- Automatic log rotation (500 MB)
- 10-day retention with compression

Logging includes:
  * Timestamp with milliseconds
  * Log level (DEBUG, INFO, WARNING, ERROR)
  * Module and function context
  * Line numbers for debugging

Log files stored in logs/ directory and rotated automatically
to prevent disk space issues."
```

---

## Part 3: Exploring Commit History

### Step 3.1: Basic Log Viewing

```bash
# Simple one-line history
git log --oneline

# Expected output (your hashes will differ):
# a1b2c3d feat(logging): add structured logging with loguru
# e4f5g6h fix(preprocessing): correct image normalization range
# i7j8k9l feat(monitoring): add performance tracking module
# ... (previous commits from Exercise 01)

# Detailed log with stats
git log --stat

# Graph view (useful for branches)
git log --oneline --graph --all

# Last 3 commits
git log -3

# Commits from specific author
git log --author="Your Name"

# Commits in last 24 hours
git log --since="24 hours ago"

# Commits with "fix" in message
git log --grep="fix"
```

### Step 3.2: Searching History for Specific Changes

```bash
# Find commits that changed a specific file
git log -- src/utils/monitoring.py

# Find when a function was added
git log -S"def normalize_image" --source --all

# Find commits that mention "preprocessing"
git log --all --grep="preprocessing"

# See what changed in last commit
git show HEAD

# See what changed 2 commits ago
git show HEAD~2

# Compare current with 3 commits ago
git diff HEAD~3 HEAD
```

### Step 3.3: Detailed Commit Inspection

```bash
# Show full details of specific commit
git show a1b2c3d

# Show only changed files
git show a1b2c3d --name-only

# Show stats
git show a1b2c3d --stat

# Show diff for specific file
git show a1b2c3d -- src/utils/logger.py
```

---

## Part 4: Amending Commits

### Step 4.1: When to Amend

**Safe to amend when:**
- ✅ Commit is NOT pushed to remote
- ✅ You're the only one working on the branch
- ✅ Fixing typos or small mistakes
- ✅ Adding forgotten files

**DO NOT amend when:**
- ❌ Commit is pushed and others may have pulled it
- ❌ Commit is on a shared branch
- ❌ Would rewrite public history

### Step 4.2: Fix Commit Message

```bash
# Oops! Typo in last commit message
# Amend the message
git commit --amend -m "feat(logging): add structured logging with loguru

Implement logging configuration with:
- Colored console output for development
- JSON file logging for production
- Automatic log rotation (500 MB)
- 10-day retention with compression

Logging includes:
  * Timestamp with milliseconds
  * Log level (DEBUG, INFO, WARNING, ERROR)
  * Module and function context
  * Line numbers for debugging

Log files stored in logs/ directory and rotated automatically
to prevent disk space issues.

Updated: Fixed typo in message"
```

### Step 4.3: Add Forgotten File to Last Commit

```bash
# Create __init__.py file we forgot
touch src/utils/__init__.py

# Stage it
git add src/utils/__init__.py

# Add to last commit (keep same message)
git commit --amend --no-edit

# Verify
git show HEAD --stat
```

---

## Part 5: Reverting Changes

### Step 5.1: Understanding Revert vs Reset

**`git revert`** (Safe - preserves history):
- Creates NEW commit that undoes changes
- Safe for shared branches
- Preserves history
- Use in production

**`git reset`** (Dangerous - rewrites history):
- Moves branch pointer backward
- Deletes commits
- NEVER use on pushed commits
- Use only locally

### Step 5.2: Revert a Commit Safely

```bash
# Suppose the normalization fix caused issues
# Let's revert it safely

# Find the commit hash
git log --oneline --grep="normalization"

# Revert it (creates new commit)
git revert e4f5g6h

# Git opens editor with message:
# "Revert 'fix(preprocessing): correct image normalization range'"
#
# Add explanation:
```

Edit the revert commit message:
```
Revert "fix(preprocessing): correct image normalization range"

This reverts commit e4f5g6h.

Reverting because the normalization change broke compatibility
with pretrained model weights that expected [0, 256] range.

Will re-implement with model retraining in separate branch.
```

```bash
# Save and close editor

# Verify revert
git log --oneline -3
```

### Step 5.3: Revert Multiple Commits

```bash
# Revert last 2 commits (creates 2 revert commits)
git revert HEAD~2..HEAD

# OR revert range without separate commits
git revert --no-commit HEAD~2..HEAD
git commit -m "Revert recent monitoring changes

Reverts commits:
- feat(monitoring): add performance tracking
- feat(logging): add structured logging

Reason: Performance overhead in production requires
optimization before deployment."
```

---

## Part 6: Using Git Diff Effectively

### Step 6.1: Different Types of Diffs

```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --cached

# Show all changes (staged + unstaged)
git diff HEAD

# Compare two commits
git diff a1b2c3d..e4f5g6h

# Compare with branch
git diff main..feature-branch

# Show only changed file names
git diff --name-only

# Show stats
git diff --stat
```

### Step 6.2: Diff Specific Files

```bash
# Diff specific file
git diff src/utils/logger.py

# Diff between commits for one file
git diff HEAD~3 HEAD -- src/utils/monitoring.py

# Show function-level context
git diff --function-context
```

---

## Part 7: Best Practices for ML Projects

### Step 7.1: Commit Frequency

**Good Rhythm:**
- After completing a logical unit of work
- Before switching tasks
- Before pulling changes
- At end of day (if work is stable)

**For ML Projects:**
```bash
# Commit experiment configuration separately
git commit -m "experiment: add hyperparameter config for run-42"

# Commit model architecture changes separately
git commit -m "feat(model): add attention layer to classifier"

# Commit training script changes separately
git commit -m "feat(training): implement early stopping callback"

# Commit results/metrics separately
git commit -m "experiment: add training results for run-42"
```

### Step 7.2: What NOT to Commit

Never commit:
- ❌ Large model files (.pth, .h5) - use Git LFS
- ❌ Training data - use DVC or remote storage
- ❌ Secrets (.env files with API keys)
- ❌ Debug code or commented blocks
- ❌ Temporary files
- ❌ Jupyter checkpoint files
- ❌ `__pycache__` directories

### Step 7.3: Organizing Commits for Review

**Strategy: Logical Progression**

```bash
# 1. Add interface/types first
git commit -m "feat(api): define inference request/response models"

# 2. Implement core logic
git commit -m "feat(inference): implement batch prediction handler"

# 3. Add error handling
git commit -m "feat(inference): add error handling for invalid images"

# 4. Add tests
git commit -m "test(inference): add unit tests for batch handler"

# 5. Add documentation
git commit -m "docs(inference): document batch prediction API"
```

This makes code review much easier!

---

## Practical Exercise

### Complete This Scenario

Add rate limiting to your ML inference API:

```bash
# 1. Create rate limiter module
cat > src/utils/rate_limiter.py << 'EOF'
"""Rate Limiting for API Requests"""
from fastapi import HTTPException
from typing import Dict
import time

class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.rate = requests_per_minute
        self.tokens: Dict[str, float] = {}

    def check_limit(self, client_id: str) -> bool:
        """Check if request should be allowed."""
        now = time.time()

        if client_id not in self.tokens:
            self.tokens[client_id] = self.rate

        # Refill tokens based on time passed
        time_passed = now - self.tokens.get(f"{client_id}_last", now)
        self.tokens[client_id] = min(
            self.rate,
            self.tokens[client_id] + time_passed * (self.rate / 60)
        )
        self.tokens[f"{client_id}_last"] = now

        # Check if we have tokens
        if self.tokens[client_id] >= 1:
            self.tokens[client_id] -= 1
            return True
        return False
EOF

# 2. Make atomic commit
git add src/utils/rate_limiter.py
git commit -m "feat(api): implement token bucket rate limiter

Add rate limiting to prevent API abuse:
- Token bucket algorithm
- Configurable requests per minute
- Per-client tracking
- Automatic token refill

Default: 60 requests/minute per client
Raises HTTPException when limit exceeded"

# 3. View your commit
git show HEAD

# 4. Check history
git log --oneline -5
```

---

## Verification and Testing

### Checklist

- [ ] Created at least 3 atomic commits
- [ ] Used conventional commit format
- [ ] Viewed commit history with multiple commands
- [ ] Successfully amended a commit message
- [ ] Performed a safe revert
- [ ] Used `git show` to inspect commits
- [ ] Used `git diff` to compare changes
- [ ] Understand when to amend vs revert

### Validation Commands

```bash
# Count your commits
git rev-list --count HEAD

# Show all your commit messages
git log --oneline

# Verify conventional commit format
git log --pretty=format:"%s" | head -5

# Check for clean working tree
git status
```

---

## Common Issues and Solutions

### Issue 1: "I committed to wrong branch"

```bash
# If not pushed yet:
# 1. Note the commit hash
git log -1

# 2. Switch to correct branch
git checkout correct-branch

# 3. Cherry-pick the commit
git cherry-pick abc123

# 4. Go back and remove from wrong branch
git checkout wrong-branch
git reset --hard HEAD~1
```

---

### Issue 2: "Commit message has typo"

```bash
# If not pushed:
git commit --amend -m "Corrected message"

# If already pushed:
# Don't amend! Make a new commit or accept the typo
```

---

### Issue 3: "Forgot to add file to commit"

```bash
# If not pushed:
git add forgotten-file.py
git commit --amend --no-edit

# If already pushed:
# Make a new commit
git add forgotten-file.py
git commit -m "Add forgotten file from previous commit"
```

---

### Issue 4: "Want to undo last commit but keep changes"

```bash
# Remove commit but keep files staged
git reset --soft HEAD~1

# Remove commit and unstage files
git reset HEAD~1

# Remove commit and discard changes (DANGEROUS!)
git reset --hard HEAD~1
```

---

## Advanced Tips

### Tip 1: Interactive Rebase (Local Only)

```bash
# Reorder/squash/edit last 3 commits (NOT PUSHED!)
git rebase -i HEAD~3

# This opens editor where you can:
# - reword: Change commit message
# - squash: Combine with previous commit
# - fixup: Like squash but discard message
# - reorder: Move lines to reorder commits
```

### Tip 2: Stashing Changes

```bash
# Save work in progress without committing
git stash

# List stashes
git stash list

# Apply most recent stash
git stash pop

# Apply specific stash
git stash apply stash@{1}
```

### Tip 3: Bisecting for Bugs

```bash
# Find which commit introduced a bug
git bisect start
git bisect bad                 # Current version is bad
git bisect good abc123         # This older commit was good

# Git checks out middle commit, you test it:
git bisect good  # or git bisect bad

# Repeat until Git finds the exact commit
git bisect reset  # When done
```

---

## Next Steps

After mastering commits and history:

1. **Practice More:**
   - Create 10+ commits with good messages
   - Practice reverting and amending
   - Experiment with different log formats

2. **Learn Branching:**
   - Move to Exercise 03: Branching and Merging
   - Practice feature branches
   - Learn merge strategies

3. **Set Up Pre-commit Hooks:**
   - Lint commit messages
   - Run tests before commit
   - Format code automatically

4. **Explore Automation:**
   - GitHub Actions for CI/CD
   - Automated changelog generation
   - Commit message validation

---

## Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Git Log Documentation](https://git-scm.com/docs/git-log)
- [Interactive Rebase Guide](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)
- [Git Bisect Tutorial](https://git-scm.com/docs/git-bisect)

---

## Summary

You've mastered:
- ✅ Conventional commit message format
- ✅ Creating atomic, reviewable commits
- ✅ Navigating Git history with log, show, diff
- ✅ Amending commits safely
- ✅ Reverting changes without losing history
- ✅ Best practices for ML project commits

**Key Takeaways:**
- Atomic commits make code review easier
- Conventional commits enable automation
- `git revert` is safer than `git reset` for shared branches
- Good commit messages explain WHY, not just WHAT
- Never rewrite pushed history

**Time to Complete:** ~90 minutes with practice

**Next Exercise:** Exercise 03 - Branching and Merging Strategies
