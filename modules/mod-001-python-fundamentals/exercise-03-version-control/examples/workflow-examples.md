# Git Workflow Examples

Real-world Git workflow scenarios for AI infrastructure engineers.

## Table of Contents

1. [Daily Development Workflow](#daily-development-workflow)
2. [Feature Development](#feature-development)
3. [Bug Fix Workflow](#bug-fix-workflow)
4. [Hotfix Workflow](#hotfix-workflow)
5. [Code Review Process](#code-review-process)
6. [Handling Merge Conflicts](#handling-merge-conflicts)
7. [ML Experiment Workflow](#ml-experiment-workflow)
8. [Release Workflow](#release-workflow)

---

## Daily Development Workflow

### Starting Your Day

```bash
# 1. Navigate to project
cd ~/projects/ml-platform

# 2. Ensure you're on main/develop
git checkout main

# 3. Pull latest changes
git pull origin main

# 4. Check status
git status

# 5. View recent activity
git log --oneline -10
```

### During Development

```bash
# Make changes to files
vim src/api/endpoint.py

# Check what changed
git diff

# Stage specific files
git add src/api/endpoint.py

# Or stage all changes
git add .

# Review staged changes
git diff --staged

# Commit with meaningful message
git commit -m "feat(api): add authentication to endpoint"

# Push to remote
git push origin main
```

### End of Day

```bash
# Save work in progress (if not ready to commit)
git stash save "WIP: working on feature X"

# Or create a WIP commit
git add .
git commit -m "WIP: feature X in progress"

# Push to backup
git push origin main
```

---

## Feature Development

### Scenario: Add Model Caching Feature

```bash
# Step 1: Create feature branch
git checkout main
git pull origin main
git checkout -b feature/model-caching

# Step 2: Implement feature
echo "Implementing model caching..."

# Create cache module
cat > src/cache.py << EOF
"""Model caching implementation."""
import pickle
from pathlib import Path

class ModelCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, model_id: str):
        cache_file = self.cache_dir / f"{model_id}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, model_id: str, model):
        cache_file = self.cache_dir / f"{model_id}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(model, f)
EOF

# Step 3: Add tests
cat > tests/test_cache.py << EOF
"""Tests for model cache."""
import pytest
from src.cache import ModelCache

def test_cache_set_and_get(tmp_path):
    cache = ModelCache(str(tmp_path))
    cache.set("test_model", {"data": "test"})
    result = cache.get("test_model")
    assert result == {"data": "test"}
EOF

# Step 4: Run tests
pytest tests/test_cache.py

# Step 5: Commit implementation
git add src/cache.py tests/test_cache.py
git commit -m "feat(cache): implement model caching

- Add ModelCache class with get/set methods
- Add unit tests for cache operations
- Use pickle for serialization"

# Step 6: Update documentation
cat >> README.md << EOF

## Model Caching

The platform now supports model caching to improve performance.

\`\`\`python
from src.cache import ModelCache

cache = ModelCache("./cache")
cache.set("my_model", model)
cached_model = cache.get("my_model")
\`\`\`
EOF

git add README.md
git commit -m "docs(cache): add model caching documentation"

# Step 7: Push feature branch
git push -u origin feature/model-caching

# Step 8: Create Pull Request (via GitHub UI or CLI)
gh pr create \
  --title "feat: Add model caching" \
  --body "Implements model caching to reduce loading times.

## Changes
- Add ModelCache class
- Add unit tests
- Update documentation

## Testing
- [x] Unit tests pass
- [x] Manual testing completed
- [ ] Performance benchmarks (will add in review)

Closes #123"

# Step 9: Address review comments
# ... make changes based on feedback ...

git add .
git commit -m "refactor(cache): use LRU cache eviction policy"
git push origin feature/model-caching

# Step 10: After PR approval, merge via GitHub UI

# Step 11: Clean up local branch
git checkout main
git pull origin main
git branch -d feature/model-caching
```

---

## Bug Fix Workflow

### Scenario: Fix API Error Handling

```bash
# Step 1: Create bugfix branch
git checkout main
git pull origin main
git checkout -b bugfix/api-error-handling

# Step 2: Reproduce the bug
python -m pytest tests/test_api.py::test_invalid_input
# Test fails as expected

# Step 3: Fix the bug
cat > src/api/validators.py << EOF
"""Input validation for API."""
from typing import Any, Dict

def validate_input(data: Dict[str, Any]) -> bool:
    """Validate API input data."""
    required_fields = ['model_id', 'input_data']

    # Fix: Check for None values
    for field in required_fields:
        if field not in data or data[field] is None:
            raise ValueError(f"Missing required field: {field}")

    return True
EOF

# Step 4: Verify fix
python -m pytest tests/test_api.py::test_invalid_input
# Test passes

# Step 5: Add regression test
cat >> tests/test_api.py << EOF

def test_none_values_rejected():
    """Ensure None values are properly rejected."""
    from src.api.validators import validate_input

    with pytest.raises(ValueError, match="Missing required field"):
        validate_input({'model_id': None, 'input_data': 'test'})
EOF

# Step 6: Commit fix
git add src/api/validators.py tests/test_api.py
git commit -m "fix(api): handle None values in input validation

- Add explicit None check for required fields
- Add regression test
- Raises ValueError with clear message

Fixes #456"

# Step 7: Push and create PR
git push -u origin bugfix/api-error-handling

gh pr create \
  --title "fix: Handle None values in API input validation" \
  --body "Fixes #456

## Problem
API was not properly handling None values in required fields.

## Solution
Added explicit None checks in validation logic.

## Testing
- [x] Regression test added
- [x] All existing tests pass"
```

---

## Hotfix Workflow

### Scenario: Critical Production Bug

```bash
# Step 1: Create hotfix from production
git checkout main  # or production branch
git pull origin main
git checkout -b hotfix/critical-memory-leak

# Step 2: Fix immediately
cat > src/utils/cleanup.py << EOF
"""Memory cleanup utilities."""
import gc

def cleanup_model(model):
    """Properly cleanup model to prevent memory leak."""
    del model
    gc.collect()
EOF

# Update model loader
sed -i 's/return model/cleanup_model(old_model); return model/' src/model_loader.py

# Step 3: Test quickly
python -m pytest tests/test_model_loader.py -v

# Step 4: Commit and push ASAP
git add .
git commit -m "hotfix: fix memory leak in model loading

Critical fix for production memory leak.
Models were not being properly garbage collected.

Fixes #999"

git push -u origin hotfix/critical-memory-leak

# Step 5: Create urgent PR
gh pr create \
  --title "HOTFIX: Fix critical memory leak" \
  --body "🚨 CRITICAL FIX

## Problem
Production servers running out of memory due to model loading leak.

## Solution
Properly cleanup old models before loading new ones.

## Testing
- [x] Unit tests pass
- [ ] Deploy to staging for verification

## Merge Priority
URGENT - Memory usage growing unbounded in production."

# Step 6: After merge to main, also merge to develop
git checkout develop
git pull origin develop
git merge main
git push origin develop

# Step 7: Tag the hotfix
git checkout main
git pull origin main
git tag -a v1.0.1 -m "Hotfix: Memory leak fix"
git push origin v1.0.1
```

---

## Code Review Process

### As the Author

```bash
# Step 1: Prepare for review
# Ensure all tests pass
pytest

# Ensure code is formatted
black src/
flake8 src/

# Self-review changes
git diff main...feature/my-feature

# Step 2: Create detailed PR
gh pr create \
  --title "feat: Add model versioning" \
  --body "## Summary
Implements model versioning system.

## Changes
- Add version tracking to model metadata
- Update API to support version parameter
- Add migration script for existing models

## Testing
- [x] Unit tests (95% coverage)
- [x] Integration tests
- [x] Manual testing on staging

## Screenshots
![Model Versioning UI](url-to-image)

## Checklist
- [x] Tests added/updated
- [x] Documentation updated
- [x] No breaking changes
- [ ] Performance benchmarks (optional)

## Related Issues
Closes #123
Related to #456"

# Step 3: Address review comments
# Read feedback carefully
gh pr view 123 --comments

# Make requested changes
git add .
git commit -m "refactor: use semantic versioning format"
git push origin feature/my-feature

# Step 4: Respond to comments
# Comment on GitHub or via CLI
gh pr comment 123 --body "Updated to use semver. Good catch!"

# Step 5: Request re-review
gh pr review 123 --request-review
```

### As the Reviewer

```bash
# Step 1: Fetch PR
gh pr checkout 123

# Step 2: Review changes
git diff main...HEAD

# Step 3: Test locally
pytest
python -m src.main  # Manual testing

# Step 4: Leave review
gh pr review 123 \
  --comment \
  --body "Great work! Just a few suggestions:

1. Consider adding error handling in line 45
2. The test coverage could be improved for edge cases
3. Documentation looks good

Overall looks good to merge after addressing #1."

# Or approve
gh pr review 123 \
  --approve \
  --body "LGTM! Great implementation of model versioning."

# Or request changes
gh pr review 123 \
  --request-changes \
  --body "Please address the following before merging:

1. Add input validation
2. Fix failing test_edge_case
3. Update changelog"
```

---

## Handling Merge Conflicts

### Scenario: Conflict in Configuration File

```bash
# Step 1: Update your branch
git checkout feature/my-feature
git fetch origin
git merge origin/main

# Git reports conflict:
# CONFLICT (content): Merge conflict in config.yaml

# Step 2: View conflicted files
git status
# config.yaml (both modified)

# Step 3: Open and resolve
cat config.yaml
# <<<<<<< HEAD
# timeout: 30
# =======
# timeout: 60
# >>>>>>> origin/main

# Step 4: Decide on resolution
# Option A: Keep yours
# Option B: Keep theirs
# Option C: Combine both
# Option D: Use a different value

# Edit to resolve
cat > config.yaml << EOF
# Increased timeout for large models
timeout: 60
EOF

# Step 5: Mark as resolved
git add config.yaml

# Step 6: Complete merge
git commit -m "merge: resolve conflict in config.yaml

Kept the higher timeout value (60s) from main as it's
needed for large model loading."

# Step 7: Push
git push origin feature/my-feature
```

### Complex Conflict Resolution

```bash
# Use merge tool
git mergetool

# Or manually edit
vim config.yaml

# Check resolution
git diff --check  # Check for conflict markers

# Abort if needed
git merge --abort
```

---

## ML Experiment Workflow

### Scenario: Experiment with New Model Architecture

```bash
# Step 1: Create experiment branch
git checkout -b experiments/transformer-model

# Step 2: Track experiment with MLflow
cat > experiments/transformer_experiment.py << EOF
import mlflow

mlflow.set_experiment("transformer-architecture")

with mlflow.start_run(run_name="baseline"):
    # Train model
    model = train_transformer_model()

    # Log parameters
    mlflow.log_params({
        "architecture": "transformer",
        "layers": 12,
        "hidden_size": 768
    })

    # Log metrics
    mlflow.log_metrics({
        "accuracy": 0.92,
        "f1_score": 0.91
    })

    # Log model
    mlflow.sklearn.log_model(model, "model")
EOF

# Step 3: Commit experiment code
git add experiments/transformer_experiment.py
git commit -m "experiment: baseline transformer model

Initial experiment with transformer architecture.
Results tracked in MLflow run 'baseline'."

# Step 4: Iterate on experiment
# ... make changes ...

git commit -m "experiment: increase model capacity

Doubled hidden size to 1536.
MLflow run: 'increased-capacity'"

# Step 5: If experiment successful, prepare for merge
# Extract reusable code
git checkout -b feature/transformer-model experiments/transformer-model

# Refactor experiment into production code
cat > src/models/transformer.py << EOF
"""Production transformer model."""
class TransformerModel:
    def __init__(self, config):
        # Production-ready implementation
        pass
EOF

git add src/models/transformer.py
git commit -m "feat(models): add transformer model

Based on successful experiment in experiments/transformer-model.
Achieved 92% accuracy, 91% F1-score."

# Step 6: Create PR for production code
git push -u origin feature/transformer-model

# Step 7: Keep experiment branch for reference
git push -u origin experiments/transformer-model
# Don't merge experiment branch - keep for reproducibility
```

---

## Release Workflow

### Scenario: Prepare v2.0.0 Release

```bash
# Step 1: Create release branch
git checkout develop
git pull origin develop
git checkout -b release/2.0.0

# Step 2: Update version numbers
sed -i 's/__version__ = "1.9.0"/__version__ = "2.0.0"/' src/__init__.py

cat > VERSION << EOF
2.0.0
EOF

git add src/__init__.py VERSION
git commit -m "chore: bump version to 2.0.0"

# Step 3: Generate changelog
python solutions/commit_analyzer.py changelog 2.0.0 \
  --since v1.9.0 \
  --output CHANGELOG.md

git add CHANGELOG.md
git commit -m "docs: update changelog for v2.0.0"

# Step 4: Update documentation
cat >> README.md << EOF

## Version 2.0.0

### Major Changes
- New transformer model architecture
- Improved API performance
- Breaking changes in configuration format

### Migration Guide
See [MIGRATION.md](MIGRATION.md) for upgrade instructions.
EOF

git add README.md
git commit -m "docs: update README for v2.0.0"

# Step 5: Final testing
pytest
python -m mypy src/
python -m flake8 src/

# Step 6: Create PR to main
git push -u origin release/2.0.0

gh pr create \
  --base main \
  --title "Release v2.0.0" \
  --body "## Release v2.0.0

### Changes
- See CHANGELOG.md for full list

### Checklist
- [x] Version numbers updated
- [x] Changelog updated
- [x] Documentation updated
- [x] All tests passing
- [x] No breaking changes without migration guide

### Post-Merge Tasks
- [ ] Tag release
- [ ] Deploy to production
- [ ] Announce release"

# Step 7: After merge to main
git checkout main
git pull origin main

# Step 8: Tag release
git tag -a v2.0.0 -m "Release version 2.0.0

Major release with transformer model support.
See CHANGELOG.md for details."

git push origin v2.0.0

# Step 9: Merge back to develop
git checkout develop
git merge main
git push origin develop

# Step 10: Delete release branch
git branch -d release/2.0.0
git push origin --delete release/2.0.0

# Step 11: Create GitHub release
gh release create v2.0.0 \
  --title "Release v2.0.0" \
  --notes-file CHANGELOG.md
```

---

## Summary of Best Practices

1. **Always start from up-to-date main/develop**
2. **Use descriptive branch names**
3. **Commit early and often**
4. **Write meaningful commit messages**
5. **Test before committing**
6. **Keep PRs focused and small**
7. **Respond to reviews promptly**
8. **Clean up branches after merge**
9. **Tag releases**
10. **Document everything**

## Quick Reference Commands

```bash
# Start feature
git checkout -b feature/name

# Save work
git add .
git commit -m "feat: description"

# Update branch
git fetch origin
git rebase origin/main

# Create PR
gh pr create

# Clean up
git checkout main
git pull
git branch -d feature/name
```
