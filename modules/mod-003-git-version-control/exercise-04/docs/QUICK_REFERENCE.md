# Git Merging and Conflict Resolution - Quick Reference

## Basic Merging

### Simple Merge

```bash
# Merge branch into current branch
git merge feature/branch-name

# Merge with custom message
git merge feature/branch-name -m "Merge feature X"

# Merge without editing message
git merge feature/branch-name --no-edit
```

### Merge Types

```bash
# Fast-forward merge (default if possible)
git merge feature/branch-name

# Force merge commit (no fast-forward)
git merge --no-ff feature/branch-name

# Squash all commits into one
git merge --squash feature/branch-name
# Then commit manually:
git commit -m "feat: combined feature"

# Merge and keep merge even if fast-forward
git merge --no-ff -m "Merge feature" feature/branch-name
```

## Viewing Changes Before Merge

### Compare Branches

```bash
# See commits that will be merged
git log master..feature/branch --oneline

# See file changes
git diff master..feature/branch

# See statistics
git diff master..feature/branch --stat

# See specific file
git diff master..feature/branch -- path/to/file

# Count commits to merge
git rev-list --count master..feature/branch
```

### Test Merge

```bash
# Test merge without committing
git merge --no-commit --no-ff feature/branch

# Review what would be merged
git status
git diff --cached

# If good, commit
git commit

# If bad, abort
git merge --abort
```

## Handling Merge Conflicts

### Identify Conflicts

```bash
# Check merge status
git status

# List conflicted files
git diff --name-only --diff-filter=U

# View conflicts
git diff

# View conflicts for specific file
git diff path/to/file
```

### Conflict Markers

Understanding conflict markers in files:

```
<<<<<<< HEAD
Your changes (current branch)
=======
Their changes (merging branch)
>>>>>>> feature/branch-name
```

### Resolution Options

```bash
# Option 1: Manual edit
vim path/to/conflicted-file
# Remove markers, combine changes
git add path/to/conflicted-file

# Option 2: Keep our version
git checkout --ours path/to/file
git add path/to/file

# Option 3: Keep their version
git checkout --theirs path/to/file
git add path/to/file

# Option 4: Use merge tool
git mergetool
# Opens configured merge tool
git add path/to/file

# After resolving all conflicts
git commit  # Completes merge
```

### Merge Tools

```bash
# Configure merge tool (one-time)
git config --global merge.tool vimdiff
git config --global merge.tool meld
git config --global merge.tool kdiff3
git config --global merge.tool vscode
git config --global merge.tool p4merge

# Set VS Code as merge tool
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait $MERGED'

# Use merge tool during conflict
git mergetool

# Open specific file in merge tool
git mergetool path/to/file

# Skip backup files
git config --global mergetool.keepBackup false

# Clean up backup files after merge
rm *.orig
```

## Aborting Merges

### Cancel Merge

```bash
# Abort merge in progress
git merge --abort

# Reset to before merge
git reset --hard HEAD

# Reset to specific commit
git reset --hard abc123
```

## Merge Strategies

### Strategy Options

```bash
# Recursive strategy (default)
git merge -s recursive feature/branch

# Ours strategy (keep our version for conflicts)
git merge -s ours feature/branch

# Theirs strategy (prefer their changes)
git merge -X theirs feature/branch

# Patience algorithm (better for large files)
git merge -X patience feature/branch

# Ignore whitespace changes
git merge -X ignore-space-change feature/branch
git merge -X ignore-all-space feature/branch
```

### Strategy Options Explanation

| Option | Description | Use Case |
|--------|-------------|----------|
| `-s recursive` | Default 3-way merge | Standard merging |
| `-s ours` | Always keep our version | Obsolete branches |
| `-X theirs` | Prefer their changes | Upstream updates |
| `-X patience` | Better diff algorithm | Large changes |
| `-X ignore-space-change` | Ignore whitespace | Reformatted code |
| `-X ignore-all-space` | Ignore all whitespace | Style changes |

## Squash Merging

### Basic Squash

```bash
# Squash merge (doesn't commit)
git merge --squash feature/branch

# View squashed changes
git diff --cached

# Create single commit
git commit -m "feat: combined feature from multiple commits

Details:
- Feature A
- Feature B
- Bug fix C"

# Original feature branch still exists
git branch -d feature/branch  # Delete if done
```

### When to Squash

✅ **Use squash when:**
- Feature has many WIP commits
- Want clean main branch history
- Individual commits not important
- Cleaning up experimental work
- Following "1 PR = 1 commit" policy

❌ **Don't squash when:**
- Commits have important history
- Want to preserve authorship
- Need to revert individual changes
- Commits are well-structured

## Viewing Merge History

### List Merges

```bash
# Show merge commits
git log --merges

# Show merge commits with graph
git log --merges --graph --oneline

# Show only merge commits
git log --oneline --merges -10

# Show merge with details
git log --merges -p

# First-parent only (main branch)
git log --first-parent
```

### Merge Commit Details

```bash
# Show merge commit
git show abc123

# Show both parents
git show abc123^1  # First parent
git show abc123^2  # Second parent

# Compare with parents
git diff abc123^1 abc123
git diff abc123^2 abc123

# Show merge commit files
git show abc123 --stat
git show abc123 --name-only
```

## Undoing Merges

### Before Push

```bash
# Undo merge (removes merge commit)
git reset --hard HEAD~1

# Undo merge (keep changes staged)
git reset --soft HEAD~1

# Undo merge (keep changes unstaged)
git reset --mixed HEAD~1
```

### After Push

```bash
# Revert merge commit
git revert -m 1 HEAD

# -m 1 means keep first parent (main branch)
# Creates new commit that undoes merge

# Revert older merge
git revert -m 1 abc123

# Revert without committing
git revert -m 1 abc123 --no-commit
git commit -m "Revert: feature X due to issues"
```

### Revert Explained

```bash
# Find merge commit
git log --merges --oneline

# Revert merge
git revert -m 1 <merge-commit-hash>

# -m 1: Keep first parent (usually main)
# -m 2: Keep second parent (usually feature)

# Most common: -m 1 (keep main branch)
```

## Cherry-Picking

### Basic Cherry-Pick

```bash
# Apply specific commit to current branch
git cherry-pick abc123

# Apply multiple commits
git cherry-pick abc123 def456

# Apply range of commits
git cherry-pick abc123..def456

# Apply without committing
git cherry-pick -n abc123
git cherry-pick --no-commit abc123

# Edit commit message
git cherry-pick -e abc123
git cherry-pick --edit abc123
```

### Cherry-Pick During Conflicts

```bash
# Start cherry-pick
git cherry-pick abc123

# If conflicts
git status
# Resolve conflicts
git add <resolved-files>
git cherry-pick --continue

# Or abort
git cherry-pick --abort

# Or skip this commit
git cherry-pick --skip
```

## Merge vs Rebase

### Merge

```bash
# Merge creates merge commit
git switch main
git merge feature/branch

# Result: branched history
#   A---B---C-------M (main)
#        \         /
#         D---E---F (feature)
```

### Rebase

```bash
# Rebase rewrites history
git switch feature/branch
git rebase main

# Result: linear history
#   A---B---C (main)
#            \
#             D'---E'---F' (feature)
```

### When to Use Each

| Situation | Use | Reason |
|-----------|-----|--------|
| Public branch | Merge | Preserves history |
| Private branch | Rebase | Clean history |
| Feature complete | Merge | Track integration |
| Feature WIP | Rebase | Update with main |
| Team project | Merge | Safe for others |
| Solo project | Either | Your choice |

## Octopus Merge

### Merge Multiple Branches

```bash
# Merge multiple branches at once
git merge feature/a feature/b feature/c

# Creates single merge commit with multiple parents

# Only works if no conflicts
# If conflicts, merge one at a time
```

## Pre-Merge Validation

### Checks Before Merging

```bash
# 1. Update main branch
git switch main
git pull origin main

# 2. Check feature is ready
git switch feature/branch
git log main..HEAD --oneline  # Review commits
pytest tests/  # Run tests

# 3. Update feature with main
git rebase main  # Or: git merge main
# Resolve any conflicts

# 4. Test merge locally
git switch main
git merge --no-commit --no-ff feature/branch
pytest tests/
git merge --abort  # If just testing

# 5. Actually merge
git merge feature/branch
```

## Post-Merge Validation

### After Merging

```bash
# 1. Check for conflict markers
grep -r "<<<<<<< HEAD" src/
grep -r ">>>>>>>" src/

# 2. Check syntax
python -m py_compile src/**/*.py

# 3. Run tests
pytest tests/ -v

# 4. Check merge commit
git show HEAD
git log --graph --oneline -5

# 5. Verify files
git diff HEAD~1 HEAD --stat
```

## Merge Commit Messages

### Good Merge Messages

```bash
# Template
git merge feature/branch -m "Merge feature: <feature-name>

<Why this feature>

Changes:
- <change 1>
- <change 2>

Testing:
- <test 1>
- <test 2>

Closes #123"
```

### Examples

```bash
# Example 1: Feature merge
git merge --no-ff feature/batch-inference -m "Merge feature: batch inference

Add batch prediction capability for processing multiple images efficiently.

Changes:
- BatchProcessor class with configurable batch size
- Async batch processing support
- /predict/batch API endpoint
- Comprehensive error handling

Testing:
- Unit tests for BatchProcessor
- Integration tests for API endpoint
- Load testing with 100+ concurrent batches

Closes #456"

# Example 2: Bug fix merge
git merge --no-ff fix/memory-leak -m "Merge fix: memory leak in image processing

Resolve memory leak causing OOM errors in production.

Changes:
- Explicit cleanup of PIL Image objects
- Resource manager for image lifecycle
- Memory usage monitoring

Testing:
- Memory profiling tests
- Load tests with 1000+ images
- 24-hour stress test

Fixes #789"

# Example 3: Squash merge
git merge --squash feature/logging-improvements
git commit -m "feat: improve logging system

Consolidated logging improvements from feature branch:
- Timestamp formatting enhancements
- Log rotation configuration (10MB, 5 files)
- Structured field definitions
- Performance optimization

All changes tested individually and as integrated system.

Squashed 5 commits from feature/logging-improvements"
```

## Conflict Resolution Patterns

### Pattern 1: Config Files

```yaml
# Conflict in YAML
<<<<<<< HEAD
timeout: 30
max_connections: 100
=======
timeout: 60
enable_cache: true
>>>>>>> feature/branch

# Resolution: Combine both
timeout: 60  # Use updated value
max_connections: 100  # Keep from main
enable_cache: true  # Add new feature
```

### Pattern 2: Python Imports

```python
# Conflict in imports
<<<<<<< HEAD
import logging
from typing import Dict, List
=======
import logging
from typing import Optional
import asyncio
>>>>>>> feature/branch

# Resolution: Combine all
import logging
import asyncio
from typing import Dict, List, Optional
```

### Pattern 3: Function Definitions

```python
# Conflict in function signature
<<<<<<< HEAD
def predict(image, model):
    """Predict with validation."""
    validate_image(image)
    return model.predict(image)
=======
def predict(image, model):
    """Predict with metrics."""
    start = time.time()
    result = model.predict(image)
    monitor.record(time.time() - start)
    return result
>>>>>>> feature/branch

# Resolution: Combine both features
def predict(image, model):
    """Predict with validation and metrics."""
    # Validation from HEAD
    validate_image(image)

    # Metrics from feature
    start = time.time()
    result = model.predict(image)
    monitor.record(time.time() - start)

    return result
```

### Pattern 4: Documentation

```markdown
# Conflict in README
<<<<<<< HEAD
## Features

- Image classification
- Batch processing
=======
## Features

- Image classification
- Model caching
>>>>>>> feature/branch

# Resolution: Combine lists
## Features

- Image classification
- Batch processing
- Model caching
```

## Troubleshooting Commands

### Merge Status

```bash
# Check if merge in progress
git status

# See what's being merged
cat .git/MERGE_HEAD

# View merge message
cat .git/MERGE_MSG

# List conflicted files
git diff --name-only --diff-filter=U

# Show merge base
git merge-base main feature/branch
```

### Conflict Analysis

```bash
# See both versions
git show :1:path/to/file  # Common ancestor
git show :2:path/to/file  # Ours (current)
git show :3:path/to/file  # Theirs (merging)

# Diff between versions
git diff :2:path/to/file :3:path/to/file

# View conflict context
git log --merge path/to/file
```

## Configuration

### Merge Configuration

```bash
# Default merge strategy
git config --global merge.defaultstrategy recursive

# Always create merge commit
git config --global merge.ff false

# Fast-forward only
git config --global merge.ff only

# Configure merge tool
git config --global merge.tool vimdiff
git config --global mergetool.prompt false
git config --global mergetool.keepBackup false

# Auto-resolve simple conflicts
git config --global merge.conflictstyle diff3
```

### Diff3 Conflict Style

```bash
# Enable diff3 style
git config --global merge.conflictstyle diff3

# Shows:
<<<<<<< HEAD
your changes
||||||| base
original
=======
their changes
>>>>>>> branch

# Helps understand what changed on both sides
```

## Best Practices

### Before Merge Checklist

```bash
# 1. Update target branch
git switch main && git pull

# 2. Review changes
git log main..feature/branch --oneline
git diff main..feature/branch --stat

# 3. Test feature branch
git switch feature/branch
pytest tests/

# 4. Update feature with main
git rebase main  # Or merge main

# 5. Test again
pytest tests/

# 6. Merge
git switch main
git merge feature/branch
```

### During Merge Checklist

```bash
# 1. Understand both changes
git log --merge

# 2. Test each side
git checkout --ours file && pytest
git checkout --theirs file && pytest

# 3. Combine carefully
# Edit file to integrate both

# 4. Test merged version
pytest tests/

# 5. Verify resolution
grep -r "<<<" src/  # No markers

# 6. Commit
git add resolved-files
git commit
```

### After Merge Checklist

```bash
# 1. Validate merge
pytest tests/
grep -r "<<<" src/

# 2. Review merge commit
git show HEAD
git log --graph --oneline -5

# 3. Push changes
git push origin main

# 4. Clean up
git branch -d feature/branch
git push origin --delete feature/branch

# 5. Notify team
# (via PR, chat, email, etc.)
```

## Common Workflows

### Workflow 1: Simple Feature Merge

```bash
git switch main
git merge feature/simple-feature
# Fast-forward or automatic merge
pytest tests/
git push
git branch -d feature/simple-feature
```

### Workflow 2: Conflicted Merge

```bash
git switch main
git merge feature/complex-feature
# CONFLICT!
git status
# Edit conflicted files
git add .
git commit
pytest tests/
git push
```

### Workflow 3: Squash Merge

```bash
git switch main
git merge --squash feature/many-commits
git commit -m "feat: consolidated feature"
pytest tests/
git push
git branch -d feature/many-commits
```

### Workflow 4: No-FF Merge

```bash
git switch main
git merge --no-ff feature/important-feature -m "Merge: feature X

Important feature that needs clear history."
pytest tests/
git push
```

## Aliases

Add to `.gitconfig`:

```ini
[alias]
    # Merge shortcuts
    m = merge
    mnf = merge --no-ff
    msq = merge --squash
    ma = merge --abort

    # Merge info
    merged = branch --merged
    unmerged = branch --no-merged
    merges = log --merges --oneline

    # Conflict resolution
    conflicts = diff --name-only --diff-filter=U
    ours = checkout --ours
    theirs = checkout --theirs
```

Usage:
```bash
git m feature/branch
git mnf feature/branch
git msq feature/branch
git conflicts
```

## Resources

- [Git Merge Documentation](https://git-scm.com/docs/git-merge)
- [Pro Git: Basic Merging](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
- [Atlassian: Merge vs Rebase](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
- [GitHub: Resolving Conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts)
