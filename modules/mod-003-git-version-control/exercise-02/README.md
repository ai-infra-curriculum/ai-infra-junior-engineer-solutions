# Exercise 02: Working with Commits and History - Solution

## Overview

This solution demonstrates advanced commit techniques, history navigation, and best practices for maintaining clean Git history in ML projects. It builds on Exercise 01 and shows how to write professional commit messages, search history, amend commits, and safely revert changes.

## Learning Objectives

This exercise covers:

- ✅ Writing commit messages with conventional format (feat:, fix:, docs:)
- ✅ Navigating commit history with `git log`
- ✅ Searching for specific changes in history
- ✅ Inspecting commits with `git show`
- ✅ Amending commits to fix mistakes
- ✅ Reverting changes without losing history
- ✅ Understanding revert vs reset
- ✅ Complete feature development workflow

## Solution Structure

```
exercise-02/
├── working-repo/          # Repository with demonstrated concepts
│   ├── src/              # Source with new features
│   │   ├── api/         # API with middleware
│   │   │   └── middleware.py
│   │   │   └── metrics.py
│   │   ├── utils/       # Utilities
│   │   │   ├── monitoring.py  # Performance monitoring
│   │   │   └── rate_limiter.py  # Rate limiting
│   │   └── preprocessing/
│   │       └── simple_preprocess.py  # Bug fix example
│   ├── docs/
│   │   └── api.md       # API documentation
│   ├── tests/
│   │   └── unit/        # Unit tests
│   └── configs/         # Enhanced configuration
├── scripts/
│   └── demonstrate_commits_history.sh  # Automated demonstration
├── docs/
│   └── QUICK_REFERENCE.md  # Command reference guide
└── README.md             # This file
```

## Key Concepts Demonstrated

### 1. Conventional Commit Messages

The repository demonstrates the conventional commits format:

```bash
# Feature
git commit -m "feat: add performance monitoring for inference tracking"

# Feature with scope
git commit -m "feat(api): implement request rate limiting"

# Bug fix
git commit -m "fix: convert images to RGB before preprocessing"

# Documentation
git commit -m "docs: add API documentation"

# Tests
git commit -m "test: add unit tests for metrics endpoint"
```

**Benefits:**
- Machine-readable format
- Automatic changelog generation
- Clear categorization
- Easy to search

### 2. Viewing History

Multiple ways to explore commit history:

```bash
# Compact view
git log --oneline

# With statistics
git log --stat --oneline

# With actual changes
git log -p

# Custom format
git log --pretty=format:"%h - %an, %ar : %s"

# Search commits
git log --grep="fix"
git log -S "RateLimiter"
```

### 3. Amending Commits

Examples of fixing commits:

```bash
# Fix typo in message
git commit --amend -m "Correct message"

# Add forgotten file
git add forgotten_file.py
git commit --amend --no-edit

# Improve both message and content
git add improved_file.py
git commit --amend -m "Better message"
```

⚠️ **Important:** Only amend commits that haven't been pushed!

### 4. Reverting Changes

Safe way to undo commits:

```bash
# Revert creates new commit
git revert HEAD

# Revert multiple commits
git revert HEAD HEAD~1 HEAD~2
```

**Comparison:**

| Operation | History | Safe for pushed commits? |
|-----------|---------|--------------------------|
| `git revert` | Preserved | ✅ Yes |
| `git reset` | Rewritten | ❌ No |

## Quick Start

### Run the Automated Demonstration

```bash
cd scripts/
./demonstrate_commits_history.sh
```

This script:
1. Copies the repository from Exercise 01
2. Demonstrates conventional commits
3. Shows history viewing techniques
4. Examples of amending commits
5. Demonstrates reverting changes
6. Shows complete feature workflow

The script pauses after each section for review.

### Explore the Repository Manually

```bash
cd working-repo/

# View commit history
git log --oneline --graph --all

# Search for specific commits
git log --grep="feat"

# See what changed in last commit
git show HEAD

# View file history
git log -- src/utils/monitoring.py

# Search for when code was added
git log -S "PerformanceMonitor"
```

## Commit History Analysis

### Commits Created

The working repository contains commits demonstrating:

1. **Initial commits** (from Exercise 01)
   ```
   37013eb Initial commit: Add .gitignore and .gitattributes
   b13ec84 Add project documentation
   d59b437 Add Python dependencies
   5bc4436 Add application configuration files
   ...
   ```

2. **Feature commits** (Exercise 02)
   ```
   607058b feat: add performance monitoring
   2a91dc8 feat(api): implement request rate limiting
   ```

3. **Bug fix commits**
   ```
   8221fb5 refactor: simplify preprocessing
   1d095df fix: convert images to RGB before preprocessing
   ```

4. **Documentation commits**
   ```
   2e46549 docs: add API documentation
   ```

### View Commit Statistics

```bash
cd working-repo/

# Count commits by type
git log --oneline | grep -c "^[a-f0-9]* feat"
git log --oneline | grep -c "^[a-f0-9]* fix"
git log --oneline | grep -c "^[a-f0-9]* docs"

# See recent commits with stats
git log --stat --oneline -5

# View commits graphically
git log --oneline --graph --all --decorate
```

## Key Features Added

### 1. Performance Monitoring (`src/utils/monitoring.py`)

**Purpose:** Track ML inference performance metrics

**Features:**
- Inference latency tracking (average, P95)
- Request counting per endpoint
- Error tracking by type
- Service uptime monitoring

**Commit:** `feat: add performance monitoring for inference tracking`

```python
from src.utils.monitoring import monitor

# Track metrics
monitor.metrics.record_inference(duration)
monitor.metrics.record_request("/predict")

# Get summary
summary = monitor.metrics.get_metrics_summary()
```

### 2. Rate Limiting (`src/utils/rate_limiter.py`)

**Purpose:** Prevent API abuse with token bucket algorithm

**Features:**
- Configurable rate limits
- Burst capacity
- Wait time calculation
- Request tracking

**Commit:** `feat(api): implement request rate limiting`

```python
from src.utils.rate_limiter import RateLimiter

limiter = RateLimiter(requests_per_minute=100, burst_size=20)

if limiter.allow_request():
    # Process request
    pass
else:
    # Return 429 Too Many Requests
    wait_time = limiter.get_wait_time()
```

### 3. Image Preprocessing Bug Fix

**Problem:** Pipeline failed with grayscale/RGBA images

**Solution:** Convert all images to RGB before processing

**Commit:** `fix: convert images to RGB before preprocessing`

```python
# Before (buggy)
image = Image.open(io.BytesIO(image_bytes))
return image

# After (fixed)
image = Image.open(io.BytesIO(image_bytes))
if image.mode != "RGB":
    image = image.convert("RGB")
return image
```

### 4. API Middleware

**Purpose:** Request logging and rate limiting

**Features:**
- Request/response logging with timing
- Rate limit enforcement
- 429 responses with retry-after

**Commit:** `feat(api): add request logging and rate limiting middleware`

## Command Reference

### Essential Commands Used

```bash
# Commit with message
git commit -m "type: subject"
git commit -m "type(scope): subject\n\nbody"

# Amend last commit
git commit --amend
git commit --amend -m "New message"
git commit --amend --no-edit

# View history
git log
git log --oneline
git log --stat
git log -p
git log --graph --all

# Search history
git log --grep="pattern"
git log -S "code"
git log -- file.py

# Show commits
git show HEAD
git show abc123
git show HEAD~1

# Revert commits
git revert HEAD
git revert abc123 --no-edit

# Compare
git diff HEAD~1 HEAD
git diff abc123 def456 -- file.py
```

See `docs/QUICK_REFERENCE.md` for comprehensive command reference.

## Best Practices Demonstrated

### 1. Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

### 2. Atomic Commits

✅ **Good:**
- One logical change per commit
- Can be reverted independently
- Clear, focused purpose

❌ **Bad:**
- Multiple unrelated changes
- "WIP" or "fix stuff" messages
- Mixing features and fixes

### 3. Commit Message Body

Include:
- **What** changed
- **Why** it changed
- **Context** if needed
- **Breaking changes** if any

Example:
```
fix: convert images to RGB before preprocessing

The preprocessing pipeline now converts all images to RGB mode
before applying transformations. This fixes inference failures
when processing:
- Grayscale images (mode 'L')
- RGBA images with alpha channel
- Other non-RGB image formats

Without this conversion, the model receives incorrect tensor
dimensions causing inference to fail.

Fixes issue where grayscale medical images caused 500 errors.
```

### 4. When to Amend

✅ **Amend when:**
- Fixing typo in message
- Adding forgotten file
- Improving last commit
- Changes are local only

❌ **Don't amend when:**
- Commit has been pushed
- Working on shared branch
- Others depend on the commit

### 5. Revert vs Reset

**Use `git revert`:**
- After pushing
- On shared branches
- Want to preserve history
- Need audit trail

**Use `git reset`:**
- Before pushing
- On private branches
- Want to rewrite history
- Cleaning up local work

## Testing the Solution

### 1. Verify Commit History

```bash
cd working-repo/

# Check commit count
git rev-list --count HEAD

# View all commits
git log --oneline

# Check for conventional format
git log --pretty=format:"%s" | head -10
```

### 2. Test Search Commands

```bash
# Find feature commits
git log --grep="^feat" --oneline

# Find when monitoring was added
git log -S "PerformanceMonitor" --oneline

# See all changes to config
git log -- configs/default.yaml
```

### 3. Inspect Specific Commits

```bash
# View latest commit
git show HEAD

# Compare last two commits
git diff HEAD~1 HEAD

# Show what changed in a file
git log -p -- src/utils/monitoring.py
```

## Common Workflows

### Workflow 1: Feature Development

```bash
# 1. Create feature
cat > src/feature.py << 'EOF'
def new_feature():
    pass
EOF

git add src/feature.py
git commit -m "feat: add new feature

Implement new feature for XYZ capability."

# 2. Add tests
cat > tests/test_feature.py << 'EOF'
def test_feature():
    assert True
EOF

git add tests/test_feature.py
git commit -m "test: add tests for new feature"

# 3. Update docs
cat >> docs/api.md << 'EOF'
## New Feature
Documentation here.
EOF

git add docs/api.md
git commit -m "docs: document new feature"
```

### Workflow 2: Bug Fix

```bash
# 1. Reproduce bug
# 2. Fix code
git add src/buggy_file.py
git commit -m "fix: resolve issue with XYZ

The bug occurred because...
This fix addresses it by...

Fixes #123"

# 3. Add regression test
git add tests/test_bug_fix.py
git commit -m "test: add regression test for bug #123"
```

### Workflow 3: Amend Forgotten File

```bash
# Initial commit
git add feature.py
git commit -m "feat: add feature"

# Oops, forgot the test
git add test_feature.py
git commit --amend --no-edit
```

## Troubleshooting

### Issue: "Cannot amend - commit was pushed"

**Problem:** Trying to amend a pushed commit

**Solution:** Don't amend! Create new commit instead:
```bash
# Instead of amending
git add fix.py
git commit -m "fix: address issue in previous commit"
```

### Issue: "Want to undo last commit"

**If not pushed:**
```bash
# Keep changes
git reset --soft HEAD~1

# Discard changes
git reset --hard HEAD~1
```

**If pushed:**
```bash
# Create revert commit
git revert HEAD
```

### Issue: "Accidentally committed secret"

**If not pushed:**
```bash
git reset --hard HEAD~1
# Remove secret from file
git add .gitignore
git commit -m "chore: add secrets to gitignore"
```

**If pushed:**
1. Rotate the secret immediately!
2. Use `git filter-branch` or BFG Repo-Cleaner
3. Force push (notify team)

## Verification Checklist

- [ ] Repository copied from Exercise 01
- [ ] Conventional commit messages used
- [ ] Features added with `feat:` commits
- [ ] Bug fixes with `fix:` commits
- [ ] Documentation with `docs:` commits
- [ ] Commits are atomic and focused
- [ ] Can search history effectively
- [ ] Can amend commits correctly
- [ ] Can revert commits safely
- [ ] Understand revert vs reset

## Related Documentation

- **QUICK_REFERENCE.md**: Comprehensive command reference
- **Exercise 01 README**: Repository initialization
- **Conventional Commits**: https://www.conventionalcommits.org/

## Next Steps

After completing this exercise:

1. **Practice** - Use conventional commits in your projects
2. **Explore** - Try interactive rebase (Exercise 07)
3. **Learn** - Study branching strategies (Exercise 03)
4. **Apply** - Use these techniques in team projects

## Summary

This exercise demonstrated:

- ✅ Professional commit message format
- ✅ Effective history navigation
- ✅ Safe commit modification techniques
- ✅ Complete feature workflow
- ✅ Best practices for Git history

**Key Takeaway:** Clean, searchable Git history is essential for team collaboration and long-term project maintenance.

## Support

For questions or issues:
- Review `docs/QUICK_REFERENCE.md`
- Check commit messages for examples
- Run the demonstration script
- Refer to Git official documentation
