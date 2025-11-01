# Exercise 04: Merging and Conflict Resolution - Solution

## Overview

This solution demonstrates Git merging strategies and conflict resolution techniques for ML infrastructure projects. It covers fast-forward merges, three-way merges, handling conflicts in code and configuration files, and various merge strategies essential for team collaboration.

## Learning Objectives

This exercise covers:

- ✅ Performing fast-forward merges
- ✅ Executing three-way merges with merge commits
- ✅ Identifying and resolving merge conflicts
- ✅ Handling conflicts in Python code, YAML configs, and documentation
- ✅ Using different merge strategies (merge, squash, no-ff)
- ✅ Aborting and retrying problematic merges
- ✅ Choosing specific versions during conflicts
- ✅ Validating merges before and after integration
- ✅ Complete merge workflow for ML projects

## Solution Structure

```
exercise-04/
├── working-repo/          # Repository with merge scenarios
│   ├── master            # Main branch
│   ├── feature/health-check-enhancements  # Fast-forward merge
│   ├── feature/add-version-info           # No-FF merge
│   ├── feature/config-feature-toggles     # Config conflict 1
│   ├── feature/config-performance         # Config conflict 2
│   ├── feature/request-validation         # Code conflict 1
│   ├── feature/api-metrics-tracking       # Code conflict 2
│   └── feature/logging-improvements       # Squash merge (3 commits)
├── scripts/
│   ├── setup_repository.sh           # Create merge scenarios
│   ├── demonstrate_merging.sh        # Interactive demonstration
│   └── post_merge_check.sh          # Merge validation
├── docs/
│   └── QUICK_REFERENCE.md            # Merging command reference
└── README.md                          # This file
```

## Key Concepts

### 1. Types of Merges

#### Fast-Forward Merge

Occurs when the target branch hasn't diverged:

```
Before:           After fast-forward:
A---B---C (main)  A---B---C---D---E (main)
         \
          D---E (feature)
```

**Characteristics:**
- No merge commit created
- Linear history preserved
- Simplest type of merge
- Only possible when no divergence

**Example:**
```bash
git switch main
git merge feature/health-check-enhancements
# Output: Fast-forward
```

#### Three-Way Merge

Occurs when branches have diverged:

```
Before:                  After three-way merge:
A---B---C---F (main)     A---B---C---F---M (main)
     \                            \     /
      D---E (feature)              D---E (feature)
```

**Characteristics:**
- Creates merge commit with two parents
- Preserves both development paths
- Shows integration point
- Required when branches diverged

**Example:**
```bash
git switch main
git merge feature/batch-inference
# Creates merge commit M
```

#### No Fast-Forward Merge (--no-ff)

Forces merge commit even when fast-forward is possible:

```
Before:           After --no-ff:
A---B---C (main)  A---B---C-------M (main)
         \                 \     /
          D---E (feat)      D---E (feat)
```

**When to use:**
- Want to preserve feature branch history
- Track when features were integrated
- Maintain clear feature boundaries
- Required by team policy

**Example:**
```bash
git merge --no-ff feature/add-version-info
```

#### Squash Merge

Combines all commits into one:

```
Before:                 After squash:
A---B---C (main)        A---B---C---D' (main)
         \
          D1---D2---D3 (feature)

D' contains all changes from D1, D2, D3
```

**When to use:**
- Feature has many WIP commits
- Want clean history on main
- Individual commits not important
- Cleaning up experimental work

**Example:**
```bash
git merge --squash feature/logging-improvements
git commit -m "feat: improve logging system"
```

### 2. Merge Conflicts

Conflicts occur when:
- Same lines modified in both branches
- File deleted in one branch, modified in another
- Binary files changed differently
- Structural conflicts in configuration

**Conflict Markers:**
```
<<<<<<< HEAD
Your changes (current branch)
=======
Their changes (merging branch)
>>>>>>> feature/branch-name
```

### 3. Conflict Resolution Strategies

#### Manual Resolution

Edit file to combine both changes:

```yaml
# Original (main):
timeout: 30

# Feature branch:
timeout: 60

# Resolution - keep newer value:
timeout: 60
```

#### Choose One Side

Keep one version entirely:

```bash
# Keep ours (current branch)
git checkout --ours path/to/file
git add path/to/file

# Keep theirs (merging branch)
git checkout --theirs path/to/file
git add path/to/file
```

#### Combine Both Changes

Integrate changes from both branches:

```python
# Main branch:
def predict(image):
    return model.predict(image)

# Feature branch:
def predict(image, validate=True):
    return model.predict(image)

# Resolution - combine features:
def predict(image, validate=True):
    if validate:
        validate_image(image)
    return model.predict(image)
```

## Quick Start

### Setup Repository

```bash
cd scripts/
./setup_repository.sh
```

This creates a repository with:
- 10 branches with various merge scenarios
- Realistic conflicts in code and configs
- Examples of all merge types

### Explore Merge Scenarios

```bash
cd working-repo/

# View all branches
git branch -a

# View branch graph
git log --oneline --graph --all -15

# Check differences
git diff master..feature/health-check-enhancements
```

## Merge Scenarios

### Scenario 1: Fast-Forward Merge

**Branch:** `feature/health-check-enhancements`

**What it adds:**
- `src/api/health.py` - Enhanced health checking
- `tests/unit/test_health.py` - Health check tests

**Merge:**
```bash
cd working-repo/
git switch master

# Check if fast-forward possible
git log --oneline --graph master feature/health-check-enhancements

# Perform merge
git merge feature/health-check-enhancements

# Verify - should say "Fast-forward"
git log --oneline -5
```

**Expected output:**
```
Updating 8729ccc..ef06b63
Fast-forward
 src/api/health.py         | 57 ++++++++++++++++++++++++++++++++++++
 tests/unit/test_health.py | 21 +++++++++++++
 2 files changed, 78 insertions(+)
```

**Key points:**
- No merge commit created
- HEAD simply moves forward
- Linear history maintained

### Scenario 2: No Fast-Forward Merge

**Branch:** `feature/add-version-info`

**What it adds:**
- `src/version.py` - Version tracking module

**Merge:**
```bash
git switch master

# Force merge commit with --no-ff
git merge --no-ff feature/add-version-info -m "Merge feature: version information

Adds version tracking module for API versioning.
Provides version info endpoints and tracking."

# View merge commit
git log --oneline --graph -5
git show HEAD
```

**Expected output:**
```
*   abc123 Merge feature: version information
|\
| * def456 feat: add version information module
|/
*   789abc (previous commits)
```

**Why use --no-ff:**
- Preserves feature branch history in graph
- Shows when feature was integrated
- Makes it easy to revert entire feature
- Better for code review tracking

### Scenario 3: Config File Conflict

**Branches:**
- `feature/config-feature-toggles` - Adds feature flags
- `feature/config-performance` - Adds performance settings

**Both modify:** `configs/default.yaml`

**Steps to reproduce and resolve:**

```bash
git switch master

# Merge first feature (no conflict)
git merge feature/config-feature-toggles --no-edit

# Try to merge second feature - CONFLICT!
git merge feature/config-performance
```

**Output:**
```
Auto-merging configs/default.yaml
CONFLICT (content): Merge conflict in configs/default.yaml
Automatic merge failed; fix conflicts and then commit the result.
```

**View conflict:**
```bash
git status

# Shows:
# Unmerged paths:
#   both modified:   configs/default.yaml

cat configs/default.yaml
```

**Conflict looks like:**
```yaml
# ... existing config ...

<<<<<<< HEAD
# Feature Toggles
features:
  batch_inference: true
  caching: true
  metrics_export: true
  health_checks: true
=======
# Performance Settings
performance:
  max_workers: 4
  timeout_seconds: 30
  enable_profiling: false
  queue_size: 1000
>>>>>>> feature/config-performance
```

**Resolution:**

```bash
# Edit configs/default.yaml to keep BOTH sections:
cat >> configs/default.yaml << 'EOF'

# Feature Toggles
features:
  batch_inference: true
  caching: true
  metrics_export: true
  health_checks: true

# Performance Settings
performance:
  max_workers: 4
  timeout_seconds: 30
  enable_profiling: false
  queue_size: 1000
EOF

# Stage resolved file
git add configs/default.yaml

# Verify resolution
git status
# Should show: All conflicts fixed but you are still merging.

# Complete merge
git commit -m "Merge feature: performance configuration

Resolved conflicts by including both feature toggles
and performance settings in configuration.

Both sets of configuration options are now available."

# Verify
git log --oneline --graph -8
```

### Scenario 4: Python Code Conflict

**Branches:**
- `feature/request-validation` - Adds validators.py
- `feature/api-metrics-tracking` - Adds metrics.py

**Conflict:** Both may need integration into API endpoints

**Steps:**

```bash
git switch master

# Merge validation feature
git merge feature/request-validation --no-edit

# Merge metrics feature
git merge feature/api-metrics-tracking --no-edit
```

**If conflicts occur in shared files:**

```bash
# Check conflicted files
git status

# View conflict
git diff

# Edit to combine both features
# Stage resolved files
git add <resolved-files>

# Complete merge
git commit -m "Merge feature: API metrics tracking

Integrated metrics tracking with request validation.
Both features now work together."
```

**Resolution strategy for code conflicts:**

1. **Understand both changes**
   ```bash
   # View what each branch changed
   git log master..feature/request-validation --oneline
   git log master..feature/api-metrics-tracking --oneline
   ```

2. **Test individual features**
   ```bash
   # Test validation
   git switch feature/request-validation
   # Run tests

   # Test metrics
   git switch feature/api-metrics-tracking
   # Run tests
   ```

3. **Combine carefully**
   - Import statements from both
   - Function parameters from both
   - Logic from both in correct order
   - Error handling from both

4. **Test merged result**
   ```bash
   # After resolution
   pytest tests/
   python -m py_compile src/**/*.py
   ```

### Scenario 5: Squash Merge

**Branch:** `feature/logging-improvements` (3 commits)

**Why squash:**
- Has 3 small incremental commits
- Want single clean commit on main
- Individual commits are WIP

**View commits:**
```bash
git log feature/logging-improvements --oneline -5

# Shows:
# 3ba68a0 logging: add structured field definitions
# 89acc23 logging: add log rotation configuration
# 7007bed logging: add timestamp formatting function
```

**Squash merge:**
```bash
git switch master

# Squash merge (doesn't commit)
git merge --squash feature/logging-improvements

# Check status
git status
# Shows: All changes staged, ready to commit

# View combined changes
git diff --cached

# Create single commit
git commit -m "feat: improve logging system

Consolidated logging improvements:
- Timestamp formatting enhancements
- Log rotation configuration
- Structured field definitions

All improvements tested and integrated as single feature."

# Verify - only ONE commit on main
git log --oneline -3
```

**Result:**
- Main branch has 1 new commit
- Feature branch still has 3 commits
- Clean history on main

## Common Workflows

### Workflow 1: Simple Feature Merge

```bash
# 1. Update main
git switch main
git pull  # If working with remote

# 2. View what will be merged
git log master..feature/my-feature --oneline
git diff master..feature/my-feature --stat

# 3. Merge
git merge feature/my-feature

# 4. If conflicts, resolve them
git status
# Edit conflicted files
git add <resolved-files>
git commit

# 5. Test
pytest tests/

# 6. Push (if using remote)
git push origin main

# 7. Clean up
git branch -d feature/my-feature
```

### Workflow 2: Pre-Merge Validation

```bash
# Test merge without committing
git switch main
git merge --no-commit --no-ff feature/risky-feature

# Review what would be merged
git status
git diff --cached

# Run tests on merged code
pytest tests/

# If good, commit
git commit -m "Merge feature: risky feature

Tested and validated before final merge."

# If bad, abort
git merge --abort
```

### Workflow 3: Conflict Resolution

```bash
# Start merge
git merge feature/conflicting-branch

# CONFLICT!
# View conflicted files
git status
git diff

# For each conflicted file:

# Option 1: Manual edit
vim path/to/conflicted-file
# Remove markers, combine changes
git add path/to/conflicted-file

# Option 2: Keep one side
git checkout --ours path/to/file     # Keep main
git checkout --theirs path/to/file   # Keep feature
git add path/to/file

# Option 3: Use merge tool
git mergetool
# Interactive resolution
git add path/to/file

# After all conflicts resolved
git status  # Should say "all conflicts fixed"

# Complete merge
git commit  # Uses default message or customize

# Verify
git log --oneline --graph -5
```

### Workflow 4: Abort and Retry

```bash
# Start merge
git merge feature/complex-feature

# Too many conflicts, need to prepare better
git merge --abort

# Prepare feature branch
git switch feature/complex-feature
git rebase main  # Update feature with latest main

# Retry merge
git switch main
git merge feature/complex-feature
# Should have fewer conflicts now
```

## Merge Strategies Comparison

| Strategy | Command | History | Use Case |
|----------|---------|---------|----------|
| **Fast-Forward** | `git merge` | Linear | Simple, no divergence |
| **Three-Way** | `git merge` | Branched | Diverged branches |
| **No-FF** | `git merge --no-ff` | Always branched | Preserve feature history |
| **Squash** | `git merge --squash` | Linear, single commit | Clean up WIP commits |
| **Rebase** | `git rebase` | Linear | Rewrite history (advanced) |

## Conflict Resolution Techniques

### Technique 1: Accept One Side

```bash
# During conflict, choose entire file version

# Keep current branch (ours)
git checkout --ours conflicted-file.py
git add conflicted-file.py

# Keep merging branch (theirs)
git checkout --theirs conflicted-file.py
git add conflicted-file.py

# Complete merge
git commit
```

### Technique 2: Manual Edit

```bash
# Edit file to remove markers and combine changes

# View conflict
cat conflicted-file.py

# Shows:
# <<<<<<< HEAD
# your code
# =======
# their code
# >>>>>>> feature/branch

# Edit to combine
vim conflicted-file.py
# Remove markers
# Combine both changes intelligently

# Stage and commit
git add conflicted-file.py
git commit
```

### Technique 3: Use Merge Tool

```bash
# Configure merge tool (one-time)
git config --global merge.tool vimdiff
# Or: meld, kdiff3, vscode, etc.

# During conflict
git mergetool

# Opens visual diff tool
# Shows: LOCAL | BASE | REMOTE | MERGED
# Make changes in MERGED pane
# Save and exit

# Clean up backup files
rm *.orig

# Commit
git commit
```

### Technique 4: Cherry-Pick Specific Changes

```bash
# During complex conflict, pick specific commits

# Abort current merge
git merge --abort

# Cherry-pick specific commits instead
git log feature/branch --oneline
git cherry-pick abc123  # Pick commit we want
git cherry-pick def456  # Pick another

# Or create new solution
# Manually apply needed changes
```

## Validation Scripts

### Post-Merge Validation

Create `scripts/post_merge_check.sh`:

```bash
#!/bin/bash
# Post-merge validation script

echo "Running post-merge checks..."

# 1. Check for unresolved conflict markers
echo "Checking for conflict markers..."
if grep -r "<<<<<<< HEAD" src/ tests/ configs/ 2>/dev/null; then
    echo "ERROR: Unresolved conflict markers found!"
    exit 1
fi

if grep -r ">>>>>>>" src/ tests/ configs/ 2>/dev/null; then
    echo "ERROR: Unresolved conflict markers found!"
    exit 1
fi

echo "✓ No conflict markers found"

# 2. Check Python syntax
echo "Checking Python syntax..."
python_files=$(find src/ tests/ -name "*.py" 2>/dev/null)
for file in $python_files; do
    python -m py_compile "$file" || {
        echo "ERROR: Syntax error in $file"
        exit 1
    }
done
echo "✓ Python syntax valid"

# 3. Check YAML syntax
echo "Checking YAML syntax..."
if command -v python3 &> /dev/null; then
    python3 -c "
import yaml
import sys
import glob

for file in glob.glob('configs/**/*.yaml', recursive=True):
    try:
        with open(file) as f:
            yaml.safe_load(f)
    except Exception as e:
        print(f'ERROR: Invalid YAML in {file}: {e}')
        sys.exit(1)
print('✓ YAML syntax valid')
"
fi

# 4. Run tests
echo "Running tests..."
if [ -d "tests/" ]; then
    if command -v pytest &> /dev/null; then
        pytest tests/ -v || {
            echo "ERROR: Tests failed"
            exit 1
        }
        echo "✓ Tests passed"
    else
        echo "⚠ pytest not installed, skipping tests"
    fi
fi

echo ""
echo "✓ All post-merge checks passed!"
```

**Usage:**
```bash
chmod +x scripts/post_merge_check.sh

# After each merge
./scripts/post_merge_check.sh
```

## Best Practices

### 1. Before Merging

✅ **Do:**
- Update target branch (git pull)
- Review changes (git diff)
- Run tests on feature branch
- Check for conflicts (git merge --no-commit --no-ff)
- Communicate with team

❌ **Don't:**
- Merge without reviewing
- Merge untested code
- Merge to wrong branch
- Force merge with errors

### 2. During Conflicts

✅ **Do:**
- Understand both changes
- Test each side individually
- Combine changes carefully
- Remove all conflict markers
- Test merged result

❌ **Don't:**
- Blindly choose one side
- Leave conflict markers
- Commit without testing
- Rush resolution

### 3. After Merging

✅ **Do:**
- Run full test suite
- Check for conflict markers
- Verify application works
- Push to remote
- Clean up feature branches

❌ **Don't:**
- Skip validation
- Assume merge worked
- Leave merged branches
- Forget to push

### 4. Merge Messages

✅ **Good:**
```
Merge feature: batch inference processing

Integrates batch prediction capability:
- Process multiple images in single request
- Configurable batch size (max 32)
- Concurrent preprocessing
- Comprehensive error handling

Resolves #123
```

❌ **Bad:**
```
Merge branch 'feature'
```

## Troubleshooting

### Issue: "Cannot merge, uncommitted changes"

**Problem:** Have uncommitted changes in working directory

**Solution:**
```bash
# Option 1: Stash changes
git stash
git merge feature/branch
git stash pop

# Option 2: Commit changes
git commit -am "WIP: in-progress work"
git merge feature/branch

# Option 3: Discard changes (careful!)
git restore .
git merge feature/branch
```

### Issue: "Merge created unexpected results"

**Problem:** Merged code doesn't work as expected

**Solution:**
```bash
# Undo last merge
git reset --hard HEAD~1

# Or revert merge commit
git revert -m 1 HEAD

# Investigate and retry
git diff master..feature/branch
```

### Issue: "Too many conflicts"

**Problem:** Dozens of conflicts, overwhelming

**Solution:**
```bash
# Abort current merge
git merge --abort

# Update feature branch first
git switch feature/branch
git rebase main  # Or: git merge main

# Resolve conflicts incrementally during rebase
# Then retry merge
git switch main
git merge feature/branch
```

### Issue: "Lost track during merge"

**Problem:** Don't remember what's being merged

**Solution:**
```bash
# Check merge status
git status

# See what branch is being merged
cat .git/MERGE_HEAD

# View merge message
cat .git/MERGE_MSG

# See conflicted files
git diff --name-only --diff-filter=U

# If too confused, abort
git merge --abort
```

### Issue: "Accidentally committed with conflicts"

**Problem:** Committed without resolving all conflicts

**Solution:**
```bash
# Check for conflict markers
grep -r "<<<<<<< HEAD" src/

# Undo commit
git reset --soft HEAD~1

# Fix conflicts properly
# Edit files, remove markers
git add <fixed-files>

# Commit again
git commit
```

## Verification Checklist

After completing merges:

- [ ] All features merged to main
- [ ] No conflict markers in code
- [ ] All tests passing
- [ ] Python syntax valid
- [ ] YAML configs valid
- [ ] Application runs correctly
- [ ] Merge commits have good messages
- [ ] Feature branches cleaned up
- [ ] Changes pushed to remote
- [ ] Team notified of integration

## Related Documentation

- **QUICK_REFERENCE.md**: Comprehensive merging commands
- **Exercise 03 README**: Branching basics
- **Exercise 05**: Collaboration workflows (next)

## Summary

This exercise demonstrated:

- ✅ Fast-forward vs. three-way merges
- ✅ Creating merge commits with --no-ff
- ✅ Resolving config file conflicts
- ✅ Resolving Python code conflicts
- ✅ Squash merging multiple commits
- ✅ Aborting problematic merges
- ✅ Using merge tools and strategies
- ✅ Validating merges
- ✅ Complete merge workflows

**Key Takeaway:** Merging integrates parallel development. Understanding merge types, resolving conflicts carefully, and validating results ensures successful integration of ML infrastructure features.

## Next Steps

After completing this exercise:

1. **Practice** - Merge features in your projects
2. **Experiment** - Try different merge strategies
3. **Learn** - Study remote collaboration (Exercise 05)
4. **Apply** - Use these techniques in team projects

---

**Exercise 04 Complete** ✓

Ready for Exercise 05: Collaboration and Remote Workflows
