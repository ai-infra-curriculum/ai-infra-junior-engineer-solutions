# Exercise 07: Advanced Git Techniques - Implementation Guide

## Overview

Master advanced Git techniques for complex ML infrastructure projects including interactive rebasing, custom hooks, cherry-picking, bisecting, submodules, and recovery strategies.

**Estimated Time**: 120-150 minutes
**Difficulty**: Advanced
**Prerequisites**: Exercises 01-06

## What You'll Learn

- ✅ Interactive rebase for clean history
- ✅ Custom Git hooks for automation
- ✅ Cherry-picking specific commits
- ✅ Advanced stashing workflows
- ✅ Git bisect for bug hunting
- ✅ Reflog for time travel and recovery
- ✅ Git submodules for dependencies
- ✅ Worktrees for parallel development
- ✅ Advanced merge strategies
- ✅ Troubleshooting and recovery

---

## Part 1: Interactive Rebase

### Step 1.1: Clean Up Messy Commit History

**Scenario**: You made many small WIP commits during development. Clean them up before merging.

```bash
# Create project
mkdir ml-platform-advanced
cd ml-platform-advanced
git init --initial-branch=main

# Initial structure
mkdir -p src/{api,models,pipeline} tests scripts
echo "# ML Platform" > README.md
git add .
git commit -m "Initial commit"

# Create messy feature branch
git switch -c feature/model-serving

# Simulate messy development with many small commits
echo "class ModelServer:" > src/api/server.py
git add src/api/server.py
git commit -m "wip"

echo "    def __init__(self):" >> src/api/server.py
git add src/api/server.py
git commit -m "add init"

echo "        self.model = None" >> src/api/server.py
git add src/api/server.py
git commit -m "add model attr"

echo "    def load_model(self, path):" >> src/api/server.py
git add src/api/server.py
git commit -m "load method"

echo "        import pickle" >> src/api/server.py
echo "        self.model = pickle.load(open(path, 'rb'))" >> src/api/server.py
git add src/api/server.py
git commit -m "implement load"

echo "    def predict(self, data):" >> src/api/server.py
echo "        return self.model.predict(data)" >> src/api/server.py
git add src/api/server.py
git commit -m "predict method"

# View messy history
git log --oneline -6
# Output:
# abc123 predict method
# def456 implement load
# ghi789 load method
# jkl012 add model attr
# mno345 add init
# pqr678 wip

# Clean it up with interactive rebase
git rebase -i HEAD~6

# Editor opens with:
# pick pqr678 wip
# pick mno345 add init
# pick jkl012 add model attr
# pick ghi789 load method
# pick def456 implement load
# pick abc123 predict method
#
# Commands:
# p, pick = use commit
# r, reword = use commit, but edit message
# e, edit = use commit, but stop for amending
# s, squash = use commit, but meld into previous
# f, fixup = like squash, but discard this commit's log message
# d, drop = remove commit

# Change to:
# pick pqr678 wip
# squash mno345 add init
# squash jkl012 add model attr
# squash ghi789 load method
# squash def456 implement load
# pick abc123 predict method

# Save and close. Editor opens again for combined commit message.
# Replace with:
# feat(api): add model serving infrastructure
#
# Implement ModelServer class with:
# - Model initialization and attribute management
# - Model loading from pickle files
# - Basic prediction interface
#
# Foundation for ML model deployment.

# View cleaned history
git log --oneline -3
# Output:
# xyz789 predict method
# abc123 feat(api): add model serving infrastructure
# pqr678 Initial commit

# Much cleaner!
```

**Rebase Command Reference:**
- `pick` (p) - Keep commit as-is
- `reword` (r) - Keep changes, edit message
- `edit` (e) - Pause to amend commit
- `squash` (s) - Merge with previous, keep both messages
- `fixup` (f) - Merge with previous, discard message
- `drop` (d) - Remove commit

### Step 1.2: Reorder Commits Logically

```bash
# Add documentation and tests (wrong order)
mkdir -p docs
echo "# Tests" > tests/test_server.py
git add tests/test_server.py
git commit -m "test: add server tests"

echo "# Server API Documentation" > docs/server.md
git add docs/server.md
git commit -m "docs: document server API"

# Oops! Docs should come before tests
git log --oneline -3

# Reorder with interactive rebase
git rebase -i HEAD~2

# In editor, reorder lines:
# pick <hash2> docs: document server API
# pick <hash1> test: add server tests

# Save - commits now in logical order
git log --oneline -3
```

### Step 1.3: Split Large Commit

```bash
# Make large commit with multiple unrelated changes
cat >> src/api/server.py << 'EOF'

    def health_check(self):
        """Check server health."""
        return {"status": "healthy", "model_loaded": self.model is not None}

    def metrics(self):
        """Get server metrics."""
        return {"total_requests": 0, "avg_latency_ms": 0}
EOF

cat > src/api/config.py << 'EOF'
"""Server configuration."""

CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "timeout": 30
}
EOF

git add src/api/
git commit -m "Add stuff"

# Bad commit message and too much in one commit!
# Split it using interactive rebase:
git rebase -i HEAD~1

# Change "pick" to "edit"
# Save - Git pauses at the commit

# Reset commit but keep changes
git reset HEAD^

# Now working directory has all changes, but uncommitted
git status

# Commit pieces separately
git add src/api/server.py
git commit -m "feat(api): add health check and metrics endpoints

Add monitoring endpoints:
- /health: Health check with model status
- /metrics: Performance metrics

Enables monitoring and alerting integration."

git add src/api/config.py
git commit -m "feat(config): add server configuration module

Centralize configuration:
- Host and port settings
- Worker count
- Timeout configuration

Supports environment-based configuration."

# Continue rebase
git rebase --continue

# View split commits
git log --oneline -3
# Now two well-documented commits instead of one vague one!
```

---

## Part 2: Custom Git Hooks

### Step 2.1: Pre-Commit Hook

Prevent bad commits with automated checks.

```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook: Code quality and security checks

set -e

echo "🔍 Running pre-commit checks..."
echo ""

# Check 1: Python syntax
echo "1. Checking Python syntax..."
python_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -n "$python_files" ]; then
    for file in $python_files; do
        if [ -f "$file" ]; then
            python3 -m py_compile "$file" 2>&1
            if [ $? -ne 0 ]; then
                echo "❌ Syntax error in $file"
                exit 1
            fi
        fi
    done
    echo "✅ Python syntax valid"
else
    echo "⏭️  No Python files changed"
fi

# Check 2: Debug statements
echo ""
echo "2. Checking for debug statements..."
if git diff --cached | grep -E "(print\(|pdb\.set_trace|breakpoint\(|console\.log)" > /dev/null; then
    echo "❌ Found debug statements:"
    git diff --cached --name-only --diff-filter=ACM | while read file; do
        if grep -n -E "(print\(|pdb\.set_trace|breakpoint\()" "$file" 2>/dev/null; then
            echo "  $file"
        fi
    done
    echo ""
    echo "Remove debug statements or use: git commit --no-verify"
    exit 1
fi
echo "✅ No debug statements"

# Check 3: Large files
echo ""
echo "3. Checking file sizes..."
max_size=1048576  # 1 MB
large_files=false
for file in $(git diff --cached --name-only --diff-filter=ACM); do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file" 2>/dev/null || echo 0)
        if [ "$size" -gt "$max_size" ]; then
            echo "❌ File too large: $file ($(numfmt --to=iec-i --suffix=B $size))"
            echo "   Use Git LFS for files > 1MB"
            large_files=true
        fi
    fi
done
if [ "$large_files" = true ]; then
    exit 1
fi
echo "✅ File sizes OK"

# Check 4: Secrets detection
echo ""
echo "4. Checking for secrets..."
if git diff --cached | grep -E "(AKIA[0-9A-Z]{16}|password\s*=\s*['\"][^'\"]+|api[_-]?key\s*=)" > /dev/null; then
    echo "❌ Possible secret detected!"
    echo "   Review your changes for:"
    echo "   - AWS keys (AKIA...)"
    echo "   - Hardcoded passwords"
    echo "   - API keys"
    exit 1
fi
echo "✅ No secrets detected"

# Check 5: TODO/FIXME comments
echo ""
echo "5. Checking for TODOs..."
todo_count=$(git diff --cached | grep -c "TODO\|FIXME" || true)
if [ "$todo_count" -gt 0 ]; then
    echo "⚠️  Found $todo_count TODO/FIXME comments"
    echo "   Consider creating issues for them"
fi

echo ""
echo "✅ All pre-commit checks passed!"
echo ""
exit 0
EOF

chmod +x .git/hooks/pre-commit

# Test it - try to commit with debug statement
echo 'print("debug info")' > src/test_debug.py
git add src/test_debug.py
git commit -m "test: debug commit"
# Hook blocks it!

# Remove debug and try again
echo '# No debug' > src/test_debug.py
git add src/test_debug.py
git commit -m "test: add test module"
# Success!
```

### Step 2.2: Commit-Msg Hook

Enforce commit message format.

```bash
cat > .git/hooks/commit-msg << 'EOF'
#!/bin/bash
# Commit-msg hook: Validate conventional commit format

commit_msg_file=$1
commit_msg=$(cat "$commit_msg_file")

# Conventional commit pattern
pattern="^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .{1,100}"

if ! echo "$commit_msg" | grep -qE "$pattern"; then
    echo "❌ Invalid commit message format!"
    echo ""
    echo "Format: <type>(<scope>): <subject>"
    echo ""
    echo "Types:"
    echo "  feat:     New feature"
    echo "  fix:      Bug fix"
    echo "  docs:     Documentation"
    echo "  test:     Tests"
    echo "  refactor: Code refactoring"
    echo "  chore:    Maintenance"
    echo ""
    echo "Examples:"
    echo "  feat(api): add batch prediction endpoint"
    echo "  fix(model): correct normalization bug"
    echo "  docs(readme): update installation steps"
    echo ""
    echo "Your message:"
    echo "  $commit_msg"
    echo ""
    exit 1
fi

exit 0
EOF

chmod +x .git/hooks/commit-msg

# Test it
echo "update" > README.md
git add README.md
git commit -m "updated readme"
# Fails! Bad format

git commit -m "docs(readme): update setup instructions"
# Success!
```

### Step 2.3: Pre-Push Hook

Run tests before pushing.

```bash
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
# Pre-push hook: Validate before pushing

echo "🚀 Running pre-push checks..."
echo ""

# Check 1: Run tests
echo "1. Running test suite..."
if [ -f "pytest.ini" ] || [ -d "tests" ]; then
    # pytest tests/ -v --tb=short
    echo "✅ Tests passed (simulated)"
else
    echo "⏭️  No tests found"
fi

# Check 2: Check branch
echo ""
echo "2. Checking target branch..."
current_branch=$(git branch --show-current)
if [ "$current_branch" = "main" ] || [ "$current_branch" = "master" ]; then
    echo "⚠️  You're pushing directly to $current_branch!"
    read -p "   Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Push cancelled"
        exit 1
    fi
fi
echo "✅ Branch check passed"

# Check 3: Validate commit messages
echo ""
echo "3. Validating commit messages..."
# Get commits being pushed
remote="$1"
url="$2"

while read local_ref local_sha remote_ref remote_sha; do
    if [ "$local_sha" != "0000000000000000000000000000000000000000" ]; then
        if [ "$remote_sha" = "0000000000000000000000000000000000000000" ]; then
            # New branch, check all commits
            range="$local_sha"
        else
            # Existing branch, check new commits
            range="$remote_sha..$local_sha"
        fi

        # Validate each commit message
        git rev-list "$range" | while read commit; do
            msg=$(git log -1 --pretty=%s "$commit")
            if ! echo "$msg" | grep -qE "^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .+"; then
                echo "❌ Invalid commit message: $msg"
                echo "   Commit: $commit"
                exit 1
            fi
        done
    fi
done

echo "✅ All commit messages valid"

echo ""
echo "✅ All pre-push checks passed!"
echo ""
exit 0
EOF

chmod +x .git/hooks/pre-push
```

---

## Part 3: Cherry-Picking

### Step 3.1: Apply Specific Commit from Another Branch

```bash
# Create hotfix branch with critical fixes
git switch main
git switch -c hotfix/security-fixes

echo "def validate_input(data):" > src/api/security.py
echo "    # Prevent SQL injection" >> src/api/security.py
echo "    return sanitize(data)" >> src/api/security.py
git add src/api/security.py
git commit -m "fix(security): add input validation to prevent SQL injection

Critical security fix for CVE-2024-1234.
Sanitizes all user input before database queries.

Priority: CRITICAL"

echo "def rate_limit(request):" >> src/api/security.py
echo "    # Prevent DoS" >> src/api/security.py
git add src/api/security.py
git commit -m "fix(security): add rate limiting

Prevents denial of service attacks.
Limits: 100 requests/minute per IP."

# Log the commits
git log --oneline -2
# abc123 fix(security): add rate limiting
# def456 fix(security): add input validation

# Switch to feature branch that needs the security fix
git switch feature/model-serving

# Cherry-pick ONLY the SQL injection fix
git cherry-pick def456

# Now feature branch has the critical security fix
# without the rate limiting feature
git log --oneline -2

# View the cherry-picked commit
git show HEAD
```

### Step 3.2: Cherry-Pick Multiple Commits

```bash
# Cherry-pick multiple commits at once
git cherry-pick abc123 def456

# Or cherry-pick a range (NOT including start)
git cherry-pick start-hash..end-hash

# If conflicts occur during cherry-pick:
# 1. Fix conflicts in files
# 2. git add <resolved-files>
# 3. git cherry-pick --continue

# To abort cherry-pick:
# git cherry-pick --abort
```

---

## Part 4: Advanced Stashing

### Step 4.1: Stash with Description

```bash
# Working on a feature
echo "# WIP: caching layer" >> src/api/cache.py
echo "# TODO: implement Redis" >> src/api/cache.py

# Need to switch branches urgently
# Stash with descriptive message
git stash push -m "WIP: Redis caching layer implementation"

# List stashes
git stash list
# Output:
# stash@{0}: On feature/model-serving: WIP: Redis caching layer implementation

# Do other work...
git switch main
# ... handle urgent issue ...

# Return and restore stash
git switch feature/model-serving
git stash pop

# Work restored!
```

### Step 4.2: Partial Stashing

```bash
# Changes in multiple files
echo "# Change to API" >> src/api/server.py
echo "# Change to model" >> src/models/model.py
echo "# Change to tests" >> tests/test_api.py

git status
# All 3 files modified

# Stash only specific files
git stash push -m "API changes only" src/api/server.py

# Other files still modified
git status
# Modified: src/models/model.py, tests/test_api.py

# Restore later
git stash pop
```

### Step 4.3: Create Branch from Stash

```bash
# Stashed experimental work
echo "# Experimental optimization" >> src/models/optimized.py
git add src/models/optimized.py
git stash

# Decide to develop it properly in a branch
git stash branch experiment/model-optimization

# Creates new branch and applies stash
# Perfect for exploratory work that becomes a feature!
```

---

## Part 5: Git Bisect - Find Bug Introduction

### Step 5.1: Manual Bisect

```bash
# Create history with bug introduced at some point
git switch main

# Good commits
for i in {1..5}; do
    echo "feature $i" >> features.txt
    git add features.txt
    git commit -m "feat: add feature $i"
done

# Bug introduced here (commit 6)
echo "BUGGY_CODE = True" >> features.txt
git add features.txt
git commit -m "feat: add feature 6"

# More commits after bug
for i in {7..10}; do
    echo "feature $i" >> features.txt
    git add features.txt
    git commit -m "feat: add feature $i"
done

# Now at commit 10, bug exists but we don't know when it was introduced

# Start bisect
git bisect start

# Mark current as bad (has bug)
git bisect bad

# Mark earlier commit as good (before bug)
git bisect good HEAD~10

# Git checks out middle commit
# Test it manually
cat features.txt | grep "BUGGY_CODE"

# If bug present:
git bisect bad

# If no bug:
# git bisect good

# Git continues binary search...
# Repeat testing until Git identifies exact commit

# When found:
# Output: abc123 is the first bad commit
#         feat: add feature 6

# Reset
git bisect reset
```

### Step 5.2: Automated Bisect

```bash
# Create automated test script
cat > scripts/test_for_bug.sh << 'EOF'
#!/bin/bash
# Exit 0 if good (no bug)
# Exit 1 if bad (bug present)

if grep -q "BUGGY_CODE" features.txt; then
    echo "❌ Bug detected!"
    exit 1
else
    echo "✅ No bug"
    exit 0
fi
EOF

chmod +x scripts/test_for_bug.sh

# Run automated bisect
git bisect start HEAD HEAD~10
git bisect run scripts/test_for_bug.sh

# Git automatically finds the bad commit!
# Output:
# abc123 is the first bad commit
# feat: add feature 6

# View the bad commit
git show abc123

# Reset
git bisect reset
```

---

## Part 6: Reflog - Time Travel and Recovery

### Step 6.1: Recover Lost Commits

```bash
# Make important work
echo "Critical feature" > critical.txt
git add critical.txt
git commit -m "feat: critical feature implementation"

# Note commit hash
git log --oneline -1
# abc123 feat: critical feature implementation

# Accidentally hard reset (LOSE the commit)
git reset --hard HEAD~1

# Commit gone from log!
git log --oneline -1
# (previous commit shown)

# OH NO! But reflog saves us
git reflog

# Output shows all ref movements:
# def456 HEAD@{0}: reset: moving to HEAD~1
# abc123 HEAD@{1}: commit: feat: critical feature implementation
# xyz789 HEAD@{2}: commit: previous commit

# Recover by checking out that state
git checkout HEAD@{1}
# You're in detached HEAD, but work is here!

# Create branch to save it
git branch recovery-branch
git switch recovery-branch

# Or directly:
git reset --hard HEAD@{1}

# Work recovered!
```

### Step 6.2: Undo Bad Rebase

```bash
# Record state before rebase
git log --oneline -3

# Do rebase
git rebase -i HEAD~3
# Make changes, save

# Oops! Messed up the rebase

# Find pre-rebase state
git reflog
# Look for "rebase -i (start)"
# The entry BEFORE that is pre-rebase state

# Reset to pre-rebase
git reset --hard HEAD@{5}  # Adjust number

# Rebase undone! Back to original state
```

---

## Part 7: Git Worktrees

### Step 7.1: Multiple Working Directories

```bash
# Main working directory on main branch
git switch main

# Need to work on feature branch simultaneously
# (e.g., compare implementations)

# Create worktree
git worktree add ../ml-platform-feature feature/model-serving

# Now have two directories:
# - ml-platform-advanced/ (main branch)
# - ml-platform-feature/ (feature branch)

# Work in feature directory
cd ../ml-platform-feature
# Make changes
echo "Feature work" >> src/api/server.py
git commit -am "feat: update server"

# Back to main, different state
cd ../ml-platform-advanced
cat src/api/server.py  # Doesn't have feature work

# List all worktrees
git worktree list

# Output:
# /path/ml-platform-advanced  abc123 [main]
# /path/ml-platform-feature   def456 [feature/model-serving]

# Remove worktree when done
git worktree remove ../ml-platform-feature

# Or prune deleted worktrees
git worktree prune
```

---

## Part 8: Troubleshooting and Recovery

### Issue 1: Detached HEAD Recovery

```bash
# Accidentally checkout old commit
git checkout HEAD~5

# Warning: You are in 'detached HEAD' state

# Make changes
echo "work" > work.txt
git add work.txt
git commit -m "Work in detached HEAD"

# Save work before losing it
git branch save-work

# Switch to saved branch
git switch save-work

# Merge into main if needed
git switch main
git merge save-work
```

### Issue 2: Committed to Wrong Branch

```bash
# Made commit on main by mistake
git log --oneline -1
# abc123 feat: new feature

# Move to correct branch
git reset --hard HEAD~1  # Remove from main
git switch feature/correct-branch
git cherry-pick abc123  # Add to correct branch
```

### Issue 3: Recover Deleted Branch

```bash
# Deleted branch accidentally
git branch -D feature/important

# Find it in reflog
git reflog | grep "feature/important"

# Recreate branch at that commit
git branch feature/important HEAD@{10}
```

---

## Verification Checklist

- [ ] Successfully cleaned commit history with interactive rebase
- [ ] Created and tested custom Git hooks (pre-commit, commit-msg)
- [ ] Cherry-picked commits between branches
- [ ] Used stash for temporary work storage
- [ ] Found bug with git bisect
- [ ] Recovered lost work with reflog
- [ ] Created and used worktrees
- [ ] Recovered from mistakes (wrong branch, detached HEAD)
- [ ] Understand when to use each advanced technique

---

## Summary

You've mastered advanced Git techniques:

- ✅ Interactive rebase for professional commit history
- ✅ Custom hooks for automated quality checks
- ✅ Cherry-picking for selective commits
- ✅ Advanced stashing workflows
- ✅ Bisecting to find bug introductions
- ✅ Reflog for time travel and recovery
- ✅ Worktrees for parallel development
- ✅ Troubleshooting and recovery strategies

**Key Takeaways:**
- Interactive rebase cleans history before merging
- Hooks automate quality checks
- Reflog is your safety net
- Advanced techniques enable complex workflows
- Always test before pushing

**Time to Complete:** ~150 minutes

**Next Exercise:** Exercise 08 - Git LFS for ML Projects
