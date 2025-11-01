#!/bin/bash

#######################################################################
# Exercise 07: Advanced Git Techniques - Setup Script
#######################################################################
# Creates a complete ML platform repository demonstrating:
# - Interactive rebase scenarios
# - Git hooks (pre-commit, post-commit, pre-push)
# - Cherry-picking scenarios
# - Stashing workflows
# - Bisect demonstration
# - Reflog and recovery examples
# - Submodules
# - Advanced merge strategies
#######################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$EXERCISE_DIR/ml-platform-advanced"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Advanced Git Techniques Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Clean up existing project
if [ -d "$PROJECT_DIR" ]; then
    rm -rf "$PROJECT_DIR"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

#######################################################################
# Part 1: Initial Setup
#######################################################################

echo -e "${YELLOW}[1/10] Creating initial project structure...${NC}"

git init
git config user.name "Platform Engineer"
git config user.email "engineer@ml-platform.com"

# Create project structure
mkdir -p src/{api,models,pipeline} tests scripts docs configs

cat > README.md << 'EOF'
# ML Platform Advanced

Production ML platform with advanced Git workflows.

## Features

- Model serving API
- ML pipeline orchestration
- Monitoring and metrics
- Advanced deployment strategies

## Development

See docs/DEVELOPMENT.md for Git workflow guidelines.
EOF

cat > src/api/__init__.py << 'EOF'
"""ML Platform API."""
__version__ = "1.0.0"
EOF

cat > src/models/__init__.py << 'EOF'
"""ML Models module."""
EOF

cat > tests/__init__.py << 'EOF'
"""Test suite."""
EOF

git add .
git commit -m "init: initialize ML platform project

Project structure:
- src/api: REST API endpoints
- src/models: ML model implementations
- src/pipeline: Data pipelines
- tests: Test suite
- scripts: Utility scripts"

# Rename master to main
git branch -m master main

#######################################################################
# Part 2: Create Messy History for Rebase Practice
#######################################################################

echo -e "${YELLOW}[2/10] Creating feature branch with messy history...${NC}"

git switch -c feature/model-serving

# Simulate messy development with many small commits
cat > src/api/server.py << 'EOF'
"""Model serving API."""


class ModelServer:
EOF

git add src/api/server.py
git commit -m "wip server"

cat >> src/api/server.py << 'EOF'
    """HTTP server for model inference."""

    def __init__(self, port=8000):
        """Initialize server."""
        self.port = port
EOF

git add src/api/server.py
git commit -m "add init method"

cat >> src/api/server.py << 'EOF'
        self.model = None
        self.cache = {}
EOF

git add src/api/server.py
git commit -m "add attributes"

cat >> src/api/server.py << 'EOF'

    def load_model(self, path):
        """Load ML model from path."""
EOF

git add src/api/server.py
git commit -m "load model"

cat >> src/api/server.py << 'EOF'
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self.model
EOF

git add src/api/server.py
git commit -m "fix: implement model loading"

cat >> src/api/server.py << 'EOF'

    def predict(self, data):
        """Make prediction."""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model.predict(data)
EOF

git add src/api/server.py
git commit -m "add predict method"

cat >> src/api/server.py << 'EOF'

    def health_check(self):
        """Check server health."""
        return {"status": "healthy", "model_loaded": self.model is not None}
EOF

git add src/api/server.py
git commit -m "add health endpoint"

#######################################################################
# Part 3: Create Git Hooks
#######################################################################

echo -e "${YELLOW}[3/10] Creating Git hooks...${NC}"

# Pre-commit hook
cat > .git/hooks/pre-commit << 'HOOK_EOF'
#!/bin/bash
# Pre-commit hook: Run code quality checks

echo "🔍 Running pre-commit checks..."
echo ""

error_count=0

# Check for Python syntax errors
echo "[1/4] Checking Python syntax..."
python_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -n "$python_files" ]; then
    for file in $python_files; do
        if [ -f "$file" ]; then
            python3 -m py_compile "$file" 2>/dev/null
            if [ $? -ne 0 ]; then
                echo "❌ Syntax error in $file"
                error_count=$((error_count + 1))
            fi
        fi
    done
    if [ $error_count -eq 0 ]; then
        echo "✅ Python syntax OK"
    fi
else
    echo "⏭️  No Python files to check"
fi
echo ""

# Check for debug statements
echo "[2/4] Checking for debug statements..."
if git diff --cached | grep -E "print\(|pdb\.set_trace|breakpoint\(" > /dev/null; then
    echo "❌ Found debug statements. Remove before committing."
    echo ""
    echo "To commit anyway: git commit --no-verify"
    error_count=$((error_count + 1))
else
    echo "✅ No debug statements found"
fi
echo ""

# Check for large files
echo "[3/4] Checking file sizes..."
max_size=1048576  # 1 MB
for file in $(git diff --cached --name-only --diff-filter=ACM); do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file" 2>/dev/null || echo 0)
        if [ $size -gt $max_size ]; then
            echo "❌ File too large: $file ($(($size / 1024))KB > $(($max_size / 1024))KB)"
            echo "   Consider using Git LFS for large files"
            error_count=$((error_count + 1))
        fi
    fi
done
if [ $error_count -eq 0 ]; then
    echo "✅ File sizes OK"
fi
echo ""

# Check for secrets
echo "[4/4] Checking for secrets..."
if git diff --cached | grep -E "AKIA[0-9A-Z]{16}|sk-[a-zA-Z0-9]{48}" > /dev/null; then
    echo "❌ Possible AWS key or API token detected!"
    error_count=$((error_count + 1))
else
    echo "✅ No secrets detected"
fi
echo ""

if [ $error_count -gt 0 ]; then
    echo "❌ Pre-commit checks failed with $error_count error(s)"
    exit 1
fi

echo "✅ All pre-commit checks passed!"
exit 0
HOOK_EOF

chmod +x .git/hooks/pre-commit

# Post-commit hook
cat > .git/hooks/post-commit << 'HOOK_EOF'
#!/bin/bash
# Post-commit hook: Log commits for tracking

commit_hash=$(git rev-parse --short HEAD)
commit_msg=$(git log -1 --pretty=%B | head -1)
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
author=$(git log -1 --pretty=%an)

echo "[$timestamp] $commit_hash: $commit_msg (by $author)" >> .git/commit-log.txt

echo "✅ Commit logged: $commit_hash"
HOOK_EOF

chmod +x .git/hooks/post-commit

# Pre-push hook
cat > .git/hooks/pre-push << 'HOOK_EOF'
#!/bin/bash
# Pre-push hook: Validate before push

echo "🚀 Running pre-push checks..."
echo ""

# Check branch name
current_branch=$(git branch --show-current)
if [ "$current_branch" = "main" ]; then
    echo "⚠️  You are pushing directly to main branch!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Push aborted"
        exit 1
    fi
fi

# Validate commit messages
echo "[1/2] Validating commit messages..."
commits=$(git log @{u}.. --pretty=format:"%s" 2>/dev/null || git log --pretty=format:"%s" -5)
invalid_count=0

while read -r msg; do
    if [ -n "$msg" ]; then
        if ! echo "$msg" | grep -E "^(feat|fix|docs|test|refactor|chore|ci|perf|style)(\(.+\))?: .+" > /dev/null; then
            echo "❌ Invalid commit message: $msg"
            echo "   Format: type(scope): message"
            echo "   Example: feat(api): add new endpoint"
            invalid_count=$((invalid_count + 1))
        fi
    fi
done <<< "$commits"

if [ $invalid_count -eq 0 ]; then
    echo "✅ Commit messages valid"
else
    echo ""
    echo "To push anyway: git push --no-verify"
    exit 1
fi
echo ""

echo "[2/2] Running tests (simulated)..."
echo "✅ Tests passed"
echo ""

echo "✅ All pre-push checks passed!"
exit 0
HOOK_EOF

chmod +x .git/hooks/pre-push

git add .git/hooks/
# Note: hooks are not tracked, but we document them

cat > docs/HOOKS.md << 'EOF'
# Git Hooks

This project uses Git hooks for quality control.

## Pre-Commit Hook

Runs before each commit:
- Python syntax validation
- Debug statement detection
- File size checks
- Secret detection

## Post-Commit Hook

Logs all commits to `.git/commit-log.txt`.

## Pre-Push Hook

Validates before push:
- Commit message format
- Test execution

## Bypass Hooks

To bypass hooks (use carefully):
```bash
git commit --no-verify
git push --no-verify
```
EOF

git add docs/HOOKS.md
git commit -m "docs: document Git hooks

Added comprehensive hooks for code quality:
- pre-commit: syntax, debug statements, secrets
- post-commit: commit logging
- pre-push: message validation, tests"

#######################################################################
# Part 4: Create Branches for Cherry-Picking
#######################################################################

echo -e "${YELLOW}[4/10] Creating branches for cherry-picking...${NC}"

# Create hotfix branch with multiple fixes
git switch main
git switch -c hotfix/critical-fixes

cat > src/api/security.py << 'EOF'
"""Security utilities."""


def validate_token(token):
    """Validate authentication token."""
    if not token or len(token) < 32:
        raise ValueError("Invalid token")
    # Security validation logic
    return True


def sanitize_input(data):
    """Sanitize user input."""
    # Remove dangerous characters
    return str(data).replace("<", "").replace(">", "")
EOF

git add src/api/security.py
git commit -m "fix: add authentication token validation (CVE-2024-001)

Critical security fix for token validation.
Prevents authentication bypass vulnerability.

Severity: HIGH
Impact: Authentication system"

cat >> src/api/security.py << 'EOF'


def rate_limit_check(user_id, max_requests=100):
    """Check rate limiting."""
    # Rate limiting logic
    return True
EOF

git add src/api/security.py
git commit -m "fix: implement rate limiting

Prevents DoS attacks by limiting requests per user.

Severity: MEDIUM
Impact: API availability"

cat > src/models/memory_fix.py << 'EOF'
"""Memory management fixes."""
import gc


def cleanup_model(model):
    """Clean up model resources."""
    del model
    gc.collect()
    return True
EOF

git add src/models/memory_fix.py
git commit -m "fix: resolve memory leak in model loading

Properly cleanup model resources after use.

Fixes memory accumulation issue in production.
Reduces memory usage by ~40%."

# Create experimental branch
git switch main
git switch -c experiment/new-architecture

cat > src/pipeline/async_pipeline.py << 'EOF'
"""Experimental async pipeline."""
import asyncio


async def process_batch(items):
    """Process items asynchronously."""
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)


async def process_item(item):
    """Process single item."""
    # Async processing
    await asyncio.sleep(0.1)
    return item
EOF

git add src/pipeline/async_pipeline.py
git commit -m "experiment: add async batch processing

Experimental async pipeline for improved throughput.
Testing concurrent batch processing approach."

#######################################################################
# Part 5: Create History for Bisect
#######################################################################

echo -e "${YELLOW}[5/10] Creating history for bisect demonstration...${NC}"

git switch main
git switch -c debug/performance-regression

# Create good commits
for i in {1..5}; do
    echo "# Version $i improvements" >> src/models/inference.py
    echo "def inference_v$i(data):" >> src/models/inference.py
    echo "    return data * $i" >> src/models/inference.py
    echo "" >> src/models/inference.py
    git add src/models/inference.py
    git commit -m "perf: optimize inference v$i"
done

# Introduce performance bug
cat >> src/models/inference.py << 'EOF'
# PERFORMANCE BUG: Unnecessary nested loop
def slow_function(data):
    result = []
    for i in range(1000):
        for j in range(1000):  # O(n²) complexity!
            result.append(data)
    return result[0]

EOF

git add src/models/inference.py
git commit -m "perf: add data processing optimization"

# Continue with more commits
for i in {6..10}; do
    echo "# Version $i improvements" >> src/models/inference.py
    git add src/models/inference.py
    git commit -m "perf: optimize inference v$i"
done

# Create test script for bisect
cat > scripts/test_performance.sh << 'EOF'
#!/bin/bash
# Test script for bisect

if grep -q "for j in range(1000)" src/models/inference.py; then
    echo "❌ Performance bug detected (nested loop)"
    exit 1
else
    echo "✅ Performance looks good"
    exit 0
fi
EOF

chmod +x scripts/test_performance.sh

git add scripts/test_performance.sh
git commit -m "test: add performance test script"

#######################################################################
# Part 6: Create Stashing Scenarios
#######################################################################

echo -e "${YELLOW}[6/10] Setting up stashing examples...${NC}"

git switch main

cat > scripts/stash_examples.sh << 'EOF'
#!/bin/bash
# Examples for stashing workflows

echo "Stashing Examples:"
echo ""
echo "1. Stash with message:"
echo "   git stash push -m 'WIP: feature description'"
echo ""
echo "2. Stash specific files:"
echo "   git stash push -m 'changes' file1 file2"
echo ""
echo "3. List stashes:"
echo "   git stash list"
echo ""
echo "4. Apply stash:"
echo "   git stash pop        # Apply and remove"
echo "   git stash apply      # Apply and keep"
echo ""
echo "5. Create branch from stash:"
echo "   git stash branch new-feature stash@{0}"
EOF

chmod +x scripts/stash_examples.sh

git add scripts/stash_examples.sh
git commit -m "docs: add stashing examples script"

#######################################################################
# Part 7: Create Submodule Simulation
#######################################################################

echo -e "${YELLOW}[7/10] Setting up submodule examples...${NC}"

mkdir -p external-libs
cd external-libs

# Create mock external library
mkdir ml-utils
cd ml-utils
git init
git config user.name "Library Maintainer"
git config user.email "maintainer@ml-utils.com"

cat > README.md << 'EOF'
# ML Utilities Library

Shared utilities for ML projects.

## Features

- Data preprocessing
- Model evaluation
- Visualization tools
EOF

cat > utils.py << 'EOF'
"""ML utility functions."""


def preprocess_data(data):
    """Preprocess input data."""
    return data


def evaluate_model(model, test_data):
    """Evaluate model performance."""
    return {"accuracy": 0.95}
EOF

git add .
git commit -m "Initial library release"
git tag v1.0.0

cd ../..

# Document submodule usage
cat > docs/SUBMODULES.md << 'EOF'
# Git Submodules

This project can use submodules for external dependencies.

## Adding Submodule

```bash
git submodule add <repository-url> external-libs/ml-utils
git commit -m "chore: add ml-utils submodule"
```

## Cloning with Submodules

```bash
git clone --recursive <repo-url>
```

Or after cloning:

```bash
git submodule init
git submodule update
```

## Updating Submodule

```bash
cd external-libs/ml-utils
git pull origin main
cd ../..
git add external-libs/ml-utils
git commit -m "chore: update ml-utils to latest"
```

## Current Submodules

- `external-libs/ml-utils`: ML utility library (v1.0.0)
EOF

git add docs/SUBMODULES.md external-libs/
git commit -m "docs: add submodule documentation

Document how to work with Git submodules for
external library dependencies."

#######################################################################
# Part 8: Create Reflog Scenarios
#######################################################################

echo -e "${YELLOW}[8/10] Creating reflog demonstration...${NC}"

# Create commits that will be "lost" and recovered
git switch -c temp/reflog-demo

cat > docs/REFLOG.md << 'EOF'
# Git Reflog - Time Travel

Reflog tracks all ref updates (commits, checkouts, resets).

## View Reflog

```bash
git reflog
git reflog show HEAD
```

## Recover Lost Commits

```bash
# After accidental reset
git reflog
# Find commit: HEAD@{2}: commit: Important work
git checkout HEAD@{2}

# Or create branch
git branch recovery HEAD@{2}
```

## Undo Rebase

```bash
git reflog
# Find: HEAD@{5}: rebase -i (start)
# Entry before that is pre-rebase state
git reset --hard HEAD@{5}
```

## Reflog Expiration

- Default: 90 days for reachable refs
- 30 days for unreachable refs

## Use Cases

1. Undo destructive operations
2. Find lost commits
3. Recover from bad rebase/merge
4. Time travel debugging
EOF

git add docs/REFLOG.md
git commit -m "docs: comprehensive reflog guide"

git switch main
git branch -D temp/reflog-demo

#######################################################################
# Part 9: Create Advanced Merge Scenarios
#######################################################################

echo -e "${YELLOW}[9/10] Setting up advanced merge scenarios...${NC}"

# Create multiple feature branches for octopus merge
git switch -c feature/monitoring
cat > src/api/metrics.py << 'EOF'
"""Monitoring and metrics."""


def track_request(endpoint, duration):
    """Track API request metrics."""
    print(f"{endpoint}: {duration}ms")
EOF

git add src/api/metrics.py
git commit -m "feat(monitoring): add request tracking"

git switch main
git switch -c feature/logging
cat > src/api/logger.py << 'EOF'
"""Structured logging."""
import logging

logger = logging.getLogger(__name__)


def log_event(event, level="INFO"):
    """Log structured event."""
    logger.log(getattr(logging, level), event)
EOF

git add src/api/logger.py
git commit -m "feat(logging): add structured logging"

git switch main
git switch -c feature/caching
cat > src/api/cache.py << 'EOF'
"""Response caching."""


class Cache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
EOF

git add src/api/cache.py
git commit -m "feat(caching): implement response cache"

git switch main

#######################################################################
# Part 10: Documentation and Summary
#######################################################################

echo -e "${YELLOW}[10/10] Creating comprehensive documentation...${NC}"

cat > docs/ADVANCED_WORKFLOWS.md << 'EOF'
# Advanced Git Workflows

## Interactive Rebase

Clean up commit history before merging:

```bash
git rebase -i HEAD~5
```

Commands:
- `pick`: keep commit
- `reword`: edit message
- `squash`: merge with previous
- `fixup`: merge, discard message
- `edit`: pause to modify
- `drop`: remove commit

## Cherry-Picking

Apply specific commits:

```bash
git cherry-pick <hash>
git cherry-pick <hash1> <hash2>
git cherry-pick <start>..<end>
```

## Bisect

Find bug-introducing commit:

```bash
git bisect start
git bisect bad          # Current is bad
git bisect good HEAD~10 # Old commit was good
# Test and mark: git bisect bad/good
git bisect reset        # When done
```

Automated:

```bash
git bisect start HEAD HEAD~10
git bisect run ./test_script.sh
```

## Worktrees

Multiple working directories:

```bash
git worktree add ../project-feature feature-branch
git worktree list
git worktree remove ../project-feature
```

## Recovery

Use reflog:

```bash
git reflog
git checkout HEAD@{2}
git branch recovery HEAD@{2}
```

## Best Practices

1. **Never rewrite public history**
2. **Use interactive rebase for local branches**
3. **Cherry-pick for hotfixes**
4. **Bisect for hard-to-find bugs**
5. **Check reflog before force operations**
EOF

cat > docs/RECOVERY.md << 'EOF'
# Git Recovery Techniques

## Lost Commits

```bash
git reflog
git branch recovery <commit-hash>
```

## Detached HEAD

```bash
git branch temp-work
git switch temp-work
```

## Bad Rebase

```bash
git reflog | grep "rebase -i (start)"
git reset --hard HEAD@{N}
```

## Accidental Reset

```bash
git reflog
git reset --hard HEAD@{1}
```

## Deleted Branch

```bash
git reflog | grep <branch-name>
git branch <branch-name> <commit-hash>
```

## Committed to Wrong Branch

```bash
git log -1  # Note commit hash
git reset --hard HEAD~1
git switch correct-branch
git cherry-pick <hash>
```

## Find Lost Objects

```bash
git fsck --lost-found
```

## Prevention

1. Commit frequently
2. Use branches
3. Don't force push shared branches
4. Check reflog before destructive ops
5. Keep backups
EOF

git add docs/
git commit -m "docs: add advanced workflows and recovery guides

Comprehensive documentation for:
- Interactive rebase workflows
- Cherry-picking strategies
- Bisect debugging
- Worktrees usage
- Recovery techniques

Essential reading for advanced Git users."

#######################################################################
# Summary
#######################################################################

echo ""
echo -e "${GREEN}✓ Advanced Git setup complete!${NC}"
echo ""
echo "Repository structure:"
tree -L 2 -I '.git' . 2>/dev/null || find . -maxdepth 2 -type d | grep -v '.git' | sort

echo ""
echo "Git history:"
git log --oneline --graph --all -15

echo ""
echo "Branches created:"
git branch -a

echo ""
echo "Git hooks installed:"
ls -la .git/hooks/ | grep -E "pre-commit|post-commit|pre-push"

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  cd $PROJECT_DIR"
echo "  git log --oneline --graph --all"
echo "  cat docs/ADVANCED_WORKFLOWS.md"
echo "  cat docs/RECOVERY.md"
echo "  cat docs/HOOKS.md"
echo ""
echo "Practice areas:"
echo "  - Interactive rebase: feature/model-serving branch"
echo "  - Cherry-pick: hotfix/critical-fixes commits"
echo "  - Bisect: debug/performance-regression branch"
echo "  - Hooks: Try committing with debug statements"
echo "  - Recovery: Use reflog for time travel"
