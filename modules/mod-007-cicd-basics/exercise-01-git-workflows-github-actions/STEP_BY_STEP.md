# Step-by-Step Implementation Guide: Git Workflows & GitHub Actions

## Overview

Master Git workflows and GitHub Actions for CI/CD in AI infrastructure projects.

**Time**: 3-4 hours | **Difficulty**: Intermediate

---

## Phase 1: Git Workflow Setup (45 minutes)

### Step 1: Initialize Repository and Branching Strategy

```bash
# Clone or create repository
git clone https://github.com/your-org/ml-project.git
cd ml-project

# Set up main and develop branches
git checkout -b develop
git push origin develop

# Protect main branch (on GitHub)
# Settings → Branches → Add rule for 'main'
# ✓ Require pull request reviews
# ✓ Require status checks to pass
```

### Step 2: Feature Branch Workflow

```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/model-training

# Make changes
echo "def train_model(): pass" > ml_model.py
git add ml_model.py
git commit -m "feat: Add model training function"

# Push feature branch
git push origin feature/model-training

# Create pull request on GitHub
# Base: develop ← Compare: feature/model-training
```

### Step 3: Conventional Commits

```bash
# Commit message format: <type>(<scope>): <description>

# Types:
# feat: New feature
# fix: Bug fix
# docs: Documentation
# style: Code style (formatting)
# refactor: Code refactoring
# test: Add tests
# chore: Maintenance tasks

# Examples:
git commit -m "feat(training): Add data preprocessing pipeline"
git commit -m "fix(inference): Handle missing input features"
git commit -m "docs(readme): Update installation instructions"
git commit -m "test(model): Add unit tests for prediction"
```

---

## Phase 2: GitHub Actions - Basic CI (1 hour)

### Step 4: Create Basic CI Workflow

Create `.github/workflows/01-basic-ci.yml`:

```yaml
name: Basic CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run basic checks
      run: |
        python --version
        pip list
        echo "✅ Basic CI passed"
```

**Test the workflow**:
```bash
git add .github/workflows/01-basic-ci.yml
git commit -m "ci: Add basic CI workflow"
git push origin feature/model-training

# Check Actions tab on GitHub
```

### Step 5: Python Linting Workflow

Create `.github/workflows/02-python-lint.yml`:

```yaml
name: Python Linting

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install linting tools
      run: |
        pip install flake8 black isort pylint

    - name: Run flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

    - name: Check black formatting
      run: black --check .

    - name: Check import sorting
      run: isort --check-only .

    - name: Run pylint
      run: pylint **/*.py --fail-under=8.0 || true
```

### Step 6: Python Testing Workflow

Create `.github/workflows/03-python-test.yml`:

```yaml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install pytest pytest-cov
        pip install -r requirements.txt

    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=. --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-${{ matrix.os }}-py${{ matrix.python-version }}
```

---

## Phase 3: Code Quality Automation (1 hour)

### Step 7: Comprehensive Quality Checks

Create `.github/workflows/04-code-quality.yml`:

```yaml
name: Code Quality

on:
  pull_request:
    branches: [ main, develop ]

jobs:
  quality:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install tools
      run: |
        pip install flake8 black isort mypy bandit safety

    - name: Code formatting (black)
      run: black --check --diff .

    - name: Import sorting (isort)
      run: isort --check-only --diff .

    - name: Linting (flake8)
      run: flake8 . --max-line-length=88 --extend-ignore=E203

    - name: Type checking (mypy)
      run: mypy . --ignore-missing-imports

    - name: Security check (bandit)
      run: bandit -r . -ll

    - name: Dependency security (safety)
      run: safety check --json

    - name: Check for secrets
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
```

### Step 8: Pull Request Validation

Create `.github/workflows/05-pr-checks.yml`:

```yaml
name: PR Validation

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  pr-checks:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for commit analysis

    - name: Check PR title (conventional commits)
      uses: amannn/action-semantic-pull-request@v5
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

    - name: Check PR size
      run: |
        FILES_CHANGED=$(git diff --name-only origin/${{ github.base_ref }}...HEAD | wc -l)
        LINES_CHANGED=$(git diff --stat origin/${{ github.base_ref }}...HEAD | tail -1 | awk '{print $4 + $6}')

        if [ $FILES_CHANGED -gt 20 ]; then
          echo "⚠️  PR is large: $FILES_CHANGED files changed"
          echo "Consider splitting into smaller PRs"
        fi

        if [ $LINES_CHANGED -gt 500 ]; then
          echo "⚠️  PR is large: $LINES_CHANGED lines changed"
        fi

    - name: Check for TODOs
      run: |
        if grep -r "TODO\|FIXME" --include="*.py" .; then
          echo "⚠️  Found TODOs or FIXMEs in code"
          exit 1
        fi

    - name: Check for print statements
      run: |
        if grep -r "print(" --include="*.py" . | grep -v "test_"; then
          echo "⚠️  Found print() statements (use logging instead)"
          exit 1
        fi
```

---

## Phase 4: Release Automation (45 minutes)

### Step 9: Automated Releases

Create `.github/workflows/06-release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Generate changelog
      id: changelog
      run: |
        PREVIOUS_TAG=$(git describe --abbrev=0 --tags $(git rev-list --tags --skip=1 --max-count=1) 2>/dev/null || echo "")
        if [ -z "$PREVIOUS_TAG" ]; then
          CHANGELOG=$(git log --pretty=format:"- %s (%h)" | head -20)
        else
          CHANGELOG=$(git log ${PREVIOUS_TAG}..HEAD --pretty=format:"- %s (%h)")
        fi
        echo "changelog<<EOF" >> $GITHUB_OUTPUT
        echo "$CHANGELOG" >> $GITHUB_OUTPUT
        echo "EOF" >> $GITHUB_OUTPUT

    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        body: |
          ## Changes
          ${{ steps.changelog.outputs.changelog }}
        files: dist/*
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: |
        twine upload dist/*
```

**Create a release**:
```bash
# Tag the release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# GitHub Actions will automatically create release and publish
```

---

## Phase 5: Git Hooks (Local Validation) (30 minutes)

### Step 10: Pre-commit Hooks

Create `scripts/pre-commit-hook.sh`:

```bash
#!/bin/bash
# Pre-commit hook for code quality checks

echo "Running pre-commit checks..."

# Check Python formatting
echo "Checking code formatting..."
black --check . || {
    echo "❌ Code formatting failed. Run: black ."
    exit 1
}

# Check imports
echo "Checking import sorting..."
isort --check-only . || {
    echo "❌ Import sorting failed. Run: isort ."
    exit 1
}

# Run linting
echo "Running linter..."
flake8 . || {
    echo "❌ Linting failed"
    exit 1
}

# Run tests
echo "Running tests..."
pytest tests/ -q || {
    echo "❌ Tests failed"
    exit 1
}

echo "✅ All pre-commit checks passed"
exit 0
```

**Install Git hooks**:
```bash
# Make executable
chmod +x scripts/pre-commit-hook.sh

# Install hook
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit

# Or use setup script
./scripts/setup-git-hooks.sh

# Test
git commit -m "test: Trigger pre-commit hook"
```

---

## Phase 6: Scheduled Workflows (15 minutes)

### Step 11: Nightly Builds

Create `.github/workflows/07-schedule.yml`:

```yaml
name: Scheduled Jobs

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:  # Manual trigger

jobs:
  nightly-build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run comprehensive tests
      run: pytest tests/ --cov=. --cov-report=html

    - name: Check dependencies
      run: pip list --outdated

    - name: Security audit
      run: |
        pip install safety
        safety check

    - name: Send notification
      if: failure()
      run: |
        echo "Nightly build failed!"
        # Send Slack/email notification
```

---

## Summary

**What You Built**:
- ✅ Git branching workflow (feature branches)
- ✅ Conventional commit messages
- ✅ GitHub Actions CI pipelines (7 workflows)
- ✅ Automated linting and formatting
- ✅ Multi-OS, multi-Python testing
- ✅ Code quality checks (mypy, bandit, safety)
- ✅ PR validation automation
- ✅ Release automation
- ✅ Local Git hooks
- ✅ Scheduled nightly builds

**Key Workflows**:
```yaml
# Basic structure
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: command
```

**Git Workflow**:
```bash
# Feature branch
git checkout -b feature/name
git commit -m "feat: description"
git push origin feature/name

# Create PR → CI runs → Merge
```

**Next Steps**:
- Exercise 02: Automated Testing
- Set up branch protection rules
- Configure required status checks
- Add code owners (CODEOWNERS file)
