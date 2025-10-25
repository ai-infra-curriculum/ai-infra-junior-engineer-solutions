# Exercise 01: Git Workflows & GitHub Actions Basics

## Overview

Learn fundamental Git workflows and GitHub Actions for CI/CD in AI infrastructure projects. This exercise covers branching strategies, pull request workflows, and automated quality checks.

## Learning Objectives

- Understand Git branching strategies (feature branches, main/develop model)
- Create and manage pull requests effectively
- Write basic GitHub Actions workflows
- Implement automated code quality checks
- Use Git hooks for local validation
- Manage release workflows

## Prerequisites

- Git installed and configured
- GitHub account
- Basic command line proficiency
- Python 3.11+ installed
- Understanding of basic version control concepts

## Project Structure

```
exercise-01-git-workflows-github-actions/
├── .github/
│   └── workflows/
│       ├── 01-basic-ci.yml            # Simple CI workflow
│       ├── 02-python-lint.yml         # Python linting
│       ├── 03-python-test.yml         # Python testing
│       ├── 04-code-quality.yml        # Comprehensive quality checks
│       ├── 05-pr-checks.yml           # Pull request validation
│       ├── 06-release.yml             # Release automation
│       └── 07-schedule.yml            # Scheduled jobs
├── scripts/
│   ├── setup-git-hooks.sh             # Install Git hooks
│   ├── pre-commit-hook.sh             # Pre-commit validation
│   ├── check-code-quality.sh          # Local quality checks
│   └── create-release.sh              # Release helper script
├── examples/
│   ├── ml_model.py                    # Example ML code
│   ├── test_ml_model.py               # Example tests
│   ├── requirements.txt               # Python dependencies
│   └── .flake8                        # Flake8 configuration
├── docs/
│   ├── GIT_WORKFLOW.md                # Git workflow guide
│   ├── GITHUB_ACTIONS.md              # GitHub Actions reference
│   ├── CODE_REVIEW.md                 # Code review guidelines
│   └── TROUBLESHOOTING.md             # Common issues
└── README.md                          # This file
```

## Exercise Tasks

### Task 1: Git Branching Workflow

**Objective**: Understand and practice feature branch workflows

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/add-model-training
   ```

2. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "feat: Add model training script"
   ```

3. **Push to remote**:
   ```bash
   git push origin feature/add-model-training
   ```

4. **Create a pull request** on GitHub

5. **Merge after approval**

**Best Practices**:
- Use descriptive branch names (`feature/`, `bugfix/`, `hotfix/`)
- Write clear commit messages (conventional commits format)
- Keep branches short-lived (< 1 week)
- Regularly sync with main branch

### Task 2: Commit Message Conventions

Learn conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, no code change
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

**Examples**:
```bash
git commit -m "feat(training): Add distributed training support"
git commit -m "fix(inference): Handle edge case in preprocessing"
git commit -m "docs: Update model deployment instructions"
git commit -m "ci: Add automated testing workflow"
```

### Task 3: Basic GitHub Actions Workflow

Create your first GitHub Actions workflow that runs on every push:

**File**: `.github/workflows/01-basic-ci.yml`

```yaml
name: Basic CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  hello-world:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Say hello
        run: echo "Hello from GitHub Actions!"

      - name: Show environment
        run: |
          echo "Branch: ${{ github.ref }}"
          echo "Commit: ${{ github.sha }}"
          echo "Actor: ${{ github.actor }}"
```

### Task 4: Python Linting Workflow

Add automated Python code quality checks:

**File**: `.github/workflows/02-python-lint.yml`

```yaml
name: Python Lint

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

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 black isort mypy

      - name: Run Flake8
        run: flake8 examples/ --max-line-length=100

      - name: Check Black formatting
        run: black --check examples/

      - name: Check import sorting
        run: isort --check-only examples/

      - name: Type checking with mypy
        run: mypy examples/ --ignore-missing-imports
```

### Task 5: Automated Testing

Add automated test execution:

**File**: `.github/workflows/03-python-test.yml`

```yaml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r examples/requirements.txt
          pip install pytest pytest-cov

      - name: Run tests with coverage
        run: pytest examples/ --cov=examples --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Task 6: Pull Request Validation

Create comprehensive PR checks:

**File**: `.github/workflows/05-pr-checks.yml`

```yaml
name: PR Validation

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  validate-pr:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for proper diff

      - name: Check PR title
        run: |
          PR_TITLE="${{ github.event.pull_request.title }}"
          if [[ ! "$PR_TITLE" =~ ^(feat|fix|docs|style|refactor|test|chore|ci)(\(.+\))?:\ .+ ]]; then
            echo "PR title must follow conventional commits format"
            exit 1
          fi

      - name: Check branch name
        run: |
          BRANCH="${{ github.head_ref }}"
          if [[ ! "$BRANCH" =~ ^(feature|bugfix|hotfix|release)/.+ ]]; then
            echo "Branch name must start with feature/, bugfix/, hotfix/, or release/"
            exit 1
          fi

      - name: Check for large files
        run: |
          LARGE_FILES=$(find . -type f -size +10M)
          if [ -n "$LARGE_FILES" ]; then
            echo "Large files detected (>10MB):"
            echo "$LARGE_FILES"
            exit 1
          fi

      - name: Check for secrets
        run: |
          if grep -r "password\|secret\|api_key\|token" --exclude-dir=.git --exclude="*.md"; then
            echo "Potential secrets detected in code"
            exit 1
          fi
```

### Task 7: Git Hooks (Local Validation)

Install pre-commit hooks for local validation:

**File**: `scripts/pre-commit-hook.sh`

```bash
#!/bin/bash
# Pre-commit hook for code quality checks

set -e

echo "Running pre-commit checks..."

# Check Python formatting
if command -v black &> /dev/null; then
    echo "Checking Python formatting with Black..."
    black --check examples/ || {
        echo "Black formatting failed. Run: black examples/"
        exit 1
    }
fi

# Check for large files
echo "Checking for large files..."
LARGE_FILES=$(git diff --cached --name-only | xargs du -h 2>/dev/null | awk '$1 ~ /M$/ && $1+0 > 10')
if [ -n "$LARGE_FILES" ]; then
    echo "Large files detected (>10MB):"
    echo "$LARGE_FILES"
    exit 1
fi

# Check for potential secrets
echo "Checking for potential secrets..."
if git diff --cached | grep -E "(password|secret|api_key|token)" | grep -v "# "; then
    echo "Warning: Potential secrets detected in changes"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Pre-commit checks passed!"
```

**Installation**:
```bash
./scripts/setup-git-hooks.sh
```

## Workflow Examples

### Example 1: Feature Development Flow

```bash
# 1. Create feature branch
git checkout -b feature/add-inference-api

# 2. Make changes
vim examples/ml_model.py

# 3. Check locally
./scripts/check-code-quality.sh

# 4. Commit changes
git add examples/ml_model.py
git commit -m "feat(inference): Add FastAPI inference endpoint"

# 5. Push to remote
git push origin feature/add-inference-api

# 6. Create PR on GitHub
# GitHub Actions automatically runs all checks

# 7. Address review comments
vim examples/ml_model.py
git add examples/ml_model.py
git commit -m "fix(inference): Address review comments"
git push origin feature/add-inference-api

# 8. Merge PR after approval
# Delete feature branch
git checkout main
git pull origin main
git branch -d feature/add-inference-api
```

### Example 2: Hotfix Flow

```bash
# 1. Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/fix-critical-bug

# 2. Make fix
vim examples/ml_model.py
git add examples/ml_model.py
git commit -m "fix: Resolve critical inference bug"

# 3. Push and create PR
git push origin hotfix/fix-critical-bug

# 4. Fast-track review and merge
# After merge, create a release
git checkout main
git pull origin main
git tag -a v1.0.1 -m "Hotfix: Critical inference bug"
git push origin v1.0.1
```

## Common GitHub Actions Patterns

### 1. Matrix Builds

Test across multiple versions:

```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11']
    os: [ubuntu-latest, macos-latest, windows-latest]
```

### 2. Caching Dependencies

Speed up builds:

```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

### 3. Conditional Steps

Run steps based on conditions:

```yaml
- name: Deploy to production
  if: github.ref == 'refs/heads/main'
  run: ./deploy.sh
```

### 4. Using Secrets

```yaml
- name: Deploy
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  run: ./deploy.sh
```

### 5. Artifacts

Save build artifacts:

```yaml
- name: Upload artifacts
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-results/
```

## Best Practices

### Git Workflow

1. **Branch Naming**:
   - `feature/short-description`
   - `bugfix/issue-123`
   - `hotfix/critical-fix`
   - `release/v1.2.0`

2. **Commit Guidelines**:
   - Use conventional commit format
   - Write descriptive messages
   - Keep commits atomic (one logical change)
   - Reference issues when applicable

3. **Pull Requests**:
   - Keep PRs small and focused
   - Write clear descriptions
   - Include testing instructions
   - Update documentation
   - Link related issues

4. **Code Review**:
   - Review promptly (within 24 hours)
   - Be constructive and specific
   - Test changes locally when needed
   - Approve only after all concerns addressed

### GitHub Actions

1. **Workflow Organization**:
   - Separate workflows by purpose
   - Use descriptive job names
   - Add comments for complex steps

2. **Performance**:
   - Cache dependencies
   - Use matrix builds efficiently
   - Fail fast when appropriate
   - Run expensive checks only when needed

3. **Security**:
   - Never commit secrets
   - Use GitHub Secrets for sensitive data
   - Limit workflow permissions
   - Review third-party actions

4. **Reliability**:
   - Use specific action versions (`@v4` not `@latest`)
   - Add timeout limits
   - Handle failures gracefully
   - Test workflows in feature branches

## Testing Your Workflows

### Local Testing

```bash
# Install act (GitHub Actions local runner)
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run workflows locally
act push
act pull_request
act -l  # List available workflows
```

### Workflow Debugging

```yaml
- name: Debug
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    echo "SHA: ${{ github.sha }}"
    echo "Actor: ${{ github.actor }}"
```

## Common Commands Reference

### Git Commands

```bash
# Branch management
git branch                          # List branches
git branch -a                       # List all branches (including remote)
git checkout -b feature/new-feature # Create and switch to branch
git branch -d feature/old-feature   # Delete local branch
git push origin --delete feature/x  # Delete remote branch

# Synchronization
git fetch origin                    # Fetch changes from remote
git pull origin main                # Pull changes from main
git push origin feature/x           # Push feature branch
git rebase main                     # Rebase feature on main

# History
git log --oneline --graph           # Visual commit history
git log --author="name"             # Commits by author
git show <commit-hash>              # Show commit details

# Undo changes
git reset --soft HEAD~1             # Undo last commit (keep changes)
git reset --hard HEAD~1             # Undo last commit (discard changes)
git revert <commit-hash>            # Create revert commit
git stash                           # Temporarily save changes
git stash pop                       # Restore stashed changes
```

### GitHub CLI Commands

```bash
# Install GitHub CLI
# Linux: curl -fsSL https://cli.github.com/install | bash
# macOS: brew install gh

# Authentication
gh auth login

# Pull requests
gh pr create                        # Create PR
gh pr list                          # List PRs
gh pr view 123                      # View PR details
gh pr checkout 123                  # Checkout PR locally
gh pr merge 123                     # Merge PR

# Issues
gh issue create                     # Create issue
gh issue list                       # List issues
gh issue close 123                  # Close issue

# Workflows
gh workflow list                    # List workflows
gh workflow run ci.yml              # Trigger workflow
gh run list                         # List workflow runs
gh run view 12345                   # View run details
```

## Exercises

### Exercise 1: Create Your First Workflow

1. Create a new workflow that:
   - Runs on push to any branch
   - Checks out code
   - Prints the current date
   - Lists all Python files

### Exercise 2: Add Code Quality Checks

1. Create a workflow that:
   - Runs flake8 on all Python files
   - Checks code formatting with black
   - Reports results as comments on PR

### Exercise 3: Implement Branch Protection

1. Configure branch protection rules on GitHub:
   - Require PR reviews (1 approver)
   - Require status checks to pass
   - Require branches to be up to date
   - No force pushes

### Exercise 4: Create a Release Workflow

1. Create a workflow that:
   - Triggers on tag creation (`v*`)
   - Builds artifacts
   - Creates a GitHub release
   - Uploads artifacts to release

### Exercise 5: Matrix Testing

1. Create a workflow that tests code against:
   - Python 3.9, 3.10, 3.11
   - Ubuntu and macOS
   - With and without optional dependencies

## Troubleshooting

### Issue: Workflow not triggering

**Symptoms**: Pushed code but workflow didn't run

**Solutions**:
1. Check workflow file syntax (YAML indentation)
2. Verify `on:` trigger configuration
3. Check branch name matches filter
4. Ensure workflow file is in `.github/workflows/`
5. Look for parse errors in GitHub Actions UI

### Issue: Permission denied

**Symptoms**: Workflow fails with permission errors

**Solutions**:
1. Check repository settings → Actions → Permissions
2. Add explicit permissions to workflow:
   ```yaml
   permissions:
     contents: read
     pull-requests: write
   ```

### Issue: Workflow takes too long

**Symptoms**: Builds timeout or take >10 minutes

**Solutions**:
1. Add dependency caching
2. Use matrix builds to parallelize
3. Skip unnecessary steps with conditions
4. Optimize Docker layer caching

### Issue: Can't access secrets

**Symptoms**: Secrets are empty in workflow

**Solutions**:
1. Verify secret is created in repository settings
2. Check secret name spelling (case-sensitive)
3. Secrets not available in PRs from forks
4. Use `${{ secrets.SECRET_NAME }}` syntax

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Git Documentation](https://git-scm.com/doc)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Awesome GitHub Actions](https://github.com/sdras/awesome-actions)

## Next Steps

After completing this exercise, you should:

1. ✅ Understand Git branching workflows
2. ✅ Be able to create pull requests
3. ✅ Write basic GitHub Actions workflows
4. ✅ Implement automated code quality checks
5. ✅ Know how to debug workflow issues

**Move on to**: Exercise 02 - Automated Testing for ML Code

## Summary

You've learned the fundamentals of Git workflows and GitHub Actions for CI/CD. These skills are essential for:
- Collaborating with teams effectively
- Maintaining code quality automatically
- Catching bugs before production
- Streamlining deployment processes
- Building reliable ML infrastructure

Practice these workflows regularly to build muscle memory and develop good habits that will serve you throughout your career in AI infrastructure engineering.
