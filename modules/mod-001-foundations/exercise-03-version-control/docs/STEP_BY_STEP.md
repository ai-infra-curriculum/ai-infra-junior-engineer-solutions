# Step-by-Step Implementation Guide: Version Control with Git

## Overview

Master Git and GitHub workflows for ML infrastructure! Learn branching strategies, collaborative workflows, pull requests, and best practices for infrastructure code.

**Time**: 2-3 hours | **Difficulty**: Beginner to Intermediate

---

## Learning Objectives

✅ Understand Git fundamentals
✅ Master branching strategies
✅ Collaborate with pull requests
✅ Implement Git workflows
✅ Manage merge conflicts
✅ Use Git hooks
✅ Follow best practices for commits

---

## Git Fundamentals

### Basic Concepts

```
Working Directory → Staging Area → Local Repository → Remote Repository
      |                  |               |                    |
   (untracked)      (git add)      (git commit)         (git push)
```

### Essential Commands

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit"

# Clone existing repository
git clone https://github.com/username/repo.git
cd repo

# Check status
git status

# View history
git log
git log --oneline --graph --all

# View changes
git diff
git diff --staged
git diff HEAD~1
```

---

## Phase 1: Repository Setup

### Create New Repository

```bash
# Local setup
mkdir ml-infrastructure-project
cd ml-infrastructure-project
git init

# Create initial structure
mkdir -p {src,tests,docs,scripts,k8s}
touch README.md .gitignore requirements.txt

# Initial commit
git add .
git commit -m "Initial project structure"
```

### Connect to GitHub

```bash
# Create repo on GitHub (via CLI)
gh repo create ml-infrastructure-project \
  --public \
  --description "ML Infrastructure Project" \
  --source=. \
  --remote=origin \
  --push

# Or add remote manually
git remote add origin git@github.com:username/ml-infrastructure-project.git
git branch -M main
git push -u origin main
```

### .gitignore for ML Projects

```bash
# .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Data files
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
*.csv
*.parquet
*.h5
*.hdf5

# Model files
models/*.pth
models/*.pkl
models/*.h5
models/*.ckpt
!models/.gitkeep
*.onnx
*.pb

# Logs
*.log
logs/
mlruns/

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Docker
.dockerignore

# Kubernetes secrets
k8s/secrets/*.yaml
!k8s/secrets/example.yaml

# Terraform
*.tfstate
*.tfstate.*
.terraform/
```

---

## Phase 2: Branching Strategies

### Git Flow (Traditional)

```
main (production)
  ↓
develop (integration)
  ↓
feature/* (new features)
release/* (release prep)
hotfix/* (urgent fixes)
```

```bash
# Create develop branch
git checkout -b develop
git push -u origin develop

# Feature branch
git checkout develop
git checkout -b feature/model-serving
# ... work on feature ...
git add .
git commit -m "Add model serving endpoint"
git push -u origin feature/model-serving

# Merge to develop
git checkout develop
git merge feature/model-serving
git push origin develop

# Release branch
git checkout -b release/v1.0.0 develop
# ... final testing, version bumps ...
git commit -m "Bump version to 1.0.0"

# Merge to main and develop
git checkout main
git merge release/v1.0.0
git tag v1.0.0
git push origin main --tags

git checkout develop
git merge release/v1.0.0
git push origin develop
```

### GitHub Flow (Simplified)

```
main (production)
  ↓
feature/* (all changes)
```

```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/add-monitoring

# Make changes
git add .
git commit -m "Add Prometheus monitoring"
git push -u origin feature/add-monitoring

# Open pull request on GitHub
gh pr create \
  --title "Add Prometheus monitoring" \
  --body "Implements monitoring with Prometheus and Grafana"

# After approval, merge via GitHub UI
# Delete branch
git checkout main
git pull origin main
git branch -d feature/add-monitoring
```

### Trunk-Based Development (Modern)

```
main (always deployable)
  ↓
short-lived feature branches (< 1 day)
```

```bash
# Small, frequent commits to main
git checkout main
git pull origin main
git checkout -b add-health-check

# Quick change
# ... make small change ...
git add .
git commit -m "Add health check endpoint"
git push -u origin add-health-check

# Open PR and merge same day
gh pr create --fill
# Get quick review, merge, delete branch
```

---

## Phase 3: Commit Best Practices

### Conventional Commits

```bash
# Format
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

# Types
feat:     New feature
fix:      Bug fix
docs:     Documentation changes
style:    Formatting, missing semicolons, etc.
refactor: Code restructuring
perf:     Performance improvements
test:     Adding tests
chore:    Maintenance tasks
ci:       CI/CD changes

# Examples
git commit -m "feat(api): add model versioning endpoint"
git commit -m "fix(k8s): correct deployment replicas count"
git commit -m "docs: update README with setup instructions"
git commit -m "perf(inference): optimize batch processing"
git commit -m "ci: add Docker build workflow"
```

### Atomic Commits

```bash
# Bad: Multiple unrelated changes
git add .
git commit -m "Fix bug and add feature"

# Good: Separate commits
# Commit 1: Bug fix
git add src/api/endpoints.py
git commit -m "fix(api): handle null predictions gracefully"

# Commit 2: New feature
git add src/models/ensemble.py tests/test_ensemble.py
git commit -m "feat(models): add ensemble prediction support"
```

### Interactive Staging

```bash
# Stage parts of a file
git add -p src/api/app.py

# Interactive rebase to clean up history
git rebase -i HEAD~3

# Options:
# pick   = use commit
# reword = use commit, edit message
# edit   = use commit, stop for amending
# squash = merge with previous commit
# fixup  = like squash, discard commit message
# drop   = remove commit
```

---

## Phase 4: Pull Request Workflow

### Create Pull Request

```bash
# Create feature branch
git checkout -b feature/gpu-scheduling
# ... make changes ...
git add .
git commit -m "feat(k8s): add GPU node scheduling"
git push -u origin feature/gpu-scheduling

# Create PR with gh CLI
gh pr create \
  --title "Add GPU node scheduling" \
  --body "$(cat <<EOF
## Summary
Implements GPU node scheduling for ML workloads.

## Changes
- Add GPU node selectors to deployment
- Configure resource limits for GPU pods
- Update documentation

## Testing
- [ ] Tested on minikube with GPU
- [ ] Verified resource allocation
- [ ] Updated integration tests

## Related Issues
Closes #42
EOF
)" \
  --assignee @me \
  --label enhancement
```

### PR Template

```markdown
# .github/pull_request_template.md
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally

## Screenshots (if applicable)

## Related Issues
Closes #(issue number)
```

### Code Review Process

```bash
# Reviewer: Check out PR locally
gh pr checkout 123

# Run tests
pytest tests/

# Review code
git diff main...feature/gpu-scheduling

# Request changes via GitHub UI or:
gh pr review 123 --request-changes --body "Please add error handling"

# Author: Address feedback
git add .
git commit -m "Add error handling for GPU allocation"
git push

# Reviewer: Approve
gh pr review 123 --approve --body "LGTM!"

# Merge
gh pr merge 123 --squash --delete-branch
```

---

## Phase 5: Merge Conflicts

### Resolve Conflicts

```bash
# Scenario: Conflict during merge
git checkout main
git pull origin main
git checkout feature/new-api
git merge main

# Output:
# Auto-merging src/api/app.py
# CONFLICT (content): Merge conflict in src/api/app.py
# Automatic merge failed; fix conflicts and then commit the result.

# Check conflicted files
git status

# Open file and resolve
# <<<<<<< HEAD (current branch)
# Your changes
# =======
# Incoming changes
# >>>>>>> main

# Example conflict in src/api/app.py:
"""
<<<<<<< HEAD
@app.get("/predict")
async def predict(data: InputData):
    model = load_model("v2")
=======
@app.post("/predict")
async def predict(request: PredictRequest):
    model = get_model("latest")
>>>>>>> main
    result = model.predict(data)
    return result
"""

# Resolved:
"""
@app.post("/predict")
async def predict(request: PredictRequest):
    model = load_model("v2")
    result = model.predict(request.data)
    return result
"""

# Mark as resolved
git add src/api/app.py
git commit -m "Merge main into feature/new-api, resolve conflicts"
```

### Rebase Instead of Merge

```bash
# Rebase feature branch onto main
git checkout feature/new-api
git rebase main

# If conflicts occur
git status
# ... resolve conflicts ...
git add .
git rebase --continue

# Abort if needed
git rebase --abort

# Force push (rebase rewrites history)
git push --force-with-lease origin feature/new-api
```

---

## Phase 6: Git Hooks

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running pre-commit checks..."

# Run linters
echo "Running flake8..."
flake8 src/ tests/
if [ $? -ne 0 ]; then
    echo "❌ Flake8 failed. Commit aborted."
    exit 1
fi

# Run tests
echo "Running tests..."
pytest tests/
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Commit aborted."
    exit 1
fi

# Check for secrets
echo "Checking for secrets..."
git diff --cached --name-only | xargs grep -l "API_KEY\|SECRET\|PASSWORD" && {
    echo "❌ Possible secrets detected. Commit aborted."
    exit 1
}

echo "✅ Pre-commit checks passed!"
exit 0

# Make executable
chmod +x .git/hooks/pre-commit
```

### Pre-commit Framework

```bash
# Install pre-commit
pip install pre-commit

# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Phase 7: Advanced Git

### Stashing Changes

```bash
# Save work in progress
git stash

# List stashes
git stash list

# Apply stash
git stash apply

# Apply and remove stash
git stash pop

# Stash with message
git stash save "WIP: implementing GPU support"

# Apply specific stash
git stash apply stash@{1}

# Drop stash
git stash drop stash@{0}
```

### Cherry-picking

```bash
# Apply specific commit from another branch
git cherry-pick abc123

# Cherry-pick multiple commits
git cherry-pick abc123 def456

# Cherry-pick without committing
git cherry-pick -n abc123
```

### Bisect (Find Bug Introduction)

```bash
# Start bisect
git bisect start

# Mark current as bad
git bisect bad

# Mark known good commit
git bisect good v1.0.0

# Test each commit
# If test passes:
git bisect good
# If test fails:
git bisect bad

# Git will binary search through history
# Once found:
git bisect reset
```

### Submodules

```bash
# Add submodule
git submodule add https://github.com/org/shared-lib.git lib/shared

# Clone repo with submodules
git clone --recurse-submodules https://github.com/org/main-repo.git

# Update submodules
git submodule update --init --recursive

# Update to latest
git submodule update --remote
```

---

## Phase 8: GitHub Collaboration

### Issues and Project Management

```bash
# Create issue
gh issue create \
  --title "Add model monitoring dashboard" \
  --body "Need Grafana dashboard for model metrics" \
  --label enhancement \
  --assignee @me

# Link PR to issue
gh pr create --title "Add monitoring dashboard" --body "Closes #42"

# Comment on issue
gh issue comment 42 --body "Working on this"

# Close issue
gh issue close 42
```

### GitHub Actions Integration

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install linters
        run: pip install black flake8

      - name: Run black
        run: black --check src/ tests/

      - name: Run flake8
        run: flake8 src/ tests/
```

### Protected Branches

```bash
# Set via GitHub UI: Settings → Branches → Add rule
# Or via gh CLI
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field "required_status_checks[strict]=true" \
  --field "required_status_checks[contexts][]=ci/test" \
  --field "required_pull_request_reviews[required_approving_review_count]=1" \
  --field "enforce_admins=true"
```

---

## Phase 9: Monorepo Management

### Directory Structure

```
ml-infrastructure/
├── services/
│   ├── api/
│   ├── training/
│   └── inference/
├── libraries/
│   ├── ml-utils/
│   └── data-processing/
├── infrastructure/
│   ├── terraform/
│   └── kubernetes/
└── tools/
    └── scripts/
```

### Sparse Checkout

```bash
# Clone only specific directories
git clone --filter=blob:none --sparse https://github.com/org/monorepo.git
cd monorepo
git sparse-checkout set services/api infrastructure/kubernetes
```

### CODEOWNERS

```bash
# .github/CODEOWNERS
# Global owners
* @ml-team

# API team owns services
/services/api/ @api-team

# Infrastructure team owns K8s configs
/infrastructure/kubernetes/ @infra-team

# ML team owns models
/libraries/ml-utils/ @ml-team
```

---

## Best Practices

✅ Commit early and often
✅ Write clear, descriptive commit messages
✅ Use branches for all changes
✅ Keep commits atomic and focused
✅ Review your changes before committing
✅ Pull before you push
✅ Use .gitignore to exclude generated files
✅ Never commit secrets or credentials
✅ Tag releases with semantic versioning
✅ Document your workflow

---

## Common Workflows

### Feature Development

```bash
# 1. Update main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/add-caching

# 3. Make changes
# ... code ...
git add .
git commit -m "feat: add Redis caching layer"

# 4. Push and create PR
git push -u origin feature/add-caching
gh pr create --fill

# 5. Address review feedback
# ... make changes ...
git add .
git commit -m "Address review feedback"
git push

# 6. After merge, clean up
git checkout main
git pull origin main
git branch -d feature/add-caching
```

### Hotfix

```bash
# 1. Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/memory-leak

# 2. Fix issue
# ... fix bug ...
git add .
git commit -m "fix: resolve memory leak in inference loop"

# 3. Create PR and fast-track review
git push -u origin hotfix/memory-leak
gh pr create --title "HOTFIX: Memory leak" --label urgent

# 4. After merge, deploy immediately
```

---

## Troubleshooting

### Undo Last Commit (Keep Changes)

```bash
git reset --soft HEAD~1
```

### Undo Last Commit (Discard Changes)

```bash
git reset --hard HEAD~1
```

### Recover Deleted Branch

```bash
git reflog
git checkout -b recovered-branch abc123
```

### Remove File from Git (Keep Local)

```bash
git rm --cached sensitive-file.txt
echo "sensitive-file.txt" >> .gitignore
git commit -m "Remove sensitive file from tracking"
```

---

**Version Control with Git mastered!** 🚀

**Congratulations!** You've completed the entire Foundations module!

**Next Module**: Python Programming (mod-002)
