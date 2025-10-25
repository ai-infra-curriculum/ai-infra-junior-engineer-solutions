# Exercise 03: Version Control with Git

## Overview

Master Git and GitHub workflows to collaborate effectively on code, track changes, and build a professional portfolio. This exercise covers Git fundamentals through advanced workflows used in professional software development.

## Learning Objectives

- ✅ Understand Git fundamentals (commits, branches, merges)
- ✅ Create and manage GitHub repositories
- ✅ Use branching strategies for professional development
- ✅ Handle merge conflicts confidently
- ✅ Implement Git hooks for automation
- ✅ Set up GitHub Actions for CI/CD
- ✅ Build a portfolio repository structure
- ✅ Follow industry-standard commit conventions

## Prerequisites

- Git installed (from Exercise 02)
- GitHub account created
- Terminal/command line basics

---

## Git Fundamentals

### What is Git?

Git is a **distributed version control system** that tracks changes in source code during software development. It enables:

- **Version History**: Track every change made to code
- **Collaboration**: Multiple developers working simultaneously
- **Branching**: Experiment without affecting main code
- **Backup**: Distributed copies prevent data loss

### Git vs. GitHub

- **Git**: Version control system (runs locally)
- **GitHub**: Cloud hosting platform for Git repositories (with collaboration features)

---

## Git Workflow Basics

### The Three States

Git has three main states for files:

1. **Working Directory**: Where you edit files
2. **Staging Area (Index)**: Files marked for commit
3. **Repository (.git directory)**: Committed snapshots

```
Working Directory → Staging Area → Repository
     (add)              (commit)
```

### Basic Commands

```bash
# Initialize repository
git init

# Check status
git status

# Add files to staging
git add file.txt
git add .  # Add all files

# Commit changes
git commit -m "Add feature X"

# View commit history
git log
git log --oneline --graph

# View changes
git diff  # Unstaged changes
git diff --staged  # Staged changes
```

---

## Setting Up Your Portfolio Repository

### Step 1: Create GitHub Repository

**On GitHub:**
1. Go to github.com
2. Click "New repository"
3. Name: `ai-infra-portfolio`
4. Description: "My AI Infrastructure Engineering Portfolio"
5. Public visibility
6. Add README
7. Add .gitignore (Python template)
8. Add MIT License
9. Click "Create repository"

### Step 2: Clone to Local

```bash
# Clone repository
git clone https://github.com/yourusername/ai-infra-portfolio.git
cd ai-infra-portfolio

# Verify remote
git remote -v
```

### Step 3: Create Portfolio Structure

```bash
# Create directory structure
mkdir -p projects/{docker,kubernetes,ml-serving,monitoring}
mkdir -p learning/{notes,tutorials}
mkdir -p docs

# Create README files
touch projects/README.md
touch learning/README.md
touch docs/README.md

# Commit structure
git add .
git commit -m "feat: add portfolio structure"
git push origin main
```

---

## Branching Strategies

### Why Branches?

Branches allow you to:
- Develop features independently
- Keep main branch stable
- Experiment without risk
- Organize work by feature/bugfix

### Common Branch Types

1. **main/master**: Production-ready code
2. **develop**: Integration branch
3. **feature/**: New features (`feature/add-login`)
4. **bugfix/**: Bug fixes (`bugfix/fix-api-error`)
5. **hotfix/**: Urgent production fixes

### Branch Commands

```bash
# List branches
git branch
git branch -a  # Include remote branches

# Create and switch to new branch
git checkout -b feature/new-project

# Switch branches
git checkout main

# Push branch to remote
git push origin feature/new-project

# Delete branch (local)
git branch -d feature/new-project

# Delete branch (remote)
git push origin --delete feature/new-project
```

### Example Workflow

```bash
# 1. Start from main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/docker-project

# 3. Make changes
echo "# Docker Project" > projects/docker/README.md
git add projects/docker/README.md
git commit -m "feat(docker): add project README"

# 4. Push to GitHub
git push origin feature/docker-project

# 5. Create Pull Request on GitHub
# (via web interface)

# 6. After merge, cleanup
git checkout main
git pull origin main
git branch -d feature/docker-project
```

---

## Commit Message Conventions

### Conventional Commits Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```bash
# Good commit messages
git commit -m "feat(auth): add JWT authentication"
git commit -m "fix(api): handle null response from service"
git commit -m "docs: update installation instructions"
git commit -m "refactor(utils): extract helper functions"

# Bad commit messages (avoid these)
git commit -m "changes"
git commit -m "fix stuff"
git commit -m "WIP"
git commit -m "asdfasdf"
```

### Multi-line Commits

```bash
git commit -m "feat(ml-serving): add model serving API

- Implement FastAPI endpoint for predictions
- Add model loading and caching
- Include input validation
- Add comprehensive logging

Closes #123"
```

---

## Handling Merge Conflicts

### What Causes Conflicts?

- Two branches modify the same lines
- One branch deletes while another modifies
- Binary files changed in both branches

### Resolving Conflicts

```bash
# 1. Attempt merge
git checkout main
git merge feature/new-feature

# 2. If conflicts occur, Git will notify you
# CONFLICT (content): Merge conflict in file.py

# 3. Check status
git status

# 4. Open conflicted files
code file.py

# 5. Look for conflict markers
<<<<<<< HEAD
current_branch_code()
=======
merging_branch_code()
>>>>>>> feature/new-feature

# 6. Resolve conflicts (keep one, both, or modify)
# Remove conflict markers
both_functions_combined()

# 7. Stage resolved files
git add file.py

# 8. Complete merge
git commit -m "merge: resolve conflicts in feature/new-feature"
```

### Tips for Avoiding Conflicts

- Pull frequently
- Keep feature branches short-lived
- Communicate with team
- Use `.gitignore` for generated files

---

## Git Hooks

Git hooks are scripts that run automatically on certain Git events.

### Common Hooks

- **pre-commit**: Run before commit (linting, formatting)
- **commit-msg**: Validate commit messages
- **pre-push**: Run before push (tests)
- **post-commit**: Run after commit (notifications)

### Example: Pre-commit Hook for Python

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Run Python linter
echo "Running flake8..."
flake8 . --exclude=venv,__pycache__

if [ $? -ne 0 ]; then
    echo "Linting failed! Fix errors before committing."
    exit 1
fi

# Run Python formatter
echo "Running black..."
black --check .

if [ $? -ne 0 ]; then
    echo "Code formatting issues! Run 'black .' to fix."
    exit 1
fi

echo "Pre-commit checks passed!"
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

### Using pre-commit Framework

Better approach: Use the `pre-commit` Python package.

**Install:**
```bash
pip install pre-commit
```

**Create `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

**Install hooks:**
```bash
pre-commit install
```

Now hooks run automatically on every commit!

---

## GitHub Actions (CI/CD)

Automate testing, linting, and deployment with GitHub Actions.

### Example: Python CI Workflow

Create `.github/workflows/python-ci.yml`:

```yaml
name: Python CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest black
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Check formatting with black
      run: |
        black --check .

    - name: Run tests
      run: |
        pytest tests/ -v
```

This workflow:
- Runs on push to main/develop and on pull requests
- Tests on Python 3.10 and 3.11
- Runs linting (flake8)
- Checks code formatting (black)
- Runs tests (pytest)

---

## .gitignore Patterns

### What to Ignore

**Never commit:**
- Secrets and credentials
- Dependencies (node_modules, venv)
- Build artifacts
- IDE settings (personal)
- OS files (.DS_Store)

### Python .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local
*.key
*.pem

# OS
.DS_Store
Thumbs.db

# Project-specific
*.log
data/
models/*.h5
models/*.pkl
```

### ML-Specific Ignores

```gitignore
# Large files
*.h5
*.pkl
*.joblib
*.pt
*.pth
*.onnx

# Data
data/raw/
data/processed/
*.csv
*.parquet

# Notebooks
.ipynb_checkpoints/
*.ipynb

# MLflow
mlruns/
mlartifacts/
```

---

## Collaboration Workflow

### Fork and Pull Request

**For open source contributions:**

1. **Fork** repository on GitHub
2. **Clone** your fork
3. **Create branch** for changes
4. **Make changes** and commit
5. **Push** to your fork
6. **Create Pull Request** to original repo

### Code Review Best Practices

**When reviewing:**
- Check for bugs and logic errors
- Verify tests exist and pass
- Check code style and formatting
- Suggest improvements politely
- Approve when satisfied

**When receiving review:**
- Address all comments
- Ask questions if unclear
- Don't take criticism personally
- Thank reviewers

---

## Portfolio Repository Structure

### Recommended Layout

```
ai-infra-portfolio/
├── README.md (Professional overview)
├── projects/
│   ├── 01-docker-ml-app/
│   │   ├── README.md
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── 02-kubernetes-deploy/
│   ├── 03-ml-api/
│   └── 04-terraform-infra/
├── learning/
│   ├── notes/
│   │   ├── docker.md
│   │   ├── kubernetes.md
│   │   └── python.md
│   └── tutorials/
│       └── completed-courses.md
├── docs/
│   ├── resume.md
│   ├── skills.md
│   └── certifications.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── LICENSE
└── CONTRIBUTING.md
```

### Professional README Template

See [examples/portfolio-readme-template.md](examples/portfolio-readme-template.md)

---

## Deliverables

By the end of this exercise, you should have:

✅ **Portfolio Repository on GitHub**
- Public repository with professional structure
- Clear README with introduction
- Organized projects folder
- Learning notes section

✅ **Git Workflow Mastery**
- Comfortable with branches and merges
- Can handle merge conflicts
- Follows commit conventions

✅ **Pre-commit Hooks Configured**
- `.pre-commit-config.yaml` created
- Hooks installed and working
- Automatic code quality checks

✅ **GitHub Actions Setup**
- CI/CD workflow for Python projects
- Automated testing on pull requests
- Badge in README showing build status

✅ **.gitignore Configured**
- Appropriate patterns for Python/ML projects
- No sensitive data committed
- Clean repository

---

## Validation

Run the validation script:

```bash
python scripts/validate_git_setup.py
```

Expected output:
```
✅ Git configured with user name and email
✅ Portfolio repository exists on GitHub
✅ Repository has proper structure
✅ .gitignore file present
✅ Pre-commit hooks installed
✅ GitHub Actions workflow present
✅ At least one feature branch created
✅ README.md is professional and complete

🎉 Exercise 03 Complete!
```

---

## Tips for Success

💡 **Commit Often**: Small, frequent commits are better than large, infrequent ones

💡 **Write Good Messages**: Future you will thank present you

💡 **Branch Fearlessly**: Branches are cheap, experiment freely

💡 **Pull Before Push**: Always pull latest changes before pushing

💡 **Review Before Commit**: Use `git diff` to check changes

---

## Common Pitfalls

❌ **Committing Secrets**: Never commit API keys, passwords, or credentials

❌ **Large Binary Files**: Avoid committing large model files (use Git LFS or separate storage)

❌ **Working Directly on Main**: Always use feature branches

❌ **Force Pushing**: Avoid `git push --force` unless absolutely necessary

❌ **Vague Commit Messages**: "fix stuff" tells nobody anything

---

## Next Steps

After completing this exercise:

1. **Module 002: Python Programming** - Build your coding skills
2. Contribute to an open-source project
3. Keep your portfolio updated with new projects
4. Practice Git workflows daily

---

## Resources

### Official Documentation
- [Git Book](https://git-scm.com/book/en/v2)
- [GitHub Docs](https://docs.github.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)

### Interactive Learning
- [Learn Git Branching](https://learngitbranching.js.org/)
- [GitHub Skills](https://skills.github.com/)
- [Git Immersion](https://gitimmersion.com/)

### Cheat Sheets
- [GitHub Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Atlassian Git Cheat Sheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)

---

**Congratulations! You've mastered Git fundamentals. Time to build amazing projects! 🚀**
