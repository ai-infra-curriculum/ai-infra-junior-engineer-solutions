# Exercise 01: Creating Your First ML Project Repository - Implementation Guide

## Overview

This implementation guide walks you through creating a proper Git repository for an ML inference API project from scratch. You'll learn Git fundamentals while building a production-ready project structure.

**Estimated Time**: 60-90 minutes
**Difficulty**: Beginner
**Prerequisites**: Git 2.x+, Python 3.11+, text editor

## What You'll Build

A complete ML inference API repository with:
- ✅ Proper Git initialization and configuration
- ✅ Professional `.gitignore` for ML projects
- ✅ `.gitattributes` for binary file handling
- ✅ Atomic commits with clear messages
- ✅ FastAPI application structure
- ✅ Model serving code (PyTorch)
- ✅ Comprehensive documentation

---

## Part 1: Initial Setup and Configuration

### Step 1.1: Configure Git Identity

Before creating commits, Git needs to know who you are.

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use your GitHub email if you have one)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list | grep user
```

**Expected Output:**
```
user.name=Your Name
user.email=your.email@example.com
```

**Why This Matters:**
- Every commit is signed with author information
- GitHub/GitLab use email to link commits to your account
- Teams use this to track who made changes
- Required for commit attribution in open source

**Pro Tip:** Use the same email as your GitHub account to ensure commits are linked to your profile.

---

### Step 1.2: Create Project Directory and Initialize Repository

```bash
# Create project directory
mkdir ml-inference-api
cd ml-inference-api

# Initialize Git repository
git init

# Verify .git directory was created
ls -la
```

**Expected Output:**
```
Initialized empty Git repository in /path/to/ml-inference-api/.git/
```

You should see a `.git/` directory (hidden folder containing all Git data).

**Understanding git init:**
- Creates `.git/` subdirectory with Git metadata
- Establishes this folder as a Git repository
- Sets up default branch (usually `main` or `master`)
- Does NOT create any commits yet

```bash
# Check repository status
git status
```

**Output:**
```
On branch main

No commits yet

nothing to commit (create/copy files and use "git add" to track)
```

---

### Step 1.3: Set Git Behavior Preferences

Configure helpful Git settings for better experience:

```bash
# Set default branch name to 'main'
git config --global init.defaultBranch main

# Enable colored output for better readability
git config --global color.ui auto

# Set default editor (use your preferred editor)
git config --global core.editor "code --wait"  # VS Code
# OR
git config --global core.editor "vim"          # Vim
# OR
git config --global core.editor "nano"         # Nano

# Show branch information in prompts
git config --global core.status auto
```

**Verify All Settings:**
```bash
git config --list --global
```

---

## Part 2: Create .gitignore File

### Step 2.1: Understand What to Ignore

ML projects generate many files that shouldn't be tracked:
- Python bytecode (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `.venv/`)
- IDE files (`.vscode/`, `.idea/`)
- Model files (large binaries)
- Data files (training datasets)
- Environment variables (`.env`)
- Logs and temporary files

### Step 2.2: Create Comprehensive .gitignore

```bash
# Create .gitignore file
cat > .gitignore << 'EOF'
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

# Virtual Environments
venv/
ENV/
env/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# ML Models and Data
models/*.pth
models/*.pt
models/*.onnx
models/*.h5
data/raw/
data/processed/
*.pkl
*.pickle

# Environment Variables
.env
.env.local
.env.*.local

# Logs
*.log
logs/
*.out

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation
docs/_build/
EOF
```

**Verify File Creation:**
```bash
cat .gitignore
ls -la | grep gitignore
```

---

### Step 2.3: Stage and Commit .gitignore

```bash
# Check status
git status
```

**Output:**
```
On branch main
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .gitignore

nothing added to commit but untracked files present
```

```bash
# Stage .gitignore
git add .gitignore

# Check status again
git status
```

**Output:**
```
On branch main
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   .gitignore
```

**Understanding the Three States:**
1. **Untracked**: File exists but Git doesn't track it
2. **Staged**: File changes added to staging area (ready to commit)
3. **Committed**: Changes saved to Git history

```bash
# Create first commit
git commit -m "Initial commit: Add comprehensive .gitignore for ML project

- Python bytecode and virtual environments
- IDE configuration files
- ML models and datasets (to be tracked with Git LFS)
- Environment variables and secrets
- Logs and temporary files

Rationale: Prevent committing generated files, secrets, and large binaries
to keep repository clean and secure."
```

**Understanding Commit Messages:**
- **Subject line**: Brief summary (50 chars or less)
- **Blank line**: Separates subject from body
- **Body**: Detailed explanation of WHY (optional but recommended)
- **Format**: Use imperative mood ("Add" not "Added" or "Adds")

---

## Part 3: Create Project Structure

### Step 3.1: Create Directory Structure

```bash
# Create all project directories
mkdir -p src/{api,models,preprocessing,utils}
mkdir -p configs
mkdir -p tests
mkdir -p scripts
mkdir -p docs
mkdir -p data/{raw,processed}
mkdir -p logs
mkdir -p models

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/api/__init__.py
touch src/models/__init__.py
touch src/preprocessing/__init__.py
touch src/utils/__init__.py
```

**Directory Structure:**
```
ml-inference-api/
├── src/               # Source code
│   ├── api/          # FastAPI application
│   ├── models/       # Model loading and inference
│   ├── preprocessing/ # Image preprocessing
│   └── utils/        # Utilities (logging, config)
├── configs/          # Configuration files (YAML)
├── tests/            # Test files
├── scripts/          # Utility scripts (setup, deployment)
├── docs/             # Documentation
├── data/             # Data directories (empty in repo)
│   ├── raw/         # Raw data
│   └── processed/   # Processed data
├── models/           # Model files (tracked with Git LFS)
└── logs/             # Log files
```

### Step 3.2: Create README.md

```bash
cat > README.md << 'EOF'
# ML Inference API

A production-ready REST API for serving image classification predictions using PyTorch models.

## Features

- 🚀 FastAPI-based REST API
- 🔥 PyTorch model inference
- 📊 Confidence score reporting
- 📝 Prediction logging
- 🔄 Multiple model version support
- 🐳 Docker containerization ready

## Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- FastAPI

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Start development server
uvicorn src.api.main:app --reload --port 8000
```

API will be available at: http://localhost:8000

### API Documentation

Interactive API docs: http://localhost:8000/docs

## Project Structure

See [docs/architecture.md](docs/architecture.md) for detailed architecture.

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your settings
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
EOF
```

**Commit README:**
```bash
git add README.md
git commit -m "Add README with project overview and quick start guide

Includes:
- Project description and features
- Installation instructions
- Quick start guide
- Project structure overview
- Testing and contribution guidelines"
```

---

### Step 3.3: Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# ML Framework
torch==2.1.0
torchvision==0.16.0
pillow==10.1.0

# Data Processing
numpy==1.26.2
pandas==2.1.3

# API Utilities
python-multipart==0.0.6
aiofiles==23.2.1

# Configuration
pyyaml==6.0.1
python-dotenv==1.0.0

# Logging and Monitoring
loguru==0.7.2
prometheus-client==0.19.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2

# Development
black==23.12.0
flake8==6.1.0
mypy==1.7.1
isort==5.13.0
EOF
```

**Commit Dependencies:**
```bash
git add requirements.txt
git commit -m "Add project dependencies with pinned versions

Dependencies include:
- FastAPI and Uvicorn for REST API
- PyTorch and torchvision for ML inference
- Testing tools (pytest, coverage)
- Development tools (black, flake8, mypy)

All versions pinned for reproducibility."
```

---

## Part 4: Create .gitattributes

### Step 4.1: Configure Git LFS for Large Files

`.gitattributes` tells Git how to handle different file types.

```bash
cat > .gitattributes << 'EOF'
# Text files - normalize line endings
*.py text eol=lf
*.md text eol=lf
*.txt text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
*.json text eol=lf
*.sh text eol=lf

# Binary files
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.pdf binary

# ML Model files (mark for Git LFS)
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text

# Data files (mark for Git LFS)
*.parquet filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.feather filter=lfs diff=lfs merge=lfs -text
EOF
```

**Understanding .gitattributes:**
- `text eol=lf`: Ensures consistent line endings (Unix-style)
- `binary`: Treats file as binary (no line ending conversion)
- `filter=lfs`: Uses Git LFS for large files (if installed)

**Commit .gitattributes:**
```bash
git add .gitattributes
git commit -m "Add .gitattributes for file type handling

Configure:
- Line ending normalization for text files (LF)
- Binary handling for images and PDFs
- Git LFS configuration for model files
- Git LFS configuration for large data files

Ensures consistent behavior across platforms."
```

---

## Part 5: Create .env.example Template

### Step 5.1: Create Environment Template

```bash
cat > .env.example << 'EOF'
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Model Configuration
MODEL_PATH=models/resnet50.pth
MODEL_NAME=resnet50
MODEL_VERSION=1.0.0
DEVICE=cpu  # or 'cuda' for GPU

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
EOF
```

**Why .env.example:**
- Provides template for required environment variables
- Safe to commit (no actual secrets)
- Developers copy to `.env` and fill in real values
- `.env` is in `.gitignore` (never committed)

**Commit Environment Template:**
```bash
git add .env.example
git commit -m "Add environment variable template

Provides configuration template for:
- API server settings
- Model configuration
- Logging preferences
- Monitoring settings

Developers should copy to .env and customize."
```

---

## Part 6: Verify Repository State

### Step 6.1: Review Commit History

```bash
# View commit history
git log --oneline
```

**Expected Output:**
```
a1b2c3d Add environment variable template
e4f5g6h Add .gitattributes for file type handling
i7j8k9l Add project dependencies with pinned versions
m0n1o2p Add README with project overview and quick start guide
q3r4s5t Initial commit: Add comprehensive .gitignore for ML project
```

**Detailed History:**
```bash
# View detailed history
git log --stat
```

---

### Step 6.2: Check Repository Status

```bash
# Verify clean working directory
git status
```

**Expected Output:**
```
On branch main
nothing to commit, working tree clean
```

**This means:**
- All changes are committed
- No untracked files
- No uncommitted changes
- Repository is in clean state

---

### Step 6.3: Inspect Files Tracked by Git

```bash
# List all tracked files
git ls-files
```

**Expected Output:**
```
.env.example
.gitattributes
.gitignore
README.md
requirements.txt
```

---

## Part 7: Understanding Git's Three-State Workflow

### Step 7.1: Working Directory vs Staging Area vs Repository

**The Three States:**

1. **Working Directory** (Modified)
   - Files you're currently editing
   - Changes not yet staged
   - Command: `git status` shows "Changes not staged"

2. **Staging Area** (Staged)
   - Changes marked for next commit
   - Preview of next commit
   - Command: `git add` moves files here

3. **Repository** (Committed)
   - Permanently stored in Git history
   - Immutable (mostly)
   - Command: `git commit` moves staged changes here

### Step 7.2: Practical Example

```bash
# Create a test file
echo "Test content" > test.txt

# Check status (untracked)
git status
# Shows: Untracked files: test.txt

# Stage the file
git add test.txt

# Check status (staged)
git status
# Shows: Changes to be committed: new file: test.txt

# Modify the file before committing
echo "More content" >> test.txt

# Check status (both staged and modified!)
git status
# Shows:
#   Changes to be committed: new file: test.txt
#   Changes not staged: modified: test.txt

# Stage the new changes
git add test.txt

# Commit
git commit -m "Add test file"

# Remove test file (this was just a demo)
git rm test.txt
git commit -m "Remove test file"
```

---

## Part 8: Best Practices for Commit Messages

### Step 8.1: Commit Message Format

**Structure:**
```
Short summary (50 chars or less)

Detailed explanation of what and why (optional)
- Can use bullet points
- Wrap at 72 characters
- Blank line between summary and body
```

**Good Example:**
```
Add user authentication middleware

Implement JWT-based authentication for API endpoints:
- Add authentication middleware to FastAPI
- Create token generation and validation functions
- Add login and refresh token endpoints
- Include comprehensive unit tests

Addresses security requirement SR-001.
```

**Bad Examples:**
```
# Too vague
"Update files"

# Too technical
"Fix bug in line 42"

# Past tense (use imperative)
"Added authentication"

# Too long subject
"Add user authentication middleware with JWT tokens and refresh token support including comprehensive testing"
```

### Step 8.2: When to Commit

**Commit When:**
- ✅ Feature is complete and tested
- ✅ Code compiles/runs without errors
- ✅ Logical unit of work is done
- ✅ Tests pass
- ✅ Before switching tasks

**Don't Commit:**
- ❌ Broken/non-working code
- ❌ Half-finished features
- ❌ Debug print statements
- ❌ Commented-out code
- ❌ Everything at once (make atomic commits)

---

## Verification and Testing

### Checklist

- [ ] Git repository initialized successfully
- [ ] User name and email configured
- [ ] .gitignore file created and committed
- [ ] .gitattributes file created and committed
- [ ] README.md exists with project information
- [ ] requirements.txt lists all dependencies
- [ ] .env.example provides configuration template
- [ ] At least 5 commits with clear messages
- [ ] `git status` shows clean working tree
- [ ] `git log` shows commit history

### Validation Commands

```bash
# Verify configuration
git config user.name
git config user.email

# Verify repository
git status
git log --oneline
git ls-files

# Verify all expected files exist
ls -la

# Count commits (should be 5+)
git rev-list --count HEAD
```

---

## Common Issues and Solutions

### Issue 1: "Author identity unknown"

**Error:**
```
*** Please tell me who you are.
```

**Solution:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

### Issue 2: "File still shows in git status after adding to .gitignore"

**Cause:** File was already tracked by Git before adding to .gitignore

**Solution:**
```bash
# Remove from Git tracking (keeps file locally)
git rm --cached filename

# Commit the removal
git commit -m "Stop tracking filename"
```

---

### Issue 3: "Accidentally committed .env file with secrets"

**Solution:**
```bash
# Remove from repository
git rm --cached .env

# Add to .gitignore if not already there
echo ".env" >> .gitignore

# Commit the changes
git add .gitignore
git commit -m "Stop tracking .env file"

# IMPORTANT: Rotate any exposed secrets immediately!
```

---

### Issue 4: "Wrong commit message"

**Solution (if not pushed yet):**
```bash
# Amend last commit message
git commit --amend -m "New correct message"
```

---

## Next Steps

After completing this exercise:

1. **Explore Git History:**
   ```bash
   git log --graph --oneline --all
   git show HEAD
   git diff HEAD~1
   ```

2. **Practice More Commits:**
   - Add actual source code files
   - Make incremental commits
   - Experiment with `git diff` before committing

3. **Learn Branching:**
   - Move to Exercise 02: Commits and History
   - Then Exercise 03: Branching and Merging

4. **Set Up Remote:**
   - Create GitHub repository
   - Add remote: `git remote add origin <url>`
   - Push: `git push -u origin main`

---

## Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [Pro Git Book](https://git-scm.com/book/en/v2) (Free)
- [GitHub Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git LFS Documentation](https://git-lfs.github.com/)

---

## Summary

You've successfully:
- ✅ Initialized a Git repository
- ✅ Configured Git with your identity
- ✅ Created comprehensive .gitignore for ML projects
- ✅ Set up .gitattributes for file handling
- ✅ Created project documentation (README)
- ✅ Made atomic commits with clear messages
- ✅ Understood Git's three-state workflow
- ✅ Learned commit message best practices

**Key Takeaways:**
- `.gitignore` prevents tracking unwanted files
- `.gitattributes` controls file handling behavior
- Atomic commits make history clean and reviewable
- Good commit messages explain WHY, not just WHAT
- Always verify with `git status` before committing

**Time to Complete:** ~90 minutes for thorough practice

**Next Exercise:** Exercise 02 - Commits and History Management
