# Setup Notes and Learning Guide

This document contains additional learning notes, tips, and troubleshooting guidance for the sentiment-classifier project setup.

## Table of Contents

1. [Virtual Environments Deep Dive](#virtual-environments-deep-dive)
2. [Dependency Management Best Practices](#dependency-management-best-practices)
3. [Environment Variables and Security](#environment-variables-and-security)
4. [Common Issues and Solutions](#common-issues-and-solutions)
5. [Advanced Topics](#advanced-topics)

---

## Virtual Environments Deep Dive

### Why Virtual Environments?

**Problem Without Virtual Environments:**
```bash
# Install package globally
pip install torch==2.1.0

# Another project needs torch==2.0.0
pip install torch==2.0.0  # Breaks first project!
```

**Solution With Virtual Environments:**
```bash
# Project 1
cd project1
python -m venv venv1
source venv1/bin/activate
pip install torch==2.1.0

# Project 2
cd project2
python -m venv venv2
source venv2/bin/activate
pip install torch==2.0.0  # No conflict!
```

### How Virtual Environments Work

1. **Directory Structure Created:**
   ```
   venv/
   ├── bin/              # Executables (python, pip, activate)
   ├── lib/              # Python packages
   │   └── python3.11/
   │       └── site-packages/
   ├── include/          # C headers
   └── pyvenv.cfg       # Configuration
   ```

2. **Activation Mechanism:**
   ```bash
   # Before activation
   which python  # /usr/bin/python

   # Activate
   source venv/bin/activate

   # After activation
   which python  # /path/to/venv/bin/python
   ```

3. **What `activate` Does:**
   - Prepends `venv/bin` to `PATH`
   - Sets `VIRTUAL_ENV` environment variable
   - Modifies prompt to show `(venv)`
   - Does NOT modify system Python

### Virtual Environment Tools Comparison

| Tool | Pros | Cons | Use Case |
|------|------|------|----------|
| **venv** | Built-in, simple, lightweight | Manual dependency tracking | Simple projects |
| **virtualenv** | More features, faster | Extra dependency | Projects needing speed |
| **conda** | Handles non-Python deps | Large, slow | Data science |
| **Poetry** | Modern, handles dependencies | Learning curve | Production projects |
| **pipenv** | Combines pip + venv | Slower than pip | Web projects |

---

## Dependency Management Best Practices

### Pinning Strategy

**Three Levels of Pinning:**

1. **Loose (Development):**
   ```txt
   # requirements-loose.txt
   torch>=2.0
   transformers>=4.30
   ```
   - Allows bug fixes and patches
   - Good for active development
   - Use in `requirements-dev.txt`

2. **Exact (Production):**
   ```txt
   # requirements.txt
   torch==2.1.0
   transformers==4.35.0
   ```
   - Guarantees reproducibility
   - Prevents breaking changes
   - Use in production deployments

3. **Frozen (Complete Lock):**
   ```txt
   # requirements-frozen.txt (pip freeze output)
   torch==2.1.0
   transformers==4.35.0
   numpy==1.24.0
   certifi==2023.7.22
   ... (all transitive dependencies)
   ```
   - Includes all transitive dependencies
   - Maximum reproducibility
   - Use for critical deployments

### Using pip-tools for Better Dependency Management

```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in (high-level dependencies)
cat > requirements.in <<EOF
torch>=2.0
transformers>=4.30
pandas>=2.0
EOF

# Compile to requirements.txt (with transitive dependencies)
pip-compile requirements.in

# Install compiled requirements
pip-sync requirements.txt

# Update all dependencies
pip-compile --upgrade requirements.in
```

### Handling Conflicts

**Example Conflict:**
```
package-a requires numpy==1.24.0
package-b requires numpy==1.23.0
```

**Solutions:**

1. **Find compatible versions:**
   ```bash
   pip install package-a package-b --dry-run
   ```

2. **Use dependency resolver:**
   ```bash
   pip install --use-feature=2020-resolver package-a package-b
   ```

3. **Separate environments:**
   ```bash
   # One venv per incompatible dependency set
   ```

---

## Environment Variables and Security

### Security Best Practices

**✗ Never Commit:**
```bash
# BAD - Secrets in code
API_KEY = "sk-1234567890abcdef"

# BAD - Secrets in git
git add .env
git commit -m "Added config"
```

**✓ Always Use .env:**
```bash
# GOOD - Secrets in .env
echo "API_KEY=sk-1234567890abcdef" >> .env

# GOOD - .env in .gitignore
echo ".env" >> .gitignore
```

### Environment Variable Loading Order

1. **System environment variables** (highest priority)
   ```bash
   export MODEL_NAME=bert-base
   ```

2. **.env file** (loaded by python-dotenv)
   ```bash
   MODEL_NAME=distilbert-base
   ```

3. **Default values in code** (lowest priority)
   ```python
   model = os.getenv("MODEL_NAME", "distilbert-base-uncased")
   ```

### Advanced .env Usage

**Multiple Environment Files:**
```bash
.env                # Common variables
.env.local          # Local overrides (gitignored)
.env.development    # Development-specific
.env.production     # Production-specific
```

**Loading in Python:**
```python
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Override with .env.local
load_dotenv(".env.local", override=True)

# Override with environment-specific
import os
env = os.getenv("ENVIRONMENT", "development")
load_dotenv(f".env.{env}", override=True)
```

---

## Common Issues and Solutions

### Issue 1: Virtual Environment Not Activating

**Symptoms:**
```bash
$ source venv/bin/activate
$ which python
/usr/bin/python  # Still system Python!
```

**Causes:**
- Using `sh` instead of `source`
- Shell alias conflicts
- Corrupted venv

**Solutions:**
```bash
# Correct activation:
source venv/bin/activate  # bash/zsh
. venv/bin/activate       # sh

# Windows:
venv\Scripts\activate.bat  # cmd.exe
venv\Scripts\Activate.ps1  # PowerShell

# If still not working, recreate venv:
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

### Issue 2: Package Installation Fails

**Symptoms:**
```bash
$ pip install torch
ERROR: Could not find a version that satisfies the requirement
```

**Solutions:**

1. **Check Python version:**
   ```bash
   python --version  # Needs 3.8+
   ```

2. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Use specific index:**
   ```bash
   pip install torch -f https://download.pytorch.org/whl/torch_stable.html
   ```

4. **Check network/proxy:**
   ```bash
   pip install --proxy http://proxy:port package
   ```

### Issue 3: Import Errors After Installation

**Symptoms:**
```python
>>> import torch
ModuleNotFoundError: No module named 'torch'
```

**Causes:**
- Wrong Python interpreter
- Package installed in different venv
- Corrupted installation

**Solutions:**
```bash
# Check which Python is running:
which python

# Check installed packages:
pip list | grep torch

# Reinstall in current environment:
pip uninstall torch
pip install torch

# Verify installation:
python -c "import torch; print(torch.__version__)"
```

### Issue 4: Permission Denied on setup.sh

**Symptoms:**
```bash
$ ./setup.sh
bash: ./setup.sh: Permission denied
```

**Solution:**
```bash
# Make executable:
chmod +x setup.sh

# Or run directly with bash:
bash setup.sh
```

### Issue 5: .env Variables Not Loading

**Symptoms:**
```python
>>> import os
>>> os.getenv("MODEL_NAME")
None
```

**Solutions:**

1. **Check .env exists:**
   ```bash
   ls -la .env
   ```

2. **Load with python-dotenv:**
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Must call this first!

   import os
   print(os.getenv("MODEL_NAME"))
   ```

3. **Check .env format:**
   ```bash
   # Correct:
   MODEL_NAME=distilbert-base-uncased

   # Incorrect (spaces):
   MODEL_NAME = distilbert-base-uncased  # Won't work!
   ```

---

## Advanced Topics

### 1. Editable Installations

For development, install your package in editable mode:

```bash
# Create setup.py or pyproject.toml
pip install -e .

# Now changes to src/ are immediately available:
python
>>> from src.utils import load_dataset  # Works!
```

### 2. Using Poetry for Modern Dependency Management

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Initialize project
poetry init

# Add dependencies
poetry add torch transformers pandas

# Add dev dependencies
poetry add --group dev pytest black mypy

# Install all dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 3. Docker for Maximum Reproducibility

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run training
CMD ["python", "src/train.py"]
```

```bash
# Build image
docker build -t sentiment-classifier .

# Run training
docker run --rm sentiment-classifier
```

### 4. Pre-commit Hooks for Quality

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
```

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Now runs on every commit!
```

### 5. GitHub Actions for CI/CD

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt

    - name: Run tests
      run: pytest tests/ -v --cov=src

    - name: Check code quality
      run: |
        black --check src/
        mypy src/
```

---

## Tips for Production

1. **Always Pin Dependencies:**
   - Use `pip freeze > requirements.txt` after testing
   - Consider using `pip-tools` or Poetry
   - Document why specific versions are needed

2. **Separate Dev and Prod Dependencies:**
   - `requirements.txt` - Production only
   - `requirements-dev.txt` - Includes dev tools
   - Don't deploy pytest, black, etc. to production

3. **Use Environment-Specific Configs:**
   - `.env.development` - Local development
   - `.env.staging` - Staging environment
   - `.env.production` - Production (in secrets manager!)

4. **Document Python Version:**
   - Create `.python-version` file
   - Or use `pyproject.toml`: `python = "^3.11"`
   - Include in README.md

5. **Test Your Setup:**
   - Run `scripts/verify_setup.py` regularly
   - Include in CI/CD pipeline
   - Document any manual steps

---

## Learning Resources

### Official Documentation
- [Python venv](https://docs.python.org/3/library/venv.html)
- [pip User Guide](https://pip.pypa.io/en/stable/user_guide/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

### Tutorials
- [Real Python: Virtual Environments](https://realpython.com/python-virtual-environments-a-primer/)
- [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/)
- [Managing Application Dependencies](https://realpython.com/python-application-layouts/)

### Tools
- [pip-tools](https://github.com/jazzband/pip-tools)
- [Poetry](https://python-poetry.org/)
- [pipenv](https://pipenv.pypa.io/)
- [pyenv](https://github.com/pyenv/pyenv)

---

**Last Updated:** 2025-10-30
**Version:** 1.0.0
