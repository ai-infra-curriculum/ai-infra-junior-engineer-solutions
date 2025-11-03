# Exercise Answers - Environment Setup and Management

This document provides comprehensive answers to all reflection questions from the exercise.

---

## Question 1: How do virtual environments provide isolation?

### Short Answer
Virtual environments create isolated Python installations by:
1. Creating a separate directory with its own `site-packages`
2. Modifying `sys.path` to prioritize the venv's packages
3. Using a dedicated `pip` installation
4. Setting `sys.prefix` to point to the venv directory

### Detailed Explanation

**Before Virtual Environment:**
```python
import sys

print(sys.prefix)
# Output: /usr/local

print(sys.executable)
# Output: /usr/bin/python3

print(sys.path)
# Output: ['/usr/lib/python3.11', '/usr/lib/python3.11/site-packages', ...]
```

**After Activating Virtual Environment:**
```python
import sys

print(sys.prefix)
# Output: /path/to/project/venv

print(sys.executable)
# Output: /path/to/project/venv/bin/python

print(sys.path)
# Output: ['/path/to/project/venv/lib/python3.11/site-packages', ...]
```

**Mechanism:**

1. **Directory Structure:**
   ```
   venv/
   ├── bin/python       # Symlink to system Python
   ├── bin/pip          # Separate pip installation
   ├── lib/python3.11/
   │   └── site-packages/  # Isolated package directory
   └── pyvenv.cfg       # Configuration
   ```

2. **PATH Modification:**
   ```bash
   # Activation script prepends venv/bin to PATH:
   export PATH="/path/to/venv/bin:$PATH"
   ```

3. **sys.path Priority:**
   - Python checks venv's `site-packages` first
   - Only falls back to system packages if not found in venv
   - System packages are still accessible (read-only)

4. **Package Installation:**
   ```bash
   # pip installs to venv's site-packages:
   pip install torch
   # → /path/to/venv/lib/python3.11/site-packages/torch/
   ```

### Example: Proving Isolation

```python
# Create two venvs with different numpy versions
import subprocess
import sys

# Venv 1
subprocess.run(["python3", "-m", "venv", "venv1"])
subprocess.run(["venv1/bin/pip", "install", "numpy==1.24.0"])

# Venv 2
subprocess.run(["python3", "-m", "venv", "venv2"])
subprocess.run(["venv2/bin/pip", "install", "numpy==1.23.0"])

# Test isolation:
result1 = subprocess.run(
    ["venv1/bin/python", "-c", "import numpy; print(numpy.__version__)"],
    capture_output=True,
    text=True,
)
print(f"Venv 1: {result1.stdout.strip()}")  # 1.24.0

result2 = subprocess.run(
    ["venv2/bin/python", "-c", "import numpy; print(numpy.__version__)"],
    capture_output=True,
    text=True,
)
print(f"Venv 2: {result2.stdout.strip()}")  # 1.23.0
```

**Result:** Each venv has its own numpy version without conflict!

---

## Question 2: Why pin exact versions in production?

### Short Answer
Pinning exact versions (e.g., `torch==2.1.0`) ensures:
- **Reproducibility**: Same code produces same results
- **Stability**: No unexpected breaking changes
- **Debugging**: Easier to reproduce bugs
- **Compliance**: Audit trail for regulated industries

### Comparison: Pinned vs. Unpinned

**Production (Pinned):**
```txt
# requirements.txt
torch==2.1.0
transformers==4.35.0
pandas==2.1.0
```

**Benefits:**
- ✓ Reproducible builds
- ✓ No surprise updates
- ✓ Known vulnerabilities can be tracked
- ✓ Easier rollback if issues occur

**Development (Flexible):**
```txt
# requirements-dev.txt
torch>=2.0.0
transformers>=4.30.0
pandas>=2.0.0
```

**Benefits:**
- ✓ Automatic bug fixes
- ✓ Security patches
- ✓ Testing with newer versions
- ✓ Easier dependency resolution

### Real-World Example

**Scenario:** Your model training code works perfectly.

**Without Pinning:**
```txt
# requirements.txt
torch>=2.0
```

**6 months later:**
```bash
# New developer joins team:
pip install -r requirements.txt
# Installs torch==2.3.0 (newer version)

# Training produces different results!
# Model accuracy: 0.87 → 0.82 (regression)
```

**Cause:** PyTorch 2.3.0 changed default behavior of some optimizer.

**With Pinning:**
```txt
# requirements.txt
torch==2.1.0
```

**6 months later:**
```bash
# New developer joins team:
pip install -r requirements.txt
# Installs torch==2.1.0 (exact same version)

# Training produces identical results!
# Model accuracy: 0.87 ✓
```

### When to Upgrade

1. **Security vulnerabilities** discovered
2. **Bug fixes** that affect your code
3. **Performance improvements** (test thoroughly)
4. **New features** you need

**Process:**
```bash
# 1. Create test branch
git checkout -b upgrade-torch

# 2. Update version
echo "torch==2.2.0" >> requirements.txt

# 3. Run full test suite
pytest tests/ -v

# 4. Run training validation
python src/train.py --validate

# 5. Compare metrics
python scripts/compare_results.py

# 6. If all pass, merge
git merge upgrade-torch
```

### ML-Specific Considerations

**Why Pinning is CRITICAL for ML:**

1. **Non-Deterministic Operations:**
   ```python
   # Same code, different versions → different results
   torch.nn.functional.dropout(x, p=0.5)
   # PyTorch 2.0: One behavior
   # PyTorch 2.1: Slightly different randomness
   ```

2. **Numerical Precision:**
   ```python
   # Floating-point operations can differ:
   torch.matmul(A, B)
   # Different versions may use different CUDA kernels
   ```

3. **Model Checkpoints:**
   ```python
   # Model trained with torch==2.0.0
   torch.save(model.state_dict(), "model.pth")

   # Loading with torch==2.1.0 may fail or behave differently
   model.load_state_dict(torch.load("model.pth"))
   ```

### Best Practice: Three-Tier Strategy

```
1. requirements.in          # High-level, flexible
   torch>=2.0

2. requirements.txt         # Compiled, pinned
   torch==2.1.0
   (via pip-compile)

3. requirements-frozen.txt  # Complete lock
   torch==2.1.0
   numpy==1.24.0
   ... (all transitive deps)
   (via pip freeze)
```

---

## Question 3: What are transitive dependencies?

### Short Answer
Transitive dependencies are packages required by your direct dependencies, but not explicitly listed in your `requirements.txt`.

### Example

**Your requirements.txt:**
```txt
torch==2.1.0
transformers==4.35.0
```

**pip freeze output:**
```txt
torch==2.1.0
transformers==4.35.0
# Plus transitive dependencies:
numpy==1.24.0          # Required by torch
regex==2023.10.3       # Required by transformers
requests==2.31.0       # Required by transformers
certifi==2023.7.22     # Required by requests
charset-normalizer==3.3.2  # Required by requests
idna==3.4              # Required by requests
urllib3==2.0.7         # Required by requests
filelock==3.13.1       # Required by transformers
fsspec==2023.10.0      # Required by transformers
jinja2==3.1.2          # Required by transformers
MarkupSafe==2.1.3      # Required by jinja2
packaging==23.2        # Required by transformers
pyyaml==6.0.1          # Required by transformers
tqdm==4.66.1           # Required by transformers
tokenizers==0.15.0     # Required by transformers
safetensors==0.4.1     # Required by transformers
huggingface-hub==0.19.4  # Required by transformers
typing-extensions==4.8.0  # Required by torch
sympy==1.12            # Required by torch
networkx==3.2.1        # Required by torch
mpmath==1.3.0          # Required by sympy
... (50+ more packages!)
```

### Dependency Tree Visualization

```
your-project
├── torch==2.1.0
│   ├── numpy==1.24.0
│   ├── typing-extensions==4.8.0
│   ├── sympy==1.12
│   │   └── mpmath==1.3.0
│   └── networkx==3.2.1
└── transformers==4.35.0
    ├── numpy==1.24.0 (already installed)
    ├── regex==2023.10.3
    ├── requests==2.31.0
    │   ├── certifi==2023.7.22
    │   ├── charset-normalizer==3.3.2
    │   ├── idna==3.4
    │   └── urllib3==2.0.7
    ├── tokenizers==0.15.0
    ├── huggingface-hub==0.19.4
    └── ... (10+ more)
```

### Why This Matters

**Problem 1: Hidden Conflicts**
```txt
# requirements.txt
package-a==1.0  # Requires numpy==1.24
package-b==2.0  # Requires numpy==1.23

# pip install fails!
ERROR: Cannot install package-a and package-b because these package versions have conflicting dependencies.
```

**Problem 2: Unexpected Changes**
```bash
# Day 1: Install dependencies
pip install -r requirements.txt
# Installs torch==2.1.0 + numpy==1.24.0

# Day 90: Reinstall from scratch
pip install -r requirements.txt
# Installs torch==2.1.0 + numpy==1.25.0 (newer!)

# Code breaks because numpy API changed!
```

### Solution: Lock ALL Dependencies

**Option 1: pip freeze**
```bash
pip install -r requirements.txt
pip freeze > requirements-frozen.txt

# Now install from frozen file:
pip install -r requirements-frozen.txt
```

**Option 2: pip-tools**
```bash
# requirements.in (high-level)
torch>=2.0
transformers>=4.30

# Compile to requirements.txt (locked)
pip-compile requirements.in

# Output: requirements.txt with ALL dependencies pinned
```

**Option 3: Poetry**
```bash
# pyproject.toml (your deps)
[tool.poetry.dependencies]
torch = "^2.1"
transformers = "^4.35"

# poetry.lock (all transitive deps)
# Auto-generated by Poetry
```

### Checking Dependency Tree

```bash
# Install pipdeptree
pip install pipdeptree

# View dependency tree
pipdeptree

# View specific package
pipdeptree -p transformers

# Find conflicts
pipdeptree --warn fail
```

### Best Practice

**For Production:**
```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Freeze ALL dependencies
pip freeze > requirements-locked.txt

# 3. Use locked file for deployments
pip install -r requirements-locked.txt
```

**Result:**
- All 50+ transitive dependencies pinned
- Perfect reproducibility
- No surprise updates

---

## Question 4: Why separate .env.example from .env?

### Short Answer
- **`.env.example`**: Template with placeholder values (committed to git)
- **`.env`**: Actual secrets and configuration (NEVER committed)

This separation:
- Documents required configuration
- Prevents secret leakage
- Provides safe defaults for new developers

### Security Risk Example

**BAD - Committing .env:**
```bash
# Developer accidentally commits secrets
cat .env
API_KEY=sk-1234567890abcdef
DATABASE_PASSWORD=SuperSecret123!

git add .env
git commit -m "Added config"
git push

# ⚠️ SECRETS ARE NOW IN GIT HISTORY FOREVER!
# Even if you delete the file later:
git rm .env
git commit -m "Remove secrets"

# Secrets still accessible in history:
git log --all -- .env
git show abc123:.env  # Can still see secrets!
```

**Consequences:**
- Attackers can access your API keys
- Database can be compromised
- AWS/GCP bills can skyrocket (crypto mining)
- Compliance violations (GDPR, HIPAA, SOC2)
- Company reputation damage

**GOOD - Using .env.example:**
```bash
# .env.example (safe to commit)
API_KEY=your_api_key_here
DATABASE_PASSWORD=your_password_here
AWS_ACCESS_KEY_ID=your_aws_key_here

# .env (never committed)
API_KEY=sk-1234567890abcdef
DATABASE_PASSWORD=SuperSecret123!
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE

# .gitignore
.env
.env.local
.env.*.local
```

### Setup Workflow

**New Developer Joins:**
```bash
# 1. Clone repository
git clone repo.git
cd repo

# 2. Copy example to .env
cp .env.example .env

# 3. Fill in actual values
nano .env

# 4. Verify
python scripts/test_env.py
```

### Advanced: Multiple Environments

**Project Structure:**
```
.env.example          # Template
.env.local            # Local development (gitignored)
.env.development      # Dev environment (gitignored)
.env.staging          # Staging environment (in secrets manager)
.env.production       # Production (in secrets manager)
```

**Loading Priority:**
```python
from dotenv import load_dotenv
import os

# 1. Load defaults from .env
load_dotenv()

# 2. Override with environment-specific
env = os.getenv("ENVIRONMENT", "development")
load_dotenv(f".env.{env}", override=True)

# 3. Override with local settings
load_dotenv(".env.local", override=True)
```

### Production: Use Secrets Manager

**Don't use .env files in production!**

Instead, use:
- **AWS Secrets Manager**
- **GCP Secret Manager**
- **Azure Key Vault**
- **HashiCorp Vault**
- **Kubernetes Secrets**

**Example with AWS Secrets Manager:**
```python
import boto3
import json

def load_secrets():
    client = boto3.client("secretsmanager", region_name="us-east-1")
    response = client.get_secret_value(SecretId="ml-app/production")
    secrets = json.loads(response["SecretString"])

    # Set as environment variables
    for key, value in secrets.items():
        os.environ[key] = value
```

### Checklist: Configuration Security

- [ ] `.env` in `.gitignore`
- [ ] `.env.example` committed with placeholders
- [ ] No secrets in code
- [ ] No secrets in docker images
- [ ] No secrets in logs
- [ ] Secrets rotated regularly
- [ ] Use secrets manager in production
- [ ] Document all required env vars

---

## Question 5: Why automate setup with scripts?

### Short Answer
Automation (like `setup.sh`) provides:
- **Consistency**: Same setup for all developers
- **Speed**: 5 minutes vs. 30+ minutes manual
- **Documentation**: Script IS the documentation
- **Reliability**: No human error

### Time Comparison

**Manual Setup (30-45 minutes):**
```bash
# Developer follows README:
python3 -m venv venv         # 2 min
source venv/bin/activate     # 30 sec
pip install --upgrade pip    # 1 min
pip install -r requirements-dev.txt  # 10 min
cp .env.example .env         # 30 sec
nano .env                    # 5 min (editing)
mkdir -p data models         # 30 sec
# ... forgot a step, debugs for 15 min
# Total: 30-45 minutes
```

**Automated Setup (3-5 minutes):**
```bash
./setup.sh
# Total: 3-5 minutes
```

**Time Saved:**
```
30 min × 5 developers × 2 times/year = 5 hours
vs.
5 min × 5 developers × 2 times/year = 50 minutes

Saved: 4+ hours per year
```

### Error Prevention

**Manual Setup - Common Errors:**
1. Forget to activate venv → packages install globally
2. Forget to upgrade pip → old packages
3. Install wrong requirements file
4. Typo in directory names
5. Forget to make scripts executable
6. Skip .env setup

**Automated Setup:**
- ✓ All steps executed correctly
- ✓ Error handling built-in
- ✓ Validation included
- ✓ Consistent across team

### Features of Good Setup Scripts

**1. Error Handling:**
```bash
set -e  # Exit on error
set -u  # Error on undefined variables
set -o pipefail  # Error on pipe failures
```

**2. Validation:**
```bash
# Check Python version
if [ "$PYTHON_MINOR" -lt 11 ]; then
    echo "Error: Python 3.11+ required"
    exit 1
fi
```

**3. Idempotency:**
```bash
# Can run multiple times safely
if [ -d "venv" ]; then
    echo "venv exists, skipping creation"
else
    python3 -m venv venv
fi
```

**4. User Feedback:**
```bash
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
```

**5. Instructions:**
```bash
echo "Next steps:"
echo "  1. Activate: source venv/bin/activate"
echo "  2. Edit .env with your configuration"
```

### Beyond setup.sh: Full Automation

**Development Container (DevContainer):**
```json
// .devcontainer/devcontainer.json
{
  "name": "Sentiment Classifier",
  "image": "python:3.11",
  "postCreateCommand": "pip install -r requirements-dev.txt",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python"]
    }
  }
}
```

**Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    volumes:
      - .:/app
    environment:
      - MODEL_NAME=${MODEL_NAME}
    command: python src/train.py
```

**Makefile:**
```makefile
.PHONY: setup test train

setup:
    python3 -m venv venv
    . venv/bin/activate && pip install -r requirements-dev.txt

test:
    pytest tests/ -v

train:
    python src/train.py
```

### ROI Calculation

**Initial Investment:**
- Write setup.sh: 1 hour
- Test on different systems: 1 hour
- Document: 30 minutes
- **Total: 2.5 hours**

**Savings:**
- Time saved per setup: 25 minutes
- Number of setups per year: 10 (5 devs × 2 times)
- **Total saved: 250 minutes = 4.2 hours/year**

**ROI: 168% in first year**

### Best Practices

1. **Make Scripts Executable:**
   ```bash
   chmod +x setup.sh
   ```

2. **Support Multiple Platforms:**
   ```bash
   if [[ "$OSTYPE" == "darwin"* ]]; then
       # macOS
   elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
       # Linux
   fi
   ```

3. **Include Verification:**
   ```bash
   ./setup.sh
   python scripts/verify_setup.py
   ```

4. **Version Control:**
   ```bash
   git add setup.sh scripts/
   git commit -m "Add setup automation"
   ```

5. **Document in README:**
   ```markdown
   ## Quick Start
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```
   ```

---

## Summary

These five questions cover fundamental concepts in Python environment management:

1. **Virtual Environments**: Isolation through separate site-packages and PATH manipulation
2. **Version Pinning**: Reproducibility through exact version specification
3. **Transitive Dependencies**: Hidden dependencies that affect builds
4. **Configuration Security**: Preventing secret leakage with .env patterns
5. **Automation**: Consistency and efficiency through scripting

**Key Takeaway:** Professional Python development requires careful environment management, security practices, and automation. These practices prevent bugs, security incidents, and wasted time.

---

**Last Updated:** 2025-10-30
**Version:** 1.0.0
