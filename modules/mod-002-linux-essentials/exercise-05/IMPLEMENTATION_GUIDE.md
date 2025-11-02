# Implementation Guide: Package Management for ML Stack Installation

## Overview

This guide teaches you to install and manage software packages for ML infrastructure using Linux package managers. You'll learn apt/yum basics, Python package management, Conda for ML environments, and production best practices for dependency management.

**Estimated Time:** 90-120 minutes
**Difficulty:** Intermediate

## Prerequisites

- Completed Exercises 01-04
- Linux system with sudo privileges
- Internet connection
- At least 5GB free disk space (for ML packages)

## Phase 1: System Package Management (30 minutes)

### Step 1.1: Detect Your Distribution

```bash
# Create workspace
mkdir -p ~/package-mgmt-lab
cd ~/package-mgmt-lab

# Detect distribution
cat /etc/os-release
lsb_release -a 2>/dev/null || echo "lsb_release not available"

# Check package manager
if command -v apt &> /dev/null; then
    echo "Package Manager: apt (Debian/Ubuntu)"
elif command -v dnf &> /dev/null; then
    echo "Package Manager: dnf (Fedora/RHEL 8+)"
elif command -v yum &> /dev/null; then
    echo "Package Manager: yum (CentOS/RHEL 7)"
fi
```

**Validation:**
- [ ] Identified your Linux distribution
- [ ] Know which package manager to use

### Step 1.2: Update Package Cache

**For Debian/Ubuntu:**
```bash
sudo apt update
sudo apt list --upgradable
```

**For RHEL/CentOS:**
```bash
sudo yum check-update
# or
sudo dnf check-update
```

### Step 1.3: Install Essential Build Tools

```bash
# Debian/Ubuntu
sudo apt install -y build-essential cmake git curl wget \
    libssl-dev libffi-dev python3-dev pkg-config

# RHEL/CentOS
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git curl wget openssl-devel \
    libffi-devel python3-devel
```

**Validation:**
```bash
gcc --version
cmake --version
git --version
```

**Expected Output:**
```
gcc (Ubuntu 11.4.0) ...
cmake version 3.22.1
git version 2.34.1
```

### Step 1.4: Search and Install Packages

```bash
# Search for packages
apt search python3 | head -10
# or
yum search python3 | head -10

# Get package info
apt show python3-pip
# or
yum info python3-pip

# Install a package
sudo apt install -y python3-pip
# or
sudo yum install -y python3-pip
```

### Step 1.5: Manage Package Versions

```bash
# List installed packages
apt list --installed | grep python
# or
yum list installed | grep python

# Check package version
dpkg -l python3-pip
# or
rpm -q python3-pip

# Hold a package version (prevent upgrades)
sudo apt-mark hold python3-pip
sudo apt-mark showhold

# Unhold
sudo apt-mark unhold python3-pip
```

**Validation:**
- [ ] Can search for packages
- [ ] Can install packages
- [ ] Can check installed versions
- [ ] Understand package hold mechanism

## Phase 2: Python Package Management with pip (45 minutes)

### Step 2.1: Verify pip Installation

```bash
# Check pip version
python3 -m pip --version
pip3 --version

# Upgrade pip
python3 -m pip install --upgrade pip

# Check installation location
which pip3
python3 -m pip --version  # Shows location
```

### Step 2.2: Virtual Environments (CRITICAL for ML)

```bash
# Install venv (if not already installed)
sudo apt install -y python3-venv
# or
sudo yum install -y python3-venv

# Create virtual environment
python3 -m venv ~/ml-env

# Activate virtual environment
source ~/ml-env/bin/activate

# Verify activation
which python
which pip
python --version
```

**Expected Output:**
```
/home/username/ml-env/bin/python
/home/username/ml-env/bin/pip
Python 3.10.12
```

### Step 2.3: Install ML Packages

```bash
# Install NumPy
pip install numpy

# Install specific version
pip install pandas==2.0.3

# Install with version constraints
pip install "scikit-learn>=1.3,<1.4"

# Install from requirements.txt
cat > requirements.txt << EOF
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
jupyter==1.0.0
EOF

pip install -r requirements.txt
```

### Step 2.4: Manage Dependencies

```bash
# List installed packages
pip list

# Show package details
pip show numpy

# Check for outdated packages
pip list --outdated

# Freeze dependencies
pip freeze > requirements-frozen.txt
cat requirements-frozen.txt

# Generate dependency tree
pip install pipdeptree
pipdeptree
```

### Step 2.5: Uninstall and Clean Up

```bash
# Uninstall a package
pip uninstall -y matplotlib

# Uninstall multiple packages
pip uninstall -y numpy pandas scikit-learn

# Reinstall from requirements
pip install -r requirements.txt

# Clear pip cache
pip cache purge
pip cache dir
```

**Validation:**
- [ ] Can create and activate virtual environments
- [ ] Can install packages with version constraints
- [ ] Can freeze dependencies
- [ ] Understand pip cache management

## Phase 3: Conda for ML Environments (45 minutes)

### Step 3.1: Install Miniforge (Recommended over Anaconda)

```bash
# Download Miniforge installer
cd ~/package-mgmt-lab
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Make executable
chmod +x Miniforge3-Linux-x86_64.sh

# Install
./Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# Initialize conda
$HOME/miniforge3/bin/conda init bash

# Restart shell or source bashrc
source ~/.bashrc

# Verify installation
conda --version
which conda
```

### Step 3.2: Create Conda Environments

```bash
# Create Python 3.10 environment
conda create -n pytorch-env python=3.10 -y

# List environments
conda env list

# Activate environment
conda activate pytorch-env

# Verify
python --version
which python
```

### Step 3.3: Install ML Frameworks

**Install PyTorch (CPU version):**
```bash
conda activate pytorch-env

# Install PyTorch from conda-forge
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Create TensorFlow environment:**
```bash
conda create -n tf-env python=3.10 -y
conda activate tf-env

# Install TensorFlow
pip install tensorflow==2.13.0

# Verify
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

### Step 3.4: Manage Conda Environments

```bash
# Export environment
conda env export > pytorch-env.yml
cat pytorch-env.yml

# Create environment from file
conda env create -f pytorch-env.yml -n pytorch-env-copy

# Clone environment
conda create --name pytorch-clone --clone pytorch-env

# Remove environment
conda deactivate
conda env remove -n pytorch-clone -y

# Update packages in environment
conda activate pytorch-env
conda update --all -y
```

### Step 3.5: Conda vs pip Comparison

**Create comparison script:**
```bash
cat > conda_vs_pip.md << 'EOF'
# Conda vs pip Comparison

## When to Use Conda

✅ Installing packages with C/C++ dependencies (NumPy, SciPy, PyTorch)
✅ Managing Python versions within environments
✅ Installing non-Python tools (CUDA, cuDNN, compilers)
✅ Complex ML frameworks (TensorFlow, PyTorch with CUDA)
✅ Reproducible environments across platforms

**Example:**
```bash
# Conda handles CUDA dependencies automatically
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## When to Use pip

✅ Lightweight pure-Python packages
✅ Latest package versions (conda can lag)
✅ Installing from Git repositories
✅ Development packages not in conda channels
✅ CI/CD pipelines (faster, smaller)

**Example:**
```bash
# pip for latest development version
pip install git+https://github.com/huggingface/transformers.git
```

## Best Practice: Hybrid Approach

1. Use conda to create base environment with Python
2. Use conda for major ML frameworks (PyTorch, TensorFlow)
3. Use pip for pure-Python packages within conda environment

**Example:**
```bash
# Create conda environment
conda create -n ml-project python=3.10 pytorch -c pytorch -y
conda activate ml-project

# Install additional packages with pip
pip install transformers datasets wandb
```

## Key Differences

| Feature | Conda | pip |
|---------|-------|-----|
| Language | Any (Python, C++, R) | Python only |
| Dependency solver | Better | Can have conflicts |
| Package source | Conda channels | PyPI |
| Environment isolation | Full (Python + libs) | Python packages only |
| Speed | Slower | Faster |
| Disk space | More | Less |
EOF

cat conda_vs_pip.md
```

**Validation:**
- [ ] Conda installed and working
- [ ] Can create and activate conda environments
- [ ] Can install PyTorch/TensorFlow
- [ ] Understand conda vs pip trade-offs

## Phase 4: CUDA and GPU Drivers (Optional - 30 minutes)

### Step 4.1: Check GPU Availability

```bash
# Check for NVIDIA GPU
lspci | grep -i nvidia

# Check if NVIDIA drivers installed
nvidia-smi 2>/dev/null && echo "NVIDIA drivers installed" || echo "NVIDIA drivers NOT installed"

# Check CUDA version (if installed)
nvcc --version 2>/dev/null || echo "CUDA not installed"
```

### Step 4.2: Install NVIDIA Drivers (Ubuntu/Debian)

```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install -y nvidia-driver-535

# Reboot required
sudo reboot
```

### Step 4.3: Install CUDA Toolkit

```bash
# Download CUDA installer from NVIDIA
# Example for CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Install
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

### Step 4.4: Install CUDA-Enabled PyTorch

```bash
conda create -n pytorch-gpu python=3.10 -y
conda activate pytorch-gpu

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Validation:**
- [ ] Can detect GPU
- [ ] NVIDIA drivers installed (if GPU present)
- [ ] Can verify CUDA availability in PyTorch

## Phase 5: Production Best Practices (45 minutes)

### Step 5.1: Pin Dependency Versions

**Create reproducible requirements:**
```bash
# Create environment with specific versions
cat > requirements-prod.txt << EOF
# Core dependencies
numpy==1.24.3
pandas==2.0.3

# ML frameworks
torch==2.0.1
torchvision==0.15.2
transformers==4.30.2

# Data processing
scikit-learn==1.3.0
pillow==10.0.0

# Development tools
pytest==7.4.0
black==23.7.0
flake8==6.0.0
EOF

# Install in new environment
python3 -m venv prod-env
source prod-env/bin/activate
pip install --no-cache-dir -r requirements-prod.txt

# Verify exact versions
pip list
```

### Step 5.2: Security Vulnerability Scanning

```bash
# Install safety
pip install safety

# Check for known vulnerabilities
safety check

# Check specific requirements file
safety check -r requirements-prod.txt

# Generate detailed report
safety check --full-report
```

### Step 5.3: Dependency Audit Script

**Create audit script:**
```bash
cat > audit_dependencies.sh << 'EOF'
#!/bin/bash
# Audit Python dependencies

echo "=== Python Package Audit ==="
echo "Date: $(date)"
echo ""

# Check Python version
echo "Python version:"
python --version
echo ""

# List installed packages
echo "Installed packages:"
pip list
echo ""

# Check for outdated packages
echo "Outdated packages:"
pip list --outdated
echo ""

# Security vulnerabilities
echo "Security check:"
if command -v safety &> /dev/null; then
    safety check
else
    echo "safety not installed. Install with: pip install safety"
fi
echo ""

# Dependency tree
echo "Dependency tree:"
if command -v pipdeptree &> /dev/null; then
    pipdeptree
else
    echo "pipdeptree not installed. Install with: pip install pipdeptree"
fi
EOF

chmod +x audit_dependencies.sh
./audit_dependencies.sh
```

### Step 5.4: Automated Environment Creation

**Create setup script:**
```bash
cat > setup_ml_env.sh << 'EOF'
#!/bin/bash
# Setup ML environment reproducibly

set -e  # Exit on error

ENV_NAME="ml-project"
PYTHON_VERSION="3.10"

echo "Creating ML environment: $ENV_NAME"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using conda..."

    # Remove existing environment
    conda env remove -n $ENV_NAME -y 2>/dev/null || true

    # Create new environment
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME

    # Install packages
    conda install pytorch torchvision -c pytorch -y
    pip install -r requirements.txt

else
    echo "Using venv..."

    # Remove existing environment
    rm -rf $ENV_NAME

    # Create new environment
    python3 -m venv $ENV_NAME
    source $ENV_NAME/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install packages
    pip install -r requirements.txt
fi

echo "Environment created successfully!"
echo "Activate with: conda activate $ENV_NAME (or source $ENV_NAME/bin/activate)"
EOF

chmod +x setup_ml_env.sh
```

### Step 5.5: Docker for Reproducibility

**Create Dockerfile:**
```bash
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Default command
CMD ["python", "--version"]
EOF

# Build image
docker build -t ml-environment:latest .

# Run container
docker run --rm ml-environment:latest

# Interactive shell
docker run -it --rm ml-environment:latest /bin/bash
```

**Validation:**
- [ ] Can pin dependency versions
- [ ] Can scan for security vulnerabilities
- [ ] Have automated setup scripts
- [ ] Understand Docker for reproducibility

## Common Issues and Solutions

### Issue 1: pip Install Fails with Permission Error

**Symptoms:**
```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied
```

**Solution:**
```bash
# NEVER use sudo pip install
# Instead, use --user flag or virtual environment

# Option 1: Install in user directory
pip install --user package_name

# Option 2: Use virtual environment (RECOMMENDED)
python3 -m venv myenv
source myenv/bin/activate
pip install package_name
```

### Issue 2: Conda Install Very Slow

**Symptoms:**
Conda taking 30+ minutes to solve environment

**Solution:**
```bash
# Use mamba (faster conda alternative)
conda install mamba -c conda-forge -y

# Use mamba instead of conda
mamba install pytorch torchvision -c pytorch

# Or use libmamba solver
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Issue 3: Package Conflicts

**Symptoms:**
```
ERROR: package-a 1.0 requires package-b<2.0, but you have package-b 2.1
```

**Solution:**
```bash
# Create fresh environment
python3 -m venv fresh-env
source fresh-env/bin/activate

# Install in order (dependencies first)
pip install package-b==1.9
pip install package-a

# Or use constraints file
pip install -c constraints.txt package-a package-b
```

### Issue 4: CUDA Version Mismatch

**Symptoms:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**
```bash
# Check CUDA version
nvidia-smi  # Driver CUDA version

# Match PyTorch CUDA version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or use conda
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Issue 5: Import Errors After Installation

**Symptoms:**
```
ImportError: No module named 'package_name'
```

**Solution:**
```bash
# Verify environment is activated
which python
pip list | grep package_name

# Reinstall package
pip uninstall package_name
pip install package_name

# Check for conflicting installations
pip show package_name
```

## Best Practices Summary

### Virtual Environments

✅ **Always use virtual environments** (never install globally)
✅ One virtual environment per project
✅ Name environments descriptively: `project-name-env`
✅ Keep base Python clean (minimal packages)

### Dependency Management

✅ **Pin exact versions** in production: `package==1.2.3`
✅ Use ranges in development: `package>=1.2,<2.0`
✅ Keep `requirements.txt` updated
✅ Generate `requirements-frozen.txt` for exact reproduction
✅ Document why specific versions are required

### Security

✅ **Scan dependencies regularly** with `safety check`
✅ Update packages for security patches
✅ Review CVEs before deploying
✅ Use trusted package sources only

### Conda vs pip

✅ Use conda for base environment + ML frameworks
✅ Use pip for pure-Python packages
✅ Don't mix conda and pip for the same package
✅ Install conda packages first, then pip

### Reproducibility

✅ **Document Python version** in README
✅ Provide both `requirements.txt` and `environment.yml`
✅ Use Docker for complete reproducibility
✅ Test environment setup on clean system
✅ Automate with setup scripts

### Performance

✅ Use `--no-cache-dir` in Docker to save space
✅ Clean pip cache: `pip cache purge`
✅ Use mamba instead of conda for faster installs
✅ Pin versions to avoid resolution time

## Completion Checklist

### System Packages
- [ ] Can identify Linux distribution and package manager
- [ ] Can update package cache
- [ ] Can search, install, and manage system packages
- [ ] Can install build tools and development libraries

### Python Packaging
- [ ] pip installed and upgraded
- [ ] Can create and activate virtual environments
- [ ] Can install packages with version constraints
- [ ] Can use requirements.txt
- [ ] Can freeze and export dependencies
- [ ] Understand pip cache management

### Conda
- [ ] Conda/Miniforge installed
- [ ] Can create conda environments
- [ ] Can install PyTorch/TensorFlow
- [ ] Can export and recreate environments
- [ ] Understand conda vs pip trade-offs

### GPU/CUDA (Optional)
- [ ] Can check for GPU availability
- [ ] NVIDIA drivers installed (if GPU present)
- [ ] CUDA toolkit installed (if needed)
- [ ] Can install CUDA-enabled packages

### Production Practices
- [ ] Can pin dependency versions
- [ ] Can scan for security vulnerabilities
- [ ] Have automated setup scripts
- [ ] Understand Docker for reproducibility
- [ ] Can audit dependencies

### Troubleshooting
- [ ] Can resolve permission errors
- [ ] Can handle package conflicts
- [ ] Can troubleshoot CUDA issues
- [ ] Can debug import errors

## Next Steps

1. **Exercise 06: Log Management** - Learn to analyze system and application logs
2. **Advanced Topics:**
   - Multi-stage Docker builds for ML
   - Private PyPI repositories
   - Dependency caching in CI/CD
   - Security scanning automation

3. **Production Skills:**
   - Package versioning strategies
   - Dependency update workflows
   - Security patch management
   - Build reproducibility validation

## Resources

- [pip Documentation](https://pip.pypa.io/)
- [Python Virtual Environments Guide](https://docs.python.org/3/library/venv.html)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [TensorFlow Installation](https://www.tensorflow.org/install)
- [Docker Best Practices for Python](https://docs.docker.com/language/python/)

Congratulations! You now know how to manage packages for ML infrastructure.
