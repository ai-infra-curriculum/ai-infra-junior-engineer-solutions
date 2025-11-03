# Exercise 05: Package Management for ML Stack Installation - Solution

## Overview

This solution provides production-ready scripts for installing and managing a complete ML infrastructure stack across different Linux distributions. It includes comprehensive package management for system tools, Python packages, GPU drivers, Docker, and ML frameworks with dependency resolution and version management.

## Learning Objectives Covered

- ✅ Understand different package management systems (apt, yum, dnf)
- ✅ Install system packages and development tools
- ✅ Manage Python packages with pip and conda
- ✅ Install CUDA and GPU drivers
- ✅ Set up Docker and container runtime
- ✅ Install ML frameworks (TensorFlow, PyTorch)
- ✅ Handle dependency conflicts and version management
- ✅ Create reproducible installation scripts

## Solution Structure

```
exercise-05/
├── README.md                          # This file - solution overview
├── IMPLEMENTATION_GUIDE.md            # Step-by-step implementation guide
├── scripts/                           # Installation and management scripts
│   ├── detect_system.sh               # System detection and analysis
│   ├── install_ml_stack.sh            # Complete ML stack installer
│   ├── setup_python_env.sh            # Python environment setup
│   ├── install_cuda.sh                # CUDA and GPU driver installation
│   ├── install_docker.sh              # Docker installation
│   ├── manage_packages.sh             # Package management utilities
│   ├── validate_installation.sh       # Installation validation
│   └── troubleshoot_packages.sh       # Troubleshooting utilities
├── examples/                          # Example configurations
│   ├── requirements/                  # Python requirements files
│   ├── conda_envs/                    # Conda environment specs
│   └── docker_configs/                # Docker configurations
└── docs/
    └── ANSWERS.md                     # Reflection question answers
```

## Key Features

### 1. System Detection (detect_system.sh)

Automatically detects Linux distribution and package manager:

```bash
./scripts/detect_system.sh
```

**Features:**
- Detects distribution (Ubuntu, Debian, RHEL, CentOS, Fedora)
- Identifies package manager (apt, yum, dnf)
- Checks system architecture and kernel version
- Verifies available package managers
- Generates system compatibility report

### 2. Complete ML Stack Installer (install_ml_stack.sh)

One-command installation of complete ML infrastructure:

```bash
./scripts/install_ml_stack.sh [options]
```

**Options:**
- `--full`: Install everything (system packages, Python, Docker, CUDA)
- `--system-only`: Install only system packages
- `--python-only`: Install only Python packages
- `--docker-only`: Install only Docker
- `--cuda-only`: Install only CUDA and GPU drivers
- `--dry-run`: Show what would be installed without installing

**Components Installed:**
- Build essentials (gcc, g++, make, cmake)
- Python 3.x with development headers
- System libraries (BLAS, LAPACK, HDF5, image processing)
- Docker and docker-compose
- NVIDIA drivers and CUDA toolkit
- Python ML packages (NumPy, Pandas, scikit-learn)
- ML frameworks (TensorFlow, PyTorch)
- Monitoring tools (htop, iotop, nvidia-smi)

### 3. Python Environment Setup (setup_python_env.sh)

Creates isolated Python environments with ML packages:

```bash
./scripts/setup_python_env.sh [env_name] [--with-gpu] [--framework=tensorflow|pytorch|both]
```

**Features:**
- Creates virtual environments with venv or conda
- Installs specified ML frameworks
- GPU support configuration
- Exports requirements.txt
- Environment activation instructions

**Environment Types:**
- `ml_cpu`: CPU-only ML environment
- `ml_gpu`: GPU-accelerated ML environment
- `datascience`: Data science tools
- `tensorflow`: TensorFlow-focused
- `pytorch`: PyTorch-focused

### 4. CUDA Installation (install_cuda.sh)

Automated NVIDIA driver and CUDA toolkit installation:

```bash
./scripts/install_cuda.sh [--version=12.2] [--driver=535]
```

**Features:**
- Detects NVIDIA GPU
- Installs compatible drivers
- Installs CUDA toolkit
- Installs cuDNN
- Configures environment variables
- Verifies installation
- Troubleshoots common issues

### 5. Docker Installation (install_docker.sh)

Installs Docker with NVIDIA GPU support:

```bash
./scripts/install_docker.sh [--with-nvidia-docker]
```

**Features:**
- Installs Docker CE
- Installs docker-compose
- Adds user to docker group
- Installs NVIDIA Container Toolkit (optional)
- Starts and enables Docker service
- Verifies installation

### 6. Package Management Utilities (manage_packages.sh)

Unified interface for package management:

```bash
./scripts/manage_packages.sh <command> [package_names...]
```

**Commands:**
- `install`: Install packages
- `remove`: Remove packages
- `update`: Update package lists
- `upgrade`: Upgrade installed packages
- `search`: Search for packages
- `info`: Show package information
- `list`: List installed packages
- `clean`: Clean package cache

**Automatically detects and uses correct package manager (apt/yum/dnf)**

### 7. Installation Validation (validate_installation.sh)

Comprehensive installation verification:

```bash
./scripts/validate_installation.sh [--full] [--gpu] [--docker]
```

**Checks:**
- System tools (gcc, python, git)
- Python packages (numpy, pandas, tensorflow, pytorch)
- GPU drivers and CUDA
- Docker and docker-compose
- ML framework GPU support
- Generates detailed validation report

### 8. Troubleshooting Utilities (troubleshoot_packages.sh)

Diagnoses and fixes common package issues:

```bash
./scripts/troubleshoot_packages.sh [issue_type]
```

**Issue Types:**
- `dependencies`: Resolve dependency conflicts
- `broken-packages`: Fix broken packages
- `cuda-mismatch`: Fix CUDA version mismatches
- `python-env`: Fix Python environment issues
- `docker-permissions`: Fix Docker permission issues
- `gpu-not-detected`: Troubleshoot GPU detection

## Quick Start

### 1. Detect System

```bash
cd /path/to/exercise-05/scripts
chmod +x *.sh
./detect_system.sh
```

### 2. Install Complete ML Stack

```bash
# Full installation (requires sudo)
./install_ml_stack.sh --full

# Or install components individually
./install_ml_stack.sh --system-only
./install_ml_stack.sh --python-only
./install_ml_stack.sh --docker-only
./install_ml_stack.sh --cuda-only  # Only if you have NVIDIA GPU
```

### 3. Create Python Environment

```bash
# Create CPU-only environment
./setup_python_env.sh ml_cpu

# Create GPU environment with TensorFlow
./setup_python_env.sh ml_gpu --with-gpu --framework=tensorflow

# Create GPU environment with PyTorch
./setup_python_env.sh ml_gpu --with-gpu --framework=pytorch
```

### 4. Validate Installation

```bash
# Validate all components
./validate_installation.sh --full

# Validate specific components
./validate_installation.sh --gpu
./validate_installation.sh --docker
```

## Package Management Examples

### System Packages

```bash
# Install development tools
./manage_packages.sh install build-essential python3-dev

# Search for packages
./manage_packages.sh search tensorflow

# Update package lists
./manage_packages.sh update

# Upgrade all packages
./manage_packages.sh upgrade

# Clean package cache
./manage_packages.sh clean
```

### Python Packages (using pip)

```bash
# Activate environment
source ~/ml_env/bin/activate

# Install specific versions
pip install numpy==1.24.3 pandas==2.0.3

# Install from requirements file
pip install -r requirements.txt

# Upgrade packages
pip install --upgrade tensorflow

# Freeze current environment
pip freeze > requirements.txt

# Uninstall packages
pip uninstall package_name
```

### Python Packages (using conda)

```bash
# Create environment
conda create -n myenv python=3.10

# Activate environment
conda activate myenv

# Install packages
conda install numpy pandas scikit-learn

# Install from conda-forge
conda install -c conda-forge tensorflow-gpu

# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml

# Remove environment
conda remove -n myenv --all
```

## Supported Distributions

### Tested Distributions
- Ubuntu 20.04 LTS, 22.04 LTS
- Debian 11, 12
- RHEL 8, 9
- CentOS 8 Stream, 9 Stream
- Fedora 38, 39

### Package Manager Support
- **apt** (Ubuntu, Debian)
- **yum** (CentOS 7, RHEL 7)
- **dnf** (CentOS 8+, RHEL 8+, Fedora)

## Version Management

### Pinning Python Package Versions

```bash
# requirements.txt with version pinning
numpy==1.24.3          # Exact version
pandas>=2.0.0,<3.0.0   # Version range
tensorflow~=2.13.0     # Compatible release (~= 2.13.0 means >=2.13.0,<2.14.0)
```

### Managing Multiple CUDA Versions

```bash
# Install specific CUDA version
./install_cuda.sh --version=11.8

# Switch between CUDA versions
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### Docker Image Versioning

```bash
# Use specific framework versions
docker pull tensorflow/tensorflow:2.13.0-gpu
docker pull pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
```

## Troubleshooting

### Issue 1: Package Installation Fails

```bash
# Run troubleshooter
./troubleshoot_packages.sh broken-packages

# Manual fix
sudo apt --fix-broken install    # Ubuntu/Debian
sudo yum clean all               # RHEL/CentOS
```

### Issue 2: CUDA Version Mismatch

```bash
# Check versions
nvidia-smi  # Shows driver version
nvcc --version  # Shows CUDA version

# Fix mismatch
./troubleshoot_packages.sh cuda-mismatch
```

### Issue 3: Python Package Conflicts

```bash
# Create fresh environment
python3 -m venv --clear fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue 4: Docker Permission Denied

```bash
# Fix permissions
./troubleshoot_packages.sh docker-permissions

# Or manually
sudo usermod -aG docker $USER
# Log out and back in
```

### Issue 5: GPU Not Detected in TensorFlow/PyTorch

```bash
# Run GPU troubleshooter
./troubleshoot_packages.sh gpu-not-detected

# Verify CUDA installation
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Best Practices

### 1. Always Use Virtual Environments

```bash
# Never install system-wide with pip
# BAD: sudo pip install tensorflow

# GOOD: Use virtual environment
python3 -m venv ml_env
source ml_env/bin/activate
pip install tensorflow
```

### 2. Pin Package Versions in Production

```bash
# requirements.txt for production
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3
```

### 3. Regular Package Updates

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package_name

# Update all packages in requirements
pip install --upgrade -r requirements.txt
```

### 4. Clean Up Unused Packages

```bash
# Remove unused dependencies
sudo apt autoremove    # Ubuntu/Debian
sudo yum autoremove    # RHEL/CentOS

# Clean package cache
sudo apt clean
sudo yum clean all
```

### 5. Documentation of Installed Packages

```bash
# Save Python environment
pip freeze > requirements_$(date +%Y%m%d).txt

# Save conda environment
conda env export > environment_$(date +%Y%m%d).yml

# Save system packages
dpkg --get-selections > system_packages_$(date +%Y%m%d).txt  # Debian/Ubuntu
rpm -qa > system_packages_$(date +%Y%m%d).txt              # RHEL/CentOS
```

## Real-World Use Cases

### Use Case 1: Setting Up New ML Development Machine

```bash
# 1. Install complete stack
./install_ml_stack.sh --full

# 2. Create development environment
./setup_python_env.sh ml_dev --with-gpu --framework=both

# 3. Activate environment
source ~/ml_dev/bin/activate

# 4. Install project dependencies
pip install -r my_project/requirements.txt

# 5. Validate setup
./validate_installation.sh --full
```

### Use Case 2: Installing on Production Server

```bash
# 1. Detect system
./detect_system.sh > system_info.txt

# 2. Install system packages only (no GUI tools)
./install_ml_stack.sh --system-only

# 3. Install Docker for containerized deployments
./install_docker.sh --with-nvidia-docker

# 4. Validate
./validate_installation.sh --docker --gpu
```

### Use Case 3: Upgrading ML Frameworks

```bash
# 1. Check current versions
./validate_installation.sh

# 2. Backup current environment
pip freeze > requirements_backup.txt

# 3. Upgrade frameworks
pip install --upgrade tensorflow torch

# 4. Test compatibility
python -c "import tensorflow as tf; import torch; print('OK')"

# 5. Rollback if needed
pip install -r requirements_backup.txt
```

## Integration with Previous Exercises

- **Exercise 01**: Uses file system navigation and operations
- **Exercise 02**: Applies proper permissions to scripts and packages
- **Exercise 03**: Manages package installation processes
- **Exercise 04**: Builds on bash scripting for automation
- **Future**: Foundation for Docker deployment (Exercise 08)

## Skills Acquired

- ✅ Package manager operations (apt, yum, dnf)
- ✅ System package installation and management
- ✅ Python virtual environment management
- ✅ pip package management
- ✅ Conda environment management
- ✅ CUDA and GPU driver installation
- ✅ Docker installation and configuration
- ✅ ML framework installation
- ✅ Dependency resolution
- ✅ Version management
- ✅ Troubleshooting package issues
- ✅ Creating reproducible environments

## Time to Complete

- **Setup and understanding**: 15 minutes
- **System detection script**: 20 minutes
- **ML stack installer**: 45 minutes
- **Python environment script**: 30 minutes
- **CUDA installation script**: 30 minutes
- **Docker installation script**: 20 minutes
- **Package management utilities**: 25 minutes
- **Validation script**: 20 minutes
- **Troubleshooting script**: 25 minutes
- **Testing and validation**: 20 minutes
- **Total**: 250-270 minutes (4-4.5 hours)

## Next Steps

- Complete Exercise 06: Log Analysis
- Complete Exercise 07: Troubleshooting
- Learn about system updates and patch management
- Explore Docker containerization in depth

## Resources

- [Ubuntu Package Management](https://ubuntu.com/server/docs/package-management)
- [pip User Guide](https://pip.pypa.io/en/stable/user_guide/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Docker Installation](https://docs.docker.com/engine/install/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

## Conclusion

This solution provides production-ready package management automation for ML infrastructure. The scripts demonstrate industry best practices for installing and managing complex software stacks, handling dependencies, and ensuring reproducibility - essential skills for ML infrastructure engineering.

**Key Achievement**: Complete, automated ML stack installation with dependency management, version control, and troubleshooting capabilities across multiple Linux distributions.

---

**Exercise 05: Package Management for ML Stack Installation - ✅ READY FOR IMPLEMENTATION**
