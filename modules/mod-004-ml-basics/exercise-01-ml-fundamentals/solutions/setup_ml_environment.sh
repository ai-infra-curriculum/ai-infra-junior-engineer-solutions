#!/bin/bash
#
# setup_ml_environment.sh - Automated ML environment setup
#
# Description:
#   Automates the setup of machine learning development environments
#   including Python, ML frameworks, and GPU drivers verification.
#
# Usage:
#   ./setup_ml_environment.sh [OPTIONS]
#
# Options:
#   -f, --frameworks FRAMEWORKS  Comma-separated list (pytorch,tensorflow,sklearn)
#   -p, --python VERSION        Python version (default: 3.10)
#   -n, --name ENV_NAME         Environment name (default: ml-env)
#   -g, --gpu-check             Verify GPU setup
#   -r, --requirements FILE     Install from requirements file
#   -v, --verbose               Verbose output
#   -h, --help                  Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
FRAMEWORKS="pytorch,tensorflow,sklearn"
PYTHON_VERSION="3.10"
ENV_NAME="ml-env"
GPU_CHECK=false
REQUIREMENTS_FILE=""
VERBOSE=false

# ===========================
# Colors
# ===========================

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RESET='\033[0m'
readonly BOLD='\033[1m'

# ===========================
# Logging
# ===========================

log_info() {
    echo -e "${BLUE}[INFO]${RESET} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $*"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${CYAN}[DEBUG]${RESET} $*"
    fi
}

# ===========================
# System Detection
# ===========================

detect_system() {
    echo -e "${BOLD}${CYAN}Detecting System Configuration${RESET}"
    echo "========================================"

    # OS Detection
    if [[ -f /etc/os-release ]]; then
        OS_NAME=$(grep "^NAME=" /etc/os-release | cut -d'"' -f2)
        OS_VERSION=$(grep "^VERSION=" /etc/os-release | cut -d'"' -f2)
        echo "OS: $OS_NAME $OS_VERSION"
    else
        OS_NAME=$(uname -s)
        echo "OS: $OS_NAME"
    fi

    # CPU Info
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
    CPU_CORES=$(nproc)
    echo "CPU: $CPU_MODEL"
    echo "CPU Cores: $CPU_CORES"

    # Memory
    TOTAL_MEM=$(free -h | awk 'NR==2 {print $2}')
    echo "Memory: $TOTAL_MEM"

    # Python
    if command -v python3 &> /dev/null; then
        PYTHON_CURRENT=$(python3 --version)
        echo "Python: $PYTHON_CURRENT"
    else
        log_warning "Python 3 not found"
    fi

    # GPU Detection
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        echo "GPU: $GPU_NAME (x$GPU_COUNT)"

        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        echo "CUDA: $CUDA_VERSION"
    else
        echo "GPU: Not detected"
    fi

    echo ""
}

# ===========================
# Environment Setup
# ===========================

setup_virtual_env() {
    echo -e "${BOLD}${CYAN}Setting Up Virtual Environment${RESET}"
    echo "========================================"

    # Check if environment already exists
    if [[ -d "$ENV_NAME" ]]; then
        log_warning "Environment '$ENV_NAME' already exists"
        echo "Remove it? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf "$ENV_NAME"
        else
            log_error "Environment setup cancelled"
            return 1
        fi
    fi

    # Create virtual environment
    log_info "Creating virtual environment: $ENV_NAME"
    python3 -m venv "$ENV_NAME"

    # Activate environment
    source "$ENV_NAME/bin/activate"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1

    log_success "Virtual environment created: $ENV_NAME"
    echo "  Activate with: source $ENV_NAME/bin/activate"
    echo ""
}

# ===========================
# Framework Installation
# ===========================

install_pytorch() {
    echo -e "${BOLD}Installing PyTorch${RESET}"
    echo "--------------------------------------"

    # Detect CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
        log_info "Detected CUDA version: $CUDA_VERSION"

        case "$CUDA_VERSION" in
            11.8)
                log_info "Installing PyTorch with CUDA 11.8 support..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                ;;
            12.1)
                log_info "Installing PyTorch with CUDA 12.1 support..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                ;;
            *)
                log_warning "CUDA version $CUDA_VERSION not directly supported, installing latest PyTorch..."
                pip install torch torchvision torchaudio
                ;;
        esac
    else
        log_info "No GPU detected, installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Verify installation
    if python -c "import torch; print(torch.__version__)" > /dev/null 2>&1; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        log_success "PyTorch installed: v$TORCH_VERSION (CUDA: $CUDA_AVAILABLE)"
    else
        log_error "PyTorch installation failed"
        return 1
    fi

    echo ""
}

install_tensorflow() {
    echo -e "${BOLD}Installing TensorFlow${RESET}"
    echo "--------------------------------------"

    log_info "Installing TensorFlow..."
    pip install tensorflow

    # Verify installation
    if python -c "import tensorflow as tf; print(tf.__version__)" > /dev/null 2>&1; then
        TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)")
        GPU_COUNT=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))")
        log_success "TensorFlow installed: v$TF_VERSION ($GPU_COUNT GPUs detected)"
    else
        log_error "TensorFlow installation failed"
        return 1
    fi

    echo ""
}

install_sklearn() {
    echo -e "${BOLD}Installing scikit-learn${RESET}"
    echo "--------------------------------------"

    log_info "Installing scikit-learn and dependencies..."
    pip install scikit-learn pandas numpy matplotlib seaborn

    # Verify installation
    if python -c "import sklearn; print(sklearn.__version__)" > /dev/null 2>&1; then
        SKLEARN_VERSION=$(python -c "import sklearn; print(sklearn.__version__)")
        log_success "scikit-learn installed: v$SKLEARN_VERSION"
    else
        log_error "scikit-learn installation failed"
        return 1
    fi

    echo ""
}

install_ml_tools() {
    echo -e "${BOLD}Installing ML Tools${RESET}"
    echo "--------------------------------------"

    log_info "Installing Jupyter, MLflow, TensorBoard..."
    pip install jupyter jupyterlab mlflow tensorboard tqdm

    log_success "ML tools installed"
    echo ""
}

# ===========================
# GPU Verification
# ===========================

verify_gpu_setup() {
    echo -e "${BOLD}${CYAN}Verifying GPU Setup${RESET}"
    echo "========================================"

    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found - no NVIDIA GPU or drivers not installed"
        return 1
    fi

    # NVIDIA Driver
    echo -e "${BOLD}NVIDIA Driver:${RESET}"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    echo ""

    # CUDA
    if command -v nvcc &> /dev/null; then
        echo -e "${BOLD}CUDA:${RESET}"
        nvcc --version | grep "release"
        echo ""
    fi

    # PyTorch GPU Test
    if python -c "import torch" > /dev/null 2>&1; then
        echo -e "${BOLD}PyTorch GPU Test:${RESET}"
        python3 <<'EOF'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Quick GPU test
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("  ✓ GPU computation test passed")
EOF
        echo ""
    fi

    # TensorFlow GPU Test
    if python -c "import tensorflow" > /dev/null 2>&1; then
        echo -e "${BOLD}TensorFlow GPU Test:${RESET}"
        python3 <<'EOF'
import tensorflow as tf
print(f"  TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPU devices: {len(gpus)}")
for gpu in gpus:
    print(f"    - {gpu.name}")
if gpus:
    print("  ✓ GPU detection test passed")
EOF
        echo ""
    fi
}

# ===========================
# Report Generation
# ===========================

generate_report() {
    local report_file="ml_environment_report.txt"

    cat > "$report_file" <<EOF
ML Environment Setup Report
===========================
Generated: $(date '+%Y-%m-%d %H:%M:%S')

System Information:
-------------------
OS: $(grep "^NAME=" /etc/os-release | cut -d'"' -f2 || echo "Unknown")
Kernel: $(uname -r)
CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
CPU Cores: $(nproc)
Memory: $(free -h | awk 'NR==2 {print $2}')

GPU Information:
----------------
$(if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "No NVIDIA GPU detected"
fi)

Python Environment:
-------------------
Environment: $ENV_NAME
Python: $(python --version 2>&1)
Pip: $(pip --version 2>&1)

Installed Packages:
-------------------
$(pip list)

GPU Verification:
-----------------
$(if command -v nvidia-smi &> /dev/null; then
    python3 <<'PYEOF'
import sys
try:
    import torch
    print(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError:
    print("PyTorch: Not installed")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__} (GPUs: {len(tf.config.list_physical_devices('GPU'))})")
except ImportError:
    print("TensorFlow: Not installed")

try:
    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")
except ImportError:
    print("scikit-learn: Not installed")
PYEOF
else
    echo "No GPU available"
fi)

===========================
EOF

    log_success "Report saved to: $report_file"
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Automated ML environment setup.

OPTIONS:
    -f, --frameworks FRAMEWORKS  Frameworks to install (pytorch,tensorflow,sklearn)
                                Default: $FRAMEWORKS
    -p, --python VERSION        Python version (default: $PYTHON_VERSION)
    -n, --name ENV_NAME         Environment name (default: $ENV_NAME)
    -g, --gpu-check             Verify GPU setup
    -r, --requirements FILE     Install from requirements file
    -v, --verbose               Verbose output
    -h, --help                  Display this help message

EXAMPLES:
    # Setup with all frameworks
    $SCRIPT_NAME

    # Setup PyTorch only
    $SCRIPT_NAME -f pytorch

    # Setup with GPU verification
    $SCRIPT_NAME -g

    # Custom environment name
    $SCRIPT_NAME -n my-ml-env

    # Install from requirements file
    $SCRIPT_NAME -r requirements.txt

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--frameworks)
                FRAMEWORKS="$2"
                shift 2
                ;;
            -p|--python)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            -n|--name)
                ENV_NAME="$2"
                shift 2
                ;;
            -g|--gpu-check)
                GPU_CHECK=true
                shift
                ;;
            -r|--requirements)
                REQUIREMENTS_FILE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# ===========================
# Main Function
# ===========================

main() {
    parse_arguments "$@"

    echo -e "${BOLD}${CYAN}ML Environment Setup${RESET}"
    echo "========================================"
    echo ""

    # Detect system
    detect_system

    # Setup virtual environment
    setup_virtual_env

    # Activate environment
    source "$ENV_NAME/bin/activate"

    # Install frameworks
    IFS=',' read -ra FRAMEWORK_ARRAY <<< "$FRAMEWORKS"
    for framework in "${FRAMEWORK_ARRAY[@]}"; do
        framework=$(echo "$framework" | xargs)  # Trim whitespace

        case "$framework" in
            pytorch)
                install_pytorch
                ;;
            tensorflow)
                install_tensorflow
                ;;
            sklearn|scikit-learn)
                install_sklearn
                ;;
            *)
                log_warning "Unknown framework: $framework"
                ;;
        esac
    done

    # Install ML tools
    install_ml_tools

    # Install from requirements file if provided
    if [[ -n "$REQUIREMENTS_FILE" ]] && [[ -f "$REQUIREMENTS_FILE" ]]; then
        log_info "Installing from requirements file: $REQUIREMENTS_FILE"
        pip install -r "$REQUIREMENTS_FILE"
    fi

    # Verify GPU setup
    if [[ "$GPU_CHECK" == true ]]; then
        verify_gpu_setup
    fi

    # Generate report
    generate_report

    echo ""
    echo -e "${GREEN}${BOLD}Setup Complete!${RESET}"
    echo "========================================"
    echo "Environment: $ENV_NAME"
    echo ""
    echo "To activate the environment:"
    echo "  source $ENV_NAME/bin/activate"
    echo ""
    echo "To deactivate:"
    echo "  deactivate"
    echo ""
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
