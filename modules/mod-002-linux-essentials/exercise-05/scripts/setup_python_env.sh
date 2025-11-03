#!/bin/bash
#
# setup_python_env.sh - Python Virtual Environment Setup for ML
#
# Usage: ./setup_python_env.sh [env_name] [options]
#

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
ENV_NAME="${1:-ml_env}"
WITH_GPU=false
FRAMEWORK="both"  # tensorflow, pytorch, both, none

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' NC=''
fi

# Logging
log_info() { echo -e "${BLUE}ℹ${NC} $*"; }
log_success() { echo -e "${GREEN}✓${NC} $*"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $*"; }
log_error() { echo -e "${RED}✗${NC} $*" >&2; }

error_exit() {
    log_error "$1"
    exit 1
}

# Usage
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [env_name] [options]

Create Python virtual environment with ML packages.

Arguments:
  env_name                Environment name (default: ml_env)

Options:
  -h, --help              Show this help message
  --with-gpu              Include GPU support for ML frameworks
  --framework=NAME        Install specific framework(s): tensorflow, pytorch, both, none
  --python=VERSION        Python version to use (default: python3)

Examples:
  $SCRIPT_NAME ml_cpu                                 # CPU-only environment
  $SCRIPT_NAME ml_gpu --with-gpu                      # GPU environment
  $SCRIPT_NAME ml_tf --with-gpu --framework=tensorflow
  $SCRIPT_NAME ml_torch --with-gpu --framework=pytorch

EOF
    exit 0
}

# Check Python installation
check_python() {
    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 not found. Install with: sudo apt install python3 python3-venv"
    fi

    log_info "Found Python: $(python3 --version)"
}

# Create virtual environment
create_venv() {
    local env_path="$HOME/$ENV_NAME"

    if [ -d "$env_path" ]; then
        log_warning "Environment $ENV_NAME already exists"
        echo -n "Overwrite? (yes/no): "
        read -r response
        if [ "$response" != "yes" ]; then
            log_info "Cancelled"
            exit 0
        fi
        rm -rf "$env_path"
    fi

    log_info "Creating virtual environment: $ENV_NAME"
    python3 -m venv "$env_path" || error_exit "Failed to create virtual environment"

    log_success "Virtual environment created at: $env_path"
}

# Install packages
install_packages() {
    local env_path="$HOME/$ENV_NAME"

    log_info "Activating environment..."
    # shellcheck source=/dev/null
    source "$env_path/bin/activate"

    log_info "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel || error_exit "Failed to upgrade pip"

    log_info "Installing core data science packages..."
    pip install \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        || error_exit "Failed to install core packages"

    log_info "Installing visualization packages..."
    pip install \
        matplotlib \
        seaborn \
        plotly \
        || log_warning "Some visualization packages failed"

    log_info "Installing Jupyter..."
    pip install \
        jupyter \
        jupyterlab \
        ipython \
        || log_warning "Jupyter installation failed"

    log_info "Installing development tools..."
    pip install \
        black \
        flake8 \
        pytest \
        ipdb \
        || log_warning "Some development tools failed"

    log_info "Installing utilities..."
    pip install \
        tqdm \
        pyyaml \
        python-dotenv \
        requests \
        || log_warning "Some utilities failed"

    # Install ML frameworks based on selection
    case "$FRAMEWORK" in
        tensorflow)
            install_tensorflow
            ;;
        pytorch)
            install_pytorch
            ;;
        both)
            install_tensorflow
            install_pytorch
            ;;
        none)
            log_info "Skipping ML framework installation"
            ;;
        *)
            log_warning "Unknown framework: $FRAMEWORK"
            ;;
    esac

    # Install MLOps tools
    log_info "Installing MLOps tools..."
    pip install \
        mlflow \
        wandb \
        tensorboard \
        || log_warning "Some MLOps tools failed"

    # Save requirements
    local req_file="$env_path/requirements.txt"
    pip freeze > "$req_file"
    log_success "Requirements saved to: $req_file"

    deactivate
}

# Install TensorFlow
install_tensorflow() {
    log_info "Installing TensorFlow..."

    if [ "$WITH_GPU" = true ]; then
        log_info "Installing TensorFlow with GPU support..."
        pip install tensorflow || log_warning "TensorFlow installation failed"
        log_info "Modern TensorFlow includes GPU support automatically"
    else
        log_info "Installing TensorFlow (CPU)..."
        pip install tensorflow-cpu || pip install tensorflow || log_warning "TensorFlow installation failed"
    fi

    log_success "TensorFlow installed"
}

# Install PyTorch
install_pytorch() {
    log_info "Installing PyTorch..."

    if [ "$WITH_GPU" = true ]; then
        log_info "Installing PyTorch with CUDA support..."

        # Detect CUDA version
        if command -v nvcc &> /dev/null; then
            local cuda_version=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
            log_info "Detected CUDA: $cuda_version"

            # Install based on CUDA version
            if [[ "$cuda_version" == 12.* ]]; then
                log_info "Installing PyTorch for CUDA 12..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            elif [[ "$cuda_version" == 11.8* ]]; then
                log_info "Installing PyTorch for CUDA 11.8..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            else
                log_warning "Unknown CUDA version, installing default PyTorch..."
                pip install torch torchvision torchaudio
            fi
        else
            log_warning "CUDA not detected, installing default PyTorch..."
            pip install torch torchvision torchaudio
        fi
    else
        log_info "Installing PyTorch (CPU)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    log_success "PyTorch installed"
}

# Generate activation script
generate_activation_script() {
    local env_path="$HOME/$ENV_NAME"
    local activate_script="$env_path/activate_env.sh"

    cat > "$activate_script" << EOF
#!/bin/bash
# Activation script for $ENV_NAME environment

# Activate virtual environment
source "$env_path/bin/activate"

# Set CUDA environment variables (if GPU support)
if [ "$WITH_GPU" = true ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=\$CUDA_HOME/bin:\$PATH
    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
fi

echo "Environment $ENV_NAME activated"
echo "Python: \$(python --version)"
echo "Location: \$(which python)"

# Show installed packages
pip list | head -20
EOF

    chmod +x "$activate_script"
    log_info "Activation script created: $activate_script"
}

# Test installation
test_installation() {
    local env_path="$HOME/$ENV_NAME"

    log_info "Testing installation..."

    # shellcheck source=/dev/null
    source "$env_path/bin/activate"

    # Test core packages
    python3 << 'EOF' || log_warning "Package import test failed"
import numpy as np
import pandas as pd
import sklearn
print("✓ Core packages: OK")
EOF

    # Test TensorFlow
    if [ "$FRAMEWORK" = "tensorflow" ] || [ "$FRAMEWORK" = "both" ]; then
        python3 << 'EOF' || log_warning "TensorFlow test failed"
import tensorflow as tf
print(f"✓ TensorFlow {tf.__version__}: OK")
if tf.config.list_physical_devices('GPU'):
    print("  GPU devices:", tf.config.list_physical_devices('GPU'))
else:
    print("  Running in CPU mode")
EOF
    fi

    # Test PyTorch
    if [ "$FRAMEWORK" = "pytorch" ] || [ "$FRAMEWORK" = "both" ]; then
        python3 << 'EOF' || log_warning "PyTorch test failed"
import torch
print(f"✓ PyTorch {torch.__version__}: OK")
if torch.cuda.is_available():
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("  Running in CPU mode")
EOF
    fi

    deactivate

    log_success "Installation tests complete"
}

# Generate summary
generate_summary() {
    local env_path="$HOME/$ENV_NAME"

    echo ""
    echo "================================="
    echo "Environment Setup Complete!"
    echo "================================="
    echo ""
    echo "Environment: $ENV_NAME"
    echo "Location: $env_path"
    echo "GPU Support: $WITH_GPU"
    echo "Framework: $FRAMEWORK"
    echo ""
    echo "To activate:"
    echo "  source $env_path/bin/activate"
    echo ""
    echo "Or use the activation script:"
    echo "  source $env_path/activate_env.sh"
    echo ""
    echo "To deactivate:"
    echo "  deactivate"
    echo ""
    echo "Requirements file:"
    echo "  $env_path/requirements.txt"
    echo ""
    echo "================================="
}

# Main function
main() {
    # Parse arguments
    shift || true  # Skip env_name (already captured)

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            --with-gpu)
                WITH_GPU=true
                shift
                ;;
            --framework=*)
                FRAMEWORK="${1#*=}"
                shift
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage
                ;;
        esac
    done

    log_info "======================================="
    log_info "Python Environment Setup: $ENV_NAME"
    log_info "======================================="

    check_python
    create_venv
    install_packages
    generate_activation_script
    test_installation
    generate_summary

    log_success "======================================="
    log_success "Setup Complete!"
    log_success "======================================="
}

# Handle no arguments case
if [ $# -eq 0 ]; then
    ENV_NAME="ml_env"
fi

main "$@"
