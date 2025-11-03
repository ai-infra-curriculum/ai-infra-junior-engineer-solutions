#!/bin/bash
#
# install_ml_stack.sh - Complete ML Infrastructure Stack Installation
#
# Usage: ./install_ml_stack.sh [options]
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_DIR="${SCRIPT_DIR}/../logs"
readonly LOG_FILE="${LOG_DIR}/ml_stack_install.log"

# Installation flags
INSTALL_SYSTEM=false
INSTALL_PYTHON=false
INSTALL_DOCKER=false
INSTALL_CUDA=false
DRY_RUN=false

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
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; echo -e "${BLUE}ℹ${NC} $*"; }
log_success() { log "SUCCESS" "$@"; echo -e "${GREEN}✓${NC} $*"; }
log_warning() { log "WARNING" "$@"; echo -e "${YELLOW}⚠${NC} $*"; }
log_error() { log "ERROR" "$@" >&2; echo -e "${RED}✗${NC} $*" >&2; }

error_exit() {
    log_error "$1"
    exit 1
}

# Usage
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [options]

Install complete ML infrastructure stack with all dependencies.

Options:
  -h, --help              Show this help message
  --full                  Install everything (system, python, docker, cuda)
  --system-only           Install only system packages
  --python-only           Install only Python packages
  --docker-only           Install only Docker
  --cuda-only             Install only CUDA (requires NVIDIA GPU)
  --dry-run               Show what would be installed without installing
  --yes                   Skip confirmation prompts

Examples:
  $SCRIPT_NAME --full               # Install complete stack
  $SCRIPT_NAME --system-only        # Install system packages only
  $SCRIPT_NAME --python-only        # Install Python packages only
  $SCRIPT_NAME --dry-run --full     # Show what would be installed

EOF
    exit 0
}

# Check if running as root
check_not_root() {
    if [ "$EUID" -eq 0 ]; then
        error_exit "Do not run as root. Script will use sudo when needed."
    fi
}

# Detect distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        DISTRO_ID=$ID
        DISTRO_NAME=$NAME
        DISTRO_VERSION=$VERSION_ID
        log_info "Detected: $DISTRO_NAME $DISTRO_VERSION"
    else
        error_exit "Cannot detect Linux distribution"
    fi
}

# Install system packages
install_system_packages() {
    log_info "Installing system packages..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would install system packages"
        return 0
    fi

    case $DISTRO_ID in
        ubuntu|debian)
            log_info "Updating package lists..."
            sudo apt update || error_exit "Failed to update package lists"

            log_info "Installing build essentials..."
            sudo apt install -y \
                build-essential \
                gcc g++ make \
                cmake \
                gdb \
                || error_exit "Failed to install build tools"

            log_info "Installing Python development packages..."
            sudo apt install -y \
                python3 \
                python3-pip \
                python3-dev \
                python3-venv \
                python3-wheel \
                python3-setuptools \
                || error_exit "Failed to install Python packages"

            log_info "Installing system libraries..."
            sudo apt install -y \
                libopenblas-dev \
                liblapack-dev \
                libblas-dev \
                libhdf5-dev \
                libjpeg-dev \
                libpng-dev \
                libfreetype6-dev \
                zlib1g-dev \
                liblzma-dev \
                libbz2-dev \
                || log_warning "Some libraries failed to install"

            log_info "Installing utilities..."
            sudo apt install -y \
                git \
                curl \
                wget \
                vim \
                htop \
                iotop \
                net-tools \
                ca-certificates \
                gnupg \
                lsb-release \
                || log_warning "Some utilities failed to install"

            log_success "System packages installed"
            ;;

        centos|rhel|fedora)
            log_info "Installing EPEL repository..."
            sudo yum install -y epel-release || log_warning "EPEL install failed"

            log_info "Installing Development Tools..."
            sudo yum groupinstall -y "Development Tools" || error_exit "Failed to install Development Tools"

            log_info "Installing Python..."
            sudo yum install -y \
                python3 \
                python3-pip \
                python3-devel \
                || error_exit "Failed to install Python"

            log_info "Installing system libraries..."
            sudo yum install -y \
                openblas-devel \
                lapack-devel \
                hdf5-devel \
                libjpeg-devel \
                libpng-devel \
                zlib-devel \
                xz-devel \
                bzip2-devel \
                || log_warning "Some libraries failed to install"

            log_info "Installing utilities..."
            sudo yum install -y \
                git \
                curl \
                wget \
                vim \
                htop \
                net-tools \
                || log_warning "Some utilities failed to install"

            log_success "System packages installed"
            ;;

        *)
            error_exit "Unsupported distribution: $DISTRO_ID"
            ;;
    esac
}

# Install Python packages
install_python_packages() {
    log_info "Installing Python packages..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would install Python packages"
        return 0
    fi

    # Create virtual environment
    if [ ! -d ~/ml_env ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv ~/ml_env || error_exit "Failed to create virtual environment"
    else
        log_info "Virtual environment already exists"
    fi

    # Activate environment
    # shellcheck source=/dev/null
    source ~/ml_env/bin/activate

    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel || error_exit "Failed to upgrade pip"

    log_info "Installing core ML packages..."
    pip install numpy pandas scikit-learn scipy || error_exit "Failed to install core packages"

    log_info "Installing visualization packages..."
    pip install matplotlib seaborn plotly || log_warning "Some visualization packages failed"

    log_info "Installing ML frameworks..."
    pip install tensorflow torch torchvision torchaudio || log_warning "Some ML frameworks failed to install"

    log_info "Installing ML utilities..."
    pip install \
        jupyter \
        jupyterlab \
        ipython \
        mlflow \
        wandb \
        tqdm \
        pyyaml \
        python-dotenv \
        requests \
        || log_warning "Some utilities failed to install"

    # Save requirements
    pip freeze > ~/ml_env_requirements.txt
    log_info "Requirements saved to ~/ml_env_requirements.txt"

    deactivate

    log_success "Python packages installed"
}

# Install Docker
install_docker() {
    log_info "Installing Docker..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would install Docker"
        return 0
    fi

    # Check if Docker is already installed
    if command -v docker &> /dev/null; then
        log_info "Docker already installed: $(docker --version)"
        return 0
    fi

    case $DISTRO_ID in
        ubuntu|debian)
            log_info "Adding Docker repository..."

            # Add Docker's official GPG key
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
                sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

            # Set up repository
            echo \
                "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
                https://download.docker.com/linux/ubuntu \
                $(lsb_release -cs) stable" | \
                sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

            # Update package index
            sudo apt update

            # Install Docker
            log_info "Installing Docker packages..."
            sudo apt install -y \
                docker-ce \
                docker-ce-cli \
                containerd.io \
                docker-buildx-plugin \
                docker-compose-plugin \
                || error_exit "Failed to install Docker"

            # Start Docker service
            sudo systemctl start docker
            sudo systemctl enable docker

            # Add user to docker group
            sudo usermod -aG docker "$USER"

            log_success "Docker installed successfully"
            log_warning "Log out and back in for Docker group changes to take effect"
            ;;

        centos|rhel|fedora)
            log_info "Installing Docker from repository..."

            # Remove old versions
            sudo yum remove -y docker docker-client docker-client-latest docker-common \
                docker-latest docker-latest-logrotate docker-logrotate docker-engine || true

            # Install dependencies
            sudo yum install -y yum-utils

            # Add Docker repository
            sudo yum-config-manager --add-repo \
                https://download.docker.com/linux/centos/docker-ce.repo

            # Install Docker
            sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

            # Start Docker
            sudo systemctl start docker
            sudo systemctl enable docker

            # Add user to docker group
            sudo usermod -aG docker "$USER"

            log_success "Docker installed successfully"
            log_warning "Log out and back in for Docker group changes to take effect"
            ;;

        *)
            log_warning "Docker installation not supported for $DISTRO_ID in this script"
            ;;
    esac
}

# Install CUDA
install_cuda() {
    log_info "Installing CUDA and NVIDIA drivers..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would install CUDA"
        return 0
    fi

    # Check if NVIDIA GPU is present
    if ! lspci | grep -i nvidia &> /dev/null; then
        log_warning "No NVIDIA GPU detected. Skipping CUDA installation."
        return 0
    fi

    log_info "NVIDIA GPU detected"

    case $DISTRO_ID in
        ubuntu|debian)
            log_info "Detecting available NVIDIA drivers..."

            # Check if ubuntu-drivers is available
            if command -v ubuntu-drivers &> /dev/null; then
                log_info "Installing recommended NVIDIA driver..."
                sudo ubuntu-drivers autoinstall || log_warning "Failed to auto-install drivers"
            else
                log_warning "ubuntu-drivers not available, installing manually..."
                # Add NVIDIA repository
                wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb || {
                    log_error "Failed to download CUDA keyring"
                    return 1
                }
                sudo dpkg -i cuda-keyring_1.1-1_all.deb
                rm cuda-keyring_1.1-1_all.deb

                sudo apt update

                # Install CUDA toolkit
                sudo apt install -y cuda-toolkit-12-2 || log_warning "Failed to install CUDA toolkit"

                # Install NVIDIA driver
                sudo apt install -y nvidia-driver-535 || log_warning "Failed to install NVIDIA driver"
            fi

            log_success "CUDA installation complete"
            log_warning "Reboot required for changes to take effect"
            ;;

        centos|rhel|fedora)
            log_info "Adding NVIDIA CUDA repository..."

            # Add CUDA repository
            sudo yum-config-manager --add-repo \
                https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

            # Install CUDA
            sudo yum install -y cuda || log_warning "Failed to install CUDA"

            log_success "CUDA installation complete"
            log_warning "Reboot required for changes to take effect"
            ;;

        *)
            log_warning "CUDA installation not supported for $DISTRO_ID in this script"
            ;;
    esac

    # Add CUDA to PATH (add to ~/.bashrc)
    if ! grep -q "cuda" ~/.bashrc; then
        {
            echo ""
            echo "# CUDA Configuration"
            echo "export PATH=/usr/local/cuda/bin:\$PATH"
            echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
        } >> ~/.bashrc
        log_info "CUDA paths added to ~/.bashrc"
    fi
}

# Generate installation report
generate_report() {
    local report_file="${LOG_DIR}/installation_report.txt"

    {
        echo "=================================="
        echo "ML Stack Installation Report"
        echo "=================================="
        echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "User: $(whoami)"
        echo "Host: $(hostname)"
        echo ""

        echo "System Information:"
        echo "-------------------"
        echo "Distribution: $DISTRO_NAME $DISTRO_VERSION"
        echo "Kernel: $(uname -r)"
        echo "Architecture: $(uname -m)"
        echo ""

        echo "Installed Components:"
        echo "---------------------"

        if command -v gcc &> /dev/null; then
            echo "✓ GCC: $(gcc --version | head -1)"
        fi

        if command -v python3 &> /dev/null; then
            echo "✓ Python: $(python3 --version)"
        fi

        if command -v pip3 &> /dev/null; then
            echo "✓ pip: $(pip3 --version)"
        fi

        if [ -d ~/ml_env ]; then
            echo "✓ Python virtual environment: ~/ml_env"
        fi

        if command -v docker &> /dev/null; then
            echo "✓ Docker: $(docker --version)"
        fi

        if command -v nvidia-smi &> /dev/null; then
            echo "✓ NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
        fi

        if command -v nvcc &> /dev/null; then
            echo "✓ CUDA: $(nvcc --version | grep release | awk '{print $5}')"
        fi

        echo ""
        echo "Next Steps:"
        echo "-----------"
        echo "1. Activate Python environment: source ~/ml_env/bin/activate"
        echo "2. Log out and back in for Docker group changes"
        echo "3. Reboot if CUDA was installed"
        echo "4. Run validation: ./validate_installation.sh --full"
        echo ""

    } | tee "$report_file"

    log_info "Report saved to: $report_file"
}

# Main installation workflow
main() {
    # Create log directory
    mkdir -p "$LOG_DIR"

    log_info "========================================="
    log_info "ML Infrastructure Stack Installation"
    log_info "========================================="

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            --full)
                INSTALL_SYSTEM=true
                INSTALL_PYTHON=true
                INSTALL_DOCKER=true
                INSTALL_CUDA=true
                shift
                ;;
            --system-only)
                INSTALL_SYSTEM=true
                shift
                ;;
            --python-only)
                INSTALL_PYTHON=true
                shift
                ;;
            --docker-only)
                INSTALL_DOCKER=true
                shift
                ;;
            --cuda-only)
                INSTALL_CUDA=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --yes)
                # Skip confirmations
                shift
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage
                ;;
        esac
    done

    # If no flags set, default to full install
    if [ "$INSTALL_SYSTEM" = false ] && [ "$INSTALL_PYTHON" = false ] && \
       [ "$INSTALL_DOCKER" = false ] && [ "$INSTALL_CUDA" = false ]; then
        INSTALL_SYSTEM=true
        INSTALL_PYTHON=true
        INSTALL_DOCKER=true
        INSTALL_CUDA=true
    fi

    # Checks
    check_not_root
    detect_distro

    # Installation steps
    if [ "$INSTALL_SYSTEM" = true ]; then
        install_system_packages
    fi

    if [ "$INSTALL_PYTHON" = true ]; then
        install_python_packages
    fi

    if [ "$INSTALL_DOCKER" = true ]; then
        install_docker
    fi

    if [ "$INSTALL_CUDA" = true ]; then
        install_cuda
    fi

    # Generate report
    generate_report

    log_success "========================================="
    log_success "Installation Complete!"
    log_success "========================================="

    if [ "$DRY_RUN" = true ]; then
        log_info "This was a dry run. No changes were made."
    fi
}

# Run main
main "$@"
