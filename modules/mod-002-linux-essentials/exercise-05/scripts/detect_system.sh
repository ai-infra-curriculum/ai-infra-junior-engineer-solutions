#!/bin/bash
#
# detect_system.sh - System Detection and Analysis for ML Stack Installation
#
# Usage: ./detect_system.sh [--report-file=path]
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
REPORT_FILE="${REPORT_FILE:-system_detection_report.txt}"

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

# Usage
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [options]

Detect Linux distribution, package managers, and system capabilities for ML stack installation.

Options:
  -h, --help              Show this help message
  --report-file=FILE      Save report to FILE (default: system_detection_report.txt)
  --json                  Output in JSON format
  --quiet                 Minimal output

Examples:
  $SCRIPT_NAME
  $SCRIPT_NAME --report-file=/tmp/system_info.txt
  $SCRIPT_NAME --json > system.json

EOF
    exit 0
}

# Detect Linux distribution
detect_distribution() {
    if [ -f /etc/os-release ]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        echo "DISTRO_ID=$ID"
        echo "DISTRO_NAME=$NAME"
        echo "DISTRO_VERSION=$VERSION_ID"
        echo "DISTRO_VERSION_CODENAME=${VERSION_CODENAME:-unknown}"
        echo "DISTRO_PRETTY=$PRETTY_NAME"
    elif [ -f /etc/redhat-release ]; then
        local release=$(cat /etc/redhat-release)
        echo "DISTRO_ID=rhel"
        echo "DISTRO_NAME=Red Hat Enterprise Linux"
        echo "DISTRO_VERSION=unknown"
        echo "DISTRO_VERSION_CODENAME=unknown"
        echo "DISTRO_PRETTY=$release"
    else
        echo "DISTRO_ID=unknown"
        echo "DISTRO_NAME=Unknown"
        echo "DISTRO_VERSION=unknown"
        echo "DISTRO_VERSION_CODENAME=unknown"
        echo "DISTRO_PRETTY=Unknown Linux Distribution"
    fi
}

# Detect package manager
detect_package_manager() {
    local distro_id="$1"

    case "$distro_id" in
        ubuntu|debian|linuxmint|pop)
            if command -v apt &> /dev/null; then
                echo "PKG_MANAGER=apt"
                echo "PKG_UPDATE_CMD=sudo apt update"
                echo "PKG_INSTALL_CMD=sudo apt install -y"
                echo "PKG_REMOVE_CMD=sudo apt remove"
                echo "PKG_SEARCH_CMD=apt search"
                echo "PKG_INFO_CMD=apt show"
            else
                echo "PKG_MANAGER=unknown"
            fi
            ;;
        centos|rhel|fedora|rocky|alma)
            if command -v dnf &> /dev/null; then
                echo "PKG_MANAGER=dnf"
                echo "PKG_UPDATE_CMD=sudo dnf check-update"
                echo "PKG_INSTALL_CMD=sudo dnf install -y"
                echo "PKG_REMOVE_CMD=sudo dnf remove"
                echo "PKG_SEARCH_CMD=dnf search"
                echo "PKG_INFO_CMD=dnf info"
            elif command -v yum &> /dev/null; then
                echo "PKG_MANAGER=yum"
                echo "PKG_UPDATE_CMD=sudo yum check-update"
                echo "PKG_INSTALL_CMD=sudo yum install -y"
                echo "PKG_REMOVE_CMD=sudo yum remove"
                echo "PKG_SEARCH_CMD=yum search"
                echo "PKG_INFO_CMD=yum info"
            else
                echo "PKG_MANAGER=unknown"
            fi
            ;;
        arch|manjaro)
            if command -v pacman &> /dev/null; then
                echo "PKG_MANAGER=pacman"
                echo "PKG_UPDATE_CMD=sudo pacman -Syu"
                echo "PKG_INSTALL_CMD=sudo pacman -S"
                echo "PKG_REMOVE_CMD=sudo pacman -R"
                echo "PKG_SEARCH_CMD=pacman -Ss"
                echo "PKG_INFO_CMD=pacman -Si"
            else
                echo "PKG_MANAGER=unknown"
            fi
            ;;
        *)
            echo "PKG_MANAGER=unknown"
            ;;
    esac
}

# Check available package managers
check_package_managers() {
    echo "AVAILABLE_PKG_MANAGERS="
    local managers=()

    for pm in apt apt-get dpkg yum dnf rpm zypper pacman; do
        if command -v "$pm" &> /dev/null; then
            local pm_path=$(which "$pm")
            managers+=("$pm:$pm_path")
        fi
    done

    echo "${managers[@]}"
}

# Detect system architecture
detect_architecture() {
    echo "ARCH=$(uname -m)"
    echo "KERNEL=$(uname -r)"
    echo "OS=$(uname -s)"
    echo "HOSTNAME=$(hostname)"
}

# Detect Python installations
detect_python() {
    local python_versions=()

    for py in python python2 python3 python3.8 python3.9 python3.10 python3.11 python3.12; do
        if command -v "$py" &> /dev/null; then
            local version=$($py --version 2>&1 | awk '{print $2}')
            local path=$(which "$py")
            python_versions+=("$py:$version:$path")
        fi
    done

    echo "PYTHON_VERSIONS=${python_versions[@]:-none}"

    # Check pip
    if command -v pip &> /dev/null; then
        echo "PIP_VERSION=$(pip --version | awk '{print $2}')"
        echo "PIP_PATH=$(which pip)"
    else
        echo "PIP_VERSION=not_installed"
    fi

    # Check pip3
    if command -v pip3 &> /dev/null; then
        echo "PIP3_VERSION=$(pip3 --version | awk '{print $2}')"
        echo "PIP3_PATH=$(which pip3)"
    else
        echo "PIP3_VERSION=not_installed"
    fi

    # Check conda
    if command -v conda &> /dev/null; then
        echo "CONDA_VERSION=$(conda --version | awk '{print $2}')"
        echo "CONDA_PATH=$(which conda)"
    else
        echo "CONDA_VERSION=not_installed"
    fi
}

# Detect GPU
detect_gpu() {
    # Check for NVIDIA GPU
    if lspci | grep -i nvidia &> /dev/null; then
        echo "GPU_VENDOR=NVIDIA"

        # Get GPU name
        local gpu_name=$(lspci | grep -i nvidia | grep -i vga | head -1 | cut -d: -f3 | xargs)
        echo "GPU_NAME=$gpu_name"

        # Check nvidia-smi
        if command -v nvidia-smi &> /dev/null; then
            local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
            local cuda_version=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
            echo "GPU_DRIVER=$driver_version"
            echo "GPU_CUDA_CAP=$cuda_version"
            echo "GPU_DRIVER_INSTALLED=yes"
        else
            echo "GPU_DRIVER_INSTALLED=no"
        fi

        # Check CUDA
        if command -v nvcc &> /dev/null; then
            local cuda_version=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
            echo "CUDA_VERSION=$cuda_version"
            echo "CUDA_INSTALLED=yes"
        else
            echo "CUDA_INSTALLED=no"
        fi
    elif lspci | grep -i amd | grep -i vga &> /dev/null; then
        echo "GPU_VENDOR=AMD"
        local gpu_name=$(lspci | grep -i amd | grep -i vga | head -1 | cut -d: -f3 | xargs)
        echo "GPU_NAME=$gpu_name"
        echo "GPU_DRIVER_INSTALLED=unknown"
    elif lspci | grep -i intel | grep -i vga &> /dev/null; then
        echo "GPU_VENDOR=Intel"
        local gpu_name=$(lspci | grep -i intel | grep -i vga | head -1 | cut -d: -f3 | xargs)
        echo "GPU_NAME=$gpu_name"
        echo "GPU_DRIVER_INSTALLED=unknown"
    else
        echo "GPU_VENDOR=none"
        echo "GPU_DRIVER_INSTALLED=no"
    fi
}

# Detect Docker
detect_docker() {
    if command -v docker &> /dev/null; then
        echo "DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')"
        echo "DOCKER_INSTALLED=yes"
        echo "DOCKER_PATH=$(which docker)"

        # Check if Docker is running
        if docker ps &> /dev/null; then
            echo "DOCKER_RUNNING=yes"
        else
            echo "DOCKER_RUNNING=no"
        fi

        # Check docker-compose
        if command -v docker-compose &> /dev/null; then
            echo "DOCKER_COMPOSE_VERSION=$(docker-compose --version | awk '{print $4}' | tr -d ',')"
            echo "DOCKER_COMPOSE_INSTALLED=yes"
        elif docker compose version &> /dev/null; then
            echo "DOCKER_COMPOSE_VERSION=$(docker compose version --short)"
            echo "DOCKER_COMPOSE_INSTALLED=yes (plugin)"
        else
            echo "DOCKER_COMPOSE_INSTALLED=no"
        fi

        # Check NVIDIA Docker
        if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
            echo "NVIDIA_DOCKER=yes"
        else
            echo "NVIDIA_DOCKER=no"
        fi
    else
        echo "DOCKER_INSTALLED=no"
    fi
}

# Detect system resources
detect_resources() {
    # CPU
    echo "CPU_CORES=$(nproc)"
    echo "CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"

    # Memory
    local total_mem=$(free -h | awk '/^Mem:/ {print $2}')
    local avail_mem=$(free -h | awk '/^Mem:/ {print $7}')
    echo "MEMORY_TOTAL=$total_mem"
    echo "MEMORY_AVAILABLE=$avail_mem"

    # Disk
    local root_total=$(df -h / | awk 'NR==2 {print $2}')
    local root_avail=$(df -h / | awk 'NR==2 {print $4}')
    echo "DISK_TOTAL=$root_total"
    echo "DISK_AVAILABLE=$root_avail"
}

# Detect ML frameworks
detect_ml_frameworks() {
    # TensorFlow
    if python3 -c "import tensorflow" 2>/dev/null; then
        local tf_version=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
        echo "TENSORFLOW_VERSION=$tf_version"
        echo "TENSORFLOW_INSTALLED=yes"
    else
        echo "TENSORFLOW_INSTALLED=no"
    fi

    # PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        local torch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        echo "PYTORCH_VERSION=$torch_version"
        echo "PYTORCH_INSTALLED=yes"
    else
        echo "PYTORCH_INSTALLED=no"
    fi

    # scikit-learn
    if python3 -c "import sklearn" 2>/dev/null; then
        local sklearn_version=$(python3 -c "import sklearn; print(sklearn.__version__)" 2>/dev/null)
        echo "SKLEARN_VERSION=$sklearn_version"
        echo "SKLEARN_INSTALLED=yes"
    else
        echo "SKLEARN_INSTALLED=no"
    fi
}

# Generate human-readable report
generate_report() {
    local output_file="$1"

    {
        echo "==================================="
        echo "ML Stack System Detection Report"
        echo "==================================="
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        echo "Distribution Information:"
        echo "-------------------------"
        eval "$(detect_distribution)"
        echo "  Name: $DISTRO_PRETTY"
        echo "  ID: $DISTRO_ID"
        echo "  Version: $DISTRO_VERSION"
        echo ""

        echo "Package Manager:"
        echo "----------------"
        eval "$(detect_package_manager "$DISTRO_ID")"
        echo "  Manager: $PKG_MANAGER"
        echo "  Update: $PKG_UPDATE_CMD"
        echo "  Install: $PKG_INSTALL_CMD"
        echo ""

        echo "System Architecture:"
        echo "--------------------"
        eval "$(detect_architecture)"
        echo "  Architecture: $ARCH"
        echo "  Kernel: $KERNEL"
        echo "  Hostname: $HOSTNAME"
        echo ""

        echo "Python Installations:"
        echo "---------------------"
        eval "$(detect_python)"
        echo "  Python versions: ${PYTHON_VERSIONS//:/ }"
        echo "  pip: $PIP_VERSION"
        echo "  pip3: $PIP3_VERSION"
        echo "  conda: $CONDA_VERSION"
        echo ""

        echo "GPU Detection:"
        echo "--------------"
        eval "$(detect_gpu)"
        echo "  Vendor: ${GPU_VENDOR:-none}"
        if [ "${GPU_VENDOR:-none}" != "none" ]; then
            echo "  Name: ${GPU_NAME:-unknown}"
            echo "  Driver: ${GPU_DRIVER_INSTALLED:-unknown}"
            [ "${CUDA_INSTALLED:-no}" = "yes" ] && echo "  CUDA: $CUDA_VERSION"
        fi
        echo ""

        echo "Docker:"
        echo "-------"
        eval "$(detect_docker)"
        echo "  Installed: ${DOCKER_INSTALLED:-no}"
        if [ "${DOCKER_INSTALLED:-no}" = "yes" ]; then
            echo "  Version: $DOCKER_VERSION"
            echo "  Running: ${DOCKER_RUNNING:-no}"
            echo "  docker-compose: ${DOCKER_COMPOSE_INSTALLED:-no}"
            echo "  NVIDIA Docker: ${NVIDIA_DOCKER:-no}"
        fi
        echo ""

        echo "System Resources:"
        echo "-----------------"
        eval "$(detect_resources)"
        echo "  CPU Cores: $CPU_CORES"
        echo "  Memory: $MEMORY_TOTAL total, $MEMORY_AVAILABLE available"
        echo "  Disk: $DISK_TOTAL total, $DISK_AVAILABLE available"
        echo ""

        echo "ML Frameworks:"
        echo "--------------"
        eval "$(detect_ml_frameworks)"
        echo "  TensorFlow: ${TENSORFLOW_INSTALLED:-no}"
        [ "${TENSORFLOW_INSTALLED:-no}" = "yes" ] && echo "    Version: $TENSORFLOW_VERSION"
        echo "  PyTorch: ${PYTORCH_INSTALLED:-no}"
        [ "${PYTORCH_INSTALLED:-no}" = "yes" ] && echo "    Version: $PYTORCH_VERSION"
        echo "  scikit-learn: ${SKLEARN_INSTALLED:-no}"
        [ "${SKLEARN_INSTALLED:-no}" = "yes" ] && echo "    Version: $SKLEARN_VERSION"
        echo ""

        echo "Recommendations:"
        echo "----------------"

        # Check Python
        if [ "$PIP3_VERSION" = "not_installed" ]; then
            echo "  ⚠ Install Python 3 and pip3"
        else
            echo "  ✓ Python 3 and pip3 installed"
        fi

        # Check GPU
        if [ "${GPU_VENDOR:-none}" = "NVIDIA" ] && [ "${GPU_DRIVER_INSTALLED:-no}" = "no" ]; then
            echo "  ⚠ NVIDIA GPU detected but driver not installed"
        elif [ "${GPU_VENDOR:-none}" = "NVIDIA" ] && [ "${GPU_DRIVER_INSTALLED:-no}" = "yes" ]; then
            echo "  ✓ NVIDIA GPU and driver installed"
        fi

        # Check Docker
        if [ "${DOCKER_INSTALLED:-no}" = "no" ]; then
            echo "  ⚠ Docker not installed"
        else
            echo "  ✓ Docker installed"
        fi

        echo ""
        echo "==================================="
        echo "End of Report"
        echo "==================================="
    } | tee "$output_file"
}

# Main function
main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            --report-file=*)
                REPORT_FILE="${1#*=}"
                shift
                ;;
            --json)
                # TODO: Implement JSON output
                echo "JSON output not yet implemented" >&2
                exit 1
                ;;
            --quiet)
                exec > /dev/null
                shift
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage
                ;;
        esac
    done

    echo -e "${BLUE}Detecting system configuration...${NC}"
    generate_report "$REPORT_FILE"
    echo -e "${GREEN}Report saved to: $REPORT_FILE${NC}"
}

main "$@"
