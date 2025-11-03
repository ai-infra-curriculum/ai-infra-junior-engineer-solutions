#!/bin/bash
#
# validate_installation.sh - Validate ML Stack Installation
#
# Usage: ./validate_installation.sh [options]
#

set -u

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

# Counters
PASS=0
FAIL=0
WARN=0

# Flags
CHECK_FULL=false
CHECK_GPU=false
CHECK_DOCKER=false

# Usage
usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Validate ML stack installation.

Options:
  -h, --help        Show this help message
  --full            Check all components
  --gpu             Check GPU components only
  --docker          Check Docker only

Examples:
  $(basename "$0") --full
  $(basename "$0") --gpu
  $(basename "$0") --docker

EOF
    exit 0
}

# Check command existence
check_command() {
    local cmd=$1
    local name=$2

    if command -v "$cmd" &> /dev/null; then
        local version=$($cmd --version 2>&1 | head -1)
        echo -e "  ${GREEN}✓${NC} $name: $version"
        ((PASS++))
        return 0
    else
        echo -e "  ${RED}✗${NC} $name: Not found"
        ((FAIL++))
        return 1
    fi
}

# Check system tools
check_system_tools() {
    echo -e "${BLUE}[1/7] System Tools:${NC}"

    check_command gcc "GCC"
    check_command g++ "G++"
    check_command make "Make"
    check_command cmake "CMake"
    check_command git "Git"
    check_command curl "curl"
    check_command wget "wget"

    echo ""
}

# Check Python
check_python() {
    echo -e "${BLUE}[2/7] Python:${NC}"

    if check_command python3 "Python3"; then
        # Check pip
        if command -v pip3 &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} pip: $(pip3 --version | awk '{print $2}')"
            ((PASS++))
        else
            echo -e "  ${RED}✗${NC} pip: Not found"
            ((FAIL++))
        fi

        # Check venv
        if python3 -c "import venv" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} venv module available"
            ((PASS++))
        else
            echo -e "  ${RED}✗${NC} venv module not available"
            ((FAIL++))
        fi
    fi

    # Check virtual environment
    if [ -d ~/ml_env ]; then
        echo -e "  ${GREEN}✓${NC} Virtual environment: ~/ml_env"
        ((PASS++))
    else
        echo -e "  ${YELLOW}⚠${NC} Virtual environment not found"
        ((WARN++))
    fi

    echo ""
}

# Check Python packages
check_python_packages() {
    echo -e "${BLUE}[3/7] Python Packages:${NC}"

    # Check if virtual environment exists
    if [ ! -d ~/ml_env ]; then
        echo -e "  ${YELLOW}⚠${NC} No virtual environment found, skipping package checks"
        ((WARN++))
        echo ""
        return
    fi

    # Activate and check
    # shellcheck source=/dev/null
    source ~/ml_env/bin/activate 2>/dev/null || {
        echo -e "  ${RED}✗${NC} Failed to activate virtual environment"
        ((FAIL++))
        echo ""
        return
    }

    # Core packages
    for pkg in numpy pandas scikit-learn scipy matplotlib seaborn; do
        if python3 -c "import ${pkg//-/_}" 2>/dev/null; then
            local version=$(python3 -c "import ${pkg//-/_}; print(${pkg//-/_}.__version__)" 2>/dev/null)
            echo -e "  ${GREEN}✓${NC} $pkg: $version"
            ((PASS++))
        else
            echo -e "  ${RED}✗${NC} $pkg: Not installed"
            ((FAIL++))
        fi
    done

    # Jupyter
    if python3 -c "import jupyter" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Jupyter: installed"
        ((PASS++))
    else
        echo -e "  ${YELLOW}⚠${NC} Jupyter: Not installed"
        ((WARN++))
    fi

    deactivate 2>/dev/null

    echo ""
}

# Check ML frameworks
check_ml_frameworks() {
    echo -e "${BLUE}[4/7] ML Frameworks:${NC}"

    if [ ! -d ~/ml_env ]; then
        echo -e "  ${YELLOW}⚠${NC} No virtual environment found, skipping framework checks"
        ((WARN++))
        echo ""
        return
    fi

    # shellcheck source=/dev/null
    source ~/ml_env/bin/activate 2>/dev/null || return

    # TensorFlow
    if python3 -c "import tensorflow" 2>/dev/null; then
        local tf_version=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
        echo -e "  ${GREEN}✓${NC} TensorFlow: $tf_version"
        ((PASS++))

        # Check GPU support
        if python3 -c "import tensorflow as tf; assert len(tf.config.list_physical_devices('GPU')) > 0" 2>/dev/null; then
            echo -e "    ${GREEN}✓${NC} TensorFlow GPU support: enabled"
            ((PASS++))
        else
            echo -e "    ${YELLOW}⚠${NC} TensorFlow GPU support: disabled or no GPU"
            ((WARN++))
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} TensorFlow: Not installed"
        ((WARN++))
    fi

    # PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        local torch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        echo -e "  ${GREEN}✓${NC} PyTorch: $torch_version"
        ((PASS++))

        # Check GPU support
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            local gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            echo -e "    ${GREEN}✓${NC} PyTorch CUDA: $gpu_name"
            ((PASS++))
        else
            echo -e "    ${YELLOW}⚠${NC} PyTorch CUDA: not available"
            ((WARN++))
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} PyTorch: Not installed"
        ((WARN++))
    fi

    deactivate 2>/dev/null

    echo ""
}

# Check GPU
check_gpu() {
    echo -e "${BLUE}[5/7] GPU:${NC}"

    # Check for NVIDIA GPU
    if lspci | grep -i nvidia &> /dev/null; then
        local gpu_name=$(lspci | grep -i nvidia | grep -i vga | head -1 | cut -d: -f3 | xargs)
        echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected: $gpu_name"
        ((PASS++))

        # Check nvidia-smi
        if command -v nvidia-smi &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} nvidia-smi: available"
            ((PASS++))

            local driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
            echo -e "    Driver version: $driver"

            local cuda_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
            echo -e "    CUDA capability: $cuda_cap"
        else
            echo -e "  ${RED}✗${NC} nvidia-smi: not found (driver not installed)"
            ((FAIL++))
        fi

        # Check CUDA
        if command -v nvcc &> /dev/null; then
            local cuda_version=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
            echo -e "  ${GREEN}✓${NC} CUDA toolkit: $cuda_version"
            ((PASS++))
        else
            echo -e "  ${YELLOW}⚠${NC} CUDA toolkit: not installed"
            ((WARN++))
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} No NVIDIA GPU detected"
        ((WARN++))
    fi

    echo ""
}

# Check Docker
check_docker() {
    echo -e "${BLUE}[6/7] Docker:${NC}"

    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version | awk '{print $3}' | tr -d ',')
        echo -e "  ${GREEN}✓${NC} Docker: $docker_version"
        ((PASS++))

        # Check if Docker is running
        if docker ps &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} Docker daemon: running"
            ((PASS++))
        else
            echo -e "  ${YELLOW}⚠${NC} Docker daemon: not running or permission denied"
            echo -e "    Try: sudo usermod -aG docker \$USER"
            echo -e "    Then log out and back in"
            ((WARN++))
        fi

        # Check docker-compose
        if command -v docker-compose &> /dev/null; then
            local compose_version=$(docker-compose --version | awk '{print $4}' | tr -d ',')
            echo -e "  ${GREEN}✓${NC} docker-compose: $compose_version"
            ((PASS++))
        elif docker compose version &> /dev/null; then
            local compose_version=$(docker compose version --short)
            echo -e "  ${GREEN}✓${NC} docker compose (plugin): $compose_version"
            ((PASS++))
        else
            echo -e "  ${YELLOW}⚠${NC} docker-compose: not installed"
            ((WARN++))
        fi

        # Check NVIDIA Docker (only if NVIDIA GPU present)
        if lspci | grep -i nvidia &> /dev/null && command -v nvidia-smi &> /dev/null; then
            if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
                echo -e "  ${GREEN}✓${NC} NVIDIA Container Toolkit: working"
                ((PASS++))
            else
                echo -e "  ${YELLOW}⚠${NC} NVIDIA Container Toolkit: not working or not installed"
                ((WARN++))
            fi
        fi
    else
        echo -e "  ${RED}✗${NC} Docker: not installed"
        ((FAIL++))
    fi

    echo ""
}

# Check system libraries
check_system_libraries() {
    echo -e "${BLUE}[7/7] System Libraries:${NC}"

    local libraries=("libblas.so" "liblapack.so" "libjpeg.so" "libpng.so" "libhdf5.so")
    local found=0
    local not_found=0

    for lib in "${libraries[@]}"; do
        if ldconfig -p | grep -q "$lib"; then
            ((found++))
        else
            ((not_found++))
        fi
    done

    if [ $found -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} Found $found/$((found + not_found)) system libraries"
        ((PASS++))
    fi

    if [ $not_found -gt 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} Missing $not_found/$((found + not_found)) system libraries"
        ((WARN++))
    fi

    echo ""
}

# Generate summary
generate_summary() {
    echo -e "${BLUE}=== Validation Summary ===${NC}"
    echo ""
    echo -e "  ${GREEN}Passed:${NC}   $PASS"
    echo -e "  ${RED}Failed:${NC}   $FAIL"
    echo -e "  ${YELLOW}Warnings:${NC} $WARN"
    echo ""

    if [ $FAIL -eq 0 ]; then
        echo -e "${GREEN}✓ All critical components validated!${NC}"
        echo ""

        if [ $WARN -gt 0 ]; then
            echo "Some optional components have warnings."
            echo "Review the warnings above."
        fi

        return 0
    else
        echo -e "${RED}✗ Some validations failed${NC}"
        echo ""
        echo "Fix the issues above and run validation again."
        echo ""
        return 1
    fi
}

# Main function
main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            --full)
                CHECK_FULL=true
                CHECK_GPU=true
                CHECK_DOCKER=true
                shift
                ;;
            --gpu)
                CHECK_GPU=true
                shift
                ;;
            --docker)
                CHECK_DOCKER=true
                shift
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage
                ;;
        esac
    done

    # Default to full check if no flags
    if [ "$CHECK_FULL" = false ] && [ "$CHECK_GPU" = false ] && [ "$CHECK_DOCKER" = false ]; then
        CHECK_FULL=true
        CHECK_GPU=true
        CHECK_DOCKER=true
    fi

    echo -e "${BLUE}=== ML Stack Installation Validation ===${NC}"
    echo ""

    if [ "$CHECK_FULL" = true ]; then
        check_system_tools
        check_python
        check_python_packages
        check_ml_frameworks
    fi

    if [ "$CHECK_GPU" = true ]; then
        check_gpu
    fi

    if [ "$CHECK_DOCKER" = true ]; then
        check_docker
    fi

    if [ "$CHECK_FULL" = true ]; then
        check_system_libraries
    fi

    generate_summary
}

main "$@"
