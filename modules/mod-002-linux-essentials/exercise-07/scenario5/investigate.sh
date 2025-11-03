#!/bin/bash
###############################################################################
# Scenario 5: CUDA/GPU Not Available - Investigation Script
###############################################################################
#
# Problem: RuntimeError: CUDA not available, or GPU not detected
# Common causes: Missing drivers, wrong CUDA version, PATH issues
#

set -u

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
fi

section() { echo -e "\n${BOLD}${BLUE}=== $* ===${NC}\n"; }
subsection() { echo -e "${CYAN}$*${NC}"; }
log_info() { echo -e "  $*"; }
log_error() { echo -e "  ${RED}✗${NC} $*"; }
log_success() { echo -e "  ${GREEN}✓${NC} $*"; }
log_warning() { echo -e "  ${YELLOW}⚠${NC} $*"; }

echo -e "${BOLD}${RED}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${RED}║  Scenario 5: CUDA/GPU Not Available Investigation         ║${NC}"
echo -e "${BOLD}${RED}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Problem:${NC} GPU not detected or CUDA not available"
echo ""

section "Step 1: Check if NVIDIA GPU is Present"
echo "Command: lspci | grep -i nvidia"
echo ""

if lspci | grep -i nvidia; then
    log_success "NVIDIA GPU detected"
    echo ""

    # Get GPU details
    subsection "GPU details:"
    lspci -v | grep -A 10 -i nvidia | head -15
else
    log_error "No NVIDIA GPU detected!"
    echo ""
    log_info "Possible causes:"
    log_info "  - No NVIDIA GPU installed"
    log_info "  - GPU not properly seated in PCIe slot"
    log_info "  - GPU disabled in BIOS"
    log_info "  - Running in a VM without GPU passthrough"
    echo ""
    echo "If you're in a cloud environment, ensure you launched a GPU instance."
    echo ""
    exit 1
fi

section "Step 2: Check NVIDIA Driver"
subsection "nvidia-smi (driver utility):"
echo ""

if command -v nvidia-smi &>/dev/null; then
    log_success "nvidia-smi found"
    echo ""

    if nvidia-smi; then
        log_success "NVIDIA driver is working"
        echo ""

        # Extract driver version
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        log_info "Driver version: $driver_version"
    else
        log_error "nvidia-smi failed to run"
        echo ""
        log_info "Driver may be installed but not loaded properly"
        log_info "Try: sudo modprobe nvidia"
    fi
else
    log_error "nvidia-smi not found"
    echo ""
    log_info "NVIDIA driver is not installed or not in PATH"
    log_info "Install with: sudo apt install nvidia-driver-XXX"
fi

section "Step 3: Check NVIDIA Kernel Modules"
echo "Loaded NVIDIA modules:"
echo ""

if lsmod | grep -i nvidia; then
    log_success "NVIDIA kernel modules loaded"
else
    log_error "NVIDIA kernel modules not loaded!"
    echo ""
    log_info "Try loading manually:"
    log_info "  sudo modprobe nvidia"
    log_info "  sudo modprobe nvidia_uvm"
fi

section "Step 4: Check CUDA Installation"
subsection "CUDA toolkit:"
echo ""

if command -v nvcc &>/dev/null; then
    log_success "nvcc (CUDA compiler) found"
    echo ""

    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    log_info "CUDA version: $nvcc_version"
    echo ""
    log_info "CUDA installation:"
    which nvcc
else
    log_warning "nvcc not found in PATH"
    echo ""

    # Check common CUDA installation locations
    cuda_paths=("/usr/local/cuda" "/usr/lib/cuda" "/opt/cuda")
    cuda_found=false

    for path in "${cuda_paths[@]}"; do
        if [ -d "$path" ]; then
            log_info "Found CUDA at: $path"
            cuda_found=true

            if [ -f "$path/bin/nvcc" ]; then
                log_info "nvcc exists at: $path/bin/nvcc"
                log_warning "But not in PATH!"
            fi
        fi
    done

    if [ "$cuda_found" = false ]; then
        log_error "CUDA toolkit not found"
        log_info "Install with:"
        log_info "  sudo apt install nvidia-cuda-toolkit"
        log_info "Or download from: https://developer.nvidia.com/cuda-downloads"
    fi
fi

section "Step 5: Check Environment Variables"
subsection "CUDA-related environment variables:"
echo ""

if [ -n "${CUDA_HOME:-}" ]; then
    log_success "CUDA_HOME: $CUDA_HOME"
else
    log_warning "CUDA_HOME not set"
fi

if [ -n "${CUDA_PATH:-}" ]; then
    log_success "CUDA_PATH: $CUDA_PATH"
else
    log_warning "CUDA_PATH not set"
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    log_info "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

    if echo "$LD_LIBRARY_PATH" | grep -q "cuda"; then
        log_success "  Contains CUDA paths"
    else
        log_warning "  Does not contain CUDA library paths"
    fi
else
    log_warning "LD_LIBRARY_PATH not set"
fi

echo ""
subsection "PATH:"
echo "$PATH" | tr ':' '\n' | grep -i cuda || log_warning "No CUDA paths in PATH"

section "Step 6: Check CUDA Libraries"
subsection "Looking for libcudart.so (CUDA runtime):"
echo ""

cuda_lib_paths=(
    "/usr/local/cuda/lib64"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
)

libcudart_found=false
for path in "${cuda_lib_paths[@]}"; do
    if [ -f "$path/libcudart.so" ]; then
        log_success "Found: $path/libcudart.so"
        libcudart_found=true
    fi
done

if [ "$libcudart_found" = false ]; then
    log_error "libcudart.so not found"
    echo ""
    log_info "Search manually with:"
    log_info "  find /usr -name 'libcudart.so*' 2>/dev/null"
fi

section "Step 7: Check Python/PyTorch/TensorFlow"
subsection "Python environment:"
echo ""

if command -v python3 &>/dev/null; then
    python_version=$(python3 --version)
    log_info "Python: $python_version"
    echo ""

    # Check PyTorch
    log_info "Checking PyTorch CUDA support..."
    python3 << 'EOF'
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version (PyTorch): {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ PyTorch cannot access CUDA")
except ImportError:
    print("  PyTorch not installed")
except Exception as e:
    print(f"  Error: {e}")
EOF

    echo ""

    # Check TensorFlow
    log_info "Checking TensorFlow GPU support..."
    python3 << 'EOF'
try:
    import tensorflow as tf
    print(f"  TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPUs available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"    {gpu}")
    else:
        print("  ⚠ TensorFlow cannot access GPU")
except ImportError:
    print("  TensorFlow not installed")
except Exception as e:
    print(f"  Error: {e}")
EOF

else
    log_warning "Python3 not found"
fi

section "Step 8: Check GPU Permissions"
subsection "Device files:"
echo ""

if [ -c /dev/nvidia0 ]; then
    log_success "/dev/nvidia0 exists"
    ls -l /dev/nvidia* | head -5
    echo ""

    # Check if current user has access
    if [ -r /dev/nvidia0 ] && [ -w /dev/nvidia0 ]; then
        log_success "Current user has access to GPU device"
    else
        log_error "Current user cannot access GPU device"
        echo ""
        log_info "Add user to video/render group:"
        log_info "  sudo usermod -aG video $USER"
        log_info "  sudo usermod -aG render $USER"
        log_info "Then log out and back in"
    fi
else
    log_error "/dev/nvidia0 not found"
    log_info "Driver may not be loaded"
fi

section "Step 9: Check for Conflicts"
subsection "Nouveau driver (conflicts with NVIDIA):"
echo ""

if lsmod | grep -i nouveau; then
    log_error "Nouveau driver is loaded!"
    log_warning "This conflicts with NVIDIA proprietary driver"
    echo ""
    log_info "Blacklist nouveau driver:"
    log_info "  sudo bash -c 'echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf'"
    log_info "  sudo bash -c 'echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf'"
    log_info "  sudo update-initramfs -u"
    log_info "  sudo reboot"
else
    log_success "Nouveau driver not loaded"
fi

section "Step 10: Version Compatibility Check"
echo "Checking version compatibility..."
echo ""

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')

    log_info "Driver supports up to CUDA: $cuda_version"

    if command -v nvcc &>/dev/null; then
        installed_cuda=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
        log_info "Installed CUDA toolkit: $installed_cuda"

        # Compare versions (simple comparison)
        if [ -n "$cuda_version" ] && [ -n "$installed_cuda" ]; then
            log_warning "Ensure PyTorch/TensorFlow CUDA version matches installed version"
        fi
    fi
fi

section "Analysis Summary"
echo -e "${BOLD}Diagnosis:${NC}"
echo ""

issues=0

# Check GPU presence
if ! lspci | grep -i nvidia &>/dev/null; then
    log_error "No NVIDIA GPU detected"
    issues=$((issues + 1))
else
    log_success "GPU hardware present"
fi

# Check driver
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
    log_error "NVIDIA driver not working"
    issues=$((issues + 1))
else
    log_success "NVIDIA driver working"
fi

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    log_warning "CUDA toolkit not in PATH"
    issues=$((issues + 1))
else
    log_success "CUDA toolkit accessible"
fi

# Check environment
if [ -z "${CUDA_HOME:-}" ] || [ -z "${LD_LIBRARY_PATH:-}" ]; then
    log_warning "Environment variables not properly set"
    issues=$((issues + 1))
fi

echo ""

if [ $issues -eq 0 ]; then
    log_success "No major issues detected"
    echo ""
    log_info "If still having problems, check:"
    log_info "  - Python package installation (torch with CUDA support)"
    log_info "  - Version compatibility between driver, CUDA, and framework"
else
    echo "Found $issues issue(s) that need attention."
fi

echo ""
echo -e "${BOLD}${BLUE}Next Steps:${NC}"
echo "  1. Review the analysis above"
echo "  2. Fix identified issues (driver, CUDA, environment)"
echo "  3. Run the fix script: ./fix.sh"
echo "  4. Verify GPU access after fixing"
echo ""
