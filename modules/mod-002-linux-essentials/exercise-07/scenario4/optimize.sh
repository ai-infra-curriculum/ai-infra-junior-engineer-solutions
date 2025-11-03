#!/bin/bash
###############################################################################
# Scenario 4: Out of Memory - Optimization Script
###############################################################################
#
# Usage: ./optimize.sh [--add-swap SIZE] [--tune-vm] [--show-recommendations]
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ${NC} $*"; }
log_success() { echo -e "${GREEN}✓${NC} $*"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $*"; }
log_error() { echo -e "${RED}✗${NC} $*"; }

# Default values
ADD_SWAP=false
SWAP_SIZE="4G"
TUNE_VM=false
SHOW_RECOMMENDATIONS=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --add-swap)
            ADD_SWAP=true
            if [[ $# -gt 1 ]] && [[ ! $2 =~ ^-- ]]; then
                SWAP_SIZE="$2"
                shift
            fi
            shift
            ;;
        --tune-vm)
            TUNE_VM=true
            shift
            ;;
        --show-recommendations)
            SHOW_RECOMMENDATIONS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [OPTIONS]

Optimize system memory configuration and provide recommendations.

Options:
  --add-swap [SIZE]         Add swap file (default: 4G)
                            Examples: 2G, 4G, 8G
  --tune-vm                 Tune VM parameters for ML workloads
  --show-recommendations    Show application-level recommendations only
  --dry-run                 Show what would be done without doing it
  -h, --help                Show this help message

Examples:
  $0 --show-recommendations       # Show recommendations only
  $0 --add-swap 8G               # Add 8GB swap
  $0 --tune-vm                    # Optimize VM parameters
  $0 --add-swap 4G --tune-vm     # Full optimization

EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Memory Optimization Utility                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Show current state
log_info "Current memory status:"
free -h | grep -E "Mem:|Swap:"
echo ""

# Add swap space
if [ "$ADD_SWAP" = true ]; then
    log_info "Adding swap space..."
    echo ""

    # Check if swap already exists
    if swapon --show | grep -q "/swapfile"; then
        log_warning "Swap file already exists at /swapfile"
        echo ""
        log_info "Current swap:"
        swapon --show
        echo ""
        log_info "To add more swap, manually create a different swap file"
        echo ""
    else
        # Convert size to bytes for validation
        size_num=$(echo "$SWAP_SIZE" | sed 's/[^0-9]*//g')
        size_unit=$(echo "$SWAP_SIZE" | sed 's/[0-9]*//g')

        log_info "Creating ${SWAP_SIZE} swap file..."

        if [ "$DRY_RUN" = false ]; then
            # Check available disk space
            available_space=$(df -BG /  | awk 'NR==2 {print $4}' | sed 's/G//')

            if [ "$size_num" -gt "$available_space" ]; then
                log_error "Insufficient disk space for swap file"
                log_info "Available: ${available_space}G, Requested: ${SWAP_SIZE}"
                exit 1
            fi

            # Create swap file
            log_info "  Creating /swapfile..."
            sudo fallocate -l "$SWAP_SIZE" /swapfile || {
                log_warning "  fallocate failed, trying dd..."
                sudo dd if=/dev/zero of=/swapfile bs=1M count=$((size_num * 1024)) status=progress
            }

            log_info "  Setting permissions..."
            sudo chmod 600 /swapfile

            log_info "  Setting up swap..."
            sudo mkswap /swapfile

            log_info "  Enabling swap..."
            sudo swapon /swapfile

            log_success "Swap file created and enabled"

            # Make it permanent
            if ! grep -q "/swapfile" /etc/fstab; then
                log_info "  Adding to /etc/fstab for persistence..."
                echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
                log_success "Swap will persist after reboot"
            fi

            echo ""
            log_info "New swap status:"
            swapon --show
        else
            log_info "Would run:"
            echo "    sudo fallocate -l $SWAP_SIZE /swapfile"
            echo "    sudo chmod 600 /swapfile"
            echo "    sudo mkswap /swapfile"
            echo "    sudo swapon /swapfile"
            echo "    echo '/swapfile none swap sw 0 0' >> /etc/fstab"
        fi
    fi
    echo ""
fi

# Tune VM parameters
if [ "$TUNE_VM" = true ]; then
    log_info "Tuning VM parameters for ML workloads..."
    echo ""

    # Swappiness
    current_swappiness=$(cat /proc/sys/vm/swappiness)
    recommended_swappiness=10

    log_info "Swappiness (how aggressively to use swap):"
    log_info "  Current: $current_swappiness"
    log_info "  Recommended for ML: $recommended_swappiness"

    if [ "$current_swappiness" -ne "$recommended_swappiness" ]; then
        if [ "$DRY_RUN" = false ]; then
            sudo sysctl vm.swappiness=$recommended_swappiness
            log_success "  Set swappiness to $recommended_swappiness"

            # Make permanent
            if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
                echo "vm.swappiness=$recommended_swappiness" | sudo tee -a /etc/sysctl.conf
            fi
        else
            log_info "Would run: sudo sysctl vm.swappiness=$recommended_swappiness"
        fi
    else
        log_success "  Already optimal"
    fi

    echo ""

    # VFS cache pressure
    current_vfs=$(cat /proc/sys/vm/vfs_cache_pressure)
    recommended_vfs=50

    log_info "VFS cache pressure (tendency to reclaim inode/dentry caches):"
    log_info "  Current: $current_vfs"
    log_info "  Recommended: $recommended_vfs"

    if [ "$current_vfs" -ne "$recommended_vfs" ]; then
        if [ "$DRY_RUN" = false ]; then
            sudo sysctl vm.vfs_cache_pressure=$recommended_vfs
            log_success "  Set VFS cache pressure to $recommended_vfs"

            if ! grep -q "vm.vfs_cache_pressure" /etc/sysctl.conf; then
                echo "vm.vfs_cache_pressure=$recommended_vfs" | sudo tee -a /etc/sysctl.conf
            fi
        else
            log_info "Would run: sudo sysctl vm.vfs_cache_pressure=$recommended_vfs"
        fi
    else
        log_success "  Already optimal"
    fi

    echo ""

    # Overcommit handling
    current_overcommit=$(cat /proc/sys/vm/overcommit_memory)
    log_info "Memory overcommit mode:"
    log_info "  Current: $current_overcommit"

    case $current_overcommit in
        0) log_info "  (Heuristic - reasonable for most workloads)" ;;
        1) log_warning "  (Always overcommit - risky for ML!)" ;;
        2) log_info "  (Never overcommit - conservative)" ;;
    esac

    if [ "$current_overcommit" -eq 1 ]; then
        log_warning "  Consider changing to mode 0 (heuristic)"
        if [ "$DRY_RUN" = false ]; then
            sudo sysctl vm.overcommit_memory=0
            log_success "  Set overcommit to heuristic mode"
        fi
    fi

    echo ""
    log_success "VM parameters tuned"
fi

# Show recommendations
if [ "$SHOW_RECOMMENDATIONS" = true ] || [ "$ADD_SWAP" = false ] && [ "$TUNE_VM" = false ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Application-Level Recommendations                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    echo "1. REDUCE BATCH SIZE"
    echo "   PyTorch:"
    echo "     train_loader = DataLoader(dataset, batch_size=32)  # Reduce from 64"
    echo ""
    echo "   TensorFlow:"
    echo "     model.fit(x_train, y_train, batch_size=32)"
    echo ""

    echo "2. USE DATA GENERATORS (Don't load all data into memory)"
    echo "   PyTorch:"
    echo "     class CustomDataset(Dataset):"
    echo "       def __getitem__(self, idx):"
    echo "         return load_sample(idx)  # Load on-demand"
    echo ""
    echo "   TensorFlow:"
    echo "     dataset = tf.data.Dataset.from_generator(...)"
    echo ""

    echo "3. ENABLE GRADIENT CHECKPOINTING"
    echo "   PyTorch:"
    echo "     from torch.utils.checkpoint import checkpoint"
    echo "     output = checkpoint(model.layer, input)"
    echo ""
    echo "   TensorFlow:"
    echo "     from tensorflow.python.ops import gradient_checkpoint as gc"
    echo ""

    echo "4. MIXED PRECISION TRAINING (Reduces memory)"
    echo "   PyTorch:"
    echo "     from torch.cuda.amp import autocast, GradScaler"
    echo "     scaler = GradScaler()"
    echo "     with autocast():"
    echo "       output = model(input)"
    echo ""
    echo "   TensorFlow:"
    echo "     policy = tf.keras.mixed_precision.Policy('mixed_float16')"
    echo "     tf.keras.mixed_precision.set_global_policy(policy)"
    echo ""

    echo "5. CLEAR GPU CACHE BETWEEN RUNS"
    echo "   PyTorch:"
    echo "     torch.cuda.empty_cache()"
    echo ""
    echo "   TensorFlow:"
    echo "     tf.keras.backend.clear_session()"
    echo ""

    echo "6. LIMIT GPU MEMORY GROWTH"
    echo "   PyTorch:"
    echo "     torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80%"
    echo ""
    echo "   TensorFlow:"
    echo "     gpus = tf.config.list_physical_devices('GPU')"
    echo "     tf.config.experimental.set_memory_growth(gpus[0], True)"
    echo ""

    echo "7. MONITOR MEMORY USAGE IN CODE"
    echo "   import psutil"
    echo "   import os"
    echo ""
    echo "   process = psutil.Process(os.getpid())"
    echo "   print(f'Memory: {process.memory_info().rss / 1024**2:.2f} MB')"
    echo ""

    echo "8. USE MODEL PARALLELISM FOR LARGE MODELS"
    echo "   Split model across multiple GPUs:"
    echo "     model.layer1.to('cuda:0')"
    echo "     model.layer2.to('cuda:1')"
    echo ""

    echo "9. REDUCE MODEL SIZE"
    echo "   - Use smaller architectures (e.g., MobileNet vs ResNet)"
    echo "   - Reduce number of layers/filters"
    echo "   - Apply pruning or quantization"
    echo ""

    echo "10. OPTIMIZE DATA LOADING"
    echo "   - Use num_workers in DataLoader"
    echo "   - Pin memory for faster GPU transfer"
    echo "   - Prefetch data"
    echo ""
    echo "   train_loader = DataLoader("
    echo "     dataset,"
    echo "     batch_size=32,"
    echo "     num_workers=4,"
    echo "     pin_memory=True,"
    echo "     prefetch_factor=2"
    echo "   )"
    echo ""
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  System-Level Best Practices                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "1. Monitor memory usage during training:"
echo "   watch -n 1 free -h"
echo ""

echo "2. Set memory limits for processes:"
echo "   ulimit -v 8000000  # Limit to ~8GB"
echo ""

echo "3. Use cgroups to limit memory for specific processes:"
echo "   systemd-run --scope -p MemoryLimit=8G python train.py"
echo ""

echo "4. Enable OOM score adjustment for critical processes:"
echo "   echo -1000 > /proc/<PID>/oom_score_adj  # Protected from OOM"
echo "   echo 1000 > /proc/<PID>/oom_score_adj   # First to be killed"
echo ""

echo "5. Monitor with tools:"
echo "   - htop: Interactive process viewer"
echo "   - vmstat: Virtual memory statistics"
echo "   - sar: System activity reports"
echo ""

echo "6. Check application memory leaks:"
echo "   valgrind --leak-check=full python train.py"
echo "   memory_profiler (@profile decorator)"
echo ""

log_success "Optimization complete!"
echo ""
