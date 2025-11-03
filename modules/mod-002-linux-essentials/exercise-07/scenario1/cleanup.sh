#!/bin/bash
###############################################################################
# Scenario 1: Disk Space Cleanup Script
###############################################################################
#
# Usage: ./cleanup.sh [--dry-run] [--aggressive]
#

set -euo pipefail

# Configuration
CHECKPOINT_DIR="/var/ml/checkpoints"
KEEP_COUNT=5
ARCHIVE_DIR="/mnt/archive/checkpoints"
COMPRESS_DAYS=7
DRY_RUN=false
AGGRESSIVE=false

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

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --aggressive)
            AGGRESSIVE=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [OPTIONS]

Cleanup disk space for ML infrastructure.

Options:
  --dry-run       Show what would be deleted without deleting
  --aggressive    More aggressive cleanup (clear caches, temp files)
  -h, --help      Show this help message

Examples:
  $0 --dry-run           # Preview cleanup
  $0                     # Perform cleanup
  $0 --aggressive        # Aggressive cleanup

EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No changes will be made"
    echo ""
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Disk Space Cleanup Utility                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Show before state
log_info "Disk space before cleanup:"
df -h | grep -E "Filesystem|/$" | head -2
echo ""

freed_space=0

# 1. Clean up old checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    log_info "1. Cleaning up old checkpoints..."

    checkpoint_count=$(ls -1 "$CHECKPOINT_DIR"/*.h5 2>/dev/null | wc -l || echo 0)

    if [ "$checkpoint_count" -gt "$KEEP_COUNT" ]; then
        files_to_remove=$((checkpoint_count - KEEP_COUNT))
        log_info "  Found $checkpoint_count checkpoints, keeping $KEEP_COUNT newest"
        log_info "  Will remove $files_to_remove old checkpoints"

        if [ "$DRY_RUN" = false ]; then
            # Keep only last N checkpoints
            ls -t "$CHECKPOINT_DIR"/*.h5 2>/dev/null | tail -n +$((KEEP_COUNT + 1)) | \
                while read file; do
                    size=$(du -sh "$file" | awk '{print $1}')
                    log_info "  Removing: $(basename "$file") ($size)"
                    rm -f "$file"
                done
            log_success "  Old checkpoints removed"
        else
            ls -t "$CHECKPOINT_DIR"/*.h5 2>/dev/null | tail -n +$((KEEP_COUNT + 1)) | \
                while read file; do
                    size=$(du -sh "$file" | awk '{print $1}')
                    log_info "  Would remove: $(basename "$file") ($size)"
                done
        fi
    else
        log_success "  Checkpoint count ($checkpoint_count) within limit ($KEEP_COUNT)"
    fi
else
    log_warning "  Checkpoint directory not found: $CHECKPOINT_DIR"
fi
echo ""

# 2. Compress old checkpoints
log_info "2. Compressing old checkpoints (>$COMPRESS_DAYS days)..."
if [ -d "$CHECKPOINT_DIR" ]; then
    old_uncompressed=$(find "$CHECKPOINT_DIR" -name "*.h5" -mtime +$COMPRESS_DAYS 2>/dev/null | wc -l)

    if [ "$old_uncompressed" -gt 0 ]; then
        log_info "  Found $old_uncompressed uncompressed old checkpoints"

        if [ "$DRY_RUN" = false ]; then
            find "$CHECKPOINT_DIR" -name "*.h5" -mtime +$COMPRESS_DAYS 2>/dev/null | \
                while read file; do
                    log_info "  Compressing: $(basename "$file")"
                    gzip "$file"
                done
            log_success "  Old checkpoints compressed"
        else
            find "$CHECKPOINT_DIR" -name "*.h5" -mtime +$COMPRESS_DAYS 2>/dev/null | \
                while read file; do
                    log_info "  Would compress: $(basename "$file")"
                done
        fi
    else
        log_success "  No old uncompressed checkpoints found"
    fi
else
    log_warning "  Checkpoint directory not found"
fi
echo ""

# 3. Clean package caches
log_info "3. Cleaning package caches..."
if command -v apt &>/dev/null; then
    cache_size=$(du -sh /var/cache/apt/archives 2>/dev/null | awk '{print $1}' || echo "0")
    log_info "  APT cache size: $cache_size"

    if [ "$DRY_RUN" = false ]; then
        sudo apt clean
        log_success "  APT cache cleaned"
    else
        log_info "  Would run: sudo apt clean"
    fi
fi

if command -v yum &>/dev/null; then
    if [ "$DRY_RUN" = false ]; then
        sudo yum clean all
        log_success "  YUM cache cleaned"
    else
        log_info "  Would run: sudo yum clean all"
    fi
fi
echo ""

# 4. Clean temporary files (aggressive mode)
if [ "$AGGRESSIVE" = true ]; then
    log_warning "4. Cleaning temporary files (aggressive mode)..."

    temp_size=$(du -sh /tmp 2>/dev/null | awk '{print $1}' || echo "0")
    log_info "  /tmp size: $temp_size"

    if [ "$DRY_RUN" = false ]; then
        sudo find /tmp -type f -atime +7 -delete 2>/dev/null || true
        log_success "  Old temp files removed"
    else
        count=$(find /tmp -type f -atime +7 2>/dev/null | wc -l)
        log_info "  Would remove $count old temp files"
    fi

    cache_size=$(du -sh ~/.cache 2>/dev/null | awk '{print $1}' || echo "0")
    log_info "  User cache size: $cache_size"

    if [ "$DRY_RUN" = false ]; then
        rm -rf ~/.cache/* 2>/dev/null || true
        log_success "  User cache cleared"
    else
        log_info "  Would clear user cache"
    fi
    echo ""
fi

# 5. Clean Docker (if installed)
if command -v docker &>/dev/null && docker ps &>/dev/null 2>&1; then
    log_info "5. Cleaning Docker resources..."

    if [ "$DRY_RUN" = false ]; then
        docker system prune -f
        log_success "  Docker resources cleaned"
    else
        log_info "  Would run: docker system prune -f"
    fi
    echo ""
fi

# 6. Rotate logs
log_info "6. Rotating logs..."
if [ "$DRY_RUN" = false ]; then
    sudo logrotate -f /etc/logrotate.conf 2>/dev/null || log_warning "  logrotate failed"
    log_success "  Logs rotated"
else
    log_info "  Would run: sudo logrotate -f /etc/logrotate.conf"
fi
echo ""

# Show after state
log_info "Disk space after cleanup:"
df -h | grep -E "Filesystem|/$" | head -2
echo ""

# Calculate freed space (approximate)
log_success "Cleanup complete!"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}This was a dry run. Run without --dry-run to perform cleanup.${NC}"
    echo ""
fi

# Recommendations
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Prevention Recommendations                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Set up automatic checkpoint cleanup:"
echo "   Add to crontab: 0 2 * * * /path/to/cleanup.sh"
echo ""
echo "2. Implement checkpoint retention policy in training code:"
echo "   - Keep only last N checkpoints"
echo "   - Compress checkpoints older than X days"
echo ""
echo "3. Archive old checkpoints to cheaper storage:"
echo "   - AWS S3 Glacier"
echo "   - Google Cloud Storage Archive"
echo ""
echo "4. Monitor disk usage with alerts:"
echo "   - Set alert at 80% usage"
echo "   - Set critical alert at 90%"
echo ""
echo "5. Use separate partition for ML data"
echo ""
