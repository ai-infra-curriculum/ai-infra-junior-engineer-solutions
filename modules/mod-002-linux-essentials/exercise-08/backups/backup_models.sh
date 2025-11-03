#!/bin/bash
###############################################################################
# Model Backup Script - Automated ML Model Backups with Versioning
###############################################################################
#
# Purpose: Create timestamped backups of ML models with compression,
#          checksums, retention policies, and optional cloud sync
#
# Usage: ./backup_models.sh [OPTIONS]
#

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration (can be overridden with environment variables)
MODEL_DIR="${MODEL_DIR:-/opt/ml/models}"
BACKUP_DIR="${BACKUP_DIR:-/backup/models}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
LOG_FILE="${LOG_FILE:-/var/log/ml-backup.log}"
S3_BUCKET="${S3_BUCKET:-}"
S3_STORAGE_CLASS="${S3_STORAGE_CLASS:-STANDARD_IA}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"
VERIFY_BACKUP="${VERIFY_BACKUP:-true}"
DRY_RUN="${DRY_RUN:-false}"
QUIET="${QUIET:-false}"

# Timestamp for this backup
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="models_backup_${DATE}.tar.gz"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$QUIET" != "true" ]; then
        case "$level" in
            INFO)
                echo -e "${BLUE}[INFO]${NC} $message"
                ;;
            SUCCESS)
                echo -e "${GREEN}[SUCCESS]${NC} $message"
                ;;
            WARNING)
                echo -e "${YELLOW}[WARNING]${NC} $message"
                ;;
            ERROR)
                echo -e "${RED}[ERROR]${NC} $message"
                ;;
        esac
    fi

    # Always log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Error handler
error_exit() {
    log ERROR "$1"
    exit 1
}

# Help function
show_help() {
    cat << EOF
Model Backup Script - Automated ML Model Backups

Usage: $0 [OPTIONS]

Options:
    --model-dir PATH        Directory containing models (default: /opt/ml/models)
    --backup-dir PATH       Backup destination directory (default: /backup/models)
    --retention-days N      Keep backups for N days (default: 30)
    --log-file PATH         Log file location (default: /var/log/ml-backup.log)
    --s3-bucket NAME        S3 bucket for cloud backup (optional)
    --compression N         Compression level 1-9 (default: 6)
    --no-verify             Skip backup verification
    --dry-run               Show what would be done without doing it
    --quiet                 Suppress console output (still logs to file)
    -h, --help              Show this help message

Environment Variables:
    MODEL_DIR, BACKUP_DIR, RETENTION_DAYS, LOG_FILE, S3_BUCKET

Examples:
    # Basic backup
    $0

    # Custom directories with S3 sync
    $0 --model-dir /data/models --s3-bucket my-ml-backups

    # Dry run to preview
    $0 --dry-run

    # Aggressive compression, keep for 60 days
    $0 --compression 9 --retention-days 60

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --retention-days)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --compression)
            COMPRESSION_LEVEL="$2"
            shift 2
            ;;
        --no-verify)
            VERIFY_BACKUP="false"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --quiet)
            QUIET="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Validate dependencies
if ! command -v tar &> /dev/null; then
    error_exit "tar command not found"
fi

# Initialize
log INFO "╔═══════════════════════════════════════════════════════════╗"
log INFO "║  ML Model Backup System                                   ║"
log INFO "╚═══════════════════════════════════════════════════════════╝"
log INFO ""

if [ "$DRY_RUN" = "true" ]; then
    log WARNING "DRY RUN MODE - No changes will be made"
fi

log INFO "Configuration:"
log INFO "  Model directory: $MODEL_DIR"
log INFO "  Backup directory: $BACKUP_DIR"
log INFO "  Retention period: $RETENTION_DAYS days"
log INFO "  Compression level: $COMPRESSION_LEVEL"
log INFO "  Backup name: $BACKUP_NAME"
[ -n "$S3_BUCKET" ] && log INFO "  S3 bucket: s3://$S3_BUCKET/"
log INFO ""

# Validate source directory
if [ ! -d "$MODEL_DIR" ]; then
    error_exit "Model directory does not exist: $MODEL_DIR"
fi

# Check if directory is empty
if [ -z "$(ls -A "$MODEL_DIR")" ]; then
    error_exit "Model directory is empty: $MODEL_DIR"
fi

# Create backup directory
if [ "$DRY_RUN" != "true" ]; then
    mkdir -p "$BACKUP_DIR" || error_exit "Failed to create backup directory"
fi

# Calculate source size
log INFO "Calculating source size..."
SOURCE_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
log INFO "Source size: $SOURCE_SIZE"
log INFO ""

# Check available disk space
AVAILABLE_SPACE=$(df -BG "$BACKUP_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED_SPACE=$(du -BG "$MODEL_DIR" | tail -1 | awk '{print $1}' | sed 's/G//')

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    error_exit "Insufficient disk space. Available: ${AVAILABLE_SPACE}G, Required: ~${REQUIRED_SPACE}G"
fi

# Start backup
log INFO "Starting backup process..."

if [ "$DRY_RUN" != "true" ]; then
    # Create tar archive with compression and exclude patterns
    if tar -czf "$BACKUP_PATH" \
        --exclude='*.tmp' \
        --exclude='*.temp' \
        --exclude='*.swp' \
        --exclude='*~' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='checkpoints/temp' \
        --exclude='experiments/failed' \
        -C "$(dirname "$MODEL_DIR")" \
        "$(basename "$MODEL_DIR")" 2>&1 | tee -a "$LOG_FILE" > /dev/null; then

        BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
        log SUCCESS "Backup created successfully"
        log INFO "  Backup file: $BACKUP_NAME"
        log INFO "  Backup size: $BACKUP_SIZE"
        log INFO "  Compression ratio: $(echo "scale=2; $(stat -f%z "$BACKUP_PATH" 2>/dev/null || stat -c%s "$BACKUP_PATH") / $(du -sb "$MODEL_DIR" | cut -f1) * 100" | bc)%"
    else
        error_exit "Backup failed during tar creation"
    fi
else
    log INFO "Would create: $BACKUP_PATH"
    log INFO "Would exclude: *.tmp, *.temp, __pycache__, .git, etc."
fi

# Calculate and store checksum
if [ "$DRY_RUN" != "true" ]; then
    log INFO ""
    log INFO "Calculating checksum..."

    CHECKSUM=$(sha256sum "$BACKUP_PATH" | awk '{print $1}')
    echo "$CHECKSUM  $BACKUP_NAME" >> "${BACKUP_DIR}/checksums.txt"

    log SUCCESS "Checksum calculated and stored"
    log INFO "  SHA256: $CHECKSUM"
else
    log INFO "Would calculate SHA256 checksum"
fi

# Verify backup integrity
if [ "$VERIFY_BACKUP" = "true" ] && [ "$DRY_RUN" != "true" ]; then
    log INFO ""
    log INFO "Verifying backup integrity..."

    if tar -tzf "$BACKUP_PATH" > /dev/null 2>&1; then
        log SUCCESS "Backup integrity verified"

        # Count files in backup
        FILE_COUNT=$(tar -tzf "$BACKUP_PATH" | wc -l)
        log INFO "  Files in backup: $FILE_COUNT"
    else
        error_exit "Backup integrity check failed!"
    fi
fi

# Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    log INFO ""
    log INFO "Uploading to S3..."

    if command -v aws &> /dev/null; then
        if [ "$DRY_RUN" != "true" ]; then
            if aws s3 cp "$BACKUP_PATH" "s3://${S3_BUCKET}/models/" \
                --storage-class "$S3_STORAGE_CLASS" \
                --metadata "checksum=$CHECKSUM,source=$MODEL_DIR,date=$DATE" \
                2>&1 | tee -a "$LOG_FILE" > /dev/null; then

                log SUCCESS "Uploaded to S3: s3://${S3_BUCKET}/models/${BACKUP_NAME}"

                # Also upload checksums file
                aws s3 cp "${BACKUP_DIR}/checksums.txt" "s3://${S3_BUCKET}/models/" \
                    2>&1 | tee -a "$LOG_FILE" > /dev/null
            else
                log WARNING "S3 upload failed (continuing anyway)"
            fi
        else
            log INFO "Would upload to: s3://${S3_BUCKET}/models/${BACKUP_NAME}"
            log INFO "  Storage class: $S3_STORAGE_CLASS"
        fi
    else
        log WARNING "AWS CLI not found - skipping S3 upload"
    fi
fi

# Clean up old backups (retention policy)
log INFO ""
log INFO "Applying retention policy (keep last $RETENTION_DAYS days)..."

if [ "$DRY_RUN" != "true" ]; then
    OLD_BACKUPS=$(find "$BACKUP_DIR" -name "models_backup_*.tar.gz" -mtime +$RETENTION_DAYS)

    if [ -n "$OLD_BACKUPS" ]; then
        DELETED_COUNT=0

        while IFS= read -r old_backup; do
            if [ -f "$old_backup" ]; then
                backup_size=$(du -h "$old_backup" | cut -f1)
                log INFO "  Removing: $(basename "$old_backup") ($backup_size)"
                rm -f "$old_backup"
                DELETED_COUNT=$((DELETED_COUNT + 1))
            fi
        done <<< "$OLD_BACKUPS"

        log SUCCESS "Deleted $DELETED_COUNT old backup(s)"

        # Clean up old entries from checksums file
        if [ -f "${BACKUP_DIR}/checksums.txt" ]; then
            temp_file=$(mktemp)
            grep -v -f <(find "$BACKUP_DIR" -name "models_backup_*.tar.gz" -mtime +$RETENTION_DAYS -exec basename {} \;) \
                "${BACKUP_DIR}/checksums.txt" > "$temp_file" 2>/dev/null || true
            mv "$temp_file" "${BACKUP_DIR}/checksums.txt"
        fi
    else
        log INFO "No old backups to remove"
    fi

    # Show remaining backups
    REMAINING_COUNT=$(find "$BACKUP_DIR" -name "models_backup_*.tar.gz" | wc -l)
    log INFO "Total backups retained: $REMAINING_COUNT"
else
    OLD_COUNT=$(find "$BACKUP_DIR" -name "models_backup_*.tar.gz" -mtime +$RETENTION_DAYS 2>/dev/null | wc -l)
    log INFO "Would delete: $OLD_COUNT backup(s)"
fi

# Summary
log INFO ""
log INFO "╔═══════════════════════════════════════════════════════════╗"
log INFO "║  Backup Summary                                           ║"
log INFO "╚═══════════════════════════════════════════════════════════╝"

if [ "$DRY_RUN" != "true" ]; then
    log SUCCESS "Backup completed successfully!"
    log INFO ""
    log INFO "Backup Details:"
    log INFO "  File: $BACKUP_NAME"
    log INFO "  Location: $BACKUP_DIR"
    log INFO "  Size: $BACKUP_SIZE"
    log INFO "  Checksum: $CHECKSUM"
    [ -n "$S3_BUCKET" ] && log INFO "  S3: s3://${S3_BUCKET}/models/${BACKUP_NAME}"
    log INFO ""
    log INFO "Disk Usage:"
    df -h "$BACKUP_DIR" | tail -1
else
    log INFO "Dry run completed - no changes made"
fi

log INFO ""
log SUCCESS "All backup operations completed"
log INFO "Log file: $LOG_FILE"

exit 0
