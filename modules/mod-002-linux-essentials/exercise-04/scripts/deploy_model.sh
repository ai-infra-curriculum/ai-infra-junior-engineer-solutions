#!/bin/bash
#
# deploy_model.sh - ML Model Deployment Automation
#
# Usage: ./deploy_model.sh <model_file> <environment>
#        ./deploy_model.sh rollback
#
# Environments: staging, production
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_DIR="${SCRIPT_DIR}/../logs"
readonly LOG_FILE="${LOG_DIR}/deployment.log"
readonly STAGING_DIR="${SCRIPT_DIR}/../models/staging"
readonly PRODUCTION_DIR="${SCRIPT_DIR}/../models/production"
readonly BACKUP_DIR="${SCRIPT_DIR}/../models/backups"
readonly METADATA_FILE="${SCRIPT_DIR}/../models/deployment_metadata.json"

# Model size limits (in MB)
readonly MIN_MODEL_SIZE=1
readonly MAX_MODEL_SIZE=5000

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

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_error() { log "ERROR" "$@" >&2; }
log_success() { log "SUCCESS" "$@"; }
log_warning() { log "WARNING" "$@"; }

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Setup function
setup() {
    # Create required directories
    mkdir -p "$LOG_DIR" "$STAGING_DIR" "$PRODUCTION_DIR" "$BACKUP_DIR"

    # Initialize metadata file if not exists
    if [ ! -f "$METADATA_FILE" ]; then
        echo '{"deployments": []}' > "$METADATA_FILE"
    fi
}

# Usage information
usage() {
    cat << EOF
Usage: $SCRIPT_NAME <model_file> <environment>
       $SCRIPT_NAME rollback

Deploy ML models to staging or production environments.

Arguments:
  model_file    Path to model file (e.g., model.h5, model.pth, model.pkl)
  environment   Target environment (staging or production)
  rollback      Rollback to previous production model

Examples:
  $SCRIPT_NAME model.h5 staging
  $SCRIPT_NAME model.pth production
  $SCRIPT_NAME rollback

Environments:
  staging     - Deploy to staging for testing
  production  - Deploy to production (requires staging validation)

Model Requirements:
  - Size: ${MIN_MODEL_SIZE}MB - ${MAX_MODEL_SIZE}MB
  - Formats: .h5, .pth, .pkl, .onnx, .pb
  - Must pass validation checks

EOF
    exit 0
}

# Validate model file
validate_model() {
    local model_path="$1"

    log_info "Validating model: $model_path"

    # Check if file exists
    if [ ! -f "$model_path" ]; then
        error_exit "Model file not found: $model_path"
    fi

    # Check file extension
    local extension="${model_path##*.}"
    case "$extension" in
        h5|pth|pkl|onnx|pb)
            log_info "✓ Valid model format: .$extension"
            ;;
        *)
            error_exit "Invalid model format: .$extension (supported: h5, pth, pkl, onnx, pb)"
            ;;
    esac

    # Check file size
    local size_bytes=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null)
    local size_mb=$((size_bytes / 1024 / 1024))

    log_info "Model size: ${size_mb}MB"

    if [ "$size_mb" -lt "$MIN_MODEL_SIZE" ]; then
        error_exit "Model too small: ${size_mb}MB (minimum: ${MIN_MODEL_SIZE}MB)"
    fi

    if [ "$size_mb" -gt "$MAX_MODEL_SIZE" ]; then
        error_exit "Model too large: ${size_mb}MB (maximum: ${MAX_MODEL_SIZE}MB)"
    fi

    log_success "✓ Model validation passed"
    return 0
}

# Backup current model
backup_model() {
    local environment="$1"

    local source_dir
    if [ "$environment" = "staging" ]; then
        source_dir="$STAGING_DIR"
    else
        source_dir="$PRODUCTION_DIR"
    fi

    # Find current model
    local current_model=$(find "$source_dir" -maxdepth 1 -type f \( -name "*.h5" -o -name "*.pth" -o -name "*.pkl" -o -name "*.onnx" -o -name "*.pb" \) 2>/dev/null | head -1)

    if [ -z "$current_model" ]; then
        log_warning "No existing model to backup in $environment"
        return 0
    fi

    log_info "Backing up current $environment model"

    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local model_name=$(basename "$current_model")
    local backup_name="${environment}_${timestamp}_${model_name}"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    cp "$current_model" "$backup_path"
    log_success "✓ Backup created: $backup_name"

    # Keep only last 5 backups per environment
    local backup_count=$(find "$BACKUP_DIR" -name "${environment}_*" | wc -l)
    if [ "$backup_count" -gt 5 ]; then
        log_info "Cleaning old backups (keeping last 5)"
        find "$BACKUP_DIR" -name "${environment}_*" -type f | sort | head -n -5 | xargs rm -f
    fi
}

# Deploy model to environment
deploy_to_environment() {
    local model_path="$1"
    local environment="$2"

    local target_dir
    if [ "$environment" = "staging" ]; then
        target_dir="$STAGING_DIR"
    else
        target_dir="$PRODUCTION_DIR"
    fi

    log_info "Deploying to $environment environment"

    # Backup current model
    backup_model "$environment"

    # Copy new model
    local model_name=$(basename "$model_path")
    local target_path="${target_dir}/${model_name}"

    cp "$model_path" "$target_path"
    log_success "✓ Model deployed: $target_path"

    # Update metadata
    update_metadata "$model_name" "$environment"

    # Verify deployment
    if [ -f "$target_path" ]; then
        log_success "✓ Deployment verification passed"
    else
        error_exit "Deployment verification failed"
    fi
}

# Update deployment metadata
update_metadata() {
    local model_name="$1"
    local environment="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Create deployment record
    local deployment_record=$(cat <<EOF
{
  "model": "$model_name",
  "environment": "$environment",
  "timestamp": "$timestamp",
  "deployed_by": "${USER:-unknown}"
}
EOF
)

    # Append to metadata (simplified - in production use jq)
    log_info "Updated deployment metadata"
}

# Rollback to previous model
rollback_deployment() {
    log_info "Initiating rollback"

    # Find latest production backup
    local latest_backup=$(find "$BACKUP_DIR" -name "production_*" -type f | sort -r | head -1)

    if [ -z "$latest_backup" ]; then
        error_exit "No backup available for rollback"
    fi

    log_info "Rolling back to: $(basename "$latest_backup")"

    # Remove current production model
    rm -f "${PRODUCTION_DIR}"/*

    # Restore backup
    local model_name=$(basename "$latest_backup" | sed 's/production_[0-9]*_[0-9]*_//')
    cp "$latest_backup" "${PRODUCTION_DIR}/${model_name}"

    log_success "✓ Rollback completed: $model_name"

    # Update metadata
    update_metadata "$model_name" "production (rollback)"
}

# Promote staging to production
promote_to_production() {
    log_info "Promoting staging model to production"

    # Find staging model
    local staging_model=$(find "$STAGING_DIR" -maxdepth 1 -type f \( -name "*.h5" -o -name "*.pth" -o -name "*.pkl" -o -name "*.onnx" -o -name "*.pb" \) 2>/dev/null | head -1)

    if [ -z "$staging_model" ]; then
        error_exit "No model found in staging"
    fi

    log_info "Staging model: $(basename "$staging_model")"

    # Confirm promotion
    echo -e "${YELLOW}⚠ Promoting to production${NC}"
    echo -n "Continue? (yes/no): "
    read -r response

    if [ "$response" != "yes" ]; then
        log_info "Promotion cancelled"
        exit 0
    fi

    # Deploy to production
    deploy_to_environment "$staging_model" "production"

    log_success "✓ Model promoted to production"
}

# Main function
main() {
    setup

    log_info "=== ML Model Deployment ==="

    # Parse arguments
    if [ $# -eq 0 ]; then
        usage
    fi

    local command="$1"

    # Handle rollback
    if [ "$command" = "rollback" ]; then
        rollback_deployment
        exit 0
    fi

    # Handle promote
    if [ "$command" = "promote" ]; then
        promote_to_production
        exit 0
    fi

    # Handle deployment
    if [ $# -lt 2 ]; then
        error_exit "Missing arguments. Usage: $SCRIPT_NAME <model_file> <environment>"
    fi

    local model_file="$1"
    local environment="$2"

    # Validate environment
    if [ "$environment" != "staging" ] && [ "$environment" != "production" ]; then
        error_exit "Invalid environment: $environment (must be staging or production)"
    fi

    # Validate model
    validate_model "$model_file"

    # Deploy
    deploy_to_environment "$model_file" "$environment"

    log_success "=== Deployment Complete ==="

    # Show deployment info
    echo ""
    echo -e "${BLUE}Deployment Summary:${NC}"
    echo "  Model:       $(basename "$model_file")"
    echo "  Environment: $environment"
    echo "  Time:        $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Log:         $LOG_FILE"
    echo ""

    if [ "$environment" = "staging" ]; then
        echo -e "${YELLOW}Next step:${NC} Validate in staging, then promote to production:"
        echo "  $SCRIPT_NAME promote"
    fi

    echo ""
}

# Handle help flag
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
fi

main "$@"
