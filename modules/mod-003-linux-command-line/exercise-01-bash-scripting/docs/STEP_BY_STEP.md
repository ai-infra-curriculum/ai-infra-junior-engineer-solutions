# Step-by-Step Implementation Guide: Bash Scripting for Infrastructure Automation

## Overview

Master Bash scripting by building a comprehensive infrastructure automation toolkit for ML/AI operations. You'll create four production-ready scripts that demonstrate professional Bash practices, error handling, and real-world automation scenarios.

**Time**: 4-5 hours | **Difficulty**: Beginner to Intermediate

---

## Prerequisites

```bash
# Verify Bash version (4.0+ recommended)
bash --version

# Install shellcheck for linting
sudo apt install shellcheck -y  # Ubuntu/Debian
# or
brew install shellcheck  # macOS

# Install jq for JSON processing
sudo apt install jq -y  # Ubuntu/Debian
# or
brew install jq  # macOS

# Optional: Install GPG for encryption features
sudo apt install gnupg -y
```

---

## Learning Objectives

By completing this exercise, you will be able to:

✅ Write robust Bash scripts with proper error handling
✅ Parse command-line arguments (both short and long options)
✅ Implement logging and verbosity controls
✅ Use functions for code organization and reusability
✅ Handle file operations safely
✅ Implement dry-run modes for testing
✅ Create production-ready automation scripts for ML infrastructure

---

## Phase 1: Backup Data Script (90 minutes)

### Step 1: Understanding the Requirements

The `backup_data.sh` script automates backing up ML training data with features like:
- Compression to save storage space
- Optional encryption for sensitive data
- Retention policies to manage old backups
- Checksum verification for data integrity
- Metadata tracking for backup history

**Real-world use case**: Daily backups of training datasets, model checkpoints, and experiment results.

### Step 2: Create the Script Structure

Create the file `solutions/backup_data.sh`:

```bash
cd ~/ai-infra-junior-engineer-solutions/modules/mod-003-linux-command-line/exercise-01-bash-scripting
touch solutions/backup_data.sh
chmod +x solutions/backup_data.sh
```

### Step 3: Add the Header and Configuration

Open `backup_data.sh` and start with the script header:

```bash
#!/bin/bash
#
# backup_data.sh - Backup and restore ML training data
#
# Description:
#   Automates backup of ML datasets with compression, encryption,
#   and retention policies. Supports local and remote backups.
#
# Usage:
#   ./backup_data.sh [OPTIONS] ACTION [PATH]
#
# Actions:
#   backup      Create a new backup
#   restore     Restore from backup
#   list        List available backups
#   cleanup     Remove old backups based on retention policy
#
# Options:
#   -d, --destination DIR   Backup destination directory
#   -r, --retention DAYS    Retention period in days (default: 30)
#   -c, --compress          Enable compression (default: true)
#   -e, --encrypt           Enable encryption
#   -n, --dry-run          Perform dry-run
#   -v, --verbose          Enable verbose output
#   -h, --help             Display this help message
#

set -euo pipefail
```

**Key concept**: `set -euo pipefail` is crucial for production scripts:
- `-e`: Exit on any error
- `-u`: Exit on undefined variables
- `-o pipefail`: Catch errors in pipes

### Step 4: Define Configuration Variables

Add global configuration after the header:

```bash
# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/ml-backup.log"

# Default configuration
BACKUP_DESTINATION="${BACKUP_DESTINATION:-/backup/ml-data}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESS=true
ENCRYPT=false
DRY_RUN=false
VERBOSE=false

# Backup metadata
readonly BACKUP_PREFIX="ml-data"
readonly METADATA_FILE=".backup-metadata.json"
```

**Why this matters**:
- `readonly` prevents accidental modification
- `${VAR:-default}` provides environment variable overrides
- Centralized configuration makes maintenance easier

### Step 5: Implement Logging Functions

Add logging capabilities:

```bash
# ===========================
# Logging Functions
# ===========================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG" "$@"
    fi
}

error_exit() {
    log "ERROR" "$1"
    exit "${2:-1}"
}
```

**Pattern explanation**:
- `local` keeps variables function-scoped
- `shift` removes the first argument, leaving the rest
- `$*` captures all remaining arguments
- `tee -a` writes to both stdout and log file

### Step 6: Add Utility Functions

Create helper functions for common operations:

```bash
# ===========================
# Utility Functions
# ===========================

human_readable_size() {
    local size=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit_index=0

    while [[ $size -gt 1024 ]] && [[ $unit_index -lt 4 ]]; do
        size=$((size / 1024))
        ((unit_index++))
    done

    echo "${size}${units[$unit_index]}"
}

calculate_checksum() {
    local file="$1"
    sha256sum "$file" | awk '{print $1}'
}

check_disk_space() {
    local required_space=$1  # in MB
    local destination=$2

    local available_space=$(df -m "$destination" | awk 'NR==2 {print $4}')

    if [[ $available_space -lt $required_space ]]; then
        error_exit "Insufficient disk space. Required: ${required_space}MB, Available: ${available_space}MB" 1
    fi

    log_verbose "Disk space check passed. Available: ${available_space}MB"
}
```

**Best practices shown**:
- Functions do one thing well
- Input validation prevents errors
- Descriptive function names improve readability

### Step 7: Implement Backup Creation

Now for the core functionality - creating backups:

```bash
# ===========================
# Backup Functions
# ===========================

create_backup_name() {
    local source_path="$1"
    local source_name=$(basename "$source_path")
    local timestamp=$(date +%Y%m%d_%H%M%S)

    echo "${BACKUP_PREFIX}_${source_name}_${timestamp}"
}

create_backup() {
    local source_path="$1"

    if [[ ! -e "$source_path" ]]; then
        error_exit "Source path does not exist: $source_path" 1
    fi

    log "INFO" "Starting backup of: $source_path"

    local backup_name=$(create_backup_name "$source_path")
    local backup_dir="$BACKUP_DESTINATION/$backup_name"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would create backup: $backup_dir"
        return 0
    fi

    # Create backup directory
    mkdir -p "$backup_dir"

    # Calculate source size
    local source_size=$(du -sm "$source_path" | cut -f1)
    log "INFO" "Source size: ${source_size}MB"

    # Check disk space (require 2x source size for safety)
    check_disk_space $((source_size * 2)) "$BACKUP_DESTINATION"

    # Start backup
    local backup_file="${backup_dir}/data.tar"
    log "INFO" "Creating backup archive..."

    tar -cf "$backup_file" -C "$(dirname "$source_path")" "$(basename "$source_path")"

    # Compress if enabled
    if [[ "$COMPRESS" == true ]]; then
        log "INFO" "Compressing backup..."
        gzip "$backup_file"
        backup_file="${backup_file}.gz"
    fi

    # Encrypt if enabled
    if [[ "$ENCRYPT" == true ]]; then
        log "INFO" "Encrypting backup..."
        if ! command -v gpg &> /dev/null; then
            log "WARNING" "GPG not found. Skipping encryption."
        else
            gpg --symmetric --cipher-algo AES256 "$backup_file"
            rm "$backup_file"
            backup_file="${backup_file}.gpg"
        fi
    fi

    # Calculate checksum
    log "INFO" "Calculating checksum..."
    local checksum=$(calculate_checksum "$backup_file")
    echo "$checksum" > "${backup_file}.sha256"

    # Create metadata
    create_backup_metadata "$backup_dir" "$source_path" "$backup_file" "$checksum"

    # Get final backup size
    local backup_size=$(du -sh "$backup_dir" | cut -f1)

    log "SUCCESS" "Backup created successfully!"
    log "INFO" "  Location: $backup_dir"
    log "INFO" "  Size: $backup_size"
    log "INFO" "  Checksum: $checksum"

    return 0
}
```

**Key techniques**:
- `command -v` checks if a program exists
- `&> /dev/null` suppresses all output
- Error checking before proceeding with operations
- Informative logging at each step

### Step 8: Create Metadata Function

Add metadata creation for tracking backup details:

```bash
create_backup_metadata() {
    local backup_dir="$1"
    local source_path="$2"
    local backup_file="$3"
    local checksum="$4"

    local metadata_path="$backup_dir/$METADATA_FILE"

    cat > "$metadata_path" <<EOF
{
  "backup_name": "$(basename "$backup_dir")",
  "source_path": "$source_path",
  "backup_file": "$(basename "$backup_file")",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "checksum": "$checksum",
  "compressed": $COMPRESS,
  "encrypted": $ENCRYPT,
  "hostname": "$(hostname)",
  "user": "$USER"
}
EOF

    log_verbose "Metadata created: $metadata_path"
}
```

**Heredoc pattern**: The `<<EOF ... EOF` syntax creates multi-line content cleanly.

### Step 9: Implement Restore and List Functions

Add functions to list and restore backups:

```bash
# ===========================
# Restore Functions
# ===========================

list_backups() {
    log "INFO" "Available backups in: $BACKUP_DESTINATION"
    log "INFO" "=========================================="

    if [[ ! -d "$BACKUP_DESTINATION" ]]; then
        log "WARNING" "Backup destination does not exist"
        return 1
    fi

    local backup_count=0

    for backup_dir in "$BACKUP_DESTINATION"/${BACKUP_PREFIX}_*; do
        if [[ ! -d "$backup_dir" ]]; then
            continue
        fi

        ((backup_count++))

        local metadata_file="$backup_dir/$METADATA_FILE"
        if [[ -f "$metadata_file" ]]; then
            local backup_name=$(basename "$backup_dir")
            local timestamp=$(jq -r '.timestamp' "$metadata_file" 2>/dev/null || echo "unknown")
            local source=$(jq -r '.source_path' "$metadata_file" 2>/dev/null || echo "unknown")
            local size=$(du -sh "$backup_dir" | cut -f1)

            echo "$backup_count. $backup_name"
            echo "   Timestamp: $timestamp"
            echo "   Source: $source"
            echo "   Size: $size"
            echo ""
        fi
    done

    if [[ $backup_count -eq 0 ]]; then
        log "INFO" "No backups found"
    else
        log "INFO" "Total backups: $backup_count"
    fi

    return 0
}

restore_backup() {
    local backup_name="$1"
    local restore_path="${2:-.}"

    local backup_dir="$BACKUP_DESTINATION/$backup_name"

    if [[ ! -d "$backup_dir" ]]; then
        error_exit "Backup not found: $backup_name" 1
    fi

    log "INFO" "Restoring backup: $backup_name"
    log "INFO" "Destination: $restore_path"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would restore backup to: $restore_path"
        return 0
    fi

    # Find backup file
    local backup_file=""
    for file in "$backup_dir"/data.tar*; do
        if [[ -f "$file" ]]; then
            backup_file="$file"
            break
        fi
    done

    if [[ -z "$backup_file" ]]; then
        error_exit "Backup data file not found in: $backup_dir" 1
    fi

    # Verify checksum
    local checksum_file="${backup_file}.sha256"
    if [[ -f "$checksum_file" ]]; then
        log "INFO" "Verifying checksum..."
        local expected_checksum=$(cat "$checksum_file")
        local actual_checksum=$(calculate_checksum "$backup_file")

        if [[ "$expected_checksum" != "$actual_checksum" ]]; then
            error_exit "Checksum verification failed!" 1
        fi
        log "SUCCESS" "Checksum verified"
    fi

    # Decrypt if needed
    if [[ "$backup_file" == *.gpg ]]; then
        log "INFO" "Decrypting backup..."
        gpg --decrypt "$backup_file" > "${backup_file%.gpg}"
        backup_file="${backup_file%.gpg}"
    fi

    # Decompress if needed
    if [[ "$backup_file" == *.gz ]]; then
        log "INFO" "Decompressing backup..."
        gunzip -k "$backup_file"
        backup_file="${backup_file%.gz}"
    fi

    # Extract backup
    log "INFO" "Extracting backup..."
    mkdir -p "$restore_path"

    if tar -xf "$backup_file" -C "$restore_path"; then
        log "SUCCESS" "Backup restored successfully to: $restore_path"
    else
        error_exit "Failed to extract backup" 1
    fi

    return 0
}
```

**String manipulation**:
- `${var%.gpg}` removes `.gpg` extension
- `${var#prefix}` removes prefix
- `${var:-.}` uses `.` as default if var is empty

### Step 10: Implement Cleanup Function

Add retention policy enforcement:

```bash
# ===========================
# Cleanup Functions
# ===========================

cleanup_old_backups() {
    log "INFO" "Cleaning up backups older than $RETENTION_DAYS days"

    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would clean up old backups"
    fi

    local removed_count=0
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%s)

    for backup_dir in "$BACKUP_DESTINATION"/${BACKUP_PREFIX}_*; do
        if [[ ! -d "$backup_dir" ]]; then
            continue
        fi

        local backup_time=$(stat -c %Y "$backup_dir")

        if [[ $backup_time -lt $cutoff_date ]]; then
            log "INFO" "Removing old backup: $(basename "$backup_dir")"

            if [[ "$DRY_RUN" == false ]]; then
                rm -rf "$backup_dir"
            fi

            ((removed_count++))
        fi
    done

    if [[ $removed_count -eq 0 ]]; then
        log "INFO" "No old backups to remove"
    else
        log "SUCCESS" "Removed $removed_count old backup(s)"
    fi

    return 0
}
```

### Step 11: Add Usage and Argument Parsing

Create user-friendly help and argument handling:

```bash
# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] ACTION [PATH]

Backup and restore ML training data.

ACTIONS:
    backup PATH         Create a new backup
    restore BACKUP_NAME [PATH]
                       Restore from backup
    list               List available backups
    cleanup            Remove old backups

OPTIONS:
    -d, --destination DIR   Backup destination directory
                           Default: /backup/ml-data
    -r, --retention DAYS    Retention period (default: 30)
    -c, --compress         Enable compression (default: true)
    -e, --encrypt          Enable encryption
    -n, --dry-run         Perform dry-run
    -v, --verbose         Enable verbose output
    -h, --help            Display this help message

EXAMPLES:
    # Create backup
    $SCRIPT_NAME backup /data/training-data

    # Create encrypted backup
    $SCRIPT_NAME --encrypt backup /data/models

    # List backups
    $SCRIPT_NAME list

    # Restore backup
    $SCRIPT_NAME restore ml-data_training-data_20240124_120000

    # Cleanup old backups
    $SCRIPT_NAME --retention 7 cleanup

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -d|--destination)
                BACKUP_DESTINATION="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -c|--compress)
                COMPRESS=true
                shift
                ;;
            -e|--encrypt)
                ENCRYPT=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            backup|restore|list|cleanup)
                ACTION="$1"
                shift
                break
                ;;
            -*)
                error_exit "Unknown option: $1" 1
                ;;
            *)
                error_exit "Invalid argument: $1" 1
                ;;
        esac
    done

    # Get remaining arguments
    ARGS=("$@")
}
```

**Argument parsing pattern**:
- `case` statement handles different options
- `shift 2` for options with values
- `shift` for flags
- `break` to stop parsing and capture remaining args

### Step 12: Add Main Function

Tie everything together:

```bash
# ===========================
# Main Function
# ===========================

main() {
    log "INFO" "=========================================="
    log "INFO" "ML Data Backup Script"
    log "INFO" "=========================================="

    parse_arguments "$@"

    if [[ -z "${ACTION:-}" ]]; then
        error_exit "Action is required (backup, restore, list, cleanup)" 1
    fi

    case "$ACTION" in
        backup)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                error_exit "Backup action requires PATH argument" 1
            fi
            create_backup "${ARGS[0]}"
            ;;
        restore)
            if [[ ${#ARGS[@]} -eq 0 ]]; then
                error_exit "Restore action requires BACKUP_NAME argument" 1
            fi
            restore_backup "${ARGS[0]}" "${ARGS[1]:-}"
            ;;
        list)
            list_backups
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        *)
            error_exit "Unknown action: $ACTION" 1
            ;;
    esac

    log "INFO" "=========================================="
    log "SUCCESS" "Operation completed successfully!"
    log "INFO" "=========================================="

    exit 0
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
```

### Step 13: Test the Backup Script

```bash
# Create test data
mkdir -p /tmp/test-data
echo "Sample ML dataset" > /tmp/test-data/dataset.csv
echo "Model weights" > /tmp/test-data/model.pkl

# Run dry-run first
./solutions/backup_data.sh --dry-run --verbose backup /tmp/test-data

# Create actual backup
sudo mkdir -p /backup/ml-data
sudo chown $USER:$USER /backup/ml-data
./solutions/backup_data.sh --verbose backup /tmp/test-data

# List backups
./solutions/backup_data.sh list

# Test restore
./solutions/backup_data.sh restore ml-data_test-data_<timestamp> /tmp/restored

# Verify
diff -r /tmp/test-data /tmp/restored/test-data

# Lint the script
shellcheck solutions/backup_data.sh
```

---

## Phase 2: Deploy Model Script (60 minutes)

### Summary

The `deploy_model.sh` script demonstrates model deployment automation. Key features:

- Model validation before deployment
- Backup of current production model
- Health checks post-deployment
- Rollback capability on failure
- Multiple environment support (dev, staging, prod)

**Implementation Steps**:

1. **Script Structure**: Similar to backup_data.sh with deployment-specific functions
2. **Validation**: Check model file format, size, dependencies
3. **Deployment**: Copy model to serving directory with proper permissions
4. **Health Check**: Verify model loads and responds correctly
5. **Rollback**: Restore previous model if deployment fails

**Key Functions**:
- `validate_model()`: Checks model integrity
- `backup_current_model()`: Saves current production model
- `deploy_model()`: Performs the actual deployment
- `health_check()`: Verifies deployment success
- `rollback()`: Restores previous model

---

## Phase 3: Log Analysis Script (60 minutes)

### Summary

The `analyze_logs.sh` script parses ML application logs for errors, warnings, and performance metrics.

**Features**:
- Parse multiple log formats (JSON, plain text)
- Extract error patterns
- Calculate statistics (error rates, response times)
- Generate alerts for critical issues
- Create summary reports

**Implementation Highlights**:

```bash
# Parse JSON logs
parse_json_logs() {
    local log_file="$1"

    jq -r 'select(.level == "ERROR") | .message' "$log_file"
}

# Extract error patterns
analyze_errors() {
    local log_file="$1"

    grep "ERROR" "$log_file" \
        | awk '{print $NF}' \
        | sort \
        | uniq -c \
        | sort -rn \
        | head -10
}

# Calculate metrics
calculate_metrics() {
    local log_file="$1"

    local total_requests=$(grep "REQUEST" "$log_file" | wc -l)
    local errors=$(grep "ERROR" "$log_file" | wc -l)
    local error_rate=$(echo "scale=2; ($errors / $total_requests) * 100" | bc)

    echo "Error Rate: ${error_rate}%"
}
```

---

## Phase 4: Health Check Script (60 minutes)

### Summary

The `health_check.sh` script monitors ML infrastructure services and triggers alerts.

**Components**:
- API endpoint checks
- Database connectivity
- Model serving status
- Resource utilization (CPU, memory, disk)
- Automated restarts for failed services

**Example Health Check**:

```bash
check_api_health() {
    local endpoint="$1"
    local timeout=5

    local response=$(curl -s -o /dev/null -w "%{http_code}" \
                          --max-time "$timeout" \
                          "$endpoint/health")

    if [[ "$response" == "200" ]]; then
        log "INFO" "API health check passed"
        return 0
    else
        log "ERROR" "API health check failed (HTTP $response)"
        return 1
    fi
}

check_model_serving() {
    local model_endpoint="$1"

    # Send test prediction request
    local test_payload='{"input": [1, 2, 3, 4, 5]}'
    local response=$(curl -s -X POST \
                          -H "Content-Type: application/json" \
                          -d "$test_payload" \
                          "$model_endpoint/predict")

    if [[ -n "$response" ]]; then
        log "INFO" "Model serving check passed"
        return 0
    else
        log "ERROR" "Model serving check failed"
        return 1
    fi
}
```

---

## Testing & Validation

### Run ShellCheck on All Scripts

```bash
# Lint all scripts
for script in solutions/*.sh; do
    echo "Checking $script..."
    shellcheck "$script"
done
```

### Create Test Suite

Create `tests/test_scripts.sh`:

```bash
#!/bin/bash

test_backup_script() {
    echo "Testing backup_data.sh..."

    # Test help
    ./solutions/backup_data.sh --help

    # Test dry-run
    ./solutions/backup_data.sh --dry-run backup /tmp

    echo "✓ Backup script tests passed"
}

test_deploy_script() {
    echo "Testing deploy_model.sh..."
    ./solutions/deploy_model.sh --help
    echo "✓ Deploy script tests passed"
}

# Run all tests
test_backup_script
test_deploy_script
```

---

## Best Practices Demonstrated

1. **Error Handling**: `set -euo pipefail` catches errors early
2. **Logging**: Structured logging with timestamps and levels
3. **Dry-Run Mode**: Test commands before execution
4. **Argument Parsing**: Support both short and long options
5. **Function Organization**: Single-responsibility functions
6. **Documentation**: Comprehensive comments and usage text
7. **Validation**: Check inputs before processing
8. **Idempotency**: Scripts can be run multiple times safely

---

## Common Pitfalls to Avoid

❌ **Not quoting variables**: `rm $file` (breaks with spaces)
✅ **Always quote**: `rm "$file"`

❌ **Using `cd` without checking**: `cd /path && command`
✅ **Check cd result**: `cd /path || exit 1; command`

❌ **Parsing ls output**: `for file in $(ls); do ...`
✅ **Use globs**: `for file in *; do ...`

❌ **Ignoring exit codes**: `command; do_something`
✅ **Check exit codes**: `if command; then do_something; fi`

---

## Next Steps

1. **Enhance the scripts**:
   - Add email notifications
   - Integrate with monitoring systems (Prometheus, Datadog)
   - Add remote backup support (S3, GCS)

2. **Create a deployment pipeline**:
   - Use these scripts in CI/CD
   - Add automated testing
   - Create configuration management

3. **Explore advanced topics**:
   - Parallel processing with GNU parallel
   - Advanced error recovery
   - Signal handling (SIGTERM, SIGINT)

---

## Resources

- [Bash Guide for Beginners](https://tldp.org/LDP/Bash-Beginners-Guide/html/)
- [Advanced Bash-Scripting Guide](https://tldp.org/LDP/abs/html/)
- [ShellCheck](https://www.shellcheck.net/) - Online linter
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
- [Bash Pitfalls](https://mywiki.wooledge.org/BashPitfalls)

---

**Congratulations!** You've created production-ready Bash automation scripts for ML infrastructure. These patterns apply to countless automation scenarios in your career. 🚀
