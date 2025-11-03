# Exercise 04: Implementation Guide
## Bash Scripting for ML Deployment Automation

This guide walks you through implementing production-ready bash scripts for ML infrastructure automation, step by step.

## Overview

You'll build four main automation scripts:
1. **deploy_model.sh** - Model deployment with validation and rollback
2. **process_data.sh** - Data pipeline automation
3. **monitor_system.sh** - System monitoring and alerting
4. **backup_ml_project.sh** - Backup and restore operations

## Prerequisites

- Completion of Exercises 01-03
- Basic bash knowledge (variables, functions, loops)
- Text editor (vim, nano, or VS Code)
- Access to a Linux terminal

## Step 1: Script Structure Foundation

### 1.1 Create Script Template

Every production script should follow this structure:

```bash
#!/bin/bash
# Shebang - tells system to use bash

set -euo pipefail
# -e: exit on error
# -u: exit on undefined variable
# -o pipefail: exit on pipe failures

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Cleanup on exit
cleanup() {
    # Cleanup code here
}
trap cleanup EXIT

# Usage information
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [options]
Options:
  -h, --help    Show help
EOF
    exit 0
}

# Main function
main() {
    # Your code here
}

main "$@"
```

**Key Concepts:**
- `readonly`: Makes variables immutable
- `$(command)`: Command substitution
- `trap`: Execute function on signal/exit
- `tee`: Write to both file and stdout

### 1.2 Add Colors for Better UX

```bash
# Colors for terminal output
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
echo -e "${GREEN}✓ Success${NC}"
echo -e "${RED}✗ Error${NC}"
echo -e "${YELLOW}⚠ Warning${NC}"
```

**Why `[[ -t 1 ]]`?**
- Checks if stdout is a terminal
- Disables colors when piping to file
- Prevents escape codes in log files

## Step 2: Model Deployment Script

### 2.1 Define Requirements

The deployment script must:
- Validate model format and size
- Support staging and production environments
- Create backups before deployment
- Enable rollback to previous version
- Log all operations

### 2.2 Implement Model Validation

```bash
validate_model() {
    local model_path="$1"

    # Check file exists
    if [ ! -f "$model_path" ]; then
        error_exit "Model file not found: $model_path"
    fi

    # Check file extension
    local extension="${model_path##*.}"
    case "$extension" in
        h5|pth|pkl|onnx|pb)
            log "✓ Valid model format: .$extension"
            ;;
        *)
            error_exit "Invalid format: .$extension"
            ;;
    esac

    # Check file size
    local size_bytes=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null)
    local size_mb=$((size_bytes / 1024 / 1024))

    if [ "$size_mb" -lt 1 ]; then
        error_exit "Model too small: ${size_mb}MB"
    fi

    if [ "$size_mb" -gt 5000 ]; then
        error_exit "Model too large: ${size_mb}MB"
    fi

    log "✓ Model size: ${size_mb}MB"
}
```

**Key Techniques:**
- `${var##*.}`: Extract file extension
- `stat -c%s`: Get file size (Linux)
- `stat -f%z`: Get file size (macOS)
- `case` statement for pattern matching

### 2.3 Implement Backup Function

```bash
backup_model() {
    local environment="$1"
    local source_dir="$MODELS_DIR/$environment"

    # Find current model
    local current_model=$(find "$source_dir" -maxdepth 1 -type f \
        \( -name "*.h5" -o -name "*.pth" \) | head -1)

    if [ -z "$current_model" ]; then
        log "No existing model to backup"
        return 0
    fi

    # Create backup with timestamp
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_name="${environment}_${timestamp}_$(basename "$current_model")"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    cp "$current_model" "$backup_path"
    log "✓ Backup created: $backup_name"

    # Retention policy: keep last 5 backups
    local backup_count=$(find "$BACKUP_DIR" -name "${environment}_*" | wc -l)
    if [ "$backup_count" -gt 5 ]; then
        find "$BACKUP_DIR" -name "${environment}_*" -type f | \
            sort | head -n -5 | xargs rm -f
        log "✓ Old backups cleaned up"
    fi
}
```

**Key Concepts:**
- `find` with multiple conditions using `-o` (OR)
- `head -1`: Get first result
- `sort | head -n -5`: Keep all but last 5 (oldest)
- `xargs`: Pass find results to rm

### 2.4 Implement Rollback

```bash
rollback_deployment() {
    # Find latest production backup
    local latest_backup=$(find "$BACKUP_DIR" -name "production_*" -type f | \
        sort -r | head -1)

    if [ -z "$latest_backup" ]; then
        error_exit "No backup available for rollback"
    fi

    log "Rolling back to: $(basename "$latest_backup")"

    # Remove current production model
    rm -f "${PRODUCTION_DIR}"/*

    # Restore backup
    local model_name=$(basename "$latest_backup" | \
        sed 's/production_[0-9]*_[0-9]*_//')
    cp "$latest_backup" "${PRODUCTION_DIR}/${model_name}"

    log "✓ Rollback completed: $model_name"
}
```

**Key Techniques:**
- `sort -r`: Reverse sort (newest first)
- `sed 's/pattern//'`: Remove pattern from string
- Always verify backup exists before attempting rollback

## Step 3: Data Pipeline Script

### 3.1 Design Pipeline Steps

1. Download/collect data
2. Validate data quality
3. Preprocess and clean
4. Merge datasets
5. Split into train/val/test (70/15/15)
6. Generate statistics

### 3.2 Implement Data Validation

```bash
validate_data() {
    log "Validating data quality"

    local file_count=$(find "$RAW_DIR" -name "*.csv" -type f | wc -l)

    if [ "$file_count" -eq 0 ]; then
        error_exit "No data files found in $RAW_DIR"
    fi

    log "Found $file_count data files"

    # Validate each file
    for data_file in "$RAW_DIR"/*.csv; do
        local filename=$(basename "$data_file")

        # Check file size
        local size=$(stat -c%s "$data_file" 2>/dev/null || \
                     stat -f%z "$data_file" 2>/dev/null)

        if [ "$size" -lt 100 ]; then
            error_exit "$filename: File too small ($size bytes)"
        fi

        # Check for header
        if head -1 "$data_file" | grep -q ","; then
            log "✓ $filename: Header detected"
        else
            error_exit "$filename: No header found"
        fi

        # Count records
        local record_count=$(($(wc -l < "$data_file") - 1))
        log "✓ $filename: $record_count records"
    done

    log "✓ Data validation passed"
}
```

**Key Techniques:**
- `wc -l < file`: Count lines without filename
- `$(( arithmetic ))`: Arithmetic expansion
- `grep -q`: Quiet mode (exit code only)

### 3.3 Implement Dataset Splitting

```bash
split_dataset() {
    local merged_file="$TEMP_DIR/merged_data.csv"
    local total_records=$(($(wc -l < "$merged_file") - 1))

    # Calculate split sizes (70/15/15)
    local train_size=$(( total_records * 70 / 100 ))
    local val_size=$(( total_records * 15 / 100 ))
    local test_size=$(( total_records - train_size - val_size ))

    log "Train: $train_size, Val: $val_size, Test: $test_size"

    # Extract header
    local header=$(head -1 "$merged_file")

    # Create train set
    {
        echo "$header"
        tail -n +2 "$merged_file" | head -n "$train_size"
    } > "$PROCESSED_DIR/train.csv"

    # Create val set
    {
        echo "$header"
        tail -n +2 "$merged_file" | tail -n +$(( train_size + 1 )) | \
            head -n "$val_size"
    } > "$PROCESSED_DIR/val.csv"

    # Create test set
    {
        echo "$header"
        tail -n +"$(( train_size + val_size + 2 ))" "$merged_file"
    } > "$PROCESSED_DIR/test.csv"

    log "✓ Dataset split complete"
}
```

**Key Concepts:**
- `tail -n +N`: Start from line N
- `{ commands; } > file`: Group commands, redirect all output
- Integer division for percentages

## Step 4: System Monitoring Script

### 4.1 Implement Metric Collection

```bash
get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | \
        sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | \
        awk '{print 100 - $1}'
}

get_memory_usage() {
    free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}'
}

get_disk_usage() {
    df -h / | tail -1 | awk '{print $5}' | sed 's/%//'
}

get_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu \
            --format=csv,noheader,nounits | head -1
    else
        echo "N/A"
    fi
}
```

**Key Techniques:**
- `top -bn1`: Batch mode, 1 iteration
- `awk '{printf "%.1f", expr}'`: Format floating point
- `command -v`: Check if command exists
- Pipe chain for data extraction

### 4.2 Implement Threshold Checking

```bash
check_threshold() {
    local value="$1"
    local threshold="$2"
    local metric_name="$3"

    # Handle N/A values
    if [ "$value" = "N/A" ]; then
        return 0
    fi

    # Convert to integer
    local value_int=$(printf "%.0f" "$value")

    if [ "$value_int" -gt "$threshold" ]; then
        log_alert "$metric_name: ${value}% (threshold: ${threshold}%)"
        return 1
    fi

    return 0
}

# Usage
if ! check_threshold "$cpu" 80 "CPU" || \
   ! check_threshold "$memory" 85 "Memory"; then
    send_alert "Resource threshold exceeded"
fi
```

**Key Concepts:**
- `printf "%.0f"`: Round floating point to integer
- Return codes: 0 = success, non-zero = failure
- `||` operator: Execute if previous command fails

### 4.3 Create Dashboard Display

```bash
display_dashboard() {
    clear

    echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ML Infrastructure Monitoring     ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════╣${NC}"

    printf "${BLUE}║${NC} CPU Usage:    "
    if (( $(echo "$cpu > $CPU_THRESHOLD" | bc -l) )); then
        printf "${RED}%.1f%%${NC} ⚠️\n" "$cpu"
    else
        printf "${GREEN}%.1f%%${NC}\n" "$cpu"
    fi

    # More metrics...

    echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
}
```

**Key Techniques:**
- `clear`: Clear terminal screen
- `printf` for formatted output
- `bc -l`: Floating point comparison
- Box drawing characters for visual appeal

## Step 5: Backup and Restore Script

### 5.1 Implement Compressed Backup

```bash
backup_project() {
    local project_name="$1"
    local project_dir="$PROJECTS_DIR/$project_name"

    # Verify project exists
    if [ ! -d "$project_dir" ]; then
        error_exit "Project directory not found: $project_dir"
    fi

    # Generate backup filename with timestamp
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_name="${project_name}_${timestamp}.tar.gz"
    local backup_path="$BACKUP_DIR/$backup_name"

    log "Creating compressed backup..."

    # Create backup excluding unnecessary files
    tar -czf "$backup_path" \
        --exclude='*.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.venv' \
        --exclude='node_modules' \
        --exclude='data/raw' \
        -C "$PROJECTS_DIR" \
        "$project_name"

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        error_exit "Backup creation failed"
    fi

    log "✓ Backup created: $backup_name"

    # Create metadata
    create_metadata "$project_name" "$project_dir" "$backup_path"
}
```

**Key Concepts:**
- `tar -czf`: Create (c), gzip (z), file (f)
- `--exclude`: Exclude patterns from archive
- `-C dir`: Change to directory before archiving
- `${PIPESTATUS[0]}`: Exit code of first command in pipe

### 5.2 Implement Metadata Generation

```bash
create_metadata() {
    local project_name="$1"
    local project_dir="$2"
    local backup_file="$3"
    local metadata_file="${backup_file}.meta"

    # Gather statistics
    local file_count=$(find "$project_dir" -type f | wc -l)
    local dir_count=$(find "$project_dir" -type d | wc -l)
    local total_size=$(du -sh "$project_dir" | awk '{print $1}')

    # Create JSON metadata
    cat > "$metadata_file" << EOF
{
  "project": "$project_name",
  "backup_file": "$(basename "$backup_file")",
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')",
  "created_by": "${USER:-unknown}",
  "statistics": {
    "files": $file_count,
    "directories": $dir_count,
    "size": "$total_size"
  }
}
EOF

    log "✓ Metadata created"
}
```

**Key Concepts:**
- `<< EOF`: Here document for multi-line strings
- `${USER:-unknown}`: Default value if USER unset
- `du -sh`: Disk usage, summary, human-readable

### 5.3 Implement Safe Restore

```bash
restore_backup() {
    local backup_file="$1"
    local backup_path="$BACKUP_DIR/$backup_file"

    # Verify backup exists
    if [ ! -f "$backup_path" ]; then
        error_exit "Backup file not found: $backup_path"
    fi

    # Extract project name from filename
    local project_name=$(echo "$backup_file" | \
        sed 's/_[0-9]*_[0-9]*.tar.gz//')
    local restore_dir="$PROJECTS_DIR/$project_name"

    # Check if project already exists
    if [ -d "$restore_dir" ]; then
        echo "⚠ Project directory already exists: $restore_dir"
        echo -n "Overwrite? (yes/no): "
        read -r response

        if [ "$response" != "yes" ]; then
            log "Restore cancelled"
            exit 0
        fi

        # Backup existing directory
        local existing_backup="${restore_dir}.backup_$(date '+%Y%m%d_%H%M%S')"
        mv "$restore_dir" "$existing_backup"
        log "Moved existing directory to: $existing_backup"
    fi

    # Extract backup
    tar -xzf "$backup_path" -C "$PROJECTS_DIR"

    log "✓ Restore completed"
}
```

**Key Concepts:**
- `read -r`: Read user input (raw mode)
- `mv`: Rename/move for safety before overwrite
- `tar -xzf`: Extract (x), gzip (z), file (f)

## Step 6: Testing Your Scripts

### 6.1 Create Test Data

```bash
# Test model deployment
mkdir -p ~/test-models
dd if=/dev/zero of=~/test-models/model.h5 bs=1M count=10
./deploy_model.sh ~/test-models/model.h5 staging

# Test data pipeline
./process_data.sh run
ls -lh ../data/processed/

# Test monitoring
timeout 30 ./monitor_system.sh monitor 5

# Test backup
mkdir -p ~/ml-projects/test-project
echo "Test README" > ~/ml-projects/test-project/README.md
./backup_ml_project.sh backup test-project
./backup_ml_project.sh list
```

### 6.2 Run Validation

```bash
cd scripts
chmod +x *.sh
./validate_exercise.sh
```

The validation script checks:
- All required scripts exist
- Scripts are executable
- Scripts have proper structure (shebang, set options, usage)
- Scripts respond to --help
- Basic functionality works

## Step 7: Best Practices Applied

### 7.1 Error Handling

```bash
# Always use set options
set -euo pipefail

# Trap errors
trap 'error_handler ${LINENO}' ERR

error_handler() {
    local line_number="$1"
    log_error "Script failed at line $line_number"
    cleanup
    exit 1
}
```

### 7.2 Logging

```bash
# Consistent log format
log() {
    local level="$1"
    shift
    local message="$*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | \
        tee -a "$LOG_FILE"
}

# Different log levels
log_info "Information message"
log_warning "Warning message"
log_error "Error message"
```

### 7.3 Function Organization

```bash
# Group related functions
# 1. Configuration and setup
# 2. Utility functions
# 3. Core business logic
# 4. Main function
# 5. Script invocation

# Use descriptive function names
validate_model()           # Good
check_it()                 # Bad

# Keep functions focused (single responsibility)
deploy_model() {
    validate_model "$model"
    backup_current_model
    copy_new_model "$model"
    verify_deployment
}
```

### 7.4 Variable Naming

```bash
# Use readonly for constants
readonly MAX_SIZE=5000
readonly BACKUP_DIR="/path/to/backups"

# Use local for function variables
function_name() {
    local model_path="$1"
    local environment="$2"
}

# Use descriptive names
model_path="/path/to/model.h5"  # Good
mp="/path/to/model.h5"           # Bad
```

## Common Issues and Solutions

### Issue 1: Permission Denied

**Symptom:**
```
bash: ./script.sh: Permission denied
```

**Solution:**
```bash
chmod +x script.sh
```

### Issue 2: Unbound Variable

**Symptom:**
```
script.sh: line 42: variable: unbound variable
```

**Solution:**
```bash
# Use default values
VAR=${VAR:-default_value}

# Or check before use
if [ -z "$VAR" ]; then
    error_exit "VAR is not set"
fi
```

### Issue 3: Command Not Found

**Symptom:**
```
script.sh: line 10: nvidia-smi: command not found
```

**Solution:**
```bash
# Check if command exists
if ! command -v nvidia-smi &> /dev/null; then
    log_warning "nvidia-smi not available"
    gpu_usage="N/A"
else
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits)
fi
```

### Issue 4: Array Expansion

**Symptom:**
Arrays not working as expected

**Solution:**
```bash
# Always quote array expansions
for item in "${ARRAY[@]}"; do
    echo "$item"
done

# Not: for item in ${ARRAY[@]}
```

### Issue 5: Floating Point Comparison

**Symptom:**
```
integer expression expected
```

**Solution:**
```bash
# Use bc for floating point
if (( $(echo "$value > 80.5" | bc -l) )); then
    echo "Threshold exceeded"
fi

# Or convert to integer first
value_int=$(printf "%.0f" "$value")
if [ "$value_int" -gt 80 ]; then
    echo "Threshold exceeded"
fi
```

## Advanced Topics

### Topic 1: Argument Parsing with getopts

```bash
while getopts "hv:d:n" opt; do
    case $opt in
        h)
            usage
            ;;
        v)
            VERBOSE=true
            ;;
        d)
            DEBUG=true
            ;;
        n)
            DRY_RUN=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

shift $((OPTIND-1))
```

### Topic 2: Parallel Execution

```bash
# Run tasks in parallel
process_file() {
    local file="$1"
    # Processing logic
}

export -f process_file

# Process files in parallel (4 at a time)
find . -name "*.csv" | xargs -P 4 -I {} bash -c 'process_file "$@"' _ {}
```

### Topic 3: Configuration Files

```bash
# Load configuration from file
load_config() {
    local config_file="$1"

    if [ -f "$config_file" ]; then
        # Source config file safely
        source "$config_file"
        log "✓ Configuration loaded"
    else
        log_warning "Config file not found, using defaults"
    fi
}

# Example config file (config.sh):
# BACKUP_RETENTION=7
# MAX_MODEL_SIZE=5000
# ALERT_EMAIL="admin@example.com"
```

## Completion Checklist

- [ ] All 5 scripts created and executable
- [ ] Each script has proper shebang and set options
- [ ] All scripts have usage/help functions
- [ ] Error handling implemented with trap
- [ ] Logging with timestamps
- [ ] Color-coded output for terminals
- [ ] Input validation
- [ ] Cleanup functions
- [ ] Test scripts with sample data
- [ ] Validation script passes
- [ ] Documentation reviewed

## Next Steps

After completing this exercise:

1. **Complete Exercise 05**: Package Management - Learn to manage system packages and dependencies

2. **Complete Exercise 06**: Log Analysis - Analyze system logs for troubleshooting

3. **Integrate with CI/CD**: Use these scripts in GitLab CI or GitHub Actions

4. **Add Monitoring**: Integrate with Prometheus, Grafana, or CloudWatch

5. **Extend Scripts**: Add more features like:
   - Email notifications
   - Slack/Teams integration
   - Database logging
   - Web dashboards

## Resources

- [Bash Guide](https://mywiki.wooledge.org/BashGuide)
- [ShellCheck](https://www.shellcheck.net/) - Shell script analyzer
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
- [Advanced Bash-Scripting Guide](https://tldp.org/LDP/abs/html/)

## Summary

In this exercise, you've learned:

✅ Bash scripting best practices (set options, error handling, logging)
✅ Model deployment automation with validation and rollback
✅ Data pipeline automation for ML workflows
✅ System monitoring with threshold-based alerting
✅ Backup and restore operations with metadata
✅ Production-ready script structure and organization

These scripts form the foundation of ML infrastructure automation and can be adapted for various production scenarios.

**Time investment**: 150-170 minutes
**Difficulty**: Intermediate
**Production-readiness**: High

---

**Exercise 04: Bash Scripting for ML Deployment Automation - Implementation Complete!**
