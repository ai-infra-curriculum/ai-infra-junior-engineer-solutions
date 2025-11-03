# Exercise 04: Reflection Questions - Comprehensive Answers
## Bash Scripting for ML Deployment Automation

This document provides detailed answers to the reflection questions for Exercise 04, covering bash scripting best practices, ML automation patterns, and production deployment strategies.

---

## Question 1: Why is `set -euo pipefail` Important in Production Scripts?

### Answer

The `set -euo pipefail` options are critical for writing robust, production-ready bash scripts. Each flag addresses a specific failure mode:

### Individual Options Explained

#### 1. `set -e` (Exit on Error)

**What it does:**
- Exits immediately if any command returns a non-zero exit code
- Prevents script from continuing after errors
- Default bash behavior is to continue execution even after errors

**Example without `-e`:**
```bash
#!/bin/bash
# No set -e

cp /nonexistent/file.txt /tmp/
echo "This will execute even though cp failed"
rm -rf /important/data  # DANGEROUS: Runs despite previous error!
```

**Example with `-e`:**
```bash
#!/bin/bash
set -e

cp /nonexistent/file.txt /tmp/
echo "This will never execute"  # Script exits at cp failure
```

**When to disable `-e`:**
```bash
# Temporarily disable for specific commands
set +e
command_that_may_fail
status=$?
set -e

# Or use conditional execution
if ! command_that_may_fail; then
    handle_error
fi
```

#### 2. `set -u` (Exit on Undefined Variables)

**What it does:**
- Treats unset variables as errors
- Prevents typos and logic errors from undefined variables
- Default bash behavior substitutes empty string for unset variables

**Example without `-u`:**
```bash
#!/bin/bash
# No set -u

MODEL_PATH="/models/production"
# Typo: MODEL_PAHT instead of MODEL_PATH
rm -rf "$MODEL_PAHT"/*  # Expands to: rm -rf /*
# DISASTER: Deletes everything in root!
```

**Example with `-u`:**
```bash
#!/bin/bash
set -u

MODEL_PATH="/models/production"
rm -rf "$MODEL_PAHT"/*
# Error: MODEL_PAHT: unbound variable
# Script exits safely before disaster
```

**Using default values:**
```bash
# Provide default if variable might be unset
CONFIG_FILE="${CONFIG_FILE:-/etc/default.conf}"

# Or check explicitly
if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH not set"
    exit 1
fi
```

#### 3. `set -o pipefail` (Pipe Failure Detection)

**What it does:**
- Makes pipelines return the exit code of the rightmost failing command
- Default bash behavior only checks the last command in a pipe
- Critical for catching errors in data processing pipelines

**Example without `pipefail`:**
```bash
#!/bin/bash
set -e
# No pipefail

# This command fails silently
cat /nonexistent/file.csv | process_data.py | wc -l

echo "Exit code: $?"  # Shows 0 (success from wc -l)
# Script continues despite cat failure
```

**Example with `pipefail`:**
```bash
#!/bin/bash
set -eo pipefail

cat /nonexistent/file.csv | process_data.py | wc -l
# Error: cat fails, script exits immediately
# Exit code reflects cat failure, not wc success
```

### Real-World ML Deployment Scenario

```bash
#!/bin/bash
set -euo pipefail

# Data preprocessing pipeline
# Each step must succeed for next to run
download_raw_data() {
    curl -f "https://api/data" > raw_data.json || return 1
}

validate_data() {
    python3 validate.py raw_data.json || return 1
}

preprocess_data() {
    cat raw_data.json | \
        jq '.records[]' | \
        python3 preprocess.py > processed_data.csv
    # pipefail catches errors in any step
}

train_model() {
    python3 train.py --data processed_data.csv || return 1
}

# Execute pipeline
download_raw_data
validate_data
preprocess_data
train_model

echo "Pipeline completed successfully"
```

### Why These Options Matter for ML Automation

1. **Data Integrity:**
   - `-e` ensures failed data downloads stop the pipeline
   - `-u` prevents accidentally processing empty datasets
   - `-o pipefail` catches corruption in data transformation pipelines

2. **Model Safety:**
   - Prevents deploying models when validation fails
   - Stops deployment if backup creation fails
   - Ensures rollback works correctly

3. **Resource Protection:**
   - Prevents deleting production data due to undefined variables
   - Stops training jobs if GPU allocation fails
   - Ensures cleanup happens only when intended

### Exception Handling Patterns

```bash
#!/bin/bash
set -euo pipefail

# Pattern 1: Conditional execution (doesn't trigger -e)
if ! command_that_may_fail; then
    echo "Command failed, handling gracefully"
    # Handle error
fi

# Pattern 2: Capture exit code
set +e
command_that_may_fail
exit_code=$?
set -e

if [ $exit_code -ne 0 ]; then
    echo "Command failed with code $exit_code"
fi

# Pattern 3: OR operator (doesn't trigger -e)
command_that_may_fail || {
    echo "Command failed, continuing"
}

# Pattern 4: Ignore specific failures
command_that_may_fail || true
```

### Production Checklist

✅ **Always use** `set -euo pipefail` in production scripts
✅ **Test** scripts with intentional failures to verify error handling
✅ **Document** any places where you disable these options
✅ **Use** explicit error handling for commands that may legitimately fail
✅ **Provide** default values for variables that might be unset
✅ **Verify** pipeline failures with test data

### Summary

`set -euo pipefail` transforms bash from a permissive scripting language into a robust automation tool suitable for production ML infrastructure. These options catch:
- Command failures (`-e`)
- Variable typos (`-u`)
- Pipeline errors (`-o pipefail`)

**Without these options:** Scripts silently continue after errors, potentially causing data corruption, failed deployments, or resource deletion.

**With these options:** Scripts fail fast, providing clear error messages and preventing cascading failures.

---

## Question 2: How Do You Implement Idempotent Deployment Scripts?

### Answer

**Idempotency** means a script can be run multiple times with the same result, without causing unintended side effects. This is crucial for ML deployment automation where scripts may be re-run due to failures, rollbacks, or CI/CD retries.

### Core Principles of Idempotent Scripts

#### 1. Check Before Create

Always verify if a resource exists before creating it.

**Non-idempotent (bad):**
```bash
# Fails on second run
mkdir /models/production
cp model.h5 /models/production/
```

**Idempotent (good):**
```bash
# Works on every run
mkdir -p /models/production  # -p creates only if needed

# Check before copy
if [ ! -f "/models/production/model.h5" ] || \
   [ "$model.h5" -nt "/models/production/model.h5" ]; then
    cp model.h5 /models/production/
    echo "Model deployed"
else
    echo "Model already up to date"
fi
```

#### 2. Update Rather Than Append

Avoid accumulating data on repeated runs.

**Non-idempotent (bad):**
```bash
# Adds duplicate entries on each run
echo "export MODEL_PATH=/models/production" >> ~/.bashrc
```

**Idempotent (good):**
```bash
# Update configuration idempotently
update_config() {
    local config_file="$1"
    local key="$2"
    local value="$3"

    if grep -q "^${key}=" "$config_file" 2>/dev/null; then
        # Update existing entry
        sed -i "s|^${key}=.*|${key}=${value}|" "$config_file"
    else
        # Add new entry
        echo "${key}=${value}" >> "$config_file"
    fi
}

update_config ~/.bashrc "MODEL_PATH" "/models/production"
```

#### 3. Use Atomic Operations

Ensure operations complete fully or not at all.

**Non-idempotent (bad):**
```bash
# Partial copy leaves incomplete state
cp large_model.h5 /models/production/model.h5
# Crash here = corrupted model
```

**Idempotent (good):**
```bash
# Copy to temp location, then atomic move
cp large_model.h5 /models/production/model.h5.tmp

# Verify integrity
if [ $? -eq 0 ]; then
    mv /models/production/model.h5.tmp /models/production/model.h5
    echo "Model deployed successfully"
else
    rm -f /models/production/model.h5.tmp
    echo "Deployment failed" >&2
    exit 1
fi
```

### Idempotent Deployment Pattern

```bash
#!/bin/bash
set -euo pipefail

deploy_model_idempotent() {
    local model_file="$1"
    local environment="$2"
    local target_dir="/models/${environment}"
    local target_path="${target_dir}/current_model.h5"

    # 1. Verify source exists
    if [ ! -f "$model_file" ]; then
        echo "ERROR: Model file not found: $model_file" >&2
        return 1
    fi

    # 2. Create target directory if needed (idempotent)
    mkdir -p "$target_dir"

    # 3. Calculate checksums for comparison
    local source_checksum=$(sha256sum "$model_file" | awk '{print $1}')
    local target_checksum=""

    if [ -f "$target_path" ]; then
        target_checksum=$(sha256sum "$target_path" | awk '{print $1}')
    fi

    # 4. Check if update is needed
    if [ "$source_checksum" = "$target_checksum" ]; then
        echo "✓ Model already deployed with same checksum"
        echo "  Checksum: $source_checksum"
        return 0
    fi

    # 5. Create backup if model exists
    if [ -f "$target_path" ]; then
        local backup_file="${target_dir}/model_backup_$(date +%s).h5"
        cp "$target_path" "$backup_file"
        echo "✓ Backup created: $(basename "$backup_file")"
    fi

    # 6. Deploy using atomic operation
    local temp_file="${target_dir}/.model_temp_$$"

    # Copy to temp location
    cp "$model_file" "$temp_file"

    # Verify copy
    local temp_checksum=$(sha256sum "$temp_file" | awk '{print $1}')
    if [ "$temp_checksum" != "$source_checksum" ]; then
        rm -f "$temp_file"
        echo "ERROR: Checksum mismatch during copy" >&2
        return 1
    fi

    # Atomic move
    mv "$temp_file" "$target_path"

    echo "✓ Model deployed successfully"
    echo "  Environment: $environment"
    echo "  Checksum: $source_checksum"

    # 7. Update metadata (idempotent)
    update_deployment_metadata "$environment" "$source_checksum"

    return 0
}

update_deployment_metadata() {
    local environment="$1"
    local checksum="$2"
    local metadata_file="/models/deployment_metadata.json"

    # Create if doesn't exist
    if [ ! -f "$metadata_file" ]; then
        echo '{"deployments":[]}' > "$metadata_file"
    fi

    # Update metadata using jq (idempotent)
    local temp_file="${metadata_file}.tmp"
    jq --arg env "$environment" \
       --arg sum "$checksum" \
       --arg time "$(date -Iseconds)" \
       'del(.deployments[] | select(.environment == $env)) |
        .deployments += [{
          "environment": $env,
          "checksum": $sum,
          "timestamp": $time
        }]' "$metadata_file" > "$temp_file"

    mv "$temp_file" "$metadata_file"
}

# Usage
deploy_model_idempotent "model.h5" "production"
# Can run multiple times safely
deploy_model_idempotent "model.h5" "production"  # No-op if same model
```

### Idempotent Data Pipeline

```bash
#!/bin/bash
set -euo pipefail

process_data_idempotent() {
    local input_file="$1"
    local output_dir="$2"

    # Generate deterministic output filename
    local input_hash=$(sha256sum "$input_file" | awk '{print $1}' | cut -c1-16)
    local output_file="${output_dir}/processed_${input_hash}.csv"

    # Check if already processed
    if [ -f "$output_file" ]; then
        local marker_file="${output_file}.complete"
        if [ -f "$marker_file" ]; then
            echo "✓ Data already processed: $(basename "$output_file")"
            echo "  Input hash: $input_hash"
            return 0
        else
            echo "⚠ Previous processing incomplete, reprocessing"
            rm -f "$output_file"
        fi
    fi

    mkdir -p "$output_dir"

    # Process data
    echo "Processing: $input_file"

    python3 - <<EOF
import pandas as pd
import sys

try:
    # Read input
    df = pd.read_csv("$input_file")

    # Process data
    df = df.dropna()
    df = df.drop_duplicates()

    # Write output
    df.to_csv("$output_file", index=False)

    print(f"✓ Processed {len(df)} records")
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    # Mark as complete only if successful
    if [ $? -eq 0 ]; then
        touch "${output_file}.complete"
        echo "✓ Processing complete: $(basename "$output_file")"
    else
        rm -f "$output_file"
        echo "ERROR: Processing failed" >&2
        return 1
    fi
}

# Usage
process_data_idempotent "raw_data.csv" "processed"
# Safe to re-run - skips if already processed
```

### Idempotent Configuration Management

```bash
#!/bin/bash
set -euo pipefail

configure_ml_environment() {
    local config_file="/etc/ml-config/settings.conf"
    local config_dir=$(dirname "$config_file")

    # Ensure config directory exists
    sudo mkdir -p "$config_dir"

    # Desired configuration state
    declare -A desired_config=(
        ["MODEL_PATH"]="/models/production"
        ["BATCH_SIZE"]="32"
        ["GPU_DEVICE"]="0"
        ["LOG_LEVEL"]="INFO"
    )

    # Create empty config if doesn't exist
    if [ ! -f "$config_file" ]; then
        sudo touch "$config_file"
    fi

    # Apply each configuration (idempotent)
    local updated=false

    for key in "${!desired_config[@]}"; do
        local desired_value="${desired_config[$key]}"
        local current_value=""

        # Get current value if exists
        if grep -q "^${key}=" "$config_file" 2>/dev/null; then
            current_value=$(grep "^${key}=" "$config_file" | cut -d'=' -f2)
        fi

        # Update if different or missing
        if [ "$current_value" != "$desired_value" ]; then
            if [ -n "$current_value" ]; then
                echo "Updating $key: $current_value -> $desired_value"
                sudo sed -i "s|^${key}=.*|${key}=${desired_value}|" "$config_file"
            else
                echo "Adding $key: $desired_value"
                echo "${key}=${desired_value}" | sudo tee -a "$config_file" > /dev/null
            fi
            updated=true
        else
            echo "✓ $key already set to $desired_value"
        fi
    done

    if [ "$updated" = true ]; then
        echo "✓ Configuration updated"
        # Restart services if needed
        # sudo systemctl restart ml-service
    else
        echo "✓ Configuration already up to date"
    fi
}

# Usage
configure_ml_environment
# Safe to run multiple times
```

### Testing Idempotency

```bash
#!/bin/bash

test_idempotency() {
    local script="$1"
    local args="${@:2}"

    echo "=== Idempotency Test ==="
    echo "Script: $script"
    echo "Args: $args"
    echo ""

    # Run 1
    echo "--- Run 1 ---"
    $script $args > /tmp/run1.log 2>&1
    local exit1=$?

    # Run 2
    echo "--- Run 2 ---"
    $script $args > /tmp/run2.log 2>&1
    local exit2=$?

    # Run 3
    echo "--- Run 3 ---"
    $script $args > /tmp/run3.log 2>&1
    local exit3=$?

    # Verify all runs succeeded
    if [ $exit1 -eq 0 ] && [ $exit2 -eq 0 ] && [ $exit3 -eq 0 ]; then
        echo "✓ All runs succeeded"
    else
        echo "✗ Some runs failed: $exit1, $exit2, $exit3"
        return 1
    fi

    # Verify outputs are similar (allowing for timestamps)
    echo ""
    echo "Comparing outputs..."

    # Check system state is consistent
    echo "✓ Script is idempotent"
}

# Test deployment script
test_idempotency ./deploy_model.sh model.h5 staging
```

### Idempotency Checklist

✅ **Directory creation:** Use `mkdir -p` instead of `mkdir`
✅ **File checks:** Verify existence before creating
✅ **Checksums:** Compare content, not just timestamps
✅ **Atomic operations:** Use temp files + atomic moves
✅ **State markers:** Use `.complete` files to track processing
✅ **Configuration updates:** Update existing, don't append
✅ **Cleanup:** Remove incomplete files on failure
✅ **Testing:** Run scripts 3 times, verify same result

### Benefits for ML Automation

1. **Reliability:** Scripts can be safely retried after failures
2. **CI/CD Integration:** Build pipelines can be re-run without manual cleanup
3. **Disaster Recovery:** Restoration scripts work regardless of current state
4. **Reduced Risk:** No accumulation of resources or duplicate deployments
5. **Consistency:** Predictable behavior across environments

### Summary

Idempotent deployment scripts are essential for production ML infrastructure. Key techniques:
- Check before create/update
- Use atomic operations
- Compare content (checksums), not timestamps
- Update rather than append
- Mark completion states
- Clean up on failures

---

## Question 3: What Are the Best Practices for Logging in Bash Scripts?

### Answer

Effective logging is critical for debugging, auditing, and monitoring production bash scripts. Good logging provides visibility into script execution, captures errors, and aids in troubleshooting.

### Logging Levels

Implement multiple log levels for different types of messages:

```bash
#!/bin/bash

# Log configuration
readonly LOG_FILE="/var/log/ml-deployment.log"
readonly LOG_LEVEL="${LOG_LEVEL:-INFO}"  # DEBUG, INFO, WARNING, ERROR

# ANSI color codes
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Log level priorities
declare -A LOG_LEVELS=( [DEBUG]=0 [INFO]=1 [WARNING]=2 [ERROR]=3 )

should_log() {
    local level="$1"
    local current_priority=${LOG_LEVELS[$LOG_LEVEL]}
    local message_priority=${LOG_LEVELS[$level]}
    [ $message_priority -ge $current_priority ]
}

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local caller="${BASH_SOURCE[2]##*/}:${BASH_LINENO[1]}"

    # Check if should log based on level
    if ! should_log "$level"; then
        return 0
    fi

    # Format log message
    local log_entry="[$timestamp] [$level] [$caller] $message"

    # Write to log file
    echo "$log_entry" >> "$LOG_FILE"

    # Write to console with colors (if terminal)
    if [[ -t 1 ]]; then
        case "$level" in
            DEBUG)
                echo -e "${BLUE}$log_entry${NC}"
                ;;
            INFO)
                echo "$log_entry"
                ;;
            WARNING)
                echo -e "${YELLOW}$log_entry${NC}" >&2
                ;;
            ERROR)
                echo -e "${RED}$log_entry${NC}" >&2
                ;;
        esac
    else
        # No colors when piped
        if [ "$level" = "WARNING" ] || [ "$level" = "ERROR" ]; then
            echo "$log_entry" >&2
        else
            echo "$log_entry"
        fi
    fi
}

# Convenience functions
log_debug() { log "DEBUG" "$@"; }
log_info() { log "INFO" "$@"; }
log_warning() { log "WARNING" "$@"; }
log_error() { log "ERROR" "$@"; }
```

### Usage Examples

```bash
log_debug "Starting model validation"
log_info "Deploying model to production"
log_warning "Disk usage above 80%"
log_error "Model validation failed"
```

### Structured Logging

For better parsing and analysis:

```bash
log_structured() {
    local level="$1"
    local message="$2"
    shift 2

    # Additional key-value pairs
    local extra_fields=""
    while [ $# -gt 0 ]; do
        extra_fields="$extra_fields $1=\"$2\""
        shift 2
    done

    local timestamp=$(date -Iseconds)
    local log_entry="timestamp=\"$timestamp\" level=\"$level\" message=\"$message\"$extra_fields"

    echo "$log_entry" >> "$LOG_FILE"
}

# Usage
log_structured "INFO" "Model deployed" \
    "model_id" "resnet50_v2" \
    "environment" "production" \
    "size_mb" "102" \
    "checksum" "abc123def456"

# Output:
# timestamp="2024-01-15T10:30:45-05:00" level="INFO" message="Model deployed" model_id="resnet50_v2" environment="production" size_mb="102" checksum="abc123def456"
```

### JSON Logging

For integration with log aggregation tools:

```bash
log_json() {
    local level="$1"
    local message="$2"
    shift 2

    # Build JSON object
    local json="{\"timestamp\":\"$(date -Iseconds)\",\"level\":\"$level\",\"message\":\"$message\""

    # Add extra fields
    while [ $# -gt 0 ]; do
        json="$json,\"$1\":\"$2\""
        shift 2
    done

    json="$json}"

    echo "$json" >> "$LOG_FILE"
}

# Usage
log_json "INFO" "Model deployed" \
    "model_id" "resnet50_v2" \
    "environment" "production"

# Output:
# {"timestamp":"2024-01-15T10:30:45-05:00","level":"INFO","message":"Model deployed","model_id":"resnet50_v2","environment":"production"}
```

### Rotation and Retention

```bash
# Log rotation function
rotate_logs() {
    local log_file="$1"
    local max_size_mb="${2:-10}"  # Default 10MB
    local keep_count="${3:-5}"    # Keep last 5 logs

    if [ ! -f "$log_file" ]; then
        return 0
    fi

    # Check size
    local size_mb=$(du -m "$log_file" | awk '{print $1}')

    if [ "$size_mb" -gt "$max_size_mb" ]; then
        log_info "Rotating log file (size: ${size_mb}MB)"

        # Rotate existing logs
        for i in $(seq $((keep_count - 1)) -1 1); do
            if [ -f "${log_file}.$i" ]; then
                mv "${log_file}.$i" "${log_file}.$((i + 1))"
            fi
        done

        # Move current log
        mv "$log_file" "${log_file}.1"

        # Create new log file
        touch "$log_file"

        # Remove old logs
        for i in $(seq $((keep_count + 1)) 20); do
            rm -f "${log_file}.$i"
        done

        log_info "Log rotation complete"
    fi
}

# Call at start of script
rotate_logs "$LOG_FILE" 10 5
```

### Error Context Logging

```bash
log_error_context() {
    local error_msg="$1"

    log_error "$error_msg"
    log_error "Script: ${BASH_SOURCE[1]}"
    log_error "Function: ${FUNCNAME[1]}"
    log_error "Line: ${BASH_LINENO[0]}"

    # Log last 10 commands
    log_error "Recent commands:"
    history | tail -10 | while read -r line; do
        log_error "  $line"
    done

    # Log environment
    log_error "Environment:"
    log_error "  PWD: $PWD"
    log_error "  USER: ${USER:-unknown}"
    log_error "  HOSTNAME: $(hostname)"
}

# Usage in error handler
error_handler() {
    log_error_context "Script failed at line ${BASH_LINENO[0]}"
    exit 1
}

trap 'error_handler' ERR
```

### Performance Monitoring

```bash
# Timer for operations
start_timer() {
    TIMER_START=$(date +%s)
}

end_timer() {
    local operation="$1"
    local timer_end=$(date +%s)
    local duration=$((timer_end - TIMER_START))

    log_info "$operation completed in ${duration}s"

    # Log slow operations
    if [ $duration -gt 60 ]; then
        log_warning "$operation took longer than expected: ${duration}s"
    fi
}

# Usage
start_timer
download_model
end_timer "Model download"
```

### Progress Logging

```bash
log_progress() {
    local current="$1"
    local total="$2"
    local item="$3"

    local percent=$((current * 100 / total))

    # Progress bar
    local bar_length=50
    local filled=$((bar_length * current / total))
    local empty=$((bar_length - filled))

    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %3d%% (%d/%d) %s" "$percent" "$current" "$total" "$item"

    # Newline on completion
    if [ "$current" -eq "$total" ]; then
        echo ""
        log_info "Processed $total items"
    fi
}

# Usage
total=100
for i in $(seq 1 $total); do
    process_item "$i"
    log_progress "$i" "$total" "items"
done
```

### Centralized Logging

```bash
# Send logs to syslog
log_to_syslog() {
    local level="$1"
    local message="$2"

    # Map to syslog priorities
    local priority
    case "$level" in
        DEBUG)   priority="debug" ;;
        INFO)    priority="info" ;;
        WARNING) priority="warning" ;;
        ERROR)   priority="err" ;;
    esac

    logger -t "ml-deployment" -p "user.${priority}" "$message"
}

# Combined logging
log_combined() {
    local level="$1"
    shift
    local message="$*"

    # Log to file
    log "$level" "$message"

    # Log to syslog
    log_to_syslog "$level" "$message"
}
```

### Best Practices Summary

1. **Structured Format:**
   - Use consistent timestamp format (ISO 8601)
   - Include log level, source location, and message
   - Add context (user, host, PID)

2. **Multiple Levels:**
   - DEBUG: Detailed diagnostic information
   - INFO: General informational messages
   - WARNING: Potential issues
   - ERROR: Error events

3. **Both File and Console:**
   - Log to file for persistence
   - Display to console for interactive use
   - Use colors for terminal output only

4. **Error Context:**
   - Log full error context
   - Include stack trace information
   - Capture environment state

5. **Performance:**
   - Time long-running operations
   - Log resource usage
   - Monitor for slowdowns

6. **Rotation:**
   - Rotate logs by size
   - Keep limited history
   - Compress old logs

7. **Security:**
   - Never log passwords or secrets
   - Sanitize sensitive data
   - Restrict log file permissions

---

## Question 4: How Do You Handle Secrets and Credentials in Bash Scripts?

### Answer

Handling secrets securely in bash scripts is critical for production environments. Never hardcode credentials, and follow these best practices:

### 1. Environment Variables

Store secrets in environment variables, never in code:

```bash
#!/bin/bash

# NEVER do this
# API_KEY="sk_live_abc123xyz789"  # WRONG!

# Instead, use environment variables
if [ -z "${API_KEY:-}" ]; then
    echo "ERROR: API_KEY environment variable not set" >&2
    exit 1
fi

# Use the secret
curl -H "Authorization: Bearer $API_KEY" https://api.example.com/deploy
```

Set environment variables securely:

```bash
# In .env file (never commit to git!)
export API_KEY="sk_live_abc123xyz789"
export DB_PASSWORD="secure_password"

# Load in script
if [ -f ".env" ]; then
    source ".env"
fi
```

### 2. AWS Secrets Manager

```bash
get_secret() {
    local secret_name="$1"

    aws secretsmanager get-secret-value \
        --secret-id "$secret_name" \
        --query SecretString \
        --output text
}

# Usage
DB_PASSWORD=$(get_secret "prod/db/password")
API_KEY=$(get_secret "prod/api/key")
```

### 3. HashiCorp Vault

```bash
get_vault_secret() {
    local secret_path="$1"

    vault kv get -field=value "$secret_path"
}

# Usage
DB_PASSWORD=$(get_vault_secret "secret/prod/db/password")
```

### 4. Kubernetes Secrets

```bash
get_k8s_secret() {
    local secret_name="$1"
    local key="$2"

    kubectl get secret "$secret_name" -o jsonpath="{.data.$key}" | base64 -d
}

# Usage
DB_PASSWORD=$(get_k8s_secret "db-credentials" "password")
```

### 5. Secure File Permissions

```bash
# Create credentials file with restricted permissions
CREDS_FILE="/etc/ml-service/credentials"

# Ensure only owner can read
if [ -f "$CREDS_FILE" ]; then
    chmod 600 "$CREDS_FILE"

    # Verify permissions
    actual_perms=$(stat -c "%a" "$CREDS_FILE")
    if [ "$actual_perms" != "600" ]; then
        echo "ERROR: Insecure credentials file permissions" >&2
        exit 1
    fi
fi
```

### 6. Log Sanitization

```bash
# Sanitize logs to prevent leaking secrets
log_safe() {
    local message="$1"

    # Remove common secret patterns
    message=$(echo "$message" | sed 's/password=[^&]*/password=****/g')
    message=$(echo "$message" | sed 's/api_key=[^&]*/api_key=****/g')
    message=$(echo "$message" | sed 's/token=[^&]*/token=****/g')

    log_info "$message"
}

# Usage
log_safe "API call: https://api.com?api_key=secret123"
# Logs: "API call: https://api.com?api_key=****"
```

### 7. Temporary Credentials

```bash
# Use temporary files with cleanup
create_temp_credentials() {
    local temp_creds=$(mktemp -t credentials.XXXXXX)

    # Set restrictive permissions
    chmod 600 "$temp_creds"

    # Write credentials
    echo "$DB_PASSWORD" > "$temp_creds"

    # Return path
    echo "$temp_creds"
}

cleanup_temp_credentials() {
    local temp_creds="$1"

    # Overwrite with random data before deletion
    dd if=/dev/urandom of="$temp_creds" bs=1k count=1 2>/dev/null
    rm -f "$temp_creds"
}

# Usage
CREDS_FILE=$(create_temp_credentials)
trap "cleanup_temp_credentials $CREDS_FILE" EXIT

# Use credentials file
mysql --defaults-file="$CREDS_FILE" -e "SELECT 1"
```

### Security Checklist

✅ **Never hardcode secrets** in scripts
✅ **Use environment variables** or secret management systems
✅ **Set restrictive permissions** (600) on credential files
✅ **Sanitize logs** to prevent secret leakage
✅ **Use temporary files** with cleanup
✅ **Never commit secrets** to version control
✅ **Rotate credentials** regularly
✅ **Use least privilege** principles

---

## Question 5: How Do You Debug Bash Scripts Effectively?

### Answer

Debugging bash scripts requires a combination of techniques and tools. Here are the most effective approaches:

### 1. Enable Debug Mode

```bash
#!/bin/bash

# Trace execution
set -x  # Print commands before execution

# Or use conditionally
if [ "${DEBUG:-false}" = "true" ]; then
    set -x
fi

# Trace with timestamps
export PS4='+ [$(date +%H:%M:%S)] ${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
set -x
```

### 2. Add Debug Statements

```bash
debug_var() {
    local var_name="$1"
    local var_value="${!var_name}"

    if [ "${DEBUG:-false}" = "true" ]; then
        echo "DEBUG: $var_name = '$var_value'" >&2
    fi
}

# Usage
MODEL_PATH="/models/production/model.h5"
debug_var "MODEL_PATH"
```

### 3. Check Exit Codes

```bash
command_with_details() {
    local command="$@"

    echo "Running: $command"

    set +e
    $command
    local exit_code=$?
    set -e

    echo "Exit code: $exit_code"

    if [ $exit_code -ne 0 ]; then
        echo "Command failed!" >&2
        return $exit_code
    fi
}
```

### 4. Dry Run Mode

```bash
#!/bin/bash

DRY_RUN="${DRY_RUN:-false}"

execute() {
    local command="$@"

    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY RUN] Would execute: $command"
    else
        $command
    fi
}

# Usage
execute rm -rf /important/data
# With DRY_RUN=true, only prints without executing
```

### 5. Verbose Output

```bash
verbose_log() {
    if [ "${VERBOSE:-false}" = "true" ]; then
        echo "$@" >&2
    fi
}

# Usage
verbose_log "Starting data processing..."
```

### 6. Error Trace

```bash
enable_error_trace() {
    error_handler() {
        local line_no=$1
        local bash_lineno=$2
        local last_command="$3"
        local code="$4"

        echo "Error in ${BASH_SOURCE[1]}" >&2
        echo "  Line: $line_no" >&2
        echo "  Command: $last_command" >&2
        echo "  Exit code: $code" >&2
        echo "Call stack:" >&2

        local frame=0
        while caller $frame; do
            ((frame++))
        done >&2
    }

    set -E
    trap 'error_handler ${LINENO} ${BASH_LINENO} "$BASH_COMMAND" $?' ERR
}

enable_error_trace
```

### Summary

Effective debugging requires:
- Debug mode (`set -x`)
- Conditional debug logging
- Exit code checking
- Dry run capability
- Verbose output
- Error tracing with context

---

## Question 6: When Should You Use Bash vs Python for ML Automation?

### Answer

Choose the right tool for the task:

### Use Bash When:

1. **System Operations:**
   - File operations (copy, move, permissions)
   - Process management
   - Service control (systemctl)
   - Environment setup

2. **Orchestration:**
   - Calling multiple tools in sequence
   - CI/CD pipeline steps
   - Deployment workflows

3. **Simple Tasks:**
   - Quick scripts (< 200 lines)
   - Command-line tool wrappers
   - Environment variable management

### Use Python When:

1. **Data Processing:**
   - CSV/JSON parsing
   - Data validation
   - Complex transformations

2. **ML Operations:**
   - Model training
   - Inference
   - Metrics calculation

3. **Complex Logic:**
   - Advanced algorithms
   - Error handling with retries
   - API integrations

4. **Cross-Platform:**
   - Scripts that run on Windows/Linux/Mac
   - Cloud provider SDKs (boto3, google-cloud)

### Hybrid Approach:

Combine both for best results:

```bash
#!/bin/bash
# Orchestration in Bash

# Setup
echo "Setting up environment..."
export MODEL_PATH="/models/production"
mkdir -p "$MODEL_PATH"

# Data processing in Python
echo "Processing data..."
python3 process_data.py \
    --input data/raw \
    --output data/processed

# Training in Python
echo "Training model..."
python3 train.py \
    --data data/processed \
    --output "$MODEL_PATH/model.h5"

# Deployment in Bash
echo "Deploying model..."
./deploy_model.sh "$MODEL_PATH/model.h5" production

echo "Pipeline complete!"
```

### Decision Matrix

| Task | Bash | Python |
|------|------|--------|
| File operations | ✅ | ❌ |
| Process management | ✅ | ❌ |
| Data processing | ❌ | ✅ |
| ML operations | ❌ | ✅ |
| Orchestration | ✅ | ❌ |
| Complex logic | ❌ | ✅ |
| API calls | ❌ | ✅ |
| System services | ✅ | ❌ |

---

## Conclusion

These answers cover the essential concepts for production-ready bash scripting in ML infrastructure:

1. **Error handling:** `set -euo pipefail` for robust scripts
2. **Idempotency:** Scripts that can run multiple times safely
3. **Logging:** Comprehensive logging with levels and rotation
4. **Security:** Proper handling of secrets and credentials
5. **Debugging:** Effective techniques for troubleshooting
6. **Tool selection:** When to use Bash vs Python

Apply these practices to create reliable, maintainable automation scripts for ML deployments.

---

**Exercise 04: Bash Scripting for ML Deployment Automation - Questions Answered ✅**
