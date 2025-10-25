# Exercise 01: Bash Scripting

## Overview

Master Bash scripting for infrastructure automation. Learn to write maintainable, robust scripts that automate repetitive tasks in AI/ML infrastructure.

## Learning Objectives

- ✅ Understand Bash syntax and scripting fundamentals
- ✅ Use variables, arrays, and data structures
- ✅ Implement control structures (if/else, loops)
- ✅ Write functions for code reusability
- ✅ Handle errors and exit codes properly
- ✅ Process command-line arguments
- ✅ Work with file operations and I/O
- ✅ Debug and test shell scripts

## Topics Covered

### 1. Script Basics

#### Shebang and Script Structure

```bash
#!/bin/bash
#
# Script: backup_models.sh
# Description: Backup ML models to remote storage
# Author: Your Name
# Date: 2024-01-24
#

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Global variables
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Main function
main() {
    echo "Starting backup process..."
    # Script logic here
}

# Run main function
main "$@"
```

#### Making Scripts Executable

```bash
# Make script executable
chmod +x script.sh

# Run script
./script.sh

# Or with bash
bash script.sh
```

### 2. Variables

#### Basic Variables

```bash
# String variables
name="John"
greeting="Hello, $name"
echo "$greeting"  # Hello, John

# Integer variables
count=10
((count++))
echo "$count"  # 11

# Command substitution
current_date=$(date +%Y-%m-%d)
user_count=$(who | wc -l)

# Environment variables
echo "$HOME"
echo "$USER"
echo "$PATH"
```

#### Variable Scopes

```bash
#!/bin/bash

# Global variable
global_var="I'm global"

my_function() {
    # Local variable
    local local_var="I'm local"
    echo "$local_var"
    echo "$global_var"
}

my_function
# echo "$local_var"  # Error: undefined
echo "$global_var"  # Works
```

#### Read-only Variables

```bash
readonly API_KEY="secret123"
# API_KEY="new_value"  # Error: readonly variable

# Declare constant with declare
declare -r MAX_RETRIES=3
```

### 3. Arrays

#### Indexed Arrays

```bash
# Create array
models=("bert" "gpt" "t5")

# Access elements
echo "${models[0]}"  # bert
echo "${models[@]}"  # All elements
echo "${#models[@]}" # Length

# Add elements
models+=("roberta")

# Iterate
for model in "${models[@]}"; do
    echo "Processing $model"
done

# Array slicing
echo "${models[@]:1:2}"  # gpt t5
```

#### Associative Arrays (Bash 4+)

```bash
# Declare associative array
declare -A model_versions

# Set values
model_versions["bert"]="1.0.0"
model_versions["gpt"]="3.5"
model_versions["t5"]="2.0"

# Access values
echo "${model_versions["bert"]}"

# Iterate
for model in "${!model_versions[@]}"; do
    echo "$model: ${model_versions[$model]}"
done
```

### 4. Control Structures

#### If Statements

```bash
# Basic if
if [[ -f "model.pkl" ]]; then
    echo "Model file exists"
fi

# If-else
if [[ $count -gt 10 ]]; then
    echo "Count is greater than 10"
else
    echo "Count is 10 or less"
fi

# If-elif-else
if [[ $status == "running" ]]; then
    echo "Process is running"
elif [[ $status == "stopped" ]]; then
    echo "Process is stopped"
else
    echo "Unknown status"
fi

# Multiple conditions
if [[ $age -ge 18 ]] && [[ $age -le 65 ]]; then
    echo "Working age"
fi

if [[ $env == "dev" ]] || [[ $env == "test" ]]; then
    echo "Non-production environment"
fi
```

#### Comparison Operators

```bash
# String comparison
[[ "$str1" == "$str2" ]]  # Equal
[[ "$str1" != "$str2" ]]  # Not equal
[[ -z "$str" ]]           # Empty string
[[ -n "$str" ]]           # Non-empty string

# Numeric comparison
[[ $num1 -eq $num2 ]]  # Equal
[[ $num1 -ne $num2 ]]  # Not equal
[[ $num1 -lt $num2 ]]  # Less than
[[ $num1 -le $num2 ]]  # Less than or equal
[[ $num1 -gt $num2 ]]  # Greater than
[[ $num1 -ge $num2 ]]  # Greater than or equal

# File tests
[[ -e file ]]  # Exists
[[ -f file ]]  # Regular file
[[ -d dir ]]   # Directory
[[ -r file ]]  # Readable
[[ -w file ]]  # Writable
[[ -x file ]]  # Executable
[[ -s file ]]  # Non-empty file
```

#### Case Statements

```bash
case "$environment" in
    dev|development)
        echo "Development environment"
        config_file="dev.conf"
        ;;
    staging)
        echo "Staging environment"
        config_file="staging.conf"
        ;;
    prod|production)
        echo "Production environment"
        config_file="prod.conf"
        ;;
    *)
        echo "Unknown environment: $environment"
        exit 1
        ;;
esac
```

#### Loops

```bash
# For loop with list
for model in bert gpt t5; do
    echo "Training $model"
done

# For loop with range
for i in {1..5}; do
    echo "Iteration $i"
done

# C-style for loop
for ((i=0; i<10; i++)); do
    echo "Count: $i"
done

# While loop
count=0
while [[ $count -lt 5 ]]; do
    echo "Count: $count"
    ((count++))
done

# Until loop
count=0
until [[ $count -ge 5 ]]; do
    echo "Count: $count"
    ((count++))
done

# Read file line by line
while IFS= read -r line; do
    echo "Line: $line"
done < file.txt

# Loop control
for i in {1..10}; do
    if [[ $i -eq 5 ]]; then
        continue  # Skip iteration
    fi
    if [[ $i -eq 8 ]]; then
        break  # Exit loop
    fi
    echo "$i"
done
```

### 5. Functions

#### Basic Functions

```bash
# Define function
greet() {
    echo "Hello, World!"
}

# Call function
greet

# Function with parameters
greet_user() {
    local name="$1"
    echo "Hello, $name!"
}

greet_user "Alice"

# Function with return value
add() {
    local a=$1
    local b=$2
    echo $((a + b))
}

result=$(add 5 3)
echo "Result: $result"

# Function with return code
check_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        return 0  # Success
    else
        return 1  # Failure
    fi
}

if check_file "data.csv"; then
    echo "File exists"
fi
```

#### Advanced Functions

```bash
# Function with default parameters
deploy_model() {
    local model="${1:-model.pkl}"
    local environment="${2:-dev}"

    echo "Deploying $model to $environment"
}

deploy_model  # Uses defaults
deploy_model "custom.pkl" "prod"

# Function with variable arguments
log() {
    local level="$1"
    shift  # Remove first argument

    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $level: $*"
}

log "INFO" "Application" "started" "successfully"
```

### 6. Command-Line Arguments

#### Positional Parameters

```bash
#!/bin/bash

# Script usage: ./script.sh arg1 arg2 arg3

echo "Script name: $0"
echo "First argument: $1"
echo "Second argument: $2"
echo "All arguments: $@"
echo "Number of arguments: $#"

# Shift arguments
shift  # Remove $1, $2 becomes $1
echo "New first argument: $1"
```

#### Parsing Options

```bash
#!/bin/bash

# Parse options
while getopts "m:e:hv" opt; do
    case $opt in
        m)
            model="$OPTARG"
            ;;
        e)
            environment="$OPTARG"
            ;;
        h)
            echo "Usage: $0 -m MODEL -e ENVIRONMENT"
            exit 0
            ;;
        v)
            verbose=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

# Usage: ./script.sh -m model.pkl -e prod -v
```

#### Long Options

```bash
#!/bin/bash

# Parse long options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$2"
            shift 2
            ;;
        --environment)
            environment="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --model MODEL --environment ENV"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Usage: ./script.sh --model model.pkl --environment prod
```

### 7. Input/Output

#### Reading Input

```bash
# Read user input
read -p "Enter your name: " name
echo "Hello, $name!"

# Read with timeout
if read -t 5 -p "Enter value (5s timeout): " value; then
    echo "You entered: $value"
else
    echo "Timeout!"
fi

# Read password (hidden)
read -sp "Enter password: " password
echo  # New line

# Read from file
while IFS= read -r line; do
    process "$line"
done < input.txt
```

#### Output Redirection

```bash
# Redirect stdout
echo "Success" > output.txt

# Append to file
echo "More text" >> output.txt

# Redirect stderr
command 2> error.log

# Redirect both
command > output.log 2>&1
command &> output.log  # Shorthand

# Redirect to null (discard)
command > /dev/null 2>&1

# Tee (write to file and stdout)
command | tee output.log
command | tee -a output.log  # Append
```

#### Here Documents

```bash
# Multi-line string
cat <<EOF
This is a
multi-line
document
EOF

# Write to file
cat > config.txt <<EOF
host=localhost
port=5432
database=mldb
EOF

# With variable expansion
cat <<EOF
Current user: $USER
Home directory: $HOME
EOF

# Without variable expansion
cat <<'EOF'
This will not expand: $USER
EOF
```

### 8. Error Handling

#### Exit Codes

```bash
# Exit with code
exit 0  # Success
exit 1  # Error

# Check exit code
command
if [[ $? -eq 0 ]]; then
    echo "Success"
else
    echo "Failed"
fi

# Or inline
if command; then
    echo "Success"
else
    echo "Failed"
fi
```

#### Set Options

```bash
#!/bin/bash

# Exit on error
set -e
# Or
set -o errexit

# Exit on undefined variable
set -u
# Or
set -o nounset

# Fail on pipe errors
set -o pipefail

# Combined (recommended)
set -euo pipefail
```

#### Trap for Cleanup

```bash
#!/bin/bash

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    rm -f /tmp/tempfile
}

# Set trap
trap cleanup EXIT

# Script continues
echo "Running script..."
# Even if script fails, cleanup runs
```

### 9. Text Processing

#### Grep

```bash
# Search for pattern
grep "error" logfile.txt

# Case-insensitive
grep -i "error" logfile.txt

# Recursive search
grep -r "TODO" src/

# Count matches
grep -c "error" logfile.txt

# Show line numbers
grep -n "error" logfile.txt

# Invert match
grep -v "debug" logfile.txt

# Extended regex
grep -E "error|warning" logfile.txt
```

#### Sed

```bash
# Replace text
sed 's/old/new/' file.txt
sed 's/old/new/g' file.txt  # All occurrences

# In-place editing
sed -i 's/old/new/g' file.txt

# Delete lines
sed '/pattern/d' file.txt
sed '1,5d' file.txt  # Delete lines 1-5

# Print specific lines
sed -n '10,20p' file.txt
```

#### Awk

```bash
# Print columns
awk '{print $1}' file.txt
awk '{print $1, $3}' file.txt

# With delimiter
awk -F',' '{print $1}' file.csv

# Conditional printing
awk '$3 > 100 {print $1, $3}' file.txt

# Sum column
awk '{sum += $3} END {print sum}' file.txt
```

---

## Project: Infrastructure Automation Toolkit

Build a comprehensive toolkit for ML infrastructure automation.

### Requirements

**Scripts to Create:**
1. Model deployment script
2. Data backup and restore
3. Log analysis and alerting
4. Health check and monitoring
5. Environment setup automation

**Technical Requirements:**
- Robust error handling
- Comprehensive logging
- Command-line argument parsing
- Configuration file support
- Dry-run mode
- Help documentation

### Implementation

See `solutions/` directory for complete implementations.

### Example Scripts

#### 1. Model Deployment Script

```bash
#!/bin/bash
set -euo pipefail

# deploy_model.sh - Deploy ML model to production

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/model_deployment.log"

# Configuration
MODEL_DIR="/models"
DEPLOY_DIR="/opt/ml/models"
BACKUP_DIR="/backup/models"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

validate_model() {
    local model_path="$1"

    if [[ ! -f "$model_path" ]]; then
        log "ERROR: Model file not found: $model_path"
        return 1
    fi

    log "INFO: Validating model: $model_path"
    # Add validation logic here
    return 0
}

backup_current_model() {
    local model_name="$1"

    if [[ -f "$DEPLOY_DIR/$model_name" ]]; then
        local backup_name="${model_name}.$(date +%Y%m%d_%H%M%S)"
        log "INFO: Backing up current model to $backup_name"
        cp "$DEPLOY_DIR/$model_name" "$BACKUP_DIR/$backup_name"
    fi
}

deploy_model() {
    local model_path="$1"
    local model_name=$(basename "$model_path")

    log "INFO: Starting deployment of $model_name"

    # Validate
    if ! validate_model "$model_path"; then
        log "ERROR: Model validation failed"
        return 1
    fi

    # Backup current
    backup_current_model "$model_name"

    # Deploy
    log "INFO: Copying model to $DEPLOY_DIR"
    sudo cp "$model_path" "$DEPLOY_DIR/"
    sudo chown ml-user:ml-group "$DEPLOY_DIR/$model_name"
    sudo chmod 644 "$DEPLOY_DIR/$model_name"

    log "INFO: Deployment successful"
    return 0
}

# Main
main() {
    if [[ $# -eq 0 ]]; then
        echo "Usage: $0 MODEL_PATH"
        exit 1
    fi

    local model_path="$1"

    if deploy_model "$model_path"; then
        log "SUCCESS: Model deployed successfully"
        exit 0
    else
        log "FAILURE: Model deployment failed"
        exit 1
    fi
}

main "$@"
```

---

## Practice Problems

### Problem 1: File Backup Script

```bash
#!/bin/bash
# Create a script that backs up files to a specified directory
# Features: timestamp, compression, retention policy

backup_files() {
    # Your implementation
    :
}
```

### Problem 2: Log Parser

```bash
#!/bin/bash
# Parse log files and extract error/warning statistics
# Output: error count, warning count, top errors

parse_logs() {
    # Your implementation
    :
}
```

### Problem 3: Service Health Check

```bash
#!/bin/bash
# Check if services are running and restart if necessary
# Monitor: API server, database, cache

check_service() {
    # Your implementation
    :
}
```

---

## Best Practices

### 1. Use ShellCheck

```bash
# Install
sudo apt install shellcheck

# Check script
shellcheck script.sh
```

### 2. Proper Quoting

```bash
# Always quote variables
file="my file.txt"
cat "$file"  # Correct
cat $file    # Wrong - breaks with spaces
```

### 3. Use $(command) over Backticks

```bash
# Good
date=$(date +%Y-%m-%d)

# Avoid
date=`date +%Y-%m-%d`
```

### 4. Meaningful Variable Names

```bash
# Good
max_retries=3
api_endpoint="https://api.example.com"

# Bad
x=3
url="https://api.example.com"
```

---

## Validation

Test your scripts:

```bash
# Run shellcheck
shellcheck solutions/*.sh

# Execute scripts
bash solutions/deploy_model.sh --help
bash solutions/backup_data.sh --dry-run
```

---

## Resources

- [Bash Guide](https://mywiki.wooledge.org/BashGuide)
- [ShellCheck](https://www.shellcheck.net/)
- [Bash Pitfalls](https://mywiki.wooledge.org/BashPitfalls)
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

---

## Next Steps

1. **Exercise 02: Filesystem & Processes** - Deep dive into Linux internals
2. Practice writing scripts daily
3. Automate your repetitive tasks
4. Contribute to shell script projects

---

**Automate everything with Bash! 🚀**
