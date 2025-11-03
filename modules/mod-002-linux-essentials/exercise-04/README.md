# Exercise 04: Bash Scripting for ML Deployment Automation - Solution

## Overview

This solution demonstrates production-ready bash scripting for ML deployment automation. It includes scripts for model deployment, data pipeline automation, system monitoring, and backup operations - essential tools for reliable ML infrastructure.

## Learning Objectives Covered

- ✅ Write structured bash scripts with proper syntax and best practices
- ✅ Use variables, functions, and control structures effectively
- ✅ Handle command-line arguments and user input
- ✅ Implement error handling and logging
- ✅ Create deployment automation scripts for ML models
- ✅ Build data processing pipelines with bash
- ✅ Develop monitoring and alerting scripts
- ✅ Write backup and restore scripts for ML infrastructure

## Solution Structure

```
exercise-04/
├── README.md                          # This file - solution overview
├── IMPLEMENTATION_GUIDE.md            # Step-by-step implementation guide
├── scripts/                           # Production-ready automation scripts
│   ├── deploy_model.sh                # Model deployment automation
│   ├── process_data.sh                # Data pipeline automation
│   ├── monitor_system.sh              # System monitoring and alerting
│   ├── backup_ml_project.sh           # Backup and restore
│   ├── script_template.sh             # Reusable script template
│   └── validate_exercise.sh           # Exercise validation
├── examples/                          # Example configurations and data
│   ├── model_deployment/              # Deployment examples
│   ├── data_pipeline/                 # Pipeline examples
│   └── monitoring/                    # Monitoring examples
└── docs/
    └── ANSWERS.md                     # Reflection question answers
```

## Key Features

### 1. Model Deployment Script (deploy_model.sh)

Automated ML model deployment with validation, backup, and rollback:

```bash
./scripts/deploy_model.sh model.h5 staging
./scripts/deploy_model.sh model.pth production
./scripts/deploy_model.sh rollback
```

**Features:**
- Model validation (format, size checks)
- Automatic backup before deployment
- Staging environment testing
- Production promotion
- Rollback capability
- Deployment notifications
- Comprehensive logging

### 2. Data Pipeline Script (process_data.sh)

End-to-end data processing automation:

```bash
./scripts/process_data.sh run          # Full pipeline
./scripts/process_data.sh validate     # Validate only
./scripts/process_data.sh cleanup      # Clean old data
```

**Features:**
- Data download simulation
- Quality validation
- Preprocessing and cleaning
- Dataset merging
- Train/val/test splitting (70/15/15)
- Statistics generation
- Old data cleanup

### 3. System Monitoring Script (monitor_system.sh)

Real-time monitoring with alerting:

```bash
./scripts/monitor_system.sh monitor      # Continuous monitoring
./scripts/monitor_system.sh check        # One-time check
./scripts/monitor_system.sh report       # Generate report
```

**Features:**
- CPU, memory, disk monitoring
- GPU usage tracking (nvidia-smi)
- Training process detection
- Threshold-based alerting
- Metrics logging to CSV
- Dashboard display
- Historical reports

### 4. Backup Script (backup_ml_project.sh)

Complete backup and restore system:

```bash
./scripts/backup_ml_project.sh backup my-project
./scripts/backup_ml_project.sh list
./scripts/backup_ml_project.sh restore backup_file.tar.gz
./scripts/backup_ml_project.sh verify backup_file.tar.gz
```

**Features:**
- Compressed backups (tar.gz)
- Automatic metadata generation
- Retention policy (keep last 5)
- Integrity verification
- Safe restore with confirmation
- Excludes large files (.git, cache, raw data)

## Quick Start

### 1. Set Up Workspace

```bash
cd /path/to/exercise-04/scripts
chmod +x *.sh
```

### 2. Test Model Deployment

```bash
# Create test model
dd if=/dev/zero of=test_model.h5 bs=1M count=10 2>/dev/null

# Deploy to staging
./deploy_model.sh test_model.h5 staging

# Check logs
cat logs/deployment.log

# Promote to production
./deploy_model.sh test_model.h5 production

# Test rollback
./deploy_model.sh rollback
```

### 3. Run Data Pipeline

```bash
# Run full pipeline
./process_data.sh run

# Check generated files
ls -lh data/processed/

# View statistics
cat data/processed/statistics.txt
```

### 4. Monitor System

```bash
# One-time check
./monitor_system.sh check

# Start continuous monitoring (5s interval)
./monitor_system.sh monitor

# Generate report
./monitor_system.sh report
```

### 5. Create Backups

```bash
# Create test project
mkdir -p ~/ml-projects/test-project
echo "Test README" > ~/ml-projects/test-project/README.md

# Backup project
./backup_ml_project.sh backup test-project

# List backups
./backup_ml_project.sh list

# Verify backup
./backup_ml_project.sh verify test-project_*.tar.gz
```

## Script Best Practices Implemented

### 1. Robust Error Handling

```bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

error_exit() {
    log "ERROR: $1"
    cleanup
    exit 1
}

trap 'error_handler ${LINENO}' ERR
trap cleanup EXIT
```

### 2. Comprehensive Logging

```bash
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_error() { log "ERROR" "$@" >&2; }
log_debug() { [ "$DEBUG" = true ] && log "DEBUG" "$@"; }
```

### 3. Argument Parsing

```bash
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done
```

### 4. Function Organization

```bash
# Functions grouped by purpose
# - Helper functions
# - Validation functions
# - Core logic functions
# - Main workflow function
# - Usage and error handling
```

### 5. Constants and Configuration

```bash
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="/var/log/${SCRIPT_NAME%.sh}.log"
```

## Bash Scripting Patterns

### Variables and Arrays

```bash
# String variables
MODEL_NAME="resnet50"
VERSION="1.2.3"

# Arrays
MODELS=("resnet50" "vgg16" "inception_v3")
echo "${MODELS[@]}"      # All elements
echo "${MODELS[0]}"      # First element
echo "${#MODELS[@]}"     # Array length

# Associative arrays
declare -A CONFIG
CONFIG["key"]="value"
```

### Control Structures

```bash
# If-else
if [ -f "$file" ]; then
    echo "File exists"
elif [ -d "$file" ]; then
    echo "Directory exists"
else
    echo "Not found"
fi

# Case statement
case $mode in
    staging)
        deploy_staging
        ;;
    production)
        deploy_production
        ;;
    *)
        error "Unknown mode"
        ;;
esac

# For loop
for model in "${MODELS[@]}"; do
    process_model "$model"
done

# While loop
while [ $count -lt $max ]; do
    process
    ((count++))
done
```

### Functions

```bash
# Function definition
validate_model() {
    local model_path=$1
    local size=$2

    # Validation logic

    return 0  # Success
}

# Function call
validate_model "$MODEL_PATH" "$SIZE" || {
    error "Validation failed"
}
```

## Real-World Use Cases

### Use Case 1: Continuous Deployment Pipeline

```bash
#!/bin/bash
# CI/CD deployment script

# 1. Run tests
./run_tests.sh || exit 1

# 2. Build model
./build_model.sh

# 3. Deploy to staging
./deploy_model.sh model.h5 staging

# 4. Run integration tests
./integration_tests.sh || exit 1

# 5. Promote to production
./deploy_model.sh model.h5 production

# 6. Monitor deployment
./monitor_system.sh check
```

### Use Case 2: Scheduled Data Processing

```bash
#!/bin/bash
# Cron job: 0 2 * * * /path/to/daily_pipeline.sh

# Process yesterday's data
./process_data.sh run

# Backup results
./backup_ml_project.sh backup data-processing

# Clean up old data
./process_data.sh cleanup

# Generate report
./monitor_system.sh report
```

### Use Case 3: Production Rollback

```bash
#!/bin/bash
# Emergency rollback procedure

# 1. Alert team
echo "Rolling back production model" | mail -s "ALERT" team@company.com

# 2. Rollback deployment
./deploy_model.sh rollback

# 3. Verify service
curl http://api/health || alert "Service down"

# 4. Log incident
./log_incident.sh "Production rollback $(date)"
```

## Testing the Solution

### 1. Validate All Scripts

```bash
cd scripts
./validate_exercise.sh
```

### 2. Test Deployment Workflow

```bash
# Create test model
touch model.h5

# Deploy through full workflow
./deploy_model.sh model.h5 staging
./deploy_model.sh model.h5 production
./deploy_model.sh rollback
```

### 3. Test Data Pipeline

```bash
# Run pipeline
./process_data.sh run

# Verify outputs
ls data/processed/
cat data/processed/statistics.txt
```

### 4. Test Monitoring

```bash
# Single check
./monitor_system.sh check

# Monitor for 30 seconds
timeout 30 ./monitor_system.sh monitor 5
```

### 5. Test Backup/Restore

```bash
# Create test project
mkdir -p ~/ml-projects/test
echo "test" > ~/ml-projects/test/file.txt

# Backup
./backup_ml_project.sh backup test

# Delete original
rm -rf ~/ml-projects/test

# Restore
BACKUP=$(ls ~/ml-backups/test_*.tar.gz | head -1)
./backup_ml_project.sh restore $(basename "$BACKUP")

# Verify restoration
cat ~/ml-projects/test/file.txt
```

## Common Issues and Solutions

### Issue 1: Permission Denied

**Symptom**: `bash: ./script.sh: Permission denied`

**Solution**:
```bash
chmod +x script.sh
```

### Issue 2: Command Not Found in Script

**Symptom**: Script fails with command not found

**Solution**:
```bash
# Check if command exists before using
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not available"
    exit 1
fi
```

### Issue 3: Variable Not Set

**Symptom**: `bash: variable: unbound variable`

**Solution**:
```bash
# Use default value
VAR=${VAR:-default_value}

# Or check explicitly
if [ -z "$VAR" ]; then
    echo "VAR is not set"
    exit 1
fi
```

### Issue 4: Array Issues

**Symptom**: Array not working as expected

**Solution**:
```bash
# Always quote array expansions
for item in "${ARRAY[@]}"; do
    echo "$item"
done

# Not: for item in ${ARRAY[@]}
```

### Issue 5: Comparison Errors

**Symptom**: Integer expression expected

**Solution**:
```bash
# Numeric comparison
if [ "$num" -gt 10 ]; then
    echo "Greater than 10"
fi

# String comparison
if [ "$str" = "value" ]; then
    echo "Matches"
fi
```

## Integration with Previous Exercises

- **Exercise 01**: Uses file navigation and operations
- **Exercise 02**: Applies proper file permissions to scripts and data
- **Exercise 03**: Integrates process management for monitoring
- **Future**: Foundation for system automation (Exercise 08)

## Skills Acquired

- ✅ Bash script structure and organization
- ✅ Variables, arrays, and data structures
- ✅ Control flow (if/case/for/while)
- ✅ Functions and modular code
- ✅ Error handling and logging
- ✅ Command-line argument parsing
- ✅ File operations and I/O
- ✅ Process management
- ✅ System monitoring
- ✅ Backup and restore automation

## Script Templates

### Basic Script Template

```bash
#!/bin/bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

main() {
    log "Script started"
    # Your code here
    log "Script completed"
}

main "$@"
```

### Production Script Template

See `scripts/script_template.sh` for a comprehensive template with:
- Error handling
- Logging system
- Argument parsing
- Cleanup functions
- Usage documentation

## Performance Considerations

### Script Efficiency

1. **Avoid subprocess spawning**:
   ```bash
   # Slow
   result=$(cat file | grep pattern | wc -l)

   # Fast
   result=$(grep -c pattern file)
   ```

2. **Use built-ins over external commands**:
   ```bash
   # Slow
   basename=$(basename "$path")

   # Fast
   basename="${path##*/}"
   ```

3. **Batch operations**:
   ```bash
   # Slow
   for file in *.txt; do
       process "$file"
   done

   # Fast (if supported)
   process *.txt
   ```

## Time to Complete

- **Setup and understanding**: 20 minutes
- **Model deployment script**: 40 minutes
- **Data pipeline script**: 30 minutes
- **Monitoring script**: 30 minutes
- **Backup script**: 30 minutes
- **Testing and validation**: 20 minutes
- **Total**: 150-170 minutes

## Next Steps

- Complete Exercise 05: Package Management
- Complete Exercise 06: Log Analysis
- Learn about advanced bash features (coprocesses, etc.)
- Explore shell alternatives (fish, zsh)

## Resources

- [Bash Guide](https://mywiki.wooledge.org/BashGuide)
- [ShellCheck](https://www.shellcheck.net/) - Shell script linter
- [Advanced Bash-Scripting Guide](https://tldp.org/LDP/abs/html/)
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

## Conclusion

This solution provides production-ready bash scripts for ML deployment automation. The scripts demonstrate industry best practices for error handling, logging, and maintainability - essential skills for reliable ML infrastructure operations.

**Key Achievement**: Complete automation suite for ML deployment, data processing, monitoring, and backup operations with production-grade error handling and logging.

---

**Exercise 04: Bash Scripting for ML Deployment Automation - ✅ READY FOR IMPLEMENTATION**
