#!/bin/bash
#
# process_data.sh - Data Pipeline Automation for ML
#
# Usage: ./process_data.sh <command>
#
# Commands: run, validate, cleanup, stats
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_DIR="${SCRIPT_DIR}/../logs"
readonly LOG_FILE="${LOG_DIR}/data_pipeline.log"
readonly DATA_DIR="${SCRIPT_DIR}/../data"
readonly RAW_DIR="${DATA_DIR}/raw"
readonly PROCESSED_DIR="${DATA_DIR}/processed"
readonly TEMP_DIR="${DATA_DIR}/temp"

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

# Logging
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

error_exit() {
    log_error "$1"
    cleanup_temp
    exit 1
}

# Setup
setup() {
    mkdir -p "$LOG_DIR" "$RAW_DIR" "$PROCESSED_DIR" "$TEMP_DIR"
}

# Cleanup temporary files
cleanup_temp() {
    if [ -d "$TEMP_DIR" ]; then
        log_info "Cleaning up temporary files"
        rm -rf "${TEMP_DIR:?}"/*
    fi
}

# Usage
usage() {
    cat << EOF
Usage: $SCRIPT_NAME <command>

Data pipeline automation for ML projects.

Commands:
  run         - Run full data pipeline
  validate    - Validate data quality
  cleanup     - Clean up old processed data
  stats       - Generate data statistics

Pipeline Steps:
  1. Download/collect data
  2. Validate data quality
  3. Preprocess and clean
  4. Merge datasets
  5. Split into train/val/test
  6. Generate statistics

Examples:
  $SCRIPT_NAME run
  $SCRIPT_NAME validate
  $SCRIPT_NAME cleanup
  $SCRIPT_NAME stats

EOF
    exit 0
}

# Step 1: Download/collect data (simulated)
download_data() {
    log_info "Step 1/6: Downloading data"

    # Simulate data download
    local datasets=("dataset1.csv" "dataset2.csv" "dataset3.csv")

    for dataset in "${datasets[@]}"; do
        local file_path="${RAW_DIR}/${dataset}"

        if [ -f "$file_path" ]; then
            log_info "✓ Dataset exists: $dataset"
        else
            log_info "Downloading: $dataset"
            # Simulate download with dummy data
            {
                echo "id,feature1,feature2,feature3,label"
                for i in {1..1000}; do
                    echo "$i,$((RANDOM % 100)),$((RANDOM % 100)),$((RANDOM % 100)),$((RANDOM % 2))"
                done
            } > "$file_path"
            log_success "✓ Downloaded: $dataset"
        fi
    done

    log_success "Step 1/6 complete: Data download"
}

# Step 2: Validate data quality
validate_data() {
    log_info "Step 2/6: Validating data quality"

    local validation_passed=true

    # Check if raw data exists
    local file_count=$(find "$RAW_DIR" -name "*.csv" -type f | wc -l)

    if [ "$file_count" -eq 0 ]; then
        error_exit "No data files found in $RAW_DIR"
    fi

    log_info "Found $file_count data files"

    # Validate each file
    for data_file in "$RAW_DIR"/*.csv; do
        local filename=$(basename "$data_file")

        # Check file size
        local size=$(stat -c%s "$data_file" 2>/dev/null || stat -f%z "$data_file" 2>/dev/null)

        if [ "$size" -lt 100 ]; then
            log_error "✗ $filename: File too small ($size bytes)"
            validation_passed=false
        else
            log_info "✓ $filename: Size OK ($(( size / 1024 ))KB)"
        fi

        # Check for header
        local header=$(head -1 "$data_file")
        if [[ "$header" == *","* ]]; then
            log_info "✓ $filename: Header detected"
        else
            log_error "✗ $filename: No header found"
            validation_passed=false
        fi

        # Count records
        local record_count=$(($(wc -l < "$data_file") - 1))
        log_info "✓ $filename: $record_count records"
    done

    if [ "$validation_passed" = false ]; then
        error_exit "Data validation failed"
    fi

    log_success "Step 2/6 complete: Data validation"
}

# Step 3: Preprocess and clean
preprocess_data() {
    log_info "Step 3/6: Preprocessing data"

    for data_file in "$RAW_DIR"/*.csv; do
        local filename=$(basename "$data_file")
        local output_file="${TEMP_DIR}/cleaned_${filename}"

        log_info "Processing: $filename"

        # Remove duplicates, handle missing values (simulated)
        # In real scenario, use awk, sed, or Python scripts
        cp "$data_file" "$output_file"

        log_success "✓ Processed: $filename"
    done

    log_success "Step 3/6 complete: Preprocessing"
}

# Step 4: Merge datasets
merge_datasets() {
    log_info "Step 4/6: Merging datasets"

    local merged_file="${TEMP_DIR}/merged_data.csv"

    # Get header from first file
    head -1 "${TEMP_DIR}/cleaned_dataset1.csv" > "$merged_file"

    # Append all data (skip headers)
    for cleaned_file in "${TEMP_DIR}"/cleaned_*.csv; do
        tail -n +2 "$cleaned_file" >> "$merged_file"
    done

    local total_records=$(($(wc -l < "$merged_file") - 1))
    log_success "✓ Merged $total_records records"

    log_success "Step 4/6 complete: Merging"
}

# Step 5: Split into train/val/test
split_dataset() {
    log_info "Step 5/6: Splitting dataset (70/15/15)"

    local merged_file="${TEMP_DIR}/merged_data.csv"
    local total_records=$(($(wc -l < "$merged_file") - 1))

    # Calculate split sizes
    local train_size=$(( total_records * 70 / 100 ))
    local val_size=$(( total_records * 15 / 100 ))
    local test_size=$(( total_records - train_size - val_size ))

    log_info "Train: $train_size, Val: $val_size, Test: $test_size"

    # Extract header
    local header=$(head -1 "$merged_file")

    # Create train set
    {
        echo "$header"
        tail -n +2 "$merged_file" | head -n "$train_size"
    } > "${PROCESSED_DIR}/train.csv"

    # Create val set
    {
        echo "$header"
        tail -n +2 "$merged_file" | tail -n +$(( train_size + 1 )) | head -n "$val_size"
    } > "${PROCESSED_DIR}/val.csv"

    # Create test set
    {
        echo "$header"
        tail -n +"$(( train_size + val_size + 2 ))" "$merged_file"
    } > "${PROCESSED_DIR}/test.csv"

    log_success "✓ Created train.csv ($train_size records)"
    log_success "✓ Created val.csv ($val_size records)"
    log_success "✓ Created test.csv ($test_size records)"

    log_success "Step 5/6 complete: Dataset splitting"
}

# Step 6: Generate statistics
generate_statistics() {
    log_info "Step 6/6: Generating statistics"

    local stats_file="${PROCESSED_DIR}/statistics.txt"

    {
        echo "=== Data Pipeline Statistics ==="
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "Dataset Splits:"
        echo "  Train: $(( $(wc -l < "${PROCESSED_DIR}/train.csv") - 1 )) records"
        echo "  Val:   $(( $(wc -l < "${PROCESSED_DIR}/val.csv") - 1 )) records"
        echo "  Test:  $(( $(wc -l < "${PROCESSED_DIR}/test.csv") - 1 )) records"
        echo ""
        echo "File Sizes:"
        ls -lh "${PROCESSED_DIR}"/*.csv | awk '{print "  " $9 ": " $5}'
        echo ""
        echo "Raw Data Files:"
        ls -1 "$RAW_DIR" | sed 's/^/  /'
    } > "$stats_file"

    cat "$stats_file"

    log_success "Step 6/6 complete: Statistics generated"
}

# Run full pipeline
run_pipeline() {
    log_info "=== Starting Data Pipeline ==="

    download_data
    validate_data
    preprocess_data
    merge_datasets
    split_dataset
    generate_statistics

    cleanup_temp

    log_success "=== Data Pipeline Complete ==="

    echo ""
    echo -e "${BLUE}Pipeline Summary:${NC}"
    echo "  Processed data: $PROCESSED_DIR"
    echo "  Statistics:     ${PROCESSED_DIR}/statistics.txt"
    echo "  Log file:       $LOG_FILE"
    echo ""
}

# Cleanup old data
cleanup_old_data() {
    log_info "Cleaning up old processed data"

    local days_to_keep=7

    # Find and remove old files
    local old_files=$(find "$PROCESSED_DIR" -type f -mtime +$days_to_keep)

    if [ -z "$old_files" ]; then
        log_info "No old files to clean up"
        return 0
    fi

    echo "$old_files" | while read -r file; do
        log_info "Removing: $(basename "$file")"
        rm -f "$file"
    done

    log_success "Cleanup complete"
}

# Display statistics
show_statistics() {
    local stats_file="${PROCESSED_DIR}/statistics.txt"

    if [ ! -f "$stats_file" ]; then
        error_exit "Statistics file not found. Run pipeline first: $SCRIPT_NAME run"
    fi

    cat "$stats_file"
}

# Main function
main() {
    setup

    if [ $# -eq 0 ]; then
        usage
    fi

    local command="$1"

    case "$command" in
        run)
            run_pipeline
            ;;
        validate)
            validate_data
            ;;
        cleanup)
            cleanup_old_data
            ;;
        stats)
            show_statistics
            ;;
        -h|--help)
            usage
            ;;
        *)
            error_exit "Unknown command: $command"
            ;;
    esac
}

main "$@"
