#!/bin/bash
#
# analyze_disk_space.sh - Disk space analysis for ML datasets
#
# Description:
#   Comprehensive disk space analysis and management for ML data storage
#   including usage reports, old file detection, and cleanup recommendations.
#
# Usage:
#   ./analyze_disk_space.sh [OPTIONS] [PATH]
#
# Options:
#   -t, --threshold PCT    Alert threshold percentage (default: 80)
#   -o, --old-days DAYS    Consider files older than DAYS for cleanup (default: 90)
#   -s, --min-size SIZE    Minimum file size to report (e.g., 100M, 1G)
#   -r, --report FILE      Generate report to file
#   -c, --cleanup          Interactive cleanup mode
#   -d, --depth N          Directory depth to analyze (default: 3)
#   -f, --format FORMAT    Output format (text, json, html)
#   -v, --verbose          Verbose output
#   -h, --help             Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
ANALYSIS_PATH="/data"
THRESHOLD=80
OLD_FILE_DAYS=90
MIN_SIZE="100M"
REPORT_FILE=""
CLEANUP_MODE=false
MAX_DEPTH=3
OUTPUT_FORMAT="text"
VERBOSE=false

# ===========================
# Colors
# ===========================

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly RESET='\033[0m'
readonly BOLD='\033[1m'

# ===========================
# Utility Functions
# ===========================

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "[DEBUG] $*"
    fi
}

human_readable_size() {
    local bytes=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit_index=0

    while [[ $bytes -gt 1024 ]] && [[ $unit_index -lt 4 ]]; do
        bytes=$((bytes / 1024))
        ((unit_index++))
    done

    echo "${bytes}${units[$unit_index]}"
}

size_to_bytes() {
    local size=$1

    # Extract number and unit
    local number=$(echo "$size" | sed 's/[^0-9.]//g')
    local unit=$(echo "$size" | sed 's/[0-9.]//g' | tr '[:lower:]' '[:upper:]')

    case "$unit" in
        K|KB)
            echo "$((${number%.*} * 1024))"
            ;;
        M|MB)
            echo "$((${number%.*} * 1024 * 1024))"
            ;;
        G|GB)
            echo "$((${number%.*} * 1024 * 1024 * 1024))"
            ;;
        T|TB)
            echo "$((${number%.*} * 1024 * 1024 * 1024 * 1024))"
            ;;
        *)
            echo "${number%.*}"
            ;;
    esac
}

get_color_for_usage() {
    local usage=$1
    local warning_threshold=$((THRESHOLD - 10))

    if [[ $usage -ge $THRESHOLD ]]; then
        echo "$RED"
    elif [[ $usage -ge $warning_threshold ]]; then
        echo "$YELLOW"
    else
        echo "$GREEN"
    fi
}

# ===========================
# Disk Usage Analysis
# ===========================

analyze_filesystem_usage() {
    echo -e "${BOLD}${CYAN}Filesystem Usage Analysis${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    # Overall filesystem usage
    df -h "$ANALYSIS_PATH" | awk 'NR==1 || NR==2' | while read -r line; do
        if echo "$line" | grep -q "Filesystem"; then
            echo -e "${BOLD}$line${RESET}"
        else
            local usage=$(echo "$line" | awk '{print $5}' | sed 's/%//')
            local color=$(get_color_for_usage "$usage")
            echo -e "$color$line$RESET"

            if [[ $usage -ge $THRESHOLD ]]; then
                echo -e "${RED}⚠ WARNING: Disk usage exceeds threshold (${THRESHOLD}%)${RESET}"
            fi
        fi
    done

    echo ""
}

analyze_directory_sizes() {
    echo -e "${BOLD}${CYAN}Top Directories by Size${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    log_verbose "Analyzing directory sizes in $ANALYSIS_PATH (depth: $MAX_DEPTH)..."

    # Get directory sizes
    du -h --max-depth="$MAX_DEPTH" "$ANALYSIS_PATH" 2>/dev/null | \
        sort -rh | \
        head -20 | \
        awk '{printf "%-10s  %s\n", $1, $2}'

    echo ""
}

analyze_file_types() {
    echo -e "${BOLD}${CYAN}Storage by File Type${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    log_verbose "Analyzing file types in $ANALYSIS_PATH..."

    # Common ML file extensions
    local extensions=("*.csv" "*.json" "*.pkl" "*.h5" "*.pt" "*.pth" "*.npz" "*.parquet" "*.tfrecord" "*.tar" "*.zip" "*.gz")

    declare -A file_sizes
    declare -A file_counts

    for ext in "${extensions[@]}"; do
        local ext_name="${ext#*.}"
        local size=$(find "$ANALYSIS_PATH" -type f -name "$ext" -exec du -cb {} + 2>/dev/null | tail -1 | awk '{print $1}')
        local count=$(find "$ANALYSIS_PATH" -type f -name "$ext" 2>/dev/null | wc -l)

        if [[ $count -gt 0 ]]; then
            file_sizes["$ext_name"]=$size
            file_counts["$ext_name"]=$count
        fi
    done

    # Sort and display
    printf "%-15s %15s %15s\n" "File Type" "Count" "Total Size"
    echo "-----------------------------------------------"

    for ext in "${!file_sizes[@]}"; do
        local size_hr=$(human_readable_size "${file_sizes[$ext]}")
        printf "%-15s %15s %15s\n" "$ext" "${file_counts[$ext]}" "$size_hr"
    done | sort -t ' ' -k3 -hr

    echo ""
}

find_large_files() {
    local min_size_bytes=$(size_to_bytes "$MIN_SIZE")

    echo -e "${BOLD}${CYAN}Large Files (> $MIN_SIZE)${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    log_verbose "Finding files larger than $MIN_SIZE in $ANALYSIS_PATH..."

    find "$ANALYSIS_PATH" -type f -size +"$MIN_SIZE" -exec ls -lh {} \; 2>/dev/null | \
        awk '{printf "%-10s  %-20s  %s\n", $5, $6" "$7" "$8, $9}' | \
        head -20

    local count=$(find "$ANALYSIS_PATH" -type f -size +"$MIN_SIZE" 2>/dev/null | wc -l)
    echo ""
    echo "Total large files found: $count"
    echo ""
}

find_old_files() {
    echo -e "${BOLD}${CYAN}Old Files (> $OLD_FILE_DAYS days)${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    log_verbose "Finding files older than $OLD_FILE_DAYS days in $ANALYSIS_PATH..."

    # Find old files with size
    local old_files=$(find "$ANALYSIS_PATH" -type f -mtime +$OLD_FILE_DAYS -exec ls -lh {} \; 2>/dev/null)

    if [[ -n "$old_files" ]]; then
        echo "$old_files" | \
            awk '{printf "%-10s  %-20s  %s\n", $5, $6" "$7" "$8, $9}' | \
            head -20

        # Calculate total size
        local total_size=$(find "$ANALYSIS_PATH" -type f -mtime +$OLD_FILE_DAYS -exec du -cb {} + 2>/dev/null | tail -1 | awk '{print $1}')
        local total_size_hr=$(human_readable_size "$total_size")
        local count=$(find "$ANALYSIS_PATH" -type f -mtime +$OLD_FILE_DAYS 2>/dev/null | wc -l)

        echo ""
        echo "Total old files: $count"
        echo "Total size: $total_size_hr"

        if [[ $count -gt 0 ]]; then
            echo -e "${YELLOW}💡 Consider removing old files to free up space${RESET}"
        fi
    else
        echo "No old files found"
    fi

    echo ""
}

find_duplicate_files() {
    echo -e "${BOLD}${CYAN}Potential Duplicate Files${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    log_verbose "Searching for duplicate files in $ANALYSIS_PATH..."

    # Find files with same size, then check md5sum for duplicates
    find "$ANALYSIS_PATH" -type f -exec du -b {} \; 2>/dev/null | \
        sort -n | \
        awk 'prev == $1 {print prev_file; print $2} {prev = $1; prev_file = $2}' | \
        head -20

    echo ""
    echo "Note: These are potential duplicates based on file size."
    echo "Run md5sum to confirm actual duplicates."
    echo ""
}

analyze_empty_files() {
    echo -e "${BOLD}${CYAN}Empty Files and Directories${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    # Empty files
    local empty_files=$(find "$ANALYSIS_PATH" -type f -empty 2>/dev/null)
    local empty_file_count=$(echo "$empty_files" | grep -c . || echo 0)

    echo -e "${BOLD}Empty Files:${RESET} $empty_file_count"
    if [[ $empty_file_count -gt 0 ]]; then
        echo "$empty_files" | head -10
        if [[ $empty_file_count -gt 10 ]]; then
            echo "... and $((empty_file_count - 10)) more"
        fi
    fi

    echo ""

    # Empty directories
    local empty_dirs=$(find "$ANALYSIS_PATH" -type d -empty 2>/dev/null)
    local empty_dir_count=$(echo "$empty_dirs" | grep -c . || echo 0)

    echo -e "${BOLD}Empty Directories:${RESET} $empty_dir_count"
    if [[ $empty_dir_count -gt 0 ]]; then
        echo "$empty_dirs" | head -10
        if [[ $empty_dir_count -gt 10 ]]; then
            echo "... and $((empty_dir_count - 10)) more"
        fi
    fi

    echo ""
}

# ===========================
# Cleanup Recommendations
# ===========================

generate_cleanup_recommendations() {
    echo -e "${BOLD}${CYAN}Cleanup Recommendations${RESET}"
    echo -e "${CYAN}========================================${RESET}"
    echo ""

    local recommendations=()

    # Check old files
    local old_file_count=$(find "$ANALYSIS_PATH" -type f -mtime +$OLD_FILE_DAYS 2>/dev/null | wc -l)
    if [[ $old_file_count -gt 0 ]]; then
        local old_size=$(find "$ANALYSIS_PATH" -type f -mtime +$OLD_FILE_DAYS -exec du -cb {} + 2>/dev/null | tail -1 | awk '{print $1}')
        local old_size_hr=$(human_readable_size "$old_size")
        echo "1. Remove old files (>$OLD_FILE_DAYS days): $old_file_count files, $old_size_hr"
        recommendations+=("old_files")
    fi

    # Check empty files
    local empty_count=$(find "$ANALYSIS_PATH" -type f -empty 2>/dev/null | wc -l)
    if [[ $empty_count -gt 0 ]]; then
        echo "2. Remove empty files: $empty_count files"
        recommendations+=("empty_files")
    fi

    # Check empty directories
    local empty_dir_count=$(find "$ANALYSIS_PATH" -type d -empty 2>/dev/null | wc -l)
    if [[ $empty_dir_count -gt 0 ]]; then
        echo "3. Remove empty directories: $empty_dir_count directories"
        recommendations+=("empty_dirs")
    fi

    # Check log files
    local log_count=$(find "$ANALYSIS_PATH" -type f -name "*.log" 2>/dev/null | wc -l)
    if [[ $log_count -gt 0 ]]; then
        local log_size=$(find "$ANALYSIS_PATH" -type f -name "*.log" -exec du -cb {} + 2>/dev/null | tail -1 | awk '{print $1}')
        local log_size_hr=$(human_readable_size "$log_size")
        echo "4. Archive or compress log files: $log_count files, $log_size_hr"
        recommendations+=("log_files")
    fi

    # Check temp files
    local temp_count=$(find "$ANALYSIS_PATH" -type f -name "*.tmp" -o -name "*.temp" 2>/dev/null | wc -l)
    if [[ $temp_count -gt 0 ]]; then
        echo "5. Remove temporary files: $temp_count files"
        recommendations+=("temp_files")
    fi

    if [[ ${#recommendations[@]} -eq 0 ]]; then
        echo "No cleanup recommendations at this time."
    fi

    echo ""
}

# ===========================
# Interactive Cleanup
# ===========================

interactive_cleanup() {
    echo -e "${BOLD}${YELLOW}Interactive Cleanup Mode${RESET}"
    echo -e "${YELLOW}========================================${RESET}"
    echo ""

    generate_cleanup_recommendations

    echo -e "${YELLOW}Would you like to proceed with cleanup? (y/N)${RESET}"
    read -r response

    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        return 0
    fi

    # Cleanup old files
    echo ""
    echo -e "${BOLD}Cleaning up old files (>$OLD_FILE_DAYS days)...${RESET}"
    local old_files=$(find "$ANALYSIS_PATH" -type f -mtime +$OLD_FILE_DAYS 2>/dev/null)
    local old_count=$(echo "$old_files" | grep -c . || echo 0)

    if [[ $old_count -gt 0 ]]; then
        echo "Found $old_count old files. Delete? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "$old_files" | xargs rm -f
            echo -e "${GREEN}✓ Removed $old_count old files${RESET}"
        fi
    fi

    # Cleanup empty files
    echo ""
    echo -e "${BOLD}Cleaning up empty files...${RESET}"
    local empty_files=$(find "$ANALYSIS_PATH" -type f -empty 2>/dev/null)
    local empty_count=$(echo "$empty_files" | grep -c . || echo 0)

    if [[ $empty_count -gt 0 ]]; then
        echo "Found $empty_count empty files. Delete? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "$empty_files" | xargs rm -f
            echo -e "${GREEN}✓ Removed $empty_count empty files${RESET}"
        fi
    fi

    # Cleanup empty directories
    echo ""
    echo -e "${BOLD}Cleaning up empty directories...${RESET}"
    local empty_dirs=$(find "$ANALYSIS_PATH" -type d -empty 2>/dev/null)
    local empty_dir_count=$(echo "$empty_dirs" | grep -c . || echo 0)

    if [[ $empty_dir_count -gt 0 ]]; then
        echo "Found $empty_dir_count empty directories. Delete? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "$empty_dirs" | xargs rmdir 2>/dev/null || true
            echo -e "${GREEN}✓ Removed empty directories${RESET}"
        fi
    fi

    echo ""
    echo -e "${GREEN}Cleanup completed!${RESET}"
}

# ===========================
# Report Generation
# ===========================

generate_text_report() {
    {
        echo "========================================="
        echo "Disk Space Analysis Report"
        echo "========================================="
        echo "Generated: $(date)"
        echo "Path: $ANALYSIS_PATH"
        echo ""

        analyze_filesystem_usage
        analyze_directory_sizes
        analyze_file_types
        find_large_files
        find_old_files
        analyze_empty_files
        generate_cleanup_recommendations
    } | if [[ -n "$REPORT_FILE" ]]; then
        tee "$REPORT_FILE" | sed 's/\x1b\[[0-9;]*m//g'  # Remove color codes for file
    else
        cat
    fi
}

generate_json_report() {
    local report_file="${REPORT_FILE:-/tmp/disk-analysis.json}"

    # Get filesystem usage
    local fs_usage=$(df "$ANALYSIS_PATH" | awk 'NR==2 {print $5}' | sed 's/%//')

    # Count old files
    local old_files=$(find "$ANALYSIS_PATH" -type f -mtime +$OLD_FILE_DAYS 2>/dev/null | wc -l)

    # Count large files
    local large_files=$(find "$ANALYSIS_PATH" -type f -size +"$MIN_SIZE" 2>/dev/null | wc -l)

    cat > "$report_file" <<EOF
{
  "generated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "path": "$ANALYSIS_PATH",
  "filesystem_usage": $fs_usage,
  "threshold": $THRESHOLD,
  "old_files": {
    "days": $OLD_FILE_DAYS,
    "count": $old_files
  },
  "large_files": {
    "min_size": "$MIN_SIZE",
    "count": $large_files
  },
  "analysis_depth": $MAX_DEPTH
}
EOF

    echo "JSON report saved to: $report_file"
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] [PATH]

Disk space analysis for ML datasets.

OPTIONS:
    -t, --threshold PCT       Alert threshold percentage (default: $THRESHOLD)
    -o, --old-days DAYS       Consider files older than DAYS (default: $OLD_FILE_DAYS)
    -s, --min-size SIZE       Minimum file size to report (default: $MIN_SIZE)
    -r, --report FILE         Generate report to file
    -c, --cleanup             Interactive cleanup mode
    -d, --depth N             Directory depth to analyze (default: $MAX_DEPTH)
    -f, --format FORMAT       Output format (text, json, html)
    -v, --verbose             Verbose output
    -h, --help                Display this help message

EXAMPLES:
    # Analyze default path
    $SCRIPT_NAME

    # Analyze specific path
    $SCRIPT_NAME /data/training

    # Generate report
    $SCRIPT_NAME -r disk-report.txt

    # Interactive cleanup
    $SCRIPT_NAME --cleanup

    # Find large files
    $SCRIPT_NAME --min-size 1G

    # JSON output
    $SCRIPT_NAME --format json -r report.json

EOF
}

# ===========================
# Argument Parsing
# ===========================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -t|--threshold)
                THRESHOLD="$2"
                shift 2
                ;;
            -o|--old-days)
                OLD_FILE_DAYS="$2"
                shift 2
                ;;
            -s|--min-size)
                MIN_SIZE="$2"
                shift 2
                ;;
            -r|--report)
                REPORT_FILE="$2"
                shift 2
                ;;
            -c|--cleanup)
                CLEANUP_MODE=true
                shift
                ;;
            -d|--depth)
                MAX_DEPTH="$2"
                shift 2
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                ANALYSIS_PATH="$1"
                shift
                ;;
        esac
    done
}

# ===========================
# Main Function
# ===========================

main() {
    parse_arguments "$@"

    # Validate path
    if [[ ! -d "$ANALYSIS_PATH" ]]; then
        echo -e "${RED}Error: Directory does not exist: $ANALYSIS_PATH${RESET}"
        exit 1
    fi

    if [[ "$CLEANUP_MODE" == true ]]; then
        interactive_cleanup
    else
        case "$OUTPUT_FORMAT" in
            text)
                generate_text_report
                ;;
            json)
                generate_json_report
                ;;
            *)
                echo "Unsupported format: $OUTPUT_FORMAT"
                exit 1
                ;;
        esac
    fi
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
