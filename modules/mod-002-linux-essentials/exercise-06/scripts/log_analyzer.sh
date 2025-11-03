#!/bin/bash
###############################################################################
# Comprehensive Log Analyzer for ML Systems
###############################################################################
#
# Usage: ./log_analyzer.sh [log_directory] [options]
#
# Description:
#   Analyzes log files from ML training systems, API services, and error logs.
#   Generates comprehensive reports with statistics, error patterns, performance
#   metrics, and recommendations.
#
# Options:
#   -h, --help           Show this help message
#   -o, --output FILE    Save report to file (default: auto-generated name)
#   -f, --format FORMAT  Output format: text, json, html (default: text)
#   -v, --verbose        Show detailed analysis
#   --no-color           Disable colored output
#

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${1:-../sample_logs}"
REPORT_FILE=""
OUTPUT_FORMAT="text"
VERBOSE=false
USE_COLOR=true

# Colors
if [[ -t 1 ]] && [[ "$USE_COLOR" = true ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    MAGENTA='\033[0;35m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' CYAN='' MAGENTA='' BOLD='' NC=''
fi

# Usage
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [log_directory] [options]

Comprehensive log analyzer for ML systems.

Arguments:
  log_directory        Directory containing log files (default: ../sample_logs)

Options:
  -h, --help           Show this help message
  -o, --output FILE    Save report to file
  -f, --format FORMAT  Output format: text, json, html (default: text)
  -v, --verbose        Show detailed analysis
  --no-color           Disable colored output

Examples:
  $SCRIPT_NAME                                    # Analyze default logs
  $SCRIPT_NAME /var/log/ml                       # Analyze production logs
  $SCRIPT_NAME ../sample_logs -o report.txt      # Save report to file
  $SCRIPT_NAME -f json -o report.json            # JSON output

EOF
    exit 0
}

# Logging
log_section() { echo -e "${BOLD}${BLUE}=== $* ===${NC}"; }
log_subsection() { echo -e "${CYAN}$*${NC}"; }
log_success() { echo -e "${GREEN}✓${NC} $*"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $*"; }
log_error() { echo -e "${RED}✗${NC} $*"; }
log_info() { echo -e "  $*"; }

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            -o|--output)
                REPORT_FILE="$2"
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
            --no-color)
                USE_COLOR=false
                GREEN='' RED='' YELLOW='' BLUE='' CYAN='' MAGENTA='' BOLD='' NC=''
                shift
                ;;
            *)
                if [ -z "$LOG_DIR" ] || [ "$LOG_DIR" = "../sample_logs" ]; then
                    LOG_DIR="$1"
                fi
                shift
                ;;
        esac
    done

    # Set default report file if not specified
    if [ -z "$REPORT_FILE" ]; then
        REPORT_FILE="analysis_report_$(date +%Y%m%d_%H%M%S).txt"
    fi
}

# Check if log directory exists
check_log_directory() {
    if [ ! -d "$LOG_DIR" ]; then
        log_error "Log directory not found: $LOG_DIR"
        exit 1
    fi

    local log_count=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | wc -l)
    if [ "$log_count" -eq 0 ]; then
        log_error "No log files found in: $LOG_DIR"
        exit 1
    fi
}

# Generate summary statistics
generate_summary() {
    log_section "Summary Statistics"

    local total_files=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | wc -l)
    local total_size=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
    local total_lines=$(find "$LOG_DIR" -name "*.log" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')

    echo "Analysis Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log Directory: $LOG_DIR"
    echo "Total log files: $total_files"
    echo "Total log size: $total_size"
    echo "Total log lines: $total_lines"
    echo ""
}

# Analyze log levels
analyze_log_levels() {
    log_section "Log Level Distribution"

    local info_count=$(grep -rh "INFO" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)
    local warning_count=$(grep -rh "WARNING" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)
    local error_count=$(grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)
    local debug_count=$(grep -rh "DEBUG" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)

    local total=$((info_count + warning_count + error_count + debug_count))

    if [ "$total" -gt 0 ]; then
        printf "%-10s %6d (%5.1f%%)\n" "INFO:" "$info_count" "$(echo "scale=1; $info_count * 100 / $total" | bc)"
        printf "%-10s %6d (%5.1f%%)\n" "WARNING:" "$warning_count" "$(echo "scale=1; $warning_count * 100 / $total" | bc)"
        printf "%-10s %6d (%5.1f%%)\n" "ERROR:" "$error_count" "$(echo "scale=1; $error_count * 100 / $total" | bc)"
        printf "%-10s %6d (%5.1f%%)\n" "DEBUG:" "$debug_count" "$(echo "scale=1; $debug_count * 100 / $total" | bc)"
    else
        echo "No log level information found"
    fi
    echo ""
}

# Analyze errors
analyze_errors() {
    log_section "Error Analysis"

    local total_errors=$(grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)
    echo "Total errors: $total_errors"
    echo ""

    if [ "$total_errors" -gt 0 ]; then
        log_subsection "Error Distribution by File:"
        for file in "$LOG_DIR"/*.log; do
            if [ -f "$file" ]; then
                local count=$(grep -c "ERROR" "$file" 2>/dev/null || echo 0)
                printf "  %-30s %d errors\n" "$(basename "$file")" "$count"
            fi
        done
        echo ""

        log_subsection "Top 10 Error Types:"
        grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | \
            sed 's/.*ERROR //' | \
            cut -d':' -f1 | \
            sort | uniq -c | \
            sort -rn | head -10 | \
            awk '{printf "  %3d  %s\n", $1, substr($0, index($0, $2))}'
        echo ""

        log_subsection "Error Timeline (by hour):"
        grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | \
            awk '{print $2}' | \
            cut -d':' -f1 | \
            sort | uniq -c | \
            awk '{printf "  %s:00 - %d errors\n", $2, $1}'
        echo ""
    fi
}

# Analyze critical issues
analyze_critical_issues() {
    log_section "Critical Issues"

    log_subsection "CUDA/GPU Errors:"
    local cuda_errors=$(grep -rih "cuda\|gpu" "$LOG_DIR" 2>/dev/null | grep -i "error" | wc -l)
    echo "  Count: $cuda_errors"
    if [ "$cuda_errors" -gt 0 ] && [ "$VERBOSE" = true ]; then
        grep -rih "cuda\|gpu" "$LOG_DIR" 2>/dev/null | grep -i "error" | head -3 | sed 's/^/    /'
    fi
    echo ""

    log_subsection "Memory Errors:"
    local memory_errors=$(grep -rih "memory\|oom\|out of memory" "$LOG_DIR" 2>/dev/null | grep -i "error" | wc -l)
    echo "  Count: $memory_errors"
    if [ "$memory_errors" -gt 0 ] && [ "$VERBOSE" = true ]; then
        grep -rih "memory\|oom\|out of memory" "$LOG_DIR" 2>/dev/null | grep -i "error" | head -3 | sed 's/^/    /'
    fi
    echo ""

    log_subsection "Disk/Storage Errors:"
    local disk_errors=$(grep -rih "disk\|space\|i/o" "$LOG_DIR" 2>/dev/null | grep -i "error" | wc -l)
    echo "  Count: $disk_errors"
    if [ "$disk_errors" -gt 0 ] && [ "$VERBOSE" = true ]; then
        grep -rih "disk\|space\|i/o" "$LOG_DIR" 2>/dev/null | grep -i "error" | head -3 | sed 's/^/    /'
    fi
    echo ""

    log_subsection "Network/Connectivity Errors:"
    local network_errors=$(grep -rih "connection\|network\|timeout" "$LOG_DIR" 2>/dev/null | grep -i "error\|failed" | wc -l)
    echo "  Count: $network_errors"
    if [ "$network_errors" -gt 0 ] && [ "$VERBOSE" = true ]; then
        grep -rih "connection\|network\|timeout" "$LOG_DIR" 2>/dev/null | grep -i "error\|failed" | head -3 | sed 's/^/    /'
    fi
    echo ""

    log_subsection "Authentication/Authorization Errors:"
    local auth_errors=$(grep -rih "auth.*fail\|unauthorized\|permission denied" "$LOG_DIR" 2>/dev/null | wc -l)
    echo "  Count: $auth_errors"
    if [ "$auth_errors" -gt 0 ] && [ "$VERBOSE" = true ]; then
        grep -rih "auth.*fail\|unauthorized\|permission denied" "$LOG_DIR" 2>/dev/null | head -3 | sed 's/^/    /'
    fi
    echo ""
}

# Analyze performance metrics
analyze_performance() {
    log_section "Performance Metrics"

    # API response times
    if grep -q "Duration:" "$LOG_DIR"/*.log 2>/dev/null; then
        log_subsection "API Response Times:"
        awk -F'Duration: |ms' '/Duration/ {
            sum += $(NF-1);
            count++;
            if ($(NF-1) < min || min == 0) min = $(NF-1);
            if ($(NF-1) > max) max = $(NF-1);
        }
        END {
            if (count > 0) {
                printf "  Average: %.2f ms\n", sum/count;
                printf "  Min: %.2f ms\n", min;
                printf "  Max: %.2f ms\n", max;
                printf "  Total requests: %d\n", count;
            } else {
                print "  No response time data found";
            }
        }' "$LOG_DIR"/*.log
        echo ""
    fi

    # Training metrics
    if grep -q "Epoch" "$LOG_DIR"/*.log 2>/dev/null; then
        log_subsection "Training Metrics:"
        local total_epochs=$(grep -rh "Epoch [0-9]*/[0-9]*" "$LOG_DIR" | wc -l)
        echo "  Total epochs completed: $total_epochs"

        # Extract final metrics if available
        if grep -q "Final metrics" "$LOG_DIR"/*.log 2>/dev/null; then
            echo "  Final metrics:"
            grep -h "Final metrics" "$LOG_DIR"/*.log 2>/dev/null | sed 's/^/    /'
        fi

        # Check for early stopping
        if grep -q "Early stopping" "$LOG_DIR"/*.log 2>/dev/null; then
            log_info "Early stopping was triggered"
        fi

        echo ""
    fi
}

# Analyze API endpoints
analyze_api_endpoints() {
    if grep -q "REQUEST" "$LOG_DIR"/*.log 2>/dev/null; then
        log_section "API Endpoint Analysis"

        log_subsection "Endpoint Usage:"
        awk '/REQUEST/ {
            match($0, /(GET|POST|PUT|DELETE|PATCH) ([^ ]+)/, m);
            if (m[1] && m[2])
                endpoints[m[1] " " m[2]]++
        }
        END {
            for (endpoint in endpoints)
                printf "  %3d  %s\n", endpoints[endpoint], endpoint
        }' "$LOG_DIR"/*.log | sort -rn
        echo ""

        log_subsection "HTTP Status Codes:"
        grep -h "RESPONSE" "$LOG_DIR"/*.log 2>/dev/null | \
            awk '{
                for (i=1; i<=NF; i++) {
                    if ($i ~ /^[0-9]{3}$/) {
                        codes[$i]++
                    }
                }
            }
            END {
                for (code in codes)
                    printf "  %3s: %d requests\n", code, codes[code]
            }' | sort
        echo ""

        # Analyze rate limiting
        if grep -q "Rate limit" "$LOG_DIR"/*.log 2>/dev/null; then
            log_subsection "Rate Limiting:"
            grep -h "Rate limit" "$LOG_DIR"/*.log | sed 's/^/  /'
            echo ""
        fi
    fi
}

# Generate recommendations
generate_recommendations() {
    log_section "Recommendations"

    local recommendations=()
    local total_errors=$(grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)
    local total_warnings=$(grep -rh "WARNING" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)

    # Error count recommendations
    if [ "$total_errors" -gt 50 ]; then
        recommendations+=("${RED}CRITICAL${NC}: Very high error count ($total_errors) - immediate investigation required")
    elif [ "$total_errors" -gt 10 ]; then
        recommendations+=("${YELLOW}WARNING${NC}: High error count ($total_errors) - investigate error patterns")
    fi

    # Memory issues
    if grep -riq "out of memory\|oom" "$LOG_DIR" 2>/dev/null; then
        recommendations+=("${YELLOW}WARNING${NC}: Memory issues detected - consider reducing batch size or upgrading RAM")
    fi

    # CUDA issues
    if grep -riq "cuda.*error" "$LOG_DIR" 2>/dev/null; then
        recommendations+=("${YELLOW}WARNING${NC}: CUDA errors detected - verify GPU drivers and CUDA installation")
    fi

    # Timeout issues
    if grep -riq "timeout" "$LOG_DIR" 2>/dev/null; then
        recommendations+=("${YELLOW}WARNING${NC}: Timeout issues detected - check network/system resources and timeout thresholds")
    fi

    # Disk space issues
    if grep -riq "no space left\|disk.*full" "$LOG_DIR" 2>/dev/null; then
        recommendations+=("${RED}CRITICAL${NC}: Disk space issues detected - free up disk space immediately")
    fi

    # Authentication issues
    if grep -riq "auth.*fail\|unauthorized" "$LOG_DIR" 2>/dev/null; then
        recommendations+=("${YELLOW}WARNING${NC}: Authentication failures detected - verify credentials and access policies")
    fi

    # Overfitting warnings
    if grep -riq "overfitting" "$LOG_DIR" 2>/dev/null; then
        recommendations+=("INFO: Overfitting detected - consider regularization techniques or early stopping")
    fi

    # Display recommendations
    if [ ${#recommendations[@]} -gt 0 ]; then
        for rec in "${recommendations[@]}"; do
            echo -e "  $rec"
        done
    else
        log_success "No critical issues detected - system appears healthy"
    fi
    echo ""
}

# Generate health score
generate_health_score() {
    log_section "System Health Score"

    local total_errors=$(grep -rh "ERROR" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)
    local total_warnings=$(grep -rh "WARNING" "$LOG_DIR" 2>/dev/null | wc -l || echo 0)
    local total_lines=$(find "$LOG_DIR" -name "*.log" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')

    if [ "$total_lines" -eq 0 ]; then
        total_lines=1  # Avoid division by zero
    fi

    local error_rate=$(echo "scale=2; $total_errors * 100 / $total_lines" | bc)
    local warning_rate=$(echo "scale=2; $total_warnings * 100 / $total_lines" | bc)

    local health_score=100
    health_score=$(echo "$health_score - ($error_rate * 10)" | bc | awk '{printf "%.0f", $1}')
    health_score=$(echo "$health_score - ($warning_rate * 2)" | bc | awk '{printf "%.0f", $1}')

    # Ensure score is between 0 and 100
    if [ "$health_score" -lt 0 ]; then
        health_score=0
    fi

    echo "Overall Health: $health_score/100"
    echo "Error Rate: $error_rate%"
    echo "Warning Rate: $warning_rate%"
    echo ""

    if [ "$health_score" -ge 90 ]; then
        log_success "System health: EXCELLENT"
    elif [ "$health_score" -ge 70 ]; then
        echo -e "${YELLOW}⚠${NC} System health: GOOD (minor issues)"
    elif [ "$health_score" -ge 50 ]; then
        echo -e "${YELLOW}⚠${NC} System health: FAIR (attention needed)"
    else
        log_error "System health: POOR (immediate action required)"
    fi
    echo ""
}

# Main report generation
generate_report() {
    {
        echo "========================================"
        echo "    ML System Log Analysis Report"
        echo "========================================"
        echo ""

        generate_summary
        analyze_log_levels
        analyze_errors
        analyze_critical_issues
        analyze_performance
        analyze_api_endpoints
        generate_health_score
        generate_recommendations

        echo "========================================"
        echo "End of Report"
        echo "========================================"

    } | if [ "$REPORT_FILE" != "/dev/stdout" ]; then
        tee "$REPORT_FILE"
    else
        cat
    fi
}

# Main function
main() {
    parse_args "$@"
    check_log_directory

    echo -e "${BOLD}${BLUE}ML System Log Analyzer${NC}"
    echo "Analyzing logs in: $LOG_DIR"
    echo ""

    generate_report

    if [ "$REPORT_FILE" != "/dev/stdout" ]; then
        echo ""
        log_success "Report saved to: $REPORT_FILE"
    fi
}

# Run main function
main "$@"
