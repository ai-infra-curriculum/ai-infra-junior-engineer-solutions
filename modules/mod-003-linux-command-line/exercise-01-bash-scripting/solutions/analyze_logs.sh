#!/bin/bash
#
# analyze_logs.sh - Analyze ML training and inference logs
#
# Description:
#   Parses and analyzes logs to extract metrics, identify errors,
#   and generate reports for ML infrastructure monitoring.
#
# Usage:
#   ./analyze_logs.sh [OPTIONS] LOG_FILE
#
# Options:
#   -t, --type TYPE         Log type (training, inference, system)
#   -s, --since DURATION    Analyze logs since duration (e.g., 1h, 30m, 1d)
#   -o, --output FILE       Output report to file
#   -f, --format FORMAT     Output format (text, json, html)
#   -a, --alert            Send alerts for errors
#   -v, --verbose          Enable verbose output
#   -h, --help             Display this help message
#

set -euo pipefail

# ===========================
# Configuration
# ===========================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration
LOG_TYPE="training"
SINCE_DURATION=""
OUTPUT_FILE=""
OUTPUT_FORMAT="text"
SEND_ALERTS=false
VERBOSE=false

# Alert thresholds
readonly ERROR_THRESHOLD=10
readonly WARNING_THRESHOLD=50

# ===========================
# Logging Functions
# ===========================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG: $*"
    fi
}

error_exit() {
    echo "ERROR: $1" >&2
    exit "${2:-1}"
}

# ===========================
# Log Analysis Functions
# ===========================

count_log_levels() {
    local log_file="$1"

    log_verbose "Counting log levels..."

    local info_count=$(grep -c "\[INFO\]" "$log_file" 2>/dev/null || echo 0)
    local warn_count=$(grep -c "\[WARNING\]" "$log_file" 2>/dev/null || echo 0)
    local error_count=$(grep -c "\[ERROR\]" "$log_file" 2>/dev/null || echo 0)
    local debug_count=$(grep -c "\[DEBUG\]" "$log_file" 2>/dev/null || echo 0)

    cat <<EOF
{
  "info": $info_count,
  "warning": $warn_count,
  "error": $error_count,
  "debug": $debug_count
}
EOF
}

extract_errors() {
    local log_file="$1"
    local limit="${2:-10}"

    log_verbose "Extracting top errors..."

    grep "\[ERROR\]" "$log_file" 2>/dev/null \
        | sed 's/.*\[ERROR\] //' \
        | sort \
        | uniq -c \
        | sort -rn \
        | head -n "$limit"
}

extract_warnings() {
    local log_file="$1"
    local limit="${2:-10}"

    log_verbose "Extracting top warnings..."

    grep "\[WARNING\]" "$log_file" 2>/dev/null \
        | sed 's/.*\[WARNING\] //' \
        | sort \
        | uniq -c \
        | sort -rn \
        | head -n "$limit"
}

analyze_training_logs() {
    local log_file="$1"

    log "INFO: Analyzing training logs..."

    # Extract training metrics
    local epochs=$(grep -oP "Epoch \K\d+" "$log_file" 2>/dev/null | tail -1 || echo 0)
    local final_loss=$(grep -oP "loss: \K[\d.]+" "$log_file" 2>/dev/null | tail -1 || echo "N/A")
    local final_acc=$(grep -oP "accuracy: \K[\d.]+" "$log_file" 2>/dev/null | tail -1 || echo "N/A")

    # Calculate training time
    local start_time=$(head -1 "$log_file" | grep -oP "^\[\K[0-9:\- ]+" || echo "")
    local end_time=$(tail -1 "$log_file" | grep -oP "^\[\K[0-9:\- ]+" || echo "")

    cat <<EOF
{
  "training_summary": {
    "epochs_completed": $epochs,
    "final_loss": "$final_loss",
    "final_accuracy": "$final_acc",
    "start_time": "$start_time",
    "end_time": "$end_time"
  }
}
EOF
}

analyze_inference_logs() {
    local log_file="$1"

    log "INFO: Analyzing inference logs..."

    # Count predictions
    local total_predictions=$(grep -c "prediction" "$log_file" 2>/dev/null || echo 0)

    # Calculate average latency
    local avg_latency=$(grep -oP "latency: \K[\d.]+" "$log_file" 2>/dev/null \
        | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')

    # Find p95 and p99 latencies
    local latencies=$(grep -oP "latency: \K[\d.]+" "$log_file" 2>/dev/null | sort -n)
    local total_count=$(echo "$latencies" | wc -l)

    local p95_index=$((total_count * 95 / 100))
    local p99_index=$((total_count * 99 / 100))

    local p95_latency=$(echo "$latencies" | sed -n "${p95_index}p" || echo "N/A")
    local p99_latency=$(echo "$latencies" | sed -n "${p99_index}p" || echo "N/A")

    cat <<EOF
{
  "inference_summary": {
    "total_predictions": $total_predictions,
    "avg_latency_ms": "$avg_latency",
    "p95_latency_ms": "$p95_latency",
    "p99_latency_ms": "$p99_latency"
  }
}
EOF
}

analyze_system_logs() {
    local log_file="$1"

    log "INFO: Analyzing system logs..."

    # Extract resource metrics
    local avg_cpu=$(grep -oP "CPU: \K[\d.]+" "$log_file" 2>/dev/null \
        | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')

    local avg_memory=$(grep -oP "Memory: \K[\d.]+" "$log_file" 2>/dev/null \
        | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')

    local max_cpu=$(grep -oP "CPU: \K[\d.]+" "$log_file" 2>/dev/null \
        | sort -n | tail -1 || echo "N/A")

    local max_memory=$(grep -oP "Memory: \K[\d.]+" "$log_file" 2>/dev/null \
        | sort -n | tail -1 || echo "N/A")

    cat <<EOF
{
  "system_summary": {
    "avg_cpu_percent": "$avg_cpu",
    "max_cpu_percent": "$max_cpu",
    "avg_memory_percent": "$avg_memory",
    "max_memory_percent": "$max_memory"
  }
}
EOF
}

# ===========================
# Report Generation
# ===========================

generate_text_report() {
    local log_file="$1"
    local log_counts="$2"
    local errors="$3"
    local warnings="$4"
    local type_analysis="$5"

    cat <<EOF
================================================================================
                        Log Analysis Report
================================================================================

File: $log_file
Analyzed: $(date)
Log Type: $LOG_TYPE

--------------------------------------------------------------------------------
Log Level Summary:
--------------------------------------------------------------------------------

$(echo "$log_counts" | jq -r 'to_entries | .[] | "  \(.key | ascii_upcase): \(.value)"' 2>/dev/null || echo "$log_counts")

--------------------------------------------------------------------------------
Top 10 Errors:
--------------------------------------------------------------------------------

$errors

--------------------------------------------------------------------------------
Top 10 Warnings:
--------------------------------------------------------------------------------

$warnings

--------------------------------------------------------------------------------
Detailed Analysis:
--------------------------------------------------------------------------------

$type_analysis

================================================================================
                            End of Report
================================================================================
EOF
}

generate_json_report() {
    local log_file="$1"
    local log_counts="$2"
    local errors="$3"
    local warnings="$4"
    local type_analysis="$5"

    # Convert errors and warnings to JSON arrays
    local errors_json=$(echo "$errors" | awk '{
        count=$1;
        $1="";
        gsub(/^[ \t]+/, "");
        printf "{\"count\": %d, \"message\": \"%s\"},\n", count, $0
    }' | sed '$ s/,$//')

    local warnings_json=$(echo "$warnings" | awk '{
        count=$1;
        $1="";
        gsub(/^[ \t]+/, "");
        printf "{\"count\": %d, \"message\": \"%s\"},\n", count, $0
    }' | sed '$ s/,$//')

    cat <<EOF
{
  "report": {
    "file": "$log_file",
    "analyzed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "log_type": "$LOG_TYPE",
    "log_counts": $log_counts,
    "top_errors": [$errors_json],
    "top_warnings": [$warnings_json],
    "analysis": $type_analysis
  }
}
EOF
}

generate_html_report() {
    local log_file="$1"
    local log_counts="$2"
    local errors="$3"
    local warnings="$4"
    local type_analysis="$5"

    cat <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>Log Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #666; margin-top: 30px; }
        .info { background: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { font-size: 24px; color: #2196F3; }
        .error { color: #f44336; }
        .warning { color: #ff9800; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        pre { background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Log Analysis Report</h1>

        <div class="info">
            <strong>File:</strong> $log_file<br>
            <strong>Analyzed:</strong> $(date)<br>
            <strong>Type:</strong> $LOG_TYPE
        </div>

        <h2>Log Level Summary</h2>
        <div>
            $(echo "$log_counts" | jq -r '
                to_entries |
                map("<div class=\"metric\"><span class=\"metric-label\">\(.key | ascii_upcase):</span> <span class=\"metric-value\">\(.value)</span></div>") |
                join("")
            ' 2>/dev/null || echo "<pre>$log_counts</pre>")
        </div>

        <h2>Top Errors</h2>
        <table>
            <tr><th>Count</th><th>Error Message</th></tr>
            $(echo "$errors" | awk '{count=$1; $1=""; printf "<tr><td class=\"error\">%d</td><td>%s</td></tr>\n", count, $0}')
        </table>

        <h2>Top Warnings</h2>
        <table>
            <tr><th>Count</th><th>Warning Message</th></tr>
            $(echo "$warnings" | awk '{count=$1; $1=""; printf "<tr><td class=\"warning\">%d</td><td>%s</td></tr>\n", count, $0}')
        </table>

        <h2>Detailed Analysis</h2>
        <pre>$type_analysis</pre>
    </div>
</body>
</html>
EOF
}

# ===========================
# Alert Functions
# ===========================

send_alert() {
    local subject="$1"
    local message="$2"

    log "ALERT: $subject"

    # Send via email (if mail command is available)
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "$subject" admin@example.com
    fi

    # Send to Slack (if webhook URL is configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"$subject\\n$message\"}" \
            2>/dev/null || true
    fi

    # Log to syslog
    logger -t "ml-log-analyzer" "$subject: $message"
}

check_alert_thresholds() {
    local error_count="$1"
    local warning_count="$2"

    if [[ $error_count -ge $ERROR_THRESHOLD ]]; then
        send_alert \
            "High Error Rate Detected" \
            "Found $error_count errors in logs (threshold: $ERROR_THRESHOLD)"
    fi

    if [[ $warning_count -ge $WARNING_THRESHOLD ]]; then
        send_alert \
            "High Warning Rate Detected" \
            "Found $warning_count warnings in logs (threshold: $WARNING_THRESHOLD)"
    fi
}

# ===========================
# Main Analysis
# ===========================

analyze_logs() {
    local log_file="$1"

    if [[ ! -f "$log_file" ]]; then
        error_exit "Log file not found: $log_file" 1
    fi

    log "INFO: Starting log analysis for: $log_file"

    # Filter logs by time if specified
    local filtered_log="$log_file"
    if [[ -n "$SINCE_DURATION" ]]; then
        log_verbose "Filtering logs since: $SINCE_DURATION"
        # Create temporary filtered log
        filtered_log=$(mktemp)
        trap "rm -f $filtered_log" EXIT

        # Simple time-based filtering (assumes timestamp at start of each line)
        # For more sophisticated filtering, use journalctl-style parsing
        tail -n 10000 "$log_file" > "$filtered_log"
    fi

    # Count log levels
    local log_counts=$(count_log_levels "$filtered_log")

    # Extract errors and warnings
    local top_errors=$(extract_errors "$filtered_log")
    local top_warnings=$(extract_warnings "$filtered_log")

    # Type-specific analysis
    local type_analysis=""
    case "$LOG_TYPE" in
        training)
            type_analysis=$(analyze_training_logs "$filtered_log")
            ;;
        inference)
            type_analysis=$(analyze_inference_logs "$filtered_log")
            ;;
        system)
            type_analysis=$(analyze_system_logs "$filtered_log")
            ;;
        *)
            type_analysis='{"message": "No type-specific analysis performed"}'
            ;;
    esac

    # Generate report
    local report=""
    case "$OUTPUT_FORMAT" in
        json)
            report=$(generate_json_report "$log_file" "$log_counts" "$top_errors" "$top_warnings" "$type_analysis")
            ;;
        html)
            report=$(generate_html_report "$log_file" "$log_counts" "$top_errors" "$top_warnings" "$type_analysis")
            ;;
        *)
            report=$(generate_text_report "$log_file" "$log_counts" "$top_errors" "$top_warnings" "$type_analysis")
            ;;
    esac

    # Output report
    if [[ -n "$OUTPUT_FILE" ]]; then
        echo "$report" > "$OUTPUT_FILE"
        log "SUCCESS: Report saved to: $OUTPUT_FILE"
    else
        echo "$report"
    fi

    # Send alerts if enabled
    if [[ "$SEND_ALERTS" == true ]]; then
        local error_count=$(echo "$log_counts" | jq -r '.error' 2>/dev/null || echo 0)
        local warning_count=$(echo "$log_counts" | jq -r '.warning' 2>/dev/null || echo 0)
        check_alert_thresholds "$error_count" "$warning_count"
    fi

    return 0
}

# ===========================
# Usage
# ===========================

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] LOG_FILE

Analyze ML training and inference logs.

OPTIONS:
    -t, --type TYPE         Log type (training, inference, system)
                           Default: training
    -s, --since DURATION    Analyze logs since duration (e.g., 1h, 30m, 1d)
    -o, --output FILE       Output report to file
    -f, --format FORMAT     Output format (text, json, html)
                           Default: text
    -a, --alert            Send alerts for errors
    -v, --verbose          Enable verbose output
    -h, --help             Display this help message

EXAMPLES:
    # Analyze training logs
    $SCRIPT_NAME --type training training.log

    # Generate JSON report
    $SCRIPT_NAME --format json --output report.json inference.log

    # Analyze with alerts
    $SCRIPT_NAME --alert --type system system.log

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
            -t|--type)
                LOG_TYPE="$2"
                shift 2
                ;;
            -s|--since)
                SINCE_DURATION="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -a|--alert)
                SEND_ALERTS=true
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
            -*)
                error_exit "Unknown option: $1" 1
                ;;
            *)
                LOG_FILE="$1"
                shift
                ;;
        esac
    done

    if [[ -z "${LOG_FILE:-}" ]]; then
        error_exit "LOG_FILE is required" 1
    fi
}

# ===========================
# Main Function
# ===========================

main() {
    parse_arguments "$@"

    log "INFO: =========================================="
    log "INFO: ML Log Analyzer"
    log "INFO: =========================================="

    analyze_logs "$LOG_FILE"

    log "INFO: =========================================="
    log "SUCCESS: Analysis complete!"
    log "INFO: =========================================="

    exit 0
}

# ===========================
# Script Entry Point
# ===========================

main "$@"
