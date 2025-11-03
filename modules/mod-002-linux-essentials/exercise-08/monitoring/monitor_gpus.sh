#!/bin/bash
###############################################################################
# GPU Health Monitoring Script - Automated GPU Monitoring with Alerts
###############################################################################
#
# Purpose: Monitor GPU health metrics and send alerts when thresholds exceeded
#
# Usage: ./monitor_gpus.sh [OPTIONS]
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration (can be overridden with environment variables)
TEMP_THRESHOLD="${TEMP_THRESHOLD:-80}"  # Celsius
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-90}"  # Percent
POWER_THRESHOLD="${POWER_THRESHOLD:-95}"  # Percent of power limit
ALERT_EMAIL="${ALERT_EMAIL:-ml-ops@company.com}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"  # Slack/Discord webhook URL
LOG_FILE="${LOG_FILE:-/var/log/gpu-monitor.log}"
METRICS_FILE="${METRICS_FILE:-/var/log/gpu-metrics.csv}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"  # seconds
ALERT_COOLDOWN="${ALERT_COOLDOWN:-3600}"  # seconds between repeat alerts
QUIET="${QUIET:-false}"

# State file for alert cooldowns
STATE_DIR="/var/tmp/gpu-monitor"
mkdir -p "$STATE_DIR"

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$QUIET" != "true" ]; then
        case "$level" in
            INFO)
                echo -e "${BLUE}[INFO]${NC} $message"
                ;;
            SUCCESS)
                echo -e "${GREEN}[OK]${NC} $message"
                ;;
            WARNING)
                echo -e "${YELLOW}[WARNING]${NC} $message"
                ;;
            ERROR)
                echo -e "${RED}[ERROR]${NC} $message"
                ;;
        esac
    fi

    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Help function
show_help() {
    cat << EOF
GPU Health Monitoring Script

Usage: $0 [OPTIONS]

Options:
    --temp-threshold N      Temperature alert threshold in °C (default: 80)
    --memory-threshold N    Memory utilization alert threshold % (default: 90)
    --power-threshold N     Power usage alert threshold % (default: 95)
    --alert-email EMAIL     Email address for alerts
    --alert-webhook URL     Slack/Discord webhook URL for alerts
    --log-file PATH         Log file location (default: /var/log/gpu-monitor.log)
    --metrics-file PATH     Metrics CSV file (default: /var/log/gpu-metrics.csv)
    --cooldown SECONDS      Seconds between repeat alerts (default: 3600)
    --quiet                 Suppress console output
    -h, --help              Show this help message

Environment Variables:
    TEMP_THRESHOLD, MEMORY_THRESHOLD, POWER_THRESHOLD, ALERT_EMAIL, ALERT_WEBHOOK

Examples:
    # Basic monitoring
    $0

    # With email alerts
    $0 --alert-email ops@company.com

    # With Slack webhook
    $0 --alert-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL

    # Custom thresholds
    $0 --temp-threshold 85 --memory-threshold 95

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --temp-threshold)
            TEMP_THRESHOLD="$2"
            shift 2
            ;;
        --memory-threshold)
            MEMORY_THRESHOLD="$2"
            shift 2
            ;;
        --power-threshold)
            POWER_THRESHOLD="$2"
            shift 2
            ;;
        --alert-email)
            ALERT_EMAIL="$2"
            shift 2
            ;;
        --alert-webhook)
            ALERT_WEBHOOK="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --metrics-file)
            METRICS_FILE="$2"
            shift 2
            ;;
        --cooldown)
            ALERT_COOLDOWN="$2"
            shift 2
            ;;
        --quiet)
            QUIET="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log ERROR "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    log ERROR "nvidia-smi not found. GPU monitoring requires NVIDIA drivers."
    exit 1
fi

# Test nvidia-smi works
if ! nvidia-smi &> /dev/null; then
    log ERROR "nvidia-smi failed to run. Check NVIDIA driver installation."
    exit 1
fi

# Function to check if alert should be sent (cooldown period)
should_alert() {
    local alert_key="$1"
    local state_file="${STATE_DIR}/${alert_key}"

    if [ ! -f "$state_file" ]; then
        return 0  # No previous alert, should send
    fi

    local last_alert=$(cat "$state_file")
    local current_time=$(date +%s)
    local time_diff=$((current_time - last_alert))

    if [ $time_diff -gt $ALERT_COOLDOWN ]; then
        return 0  # Cooldown expired, should send
    fi

    return 1  # Still in cooldown period
}

# Function to record alert sent
record_alert() {
    local alert_key="$1"
    local state_file="${STATE_DIR}/${alert_key}"
    date +%s > "$state_file"
}

# Function to send email alert
send_email_alert() {
    local subject="$1"
    local message="$2"

    if [ -z "$ALERT_EMAIL" ]; then
        return
    fi

    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "$subject" "$ALERT_EMAIL"
        log INFO "Email alert sent to $ALERT_EMAIL"
    elif command -v sendmail &> /dev/null; then
        {
            echo "Subject: $subject"
            echo ""
            echo "$message"
        } | sendmail "$ALERT_EMAIL"
        log INFO "Email alert sent via sendmail to $ALERT_EMAIL"
    else
        log WARNING "Email command not found (install mailutils or sendmail)"
    fi
}

# Function to send webhook alert (Slack/Discord)
send_webhook_alert() {
    local message="$1"
    local severity="$2"  # warning, critical

    if [ -z "$ALERT_WEBHOOK" ]; then
        return
    fi

    local color
    case "$severity" in
        critical)
            color="#FF0000"  # Red
            ;;
        warning)
            color="#FFA500"  # Orange
            ;;
        *)
            color="#0000FF"  # Blue
            ;;
    esac

    local payload=$(cat <<EOF
{
    "text": "GPU Alert",
    "attachments": [{
        "color": "$color",
        "text": "$message",
        "footer": "GPU Monitor",
        "ts": $(date +%s)
    }]
}
EOF
)

    if curl -s -X POST -H 'Content-type: application/json' \
        --data "$payload" "$ALERT_WEBHOOK" > /dev/null; then
        log INFO "Webhook alert sent"
    else
        log WARNING "Webhook alert failed"
    fi
}

# Function to send alert
send_alert() {
    local alert_key="$1"
    local subject="$2"
    local message="$3"
    local severity="${4:-warning}"

    # Check cooldown
    if ! should_alert "$alert_key"; then
        log INFO "Alert suppressed (cooldown active): $subject"
        return
    fi

    log WARNING "ALERT: $subject"

    # Log to syslog
    logger -t gpu-monitor -p user.warning "$subject: $message"

    # Send email
    send_email_alert "$subject" "$message"

    # Send webhook
    send_webhook_alert "$message" "$severity"

    # Record alert sent
    record_alert "$alert_key"
}

# Initialize metrics file with header
if [ ! -f "$METRICS_FILE" ]; then
    echo "timestamp,gpu_index,gpu_name,temperature_c,utilization_gpu_%,utilization_memory_%,memory_used_mib,memory_total_mib,power_draw_w,power_limit_w" > "$METRICS_FILE"
fi

# Start monitoring
log INFO "═══════════════════════════════════════════════════════════"
log INFO "GPU Health Monitor Started"
log INFO "═══════════════════════════════════════════════════════════"
log INFO "Configuration:"
log INFO "  Temperature threshold: ${TEMP_THRESHOLD}°C"
log INFO "  Memory threshold: ${MEMORY_THRESHOLD}%"
log INFO "  Power threshold: ${POWER_THRESHOLD}%"
log INFO "  Alert cooldown: ${ALERT_COOLDOWN}s"
[ -n "$ALERT_EMAIL" ] && log INFO "  Alert email: $ALERT_EMAIL"
[ -n "$ALERT_WEBHOOK" ] && log INFO "  Webhook configured: Yes"
log INFO ""

# Query GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
log INFO "Monitoring $GPU_COUNT GPU(s)..."
log INFO ""

# Query GPU information
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit \
    --format=csv,noheader,nounits | while IFS=',' read -r idx name temp util_gpu util_mem mem_used mem_total power_draw power_limit; do

    # Clean up whitespace
    idx=$(echo "$idx" | xargs)
    name=$(echo "$name" | xargs)
    temp=$(echo "$temp" | xargs)
    util_gpu=$(echo "$util_gpu" | xargs)
    util_mem=$(echo "$util_mem" | xargs)
    mem_used=$(echo "$mem_used" | xargs)
    mem_total=$(echo "$mem_total" | xargs)
    power_draw=$(echo "$power_draw" | xargs)
    power_limit=$(echo "$power_limit" | xargs)

    # Log current metrics
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    log INFO "GPU $idx ($name):"
    log INFO "  Temperature: ${temp}°C"
    log INFO "  GPU Utilization: ${util_gpu}%"
    log INFO "  Memory Utilization: ${util_mem}%"
    log INFO "  Memory: ${mem_used}/${mem_total} MiB"
    log INFO "  Power: ${power_draw}/${power_limit} W"

    # Save metrics to CSV
    echo "$timestamp,$idx,$name,$temp,$util_gpu,$util_mem,$mem_used,$mem_total,$power_draw,$power_limit" >> "$METRICS_FILE"

    # Check temperature threshold
    if [ "$temp" -gt "$TEMP_THRESHOLD" ]; then
        severity="warning"
        if [ "$temp" -gt $((TEMP_THRESHOLD + 10)) ]; then
            severity="critical"
        fi

        send_alert "gpu${idx}_temp" \
            "GPU Temperature Alert - GPU $idx" \
            "GPU $idx ($name) temperature is ${temp}°C (threshold: ${TEMP_THRESHOLD}°C)\n\nCurrent status:\n  GPU Util: ${util_gpu}%\n  Mem Util: ${util_mem}%\n  Power: ${power_draw}W/${power_limit}W" \
            "$severity"
    else
        log SUCCESS "  Temperature OK"
    fi

    # Check memory utilization threshold
    if [ "$util_mem" -gt "$MEMORY_THRESHOLD" ]; then
        send_alert "gpu${idx}_memory" \
            "GPU Memory Alert - GPU $idx" \
            "GPU $idx ($name) memory utilization is ${util_mem}% (threshold: ${MEMORY_THRESHOLD}%)\n\nMemory: ${mem_used}/${mem_total} MiB\n  GPU Util: ${util_gpu}%\n  Temperature: ${temp}°C" \
            "warning"
    else
        log SUCCESS "  Memory usage OK"
    fi

    # Check power draw (as percentage of limit)
    if [ -n "$power_limit" ] && [ "$power_limit" != "N/A" ] && [ "$power_limit" -gt 0 ]; then
        power_pct=$(echo "scale=0; $power_draw * 100 / $power_limit" | bc)

        if [ "$power_pct" -gt "$POWER_THRESHOLD" ]; then
            send_alert "gpu${idx}_power" \
                "GPU Power Alert - GPU $idx" \
                "GPU $idx ($name) power usage is ${power_pct}% of limit (threshold: ${POWER_THRESHOLD}%)\n\nPower: ${power_draw}W/${power_limit}W\n  GPU Util: ${util_gpu}%\n  Temperature: ${temp}°C" \
                "warning"
        else
            log SUCCESS "  Power usage OK"
        fi
    fi

    # Check for any GPU errors
    gpu_status=$(nvidia-smi --query-gpu=index --format=csv,noheader -i "$idx" 2>&1)
    if [ $? -ne 0 ]; then
        send_alert "gpu${idx}_error" \
            "GPU Error - GPU $idx" \
            "GPU $idx encountered an error:\n$gpu_status" \
            "critical"
    fi

    log INFO ""
done

# Check for GPU compute processes
log INFO "Active GPU Processes:"
process_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)

if [ "$process_count" -eq 0 ]; then
    log INFO "  No active GPU processes"
else
    log INFO "  $process_count active process(es)"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | while IFS=',' read -r pid pname mem; do
        log INFO "    PID $pid: $pname (Memory: $mem)"
    done
fi

log INFO ""
log SUCCESS "GPU monitoring check completed"
log INFO "Next check in ${CHECK_INTERVAL}s (if running as daemon)"

# Cleanup old metrics (keep last 7 days)
if [ -f "$METRICS_FILE" ]; then
    METRICS_LINES=$(wc -l < "$METRICS_FILE")
    if [ "$METRICS_LINES" -gt 100000 ]; then
        log INFO "Rotating metrics file (size: $METRICS_LINES lines)"
        tail -50000 "$METRICS_FILE" > "${METRICS_FILE}.tmp"
        mv "${METRICS_FILE}.tmp" "$METRICS_FILE"
    fi
fi

exit 0
