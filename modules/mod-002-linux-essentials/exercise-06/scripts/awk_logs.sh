#!/bin/bash
#
# awk_logs.sh - Advanced Log Parsing with awk
#
# Description:
#   Demonstrates awk for extracting structured data from logs,
#   calculating statistics, and generating reports.
#

set -euo pipefail

LOG_DIR="${1:-../sample_logs}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

section() { echo -e "\n${BLUE}=== $* ===${NC}\n"; }
subsection() { echo -e "${CYAN}$*${NC}"; }

section "Extract timestamps and log levels"
awk '{print $1, $2, $3}' "$LOG_DIR/training.log"

section "Extract only ERROR messages"
awk '/ERROR/ {print $0}' "$LOG_DIR/training.log"

section "Count log levels"
echo "Log level distribution:"
awk '{print $3}' "$LOG_DIR/training.log" | sort | uniq -c | sort -rn

section "Extract metrics from training logs"
echo "Epoch metrics summary:"
awk '/Epoch [0-9]+\/100/ {
    match($0, /Epoch ([0-9]+)/, epoch);
    match($0, /loss: ([0-9.]+)/, loss);
    match($0, /accuracy: ([0-9.]+)/, acc);
    match($0, /val_loss: ([0-9.]+)/, val_loss);
    match($0, /val_accuracy: ([0-9.]+)/, val_acc);

    printf "Epoch %2d: Loss=%.4f Acc=%.4f Val_Loss=%.4f Val_Acc=%.4f\n",
        epoch[1], loss[1], acc[1], val_loss[1], val_acc[1]
}' "$LOG_DIR/training.log"

section "Calculate average response time from API logs"
awk -F'Duration: |ms' '/Duration/ {
    duration = $(NF-1);
    sum += duration;
    count++;
    if (duration < min || min == 0) min = duration;
    if (duration > max) max = duration;
}
END {
    if (count > 0) {
        printf "API Response Time Statistics:\n";
        printf "  Total requests: %d\n", count;
        printf "  Average: %.2f ms\n", sum/count;
        printf "  Min: %.2f ms\n", min;
        printf "  Max: %.2f ms\n", max;
    } else {
        print "No response time data found";
    }
}' "$LOG_DIR/api.log"

section "Extract API endpoints and methods"
awk '/REQUEST/ {
    match($0, /(GET|POST|PUT|DELETE|PATCH) ([^ ]+)/, m);
    if (m[1] && m[2])
        print m[1], m[2]
}' "$LOG_DIR/api.log" | sort | uniq -c | sort -rn

section "Count errors by type"
echo "Error type frequency:"
awk -F': ' '/ERROR/ {
    error_type = $2;
    if (length(error_type) > 50)
        error_type = substr(error_type, 1, 50) "...";
    errors[error_type]++
}
END {
    for (error in errors)
        printf "%3d  %s\n", errors[error], error
}' "$LOG_DIR/errors.log" | sort -rn

section "Extract HTTP status codes from API logs"
awk '/RESPONSE/ {
    for (i=1; i<=NF; i++) {
        if ($i ~ /^[0-9]{3}$/) {
            status_codes[$i]++
        }
    }
}
END {
    printf "HTTP Status Code Distribution:\n";
    for (code in status_codes)
        printf "  %s: %d requests\n", code, status_codes[code]
}' "$LOG_DIR/api.log" | sort

section "Analyze errors by hour"
awk '/ERROR/ {
    hour = substr($2, 1, 2);
    errors_by_hour[hour]++
}
END {
    printf "Errors by Hour:\n";
    for (hour in errors_by_hour)
        printf "  %s:00 - %d errors\n", hour, errors_by_hour[hour]
}' "$LOG_DIR/errors.log" | sort

section "Calculate training progress statistics"
awk '
BEGIN {
    print "Training Progress Analysis:"
}
/Epoch [0-9]+\/100/ {
    epochs++;
    match($0, /loss: ([0-9.]+)/, loss);
    match($0, /accuracy: ([0-9.]+)/, acc);

    loss_sum += loss[1];
    acc_sum += acc[1];

    if (epochs == 1) {
        first_loss = loss[1];
        first_acc = acc[1];
    }
    last_loss = loss[1];
    last_acc = acc[1];
}
END {
    if (epochs > 0) {
        printf "  Total epochs: %d\n", epochs;
        printf "  Average loss: %.4f\n", loss_sum/epochs;
        printf "  Average accuracy: %.4f\n", acc_sum/epochs;
        printf "  Loss improvement: %.4f (%.1f%%)\n",
            first_loss - last_loss,
            (first_loss - last_loss) / first_loss * 100;
        printf "  Accuracy improvement: %.4f (%.1f%%)\n",
            last_acc - first_acc,
            (last_acc - first_acc) / first_acc * 100;
    }
}' "$LOG_DIR/training.log"

section "Extract IP addresses from API logs"
awk '/IP:/ {
    match($0, /IP: ([0-9.]+)/, ip);
    if (ip[1])
        ips[ip[1]]++
}
END {
    printf "Top IP Addresses:\n";
    for (ip in ips)
        printf "  %15s: %d requests\n", ip, ips[ip]
}' "$LOG_DIR/api.log" | sort -t: -k2 -rn

section "Filter logs by time range"
echo "Logs between 10:04:00 and 10:08:00:"
awk '$2 >= "10:04:00" && $2 <= "10:08:00" {print $0}' "$LOG_DIR/training.log"

section "Generate performance summary"
awk '
BEGIN {
    print "Performance Summary Report"
    print "=========================="
}
/Duration:/ {
    match($0, /Duration: ([0-9]+)ms/, dur);
    if (dur[1] < 100) fast++;
    else if (dur[1] < 1000) medium++;
    else slow++;
    total_requests++;
}
END {
    if (total_requests > 0) {
        printf "\nResponse Time Categories:\n";
        printf "  Fast (<100ms):    %d (%.1f%%)\n", fast, fast/total_requests*100;
        printf "  Medium (100-1s):  %d (%.1f%%)\n", medium, medium/total_requests*100;
        printf "  Slow (>1s):       %d (%.1f%%)\n", slow, slow/total_requests*100;
    }
}' "$LOG_DIR/api.log"

echo -e "\n${GREEN}Awk analysis complete!${NC}"
