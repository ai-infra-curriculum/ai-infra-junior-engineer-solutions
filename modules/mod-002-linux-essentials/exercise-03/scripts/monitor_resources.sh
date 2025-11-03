#!/bin/bash
#
# monitor_resources.sh - Monitor training process resources
#
# Usage: ./monitor_resources.sh [PID]
#
# If PID is not provided, reads from training.pid file
#

set -e
set -u

PID_FILE="training.pid"
OUTPUT_FILE="resource_usage.csv"
UPDATE_INTERVAL=2  # seconds

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

# Get PID
if [ $# -gt 0 ]; then
    PID=$1
elif [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
else
    echo -e "${RED}Error: No PID provided and $PID_FILE not found${NC}" >&2
    echo "Usage: $0 [PID]" >&2
    exit 1
fi

# Verify process exists
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${RED}Error: Process $PID not running${NC}" >&2
    exit 1
fi

# Create CSV header
echo "timestamp,cpu_percent,mem_percent,mem_rss_mb,mem_vsz_mb" > "$OUTPUT_FILE"

# Display header
echo ""
echo -e "${BLUE}=== Resource Monitor ===${NC}"
echo -e "${BLUE}Monitoring PID: $PID${NC}"
echo -e "${BLUE}Update interval: ${UPDATE_INTERVAL}s${NC}"
echo -e "${BLUE}Output file: $OUTPUT_FILE${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
echo ""

printf "%-20s %8s %8s %12s %12s\n" "TIME" "CPU%" "MEM%" "RSS(MB)" "VSZ(MB)"
printf "%-20s %8s %8s %12s %12s\n" "----" "----" "----" "-------" "-------"

# Monitor loop
while ps -p "$PID" > /dev/null 2>&1; do
    # Get process stats
    if ! STATS=$(ps -p "$PID" -o %cpu,%mem,rss,vsz --no-headers 2>/dev/null); then
        echo ""
        echo -e "${RED}Process $PID terminated${NC}"
        break
    fi

    # Parse stats
    CPU=$(echo "$STATS" | awk '{print $1}')
    MEM=$(echo "$STATS" | awk '{print $2}')
    RSS=$(echo "$STATS" | awk '{printf "%.1f", $3/1024}')  # Convert KB to MB
    VSZ=$(echo "$STATS" | awk '{printf "%.1f", $4/1024}')  # Convert KB to MB

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Display
    printf "%-20s %7s%% %7s%% %11s %11s\n" \
        "$TIMESTAMP" "$CPU" "$MEM" "$RSS" "$VSZ"

    # Log to CSV
    echo "$TIMESTAMP,$CPU,$MEM,$RSS,$VSZ" >> "$OUTPUT_FILE"

    # Sleep
    sleep "$UPDATE_INTERVAL"
done

echo ""
echo -e "${GREEN}Monitoring complete${NC}"
echo -e "${BLUE}Resource usage logged to: $OUTPUT_FILE${NC}"
echo ""
echo -e "${YELLOW}Analyze with: ./analyze_resources.py $OUTPUT_FILE${NC}"
echo ""
