#!/bin/bash
###############################################################################
# Network Latency Analysis Tool
###############################################################################
#
# Purpose: Analyze network latency issues with statistical analysis
#
# Usage: ./analyze_latency.sh <host>
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <host>"
    echo ""
    echo "Examples:"
    echo "  $0 ml-api.internal"
    echo "  $0 192.168.1.100"
    echo "  $0 google.com"
    exit 1
fi

TARGET="$1"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Latency Analysis for $TARGET${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# 1. RTT Measurement
echo -e "${CYAN}[1] Round-Trip Time (RTT) Measurement${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Sending 100 pings with 0.2s interval..."
echo ""

PING_OUTPUT=$(ping -c 100 -i 0.2 "$TARGET" 2>&1)

if echo "$PING_OUTPUT" | grep -q "0% packet loss"; then
    echo -e "${GREEN}✓${NC} No packet loss detected"
else
    PACKET_LOSS=$(echo "$PING_OUTPUT" | grep "packet loss" | awk '{print $6}')
    echo -e "${YELLOW}!${NC} Packet loss detected: $PACKET_LOSS"
fi

echo ""
echo "Summary statistics:"
echo "$PING_OUTPUT" | tail -n 2
echo ""

# 2. Latency Distribution
echo -e "${CYAN}[2] Latency Distribution${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Analyzing 50 samples..."
echo ""

# Collect latency samples
LATENCIES=$(ping -c 50 "$TARGET" 2>&1 | grep 'time=' | awk -F'time=' '{print $2}' | awk '{print $1}' | sort -n)

if [ -z "$LATENCIES" ]; then
    echo -e "${RED}✗${NC} Failed to collect latency samples"
    exit 1
fi

# Calculate statistics
LATENCY_ARRAY=($LATENCIES)
COUNT=${#LATENCY_ARRAY[@]}

MIN=${LATENCY_ARRAY[0]}
MAX=${LATENCY_ARRAY[$((COUNT-1))]}
P50=${LATENCY_ARRAY[$((COUNT/2))]}
P95=${LATENCY_ARRAY[$((COUNT*95/100))]}
P99=${LATENCY_ARRAY[$((COUNT*99/100))]}

# Calculate average
SUM=0
for lat in $LATENCIES; do
    SUM=$(awk "BEGIN {print $SUM + $lat}")
done
AVG=$(awk "BEGIN {printf \"%.2f\", $SUM / $COUNT}")

echo "Min:  ${MIN} ms"
echo "P50:  ${P50} ms"
echo "P95:  ${P95} ms"
echo "P99:  ${P99} ms"
echo "Max:  ${MAX} ms"
echo "Avg:  ${AVG} ms"
echo ""

# Latency assessment
echo "Assessment:"
if (( $(echo "$P95 < 20" | bc -l) )); then
    echo -e "${GREEN}✓ Excellent${NC} - Very low latency (<20ms)"
elif (( $(echo "$P95 < 50" | bc -l) )); then
    echo -e "${GREEN}✓ Good${NC} - Low latency (20-50ms)"
elif (( $(echo "$P95 < 100" | bc -l) )); then
    echo -e "${YELLOW}! Fair${NC} - Moderate latency (50-100ms)"
elif (( $(echo "$P95 < 200" | bc -l) )); then
    echo -e "${YELLOW}! Poor${NC} - High latency (100-200ms)"
else
    echo -e "${RED}✗ Critical${NC} - Very high latency (>200ms)"
fi
echo ""

# 3. Jitter Analysis
echo -e "${CYAN}[3] Jitter Analysis (Latency Variability)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Calculate standard deviation
VARIANCE=0
for lat in $LATENCIES; do
    DIFF=$(awk "BEGIN {print $lat - $AVG}")
    DIFF_SQ=$(awk "BEGIN {print $DIFF * $DIFF}")
    VARIANCE=$(awk "BEGIN {print $VARIANCE + $DIFF_SQ}")
done
VARIANCE=$(awk "BEGIN {print $VARIANCE / $COUNT}")
STDDEV=$(awk "BEGIN {printf \"%.2f\", sqrt($VARIANCE)}")

echo "Standard Deviation: ${STDDEV} ms"

# Jitter assessment
if (( $(echo "$STDDEV < 5" | bc -l) )); then
    echo -e "${GREEN}✓ Excellent${NC} - Very consistent latency"
elif (( $(echo "$STDDEV < 10" | bc -l) )); then
    echo -e "${GREEN}✓ Good${NC} - Consistent latency"
elif (( $(echo "$STDDEV < 20" | bc -l) )); then
    echo -e "${YELLOW}! Fair${NC} - Some variability"
else
    echo -e "${RED}✗ Poor${NC} - High variability (network congestion?)"
fi
echo ""

# 4. Packet Loss Test
echo -e "${CYAN}[4] Packet Loss Test${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing with 100 packets, 0.1s interval..."
echo ""

LOSS_TEST=$(ping -c 100 -i 0.1 "$TARGET" 2>&1 | grep 'packet loss')
echo "$LOSS_TEST"

LOSS_PCT=$(echo "$LOSS_TEST" | awk '{print $6}' | sed 's/%//')

if [ "$LOSS_PCT" = "0" ]; then
    echo -e "${GREEN}✓ No packet loss${NC}"
elif (( $(echo "$LOSS_PCT < 1" | bc -l) )); then
    echo -e "${YELLOW}! Minor packet loss${NC} (acceptable for most applications)"
elif (( $(echo "$LOSS_PCT < 5" | bc -l) )); then
    echo -e "${YELLOW}! Moderate packet loss${NC} (may affect real-time applications)"
else
    echo -e "${RED}✗ Significant packet loss${NC} (network issues)"
fi
echo ""

# 5. Per-Hop Latency (if mtr available)
if command -v mtr &> /dev/null; then
    echo -e "${CYAN}[5] Per-Hop Latency (mtr)${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Analyzing route with 10 cycles..."
    echo ""

    mtr --report --report-cycles 10 "$TARGET" 2>/dev/null

    echo ""
    echo "Note: Look for hops with high latency or packet loss"
    echo ""
elif command -v traceroute &> /dev/null; then
    echo -e "${CYAN}[5] Route Analysis (traceroute)${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    traceroute "$TARGET" 2>&1 | head -n 20
    echo ""
else
    echo -e "${YELLOW}! mtr and traceroute not available for per-hop analysis${NC}"
    echo ""
fi

# 6. Latency Over Time
echo -e "${CYAN}[6] Latency Stability Over Time${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing latency for 30 seconds..."
echo ""

# Sample latency every second for 30 seconds
declare -a SAMPLES
for i in {1..30}; do
    LAT=$(ping -c 1 -W 1 "$TARGET" 2>&1 | grep 'time=' | awk -F'time=' '{print $2}' | awk '{print $1}')
    if [ -n "$LAT" ]; then
        SAMPLES+=($LAT)
        # Simple progress indicator
        if [ $((i % 5)) -eq 0 ]; then
            echo -n "."
        fi
    fi
done
echo ""
echo ""

# Check for trends
if [ ${#SAMPLES[@]} -gt 10 ]; then
    FIRST_10_AVG=0
    LAST_10_AVG=0

    for i in {0..9}; do
        FIRST_10_AVG=$(awk "BEGIN {print $FIRST_10_AVG + ${SAMPLES[$i]}}")
    done
    FIRST_10_AVG=$(awk "BEGIN {printf \"%.2f\", $FIRST_10_AVG / 10}")

    START_IDX=$((${#SAMPLES[@]} - 10))
    for i in $(seq $START_IDX $((${#SAMPLES[@]} - 1))); do
        LAST_10_AVG=$(awk "BEGIN {print $LAST_10_AVG + ${SAMPLES[$i]}}")
    done
    LAST_10_AVG=$(awk "BEGIN {printf \"%.2f\", $LAST_10_AVG / 10}")

    echo "First 10 samples avg: ${FIRST_10_AVG} ms"
    echo "Last 10 samples avg:  ${LAST_10_AVG} ms"
    echo ""

    DIFF=$(awk "BEGIN {print $LAST_10_AVG - $FIRST_10_AVG}")
    DIFF_PCT=$(awk "BEGIN {printf \"%.1f\", ($DIFF / $FIRST_10_AVG) * 100}")

    if (( $(echo "$DIFF_PCT > 50" | bc -l) )); then
        echo -e "${RED}✗ Latency increasing significantly${NC} (${DIFF_PCT}% increase)"
        echo "Possible causes: Network congestion, bandwidth saturation"
    elif (( $(echo "$DIFF_PCT < -50" | bc -l) )); then
        echo -e "${GREEN}✓ Latency decreasing${NC} (${DIFF_PCT}% decrease)"
    else
        echo -e "${GREEN}✓ Latency stable${NC}"
    fi
fi
echo ""

# Summary and Recommendations
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Summary and Recommendations${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo "Latency Profile:"
echo "  Min:     ${MIN} ms"
echo "  Average: ${AVG} ms"
echo "  P95:     ${P95} ms"
echo "  Max:     ${MAX} ms"
echo "  Jitter:  ${STDDEV} ms"
echo "  Loss:    ${LOSS_PCT}%"
echo ""

# Recommendations
echo "Recommendations:"

if (( $(echo "$LOSS_PCT > 1" | bc -l) )); then
    echo "  ${RED}✗${NC} Packet Loss:"
    echo "    - Check network hardware (cables, switches)"
    echo "    - Check for bandwidth saturation"
    echo "    - Check for buffer overflow (increase buffers)"
    echo "    - Check for wireless interference"
fi

if (( $(echo "$P95 > 100" | bc -l) )); then
    echo "  ${YELLOW}!${NC} High Latency:"
    echo "    - Check network path (traceroute/mtr)"
    echo "    - Check for geographic distance"
    echo "    - Check for network congestion"
    echo "    - Consider using faster connection (fiber, 10GbE)"
    echo "    - Check for DNS resolution delays"
fi

if (( $(echo "$STDDEV > 20" | bc -l) )); then
    echo "  ${YELLOW}!${NC} High Jitter:"
    echo "    - Check for network congestion"
    echo "    - Check for competing traffic (QoS needed?)"
    echo "    - Check for CPU throttling on endpoints"
    echo "    - Consider traffic shaping/QoS"
fi

if (( $(echo "$P95 < 50" | bc -l) )) && [ "$LOSS_PCT" = "0" ] && (( $(echo "$STDDEV < 10" | bc -l) )); then
    echo -e "  ${GREEN}✓${NC} Network performance is excellent!"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
