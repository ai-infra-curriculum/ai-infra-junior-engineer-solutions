#!/bin/bash
# Prometheus Cardinality Analysis
# Checks metric cardinality to identify high-cardinality metrics that impact performance

set -e

# Configuration
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
OUTPUT_FILE="${OUTPUT_FILE:-cardinality_report.txt}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Prometheus Cardinality Analysis"
echo "=========================================="
echo "Prometheus: $PROMETHEUS_URL"
echo "Timestamp: $(date)"
echo ""

# Clear previous report
> "$OUTPUT_FILE"

# Function to query Prometheus
prom_query() {
    local query="$1"
    curl -s -G --data-urlencode "query=$query" "$PROMETHEUS_URL/api/v1/query" | \
        grep -o '"result":\[.*\]' | \
        grep -o '\[.*\]' || echo "[]"
}

# Function to extract value from result
extract_value() {
    local result="$1"
    echo "$result" | grep -o '"value":\[[^]]*\]' | grep -o '[0-9.]*' | tail -1
}

echo "=========================================="
echo "1. Overall Cardinality Metrics"
echo "=========================================="

# Total number of time series
echo -n "Counting total time series... "
total_series_query='count({__name__=~".+"})'
total_series_result=$(prom_query "$total_series_query")
total_series=$(extract_value "$total_series_result")

if [ -n "$total_series" ]; then
    echo -e "${GREEN}${total_series}${NC}"
    echo "Total time series: $total_series" >> "$OUTPUT_FILE"

    # Interpret results
    if (( $(echo "$total_series < 100000" | bc -l) )); then
        echo -e "  Status: ${GREEN}GOOD${NC} (<100k series)"
    elif (( $(echo "$total_series < 500000" | bc -l) )); then
        echo -e "  Status: ${YELLOW}ACCEPTABLE${NC} (100k-500k series)"
    else
        echo -e "  Status: ${RED}PROBLEM${NC} (>500k series) - Consider reducing cardinality"
    fi
else
    echo -e "${RED}ERROR${NC} - Could not query total series"
fi

echo ""

# Number of unique metrics
echo -n "Counting unique metric names... "
unique_metrics_query='count(count by (__name__)({__name__=~".+"}))'
unique_metrics_result=$(prom_query "$unique_metrics_query")
unique_metrics=$(extract_value "$unique_metrics_result")

if [ -n "$unique_metrics" ]; then
    echo -e "${GREEN}${unique_metrics}${NC}"
    echo "Unique metrics: $unique_metrics" >> "$OUTPUT_FILE"
else
    echo -e "${RED}ERROR${NC}"
fi

echo ""

# Average series per metric
if [ -n "$total_series" ] && [ -n "$unique_metrics" ] && [ "$unique_metrics" != "0" ]; then
    avg_series_per_metric=$(echo "scale=2; $total_series / $unique_metrics" | bc)
    echo "Average series per metric: ${avg_series_per_metric}"
    echo "Average series per metric: $avg_series_per_metric" >> "$OUTPUT_FILE"
fi

echo ""
echo "=========================================="
echo "2. Top Metrics by Cardinality"
echo "=========================================="

# Top 20 metrics by time series count
echo "Finding top 20 metrics by cardinality..."
echo "" >> "$OUTPUT_FILE"
echo "Top 20 Metrics by Cardinality:" >> "$OUTPUT_FILE"
echo "================================" >> "$OUTPUT_FILE"

# This query groups by metric name and counts series
top_metrics_query='topk(20, count by (__name__)({__name__=~".+"}))'

# Get the metric names and their counts
top_metrics=$(curl -s -G --data-urlencode "query=$top_metrics_query" "$PROMETHEUS_URL/api/v1/query")

# Parse the results (simplified parsing)
echo "$top_metrics" | grep -o '"metric":{"__name__":"[^"]*"}' | \
    grep -o '"__name__":"[^"]*"' | \
    cut -d'"' -f4 | \
    head -20 | \
    while read -r metric_name; do
        # Count series for this metric
        count_query="count({__name__=\"$metric_name\"})"
        count_result=$(prom_query "$count_query")
        count=$(extract_value "$count_result")

        if [ -n "$count" ]; then
            printf "%-50s %10s series\n" "$metric_name" "$count"
            printf "%-50s %10s series\n" "$metric_name" "$count" >> "$OUTPUT_FILE"

            # Warning if too high
            if (( $(echo "$count > 10000" | bc -l) )); then
                echo -e "  ${RED}⚠ HIGH CARDINALITY${NC} - Consider aggregation or relabeling"
            fi
        fi
    done

echo ""
echo "=========================================="
echo "3. ML Platform Metrics Analysis"
echo "=========================================="

# Check cardinality of key ML metrics
ml_metrics=(
    "model_predictions_total"
    "model_prediction_errors_total"
    "model_prediction_duration_seconds_bucket"
    "model_cache_hits_total"
    "model_cache_misses_total"
)

echo "" >> "$OUTPUT_FILE"
echo "ML Platform Metrics:" >> "$OUTPUT_FILE"
echo "====================" >> "$OUTPUT_FILE"

for metric in "${ml_metrics[@]}"; do
    echo -n "Checking $metric... "
    count_query="count({__name__=\"$metric\"})"
    count_result=$(prom_query "$count_query")
    count=$(extract_value "$count_result")

    if [ -n "$count" ] && [ "$count" != "0" ]; then
        echo -e "${GREEN}${count} series${NC}"
        printf "%-50s %10s series\n" "$metric" "$count" >> "$OUTPUT_FILE"

        # Check labels
        labels_query="count(count by (model) ({__name__=\"$metric\"}))"
        labels_result=$(prom_query "$labels_query")
        model_count=$(extract_value "$labels_result")

        if [ -n "$model_count" ]; then
            echo "  └─ $model_count unique models"

            # Check for high cardinality labels
            if (( $(echo "$count / $model_count > 50" | bc -l 2>/dev/null || echo 0) )); then
                echo -e "  └─ ${YELLOW}⚠ Many series per model - check for high-cardinality labels${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Not found${NC}"
    fi
done

echo ""
echo "=========================================="
echo "4. Recording Rules Efficiency"
echo "=========================================="

# Check if recording rules are reducing cardinality
recording_rules=(
    "model:predictions:rate5m"
    "model:latency:p95"
    "model:error_ratio:rate5m"
    "platform:predictions:rate5m"
)

echo "" >> "$OUTPUT_FILE"
echo "Recording Rules Cardinality:" >> "$OUTPUT_FILE"
echo "============================" >> "$OUTPUT_FILE"

total_recording_series=0
for rule in "${recording_rules[@]}"; do
    echo -n "Checking $rule... "
    count_query="count({__name__=\"$rule\"})"
    count_result=$(prom_query "$count_query")
    count=$(extract_value "$count_result")

    if [ -n "$count" ] && [ "$count" != "0" ]; then
        echo -e "${GREEN}${count} series${NC}"
        printf "%-50s %10s series\n" "$rule" "$count" >> "$OUTPUT_FILE"
        total_recording_series=$((total_recording_series + ${count%.*}))
    else
        echo -e "${YELLOW}Not configured${NC}"
    fi
done

if [ "$total_recording_series" -gt 0 ]; then
    echo ""
    echo "Total recording rule series: $total_recording_series"
    echo "Total recording rule series: $total_recording_series" >> "$OUTPUT_FILE"
    echo -e "${GREEN}✓ Recording rules active${NC}"
fi

echo ""
echo "=========================================="
echo "5. Label Cardinality Analysis"
echo "=========================================="

# Check cardinality of common labels
labels=("model" "version" "status_code" "endpoint")

echo "" >> "$OUTPUT_FILE"
echo "Label Cardinality:" >> "$OUTPUT_FILE"
echo "==================" >> "$OUTPUT_FILE"

for label in "${labels[@]}"; do
    echo -n "Checking label: $label... "

    # Count unique values for this label
    label_query="count(count by ($label)({__name__=~\"model.*\"}))"
    label_result=$(prom_query "$label_query")
    label_count=$(extract_value "$label_result")

    if [ -n "$label_count" ] && [ "$label_count" != "0" ]; then
        echo -e "${GREEN}${label_count} unique values${NC}"
        printf "%-20s %10s unique values\n" "$label" "$label_count" >> "$OUTPUT_FILE"

        # Warning for high cardinality labels
        if (( $(echo "$label_count > 100" | bc -l) )); then
            echo -e "  ${RED}⚠ HIGH CARDINALITY LABEL${NC} - Each value multiplies series count"
        elif (( $(echo "$label_count > 50" | bc -l) )); then
            echo -e "  ${YELLOW}⚠ MODERATE CARDINALITY${NC} - Monitor growth"
        fi
    else
        echo -e "${YELLOW}Not found${NC}"
    fi
done

echo ""
echo "=========================================="
echo "6. Storage and Performance Impact"
echo "=========================================="

if [ -n "$total_series" ]; then
    # Estimate storage (rough approximation)
    # Assume ~1-2 KB per series per day
    storage_per_day_mb=$(echo "scale=2; $total_series * 1.5 / 1024" | bc)
    storage_per_month_gb=$(echo "scale=2; $storage_per_day_mb * 30 / 1024" | bc)

    echo "" >> "$OUTPUT_FILE"
    echo "Storage Estimates:" >> "$OUTPUT_FILE"
    echo "==================" >> "$OUTPUT_FILE"

    echo "Estimated storage usage:"
    echo "  • Per day:   ~${storage_per_day_mb} MB"
    echo "  • Per month: ~${storage_per_month_gb} GB"

    echo "Estimated storage per day: ${storage_per_day_mb} MB" >> "$OUTPUT_FILE"
    echo "Estimated storage per month: ${storage_per_month_gb} GB" >> "$OUTPUT_FILE"

    # Memory estimate (rough)
    # Assume ~3 KB per series in memory
    memory_mb=$(echo "scale=2; $total_series * 3 / 1024" | bc)
    echo "  • Memory:    ~${memory_mb} MB"
    echo "Estimated memory usage: ${memory_mb} MB" >> "$OUTPUT_FILE"

    if (( $(echo "$memory_mb > 2048" | bc -l) )); then
        echo -e "  ${RED}⚠ High memory usage${NC} - Consider scaling vertically or reducing cardinality"
    fi
fi

echo ""
echo "=========================================="
echo "7. Recommendations"
echo "=========================================="

echo "" >> "$OUTPUT_FILE"
echo "Recommendations:" >> "$OUTPUT_FILE"
echo "================" >> "$OUTPUT_FILE"

recommendations=()

# Check if cardinality is high
if [ -n "$total_series" ] && (( $(echo "$total_series > 500000" | bc -l) )); then
    recommendations+=("🔴 CRITICAL: Reduce overall cardinality (<500k series)")
    recommendations+=("   • Drop unused metrics via relabeling")
    recommendations+=("   • Aggregate high-cardinality labels")
    recommendations+=("   • Use recording rules to pre-aggregate")
fi

# Check if recording rules are helping
if [ "$total_recording_series" -eq 0 ]; then
    recommendations+=("🟡 Deploy recording rules to improve query performance")
    recommendations+=("   • See kubernetes/recording-rules.yaml")
fi

# Check for high-cardinality metrics
high_card_count=$(echo "$top_metrics" | grep -c "10000" || echo "0")
if [ "$high_card_count" -gt 0 ]; then
    recommendations+=("🟡 Found metrics with >10k series - review and optimize")
    recommendations+=("   • Consider dropping unnecessary labels")
    recommendations+=("   • Use relabeling to reduce dimensions")
fi

if [ ${#recommendations[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ No major issues detected${NC}"
    echo "✓ Cardinality is within acceptable limits"
    echo "✓ Recording rules are active"
    echo "✓ Continue monitoring as metrics grow"

    echo "✓ No major issues detected" >> "$OUTPUT_FILE"
else
    for rec in "${recommendations[@]}"; do
        echo "$rec"
        echo "$rec" >> "$OUTPUT_FILE"
    done
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Analysis Complete!${NC}"
echo "=========================================="
echo "Full report saved to: $OUTPUT_FILE"
echo ""
echo "For more details, run:"
echo "  • curl $PROMETHEUS_URL/tsdb-status"
echo "  • curl $PROMETHEUS_URL/api/v1/status/tsdb"
echo ""
