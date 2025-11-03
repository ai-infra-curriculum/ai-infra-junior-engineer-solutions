#!/bin/bash
#
# extract_metrics.sh - Extract Training Metrics from Logs
#
# Description:
#   Extracts structured metrics from training logs and exports to CSV format.
#   Calculates statistics and generates summaries.
#

set -euo pipefail

LOG_FILE="${1:-../sample_logs/training.log}"
OUTPUT_CSV="${2:-training_metrics.csv}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Extracting training metrics from: $LOG_FILE${NC}"
echo ""

# Create CSV header
echo "epoch,loss,accuracy,val_loss,val_accuracy" > "$OUTPUT_CSV"

# Extract metrics using awk
awk '/Epoch [0-9]+\/[0-9]+/ {
    match($0, /Epoch ([0-9]+)/, epoch);
    match($0, /loss: ([0-9.]+)/, loss);
    match($0, /accuracy: ([0-9.]+)/, acc);
    match($0, /val_loss: ([0-9.]+)/, val_loss);
    match($0, /val_accuracy: ([0-9.]+)/, val_acc);

    if (epoch[1] && loss[1] && acc[1] && val_loss[1] && val_acc[1]) {
        printf "%d,%.4f,%.4f,%.4f,%.4f\n",
            epoch[1], loss[1], acc[1], val_loss[1], val_acc[1]
    }
}' "$LOG_FILE" >> "$OUTPUT_CSV"

echo -e "${GREEN}✓ Metrics saved to: $OUTPUT_CSV${NC}"
echo ""

# Display extracted metrics
echo "Extracted Metrics:"
cat "$OUTPUT_CSV"
echo ""

# Generate statistics
echo -e "${BLUE}=== Training Statistics ===${NC}"
awk -F',' 'NR>1 {
    loss_sum += $2;
    acc_sum += $3;
    val_loss_sum += $4;
    val_acc_sum += $5;
    count++;

    if (NR == 2) {
        first_loss = $2;
        first_acc = $3;
        best_loss = $2;
        best_acc = $3;
    }

    last_loss = $2;
    last_acc = $3;

    if ($2 < best_loss) best_loss = $2;
    if ($3 > best_acc) best_acc = $3;
}
END {
    if (count > 0) {
        printf "Total epochs: %d\n", count;
        printf "\nLoss:\n";
        printf "  First: %.4f\n", first_loss;
        printf "  Last: %.4f\n", last_loss;
        printf "  Best: %.4f\n", best_loss;
        printf "  Average: %.4f\n", loss_sum/count;
        printf "  Improvement: %.4f (%.1f%%)\n",
            first_loss - last_loss,
            (first_loss - last_loss) / first_loss * 100;

        printf "\nAccuracy:\n";
        printf "  First: %.4f\n", first_acc;
        printf "  Last: %.4f\n", last_acc;
        printf "  Best: %.4f\n", best_acc;
        printf "  Average: %.4f\n", acc_sum/count;
        printf "  Improvement: %.4f (%.1f%%)\n",
            last_acc - first_acc,
            (last_acc - first_acc) / first_acc * 100;

        printf "\nValidation Metrics:\n";
        printf "  Average Val Loss: %.4f\n", val_loss_sum/count;
        printf "  Average Val Accuracy: %.4f\n", val_acc_sum/count;
    }
}' "$OUTPUT_CSV"

echo ""

# Check for overfitting
echo -e "${BLUE}=== Overfitting Detection ===${NC}"
awk -F',' 'NR>1 {
    train_loss = $2;
    val_loss = $4;
    diff = val_loss - train_loss;

    if (diff > 0.1) {
        printf "Epoch %d: Val loss (%.4f) > Train loss (%.4f) by %.4f - possible overfitting\n",
            $1, val_loss, train_loss, diff
    }
}' "$OUTPUT_CSV"

echo ""

# Generate training summary
echo -e "${BLUE}=== Training Summary ===${NC}"

# Extract training duration
if grep -q "Training completed" "$LOG_FILE"; then
    duration=$(grep "Training completed" "$LOG_FILE" | grep -oP '\d+ minutes \d+ seconds')
    echo "Training duration: $duration"
fi

# Extract early stopping info
if grep -q "Early stopping" "$LOG_FILE"; then
    echo -e "${YELLOW}Early stopping was triggered${NC}"
fi

# Extract checkpoint info
checkpoint_count=$(grep -c "Checkpoint saved" "$LOG_FILE" || echo 0)
echo "Checkpoints saved: $checkpoint_count"

# Extract final model location
if grep -q "Best model saved" "$LOG_FILE"; then
    model_path=$(grep "Best model saved" "$LOG_FILE" | awk '{print $NF}')
    echo "Final model location: $model_path"
fi

echo ""
echo -e "${GREEN}✓ Metric extraction complete!${NC}"
echo ""
echo "To visualize metrics, run:"
echo "  python3 ../scripts/visualize_metrics.py"
