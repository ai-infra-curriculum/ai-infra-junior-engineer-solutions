#!/bin/bash
# Train model and log to MLflow

set -euo pipefail

# Default values
DATA_PATH="data/dataset.csv"
MODEL_TYPE="random_forest"
EXPERIMENT_NAME="production-training"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Training model with MLflow tracking..."
echo "Data: $DATA_PATH"
echo "Model: $MODEL_TYPE"
echo "Experiment: $EXPERIMENT_NAME"

# Train model
python mlflow/train.py \
    --data-path "$DATA_PATH" \
    --model-type "$MODEL_TYPE" \
    --experiment-name "$EXPERIMENT_NAME" \
    --n-estimators 100 \
    --max-depth 10

echo "✓ Training complete!"
