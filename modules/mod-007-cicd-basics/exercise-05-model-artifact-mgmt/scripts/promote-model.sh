#!/bin/bash
# Promote model to new stage

set -euo pipefail

# Parse arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <model-name> <version> <stage>"
    echo "Example: $0 RandomForestClassifier 5 Production"
    exit 1
fi

MODEL_NAME=$1
VERSION=$2
STAGE=$3

echo "Promoting model..."
echo "Model: $MODEL_NAME"
echo "Version: $VERSION"
echo "Stage: $STAGE"

# Confirm if promoting to production
if [ "$STAGE" == "Production" ]; then
    read -p "Are you sure you want to promote to Production? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Promotion cancelled"
        exit 0
    fi
fi

# Promote model
python mlflow/promote_model.py \
    --model-name "$MODEL_NAME" \
    --version "$VERSION" \
    --stage "$STAGE"

echo "✓ Model promoted successfully!"
