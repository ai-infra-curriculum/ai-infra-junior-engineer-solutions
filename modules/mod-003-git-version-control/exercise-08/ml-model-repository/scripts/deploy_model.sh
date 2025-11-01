#!/bin/bash
# Deploy specific model version

if [ -z "$1" ]; then
    echo "Usage: ./deploy_model.sh <model-tag>"
    echo "Example: ./deploy_model.sh model-bert-v1.1.0"
    exit 1
fi

MODEL_TAG=$1

echo "Deploying model: $MODEL_TAG"

# Checkout model version
git checkout $MODEL_TAG

# Download model files
echo "Downloading model files..."
git lfs pull --include="models/production/*.onnx"

# Verify model exists
MODEL_FILE=$(ls models/production/*.onnx 2>/dev/null | head -1)
if [ -z "$MODEL_FILE" ]; then
    echo "Error: No model file found"
    exit 1
fi

echo "Model ready: $MODEL_FILE"
echo "Size: $(du -h $MODEL_FILE | cut -f1)"
echo ""
echo "To deploy to Kubernetes:"
echo "  kubectl apply -f deployment/${MODEL_TAG}.yaml"
