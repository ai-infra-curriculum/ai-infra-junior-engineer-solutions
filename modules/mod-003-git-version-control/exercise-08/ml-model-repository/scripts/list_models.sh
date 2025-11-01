#!/bin/bash
# List all model versions

echo "=== Model Versions (Git Tags) ==="
git tag -l "model-*" | sort -V

echo ""
echo "=== Production Models ==="
ls -lh models/production/*.onnx 2>/dev/null || echo "No models downloaded yet"

echo ""
echo "=== Model Metadata ==="
ls -1 models/production/*.yaml
