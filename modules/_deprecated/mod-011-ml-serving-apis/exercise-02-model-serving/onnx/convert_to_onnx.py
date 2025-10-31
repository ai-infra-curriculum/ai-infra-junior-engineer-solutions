"""Convert ML models to ONNX format."""

import numpy as np
import onnx
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import torch
import torch.nn as nn


def convert_sklearn_to_onnx(model_path="model_sklearn.onnx"):
    """Convert scikit-learn model to ONNX."""
    print("Converting scikit-learn model to ONNX...")

    # Train a simple model
    X = np.random.rand(1000, 5).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)

    # Define input type
    initial_type = [('float_input', FloatTensorType([None, 5]))]

    # Convert to ONNX
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=13,
        options={id(model): {'zipmap': False}}
    )

    # Save model
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✓ Model saved to {model_path}")

    # Verify
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")

    return model_path


def convert_pytorch_to_onnx(model_path="model_pytorch.onnx"):
    """Convert PyTorch model to ONNX."""
    print("\nConverting PyTorch model to ONNX...")

    # Define a simple PyTorch model
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create and train model
    model = SimpleNN()
    model.eval()

    # Dummy input for export
    dummy_input = torch.randn(1, 5)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✓ Model saved to {model_path}")

    # Verify
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")

    return model_path


def print_model_info(model_path):
    """Print ONNX model information."""
    print(f"\nModel Info: {model_path}")
    print("-" * 50)

    model = onnx.load(model_path)

    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    print(f"Opset: {model.opset_import[0].version}")

    print("\nInputs:")
    for input in model.graph.input:
        print(f"  {input.name}: {input.type}")

    print("\nOutputs:")
    for output in model.graph.output:
        print(f"  {output.name}: {output.type}")


if __name__ == "__main__":
    print("="*50)
    print("ONNX Model Conversion")
    print("="*50)

    # Convert scikit-learn model
    sklearn_path = convert_sklearn_to_onnx()
    print_model_info(sklearn_path)

    # Convert PyTorch model
    pytorch_path = convert_pytorch_to_onnx()
    print_model_info(pytorch_path)

    print("\n" + "="*50)
    print("Conversion complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Run inference: python onnx_inference.py")
    print("2. Optimize model: python optimize.py")
    print("3. Benchmark: python ../benchmarks/benchmark_all.py")
