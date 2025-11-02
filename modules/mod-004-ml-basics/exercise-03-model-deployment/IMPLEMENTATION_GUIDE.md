# IMPLEMENTATION GUIDE: Model Format Conversion (ONNX)

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [PyTorch to ONNX Conversion](#pytorch-to-onnx-conversion)
- [TensorFlow to ONNX Conversion](#tensorflow-to-onnx-conversion)
- [ONNX Model Inspection](#onnx-model-inspection)
- [ONNX Runtime Inference](#onnx-runtime-inference)
- [Performance Comparison](#performance-comparison)
- [Production Converter Tool](#production-converter-tool)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Best Practices](#best-practices)
- [Validation Checklist](#validation-checklist)

---

## Overview

### What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open-source format for representing machine learning models. It enables interoperability between different ML frameworks, allowing you to:

- Train models in PyTorch and deploy with TensorFlow Serving
- Convert models to optimized formats for edge devices
- Use framework-agnostic serving infrastructure
- Leverage hardware-specific acceleration (NVIDIA TensorRT, Intel OpenVINO, Apple CoreML)

### Why Infrastructure Engineers Need ONNX

```
Training Framework    →    ONNX Format    →    Deployment Target
─────────────────────      ───────────         ─────────────────
PyTorch              ─┐                    ┌→  ONNX Runtime (CPU)
TensorFlow           ─┤→   Universal      ─┤→  TensorRT (GPU)
Keras                ─┤    Representation  ┤→  OpenVINO (Intel)
scikit-learn         ─┘                    ├→  CoreML (Apple)
                                           ├→  Web browsers
                                           └→  Mobile devices
```

### Key Benefits

1. **Framework Independence**: Deploy without framework lock-in
2. **Performance Optimization**: ONNX Runtime often outperforms native frameworks
3. **Reduced Dependencies**: Smaller deployment footprint
4. **Hardware Acceleration**: Seamless integration with specialized hardware
5. **Cost Reduction**: Faster inference = lower cloud costs

---

## Environment Setup

### Step 1: Install Core Dependencies

```bash
# Create isolated environment
python3 -m venv onnx-conversion-env
source onnx-conversion-env/bin/activate  # Windows: onnx-conversion-env\Scripts\activate

# Install PyTorch (CPU version for simplicity)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow
pip install tensorflow==2.14.0

# Install ONNX ecosystem
pip install onnx==1.15.0 \
            onnxruntime==1.16.3 \
            onnxruntime-gpu==1.16.3  # Only if CUDA available

# Install TensorFlow to ONNX converter
pip install tf2onnx==1.15.1

# Install utilities
pip install numpy==1.24.3 \
            pillow==10.0.0 \
            matplotlib==3.7.2 \
            psutil==5.9.5
```

### Step 2: Verify Installation

Create `verify_environment.py`:

```python
#!/usr/bin/env python3
"""
Verify ONNX conversion environment setup
"""
import sys

def check_imports():
    """Check all required imports"""
    print("=" * 70)
    print("ONNX Conversion Environment Verification")
    print("=" * 70)

    checks = {
        'PyTorch': 'torch',
        'TorchVision': 'torchvision',
        'TensorFlow': 'tensorflow',
        'ONNX': 'onnx',
        'ONNX Runtime': 'onnxruntime',
        'TF2ONNX': 'tf2onnx',
        'NumPy': 'numpy',
        'Pillow': 'PIL'
    }

    failed = []

    for name, module in checks.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:<20} {version}")
        except ImportError as e:
            print(f"✗ {name:<20} NOT INSTALLED")
            failed.append(name)

    return len(failed) == 0

def check_versions():
    """Check framework versions and compatibility"""
    print("\n" + "=" * 70)
    print("Framework Details")
    print("=" * 70)

    import torch
    import tensorflow as tf
    import onnxruntime as ort

    print(f"\nPyTorch:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")

    print(f"\nTensorFlow:")
    print(f"  Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPU Count: {len(gpus)}")

    print(f"\nONNX Runtime:")
    print(f"  Version: {ort.__version__}")
    print(f"  Available Providers:")
    for provider in ort.get_available_providers():
        print(f"    - {provider}")

def check_onnx_features():
    """Test basic ONNX functionality"""
    print("\n" + "=" * 70)
    print("ONNX Functionality Tests")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
        import onnx
        import onnxruntime as ort
        import tempfile
        import os

        # Create simple model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = TestModel()
        model.eval()

        # Export to ONNX
        dummy_input = torch.randn(1, 10)
        temp_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        torch.onnx.export(
            model, dummy_input, temp_path,
            input_names=['input'],
            output_names=['output']
        )

        print("✓ PyTorch to ONNX export: SUCCESS")

        # Load and validate
        onnx_model = onnx.load(temp_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation: SUCCESS")

        # Run inference
        session = ort.InferenceSession(temp_path)
        result = session.run(None, {'input': dummy_input.numpy()})
        print("✓ ONNX Runtime inference: SUCCESS")

        # Cleanup
        os.remove(temp_path)

        return True

    except Exception as e:
        print(f"✗ ONNX functionality test FAILED: {str(e)}")
        return False

def main():
    """Run all verification checks"""
    print("\nStarting environment verification...\n")

    success = True

    # Check imports
    if not check_imports():
        print("\n⚠ Some packages are missing. Install them before proceeding.")
        success = False

    # Check versions
    try:
        check_versions()
    except Exception as e:
        print(f"\n⚠ Version check failed: {str(e)}")
        success = False

    # Check ONNX features
    if not check_onnx_features():
        success = False

    # Final summary
    print("\n" + "=" * 70)
    if success:
        print("✓ Environment verification PASSED")
        print("  You are ready to proceed with ONNX conversion exercises!")
    else:
        print("✗ Environment verification FAILED")
        print("  Please fix the issues above before continuing.")
    print("=" * 70)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

Run verification:

```bash
python verify_environment.py
```

**Expected Output:**
```
======================================================================
ONNX Conversion Environment Verification
======================================================================
✓ PyTorch              2.1.0+cpu
✓ TorchVision          0.16.0+cpu
✓ TensorFlow           2.14.0
✓ ONNX                 1.15.0
✓ ONNX Runtime         1.16.3
✓ TF2ONNX              1.15.1
✓ NumPy                1.24.3
✓ Pillow               10.0.0

======================================================================
Framework Details
======================================================================

PyTorch:
  Version: 2.1.0+cpu
  CUDA Available: False

TensorFlow:
  Version: 2.14.0
  GPU Count: 0

ONNX Runtime:
  Version: 1.16.3
  Available Providers:
    - CPUExecutionProvider

======================================================================
ONNX Functionality Tests
======================================================================
✓ PyTorch to ONNX export: SUCCESS
✓ ONNX model validation: SUCCESS
✓ ONNX Runtime inference: SUCCESS

======================================================================
✓ Environment verification PASSED
  You are ready to proceed with ONNX conversion exercises!
======================================================================
```

### Step 3: Create Project Structure

```bash
mkdir -p onnx-conversion/{models,scripts,data,reports}
cd onnx-conversion

# Download test resources
wget https://github.com/pytorch/hub/raw/master/images/dog.jpg -O data/dog.jpg
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt -O data/imagenet_classes.txt
```

---

## PyTorch to ONNX Conversion

### Understanding torch.onnx.export()

The `torch.onnx.export()` function is the primary interface for converting PyTorch models:

```python
torch.onnx.export(
    model,              # PyTorch model (must be in eval mode)
    args,               # Example inputs for tracing
    f,                  # Output file path
    export_params=True, # Include trained weights
    opset_version=14,   # ONNX opset version
    do_constant_folding=True,  # Optimize constant expressions
    input_names=None,   # Input tensor names
    output_names=None,  # Output tensor names
    dynamic_axes=None,  # Variable dimension axes
    verbose=False       # Print export details
)
```

### Simple Model Conversion

Create `scripts/pytorch_basic_export.py`:

```python
#!/usr/bin/env python3
"""
Basic PyTorch to ONNX conversion example
"""
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

class SimpleClassifier(nn.Module):
    """Simple feedforward classifier"""

    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def export_model():
    """Export PyTorch model to ONNX"""

    print("=" * 70)
    print("PyTorch to ONNX Conversion - Simple Model")
    print("=" * 70)

    # Create model
    print("\n1. Creating PyTorch model...")
    model = SimpleClassifier(input_size=784, hidden_size=256, num_classes=10)
    model.eval()

    # Create dummy input (batch_size=1, features=784)
    print("2. Preparing dummy input...")
    batch_size = 1
    dummy_input = torch.randn(batch_size, 784)
    print(f"   Input shape: {dummy_input.shape}")

    # Run PyTorch inference for comparison
    print("3. Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)
    print(f"   PyTorch output shape: {pytorch_output.shape}")

    # Export to ONNX
    output_path = "models/simple_classifier.onnx"
    print(f"\n4. Exporting to ONNX: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("   ✓ Export complete")

    # Verify ONNX model
    print("\n5. Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("   ✓ ONNX model is valid")

    # Display model info
    print("\n6. Model Information:")
    print(f"   Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
    print(f"   IR Version: {onnx_model.ir_version}")
    print(f"   Opset Version: {onnx_model.opset_import[0].version}")
    print(f"   Graph Nodes: {len(onnx_model.graph.node)}")

    # Test ONNX Runtime inference
    print("\n7. Testing ONNX Runtime inference...")
    session = ort.InferenceSession(output_path)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"   Input name: {input_name}")
    print(f"   Output name: {output_name}")

    # Run inference
    onnx_output = session.run([output_name], {input_name: dummy_input.numpy()})[0]
    print(f"   ONNX output shape: {onnx_output.shape}")

    # Compare outputs
    print("\n8. Comparing outputs...")
    pytorch_np = pytorch_output.detach().numpy()
    max_diff = np.abs(pytorch_np - onnx_output).max()
    mean_diff = np.abs(pytorch_np - onnx_output).mean()

    print(f"   Max difference: {max_diff:.6e}")
    print(f"   Mean difference: {mean_diff:.6e}")

    if np.allclose(pytorch_np, onnx_output, rtol=1e-5, atol=1e-5):
        print("   ✓ Outputs match (within tolerance)")
    else:
        print("   ✗ Outputs differ significantly")

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)

if __name__ == "__main__":
    export_model()
```

### Pre-trained Model Conversion

Create `scripts/pytorch_pretrained_export.py`:

```python
#!/usr/bin/env python3
"""
Convert pre-trained PyTorch models to ONNX
"""
import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path):
    """Preprocess image for ImageNet models"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def export_resnet50():
    """Export ResNet50 to ONNX"""

    print("=" * 70)
    print("Exporting ResNet50 to ONNX")
    print("=" * 70)

    # Load pre-trained model
    print("\n1. Loading pre-trained ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    print("   ✓ Model loaded")

    # Create dummy input
    print("\n2. Creating dummy input...")
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"   Input shape: {dummy_input.shape}")

    # Export to ONNX
    output_path = "models/resnet50.onnx"
    print(f"\n3. Exporting to {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✓ Export complete ({file_size_mb:.2f} MB)")

    # Verify model
    print("\n4. Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ✓ Valid ONNX model")
    print(f"   ✓ Graph has {len(onnx_model.graph.node)} nodes")

    return output_path

def test_inference(onnx_path, image_path, labels_path):
    """Test ONNX model inference"""

    print("\n" + "=" * 70)
    print("Testing ONNX Inference")
    print("=" * 70)

    # Load labels
    print("\n1. Loading ImageNet labels...")
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"   ✓ Loaded {len(labels)} labels")

    # Load ONNX model
    print("\n2. Loading ONNX model...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    print(f"   ✓ Model loaded (input: {input_name})")

    # Preprocess image
    print(f"\n3. Preprocessing image: {image_path}")
    input_tensor = preprocess_image(image_path)
    print(f"   ✓ Image tensor shape: {input_tensor.shape}")

    # Run inference
    print("\n4. Running inference...")
    outputs = session.run(None, {input_name: input_tensor.numpy()})
    logits = outputs[0][0]

    # Get top-5 predictions
    top5_indices = np.argsort(logits)[-5:][::-1]

    # Apply softmax
    exp_logits = np.exp(logits - logits.max())
    probabilities = exp_logits / exp_logits.sum()

    print("\n5. Top-5 Predictions:")
    for i, idx in enumerate(top5_indices):
        print(f"   {i+1}. {labels[idx]:<35} {probabilities[idx]*100:6.2f}%")

    print("\n" + "=" * 70)

def main():
    """Main execution"""

    # Export model
    onnx_path = export_resnet50()

    # Test inference if test data exists
    image_path = "data/dog.jpg"
    labels_path = "data/imagenet_classes.txt"

    if os.path.exists(image_path) and os.path.exists(labels_path):
        test_inference(onnx_path, image_path, labels_path)
    else:
        print("\n⚠ Test data not found. Skipping inference test.")
        print(f"  Download test image to: {image_path}")
        print(f"  Download labels to: {labels_path}")

if __name__ == "__main__":
    main()
```

### Advanced Export Options

Create `scripts/pytorch_advanced_export.py`:

```python
#!/usr/bin/env python3
"""
Advanced PyTorch to ONNX export options
"""
import torch
import torch.nn as nn
import onnx

class CustomModel(nn.Module):
    """Model with multiple inputs/outputs"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.decoder1 = nn.Linear(20, 5)
        self.decoder2 = nn.Linear(20, 3)

    def forward(self, x, use_decoder2=False):
        encoded = self.encoder(x)
        output1 = self.decoder1(encoded)

        if use_decoder2:
            output2 = self.decoder2(encoded)
            return output1, output2
        return output1

def export_with_multiple_outputs():
    """Export model with multiple outputs"""

    print("Exporting model with multiple outputs...")

    model = CustomModel()
    model.eval()

    # Dummy inputs
    x = torch.randn(1, 10)

    # Export with both outputs
    torch.onnx.export(
        model,
        (x, True),  # Tuple of arguments
        "models/multi_output.onnx",
        input_names=['features', 'use_decoder2'],
        output_names=['output1', 'output2'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'output1': {0: 'batch_size'},
            'output2': {0: 'batch_size'}
        }
    )

    print("✓ Multi-output model exported")

def export_with_control_flow():
    """Export model with conditional logic"""

    class ConditionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(10, 20)

        def forward(self, x, condition):
            # ONNX export handles conditional logic
            if condition.item() > 0:
                return self.linear1(x)
            else:
                return self.linear2(x)

    print("\nExporting model with control flow...")

    model = ConditionalModel()
    model.eval()

    x = torch.randn(1, 10)
    condition = torch.tensor([1.0])

    torch.onnx.export(
        model,
        (x, condition),
        "models/conditional.onnx",
        input_names=['input', 'condition'],
        output_names=['output']
    )

    print("✓ Conditional model exported")

def export_with_custom_operators():
    """Handle custom operators during export"""

    print("\nExporting model with custom operators...")
    print("⚠ Custom operators require symbolic function registration")
    print("  See PyTorch documentation for torch.onnx.register_custom_op_symbolic()")

    # Example structure (simplified):
    # @torch.onnx.symbolic_helper.parse_args('v', 'v')
    # def custom_op_symbolic(g, input1, input2):
    #     return g.op("CustomDomain::CustomOp", input1, input2)

if __name__ == "__main__":
    export_with_multiple_outputs()
    export_with_control_flow()
    export_with_custom_operators()
```

---

## TensorFlow to ONNX Conversion

### Using tf2onnx for Conversion

Create `scripts/tensorflow_basic_export.py`:

```python
#!/usr/bin/env python3
"""
TensorFlow/Keras to ONNX conversion
"""
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
import numpy as np

def create_keras_model():
    """Create a simple Keras model"""

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model

def export_keras_direct():
    """Export Keras model directly to ONNX"""

    print("=" * 70)
    print("TensorFlow/Keras to ONNX - Direct Conversion")
    print("=" * 70)

    # Create model
    print("\n1. Creating Keras model...")
    model = create_keras_model()
    model.summary()

    # Define input signature
    print("\n2. Defining input signature...")
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)

    # Convert to ONNX
    output_path = "models/mnist_classifier.onnx"
    print(f"\n3. Converting to ONNX: {output_path}")

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=14,
        output_path=output_path
    )

    print("   ✓ Conversion complete")

    # Verify
    print("\n4. Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("   ✓ ONNX model is valid")

    # Test inference
    print("\n5. Testing ONNX Runtime inference...")
    session = ort.InferenceSession(output_path)

    # Create test input
    test_input = np.random.randn(1, 28, 28, 1).astype(np.float32)

    # TensorFlow inference
    tf_output = model.predict(test_input, verbose=0)

    # ONNX inference
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input})[0]

    # Compare
    max_diff = np.abs(tf_output - onnx_output).max()
    print(f"   Max difference: {max_diff:.6e}")

    if np.allclose(tf_output, onnx_output, rtol=1e-5, atol=1e-5):
        print("   ✓ Outputs match")
    else:
        print("   ✗ Outputs differ")

    print("\n" + "=" * 70)

def export_savedmodel():
    """Export from TensorFlow SavedModel format"""

    print("=" * 70)
    print("TensorFlow SavedModel to ONNX")
    print("=" * 70)

    # Create and save model
    print("\n1. Creating and saving TensorFlow model...")
    model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    saved_model_dir = "models/mobilenet_saved"
    model.save(saved_model_dir)
    print(f"   ✓ Saved to {saved_model_dir}")

    # Convert using command line (more reliable for complex models)
    import subprocess

    output_path = "models/mobilenet_v2_tf.onnx"
    print(f"\n2. Converting to ONNX: {output_path}")

    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", output_path,
        "--opset", "14"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("   ✓ Conversion successful")
    else:
        print(f"   ✗ Conversion failed: {result.stderr}")
        return

    # Verify
    print("\n3. Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("   ✓ ONNX model is valid")

    import os
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✓ File size: {file_size:.2f} MB")

    print("\n" + "=" * 70)

def export_with_concrete_function():
    """Export using concrete function (advanced)"""

    print("=" * 70)
    print("TensorFlow Concrete Function to ONNX")
    print("=" * 70)

    # Create model
    print("\n1. Creating model...")
    model = create_keras_model()

    # Get concrete function
    print("\n2. Getting concrete function...")
    concrete_func = tf.function(lambda x: model(x))
    concrete_func = concrete_func.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    # Convert
    output_path = "models/mnist_concrete.onnx"
    print(f"\n3. Converting to ONNX: {output_path}")

    model_proto, _ = tf2onnx.convert.from_function(
        concrete_func,
        input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32, name="input")],
        opset=14,
        output_path=output_path
    )

    print("   ✓ Conversion complete")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    export_keras_direct()
    export_savedmodel()
    export_with_concrete_function()
```

---

## ONNX Model Inspection

Create `scripts/onnx_model_inspector.py`:

```python
#!/usr/bin/env python3
"""
Inspect and analyze ONNX models
"""
import onnx
import onnx.helper as helper
import onnx.checker as checker
from onnx import numpy_helper
import numpy as np
from typing import List, Dict
import json

class ONNXModelInspector:
    """Comprehensive ONNX model inspection tool"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.graph = self.model.graph

    def validate(self) -> bool:
        """Validate ONNX model"""
        print("=" * 70)
        print("Model Validation")
        print("=" * 70)

        try:
            checker.check_model(self.model)
            print("\n✓ Model is valid ONNX")
            return True
        except Exception as e:
            print(f"\n✗ Model validation failed: {str(e)}")
            return False

    def get_metadata(self) -> Dict:
        """Extract model metadata"""
        print("\n" + "=" * 70)
        print("Model Metadata")
        print("=" * 70)

        metadata = {
            'producer_name': self.model.producer_name,
            'producer_version': self.model.producer_version,
            'domain': self.model.domain,
            'model_version': self.model.model_version,
            'doc_string': self.model.doc_string,
            'ir_version': self.model.ir_version,
            'opset_version': self.model.opset_import[0].version if self.model.opset_import else None
        }

        for key, value in metadata.items():
            print(f"  {key:<20}: {value}")

        return metadata

    def get_graph_info(self) -> Dict:
        """Get graph structure information"""
        print("\n" + "=" * 70)
        print("Graph Structure")
        print("=" * 70)

        info = {
            'name': self.graph.name,
            'num_nodes': len(self.graph.node),
            'num_inputs': len(self.graph.input),
            'num_outputs': len(self.graph.output),
            'num_initializers': len(self.graph.initializer)
        }

        print(f"\n  Graph Name: {info['name']}")
        print(f"  Nodes: {info['num_nodes']}")
        print(f"  Inputs: {info['num_inputs']}")
        print(f"  Outputs: {info['num_outputs']}")
        print(f"  Initializers (weights): {info['num_initializers']}")

        return info

    def get_inputs(self) -> List[Dict]:
        """Get input tensor information"""
        print("\n" + "=" * 70)
        print("Input Tensors")
        print("=" * 70)

        inputs = []

        for inp in self.graph.input:
            # Skip initializers (weights)
            if inp.name in [init.name for init in self.graph.initializer]:
                continue

            input_info = {
                'name': inp.name,
                'type': inp.type.tensor_type.elem_type,
                'shape': [
                    dim.dim_value if dim.dim_value else dim.dim_param
                    for dim in inp.type.tensor_type.shape.dim
                ]
            }

            inputs.append(input_info)

            print(f"\n  Name: {input_info['name']}")
            print(f"  Type: {onnx.TensorProto.DataType.Name(input_info['type'])}")
            print(f"  Shape: {input_info['shape']}")

        return inputs

    def get_outputs(self) -> List[Dict]:
        """Get output tensor information"""
        print("\n" + "=" * 70)
        print("Output Tensors")
        print("=" * 70)

        outputs = []

        for out in self.graph.output:
            output_info = {
                'name': out.name,
                'type': out.type.tensor_type.elem_type,
                'shape': [
                    dim.dim_value if dim.dim_value else dim.dim_param
                    for dim in out.type.tensor_type.shape.dim
                ]
            }

            outputs.append(output_info)

            print(f"\n  Name: {output_info['name']}")
            print(f"  Type: {onnx.TensorProto.DataType.Name(output_info['type'])}")
            print(f"  Shape: {output_info['shape']}")

        return outputs

    def get_operators(self) -> Dict[str, int]:
        """Get operator statistics"""
        print("\n" + "=" * 70)
        print("Operator Statistics")
        print("=" * 70)

        op_counts = {}

        for node in self.graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        # Sort by count
        sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Total unique operators: {len(op_counts)}")
        print(f"\n  Top operators:")

        for op, count in sorted_ops[:10]:
            print(f"    {op:<30} {count:>5}")

        return op_counts

    def get_model_size(self) -> Dict:
        """Calculate model size breakdown"""
        print("\n" + "=" * 70)
        print("Model Size Analysis")
        print("=" * 70)

        import os

        file_size = os.path.getsize(self.model_path)

        # Calculate parameter count
        param_count = 0
        param_size = 0

        for initializer in self.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            param_count += tensor.size
            param_size += tensor.nbytes

        size_info = {
            'file_size_mb': file_size / (1024 * 1024),
            'parameter_count': param_count,
            'parameter_size_mb': param_size / (1024 * 1024)
        }

        print(f"\n  File Size: {size_info['file_size_mb']:.2f} MB")
        print(f"  Parameters: {size_info['parameter_count']:,}")
        print(f"  Parameter Size: {size_info['parameter_size_mb']:.2f} MB")

        overhead = size_info['file_size_mb'] - size_info['parameter_size_mb']
        print(f"  Overhead: {overhead:.2f} MB ({overhead/size_info['file_size_mb']*100:.1f}%)")

        return size_info

    def generate_report(self, output_path: str = "reports/model_inspection.json"):
        """Generate comprehensive inspection report"""

        report = {
            'model_path': self.model_path,
            'is_valid': self.validate(),
            'metadata': self.get_metadata(),
            'graph_info': self.get_graph_info(),
            'inputs': self.get_inputs(),
            'outputs': self.get_outputs(),
            'operators': self.get_operators(),
            'size_info': self.get_model_size()
        }

        # Save report
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n" + "=" * 70)
        print(f"Report saved to: {output_path}")
        print("=" * 70)

        return report

def main():
    """Inspect ONNX model"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python onnx_model_inspector.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    inspector = ONNXModelInspector(model_path)
    report = inspector.generate_report()

if __name__ == "__main__":
    main()
```

---

## ONNX Runtime Inference

Create `scripts/onnx_runtime_inference.py`:

```python
#!/usr/bin/env python3
"""
ONNX Runtime inference examples
"""
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Optional
import time

class ONNXInferenceEngine:
    """Production-ready ONNX Runtime inference engine"""

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None
    ):
        """
        Initialize inference engine

        Args:
            model_path: Path to ONNX model
            providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            session_options: ONNX Runtime session options
        """
        self.model_path = model_path

        # Set providers
        if providers is None:
            providers = ['CPUExecutionProvider']

        # Create session options if not provided
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers
        )

        # Get model metadata
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Cache input/output metadata
        self.input_metadata = {
            inp.name: {
                'shape': inp.shape,
                'type': inp.type,
                'dtype': self._onnx_type_to_numpy(inp.type)
            }
            for inp in self.session.get_inputs()
        }

        self.output_metadata = {
            out.name: {
                'shape': out.shape,
                'type': out.type
            }
            for out in self.session.get_outputs()
        }

    def _onnx_type_to_numpy(self, onnx_type: str) -> np.dtype:
        """Convert ONNX type string to NumPy dtype"""
        type_map = {
            'tensor(float)': np.float32,
            'tensor(double)': np.float64,
            'tensor(int32)': np.int32,
            'tensor(int64)': np.int64,
            'tensor(uint8)': np.uint8,
        }
        return type_map.get(onnx_type, np.float32)

    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'providers': self.session.get_providers(),
            'inputs': self.input_metadata,
            'outputs': self.output_metadata
        }

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Run inference

        Args:
            inputs: Dictionary of input name to NumPy array
            output_names: Optional list of output names to return

        Returns:
            List of output arrays
        """
        # Validate inputs
        for name in inputs:
            if name not in self.input_names:
                raise ValueError(f"Unknown input: {name}")

        # Run inference
        if output_names is None:
            output_names = self.output_names

        outputs = self.session.run(output_names, inputs)

        return outputs

    def benchmark(
        self,
        inputs: Dict[str, np.ndarray],
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Benchmark inference performance

        Args:
            inputs: Input dictionary
            iterations: Number of iterations
            warmup: Number of warmup iterations

        Returns:
            Performance statistics
        """
        # Warmup
        for _ in range(warmup):
            self.infer(inputs)

        # Measure latency
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            self.infer(inputs)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

        latencies = np.array(latencies)

        return {
            'iterations': iterations,
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'throughput_qps': 1000.0 / float(np.mean(latencies))
        }

def example_single_input():
    """Example: Single input model"""

    print("=" * 70)
    print("ONNX Runtime Inference - Single Input")
    print("=" * 70)

    # Create engine
    engine = ONNXInferenceEngine("models/simple_classifier.onnx")

    # Display info
    info = engine.get_info()
    print("\nModel Information:")
    print(f"  Providers: {info['providers']}")
    print(f"  Inputs: {list(info['inputs'].keys())}")
    print(f"  Outputs: {list(info['outputs'].keys())}")

    # Prepare input
    input_name = engine.input_names[0]
    input_shape = info['inputs'][input_name]['shape']
    input_dtype = info['inputs'][input_name]['dtype']

    # Create batch of random inputs
    batch_size = 4
    actual_shape = [batch_size if dim == 'batch_size' else dim for dim in input_shape]
    test_input = np.random.randn(*actual_shape).astype(input_dtype)

    print(f"\nInput:")
    print(f"  Name: {input_name}")
    print(f"  Shape: {test_input.shape}")
    print(f"  Dtype: {test_input.dtype}")

    # Run inference
    outputs = engine.infer({input_name: test_input})

    print(f"\nOutput:")
    print(f"  Shape: {outputs[0].shape}")
    print(f"  Sample: {outputs[0][0][:5]}")

    # Benchmark
    print("\nBenchmarking...")
    stats = engine.benchmark({input_name: test_input}, iterations=100)

    print(f"  Mean latency: {stats['mean_ms']:.2f} ms")
    print(f"  Std latency: {stats['std_ms']:.2f} ms")
    print(f"  P95 latency: {stats['p95_ms']:.2f} ms")
    print(f"  P99 latency: {stats['p99_ms']:.2f} ms")
    print(f"  Throughput: {stats['throughput_qps']:.2f} QPS")

    print("\n" + "=" * 70)

def example_execution_providers():
    """Example: Different execution providers"""

    print("=" * 70)
    print("ONNX Runtime - Execution Providers")
    print("=" * 70)

    # Check available providers
    available = ort.get_available_providers()
    print(f"\nAvailable providers:")
    for provider in available:
        print(f"  - {provider}")

    # Test each provider
    model_path = "models/simple_classifier.onnx"
    test_input = np.random.randn(1, 784).astype(np.float32)

    providers_to_test = [
        ['CPUExecutionProvider'],
    ]

    if 'CUDAExecutionProvider' in available:
        providers_to_test.append(['CUDAExecutionProvider', 'CPUExecutionProvider'])

    print("\nPerformance by provider:")

    for providers in providers_to_test:
        engine = ONNXInferenceEngine(model_path, providers=providers)
        input_name = engine.input_names[0]

        stats = engine.benchmark({input_name: test_input}, iterations=100)

        provider_name = providers[0]
        print(f"\n  {provider_name}:")
        print(f"    Mean latency: {stats['mean_ms']:.2f} ms")
        print(f"    Throughput: {stats['throughput_qps']:.2f} QPS")

    print("\n" + "=" * 70)

def example_optimized_session():
    """Example: Optimized session configuration"""

    print("=" * 70)
    print("ONNX Runtime - Optimized Session")
    print("=" * 70)

    # Create optimized session options
    session_options = ort.SessionOptions()

    # Enable all graph optimizations
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Set intra-op parallelism (threads within operations)
    session_options.intra_op_num_threads = 4

    # Set inter-op parallelism (threads between operations)
    session_options.inter_op_num_threads = 4

    # Enable memory pattern optimization
    session_options.enable_mem_pattern = True

    # Enable memory reuse
    session_options.enable_cpu_mem_arena = True

    print("\nSession Configuration:")
    print(f"  Graph optimization: {session_options.graph_optimization_level}")
    print(f"  Intra-op threads: {session_options.intra_op_num_threads}")
    print(f"  Inter-op threads: {session_options.inter_op_num_threads}")
    print(f"  Memory pattern: {session_options.enable_mem_pattern}")
    print(f"  Memory arena: {session_options.enable_cpu_mem_arena}")

    # Create engine with optimized session
    engine = ONNXInferenceEngine(
        "models/simple_classifier.onnx",
        session_options=session_options
    )

    # Benchmark
    input_name = engine.input_names[0]
    test_input = np.random.randn(1, 784).astype(np.float32)

    stats = engine.benchmark({input_name: test_input}, iterations=100)

    print(f"\nPerformance:")
    print(f"  Mean latency: {stats['mean_ms']:.2f} ms")
    print(f"  Throughput: {stats['throughput_qps']:.2f} QPS")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    example_single_input()
    example_execution_providers()
    example_optimized_session()
```

---

## Performance Comparison

Create `scripts/performance_comparison.py`:

```python
#!/usr/bin/env python3
"""
Compare performance across PyTorch, TensorFlow, and ONNX
"""
import torch
import torch.nn as nn
import tensorflow as tf
import onnxruntime as ort
import numpy as np
import time
from typing import Dict, Callable
import matplotlib.pyplot as plt
import json

class FrameworkBenchmark:
    """Benchmark ML frameworks"""

    def __init__(self, iterations: int = 100, warmup: int = 10):
        self.iterations = iterations
        self.warmup = warmup
        self.results = {}

    def benchmark(self, name: str, func: Callable, *args) -> Dict:
        """
        Benchmark a function

        Args:
            name: Benchmark name
            func: Function to benchmark
            *args: Function arguments

        Returns:
            Performance statistics
        """
        print(f"\nBenchmarking {name}...")

        # Warmup
        for _ in range(self.warmup):
            func(*args)

        # Measure latency
        latencies = []

        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

        latencies = np.array(latencies)

        stats = {
            'name': name,
            'iterations': self.iterations,
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'throughput_qps': 1000.0 / float(np.mean(latencies))
        }

        self.results[name] = stats

        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  P95: {stats['p95_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_qps']:.2f} QPS")

        return stats

    def compare(self, baseline: str = None):
        """Compare results"""
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)

        if not self.results:
            print("No results to compare")
            return

        # Table header
        print(f"\n{'Framework':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput (QPS)':<18}")
        print("-" * 70)

        # Sort by mean latency
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['mean_ms']
        )

        for name, stats in sorted_results:
            print(
                f"{name:<20} "
                f"{stats['mean_ms']:<12.2f} "
                f"{stats['p95_ms']:<12.2f} "
                f"{stats['throughput_qps']:<18.2f}"
            )

        # Speedup comparison
        if baseline and baseline in self.results:
            print(f"\nSpeedup vs {baseline}:")
            baseline_mean = self.results[baseline]['mean_ms']

            for name, stats in sorted_results:
                if name != baseline:
                    speedup = baseline_mean / stats['mean_ms']
                    print(f"  {name:<20} {speedup:>6.2f}x")

        # Find fastest
        fastest = sorted_results[0]
        print(f"\nFastest: {fastest[0]} ({fastest[1]['mean_ms']:.2f} ms)")

    def plot_results(self, output_path: str = "reports/performance_comparison.png"):
        """Plot performance comparison"""
        if not self.results:
            return

        names = list(self.results.keys())
        means = [self.results[name]['mean_ms'] for name in names]
        p95s = [self.results[name]['p95_ms'] for name in names]

        x = np.arange(len(names))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Latency comparison
        ax1.bar(x - width/2, means, width, label='Mean', alpha=0.8)
        ax1.bar(x + width/2, p95s, width, label='P95', alpha=0.8)
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Throughput comparison
        throughputs = [self.results[name]['throughput_qps'] for name in names]
        ax2.bar(names, throughputs, alpha=0.8)
        ax2.set_ylabel('Throughput (QPS)')
        ax2.set_title('Throughput Comparison')
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    def save_results(self, output_path: str = "reports/benchmark_results.json"):
        """Save results to JSON"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

def compare_simple_model():
    """Compare simple model across frameworks"""

    print("=" * 70)
    print("Comparing Simple Feedforward Model")
    print("=" * 70)

    # Model parameters
    input_size = 784
    hidden_size = 256
    output_size = 10
    batch_size = 1

    # PyTorch model
    class PyTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    pytorch_model = PyTorchModel()
    pytorch_model.eval()
    pytorch_input = torch.randn(batch_size, input_size)

    # TensorFlow model
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(output_size)
    ])
    tf_input = np.random.randn(batch_size, input_size).astype(np.float32)

    # Export PyTorch to ONNX
    onnx_path = "models/comparison_model.onnx"
    torch.onnx.export(
        pytorch_model,
        pytorch_input,
        onnx_path,
        input_names=['input'],
        output_names=['output']
    )

    # ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = pytorch_input.numpy()

    # Create benchmark
    benchmark = FrameworkBenchmark(iterations=200, warmup=20)

    # Benchmark PyTorch
    def pytorch_infer():
        with torch.no_grad():
            return pytorch_model(pytorch_input)

    benchmark.benchmark("PyTorch (CPU)", pytorch_infer)

    # Benchmark TensorFlow
    def tf_infer():
        return tf_model.predict(tf_input, verbose=0)

    benchmark.benchmark("TensorFlow (CPU)", tf_infer)

    # Benchmark ONNX Runtime
    def onnx_infer():
        return ort_session.run(None, {'input': onnx_input})

    benchmark.benchmark("ONNX Runtime (CPU)", onnx_infer)

    # Compare results
    benchmark.compare(baseline="PyTorch (CPU)")
    benchmark.plot_results()
    benchmark.save_results()

def compare_batch_sizes():
    """Compare performance across different batch sizes"""

    print("\n" + "=" * 70)
    print("Comparing Different Batch Sizes")
    print("=" * 70)

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    model.eval()

    # Export to ONNX
    onnx_path = "models/batch_comparison.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 784),
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    ort_session = ort.InferenceSession(onnx_path)

    batch_sizes = [1, 4, 8, 16, 32, 64]

    print(f"\n{'Batch Size':<12} {'PyTorch (ms)':<15} {'ONNX (ms)':<15} {'Speedup':<10}")
    print("-" * 55)

    for bs in batch_sizes:
        # PyTorch
        pytorch_input = torch.randn(bs, 784)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                model(pytorch_input)
        pytorch_time = (time.perf_counter() - start) / 50 * 1000

        # ONNX
        onnx_input = pytorch_input.numpy()

        start = time.perf_counter()
        for _ in range(50):
            ort_session.run(None, {'input': onnx_input})
        onnx_time = (time.perf_counter() - start) / 50 * 1000

        speedup = pytorch_time / onnx_time

        print(f"{bs:<12} {pytorch_time:<15.2f} {onnx_time:<15.2f} {speedup:<10.2f}x")

if __name__ == "__main__":
    compare_simple_model()
    compare_batch_sizes()
```

---

## Production Converter Tool

Create `scripts/production_converter.py`:

```python
#!/usr/bin/env python3
"""
Production-ready model conversion tool
"""
import torch
import torch.nn as nn
import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import os
import json
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelConverter:
    """
    Production-ready ML model converter

    Features:
    - PyTorch to ONNX conversion
    - TensorFlow to ONNX conversion
    - Automatic validation
    - Performance benchmarking
    - Detailed reporting
    """

    def __init__(
        self,
        validate: bool = True,
        benchmark: bool = False,
        opset_version: int = 14,
        verbose: bool = True
    ):
        self.validate = validate
        self.benchmark = benchmark
        self.opset_version = opset_version
        self.verbose = verbose
        self.report = {}

    def convert_pytorch(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        input_names: List[str] = ['input'],
        output_names: List[str] = ['output'],
        dynamic_axes: Optional[Dict] = None
    ) -> bool:
        """
        Convert PyTorch model to ONNX

        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            output_path: Output ONNX file path
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic dimension configuration

        Returns:
            Success status
        """
        logger.info("=" * 70)
        logger.info("PyTorch to ONNX Conversion")
        logger.info("=" * 70)

        try:
            # Ensure model is in eval mode
            model.eval()

            # Default dynamic axes
            if dynamic_axes is None:
                dynamic_axes = {
                    input_names[0]: {0: 'batch_size'},
                    output_names[0]: {0: 'batch_size'}
                }

            # Get PyTorch output for validation
            logger.info("Running PyTorch inference for validation...")
            with torch.no_grad():
                pytorch_output = model(dummy_input)

            # Export to ONNX
            logger.info(f"Exporting to {output_path}...")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=self.verbose
            )

            # Verify ONNX model
            logger.info("Verifying ONNX model...")
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model is valid")

            # Store model info in report
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            self.report.update({
                'framework': 'PyTorch',
                'output_path': output_path,
                'file_size_mb': round(file_size, 2),
                'opset_version': self.opset_version,
                'num_nodes': len(onnx_model.graph.node),
                'input_names': input_names,
                'output_names': output_names
            })

            logger.info(f"Model size: {file_size:.2f} MB")
            logger.info(f"Graph nodes: {len(onnx_model.graph.node)}")

            # Validate conversion
            if self.validate:
                logger.info("\nValidating conversion accuracy...")
                validation_passed = self._validate_pytorch(
                    model,
                    output_path,
                    dummy_input,
                    pytorch_output
                )

                if not validation_passed:
                    logger.error("Validation failed!")
                    return False

            # Benchmark if requested
            if self.benchmark:
                logger.info("\nBenchmarking performance...")
                self._benchmark_onnx(output_path, dummy_input.numpy(), input_names[0])

            logger.info("\n✓ Conversion successful!")
            return True

        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}", exc_info=True)
            self.report['error'] = str(e)
            return False

    def _validate_pytorch(
        self,
        pytorch_model: nn.Module,
        onnx_path: str,
        test_input: torch.Tensor,
        pytorch_output: torch.Tensor,
        tolerance: float = 1e-5
    ) -> bool:
        """Validate PyTorch to ONNX conversion"""

        try:
            # ONNX inference
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            onnx_output = session.run(None, {input_name: test_input.numpy()})[0]

            # Compare outputs
            pytorch_np = pytorch_output.detach().numpy()

            abs_diff = np.abs(pytorch_np - onnx_output)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)

            # Calculate relative difference
            rel_diff = np.mean(
                np.abs(pytorch_np - onnx_output) / (np.abs(pytorch_np) + 1e-8)
            )

            logger.info(f"  Max absolute difference: {max_diff:.6e}")
            logger.info(f"  Mean absolute difference: {mean_diff:.6e}")
            logger.info(f"  Mean relative difference: {rel_diff:.6e}")

            # Store in report
            self.report.update({
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'relative_diff': float(rel_diff)
            })

            # Check if outputs are close
            passed = np.allclose(pytorch_np, onnx_output, rtol=tolerance, atol=tolerance)

            if passed:
                logger.info(f"  ✓ Validation PASSED (tolerance: {tolerance})")
            else:
                logger.warning(f"  ✗ Validation FAILED (tolerance: {tolerance})")

            self.report['validation_passed'] = passed
            return passed

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            self.report['validation_error'] = str(e)
            return False

    def _benchmark_onnx(
        self,
        onnx_path: str,
        test_input: np.ndarray,
        input_name: str,
        iterations: int = 100,
        warmup: int = 10
    ):
        """Benchmark ONNX model"""

        try:
            session = ort.InferenceSession(onnx_path)

            # Warmup
            for _ in range(warmup):
                session.run(None, {input_name: test_input})

            # Measure
            import time
            latencies = []

            for _ in range(iterations):
                start = time.perf_counter()
                session.run(None, {input_name: test_input})
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)

            latencies = np.array(latencies)

            stats = {
                'mean_ms': float(np.mean(latencies)),
                'std_ms': float(np.std(latencies)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'p99_ms': float(np.percentile(latencies, 99)),
                'throughput_qps': 1000.0 / float(np.mean(latencies))
            }

            logger.info(f"  Mean latency: {stats['mean_ms']:.2f} ms")
            logger.info(f"  P95 latency: {stats['p95_ms']:.2f} ms")
            logger.info(f"  Throughput: {stats['throughput_qps']:.2f} QPS")

            self.report['benchmark'] = stats

        except Exception as e:
            logger.error(f"Benchmark error: {str(e)}")

    def save_report(self, output_path: str = "conversion_report.json"):
        """Save conversion report"""

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2)

        logger.info(f"\nReport saved to: {output_path}")

def load_pytorch_model(model_name: str) -> Tuple[nn.Module, torch.Tensor]:
    """Load pre-trained PyTorch model"""

    logger.info(f"Loading PyTorch model: {model_name}")

    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        import torchvision.models as models
        model = getattr(models, model_name)(weights='DEFAULT')
        dummy_input = torch.randn(1, 3, 224, 224)

    elif model_name in ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']:
        import torchvision.models as models
        model = getattr(models, model_name)(weights='DEFAULT')
        dummy_input = torch.randn(1, 3, 224, 224)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, dummy_input

def main():
    """Main CLI interface"""

    parser = argparse.ArgumentParser(
        description='Convert ML models to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert ResNet18
  python production_converter.py --model resnet18 --output models/resnet18.onnx

  # Convert with benchmarking
  python production_converter.py --model mobilenet_v2 --output models/mobilenet.onnx --benchmark

  # Skip validation (faster)
  python production_converter.py --model resnet50 --output models/resnet50.onnx --no-validate
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='PyTorch model name (resnet18, resnet50, mobilenet_v2, etc.)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output ONNX file path'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark performance'
    )

    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )

    parser.add_argument(
        '--report',
        type=str,
        default='conversion_report.json',
        help='Report output path'
    )

    args = parser.parse_args()

    try:
        # Load model
        model, dummy_input = load_pytorch_model(args.model)

        # Create converter
        converter = ModelConverter(
            validate=not args.no_validate,
            benchmark=args.benchmark,
            opset_version=args.opset
        )

        # Convert
        success = converter.convert_pytorch(
            model=model,
            dummy_input=dummy_input,
            output_path=args.output
        )

        # Save report
        if success:
            converter.save_report(args.report)
            logger.info("\n✓ Conversion complete!")
            return 0
        else:
            logger.error("\n✗ Conversion failed!")
            return 1

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Export Fails with "torch.onnx export failed"

**Problem**: Model export fails during tracing

**Solutions**:
```python
# Solution 1: Ensure model is in eval mode
model.eval()

# Solution 2: Disable dropout/batch norm training mode
for module in model.modules():
    if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm2d)):
        module.eval()

# Solution 3: Use torch.jit.trace instead
traced_model = torch.jit.trace(model, dummy_input)
torch.onnx.export(traced_model, dummy_input, "model.onnx")
```

#### 2. Dynamic Axes Not Working

**Problem**: Batch size is fixed after export

**Solution**:
```python
# Explicitly specify dynamic axes
dynamic_axes = {
    'input': {
        0: 'batch_size',    # Variable batch dimension
        2: 'height',        # Variable height (if needed)
        3: 'width'          # Variable width (if needed)
    },
    'output': {
        0: 'batch_size'
    }
}

torch.onnx.export(
    model, dummy_input, "model.onnx",
    dynamic_axes=dynamic_axes
)
```

#### 3. Validation Fails (Outputs Differ)

**Problem**: ONNX outputs don't match PyTorch

**Diagnostic steps**:
```python
# Check intermediate outputs
import onnx
from onnx import numpy_helper

model_onnx = onnx.load("model.onnx")

# Print all intermediate tensor names
for node in model_onnx.graph.node:
    print(f"Node: {node.op_type}")
    print(f"  Inputs: {node.input}")
    print(f"  Outputs: {node.output}")

# Run with intermediate outputs
session = ort.InferenceSession("model.onnx")
# Enable verbose mode to see what's happening
```

**Common causes**:
- Random number generation (use fixed seed)
- Batch normalization in training mode
- Dropout enabled during inference
- Custom operators not supported

#### 4. TensorFlow Conversion Fails

**Problem**: tf2onnx fails to convert model

**Solutions**:
```python
# Solution 1: Save as SavedModel first
model.save("saved_model_dir")

# Then convert from SavedModel
import subprocess
subprocess.run([
    "python", "-m", "tf2onnx.convert",
    "--saved-model", "saved_model_dir",
    "--output", "model.onnx",
    "--opset", "14"
])

# Solution 2: Use specific input signature
spec = tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input")
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[spec],
    opset=14,
    output_path="model.onnx"
)
```

#### 5. ONNX Runtime Execution Error

**Problem**: Model runs but throws runtime errors

**Solutions**:
```python
# Check execution providers
providers = ort.get_available_providers()
print(f"Available providers: {providers}")

# Try with specific provider
session = ort.InferenceSession(
    "model.onnx",
    providers=['CPUExecutionProvider']
)

# Enable verbose logging
session_options = ort.SessionOptions()
session_options.log_severity_level = 0  # Verbose
session = ort.InferenceSession("model.onnx", session_options)
```

#### 6. Performance Not Improved

**Problem**: ONNX Runtime slower than native framework

**Optimization steps**:
```python
# 1. Enable graph optimizations
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 2. Set thread count
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 4

# 3. Enable memory optimizations
session_options.enable_mem_pattern = True
session_options.enable_cpu_mem_arena = True

# 4. Use appropriate execution provider
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", session_options, providers=providers)
```

---

## Best Practices

### 1. Model Preparation

```python
# Always set model to eval mode
model.eval()

# Freeze batch normalization
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None

# Use deterministic operations
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

### 2. Input Specification

```python
# Be explicit about input shapes and types
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

# Name inputs descriptively
input_names = ['image']  # Not just 'input'
output_names = ['class_probabilities']  # Not just 'output'

# Specify all variable dimensions
dynamic_axes = {
    'image': {0: 'batch_size', 2: 'height', 3: 'width'},
    'class_probabilities': {0: 'batch_size'}
}
```

### 3. Validation Strategy

```python
# Test multiple inputs
test_inputs = [
    torch.randn(1, 3, 224, 224),      # Single sample
    torch.randn(8, 3, 224, 224),      # Small batch
    torch.zeros(1, 3, 224, 224),      # Edge case: zeros
    torch.ones(1, 3, 224, 224),       # Edge case: ones
]

for test_input in test_inputs:
    validate_conversion(model, onnx_session, test_input)
```

### 4. Version Pinning

```txt
# requirements.txt
torch==2.1.0
onnx==1.15.0
onnxruntime==1.16.3
tf2onnx==1.15.1
```

### 5. Documentation

```python
# Document conversion metadata
metadata = {
    'model_name': 'ResNet50',
    'framework': 'PyTorch',
    'framework_version': torch.__version__,
    'onnx_version': onnx.__version__,
    'opset_version': 14,
    'input_shape': [1, 3, 224, 224],
    'output_shape': [1, 1000],
    'preprocessing': 'ImageNet normalization',
    'postprocessing': 'Softmax for probabilities',
    'conversion_date': '2024-10-15',
    'validated': True
}
```

---

## Validation Checklist

Use this checklist before deploying ONNX models to production:

### Pre-Conversion
- [ ] Model is in evaluation mode (`model.eval()`)
- [ ] All dropout layers are disabled
- [ ] Batch normalization is in inference mode
- [ ] Input shapes are clearly defined
- [ ] Dynamic axes are specified if needed
- [ ] Test data is prepared

### During Conversion
- [ ] Conversion completes without errors
- [ ] ONNX model passes validation (`onnx.checker.check_model()`)
- [ ] File size is reasonable
- [ ] Graph structure looks correct (use Netron to visualize)

### Post-Conversion
- [ ] Numerical validation passes (outputs match within tolerance)
- [ ] Multiple test cases validated
- [ ] Edge cases tested (zeros, ones, random values)
- [ ] Different batch sizes work correctly
- [ ] Performance benchmarked
- [ ] Memory usage acceptable

### Production Readiness
- [ ] Model tested on target hardware
- [ ] Execution providers configured correctly
- [ ] Session options optimized
- [ ] Error handling implemented
- [ ] Monitoring/logging added
- [ ] Documentation complete
- [ ] Version tracking in place

---

## Summary

This implementation guide covered:

1. **Environment Setup**: Installing and verifying ONNX ecosystem
2. **PyTorch Conversion**: Exporting models with torch.onnx.export()
3. **TensorFlow Conversion**: Using tf2onnx for Keras and SavedModel
4. **Model Inspection**: Analyzing ONNX model structure and properties
5. **ONNX Runtime**: Optimized inference with different execution providers
6. **Performance Comparison**: Benchmarking across frameworks
7. **Production Tool**: Complete converter with validation and reporting

### Key Takeaways

- ONNX provides framework-independent model deployment
- ONNX Runtime often outperforms native frameworks for inference
- Always validate converted models for numerical accuracy
- Use appropriate execution providers for your hardware
- Document conversion process and model metadata

### Next Steps

1. Practice converting your own models
2. Experiment with optimization techniques (quantization, pruning)
3. Deploy ONNX models in production serving systems
4. Explore hardware-specific acceleration (TensorRT, OpenVINO)
5. Build CI/CD pipelines for model conversion and validation

