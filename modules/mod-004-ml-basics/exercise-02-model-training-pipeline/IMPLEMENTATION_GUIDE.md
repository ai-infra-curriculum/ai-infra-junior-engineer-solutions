# Implementation Guide: TensorFlow Model Inference
## Exercise 02 - ML Infrastructure Engineering

**Difficulty**: Beginner
**Estimated Time**: 2-3 hours
**Focus**: Production TensorFlow inference, performance optimization, SavedModel deployment

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Environment Setup](#phase-1-environment-setup)
4. [Phase 2: Model Loading & Inspection](#phase-2-model-loading--inspection)
5. [Phase 3: Image Preprocessing Pipeline](#phase-3-image-preprocessing-pipeline)
6. [Phase 4: Inference Execution](#phase-4-inference-execution)
7. [Phase 5: Performance Optimization](#phase-5-performance-optimization)
8. [Phase 6: SavedModel Export & Serving](#phase-6-savedmodel-export--serving)
9. [Phase 7: Production Application](#phase-7-production-application)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [Validation Checklist](#validation-checklist)

---

## Overview

### Learning Objectives

By completing this guide, you will:

- Configure TensorFlow environments with proper GPU/CPU detection
- Load and inspect pre-trained models from keras.applications
- Implement production-grade image preprocessing pipelines
- Execute optimized inference using TensorFlow's functional APIs
- Measure and optimize inference latency and throughput
- Export models in SavedModel format for deployment
- Understand TensorFlow Serving architecture and deployment patterns
- Build complete production-ready inference applications

### Infrastructure Focus

This exercise emphasizes ML infrastructure engineering skills:

- **Device Management**: GPU memory configuration, device placement
- **Performance**: Inference optimization with `@tf.function`, batching strategies
- **Deployment**: SavedModel format, versioning, TensorFlow Serving integration
- **Production Code**: Error handling, logging, monitoring-ready implementations

---

## Prerequisites

### Required Knowledge

- Basic Python programming (variables, functions, classes)
- Understanding of ML concepts (models, inference, preprocessing)
- Familiarity with NumPy arrays
- Command line basics

### System Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- 2GB free disk space
- Optional: NVIDIA GPU with CUDA support

### Pre-Installation Check

```bash
# Verify Python version
python --version  # Should be 3.8+

# Check pip
pip --version

# Check available disk space
df -h .

# Check for GPU (optional)
nvidia-smi
```

---

## Phase 1: Environment Setup

### Step 1.1: Create Virtual Environment

**Why Virtual Environments?**
- Isolated dependency management
- Reproducible environments
- Prevent version conflicts
- Easy cleanup and recreation

```bash
# Create virtual environment
python -m venv tensorflow_inference_env

# Activate environment
# Linux/Mac:
source tensorflow_inference_env/bin/activate

# Windows:
# tensorflow_inference_env\Scripts\activate

# Verify activation (should show venv path)
which python
```

**Expected Output**:
```
/path/to/tensorflow_inference_env/bin/python
```

### Step 1.2: Install TensorFlow and Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install TensorFlow and core dependencies
pip install tensorflow==2.15.0 pillow==10.1.0 requests==2.31.0 numpy==1.24.3 matplotlib==3.8.0

# For GPU support (optional - only if you have NVIDIA GPU)
# TensorFlow 2.15+ includes GPU support automatically if CUDA is available

# Verify installations
pip list | grep -E "tensorflow|pillow|numpy"
```

**Installation Notes**:
- TensorFlow 2.x includes Keras by default
- GPU support is automatic if CUDA drivers are installed
- Installation takes 5-10 minutes depending on connection

### Step 1.3: Verify Installation and Device Configuration

Create `verify_installation.py`:

```python
#!/usr/bin/env python3
"""
TensorFlow Installation Verification Script
Checks TensorFlow, Keras, and GPU availability
"""

import sys
import tensorflow as tf
import numpy as np
from datetime import datetime

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

def verify_versions():
    """Verify package versions"""
    print_section("Package Versions")
    print(f"Python version:     {sys.version.split()[0]}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version:      {tf.keras.__version__}")
    print(f"NumPy version:      {np.__version__}")

def check_gpu_availability():
    """Check and configure GPU devices"""
    print_section("Device Configuration")

    # List physical devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')

    print(f"CPUs available: {len(cpus)}")
    for i, cpu in enumerate(cpus):
        print(f"  CPU {i}: {cpu.name}")

    print(f"\nGPUs available: {len(gpus)}")

    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")

        # Configure GPU memory growth
        try:
            # Prevent TensorFlow from allocating all GPU memory at once
            # This allows multiple processes to share the GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print("\n✓ GPU memory growth enabled")
            print("  (TensorFlow will allocate memory as needed)")

            # Get GPU details
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            if gpu_details:
                print(f"\nGPU Details:")
                for key, value in gpu_details.items():
                    print(f"  {key}: {value}")

        except RuntimeError as e:
            print(f"\n✗ GPU configuration error: {e}")
            print("  This is usually not critical - continuing with default settings")
    else:
        print("\nNo GPU detected - using CPU")
        print("This is fine for learning, but GPU is recommended for production")

def test_basic_operations():
    """Test basic TensorFlow operations"""
    print_section("Basic Operations Test")

    # Create test tensors
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

    # Matrix multiplication
    c = tf.matmul(a, b)

    print("Matrix A:")
    print(a.numpy())
    print("\nMatrix B:")
    print(b.numpy())
    print("\nA @ B:")
    print(c.numpy())
    print("\n✓ Basic tensor operations working correctly")

def test_keras_import():
    """Test Keras functionality"""
    print_section("Keras Functionality Test")

    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Test forward pass
    test_input = tf.random.normal([1, 5])
    output = model(test_input)

    print(f"Test model created with {len(model.layers)} layers")
    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.numpy()[0][0]:.4f}")
    print("\n✓ Keras functionality working correctly")

def check_model_availability():
    """Check if pre-trained models are accessible"""
    print_section("Pre-trained Model Access Test")

    try:
        # Try to access MobileNetV2 (will download if not cached)
        print("Attempting to access MobileNetV2 from keras.applications...")
        print("(First run will download ~14MB - this is expected)")

        model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )

        print(f"\n✓ Model loaded successfully")
        print(f"  Model name:   {model.name}")
        print(f"  Input shape:  {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total layers: {len(model.layers)}")

        # Count parameters
        trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
        total = sum([tf.size(w).numpy() for w in model.weights])

        print(f"  Parameters:   {total:,} total, {trainable:,} trainable")

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("  Check your internet connection and try again")

def main():
    """Run all verification tests"""
    print(f"\n{'#'*60}")
    print(f"# TensorFlow Installation Verification")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    try:
        verify_versions()
        check_gpu_availability()
        test_basic_operations()
        test_keras_import()
        check_model_availability()

        print_section("Verification Complete")
        print("✓ All checks passed!")
        print("\nYour TensorFlow environment is ready for inference exercises.")
        print("You can now proceed to the next phase.")

    except Exception as e:
        print_section("Verification Failed")
        print(f"✗ Error during verification: {e}")
        print("\nPlease check the error messages above and:")
        print("  1. Ensure TensorFlow is installed: pip install tensorflow")
        print("  2. Try reinstalling in a fresh virtual environment")
        print("  3. Check system requirements and dependencies")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
```

**Run the verification**:

```bash
python verify_installation.py
```

**Expected Output**:

```
############################################################
# TensorFlow Installation Verification
# 2025-11-02 14:30:00
############################################################

============================================================
 Package Versions
============================================================

Python version:     3.10.12
TensorFlow version: 2.15.0
Keras version:      2.15.0
NumPy version:      1.24.3

============================================================
 Device Configuration
============================================================

CPUs available: 1
  CPU 0: /physical_device:CPU:0

GPUs available: 1
  GPU 0: /physical_device:GPU:0

✓ GPU memory growth enabled
  (TensorFlow will allocate memory as needed)

============================================================
 Basic Operations Test
============================================================

Matrix A:
[[1. 2.]
 [3. 4.]]

Matrix B:
[[5. 6.]
 [7. 8.]]

A @ B:
[[19. 22.]
 [43. 50.]]

✓ Basic tensor operations working correctly

============================================================
 Keras Functionality Test
============================================================

Test model created with 2 layers
Input shape:  (1, 5)
Output shape: (1, 1)
Output value: 0.5234

✓ Keras functionality working correctly

============================================================
 Pre-trained Model Access Test
============================================================

Attempting to access MobileNetV2 from keras.applications...
(First run will download ~14MB - this is expected)

✓ Model loaded successfully
  Model name:   mobilenetv2_1.00_224
  Input shape:  (None, 224, 224, 3)
  Output shape: (None, 1000)
  Total layers: 158
  Parameters:   3,538,984 total, 3,538,984 trainable

============================================================
 Verification Complete
============================================================

✓ All checks passed!

Your TensorFlow environment is ready for inference exercises.
You can now proceed to the next phase.
```

### Step 1.4: Download Test Images

Create `download_test_images.py`:

```python
#!/usr/bin/env python3
"""
Download test images for inference testing
"""

import urllib.request
import os
import sys
from pathlib import Path

# Test images from Wikimedia Commons (freely licensed)
TEST_IMAGES = {
    "elephant.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg",
    "tiger.jpg": "https://upload.wikimedia.org/wikipedia/commons/5/56/Tiger.50.jpg",
    "airplane.jpg": "https://upload.wikimedia.org/wikipedia/commons/2/2d/Boeing_777-31H_Air_China_B-2086.jpg",
    "golden_retriever.jpg": "https://upload.wikimedia.org/wikipedia/commons/9/93/Golden_Retriever_Carlos_%2810581910556%29.jpg",
}

def download_image(url, filepath):
    """Download image from URL to filepath"""
    try:
        print(f"  Downloading {filepath.name}...")
        urllib.request.urlretrieve(url, filepath)

        # Verify file was created and has content
        if filepath.exists() and filepath.stat().st_size > 0:
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ Downloaded {filepath.name} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"  ✗ Download failed: file is empty")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Download all test images"""
    print("Downloading test images for TensorFlow inference exercises...")
    print("=" * 60)

    # Create images directory
    images_dir = Path("test_images")
    images_dir.mkdir(exist_ok=True)
    print(f"\nImages directory: {images_dir.absolute()}\n")

    # Download each image
    success_count = 0
    total_count = len(TEST_IMAGES)

    for filename, url in TEST_IMAGES.items():
        filepath = images_dir / filename

        # Skip if already exists
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename} already exists ({size_kb:.1f} KB)")
            success_count += 1
            continue

        # Download image
        if download_image(url, filepath):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Download complete: {success_count}/{total_count} images")

    if success_count == total_count:
        print("\n✓ All test images ready!")
        print(f"Location: {images_dir.absolute()}")

        # List downloaded images
        print("\nDownloaded images:")
        for img in sorted(images_dir.glob("*.jpg")):
            size_kb = img.stat().st_size / 1024
            print(f"  - {img.name} ({size_kb:.1f} KB)")

        return 0
    else:
        print(f"\n✗ {total_count - success_count} downloads failed")
        print("Check your internet connection and try again")
        return 1

if __name__ == "__main__":
    exit(main())
```

**Run the download script**:

```bash
python download_test_images.py
```

**Phase 1 Checkpoint**:
- [ ] Virtual environment created and activated
- [ ] TensorFlow 2.15.0 installed
- [ ] GPU detected (if available) and configured
- [ ] Verification script passes all tests
- [ ] Test images downloaded successfully

---

## Phase 2: Model Loading & Inspection

### Step 2.1: Load Pre-trained MobileNetV2

Create `load_model.py`:

```python
#!/usr/bin/env python3
"""
Load and inspect pre-trained MobileNetV2 model
Demonstrates keras.applications model loading
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_mobilenetv2():
    """
    Load pre-trained MobileNetV2 from keras.applications

    Returns:
        model: Loaded MobileNetV2 model
    """
    print("Loading MobileNetV2 from keras.applications...")
    print("-" * 60)

    # Load pre-trained model
    # weights='imagenet': Use ImageNet pre-trained weights
    # include_top=True: Include classification head (1000 classes)
    # input_shape: Expected input dimensions (H, W, C)
    model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=True,
        input_shape=(224, 224, 3)
    )

    print(f"\n✓ Model loaded: {model.name}")

    return model

def inspect_model(model):
    """Print model information"""
    print("\n" + "=" * 60)
    print(" Model Information")
    print("=" * 60)

    # Basic info
    print(f"\nModel name:   {model.name}")
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total layers: {len(model.layers)}")

    # Parameter counts
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    print(f"\nParameters:")
    print(f"  Trainable:     {trainable_params:,}")
    print(f"  Non-trainable: {non_trainable_params:,}")
    print(f"  Total:         {total_params:,}")

    # Model size estimate
    # Assuming float32 (4 bytes per parameter)
    size_mb = (total_params * 4) / (1024 * 1024)
    print(f"\nEstimated model size: {size_mb:.1f} MB")

def examine_architecture(model):
    """Examine model architecture details"""
    print("\n" + "=" * 60)
    print(" Architecture Analysis")
    print("=" * 60)

    # Count layers by type
    layer_counts = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1

    print("\nLayer type distribution:")
    for layer_type, count in sorted(layer_counts.items()):
        print(f"  {layer_type:30s}: {count:3d}")

    # Show first few layers
    print("\nFirst 5 layers:")
    for i, layer in enumerate(model.layers[:5]):
        print(f"  {i+1:3d}. {layer.name:40s} ({layer.__class__.__name__})")
        if hasattr(layer, 'output_shape'):
            print(f"       Output shape: {layer.output_shape}")

    # Show last few layers
    print("\nLast 5 layers:")
    for i, layer in enumerate(model.layers[-5:], start=len(model.layers)-4):
        print(f"  {i:3d}. {layer.name:40s} ({layer.__class__.__name__})")
        if hasattr(layer, 'output_shape'):
            print(f"       Output shape: {layer.output_shape}")

def test_forward_pass(model):
    """Test a forward pass with dummy input"""
    print("\n" + "=" * 60)
    print(" Forward Pass Test")
    print("=" * 60)

    # Create random input
    dummy_input = tf.random.normal([1, 224, 224, 3])
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Input dtype:  {dummy_input.dtype}")
    print(f"Input range:  [{tf.reduce_min(dummy_input).numpy():.2f}, {tf.reduce_max(dummy_input).numpy():.2f}]")

    # Run inference
    print("\nRunning forward pass...")
    output = model(dummy_input, training=False)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{tf.reduce_min(output).numpy():.4f}, {tf.reduce_max(output).numpy():.4f}]")
    print(f"Output sum:   {tf.reduce_sum(output).numpy():.6f}")

    # Check if output is valid probabilities
    if output.shape[-1] == 1000:
        print("\n✓ Output has 1000 classes (ImageNet)")
        # Apply softmax to get probabilities
        probs = tf.nn.softmax(output[0])
        print(f"  Probabilities sum: {tf.reduce_sum(probs).numpy():.6f} (should be ~1.0)")
        print(f"  Max probability:   {tf.reduce_max(probs).numpy():.6f}")
        print(f"  Top class index:   {tf.argmax(probs).numpy()}")

def main():
    """Main execution"""
    print("\n" + "#" * 60)
    print("# MobileNetV2 Model Loading and Inspection")
    print("#" * 60)

    # Load model
    model = load_mobilenetv2()

    # Inspect model
    inspect_model(model)
    examine_architecture(model)
    test_forward_pass(model)

    print("\n" + "=" * 60)
    print("✓ Model loading and inspection complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
```

**Run the script**:

```bash
python load_model.py
```

### Step 2.2: Compare Multiple Models

Create `compare_models.py`:

```python
#!/usr/bin/env python3
"""
Compare different keras.applications models
"""

import tensorflow as tf
from tensorflow import keras
import time

# Models to compare
MODELS_TO_TEST = {
    'MobileNetV2': {
        'class': keras.applications.MobileNetV2,
        'preprocess': keras.applications.mobilenet_v2.preprocess_input,
    },
    'ResNet50': {
        'class': keras.applications.ResNet50,
        'preprocess': keras.applications.resnet50.preprocess_input,
    },
    'EfficientNetB0': {
        'class': keras.applications.EfficientNetB0,
        'preprocess': keras.applications.efficientnet.preprocess_input,
    },
}

def load_and_analyze_model(model_name, model_info):
    """Load model and get statistics"""
    print(f"\n{model_name}:")
    print("-" * 60)

    # Load model
    print("  Loading model...")
    start_time = time.time()
    model = model_info['class'](weights='imagenet')
    load_time = time.time() - start_time

    # Get info
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    size_mb = (total_params * 4) / (1024 * 1024)

    print(f"  Load time:    {load_time:.2f}s")
    print(f"  Parameters:   {total_params:,}")
    print(f"  Size:         {size_mb:.1f} MB")
    print(f"  Layers:       {len(model.layers)}")
    print(f"  Input shape:  {model.input_shape}")

    # Test inference speed
    dummy_input = tf.random.normal([1, 224, 224, 3])

    # Warmup
    for _ in range(5):
        _ = model(dummy_input, training=False)

    # Measure
    iterations = 20
    start = time.time()
    for _ in range(iterations):
        _ = model(dummy_input, training=False)
    avg_latency = ((time.time() - start) / iterations) * 1000

    print(f"  Inference:    {avg_latency:.2f}ms")

    return {
        'model_name': model_name,
        'params': total_params,
        'size_mb': size_mb,
        'layers': len(model.layers),
        'latency_ms': avg_latency,
        'load_time_s': load_time,
    }

def main():
    """Compare models"""
    print("\n" + "=" * 60)
    print(" Comparing keras.applications Models")
    print("=" * 60)

    results = []

    for model_name, model_info in MODELS_TO_TEST.items():
        try:
            result = load_and_analyze_model(model_name, model_info)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}")

    # Print comparison table
    print("\n" + "=" * 60)
    print(" Comparison Summary")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Params':>12} {'Size (MB)':>10} {'Layers':>8} {'Latency (ms)':>15}")
    print("-" * 60)

    for result in results:
        print(f"{result['model_name']:<20} {result['params']:>12,} {result['size_mb']:>10.1f} "
              f"{result['layers']:>8} {result['latency_ms']:>15.2f}")

    print("\n")

if __name__ == "__main__":
    main()
```

**Phase 2 Checkpoint**:
- [ ] Successfully loaded MobileNetV2
- [ ] Understood model architecture (layers, parameters)
- [ ] Tested forward pass with dummy input
- [ ] Compared multiple model architectures

---

## Phase 3: Image Preprocessing Pipeline

### Step 3.1: Implement Preprocessing Functions

Create `preprocessing.py`:

```python
#!/usr/bin/env python3
"""
Image preprocessing pipelines for TensorFlow models
Demonstrates multiple preprocessing approaches
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from pathlib import Path

class ImagePreprocessor:
    """
    Production-grade image preprocessing for TensorFlow models
    Supports multiple preprocessing methods
    """

    def __init__(self, target_size=(224, 224), model_name='MobileNetV2'):
        """
        Initialize preprocessor

        Args:
            target_size: Tuple of (height, width) for resizing
            model_name: Model name to get appropriate preprocessing function
        """
        self.target_size = target_size
        self.model_name = model_name

        # Get model-specific preprocessing function
        self.preprocess_fn = self._get_preprocess_function(model_name)

    def _get_preprocess_function(self, model_name):
        """Get preprocessing function for specific model"""
        preprocess_map = {
            'MobileNetV2': keras.applications.mobilenet_v2.preprocess_input,
            'ResNet50': keras.applications.resnet50.preprocess_input,
            'EfficientNetB0': keras.applications.efficientnet.preprocess_input,
            'VGG16': keras.applications.vgg16.preprocess_input,
            'InceptionV3': keras.applications.inception_v3.preprocess_input,
        }

        if model_name not in preprocess_map:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(preprocess_map.keys())}")

        return preprocess_map[model_name]

    def preprocess_with_keras(self, image_path):
        """
        Preprocess image using Keras utilities (high-level API)

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor of shape (1, H, W, C)
        """
        # Load image
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=self.target_size
        )

        # Convert to array
        img_array = keras.preprocessing.image.img_to_array(img)

        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)

        # Apply model-specific preprocessing
        preprocessed = self.preprocess_fn(img_batch)

        return preprocessed

    def preprocess_with_tensorflow(self, image_path):
        """
        Preprocess image using pure TensorFlow APIs (production approach)

        This method is more suitable for:
        - TensorFlow Serving deployment
        - tf.data pipelines
        - Graph optimization

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor of shape (1, H, W, C)
        """
        # Read image file
        image_raw = tf.io.read_file(str(image_path))

        # Decode image (automatically detects format: JPEG, PNG, etc.)
        image = tf.image.decode_image(image_raw, channels=3)

        # Set shape explicitly (required for some operations)
        image.set_shape([None, None, 3])

        # Resize to target size
        image = tf.image.resize(image, self.target_size)

        # Convert to float32
        image = tf.cast(image, tf.float32)

        # Add batch dimension
        image = tf.expand_dims(image, 0)

        # Apply model-specific preprocessing
        preprocessed = self.preprocess_fn(image)

        return preprocessed

    def preprocess_with_pil(self, image_path):
        """
        Preprocess using PIL and NumPy (for comparison)

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor
        """
        # Load with PIL
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize
        img = img.resize(self.target_size, Image.BILINEAR)

        # Convert to NumPy array
        img_array = np.array(img, dtype=np.float32)

        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)

        # Apply preprocessing
        preprocessed = self.preprocess_fn(img_batch)

        return preprocessed

    def preprocess_batch(self, image_paths):
        """
        Preprocess multiple images into a batch

        Args:
            image_paths: List of image file paths

        Returns:
            Batched preprocessed tensor of shape (N, H, W, C)
        """
        preprocessed_images = []

        for image_path in image_paths:
            img = self.preprocess_with_tensorflow(image_path)
            # Remove batch dimension from individual images
            preprocessed_images.append(img[0])

        # Stack into batch
        batch = tf.stack(preprocessed_images, axis=0)

        return batch

def demonstrate_preprocessing(image_path):
    """Demonstrate different preprocessing methods"""
    print(f"\nPreprocessing: {image_path}")
    print("=" * 60)

    preprocessor = ImagePreprocessor(model_name='MobileNetV2')

    # Method 1: Keras utilities
    print("\nMethod 1: Keras utilities")
    keras_result = preprocessor.preprocess_with_keras(image_path)
    print(f"  Shape: {keras_result.shape}")
    print(f"  Dtype: {keras_result.dtype}")
    print(f"  Range: [{keras_result.min():.3f}, {keras_result.max():.3f}]")

    # Method 2: Pure TensorFlow
    print("\nMethod 2: TensorFlow APIs")
    tf_result = preprocessor.preprocess_with_tensorflow(image_path)
    print(f"  Shape: {tf_result.shape}")
    print(f"  Dtype: {tf_result.dtype}")
    print(f"  Range: [{tf.reduce_min(tf_result).numpy():.3f}, {tf.reduce_max(tf_result).numpy():.3f}]")

    # Method 3: PIL/NumPy
    print("\nMethod 3: PIL/NumPy")
    pil_result = preprocessor.preprocess_with_pil(image_path)
    print(f"  Shape: {pil_result.shape}")
    print(f"  Dtype: {pil_result.dtype}")
    print(f"  Range: [{pil_result.min():.3f}, {pil_result.max():.3f}]")

    # Verify consistency
    print("\nConsistency Check:")
    keras_np = keras_result
    tf_np = tf_result.numpy()

    max_diff = np.abs(keras_np - tf_np).max()
    mean_diff = np.abs(keras_np - tf_np).mean()

    print(f"  Max difference (Keras vs TF):  {max_diff:.6f}")
    print(f"  Mean difference (Keras vs TF): {mean_diff:.6f}")

    if max_diff < 0.001:
        print("  ✓ Methods produce consistent results")
    else:
        print("  ⚠ Methods show some differences (may be due to resizing algorithms)")

def demonstrate_batch_preprocessing():
    """Demonstrate batch preprocessing"""
    print("\n" + "=" * 60)
    print(" Batch Preprocessing")
    print("=" * 60)

    # Get all test images
    test_images = sorted(Path("test_images").glob("*.jpg"))

    if not test_images:
        print("\n⚠ No test images found. Run download_test_images.py first.")
        return

    print(f"\nFound {len(test_images)} images")

    preprocessor = ImagePreprocessor()

    # Preprocess as batch
    print("\nPreprocessing batch...")
    batch = preprocessor.preprocess_batch(test_images)

    print(f"\nBatch tensor:")
    print(f"  Shape: {batch.shape}")
    print(f"  Dtype: {batch.dtype}")
    print(f"  Range: [{tf.reduce_min(batch).numpy():.3f}, {tf.reduce_max(batch).numpy():.3f}]")
    print(f"  Memory: {batch.numpy().nbytes / (1024*1024):.2f} MB")

    print("\n✓ Batch preprocessing complete")

def main():
    """Main demonstration"""
    print("\n" + "#" * 60)
    print("# Image Preprocessing Pipeline Demonstration")
    print("#" * 60)

    # Test with first available image
    test_images = list(Path("test_images").glob("*.jpg"))

    if not test_images:
        print("\n✗ No test images found!")
        print("Run: python download_test_images.py")
        return 1

    # Demonstrate different preprocessing methods
    demonstrate_preprocessing(test_images[0])

    # Demonstrate batch preprocessing
    demonstrate_batch_preprocessing()

    print("\n" + "=" * 60)
    print("✓ Preprocessing demonstration complete!")
    print("=" * 60 + "\n")

    return 0

if __name__ == "__main__":
    exit(main())
```

**Run the preprocessing demonstration**:

```bash
python preprocessing.py
```

**Phase 3 Checkpoint**:
- [ ] Implemented multiple preprocessing methods
- [ ] Verified preprocessing consistency
- [ ] Tested batch preprocessing
- [ ] Understood model-specific preprocessing requirements

---

## Phase 4: Inference Execution

### Step 4.1: Basic Inference Implementation

Create `inference.py`:

```python
#!/usr/bin/env python3
"""
TensorFlow model inference implementation
Demonstrates multiple inference approaches
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import urllib.request
from pathlib import Path
import time

class TensorFlowInference:
    """
    Production-grade inference engine for TensorFlow models
    """

    def __init__(self, model_name='MobileNetV2', use_saved_model=False, model_path=None):
        """
        Initialize inference engine

        Args:
            model_name: Model from keras.applications
            use_saved_model: Load from SavedModel format
            model_path: Path to SavedModel directory
        """
        print(f"Initializing {model_name} inference engine...")

        self.model_name = model_name

        # Load model
        if use_saved_model:
            if not model_path:
                raise ValueError("model_path required when use_saved_model=True")
            print(f"  Loading from SavedModel: {model_path}")
            self.model = keras.models.load_model(model_path)
        else:
            print(f"  Loading from keras.applications...")
            model_class = getattr(keras.applications, model_name)
            self.model = model_class(weights='imagenet')

        # Get preprocessing function
        preprocess_module = getattr(
            keras.applications,
            model_name.lower().replace('v2', '_v2').replace('b0', '')
        )
        self.preprocess_fn = preprocess_module.preprocess_input

        # Load labels
        self.labels = self._load_imagenet_labels()

        # Create optimized inference function
        self._create_optimized_inference()

        print("  ✓ Initialization complete\n")

    def _load_imagenet_labels(self):
        """Load ImageNet class labels"""
        labels_path = Path("imagenet_class_index.json")

        if not labels_path.exists():
            print("  Downloading ImageNet labels...")
            url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
            urllib.request.urlretrieve(url, labels_path)

        with open(labels_path) as f:
            class_idx = json.load(f)
            labels = [class_idx[str(i)][1] for i in range(len(class_idx))]

        return labels

    def _create_optimized_inference(self):
        """Create @tf.function optimized inference function"""
        @tf.function
        def predict_fn(inputs):
            return self.model(inputs, training=False)

        self.predict_fn = predict_fn

        # Warmup
        print("  Warming up inference function...")
        dummy = tf.random.normal([1, 224, 224, 3])
        for _ in range(5):
            _ = self.predict_fn(dummy)

    def preprocess_image(self, image_path):
        """Preprocess single image"""
        # Load image
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        preprocessed = self.preprocess_fn(img_batch)

        return tf.convert_to_tensor(preprocessed, dtype=tf.float32)

    def predict_basic(self, image_path, top_k=5):
        """
        Basic inference using model.predict()

        Args:
            image_path: Path to image
            top_k: Number of top predictions

        Returns:
            results: List of (class_name, probability) tuples
            latency_ms: Inference time in milliseconds
        """
        # Preprocess
        preprocessed = self.preprocess_image(image_path)

        # Run inference
        start = time.time()
        predictions = self.model.predict(preprocessed, verbose=0)
        latency_ms = (time.time() - start) * 1000

        # Get top k
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]

        results = [
            (self.labels[idx], float(predictions[0][idx]))
            for idx in top_indices
        ]

        return results, latency_ms

    def predict_optimized(self, image_path, top_k=5):
        """
        Optimized inference using @tf.function

        Args:
            image_path: Path to image
            top_k: Number of top predictions

        Returns:
            results: List of (class_name, probability) tuples
            latency_ms: Inference time in milliseconds
        """
        # Preprocess
        preprocessed = self.preprocess_image(image_path)

        # Run optimized inference
        start = time.time()
        predictions = self.predict_fn(preprocessed)
        latency_ms = (time.time() - start) * 1000

        # Get top k
        top_indices = tf.argsort(predictions[0], direction='DESCENDING')[:top_k]
        top_probs = tf.gather(predictions[0], top_indices)

        results = [
            (self.labels[idx.numpy()], float(prob.numpy()))
            for idx, prob in zip(top_indices, top_probs)
        ]

        return results, latency_ms

    def predict_batch(self, image_paths, top_k=5):
        """
        Batch inference for multiple images

        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image

        Returns:
            all_results: List of result lists
            total_latency_ms: Total inference time
            per_image_latency_ms: Average latency per image
        """
        # Preprocess batch
        batch_images = []
        for path in image_paths:
            img = self.preprocess_image(path)
            batch_images.append(img[0])  # Remove batch dim

        batch = tf.stack(batch_images, axis=0)

        # Run batch inference
        start = time.time()
        predictions = self.predict_fn(batch)
        total_latency_ms = (time.time() - start) * 1000
        per_image_latency_ms = total_latency_ms / len(image_paths)

        # Process results for each image
        all_results = []
        for i in range(len(image_paths)):
            top_indices = tf.argsort(predictions[i], direction='DESCENDING')[:top_k]
            top_probs = tf.gather(predictions[i], top_indices)

            results = [
                (self.labels[idx.numpy()], float(prob.numpy()))
                for idx, prob in zip(top_indices, top_probs)
            ]
            all_results.append(results)

        return all_results, total_latency_ms, per_image_latency_ms

def demonstrate_inference():
    """Demonstrate different inference methods"""
    print("\n" + "=" * 60)
    print(" Inference Method Comparison")
    print("=" * 60)

    # Get test image
    test_images = list(Path("test_images").glob("*.jpg"))
    if not test_images:
        print("\n✗ No test images found!")
        return

    test_image = test_images[0]
    print(f"\nTest image: {test_image}")

    # Initialize inference engine
    engine = TensorFlowInference(model_name='MobileNetV2')

    # Method 1: Basic predict()
    print("\nMethod 1: model.predict()")
    results, latency = engine.predict_basic(test_image, top_k=3)
    print(f"  Latency: {latency:.2f}ms")
    print("  Top 3 predictions:")
    for class_name, prob in results:
        print(f"    {class_name:<30} {prob*100:>6.2f}%")

    # Method 2: Optimized with @tf.function
    print("\nMethod 2: @tf.function optimized")
    results, latency = engine.predict_optimized(test_image, top_k=3)
    print(f"  Latency: {latency:.2f}ms")
    print("  Top 3 predictions:")
    for class_name, prob in results:
        print(f"    {class_name:<30} {prob*100:>6.2f}%")

    # Benchmark
    print("\n" + "-" * 60)
    print(" Performance Benchmark")
    print("-" * 60)

    iterations = 50

    # Benchmark basic
    latencies = []
    for _ in range(iterations):
        _, lat = engine.predict_basic(test_image)
        latencies.append(lat)
    avg_basic = np.mean(latencies)

    # Benchmark optimized
    latencies = []
    for _ in range(iterations):
        _, lat = engine.predict_optimized(test_image)
        latencies.append(lat)
    avg_optimized = np.mean(latencies)

    print(f"\nBasic predict():      {avg_basic:.2f}ms")
    print(f"@tf.function:         {avg_optimized:.2f}ms")
    print(f"Speedup:              {avg_basic/avg_optimized:.2f}x")

def demonstrate_batch_inference():
    """Demonstrate batch inference"""
    print("\n" + "=" * 60)
    print(" Batch Inference Demonstration")
    print("=" * 60)

    test_images = list(Path("test_images").glob("*.jpg"))
    if len(test_images) < 2:
        print("\n⚠ Need multiple images for batch inference demo")
        return

    engine = TensorFlowInference(model_name='MobileNetV2')

    # Process batch
    print(f"\nProcessing batch of {len(test_images)} images...")
    all_results, total_lat, per_img_lat = engine.predict_batch(test_images, top_k=3)

    print(f"\nBatch statistics:")
    print(f"  Total latency:      {total_lat:.2f}ms")
    print(f"  Per-image latency:  {per_img_lat:.2f}ms")
    print(f"  Throughput:         {1000/per_img_lat:.2f} images/sec")

    print("\nPredictions:")
    for image_path, results in zip(test_images, all_results):
        print(f"\n  {image_path.name}:")
        for class_name, prob in results[:3]:
            print(f"    {class_name:<30} {prob*100:>6.2f}%")

def main():
    """Main execution"""
    print("\n" + "#" * 60)
    print("# TensorFlow Model Inference")
    print("#" * 60)

    demonstrate_inference()
    demonstrate_batch_inference()

    print("\n" + "=" * 60)
    print("✓ Inference demonstration complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
```

**Run inference**:

```bash
python inference.py
```

**Phase 4 Checkpoint**:
- [ ] Implemented basic inference with model.predict()
- [ ] Implemented optimized inference with @tf.function
- [ ] Tested batch inference
- [ ] Measured performance differences

---

## Phase 5: Performance Optimization

### Step 5.1: Comprehensive Performance Benchmarking

Create `performance_benchmark.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive performance benchmarking for TensorFlow inference
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from collections import defaultdict

class PerformanceBenchmark:
    """
    Comprehensive inference performance benchmarking
    """

    def __init__(self, model_name='MobileNetV2'):
        """Initialize benchmark with model"""
        print(f"Initializing benchmark for {model_name}...")

        # Load model
        model_class = getattr(keras.applications, model_name)
        self.model = model_class(weights='imagenet')
        self.model_name = model_name

        # Create optimized inference function
        @tf.function
        def predict_optimized(inputs):
            return self.model(inputs, training=False)

        self.predict_optimized = predict_optimized

        print("✓ Benchmark initialized\n")

    def warmup(self, batch_size=1, iterations=10):
        """Warmup model to ensure consistent measurements"""
        print(f"Warming up (batch_size={batch_size}, iterations={iterations})...")

        dummy_input = tf.random.normal([batch_size, 224, 224, 3])

        for _ in range(iterations):
            _ = self.predict_optimized(dummy_input)

        print("✓ Warmup complete\n")

    def benchmark_latency(self, batch_size=1, iterations=100):
        """
        Benchmark inference latency

        Args:
            batch_size: Batch size to test
            iterations: Number of iterations

        Returns:
            Dictionary with latency statistics
        """
        dummy_input = tf.random.normal([batch_size, 224, 224, 3])

        latencies = []

        for _ in range(iterations):
            start = time.time()
            _ = self.predict_optimized(dummy_input)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        latencies = np.array(latencies)

        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
        }

    def benchmark_throughput(self, batch_size=1, duration_seconds=5):
        """
        Benchmark throughput (images/second)

        Args:
            batch_size: Batch size to test
            duration_seconds: How long to run benchmark

        Returns:
            Dictionary with throughput statistics
        """
        dummy_input = tf.random.normal([batch_size, 224, 224, 3])

        start_time = time.time()
        iterations = 0

        while (time.time() - start_time) < duration_seconds:
            _ = self.predict_optimized(dummy_input)
            iterations += 1

        elapsed = time.time() - start_time
        total_images = iterations * batch_size
        throughput = total_images / elapsed

        return {
            'throughput_imgs_per_sec': throughput,
            'total_images': total_images,
            'total_iterations': iterations,
            'duration_seconds': elapsed,
        }

    def benchmark_batch_sizes(self, batch_sizes=[1, 2, 4, 8, 16, 32], iterations=50):
        """
        Benchmark performance across different batch sizes

        Args:
            batch_sizes: List of batch sizes to test
            iterations: Iterations per batch size

        Returns:
            Dictionary mapping batch_size to statistics
        """
        print("Benchmarking batch sizes...")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Iterations: {iterations}\n")

        results = {}

        for batch_size in batch_sizes:
            print(f"  Testing batch_size={batch_size}...", end='', flush=True)

            # Warmup for this batch size
            self.warmup(batch_size=batch_size, iterations=5)

            # Benchmark latency
            latency_stats = self.benchmark_latency(batch_size=batch_size, iterations=iterations)

            # Calculate per-image metrics
            per_image_latency = latency_stats['mean'] / batch_size
            throughput = (batch_size * 1000) / latency_stats['mean']

            results[batch_size] = {
                'batch_latency_ms': latency_stats['mean'],
                'batch_latency_std': latency_stats['std'],
                'per_image_latency_ms': per_image_latency,
                'throughput_imgs_per_sec': throughput,
                'p95_latency_ms': latency_stats['p95'],
                'p99_latency_ms': latency_stats['p99'],
            }

            print(f" {latency_stats['mean']:.2f}ms")

        return results

    def print_batch_size_results(self, results):
        """Print batch size benchmark results in table format"""
        print("\n" + "=" * 80)
        print(" Batch Size Performance Analysis")
        print("=" * 80)

        print(f"\n{'Batch':<7} {'Batch Lat':<12} {'Per-Image':<12} {'Throughput':<15} {'P95':<10} {'P99':<10}")
        print(f"{'Size':<7} {'(ms)':<12} {'Lat (ms)':<12} {'(imgs/sec)':<15} {'(ms)':<10} {'(ms)':<10}")
        print("-" * 80)

        for batch_size in sorted(results.keys()):
            stats = results[batch_size]
            print(f"{batch_size:<7} "
                  f"{stats['batch_latency_ms']:<12.2f} "
                  f"{stats['per_image_latency_ms']:<12.2f} "
                  f"{stats['throughput_imgs_per_sec']:<15.2f} "
                  f"{stats['p95_latency_ms']:<10.2f} "
                  f"{stats['p99_latency_ms']:<10.2f}")

        # Find optimal batch size
        print("\n" + "-" * 80)

        best_throughput_batch = max(results.keys(), key=lambda k: results[k]['throughput_imgs_per_sec'])
        best_latency_batch = min(results.keys(), key=lambda k: results[k]['per_image_latency_ms'])

        print(f"Best throughput:     batch_size={best_throughput_batch} "
              f"({results[best_throughput_batch]['throughput_imgs_per_sec']:.2f} imgs/sec)")
        print(f"Best per-img latency: batch_size={best_latency_batch} "
              f"({results[best_latency_batch]['per_image_latency_ms']:.2f}ms)")

def compare_optimization_methods():
    """Compare different optimization methods"""
    print("\n" + "=" * 80)
    print(" Optimization Method Comparison")
    print("=" * 80)

    model = keras.applications.MobileNetV2(weights='imagenet')
    dummy_input = tf.random.normal([1, 224, 224, 3])

    # Method 1: Basic model.predict()
    print("\nMethod 1: Basic model.predict()")
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)

    latencies = []
    for _ in range(50):
        start = time.time()
        _ = model.predict(dummy_input, verbose=0)
        latencies.append((time.time() - start) * 1000)

    predict_mean = np.mean(latencies)
    predict_std = np.std(latencies)
    print(f"  Latency: {predict_mean:.2f} ± {predict_std:.2f}ms")

    # Method 2: model() direct call
    print("\nMethod 2: model() direct call")
    for _ in range(10):
        _ = model(dummy_input, training=False)

    latencies = []
    for _ in range(50):
        start = time.time()
        _ = model(dummy_input, training=False)
        latencies.append((time.time() - start) * 1000)

    call_mean = np.mean(latencies)
    call_std = np.std(latencies)
    print(f"  Latency: {call_mean:.2f} ± {call_std:.2f}ms")

    # Method 3: @tf.function wrapped
    @tf.function
    def predict_optimized(inputs):
        return model(inputs, training=False)

    print("\nMethod 3: @tf.function optimized")
    for _ in range(10):
        _ = predict_optimized(dummy_input)

    latencies = []
    for _ in range(50):
        start = time.time()
        _ = predict_optimized(dummy_input)
        latencies.append((time.time() - start) * 1000)

    tf_func_mean = np.mean(latencies)
    tf_func_std = np.std(latencies)
    print(f"  Latency: {tf_func_mean:.2f} ± {tf_func_std:.2f}ms")

    # Summary
    print("\n" + "-" * 80)
    print(" Performance Comparison")
    print("-" * 80)
    print(f"model.predict():     {predict_mean:.2f}ms (baseline)")
    print(f"model():             {call_mean:.2f}ms ({predict_mean/call_mean:.2f}x speedup)")
    print(f"@tf.function:        {tf_func_mean:.2f}ms ({predict_mean/tf_func_mean:.2f}x speedup)")

def main():
    """Run comprehensive benchmarks"""
    print("\n" + "#" * 80)
    print("# TensorFlow Inference Performance Benchmark")
    print("#" * 80)

    # Compare optimization methods
    compare_optimization_methods()

    # Comprehensive batch size benchmark
    print("\n")
    benchmark = PerformanceBenchmark(model_name='MobileNetV2')
    benchmark.warmup(batch_size=1, iterations=20)

    results = benchmark.benchmark_batch_sizes(
        batch_sizes=[1, 2, 4, 8, 16, 32],
        iterations=50
    )

    benchmark.print_batch_size_results(results)

    print("\n" + "=" * 80)
    print("✓ Performance benchmark complete!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
```

**Run the benchmark**:

```bash
python performance_benchmark.py
```

**Phase 5 Checkpoint**:
- [ ] Benchmarked different optimization methods
- [ ] Tested multiple batch sizes
- [ ] Measured latency statistics (mean, p95, p99)
- [ ] Identified optimal batch size for your hardware

---

## Phase 6: SavedModel Export & Serving

### Step 6.1: Export Model in SavedModel Format

Create `export_savedmodel.py`:

```python
#!/usr/bin/env python3
"""
Export TensorFlow models in SavedModel format
Production deployment format for TensorFlow Serving
"""

import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path
import json
import time

class ModelExporter:
    """
    Export TensorFlow models for production deployment
    """

    def __init__(self, model_name='MobileNetV2', base_export_path='saved_models'):
        """
        Initialize exporter

        Args:
            model_name: Model to export
            base_export_path: Base directory for saved models
        """
        self.model_name = model_name
        self.base_export_path = Path(base_export_path)

        # Load model
        print(f"Loading {model_name}...")
        model_class = getattr(keras.applications, model_name)
        self.model = model_class(weights='imagenet')

        print(f"✓ Model loaded\n")

    def export_savedmodel(self, version=1, include_preprocessing=False):
        """
        Export model in SavedModel format

        Args:
            version: Version number for this export
            include_preprocessing: Whether to include preprocessing in model

        Returns:
            export_path: Path where model was saved
        """
        # Create export path: saved_models/model_name/version/
        export_path = self.base_export_path / self.model_name.lower() / str(version)
        export_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting {self.model_name} to SavedModel format...")
        print(f"Export path: {export_path}")
        print(f"Version: {version}")
        print(f"Include preprocessing: {include_preprocessing}\n")

        if include_preprocessing:
            # Wrap model with preprocessing
            model_to_save = self._create_model_with_preprocessing()
        else:
            model_to_save = self.model

        # Save model
        start_time = time.time()
        model_to_save.save(
            str(export_path),
            save_format='tf',
            include_optimizer=False,  # Don't save optimizer (inference only)
        )
        save_time = time.time() - start_time

        print(f"✓ Model saved in {save_time:.2f}s")

        # Get export size
        total_size = sum(f.stat().st_size for f in export_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)

        print(f"  Export size: {size_mb:.1f} MB")

        # Save metadata
        self._save_metadata(export_path, version, include_preprocessing)

        # Verify export
        self._verify_export(export_path)

        return export_path

    def _create_model_with_preprocessing(self):
        """
        Create model that includes preprocessing
        Useful for deployment where client sends raw images
        """
        # Get preprocessing function
        preprocess_module = getattr(
            keras.applications,
            self.model_name.lower().replace('v2', '_v2').replace('b0', '')
        )
        preprocess_fn = preprocess_module.preprocess_input

        # Create preprocessing layers
        inputs = keras.Input(shape=(224, 224, 3), dtype=tf.uint8, name='image_input')

        # Convert to float32 and preprocess
        x = tf.cast(inputs, tf.float32)
        x = preprocess_fn(x)

        # Apply model
        outputs = self.model(x, training=False)

        # Create combined model
        model_with_preproc = keras.Model(inputs=inputs, outputs=outputs, name=f'{self.model_name}_with_preprocessing')

        return model_with_preproc

    def _save_metadata(self, export_path, version, include_preprocessing):
        """Save metadata about the exported model"""
        metadata = {
            'model_name': self.model_name,
            'version': version,
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tensorflow_version': tf.__version__,
            'include_preprocessing': include_preprocessing,
            'input_shape': str(self.model.input_shape),
            'output_shape': str(self.model.output_shape),
            'num_classes': 1000,
            'dataset': 'ImageNet',
        }

        metadata_path = export_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata saved: {metadata_path}")

    def _verify_export(self, export_path):
        """Verify the exported SavedModel works correctly"""
        print(f"\nVerifying exported model...")

        # Load the exported model
        loaded = keras.models.load_model(str(export_path))

        # Test with dummy input
        dummy_input = tf.random.normal([1, 224, 224, 3])

        # Original model prediction
        orig_output = self.model(dummy_input, training=False)

        # Loaded model prediction
        loaded_output = loaded(dummy_input, training=False)

        # Compare outputs
        max_diff = tf.reduce_max(tf.abs(orig_output - loaded_output)).numpy()

        if max_diff < 1e-5:
            print(f"✓ Verification passed (max difference: {max_diff:.2e})")
        else:
            print(f"⚠ Verification warning: max difference = {max_diff:.2e}")

    def inspect_savedmodel_structure(self, export_path):
        """Display SavedModel directory structure"""
        print(f"\nSavedModel Directory Structure:")
        print(f"{export_path}/")

        for root, dirs, files in os.walk(export_path):
            level = root.replace(str(export_path), '').count(os.sep)
            indent = '  ' * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")

            subindent = '  ' * (level + 2)
            for file in files:
                file_path = Path(root) / file
                size_kb = file_path.stat().st_size / 1024
                print(f"{subindent}{file} ({size_kb:.1f} KB)")

def demonstrate_savedmodel_export():
    """Demonstrate SavedModel export"""
    print("\n" + "=" * 80)
    print(" SavedModel Export Demonstration")
    print("=" * 80 + "\n")

    exporter = ModelExporter(model_name='MobileNetV2')

    # Export version 1 without preprocessing
    print("Export 1: Model without preprocessing")
    print("-" * 80)
    export_path_v1 = exporter.export_savedmodel(version=1, include_preprocessing=False)
    exporter.inspect_savedmodel_structure(export_path_v1)

    # Export version 2 with preprocessing
    print("\n" + "=" * 80)
    print("Export 2: Model with preprocessing")
    print("-" * 80)
    export_path_v2 = exporter.export_savedmodel(version=2, include_preprocessing=True)
    exporter.inspect_savedmodel_structure(export_path_v2)

    print("\n" + "=" * 80)
    print("✓ SavedModel exports complete!")
    print("=" * 80)

    return export_path_v1, export_path_v2

def inspect_savedmodel_cli(export_path):
    """
    Inspect SavedModel using Python API
    (Alternative to saved_model_cli command-line tool)
    """
    print(f"\nInspecting SavedModel: {export_path}")
    print("=" * 80)

    # Load SavedModel
    loaded = tf.saved_model.load(str(export_path))

    # Show available signatures
    print("\nAvailable signatures:")
    for sig_name in loaded.signatures.keys():
        print(f"  - {sig_name}")

    # Get serving signature
    if 'serving_default' in loaded.signatures:
        serving_fn = loaded.signatures['serving_default']

        print("\nServing signature details:")
        print(f"  Inputs:")
        for input_name, input_spec in serving_fn.structured_input_signature[1].items():
            print(f"    {input_name}: shape={input_spec.shape}, dtype={input_spec.dtype}")

        print(f"  Outputs:")
        for output_name, output_spec in serving_fn.structured_outputs.items():
            print(f"    {output_name}: shape={output_spec.shape}, dtype={output_spec.dtype}")

def test_savedmodel_loading(export_path):
    """Test loading and using SavedModel"""
    print(f"\nTesting SavedModel from: {export_path}")
    print("=" * 80)

    # Load model
    print("Loading SavedModel...")
    loaded_model = keras.models.load_model(str(export_path))

    print(f"✓ Model loaded successfully")
    print(f"  Model name: {loaded_model.name}")
    print(f"  Input shape: {loaded_model.input_shape}")
    print(f"  Output shape: {loaded_model.output_shape}")

    # Test inference
    print("\nTesting inference...")
    dummy_input = tf.random.normal([1, 224, 224, 3])

    start = time.time()
    output = loaded_model(dummy_input, training=False)
    latency = (time.time() - start) * 1000

    print(f"  Inference latency: {latency:.2f}ms")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{tf.reduce_min(output).numpy():.4f}, {tf.reduce_max(output).numpy():.4f}]")

    print("\n✓ SavedModel works correctly!")

def main():
    """Main execution"""
    print("\n" + "#" * 80)
    print("# TensorFlow SavedModel Export")
    print("#" * 80)

    # Export models
    export_path_v1, export_path_v2 = demonstrate_savedmodel_export()

    # Inspect models
    print("\n" + "=" * 80)
    print(" SavedModel Inspection")
    print("=" * 80)

    inspect_savedmodel_cli(export_path_v1)

    # Test loading
    print("\n" + "=" * 80)
    print(" SavedModel Loading Test")
    print("=" * 80)

    test_savedmodel_loading(export_path_v1)

    print("\n" + "=" * 80)
    print("✓ SavedModel export and inspection complete!")
    print("=" * 80 + "\n")

    print("Your models are ready for deployment with TensorFlow Serving!")
    print(f"Model locations:")
    print(f"  Version 1 (no preprocessing): {export_path_v1}")
    print(f"  Version 2 (with preprocessing): {export_path_v2}\n")

if __name__ == "__main__":
    main()
```

**Run the export**:

```bash
python export_savedmodel.py
```

**Phase 6 Checkpoint**:
- [ ] Exported model in SavedModel format
- [ ] Created model with embedded preprocessing
- [ ] Inspected SavedModel structure
- [ ] Verified saved model loads and works correctly

---

## Phase 7: Production Application

### Step 7.1: Complete Production-Ready Classifier

Create `production_classifier.py`:

```python
#!/usr/bin/env python3
"""
Production-Ready TensorFlow Image Classifier
Complete CLI application with error handling, logging, and monitoring hooks
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import json
import time
import sys
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class InferenceResult:
    """Result from image classification inference"""
    image_path: str
    predictions: List[Dict[str, any]]
    inference_time_ms: float
    preprocessing_time_ms: float
    total_time_ms: float
    model_name: str
    timestamp: str

    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self):
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class ProductionImageClassifier:
    """
    Production-grade TensorFlow image classifier

    Features:
    - Multiple model support
    - SavedModel loading
    - Error handling
    - Performance monitoring
    - Batch processing
    - JSON output
    """

    def __init__(
        self,
        model_name: str = 'MobileNetV2',
        use_saved_model: bool = False,
        model_path: Optional[str] = None,
        top_k: int = 5,
        batch_size: int = 1
    ):
        """
        Initialize classifier

        Args:
            model_name: Model from keras.applications
            use_saved_model: Load from SavedModel format
            model_path: Path to SavedModel
            top_k: Number of top predictions
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.use_saved_model = use_saved_model
        self.model_path = model_path
        self.top_k = top_k
        self.batch_size = batch_size

        logger.info(f"Initializing {model_name} classifier...")
        logger.info(f"  Top-K: {top_k}")
        logger.info(f"  Batch size: {batch_size}")

        # Load model
        self._load_model()

        # Load labels
        self.labels = self._load_labels()

        # Create optimized inference function
        self._create_inference_function()

        logger.info("✓ Classifier ready\n")

    def _load_model(self):
        """Load model from keras.applications or SavedModel"""
        try:
            if self.use_saved_model:
                if not self.model_path:
                    raise ValueError("model_path required when use_saved_model=True")

                logger.info(f"Loading SavedModel from: {self.model_path}")
                self.model = keras.models.load_model(self.model_path)

                # Get preprocessing function (assume MobileNetV2 for saved models)
                self.preprocess_fn = keras.applications.mobilenet_v2.preprocess_input
            else:
                logger.info(f"Loading {self.model_name} from keras.applications")
                model_class = getattr(keras.applications, self.model_name)
                self.model = model_class(weights='imagenet')

                # Get model-specific preprocessing
                preprocess_module = getattr(
                    keras.applications,
                    self.model_name.lower().replace('v2', '_v2').replace('b0', '')
                )
                self.preprocess_fn = preprocess_module.preprocess_input

            logger.info(f"✓ Model loaded: {self.model.name}")
            logger.info(f"  Input shape: {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_labels(self) -> List[str]:
        """Load ImageNet class labels"""
        labels_path = Path("imagenet_class_index.json")

        try:
            if not labels_path.exists():
                logger.info("Downloading ImageNet labels...")
                url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
                urllib.request.urlretrieve(url, labels_path)

            with open(labels_path) as f:
                class_idx = json.load(f)
                labels = [class_idx[str(i)][1] for i in range(len(class_idx))]

            logger.info(f"✓ Loaded {len(labels)} class labels")
            return labels

        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            raise

    def _create_inference_function(self):
        """Create optimized @tf.function for inference"""
        @tf.function
        def predict_fn(inputs):
            return self.model(inputs, training=False)

        self.predict_fn = predict_fn

        # Warmup
        logger.info("Warming up inference function...")
        dummy_input = tf.random.normal([1, 224, 224, 3])
        for _ in range(5):
            _ = self.predict_fn(dummy_input)

        logger.info("✓ Inference function ready")

    def preprocess_image(self, image_path: Path) -> tf.Tensor:
        """
        Preprocess single image

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            img = keras.preprocessing.image.load_img(
                image_path,
                target_size=(224, 224)
            )

            # Convert to array
            img_array = keras.preprocessing.image.img_to_array(img)

            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)

            # Preprocess
            preprocessed = self.preprocess_fn(img_batch)

            # Convert to tensor
            return tf.convert_to_tensor(preprocessed, dtype=tf.float32)

        except Exception as e:
            logger.error(f"Failed to preprocess {image_path}: {e}")
            raise

    def classify_image(self, image_path: str) -> InferenceResult:
        """
        Classify a single image

        Args:
            image_path: Path to image file

        Returns:
            InferenceResult with predictions
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Classifying: {image_path}")

        # Preprocessing
        preprocess_start = time.time()
        preprocessed = self.preprocess_image(image_path)
        preprocess_time_ms = (time.time() - preprocess_start) * 1000

        # Inference
        inference_start = time.time()
        predictions = self.predict_fn(preprocessed)
        inference_time_ms = (time.time() - inference_start) * 1000

        total_time_ms = preprocess_time_ms + inference_time_ms

        # Get top-k predictions
        top_k_indices = tf.argsort(predictions[0], direction='DESCENDING')[:self.top_k]
        top_k_probs = tf.gather(predictions[0], top_k_indices)

        # Format predictions
        prediction_list = []
        for idx, prob in zip(top_k_indices.numpy(), top_k_probs.numpy()):
            prediction_list.append({
                'class_id': int(idx),
                'class_name': self.labels[idx],
                'probability': float(prob),
                'confidence_percent': float(prob * 100)
            })

        # Create result
        result = InferenceResult(
            image_path=str(image_path),
            predictions=prediction_list,
            inference_time_ms=inference_time_ms,
            preprocessing_time_ms=preprocess_time_ms,
            total_time_ms=total_time_ms,
            model_name=self.model_name,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        logger.info(f"  Top prediction: {prediction_list[0]['class_name']} "
                   f"({prediction_list[0]['confidence_percent']:.2f}%)")
        logger.info(f"  Timing: preprocess={preprocess_time_ms:.2f}ms, "
                   f"inference={inference_time_ms:.2f}ms, "
                   f"total={total_time_ms:.2f}ms")

        return result

    def classify_batch(self, image_paths: List[str]) -> List[InferenceResult]:
        """
        Classify multiple images

        Args:
            image_paths: List of image file paths

        Returns:
            List of InferenceResults
        """
        logger.info(f"Classifying batch of {len(image_paths)} images")

        results = []

        for image_path in image_paths:
            try:
                result = self.classify_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify {image_path}: {e}")
                # Continue with other images

        logger.info(f"✓ Batch complete: {len(results)}/{len(image_paths)} succeeded")

        return results

def print_result(result: InferenceResult, format: str = 'text'):
    """
    Print classification result

    Args:
        result: InferenceResult to print
        format: Output format ('text' or 'json')
    """
    if format == 'json':
        print(result.to_json())
    else:
        print(f"\nImage: {result.image_path}")
        print(f"Model: {result.model_name}")
        print(f"Time:  {result.timestamp}")
        print(f"\nTop {len(result.predictions)} Predictions:")
        print("-" * 60)

        for i, pred in enumerate(result.predictions, 1):
            print(f"  {i}. {pred['class_name']:<30} {pred['confidence_percent']:>6.2f}%")

        print(f"\nPerformance:")
        print(f"  Preprocessing: {result.preprocessing_time_ms:>6.2f}ms")
        print(f"  Inference:     {result.inference_time_ms:>6.2f}ms")
        print(f"  Total:         {result.total_time_ms:>6.2f}ms")
        print()

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Production TensorFlow Image Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Classify single image
  python production_classifier.py image.jpg

  # Use different model
  python production_classifier.py image.jpg --model ResNet50

  # Get top 10 predictions
  python production_classifier.py image.jpg --top-k 10

  # Use SavedModel
  python production_classifier.py image.jpg --saved-model saved_models/mobilenetv2/1

  # Output as JSON
  python production_classifier.py image.jpg --format json

  # Classify multiple images
  python production_classifier.py image1.jpg image2.jpg image3.jpg
        '''
    )

    parser.add_argument(
        'images',
        type=str,
        nargs='+',
        help='Path(s) to image file(s)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='MobileNetV2',
        help='Model name from keras.applications (default: MobileNetV2)'
    )

    parser.add_argument(
        '--saved-model',
        type=str,
        default=None,
        help='Path to SavedModel directory (optional)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions (default: 5)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (optional, prints to stdout if not specified)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize classifier
        classifier = ProductionImageClassifier(
            model_name=args.model,
            use_saved_model=args.saved_model is not None,
            model_path=args.saved_model,
            top_k=args.top_k
        )

        # Classify images
        results = classifier.classify_batch(args.images)

        # Output results
        if args.output:
            output_path = Path(args.output)
            logger.info(f"Saving results to: {output_path}")

            with open(output_path, 'w') as f:
                if args.format == 'json':
                    json.dump([r.to_dict() for r in results], f, indent=2)
                else:
                    for result in results:
                        f.write(f"\n{'='*60}\n")
                        print_result(result, format='text')
                        # Redirect print to file would require more work

            logger.info(f"✓ Results saved to {output_path}")
        else:
            # Print to stdout
            for result in results:
                print_result(result, format=args.format)

        return 0

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Usage examples**:

```bash
# Basic usage
python production_classifier.py test_images/elephant.jpg

# Different model
python production_classifier.py test_images/tiger.jpg --model ResNet50

# Top 10 predictions
python production_classifier.py test_images/airplane.jpg --top-k 10

# Use SavedModel
python production_classifier.py test_images/elephant.jpg --saved-model saved_models/mobilenetv2/1

# JSON output
python production_classifier.py test_images/elephant.jpg --format json

# Multiple images
python production_classifier.py test_images/*.jpg

# Save results to file
python production_classifier.py test_images/elephant.jpg --output results.json --format json

# Verbose logging
python production_classifier.py test_images/elephant.jpg --verbose
```

**Phase 7 Checkpoint**:
- [ ] Implemented production CLI application
- [ ] Added comprehensive error handling
- [ ] Implemented batch processing
- [ ] Added JSON output support
- [ ] Tested with various model configurations

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: GPU Not Detected

**Symptoms**:
```
GPUs available: 0
```

**Solutions**:
1. Check NVIDIA driver installation:
   ```bash
   nvidia-smi
   ```

2. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

3. Install GPU-enabled TensorFlow:
   ```bash
   pip install tensorflow[and-cuda]
   ```

4. Check TensorFlow GPU support:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

#### Issue 2: Out of Memory Errors

**Symptoms**:
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions**:
1. Enable GPU memory growth:
   ```python
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

2. Reduce batch size
3. Use smaller model (MobileNetV2 instead of ResNet50)

#### Issue 3: Slow Inference Performance

**Symptoms**:
- Inference takes > 100ms on GPU
- No speedup with batch processing

**Solutions**:
1. Use `@tf.function` decoration
2. Ensure proper warmup (10-20 iterations)
3. Check device placement:
   ```python
   tf.debugging.set_log_device_placement(True)
   ```

4. Enable XLA compilation:
   ```python
   @tf.function(jit_compile=True)
   def predict(x):
       return model(x, training=False)
   ```

#### Issue 4: Model Download Failures

**Symptoms**:
```
URLError: <urlopen error [Errno 11001] getaddrinfo failed>
```

**Solutions**:
1. Check internet connection
2. Use proxy if behind firewall:
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   ```

3. Download manually and load from file

#### Issue 5: Preprocessing Errors

**Symptoms**:
```
ValueError: Image size should be (224, 224)
```

**Solutions**:
1. Verify image file is valid
2. Check target size matches model requirements
3. Handle corrupted images:
   ```python
   try:
       img = keras.preprocessing.image.load_img(path, target_size=(224, 224))
   except Exception as e:
       logger.error(f"Failed to load image: {e}")
       return None
   ```

---

## Best Practices

### 1. Model Loading
- Cache loaded models to avoid reloading
- Use SavedModel format for production
- Include preprocessing in SavedModel when possible

### 2. Preprocessing
- Use TensorFlow ops for preprocessing (not NumPy) when possible
- Batch preprocessing for multiple images
- Validate image dimensions before preprocessing

### 3. Inference Optimization
- Always use `@tf.function` for production
- Warmup models before benchmarking
- Use appropriate batch sizes for your hardware
- Monitor GPU utilization

### 4. Error Handling
- Validate input images exist and are readable
- Handle model loading failures gracefully
- Provide clear error messages
- Log errors for debugging

### 5. Production Deployment
- Use SavedModel format
- Version your models (1, 2, 3...)
- Include metadata (model name, version, timestamp)
- Implement health checks
- Monitor inference latency and throughput

### 6. Performance Monitoring
- Track inference latency (p50, p95, p99)
- Monitor GPU memory usage
- Log batch sizes and throughput
- Set up alerts for performance degradation

---

## Validation Checklist

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] TensorFlow 2.15.0 installed
- [ ] GPU configured (if available)
- [ ] Test images downloaded

### Model Loading
- [ ] Successfully loaded MobileNetV2
- [ ] Inspected model architecture
- [ ] Counted parameters and layers
- [ ] Tested forward pass with dummy data

### Preprocessing
- [ ] Implemented Keras-based preprocessing
- [ ] Implemented TensorFlow-based preprocessing
- [ ] Tested batch preprocessing
- [ ] Verified preprocessing consistency

### Inference
- [ ] Basic inference with model.predict()
- [ ] Optimized inference with @tf.function
- [ ] Batch inference working
- [ ] Top-k predictions correct

### Performance
- [ ] Measured baseline latency
- [ ] Benchmarked @tf.function speedup
- [ ] Tested multiple batch sizes
- [ ] Identified optimal batch size

### SavedModel
- [ ] Exported model in SavedModel format
- [ ] Created model with preprocessing
- [ ] Verified saved model loads correctly
- [ ] Inspected SavedModel structure

### Production Application
- [ ] CLI application working
- [ ] Error handling implemented
- [ ] JSON output supported
- [ ] Batch processing functional
- [ ] Logging configured

### Documentation
- [ ] Code is well-commented
- [ ] Performance numbers documented
- [ ] Troubleshooting steps tested
- [ ] Best practices understood

---

## Summary

Congratulations! You have completed the TensorFlow Model Inference implementation guide. You should now be able to:

- Set up TensorFlow environments with proper device configuration
- Load and inspect pre-trained models from keras.applications
- Implement production-grade preprocessing pipelines
- Execute optimized inference using @tf.function
- Measure and optimize inference performance
- Export models in SavedModel format
- Build production-ready inference applications
- Troubleshoot common issues

### Key Takeaways

1. **@tf.function is essential** for production inference performance
2. **Batch processing** significantly improves throughput
3. **SavedModel** is the standard deployment format for TensorFlow
4. **Preprocessing** should be consistent between training and inference
5. **Error handling and logging** are critical for production systems

### Next Steps

1. Compare your TensorFlow implementation with PyTorch (Exercise 01)
2. Experiment with different keras.applications models
3. Deploy a model with TensorFlow Serving (optional)
4. Move on to the module projects to apply these concepts

---

**Guide Version**: 1.0
**Last Updated**: 2025-11-02
**TensorFlow Version**: 2.15.0
**Estimated Completion Time**: 2-3 hours
