# Exercise 01: PyTorch Model Inference - Implementation Guide

## Overview

This guide provides step-by-step instructions for completing Exercise 01: PyTorch Model Inference. You'll learn how to load pre-trained models, run inference, measure performance, and build production-ready ML inference pipelines.

**What You'll Build**:
- Complete image classification system
- Performance benchmarking tools
- Production-ready inference pipeline
- Device management (CPU/GPU) infrastructure

**ML Infrastructure Focus**:
- Model loading and management
- Inference optimization
- Performance monitoring
- Resource utilization
- Error handling and debugging

## Time Required

**Total**: 2.5-3 hours
- Phase 1: Environment Setup (20 minutes)
- Phase 2: Model Loading and Exploration (30 minutes)
- Phase 3: Image Preprocessing (25 minutes)
- Phase 4: Inference Execution (30 minutes)
- Phase 5: Performance Optimization (35 minutes)
- Phase 6: Production Application (30 minutes)

## Prerequisites

**Required Knowledge**:
- Python programming basics
- Command line operations
- Basic understanding of tensors and neural networks

**Required Software**:
- Python 3.8+ installed
- Virtual environment tool (venv)
- Internet connection for downloading models
- 4GB+ free disk space
- (Optional) NVIDIA GPU with CUDA support

**Recommended**:
- Completed Module 001 (Python Fundamentals)
- Familiarity with pip package management

---

## Phase 1: Environment Setup and Verification (20 minutes)

### Objective
Set up a clean Python environment with PyTorch and all necessary dependencies, verify GPU availability, and prepare test data.

### Step 1.1: Create and Activate Virtual Environment

**Why**: Isolated environments prevent dependency conflicts and ensure reproducibility.

```bash
# Create project directory
mkdir -p ~/ml-inference-lab
cd ~/ml-inference-lab

# Create virtual environment
python3 -m venv pytorch_env

# Activate environment
source pytorch_env/bin/activate  # Linux/Mac
# On Windows: pytorch_env\Scripts\activate

# Verify activation (prompt should show (pytorch_env))
which python
```

**Expected Output**:
```
/home/username/ml-inference-lab/pytorch_env/bin/python
```

**Troubleshooting**:
- If `python3` not found, try `python` instead
- If venv module missing, install: `sudo apt-get install python3-venv` (Ubuntu)

### Step 1.2: Install PyTorch and Dependencies

**Why**: We install specific versions for reproducibility and stability.

```bash
# Install PyTorch (CPU version - universal compatibility)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies
pip install pillow requests numpy

# Verify installations
pip list | grep -E "torch|Pillow|numpy"
```

**Expected Output**:
```
numpy         1.24.3
Pillow        10.0.0
torch         2.1.0+cpu
torchvision   0.16.0+cpu
```

**For GPU Support** (Optional - if you have NVIDIA GPU):
```bash
# First, verify CUDA is available
nvidia-smi

# Install GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 1.3: Verify PyTorch Installation

Create a verification script to test PyTorch:

```bash
cat > verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Verify PyTorch installation and check device availability.
"""
import torch
import torchvision
import sys

def verify_pytorch():
    """Verify PyTorch installation and configuration."""
    print("=" * 60)
    print("PyTorch Installation Verification")
    print("=" * 60)

    # Version information
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("Running on CPU (GPU not available)")

    # Test tensor operations
    print("\n" + "-" * 60)
    print("Testing tensor operations...")

    # CPU tensor
    cpu_tensor = torch.rand(3, 3)
    print(f"CPU tensor created: {cpu_tensor.shape}")
    print(f"CPU tensor device: {cpu_tensor.device}")

    # GPU tensor (if available)
    if cuda_available:
        gpu_tensor = torch.rand(3, 3).cuda()
        print(f"GPU tensor created: {gpu_tensor.shape}")
        print(f"GPU tensor device: {gpu_tensor.device}")

        # Test computation
        result = gpu_tensor @ gpu_tensor.T
        print(f"GPU computation successful: {result.shape}")

    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    try:
        verify_pytorch()
    except Exception as e:
        print(f"ERROR: Verification failed: {e}", file=sys.stderr)
        sys.exit(1)
EOF

chmod +x verify_installation.py
python verify_installation.py
```

**Expected Output** (CPU mode):
```
============================================================
PyTorch Installation Verification
============================================================

PyTorch version: 2.1.0+cpu
Torchvision version: 0.16.0+cpu

CUDA available: False
Running on CPU (GPU not available)

------------------------------------------------------------
Testing tensor operations...
CPU tensor created: torch.Size([3, 3])
CPU tensor device: cpu

============================================================
Verification Complete!
============================================================
```

**Checkpoint**: Your environment is ready if you see "Verification Complete!" without errors.

### Step 1.4: Download Test Images

Create a script to download sample images:

```bash
cat > download_test_images.py << 'EOF'
#!/usr/bin/env python3
"""
Download test images for inference testing.
"""
import urllib.request
import os
from pathlib import Path

def download_images():
    """Download sample images for testing."""

    # Create images directory
    images_dir = Path("test_images")
    images_dir.mkdir(exist_ok=True)

    # Image sources
    images = {
        "dog.jpg": "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "cat.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",  # Placeholder
    }

    print("Downloading test images...")
    print("-" * 60)

    for filename, url in images.items():
        filepath = images_dir / filename

        if filepath.exists():
            print(f" {filename} already exists, skipping...")
            continue

        try:
            print(f"Downloading {filename}...", end=" ")
            urllib.request.urlretrieve(url, filepath)

            # Verify file size
            size_kb = filepath.stat().st_size / 1024
            print(f" ({size_kb:.1f} KB)")

        except Exception as e:
            print(f" Failed: {e}")

    print("-" * 60)
    print(f"Images saved to: {images_dir.absolute()}")

    # List downloaded files
    print("\nDownloaded files:")
    for file in images_dir.iterdir():
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    download_images()
EOF

python download_test_images.py
```

**Checkpoint**: Verify you have test images:
```bash
ls -lh test_images/
```

---

## Phase 2: Model Loading and Exploration (30 minutes)

### Objective
Learn how to load pre-trained models from PyTorch Hub, understand model architecture, and prepare models for inference.

### Step 2.1: Load Pre-trained ResNet50 Model

Create a script to load and inspect the model:

```bash
cat > load_model.py << 'EOF'
#!/usr/bin/env python3
"""
Load and inspect a pre-trained ResNet50 model.
"""
import torch
import time

def load_resnet50():
    """Load ResNet50 model from PyTorch Hub."""

    print("=" * 60)
    print("Loading Pre-trained ResNet50")
    print("=" * 60)

    # Record start time
    start_time = time.time()

    # Load model from PyTorch Hub
    # This will download the model on first run (~100MB)
    print("\nLoading model from PyTorch Hub...")
    print("(First run will download ~100MB)")

    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'resnet50',
        pretrained=True,
        verbose=False
    )

    load_time = time.time() - start_time
    print(f" Model loaded in {load_time:.2f} seconds")

    # CRITICAL: Set model to evaluation mode
    # This disables dropout and batch normalization training behavior
    model.eval()
    print(" Model set to evaluation mode")

    # Model statistics
    print("\n" + "-" * 60)
    print("Model Statistics")
    print("-" * 60)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model name: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # 4 bytes per float32

    # Model input/output specifications
    print("\n" + "-" * 60)
    print("Model Specifications")
    print("-" * 60)
    print("Input requirements:")
    print("  - Shape: [batch_size, 3, 224, 224]")
    print("  - Type: torch.float32")
    print("  - Range: Normalized with ImageNet stats")
    print("\nOutput format:")
    print("  - Shape: [batch_size, 1000]")
    print("  - Type: torch.float32 (logits)")
    print("  - Classes: 1000 ImageNet classes")

    return model

def inspect_model_architecture(model):
    """Inspect and display model architecture details."""

    print("\n" + "=" * 60)
    print("Model Architecture Analysis")
    print("=" * 60)

    # Count layers by type
    layer_counts = {}
    for name, module in model.named_modules():
        layer_type = module.__class__.__name__
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1

    print("\nLayer type distribution:")
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1:  # Only show layers with multiple instances
            print(f"  {layer_type}: {count}")

    # Show main structure
    print("\nMain structure:")
    for name, child in model.named_children():
        print(f"  - {name}: {child.__class__.__name__}")

    # Test forward pass with dummy input
    print("\n" + "-" * 60)
    print("Testing Forward Pass")
    print("-" * 60)

    # Create dummy input
    dummy_input = torch.rand(1, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        start = time.time()
        output = model(dummy_input)
        forward_time = (time.time() - start) * 1000  # ms

    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {forward_time:.2f} ms")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    print("\n Model is ready for inference!")

if __name__ == "__main__":
    # Load model
    model = load_resnet50()

    # Inspect architecture
    inspect_model_architecture(model)

    # Save model for later use
    print("\nSaving model state...")
    torch.save(model.state_dict(), "resnet50_pretrained.pth")
    print(" Model saved to: resnet50_pretrained.pth")
EOF

python load_model.py
```

**Expected Output**:
```
============================================================
Loading Pre-trained ResNet50
============================================================

Loading model from PyTorch Hub...
(First run will download ~100MB)
 Model loaded in 15.32 seconds
 Model set to evaluation mode

------------------------------------------------------------
Model Statistics
------------------------------------------------------------
Model name: ResNet
Total parameters: 25,557,032
Trainable parameters: 25,557,032
Model size (MB): 97.49

------------------------------------------------------------
Model Specifications
------------------------------------------------------------
Input requirements:
  - Shape: [batch_size, 3, 224, 224]
  - Type: torch.float32
  - Range: Normalized with ImageNet stats

Output format:
  - Shape: [batch_size, 1000]
  - Type: torch.float32 (logits)
  - Classes: 1000 ImageNet classes

============================================================
Model Architecture Analysis
============================================================

Layer type distribution:
  BatchNorm2d: 53
  Bottleneck: 16
  Conv2d: 53
  ReLU: 49
  ...

 Model is ready for inference!
```

**Key Concepts**:
- **model.eval()**: Disables dropout and sets batch norm to inference mode
- **torch.no_grad()**: Disables gradient computation (saves memory)
- **Logits**: Raw model outputs before softmax (can be converted to probabilities)

**Checkpoint**: Verify `resnet50_pretrained.pth` file exists (should be ~98MB).

### Step 2.2: Understanding Model Inputs

Create a script to understand input requirements:

```bash
cat > understand_inputs.py << 'EOF'
#!/usr/bin/env python3
"""
Understand and visualize model input requirements.
"""
import torch
from PIL import Image

def analyze_input_requirements():
    """Analyze and explain model input requirements."""

    print("=" * 60)
    print("Understanding Model Input Requirements")
    print("=" * 60)

    # Load sample image
    image_path = "test_images/dog.jpg"
    image = Image.open(image_path)

    print(f"\nOriginal image:")
    print(f"  - Size: {image.size} (width x height)")
    print(f"  - Mode: {image.mode}")
    print(f"  - Format: {image.format}")

    # Show what model expects
    print("\nModel expects:")
    print("  - Tensor shape: [batch, channels, height, width]")
    print("  - Channels: 3 (RGB)")
    print("  - Height: 224 pixels")
    print("  - Width: 224 pixels")
    print("  - Value range: Normalized [-¼/Ã, (1-¼)/Ã]")

    # ImageNet normalization parameters
    print("\nImageNet normalization:")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    print(f"  Mean (RGB): {mean}")
    print(f"  Std (RGB):  {std}")
    print("\nThese values are computed from ImageNet training set.")

    # Show transformation pipeline
    print("\n" + "-" * 60)
    print("Required Preprocessing Steps:")
    print("-" * 60)
    print("1. Resize: Shortest side to 256 pixels (preserves aspect ratio)")
    print("2. CenterCrop: Crop center 224x224 pixels")
    print("3. ToTensor: Convert PIL Image to tensor [0, 1]")
    print("4. Normalize: Apply (x - mean) / std transformation")
    print("5. Add batch dimension: [C, H, W] ’ [1, C, H, W]")

if __name__ == "__main__":
    analyze_input_requirements()
EOF

python understand_inputs.py
```

**Checkpoint**: You understand why preprocessing is necessary and what transformations are required.

---

## Phase 3: Image Preprocessing Pipeline (25 minutes)

### Objective
Build a robust preprocessing pipeline that correctly prepares images for model input.

### Step 3.1: Create Preprocessing Pipeline

```bash
cat > preprocessing.py << 'EOF'
#!/usr/bin/env python3
"""
Image preprocessing pipeline for ResNet models.
"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class ImagePreprocessor:
    """Handle image preprocessing for ResNet models."""

    def __init__(self):
        """Initialize preprocessing pipeline."""

        # ImageNet statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Create transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),                # Resize shortest side to 256
            transforms.CenterCrop(224),            # Crop center 224x224
            transforms.ToTensor(),                 # Convert to tensor [0, 1]
            transforms.Normalize(                  # Normalize with ImageNet stats
                mean=self.mean,
                std=self.std
            ),
        ])

    def preprocess(self, image_path):
        """
        Preprocess an image for model input.

        Args:
            image_path: Path to image file

        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, 224, 224]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        tensor = self.transform(image)

        # Add batch dimension
        batch = tensor.unsqueeze(0)

        return batch

    def preprocess_batch(self, image_paths):
        """
        Preprocess multiple images into a batch.

        Args:
            image_paths: List of image paths

        Returns:
            torch.Tensor: Batch of preprocessed images [N, 3, 224, 224]
        """
        tensors = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            tensor = self.transform(image)
            tensors.append(tensor)

        # Stack into batch
        batch = torch.stack(tensors)
        return batch

    def visualize_preprocessing(self, image_path):
        """Show preprocessing steps and statistics."""

        print("=" * 60)
        print(f"Preprocessing: {image_path}")
        print("=" * 60)

        # Original image
        image = Image.open(image_path).convert('RGB')
        print(f"\n1. Original image:")
        print(f"   - Size: {image.size}")
        print(f"   - Mode: {image.mode}")

        # After resize
        resize = transforms.Resize(256)
        resized = resize(image)
        print(f"\n2. After resize (shortest side to 256):")
        print(f"   - Size: {resized.size}")

        # After center crop
        crop = transforms.CenterCrop(224)
        cropped = crop(resized)
        print(f"\n3. After center crop (224x224):")
        print(f"   - Size: {cropped.size}")

        # After to tensor
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(cropped)
        print(f"\n4. After ToTensor:")
        print(f"   - Shape: {tensor.shape}")
        print(f"   - Type: {tensor.dtype}")
        print(f"   - Range: [{tensor.min():.3f}, {tensor.max():.3f}]")

        # After normalization
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        normalized = normalize(tensor)
        print(f"\n5. After normalization:")
        print(f"   - Shape: {normalized.shape}")
        print(f"   - Type: {normalized.dtype}")
        print(f"   - Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"   - Mean per channel: {normalized.mean(dim=[1, 2]).tolist()}")
        print(f"   - Std per channel: {normalized.std(dim=[1, 2]).tolist()}")

        # After adding batch dimension
        batch = normalized.unsqueeze(0)
        print(f"\n6. After adding batch dimension:")
        print(f"   - Shape: {batch.shape}")
        print(f"   - Ready for model input: ")

        return batch

def test_preprocessor():
    """Test the preprocessing pipeline."""

    preprocessor = ImagePreprocessor()

    # Test single image
    print("\n" + "=" * 60)
    print("Test 1: Single Image Preprocessing")
    print("=" * 60)

    batch = preprocessor.visualize_preprocessing("test_images/dog.jpg")
    print(f"\nFinal output shape: {batch.shape}")
    assert batch.shape == (1, 3, 224, 224), "Incorrect output shape!"
    print(" Test passed!")

    # Test batch preprocessing
    print("\n" + "=" * 60)
    print("Test 2: Batch Preprocessing")
    print("=" * 60)

    import glob
    image_paths = glob.glob("test_images/*.jpg")
    if len(image_paths) > 1:
        batch = preprocessor.preprocess_batch(image_paths[:2])
        print(f"Batch shape: {batch.shape}")
        print(f"Number of images: {batch.shape[0]}")
        print(" Batch preprocessing successful!")

if __name__ == "__main__":
    test_preprocessor()
EOF

python preprocessing.py
```

**Expected Output**:
```
============================================================
Preprocessing: test_images/dog.jpg
============================================================

1. Original image:
   - Size: (576, 768)
   - Mode: RGB

2. After resize (shortest side to 256):
   - Size: (384, 256)

3. After center crop (224x224):
   - Size: (224, 224)

4. After ToTensor:
   - Shape: torch.Size([3, 224, 224])
   - Type: torch.float32
   - Range: [0.000, 1.000]

5. After normalization:
   - Shape: torch.Size([3, 224, 224])
   - Type: torch.float32
   - Range: [-2.118, 2.640]
   - Mean per channel: [-0.001, 0.002, -0.003]
   - Std per channel: [0.998, 1.001, 0.997]

6. After adding batch dimension:
   - Shape: torch.Size([1, 3, 224, 224])
   - Ready for model input: 

Final output shape: torch.Size([1, 3, 224, 224])
 Test passed!
```

**Key Concepts**:
- **Resize preserves aspect ratio**: Prevents image distortion
- **Normalization range can be negative**: This is expected after (x - mean) / std
- **Batch dimension is critical**: Models expect [N, C, H, W] format

**Checkpoint**: All preprocessing tests pass.

---

## Phase 4: Inference Execution (30 minutes)

### Objective
Run inference on images, interpret results, and build a classification pipeline.

### Step 4.1: Download ImageNet Labels

```bash
cat > download_labels.py << 'EOF'
#!/usr/bin/env python3
"""
Download ImageNet class labels.
"""
import urllib.request

def download_imagenet_labels():
    """Download ImageNet class labels."""

    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_file = "imagenet_classes.txt"

    print("Downloading ImageNet labels...")
    try:
        urllib.request.urlretrieve(labels_url, labels_file)
        print(f" Labels saved to: {labels_file}")

        # Load and show sample
        with open(labels_file) as f:
            labels = [line.strip() for line in f.readlines()]

        print(f"\nTotal classes: {len(labels)}")
        print("\nSample labels:")
        for i in [0, 100, 200, 500, 999]:
            print(f"  {i:3d}: {labels[i]}")

        return labels

    except Exception as e:
        print(f" Error downloading labels: {e}")
        return None

if __name__ == "__main__":
    download_imagenet_labels()
EOF

python download_labels.py
```

### Step 4.2: Build Inference Engine

```bash
cat > inference_engine.py << 'EOF'
#!/usr/bin/env python3
"""
Complete inference engine for image classification.
"""
import torch
import torch.nn.functional as F
from preprocessing import ImagePreprocessor
import time
from typing import List, Dict, Tuple

class InferenceEngine:
    """Handle model inference for image classification."""

    def __init__(self, model_name='resnet50', device='auto'):
        """
        Initialize inference engine.

        Args:
            model_name: Name of model to load
            device: 'cpu', 'cuda', or 'auto'
        """
        print(f"Initializing InferenceEngine with {model_name}...")

        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model
        print(f"Loading {model_name}...")
        self.model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            model_name,
            pretrained=True,
            verbose=False
        )

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        print(" Model loaded and ready")

        # Load labels
        with open('imagenet_classes.txt') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Create preprocessor
        self.preprocessor = ImagePreprocessor()

        print(" Inference engine ready!")

    def predict(self, image_path: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions list, inference_time_ms)
        """
        # Preprocess image
        input_tensor = self.preprocessor.preprocess(image_path)
        input_tensor = input_tensor.to(self.device)

        # Run inference with timing
        start_time = time.time()

        with torch.no_grad():
            logits = self.model(input_tensor)

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Convert logits to probabilities
        probabilities = F.softmax(logits[0], dim=0)

        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Format results
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class_id': idx.item(),
                'class_name': self.labels[idx.item()],
                'probability': prob.item(),
                'confidence': f"{prob.item() * 100:.2f}%"
            })

        return predictions, inference_time

    def predict_batch(self, image_paths: List[str], top_k: int = 5) -> List[Tuple[List[Dict], float]]:
        """
        Run inference on multiple images as a batch.

        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image

        Returns:
            List of (predictions, inference_time) tuples
        """
        # Preprocess all images into a batch
        batch_tensor = self.preprocessor.preprocess_batch(image_paths)
        batch_tensor = batch_tensor.to(self.device)

        # Run inference
        start_time = time.time()

        with torch.no_grad():
            logits = self.model(batch_tensor)

        total_inference_time = (time.time() - start_time) * 1000

        # Process results for each image
        results = []
        for i in range(len(image_paths)):
            # Convert logits to probabilities for this image
            probabilities = F.softmax(logits[i], dim=0)

            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)

            # Format results
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'class_id': idx.item(),
                    'class_name': self.labels[idx.item()],
                    'probability': prob.item(),
                    'confidence': f"{prob.item() * 100:.2f}%"
                })

            results.append((predictions, total_inference_time / len(image_paths)))

        return results

def test_inference():
    """Test the inference engine."""

    print("=" * 60)
    print("Testing Inference Engine")
    print("=" * 60)

    # Initialize engine
    engine = InferenceEngine(model_name='resnet50', device='auto')

    # Test single image
    print("\n" + "-" * 60)
    print("Test 1: Single Image Inference")
    print("-" * 60)

    image_path = "test_images/dog.jpg"
    predictions, inference_time = engine.predict(image_path, top_k=5)

    print(f"\nImage: {image_path}")
    print(f"Inference time: {inference_time:.2f} ms")
    print("\nTop 5 Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred['class_name']:<30} {pred['confidence']:>8}")

    # Test batch inference
    print("\n" + "-" * 60)
    print("Test 2: Batch Inference")
    print("-" * 60)

    import glob
    image_paths = glob.glob("test_images/*.jpg")[:2]

    if len(image_paths) > 0:
        results = engine.predict_batch(image_paths, top_k=3)

        for image_path, (predictions, inference_time) in zip(image_paths, results):
            print(f"\nImage: {image_path}")
            print(f"Inference time: {inference_time:.2f} ms")
            print("Top 3 predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {pred['class_name']:<25} {pred['confidence']:>8}")

    print("\n" + "=" * 60)
    print(" All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_inference()
EOF

python inference_engine.py
```

**Expected Output**:
```
============================================================
Testing Inference Engine
============================================================
Initializing InferenceEngine with resnet50...
Using device: cpu
Loading resnet50...
 Model loaded and ready
 Inference engine ready!

------------------------------------------------------------
Test 1: Single Image Inference
------------------------------------------------------------

Image: test_images/dog.jpg
Inference time: 42.35 ms

Top 5 Predictions:
  1. Samoyed                       89.32%
  2. Pomeranian                     4.02%
  3. white wolf                     3.11%
  4. keeshond                       1.89%
  5. Great Pyrenees                 1.42%

============================================================
 All tests passed!
============================================================
```

**Checkpoint**: Inference engine successfully classifies images with reasonable predictions.

---

## Phase 5: Performance Optimization and Benchmarking (35 minutes)

### Objective
Measure and optimize inference performance, understand throughput vs latency tradeoffs.

### Step 5.1: Create Performance Benchmarking Tool

```bash
cat > benchmark.py << 'EOF'
#!/usr/bin/env python3
"""
Performance benchmarking for inference.
"""
import torch
import time
import statistics
from inference_engine import InferenceEngine
from typing import Dict, List

class PerformanceBenchmark:
    """Benchmark inference performance."""

    def __init__(self, engine: InferenceEngine):
        """Initialize benchmark with an inference engine."""
        self.engine = engine
        self.device = engine.device

    def benchmark_latency(self, image_path: str, iterations: int = 100,
                         warmup: int = 10) -> Dict:
        """
        Benchmark inference latency.

        Args:
            image_path: Path to test image
            iterations: Number of iterations to run
            warmup: Number of warmup iterations

        Returns:
            Dictionary with latency statistics
        """
        print(f"Benchmarking latency ({iterations} iterations)...")

        # Prepare input
        input_tensor = self.engine.preprocessor.preprocess(image_path)
        input_tensor = input_tensor.to(self.device)

        # Warmup
        print(f"Warming up ({warmup} iterations)...", end=" ")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.engine.model(input_tensor)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print("")

        # Benchmark
        print("Running benchmark...", end=" ")
        latencies = []

        for _ in range(iterations):
            start = time.time()

            with torch.no_grad():
                _ = self.engine.model(input_tensor)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        print("")

        # Calculate statistics
        stats = {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'std_ms': statistics.stdev(latencies),
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'p50_ms': statistics.median(latencies),
            'p95_ms': sorted(latencies)[int(len(latencies) * 0.95)],
            'p99_ms': sorted(latencies)[int(len(latencies) * 0.99)],
            'throughput_ips': 1000 / statistics.mean(latencies),
        }

        return stats

    def benchmark_batch_sizes(self, batch_sizes: List[int], iterations: int = 50) -> Dict:
        """
        Benchmark different batch sizes.

        Args:
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per batch size

        Returns:
            Dictionary mapping batch size to performance metrics
        """
        print("\n" + "=" * 60)
        print("Batch Size Performance Analysis")
        print("=" * 60)

        results = {}

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")

            # Create dummy batch
            dummy_batch = torch.rand(batch_size, 3, 224, 224).to(self.device)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.engine.model(dummy_batch)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            # Benchmark
            latencies = []
            for _ in range(iterations):
                start = time.time()

                with torch.no_grad():
                    _ = self.engine.model(dummy_batch)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)

            # Calculate metrics
            avg_batch_latency = statistics.mean(latencies)
            avg_per_image_latency = avg_batch_latency / batch_size
            throughput = (batch_size * 1000) / avg_batch_latency

            results[batch_size] = {
                'batch_latency_ms': avg_batch_latency,
                'per_image_latency_ms': avg_per_image_latency,
                'throughput_ips': throughput,
            }

            print(f"  Batch latency: {avg_batch_latency:.2f} ms")
            print(f"  Per-image latency: {avg_per_image_latency:.2f} ms")
            print(f"  Throughput: {throughput:.2f} images/sec")

        return results

    def print_latency_report(self, stats: Dict):
        """Print formatted latency report."""

        print("\n" + "=" * 60)
        print("Latency Performance Report")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"\nLatency Statistics:")
        print(f"  Mean:       {stats['mean_ms']:>8.2f} ms")
        print(f"  Median:     {stats['median_ms']:>8.2f} ms")
        print(f"  Std Dev:    {stats['std_ms']:>8.2f} ms")
        print(f"  Min:        {stats['min_ms']:>8.2f} ms")
        print(f"  Max:        {stats['max_ms']:>8.2f} ms")
        print(f"\nPercentiles:")
        print(f"  P50:        {stats['p50_ms']:>8.2f} ms")
        print(f"  P95:        {stats['p95_ms']:>8.2f} ms")
        print(f"  P99:        {stats['p99_ms']:>8.2f} ms")
        print(f"\nThroughput: {stats['throughput_ips']:>8.2f} images/second")

    def print_batch_report(self, results: Dict):
        """Print formatted batch size comparison."""

        print("\n" + "=" * 60)
        print("Batch Size Performance Comparison")
        print("=" * 60)
        print(f"{'Batch Size':<12} {'Batch Latency':<16} {'Per-Image':<16} {'Throughput':<16}")
        print(f"{'':12} {'(ms)':<16} {'(ms)':<16} {'(img/s)':<16}")
        print("-" * 60)

        for batch_size, metrics in sorted(results.items()):
            print(f"{batch_size:<12} "
                  f"{metrics['batch_latency_ms']:<16.2f} "
                  f"{metrics['per_image_latency_ms']:<16.2f} "
                  f"{metrics['throughput_ips']:<16.2f}")

def run_comprehensive_benchmark():
    """Run complete performance benchmark."""

    print("=" * 60)
    print("Comprehensive Performance Benchmark")
    print("=" * 60)

    # Initialize engine
    print("\nInitializing inference engine...")
    engine = InferenceEngine(model_name='resnet50', device='auto')

    # Create benchmark
    benchmark = PerformanceBenchmark(engine)

    # Benchmark latency
    print("\n" + "-" * 60)
    print("1. Single Image Latency")
    print("-" * 60)

    stats = benchmark.benchmark_latency("test_images/dog.jpg", iterations=100)
    benchmark.print_latency_report(stats)

    # Benchmark batch sizes
    print("\n" + "-" * 60)
    print("2. Batch Size Impact")
    print("-" * 60)

    batch_sizes = [1, 2, 4, 8, 16]
    results = benchmark.benchmark_batch_sizes(batch_sizes, iterations=50)
    benchmark.print_batch_report(results)

    # Analysis
    print("\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)

    print("\nKey Insights:")
    print(f"1. Optimal batch size for throughput: {max(results.keys(), key=lambda k: results[k]['throughput_ips'])}")
    print(f"2. Best single-image latency: {stats['mean_ms']:.2f} ms")
    print(f"3. Maximum throughput: {max(r['throughput_ips'] for r in results.values()):.2f} img/s")

    # Device-specific recommendations
    if engine.device.type == 'cpu':
        print("\n  Running on CPU - consider GPU for better performance")
        print("  - Expected GPU speedup: 5-15x")
    else:
        print("\n Running on GPU - optimal performance")

    print("\n Benchmark complete!")

if __name__ == "__main__":
    run_comprehensive_benchmark()
EOF

python benchmark.py
```

**Expected Output** (CPU):
```
============================================================
Comprehensive Performance Benchmark
============================================================

Initializing inference engine...
Using device: cpu
 Inference engine ready!

------------------------------------------------------------
1. Single Image Latency
------------------------------------------------------------
Benchmarking latency (100 iterations)...
Warming up (10 iterations)... 
Running benchmark... 

============================================================
Latency Performance Report
============================================================
Device: cpu

Latency Statistics:
  Mean:          45.23 ms
  Median:        44.89 ms
  Std Dev:        2.15 ms
  Min:           42.10 ms
  Max:           52.34 ms

Percentiles:
  P50:           44.89 ms
  P95:           48.92 ms
  P99:           50.12 ms

Throughput:    22.11 images/second

------------------------------------------------------------
2. Batch Size Impact
------------------------------------------------------------

Testing batch size: 1
  Batch latency: 45.23 ms
  Per-image latency: 45.23 ms
  Throughput: 22.11 images/sec

Testing batch size: 2
  Batch latency: 76.89 ms
  Per-image latency: 38.45 ms
  Throughput: 26.01 images/sec

...

============================================================
Performance Analysis
============================================================

Key Insights:
1. Optimal batch size for throughput: 16
2. Best single-image latency: 45.23 ms
3. Maximum throughput: 27.67 img/s

  Running on CPU - consider GPU for better performance
  - Expected GPU speedup: 5-15x

 Benchmark complete!
```

**Key Insights**:
- Larger batches improve throughput but increase latency
- GPU provides 5-15x speedup for deep learning inference
- P99 latency is important for production SLAs

**Checkpoint**: You understand performance characteristics and bottlenecks.

---

## Phase 6: Production-Ready Application (30 minutes)

### Objective
Build a complete, production-ready inference application with error handling, logging, and CLI interface.

### Step 6.1: Create Production Inference Application

```bash
cat > classifier_app.py << 'EOF'
#!/usr/bin/env python3
"""
Production-ready image classification application.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import torch
from inference_engine import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageClassifierApp:
    """Production image classifier application."""

    def __init__(self, model_name: str = 'resnet50', device: str = 'auto'):
        """
        Initialize classifier application.

        Args:
            model_name: Model architecture to use
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        try:
            logger.info(f"Initializing classifier with {model_name}")
            self.engine = InferenceEngine(model_name=model_name, device=device)
            logger.info("Classifier ready")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            raise

    def classify_image(self, image_path: str, top_k: int = 5) -> None:
        """
        Classify a single image and display results.

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to show
        """
        # Validate image path
        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            print(f"ERROR: Image not found: {image_path}")
            return

        try:
            logger.info(f"Classifying: {image_path}")

            # Run inference
            predictions, inference_time = self.engine.predict(image_path, top_k=top_k)

            # Display results
            print("\n" + "=" * 60)
            print(f"Classification Results: {Path(image_path).name}")
            print("=" * 60)
            print(f"Inference time: {inference_time:.2f} ms")
            print(f"\nTop {top_k} Predictions:")
            print("-" * 60)

            for i, pred in enumerate(predictions, 1):
                confidence = pred['probability'] * 100
                bar_length = int(confidence / 2)  # Scale to 50 chars max
                bar = 'ˆ' * bar_length

                print(f"{i:2d}. {pred['class_name']:<30} {confidence:>6.2f}% {bar}")

            print("=" * 60)

            logger.info(f"Classification complete: {predictions[0]['class_name']} ({predictions[0]['confidence']})")

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            print(f"ERROR: Classification failed: {e}")

    def classify_batch(self, image_dir: str, top_k: int = 3) -> None:
        """
        Classify all images in a directory.

        Args:
            image_dir: Directory containing images
            top_k: Number of top predictions per image
        """
        # Find all images
        image_dir_path = Path(image_dir)
        if not image_dir_path.exists():
            logger.error(f"Directory not found: {image_dir}")
            print(f"ERROR: Directory not found: {image_dir}")
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = [
            str(p) for p in image_dir_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            print(f"No images found in: {image_dir}")
            return

        logger.info(f"Found {len(image_paths)} images in {image_dir}")

        try:
            # Batch inference
            results = self.engine.predict_batch(image_paths, top_k=top_k)

            # Display results
            print("\n" + "=" * 60)
            print(f"Batch Classification Results ({len(image_paths)} images)")
            print("=" * 60)

            for image_path, (predictions, inference_time) in zip(image_paths, results):
                print(f"\n{Path(image_path).name}:")
                print(f"  Inference: {inference_time:.2f} ms")
                print(f"  Top prediction: {predictions[0]['class_name']} ({predictions[0]['confidence']})")

                if top_k > 1:
                    print(f"  Other predictions:")
                    for pred in predictions[1:top_k]:
                        print(f"    - {pred['class_name']} ({pred['confidence']})")

            print("\n" + "=" * 60)
            logger.info(f"Batch classification complete")

        except Exception as e:
            logger.error(f"Batch classification failed: {e}", exc_info=True)
            print(f"ERROR: Batch classification failed: {e}")

def main():
    """Main application entry point."""

    parser = argparse.ArgumentParser(
        description='PyTorch Image Classifier - Production Inference Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Classify single image
  python classifier_app.py dog.jpg

  # Use different model
  python classifier_app.py dog.jpg --model resnet18

  # Show top 10 predictions
  python classifier_app.py dog.jpg --top-k 10

  # Force CPU usage
  python classifier_app.py dog.jpg --device cpu

  # Classify all images in directory
  python classifier_app.py --batch test_images/
        '''
    )

    parser.add_argument(
        'image',
        nargs='?',
        help='Path to image file (or directory with --batch)'
    )
    parser.add_argument(
        '--model',
        default='resnet50',
        choices=['resnet18', 'resnet50', 'resnet101', 'mobilenet_v2'],
        help='Model architecture (default: resnet50)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run on (default: auto)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all images in directory'
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

    # Validate arguments
    if not args.image:
        parser.print_help()
        sys.exit(1)

    try:
        # Initialize application
        app = ImageClassifierApp(model_name=args.model, device=args.device)

        # Run classification
        if args.batch:
            app.classify_batch(args.image, top_k=args.top_k)
        else:
            app.classify_image(args.image, top_k=args.top_k)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nApplication interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x classifier_app.py
```

### Step 6.2: Test Production Application

```bash
# Test basic classification
python classifier_app.py test_images/dog.jpg

# Test with different top-k
python classifier_app.py test_images/dog.jpg --top-k 10

# Test batch processing
python classifier_app.py --batch test_images/

# Test with verbose logging
python classifier_app.py test_images/dog.jpg --verbose
```

**Checkpoint**: Production application runs successfully with proper error handling.

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Out of Memory Error

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```python
# Reduce batch size
batch_size = 1  # Start small

# Use CPU instead
engine = InferenceEngine(device='cpu')

# Clear CUDA cache
torch.cuda.empty_cache()
```

#### Issue 2: Model Download Fails

**Symptoms**:
```
urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
```

**Solutions**:
```bash
# Check internet connection
ping google.com

# Try manual download
wget https://download.pytorch.org/models/resnet50-0676ba61.pth

# Use offline model loading
model = torch.load('resnet50-0676ba61.pth')
```

#### Issue 3: Incorrect Predictions

**Symptoms**:
- All predictions are random or nonsensical

**Diagnostic**:
```python
# Check if model is in eval mode
print(model.training)  # Should be False

# Verify preprocessing
tensor = preprocess(image)
print(f"Mean: {tensor.mean()}")  # Should be ~0
print(f"Std: {tensor.std()}")    # Should be ~1
```

**Solutions**:
```python
# Ensure eval mode
model.eval()

# Verify normalization stats
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

#### Issue 4: Slow Performance

**Symptoms**:
- Inference takes >1 second per image

**Solutions**:
```python
# Enable GPU if available
engine = InferenceEngine(device='cuda')

# Use batch inference
results = engine.predict_batch(image_paths)

# Use smaller model
engine = InferenceEngine(model_name='resnet18')

# Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

---

## Best Practices

### 1. Always Use eval() Mode

```python
# CORRECT
model.eval()
with torch.no_grad():
    output = model(input)

# INCORRECT - may give wrong results
output = model(input)
```

### 2. Proper Device Management

```python
# CORRECT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)

# INCORRECT - will fail on GPU-only code
model.cuda()  # Crashes if no GPU
```

### 3. Batch Processing for Throughput

```python
# CORRECT - efficient batch processing
batch = torch.stack([preprocess(img) for img in images])
predictions = model(batch)

# INEFFICIENT - one at a time
for img in images:
    prediction = model(preprocess(img).unsqueeze(0))
```

### 4. Error Handling

```python
# CORRECT
try:
    predictions, time = engine.predict(image_path)
except FileNotFoundError:
    logger.error(f"Image not found: {image_path}")
except torch.cuda.OutOfMemoryError:
    logger.error("GPU out of memory, retrying on CPU")
    engine = InferenceEngine(device='cpu')
```

---

## Validation Checklist

Use this checklist to verify your implementation:

- [ ] Virtual environment created and activated
- [ ] PyTorch and dependencies installed correctly
- [ ] GPU detection working (if GPU available)
- [ ] Test images downloaded
- [ ] Model loads successfully
- [ ] Model in eval mode
- [ ] Preprocessing pipeline correct
- [ ] Single image inference works
- [ ] Batch inference works
- [ ] Performance benchmark completes
- [ ] Production app runs without errors
- [ ] Error handling tested
- [ ] Logging configured properly

---

## Performance Targets

Your implementation should meet these targets:

**CPU (Reference: Intel i7)**:
- Single image latency: < 100ms
- Throughput: > 15 images/second

**GPU (Reference: NVIDIA T4)**:
- Single image latency: < 10ms
- Throughput: > 100 images/second

**Accuracy**:
- Top-1 accuracy on ImageNet validation: ~76% (ResNet50)
- Top-5 accuracy: ~93%

---

## Next Steps

After completing this exercise:

1. **Experiment with Different Models**
   ```bash
   python classifier_app.py test_images/dog.jpg --model resnet18
   python classifier_app.py test_images/dog.jpg --model mobilenet_v2
   ```

2. **Try Custom Images**
   - Download your own images
   - Test model generalization

3. **Optimize Further**
   - Explore model quantization
   - Try TorchScript compilation
   - Implement model caching

4. **Move to Exercise 02**
   - Learn TensorFlow inference
   - Compare PyTorch vs TensorFlow

---

## Summary

**What You Accomplished**:
- Built complete PyTorch inference pipeline
- Implemented production-ready classifier
- Measured and optimized performance
- Learned device management (CPU/GPU)
- Applied ML infrastructure best practices

**Key Skills Gained**:
- Model loading and management
- Image preprocessing
- Inference optimization
- Performance benchmarking
- Error handling in ML systems
- Production application development

**Time Investment**: 2.5-3 hours
**Skills Acquired**: PyTorch inference, ML infrastructure, performance optimization
**Ready For**: Exercise 02 (TensorFlow), production ML deployments

---

**Congratulations on completing Exercise 01: PyTorch Model Inference!**

You now have the skills to deploy PyTorch models in production environments with proper performance monitoring and error handling.

---

**Exercise Version**: 1.0
**Last Updated**: November 2025
**Estimated Time**: 2.5-3 hours
**Difficulty**: Beginner
