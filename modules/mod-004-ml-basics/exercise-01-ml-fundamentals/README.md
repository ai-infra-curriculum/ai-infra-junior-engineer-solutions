# Exercise 01: ML Fundamentals

## Overview

Learn the fundamentals of machine learning from an infrastructure perspective. Set up ML environments, understand popular frameworks, monitor GPU utilization, and optimize data loading for ML workloads.

## Learning Objectives

- ✅ Set up Python environments for ML development
- ✅ Install and configure PyTorch, TensorFlow, and scikit-learn
- ✅ Understand GPU architecture and CUDA basics
- ✅ Monitor GPU utilization and performance
- ✅ Optimize data loading pipelines
- ✅ Run basic ML training workloads
- ✅ Troubleshoot common ML infrastructure issues

## Topics Covered

### 1. ML Environment Setup

#### Python Virtual Environments

```bash
# Create virtual environment
python3 -m venv ml-env

# Activate environment
source ml-env/bin/activate  # Linux/Mac
ml-env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Deactivate
deactivate
```

#### Conda Environments

```bash
# Install Miniconda/Anaconda first
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create environment with specific Python version
conda create -n ml-env python=3.10

# Activate environment
conda activate ml-env

# Install packages
conda install numpy pandas matplotlib

# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml

# List environments
conda env list

# Remove environment
conda env remove -n ml-env
```

### 2. Installing ML Frameworks

#### PyTorch Installation

```bash
# CPU-only version
pip install torch torchvision torchaudio

# CUDA 11.8 (check your CUDA version with: nvcc --version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### TensorFlow Installation

```bash
# TensorFlow 2.x (includes GPU support if CUDA is available)
pip install tensorflow

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# TensorFlow with specific CUDA
# TensorFlow automatically detects compatible CUDA versions
```

#### scikit-learn Installation

```bash
# scikit-learn (CPU only, no GPU support)
pip install scikit-learn

# With additional tools
pip install scikit-learn pandas numpy matplotlib seaborn

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

#### Complete ML Environment

```bash
# requirements.txt for ML development
cat > requirements.txt <<EOF
# Core ML frameworks
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.13.0
scikit-learn>=1.3.0

# Data manipulation
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# ML utilities
mlflow>=2.5.0
tensorboard>=2.13.0

# Jupyter
jupyter>=1.0.0
jupyterlab>=4.0.0

# Other useful libraries
tqdm>=4.65.0
pillow>=10.0.0
requests>=2.31.0
EOF

# Install all dependencies
pip install -r requirements.txt
```

### 3. GPU Setup and Verification

#### NVIDIA Driver and CUDA

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check CUDA libraries
ls /usr/local/cuda/lib64/

# Test GPU with PyTorch
python3 <<EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
```

#### GPU Memory Management

```python
import torch

# Check current GPU memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear GPU cache
torch.cuda.empty_cache()

# Set memory growth (TensorFlow)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Limit GPU memory (TensorFlow)
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
    )
```

### 4. Basic ML Training Examples

#### PyTorch Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet(input_size=10, hidden_size=50, output_size=2).to(device)

# Create dummy data
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'model.pth')
```

#### TensorFlow/Keras Example

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create simple model
model = keras.Sequential([
    keras.layers.Dense(50, activation='relu', input_shape=(10,)),
    keras.layers.Dense(2, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Create dummy data
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 2, 1000)

# Train model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# Save model
model.save('model_tf')
```

#### scikit-learn Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Create dummy data
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, 'model_sklearn.pkl')
```

### 5. GPU Monitoring and Profiling

#### Real-time GPU Monitoring

```bash
# Basic monitoring
nvidia-smi -l 1  # Update every second

# Detailed monitoring
nvidia-smi dmon -s pucvmet -c 100  # Monitor for 100 iterations

# Query specific metrics
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 1

# Process-specific monitoring
nvidia-smi pmon -c 10  # Monitor processes
```

#### Python-based GPU Monitoring

```python
import subprocess
import re

def get_gpu_info():
    """Get GPU utilization and memory usage"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
         '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
        return {
            'gpu_utilization': int(gpu_util),
            'memory_used': int(mem_used),
            'memory_total': int(mem_total),
            'memory_percent': int(mem_used) / int(mem_total) * 100
        }
    return None

# Monitor during training
info = get_gpu_info()
print(f"GPU Utilization: {info['gpu_utilization']}%")
print(f"Memory: {info['memory_used']}MB / {info['memory_total']}MB")
```

#### PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = SimpleNet(10, 50, 2).cuda()
inputs = torch.randn(32, 10).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    outputs = model(inputs)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for TensorBoard
prof.export_chrome_trace("trace.json")
```

### 6. Data Loading Optimization

#### PyTorch DataLoader Optimization

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, size=10000):
        self.size = size
        self.data = np.random.randn(size, 224, 224, 3).astype(np.float32)
        self.labels = np.random.randint(0, 10, size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]

# Optimized DataLoader
dataset = CustomDataset()
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # Use 4 worker processes
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=2        # Prefetch 2 batches per worker
)

# Test loading speed
import time
start = time.time()
for batch_idx, (data, target) in enumerate(dataloader):
    if batch_idx >= 100:  # Test 100 batches
        break
end = time.time()
print(f"Time for 100 batches: {end - start:.2f}s")
```

#### TensorFlow Data Pipeline

```python
import tensorflow as tf

def create_dataset(size=10000):
    # Create dummy data
    data = tf.random.normal((size, 224, 224, 3))
    labels = tf.random.uniform((size,), maxval=10, dtype=tf.int32)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))

    # Optimize pipeline
    dataset = dataset.cache()  # Cache dataset in memory
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch automatically

    return dataset

# Use dataset
dataset = create_dataset()
for batch_data, batch_labels in dataset.take(10):
    print(f"Batch shape: {batch_data.shape}")
```

---

## Project: ML Infrastructure Setup and Monitoring

Build a comprehensive ML infrastructure setup and monitoring system.

### Requirements

**Components to Create:**
1. Environment setup automation
2. GPU monitoring dashboard
3. Framework compatibility checker
4. Data loading benchmark tool
5. Training job monitor

**Features:**
- Automated ML environment setup
- Real-time GPU monitoring
- Performance benchmarking
- Resource utilization tracking
- Alert system for issues

### Implementation

See `solutions/` directory for complete implementations.

---

## Practice Problems

### Problem 1: Environment Setup Script

Create a script that:
- Detects system configuration (CPU/GPU)
- Installs appropriate ML frameworks
- Verifies installations
- Sets up development environment
- Generates configuration report

### Problem 2: GPU Benchmark Tool

Create a tool that:
- Benchmarks GPU performance
- Tests different batch sizes
- Measures throughput
- Generates performance report
- Recommends optimal settings

### Problem 3: Data Loading Optimizer

Create a tool that:
- Tests different DataLoader configurations
- Measures loading speed
- Identifies bottlenecks
- Recommends optimal settings
- Generates comparison charts

---

## Best Practices

### 1. Environment Management

```bash
# Always use virtual environments
python -m venv ml-env
source ml-env/bin/activate

# Pin dependencies
pip freeze > requirements.txt

# Document Python version
python --version > python_version.txt

# Use .gitignore
echo "ml-env/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
```

### 2. GPU Best Practices

```python
# Check GPU availability before use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Clear GPU cache when needed
torch.cuda.empty_cache()

# Use mixed precision for better performance
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Data Loading

```python
# Use appropriate number of workers
# Rule of thumb: num_workers = 4 * num_gpus

# Monitor data loading time
import time
start = time.time()
for data in dataloader:
    pass
print(f"Loading time: {time.time() - start:.2f}s")
```

---

## Validation

Test your setup:

```bash
# Verify Python environment
python --version
pip list

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Run benchmark
python solutions/benchmark_gpu.py

# Test data loading
python solutions/test_dataloader.py

# Monitor GPU
python solutions/monitor_gpu.py
```

---

## Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [TensorFlow Installation](https://www.tensorflow.org/install)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## Next Steps

1. **Exercise 02: Model Training Pipeline** - Build production training pipelines
2. Practice with different ML frameworks
3. Experiment with different model architectures
4. Optimize training performance
5. Learn distributed training basics

---

**Build a solid ML infrastructure foundation! 🚀**
