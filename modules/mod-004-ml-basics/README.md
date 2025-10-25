# Module 004: ML Basics for Infrastructure Engineers

## Overview

Understand machine learning fundamentals from an infrastructure perspective. Learn ML workflows, frameworks, model training, deployment patterns, and the infrastructure requirements for supporting ML workloads at scale.

## Learning Objectives

- ✅ Understand ML terminology and concepts relevant to infrastructure
- ✅ Work with popular ML frameworks (PyTorch, TensorFlow, scikit-learn)
- ✅ Set up ML training environments
- ✅ Understand GPU utilization for ML workloads
- ✅ Manage ML data pipelines and storage
- ✅ Deploy and serve ML models
- ✅ Monitor ML infrastructure performance
- ✅ Troubleshoot common ML infrastructure issues

## Why ML Basics for Infrastructure Engineers?

As an AI Infrastructure Engineer, you need to:
- **Understand the workload**: Know what ML training and inference require from infrastructure
- **Optimize resources**: Allocate GPU, CPU, and memory appropriately
- **Troubleshoot effectively**: Diagnose performance issues in ML pipelines
- **Plan capacity**: Estimate infrastructure needs for ML projects
- **Communicate with ML teams**: Speak the same language as data scientists

You don't need to build models, but you need to understand:
- How models are trained and deployed
- What resources they consume
- What infrastructure bottlenecks exist
- How to support ML workflows efficiently

## Module Structure

### Exercise 01: ML Fundamentals (6-8 hours)
- ML terminology and concepts
- Understanding ML workflows
- Common ML frameworks overview
- Setting up ML environments
- GPU basics for ML

### Exercise 02: Model Training Pipeline (8-10 hours)
- Training data management
- Training infrastructure setup
- Distributed training basics
- Monitoring training jobs
- Checkpointing and model storage

### Exercise 03: Model Deployment (8-10 hours)
- Model formats and conversion
- Model serving architectures
- API development for ML models
- Performance optimization
- Production deployment patterns

**Total Time**: 22-28 hours

---

## Key ML Concepts for Infrastructure

### 1. Machine Learning Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Data      │────▶│   Training   │────▶│   Model     │────▶│  Deployment  │
│ Collection  │     │  & Tuning    │     │  Evaluation │     │  & Serving   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
       │                    │                     │                    │
       ▼                    ▼                     ▼                    ▼
   Data Lake           GPU Cluster          Validation Set       Production API
  Object Storage      Distributed           Metrics Store         Load Balancer
```

### 2. ML Infrastructure Stack

```
┌────────────────────────────────────────────────────────────┐
│                   ML Applications/Models                    │
├────────────────────────────────────────────────────────────┤
│  PyTorch │ TensorFlow │ scikit-learn │ XGBoost │ Others   │ ← ML Frameworks
├────────────────────────────────────────────────────────────┤
│  MLflow │ Weights & Biases │ TensorBoard │ Model Registry │ ← ML Tools
├────────────────────────────────────────────────────────────┤
│  Python │ Jupyter │ NumPy/Pandas │ CUDA │ cuDNN          │ ← Runtime
├────────────────────────────────────────────────────────────┤
│  Docker │ Kubernetes │ Ray │ Horovod │ DeepSpeed         │ ← Orchestration
├────────────────────────────────────────────────────────────┤
│  CPU │ GPU (NVIDIA/AMD) │ Memory │ Storage │ Network      │ ← Hardware
└────────────────────────────────────────────────────────────┘
```

### 3. Key ML Terminology

**Model Training:**
- **Epoch**: One complete pass through the training dataset
- **Batch Size**: Number of samples processed before model update
- **Learning Rate**: Step size for model parameter updates
- **Loss Function**: Measure of prediction error
- **Gradient**: Direction of parameter updates
- **Backpropagation**: Algorithm for computing gradients

**Model Types:**
- **Supervised Learning**: Learning from labeled data (classification, regression)
- **Unsupervised Learning**: Learning patterns from unlabeled data (clustering)
- **Reinforcement Learning**: Learning through rewards and penalties
- **Transfer Learning**: Using pre-trained models as starting points

**Infrastructure Terms:**
- **Training Job**: Process of training a model
- **Inference**: Making predictions with a trained model
- **Checkpoint**: Saved model state during training
- **Model Artifact**: Serialized trained model
- **Serving**: Deploying model for production use

---

## Common ML Frameworks

### PyTorch

**Overview**: Popular deep learning framework developed by Meta (Facebook)

**Infrastructure Characteristics:**
- Dynamic computation graphs (flexible, easier debugging)
- Excellent GPU utilization
- Native Python integration
- Popular for research and production

**Resource Requirements:**
- GPU: Highly recommended for deep learning
- Memory: High (models + activations + gradients)
- Storage: Moderate (model checkpoints)

**Common Use Cases:**
- Computer vision (image classification, object detection)
- Natural language processing (transformers, LLMs)
- Reinforcement learning
- Research experiments

**Installation:**
```bash
# CPU only
pip install torch torchvision torchaudio

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Typical File Sizes:**
- Small models: 10-100 MB
- Medium models: 100 MB - 1 GB
- Large models: 1-10 GB
- LLMs: 10+ GB (up to 100s of GB)

### TensorFlow

**Overview**: Comprehensive ML framework by Google

**Infrastructure Characteristics:**
- Static computation graphs (optimized execution)
- Excellent for production deployment
- TensorFlow Serving for model serving
- Strong ecosystem (TensorBoard, TFX)

**Resource Requirements:**
- GPU: Supported, but also optimized for CPU/TPU
- Memory: High for training, moderate for inference
- Storage: Model checkpoints can be large

**Common Use Cases:**
- Production ML systems
- Mobile deployment (TensorFlow Lite)
- Large-scale training (distributed)
- Edge devices

**Installation:**
```bash
# CPU only
pip install tensorflow

# GPU support
pip install tensorflow[and-cuda]

# TensorFlow Extended (TFX)
pip install tfx
```

### scikit-learn

**Overview**: Classical machine learning library

**Infrastructure Characteristics:**
- CPU-based (no GPU support in core library)
- Low memory footprint
- Fast training for small/medium datasets
- Easy to deploy

**Resource Requirements:**
- CPU: Primary compute resource
- Memory: Moderate (depends on dataset size)
- Storage: Small model files (KB to MB)

**Common Use Cases:**
- Tabular data analysis
- Classical ML algorithms (random forests, SVM, linear models)
- Feature engineering
- Quick prototyping

**Installation:**
```bash
pip install scikit-learn
```

### XGBoost / LightGBM

**Overview**: Gradient boosting frameworks

**Infrastructure Characteristics:**
- CPU-optimized (GPU support available)
- Memory efficient
- Fast training
- Small model sizes

**Common Use Cases:**
- Structured/tabular data
- Kaggle competitions
- Production ML for structured data
- Feature importance analysis

**Installation:**
```bash
pip install xgboost lightgbm
```

---

## ML Infrastructure Requirements

### Training Infrastructure

**Compute:**
- **GPUs**: Primary for deep learning (NVIDIA recommended)
  - Single GPU: Development, small models
  - Multi-GPU: Medium to large models
  - Multi-node: Very large models, distributed training
- **CPUs**: Classical ML, data preprocessing
- **TPUs**: Google Cloud TPUs for TensorFlow

**Memory:**
- **RAM**: 2-4x model size + dataset batch size
- **GPU Memory**: Critical bottleneck
  - 8-16 GB: Small models, development
  - 24-48 GB: Medium models, research
  - 40-80 GB: Large models, production training

**Storage:**
- **Training Data**: Fast access (SSD/NVMe preferred)
  - Small: < 100 GB
  - Medium: 100 GB - 1 TB
  - Large: 1-10 TB
  - Very Large: 10+ TB
- **Model Checkpoints**: Frequent writes
- **Logs & Metrics**: Moderate writes

**Network:**
- **Data Loading**: High bandwidth for remote storage
- **Distributed Training**: Low latency, high bandwidth between nodes
- **Monitoring**: Metrics collection and logging

### Inference Infrastructure

**Compute:**
- **GPU**: High throughput, batch inference
- **CPU**: Cost-effective for low-latency single predictions
- **Edge Devices**: Optimized models (quantization, pruning)

**Latency Requirements:**
- **Batch Inference**: Seconds to minutes acceptable
- **Real-time API**: < 100ms typical
- **Critical Systems**: < 10ms required

**Scaling:**
- **Horizontal**: Multiple replicas behind load balancer
- **Vertical**: Larger instances for single-model serving
- **Auto-scaling**: Based on request volume

---

## GPU Fundamentals for ML

### Why GPUs for ML?

**Parallelism**:
- CPUs: 8-64 cores for general purpose
- GPUs: 1000s of cores optimized for parallel operations
- ML operations (matrix multiplication) are highly parallelizable

**Performance Gains:**
- Training: 10-100x faster than CPU
- Inference: 10-50x faster than CPU
- Data preprocessing: 5-20x faster with GPU-accelerated libraries

### GPU Architecture Basics

```
GPU Memory (VRAM)
├── Model Weights
├── Activations (intermediate results)
├── Gradients (for backpropagation)
├── Optimizer State (Adam, SGD)
└── Temporary Buffers

GPU Compute
├── CUDA Cores (NVIDIA)
├── Tensor Cores (mixed precision)
├── RT Cores (ray tracing, not ML)
└── Memory Bandwidth
```

### GPU Utilization Metrics

**Key Metrics:**
- **GPU Utilization**: % of time GPU is active (target: > 90%)
- **Memory Usage**: % of VRAM used (target: 80-95%)
- **Memory Bandwidth**: Data transfer rate
- **Compute Throughput**: TFLOPS achieved
- **Temperature**: Operating temperature (safe: < 85°C)
- **Power Draw**: Watts consumed

**Monitoring:**
```bash
# Real-time monitoring
nvidia-smi -l 1

# Query specific metrics
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

# Persistent daemon
nvidia-persistenced

# Process monitoring
nvidia-smi pmon
```

### Common GPU Issues

**Out of Memory (OOM):**
- **Cause**: Model + data + activations exceed GPU memory
- **Solutions**:
  - Reduce batch size
  - Use gradient checkpointing
  - Enable mixed precision training
  - Use model parallelism

**Low GPU Utilization:**
- **Cause**: Data loading bottleneck, small batches
- **Solutions**:
  - Increase batch size
  - Use data loading workers
  - Pin memory for faster transfers
  - Prefetch data

**Multi-GPU Issues:**
- **Cause**: Poor parallelization, communication overhead
- **Solutions**:
  - Use DistributedDataParallel (PyTorch)
  - Optimize batch size per GPU
  - Use faster interconnect (NVLink)

---

## ML Data Management

### Data Types

**Training Data:**
- Images: JPEG, PNG, TIFF
- Text: Plain text, JSON, Parquet
- Tabular: CSV, Parquet, HDF5
- Audio: WAV, MP3, FLAC
- Video: MP4, AVI

**Data Formats:**
- **Raw Data**: Original format, largest size
- **Preprocessed**: Ready for training, optimized format
- **TFRecord**: TensorFlow optimized format
- **HDF5**: Hierarchical data format
- **Parquet**: Columnar storage for tabular data

### Data Storage Patterns

**Development:**
```
/data/
├── raw/                 # Original data
├── processed/           # Cleaned/preprocessed
├── train/              # Training split
├── val/                # Validation split
└── test/               # Test split
```

**Production:**
```
Object Storage (S3, GCS, Azure Blob)
├── datasets/
│   ├── v1.0/
│   ├── v2.0/
│   └── latest/
├── models/
│   ├── checkpoints/
│   ├── final/
│   └── archived/
└── artifacts/
    ├── logs/
    ├── metrics/
    └── predictions/
```

### Data Loading Optimization

**Bottlenecks:**
1. Disk I/O: Slow storage
2. Network I/O: Remote data
3. CPU preprocessing: Complex transforms
4. Memory: Insufficient RAM

**Solutions:**
```python
# PyTorch DataLoader optimization
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,        # Parallel data loading
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2     # Prefetch batches
)

# TensorFlow Dataset optimization
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.cache()           # Cache in memory
dataset = dataset.shuffle(10000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch
```

---

## Model Deployment Patterns

### 1. REST API Serving

**Pattern**: Model behind HTTP API

**Pros:**
- Simple integration
- Language agnostic
- Standard HTTP infrastructure

**Cons:**
- Higher latency (serialization overhead)
- Network overhead
- Scaling complexity

**Tools:**
- Flask/FastAPI (Python)
- TensorFlow Serving
- TorchServe
- Seldon Core

### 2. Batch Inference

**Pattern**: Process large datasets offline

**Pros:**
- High throughput
- Resource efficient
- Simpler infrastructure

**Cons:**
- Not real-time
- Delayed results
- Scheduling complexity

**Use Cases:**
- Daily predictions
- Large-scale feature generation
- Report generation

### 3. Edge Deployment

**Pattern**: Model on device (mobile, IoT)

**Pros:**
- Low latency
- No network required
- Privacy (data stays on device)

**Cons:**
- Limited compute/memory
- Model optimization required
- Update complexity

**Tools:**
- TensorFlow Lite
- ONNX Runtime
- PyTorch Mobile
- Core ML (iOS)

### 4. Streaming Inference

**Pattern**: Real-time processing of data streams

**Pros:**
- Real-time results
- Scalable
- Event-driven

**Cons:**
- Complex architecture
- State management
- Exactly-once semantics

**Tools:**
- Apache Kafka + ML models
- AWS Kinesis
- Google Cloud Dataflow

---

## ML Infrastructure Best Practices

### 1. Environment Management

```bash
# Use virtual environments
python -m venv ml-env
source ml-env/bin/activate

# Use conda for complex dependencies
conda create -n ml-env python=3.10
conda activate ml-env

# Pin dependencies
pip freeze > requirements.txt

# Use Docker for reproducibility
docker build -t ml-training:latest .
```

### 2. Experiment Tracking

**Track:**
- Hyperparameters
- Metrics (accuracy, loss, etc.)
- Model artifacts
- Training duration
- Resource utilization

**Tools:**
- MLflow
- Weights & Biases (W&B)
- TensorBoard
- Neptune.ai

### 3. Model Versioning

```
models/
├── model-v1.0/
│   ├── model.pth
│   ├── config.yaml
│   ├── metrics.json
│   └── metadata.json
├── model-v1.1/
└── model-v2.0/
```

### 4. Resource Allocation

**Training:**
- Allocate dedicated GPU nodes
- Set memory limits
- Use job queues (SLURM, Kubernetes)
- Monitor and kill runaway jobs

**Inference:**
- Horizontal scaling with load balancing
- Auto-scaling based on load
- Use smaller instance types for CPU inference
- Cache frequently requested predictions

---

## Troubleshooting Common Issues

### Issue: Training is Slow

**Possible Causes:**
1. Data loading bottleneck
2. Small batch size
3. CPU preprocessing overhead
4. Slow storage

**Diagnosis:**
```bash
# Monitor GPU utilization
nvidia-smi dmon -s mu

# Profile data loading
# Use PyTorch profiler or TensorFlow profiler

# Check disk I/O
iostat -x 1
```

**Solutions:**
- Increase `num_workers` in DataLoader
- Increase batch size (if memory allows)
- Move data to faster storage (SSD)
- Preprocess data offline

### Issue: Out of Memory

**Diagnosis:**
```bash
# Check GPU memory
nvidia-smi

# Check RAM
free -h
```

**Solutions:**
- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision (FP16)
- Clear cache: `torch.cuda.empty_cache()`
- Use smaller model architecture

### Issue: Model Not Improving

**Infrastructure-Related Causes:**
1. Insufficient training time
2. Resource constraints slowing training
3. Data pipeline issues (corrupted data)
4. Checkpoint loading failures

**Diagnosis:**
- Check logs for errors
- Verify data loading
- Monitor loss curves
- Check learning rate

---

## Exercises

### Exercise 01: ML Fundamentals
- Set up ML development environment
- Understand ML frameworks
- Monitor GPU utilization
- Optimize data loading

### Exercise 02: Model Training Pipeline
- Build training infrastructure
- Implement checkpointing
- Track experiments
- Optimize training performance

### Exercise 03: Model Deployment
- Convert and optimize models
- Build serving API
- Implement monitoring
- Deploy to production

---

## Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)

### Courses
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [ML Engineering for Production](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)

### Books
- "Designing Machine Learning Systems" by Chip Huyen
- "Machine Learning Engineering" by Andriy Burkov
- "Building Machine Learning Powered Applications" by Emmanuel Ameisen

### Tools & Platforms
- [MLflow](https://mlflow.org/)
- [Weights & Biases](https://wandb.ai/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Kubeflow](https://www.kubeflow.org/)

---

## Next Steps

After completing this module, you'll understand:
- How ML workloads consume infrastructure resources
- What data scientists need from infrastructure
- How to optimize ML training and inference
- Best practices for ML infrastructure management

**Next Module**: Module 005: Docker & Containerization
- Containerize ML applications
- Build ML container images
- Orchestrate ML workloads with Kubernetes
- Deploy scalable ML infrastructure

---

**Master the fundamentals to support ML at scale! 🤖**
