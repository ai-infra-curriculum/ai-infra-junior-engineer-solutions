# Implementation Guide: LLM Basics

**Exercise**: Module 004 - Exercise 04: LLM Basics
**Focus**: Running and serving Large Language Models
**Target Audience**: Junior AI Infrastructure Engineers
**Estimated Time**: 2-3 hours

## Table of Contents

1. [Overview](#overview)
2. [Architecture and Design](#architecture-and-design)
3. [Environment Setup](#environment-setup)
4. [Core Concepts](#core-concepts)
5. [Implementation Details](#implementation-details)
6. [Production Considerations](#production-considerations)
7. [Testing and Validation](#testing-and-validation)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Advanced Topics](#advanced-topics)

## Overview

### What You'll Build

This exercise implements a production-ready LLM inference service with the following components:

- **Text Generation API**: Flask-based REST API for serving LLM predictions
- **Resource Monitoring**: CPU, memory, and performance tracking
- **Model Comparison**: Benchmarking different model sizes
- **Parameter Exploration**: Understanding generation parameters
- **Production Infrastructure**: Error handling, logging, validation

### Learning Objectives

By completing this implementation, you will:

1. Understand LLM architecture and inference pipeline
2. Load and run pre-trained models using Hugging Face Transformers
3. Implement production-grade API endpoints with proper validation
4. Monitor and optimize resource usage (CPU, memory, GPU)
5. Understand generation parameters and their impact
6. Deploy ML models with proper error handling and logging
7. Benchmark and compare different model sizes
8. Implement infrastructure best practices for ML serving

### Prerequisites

- Python 3.8+
- 2GB+ available RAM
- Basic understanding of HTTP APIs
- Familiarity with Flask (helpful but not required)
- Understanding of ML concepts (from previous modules)

## Architecture and Design

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LLM Inference Service                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Flask API  │─────▶│   Generator  │─────▶│ LLM Model │ │
│  │   Endpoints  │      │   Pipeline   │      │  (GPT-2)  │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                     │                     │       │
│         ▼                     ▼                     ▼       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  Validation  │      │   Resource   │      │  Logging  │ │
│  │   & Errors   │      │  Monitoring  │      │  & Metrics│ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
    HTTP Clients         System Resources        Log Files
```

### Key Design Decisions

**1. Single Model Loading at Startup**
- Load model once during initialization, not per-request
- Reduces latency and memory fragmentation
- Improves predictable resource usage

**2. Pipeline Abstraction**
- Use Hugging Face `pipeline` API for simplicity
- Handles tokenization, generation, and decoding automatically
- Allows easy model swapping

**3. Comprehensive Validation**
- Validate all input parameters before generation
- Return meaningful error messages
- Prevent resource exhaustion attacks

**4. Environment-Based Configuration**
- Use environment variables for all settings
- Support different deployment environments
- No hardcoded values in source code

**5. Structured Logging**
- Log all requests and errors
- Include timing and resource metrics
- Enable debugging and monitoring

## Environment Setup

### Step 1: Create Project Structure

```bash
# Navigate to exercise directory
cd /path/to/exercise-04-llm-basics

# Project structure
exercise-04-llm-basics/
├── src/                          # Source code
│   ├── __init__.py
│   ├── llm_api.py               # Main API server
│   ├── basic_generation.py      # Basic examples
│   ├── parameter_exploration.py # Parameter testing
│   ├── monitor_resources.py     # Resource monitoring
│   └── compare_models.py        # Model comparison
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_api.py
│   └── test_generation.py
├── scripts/                      # Utility scripts
│   ├── setup.sh
│   ├── run.sh
│   └── test.sh
├── requirements.txt              # Dependencies
├── .env                          # Configuration
└── README.md                     # Documentation
```

### Step 2: Install Dependencies

Create `requirements.txt`:

```
# Core dependencies
transformers>=4.30.0      # Hugging Face Transformers library
torch>=2.0.0             # PyTorch backend for transformers
flask>=2.3.0             # Web framework for API
requests>=2.31.0         # HTTP client for testing
psutil>=5.9.0            # System resource monitoring

# Testing
pytest>=7.4.0            # Testing framework
pytest-flask>=1.2.0      # Flask testing utilities
pytest-cov>=4.1.0        # Code coverage

# Development
black>=23.0.0            # Code formatting
flake8>=6.0.0            # Linting
mypy>=1.4.0              # Type checking
```

Install dependencies:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Create `.env` file:

```bash
# Model Configuration
MODEL_NAME=gpt2                  # Model to use (gpt2, distilgpt2, etc.)
DEVICE=-1                        # -1 for CPU, 0 for GPU:0, 1 for GPU:1, etc.

# API Limits
MAX_LENGTH_LIMIT=200            # Maximum allowed generation length
DEFAULT_MAX_LENGTH=50           # Default generation length
DEFAULT_TEMPERATURE=0.7         # Default temperature

# Server Configuration
HOST=0.0.0.0                    # Server host (0.0.0.0 for all interfaces)
PORT=5000                       # Server port
DEBUG=false                     # Debug mode (true/false)

# Hugging Face Configuration
TRANSFORMERS_CACHE=~/.cache/huggingface  # Model cache directory
```

### Step 4: Verify Installation

Create `verify_install.py`:

```python
"""Verify installation and dependencies."""
import sys
import torch
from transformers import pipeline

def verify_installation():
    """Verify all dependencies are installed correctly."""
    print("=" * 60)
    print("Verifying Installation")
    print("=" * 60)

    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")

    # Check PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print("✓ PyTorch installed")

    # Test Transformers
    print("\nTesting Transformers...")
    try:
        generator = pipeline('text-generation', model='gpt2', max_length=20)
        result = generator("Hello", max_length=15, num_return_sequences=1)
        print(f"Test output: {result[0]['generated_text']}")
        print("✓ Transformers working")
    except Exception as e:
        print(f"❌ Transformers error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All checks passed! Ready to proceed.")
    print("=" * 60)
    return True

if __name__ == '__main__':
    verify_installation()
```

Run verification:

```bash
python verify_install.py
```

## Core Concepts

### Understanding LLM Inference

#### What is Inference?

Inference is the process of using a trained model to make predictions:

1. **Input**: Text prompt (e.g., "Machine learning is")
2. **Tokenization**: Convert text to token IDs
3. **Model Forward Pass**: Generate next token probabilities
4. **Sampling**: Select next token based on probabilities
5. **Decoding**: Convert token IDs back to text
6. **Output**: Generated text

#### Token Representation

LLMs work with tokens, not raw characters:

```python
# Example tokenization
text = "Hello, world!"
# Tokens: ["Hello", ",", " world", "!"]
# Token IDs: [15496, 11, 995, 0]
```

Token counts matter for:
- **Memory usage**: More tokens = more computation
- **Speed**: Tokens processed sequentially
- **Cost**: Cloud APIs charge per token
- **Context limits**: Models have maximum token limits

#### Generation Process

Text generation is **autoregressive** - each token depends on previous tokens:

```
Prompt: "The AI model"
Step 1: "The AI model can" (generate "can")
Step 2: "The AI model can learn" (generate "learn")
Step 3: "The AI model can learn from" (generate "from")
...continue until max_length or stop token...
```

### Generation Parameters

#### Temperature (0.0 - 2.0)

Controls randomness in token selection:

```python
# Low temperature (0.1-0.3): Deterministic, focused
generator("Machine learning is", temperature=0.1)
# Output: "Machine learning is a subset of artificial intelligence..."

# High temperature (1.0-2.0): Creative, diverse
generator("Machine learning is", temperature=1.5)
# Output: "Machine learning is revolutionizing quantum physics and art..."
```

**Infrastructure Impact**:
- Temperature doesn't affect speed
- Higher temperature = more diverse outputs
- Set based on use case (factual vs creative)

#### Max Length

Total tokens in prompt + generation:

```python
# Short generation (faster)
generator("Hello", max_length=20)

# Long generation (slower)
generator("Hello", max_length=200)
```

**Infrastructure Impact**:
- Linear relationship: 2x length = ~2x time
- Memory usage increases with length
- Set limits to prevent resource exhaustion

#### Top-K Sampling

Consider only top K most probable tokens:

```python
# More focused (K=10)
generator("AI is", top_k=10)

# More diverse (K=50)
generator("AI is", top_k=50)
```

**Infrastructure Impact**:
- Minimal performance impact
- Affects output diversity
- Typical values: 30-50

#### Top-P (Nucleus Sampling)

Select from smallest set of tokens with cumulative probability >= P:

```python
# Focused (p=0.9)
generator("The model", top_p=0.9)

# More diverse (p=0.95)
generator("The model", top_p=0.95)
```

**Infrastructure Impact**:
- Minimal performance impact
- Can combine with top_k
- Typical values: 0.9-1.0

### Resource Requirements

#### Memory Usage by Model Size

```
┌─────────────┬────────────┬──────────┬──────────────┐
│ Model       │ Parameters │ Memory   │ Storage      │
├─────────────┼────────────┼──────────┼──────────────┤
│ distilgpt2  │ 82M        │ ~350 MB  │ ~350 MB      │
│ gpt2        │ 124M       │ ~500 MB  │ ~500 MB      │
│ gpt2-medium │ 355M       │ ~1.5 GB  │ ~1.5 GB      │
│ gpt2-large  │ 774M       │ ~3 GB    │ ~3 GB        │
│ gpt2-xl     │ 1.5B       │ ~6 GB    │ ~6 GB        │
└─────────────┴────────────┴──────────┴──────────────┘
```

**Memory Formula**:
```
Memory (GB) ≈ Parameters × 4 bytes / (1024^3)
```

For FP32 (32-bit floating point), each parameter = 4 bytes.

#### CPU vs GPU Performance

**CPU Inference** (typical):
- distilgpt2: 2-3 tokens/second
- gpt2: 1-2 tokens/second
- gpt2-medium: 0.5-1 tokens/second

**GPU Inference** (NVIDIA T4):
- distilgpt2: 50-100 tokens/second
- gpt2: 30-50 tokens/second
- gpt2-medium: 20-30 tokens/second

**Infrastructure Recommendation**:
- Development/Testing: CPU acceptable
- Production: GPU required for reasonable latency
- Consider model quantization for reduced memory

## Implementation Details

### Basic Text Generation

File: `src/basic_generation.py`

```python
"""Basic Text Generation

Demonstrates the simplest way to generate text using Hugging Face transformers.
"""

from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run basic text generation example."""
    print("=" * 80)
    print("BASIC TEXT GENERATION")
    print("=" * 80)

    # Initialize the pipeline
    logger.info("Loading model...")
    generator = pipeline(
        'text-generation',
        model='gpt2',           # Model name
        device=-1                # -1 for CPU, 0 for GPU
    )
    logger.info("Model loaded successfully")

    # Generate text
    prompt = "Machine learning is"
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerating responses...\n")

    results = generator(
        prompt,
        max_length=50,          # Total tokens (prompt + generation)
        num_return_sequences=3, # Number of variations
        temperature=0.7,        # Randomness
        top_k=50,              # Top-K sampling
        do_sample=True          # Enable sampling (required for temperature)
    )

    # Display results
    for i, result in enumerate(results, 1):
        print(f"Generation {i}:")
        print(result['generated_text'])
        print("-" * 80)

    print("\n✓ Generation complete!")


if __name__ == '__main__':
    main()
```

**Key Points**:
- Pipeline abstracts complexity
- Model downloaded on first run (~500MB for GPT-2)
- Cached locally for subsequent runs
- `do_sample=True` required for temperature/top_k

### Parameter Exploration

File: `src/parameter_exploration.py`

This script demonstrates how different parameters affect output:

**Temperature Comparison**:
```python
# Low temperature (deterministic)
result = generator(prompt, temperature=0.1, max_length=50)
# Expected: Consistent, focused output

# High temperature (creative)
result = generator(prompt, temperature=1.5, max_length=50)
# Expected: Diverse, sometimes unexpected output
```

**Length Comparison**:
```python
# Short generation
result = generator(prompt, max_length=20)
# Faster, may be incomplete

# Long generation
result = generator(prompt, max_length=100)
# Slower, more complete thoughts
```

### Resource Monitoring

File: `src/monitor_resources.py`

```python
"""Resource Monitoring

Monitor CPU, memory, and performance during LLM operations.
Essential for infrastructure planning and optimization.
"""

import psutil
import time
from transformers import pipeline


class ResourceMonitor:
    """Monitor system resources during LLM operations."""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_percent(self) -> float:
        """Get CPU usage percentage."""
        return self.process.cpu_percent(interval=1.0)


def monitor_model_loading(model_name: str = 'gpt2'):
    """Monitor resources during model loading."""
    monitor = ResourceMonitor()

    # Baseline memory
    mem_before = monitor.get_memory_usage()
    print(f"Memory before loading: {mem_before:.2f} MB")

    # Load model
    start_time = time.time()
    generator = pipeline('text-generation', model=model_name, device=-1)
    load_time = time.time() - start_time

    # Memory after loading
    mem_after = monitor.get_memory_usage()
    mem_delta = mem_after - mem_before

    print(f"Memory after loading: {mem_after:.2f} MB")
    print(f"Model memory usage: {mem_delta:.2f} MB")
    print(f"Load time: {load_time:.2f} seconds")

    return generator, mem_delta, load_time


def monitor_inference(generator, prompt: str, runs: int = 5):
    """Monitor resources during inference."""
    monitor = ResourceMonitor()
    inference_times = []

    for i in range(runs):
        start_time = time.time()
        result = generator(prompt, max_length=50, num_return_sequences=1)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

    avg_time = sum(inference_times) / len(inference_times)
    print(f"\nAverage inference time: {avg_time:.3f}s")
    print(f"Min: {min(inference_times):.3f}s, Max: {max(inference_times):.3f}s")

    return avg_time
```

**Infrastructure Insights**:
- Model loading is expensive (2-5 seconds)
- Memory usage is predictable
- First inference may be slower (warmup)
- Subsequent inferences are consistent

### Model Comparison

File: `src/compare_models.py`

Compares different model sizes:

```python
models_to_test = [
    "distilgpt2",    # Smallest, fastest
    "gpt2",          # Balanced
    "gpt2-medium",   # Larger, slower (CPU)
]

for model in models_to_test:
    # Test loading time
    # Test memory usage
    # Test inference speed
    # Compare output quality
```

**Comparison Metrics**:
- Load time (seconds)
- Memory usage (MB)
- Inference time (seconds)
- Tokens per second
- Output quality (subjective)

### Production API Implementation

File: `src/llm_api.py`

The main API server with production features:

#### 1. Model Initialization

```python
# Global variable for model (loaded once at startup)
generator = None

def initialize_model():
    """Initialize the LLM model at startup."""
    global generator

    logger.info(f"Loading model: {MODEL_NAME}")
    start_time = time.time()

    generator = pipeline(
        'text-generation',
        model=MODEL_NAME,
        device=DEVICE
    )

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s")
```

**Why This Matters**:
- Loading per-request would add 2-5 seconds latency
- Memory would fragment over time
- Concurrent requests would compete for resources

#### 2. Request Validation

```python
def validate_generation_request(data: Dict[str, Any]):
    """Validate request parameters."""

    # Check prompt exists and is non-empty
    if not data or 'prompt' not in data:
        return "Missing required parameter: 'prompt'", 400

    prompt = data.get('prompt', '').strip()
    if not prompt:
        return "Prompt cannot be empty", 400

    # Validate max_length
    max_length = data.get('max_length', DEFAULT_MAX_LENGTH)
    if max_length > MAX_LENGTH_LIMIT:
        return f"max_length cannot exceed {MAX_LENGTH_LIMIT}", 400

    # Validate temperature
    temperature = data.get('temperature', DEFAULT_TEMPERATURE)
    if not (0.0 <= temperature <= 2.0):
        return "temperature must be between 0.0 and 2.0", 400

    return None, None  # No error
```

**Validation Prevents**:
- Empty requests wasting resources
- Excessive generation lengths (DoS)
- Invalid parameter values
- Type errors during generation

#### 3. Generation Endpoint

```python
@app.route('/generate', methods=['POST'])
def generate():
    """Text generation endpoint."""

    # Check model is loaded
    if generator is None:
        return jsonify({"error": "Model not loaded"}), 503

    # Get and validate request
    data = request.get_json()
    error, status_code = validate_generation_request(data)
    if error:
        return jsonify({"error": error}), status_code

    # Extract parameters
    prompt = data.get('prompt').strip()
    max_length = int(data.get('max_length', DEFAULT_MAX_LENGTH))
    temperature = float(data.get('temperature', DEFAULT_TEMPERATURE))

    # Generate text
    start_time = time.time()
    result = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True if temperature > 0 else False
    )
    inference_time = time.time() - start_time

    # Return response
    return jsonify({
        "success": True,
        "prompt": prompt,
        "generated_text": result[0]['generated_text'],
        "inference_time_seconds": round(inference_time, 3),
        "parameters": {
            "max_length": max_length,
            "temperature": temperature
        }
    }), 200
```

#### 4. Health Check Endpoint

```python
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for load balancers."""

    if generator is None:
        return jsonify({
            "status": "unhealthy",
            "error": "Model not loaded"
        }), 503

    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "device": "CPU" if DEVICE == -1 else f"GPU:{DEVICE}"
    }), 200
```

**Why Health Checks Matter**:
- Load balancers need to know service status
- Kubernetes probes use health endpoints
- Prevents routing traffic to unhealthy instances
- Enables graceful degradation

#### 5. Error Handling

```python
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/health", "/info", "/generate"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), 500
```

## Production Considerations

### Memory Management

#### Understanding Memory Usage

LLM memory consists of:

1. **Model Weights**: The parameters (~500MB for GPT-2)
2. **Activations**: Intermediate computations during inference
3. **KV Cache**: Key-value cache for faster generation
4. **Framework Overhead**: PyTorch, transformers libraries

**Total Memory Formula**:
```
Total = Model Weights + (Batch Size × Sequence Length × Hidden Size × Layers)
```

#### GPU Memory Management

If using GPU:

```python
import torch

# Clear cache before loading model
torch.cuda.empty_cache()

# Monitor GPU memory
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"GPU memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
```

#### Out-of-Memory (OOM) Prevention

```python
def safe_generate(generator, prompt, max_length):
    """Generate with OOM protection."""
    try:
        result = generator(
            prompt,
            max_length=min(max_length, 200),  # Cap max length
            num_return_sequences=1  # Don't generate multiple sequences
        )
        return result
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("OOM error, clearing cache and retrying with shorter length")
            torch.cuda.empty_cache()
            # Retry with shorter length
            return generator(prompt, max_length=50, num_return_sequences=1)
        raise
```

### Performance Optimization

#### 1. Batch Processing

Process multiple requests together:

```python
# Instead of:
for prompt in prompts:
    result = generator(prompt, max_length=50)

# Do this (if supported):
results = generator(prompts, max_length=50)  # Batch inference
```

**Benefits**:
- Better GPU utilization
- Higher throughput
- Same latency per request

#### 2. Model Quantization

Reduce memory and improve speed:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    load_in_8bit=True,  # Reduce from FP32 to INT8
    device_map="auto"
)
```

**Impact**:
- Memory: 75% reduction (4 bytes → 1 byte per parameter)
- Speed: Often faster on compatible hardware
- Quality: Minimal degradation (<1%)

#### 3. Caching Common Prompts

```python
from functools import lru_cache
import hashlib

def get_cache_key(prompt, max_length, temperature):
    """Generate cache key for request."""
    key = f"{prompt}:{max_length}:{temperature}"
    return hashlib.md5(key.encode()).hexdigest()

# Simple in-memory cache
cache = {}

def cached_generate(prompt, max_length, temperature):
    """Generate with caching."""
    key = get_cache_key(prompt, max_length, temperature)

    if key in cache:
        logger.info(f"Cache hit for prompt: {prompt[:30]}...")
        return cache[key]

    result = generator(prompt, max_length=max_length, temperature=temperature)
    cache[key] = result
    return result
```

### Monitoring and Observability

#### Logging Best Practices

```python
import logging
import json

# Structured logging
logger.info(
    "Generation request",
    extra={
        "prompt_length": len(prompt),
        "max_length": max_length,
        "temperature": temperature,
        "inference_time": inference_time,
        "request_id": request_id
    }
)

# JSON logging for aggregation
log_data = {
    "timestamp": time.time(),
    "event": "generation_complete",
    "prompt_length": len(prompt),
    "inference_time": inference_time,
    "tokens_generated": len(result[0]['generated_text'].split())
}
logger.info(json.dumps(log_data))
```

#### Metrics to Track

1. **Latency Metrics**:
   - P50, P95, P99 inference time
   - Time to first token
   - End-to-end request time

2. **Throughput Metrics**:
   - Requests per second
   - Tokens generated per second
   - Batch size utilization

3. **Resource Metrics**:
   - CPU utilization
   - Memory usage
   - GPU memory usage
   - GPU utilization

4. **Error Metrics**:
   - Error rate by type
   - OOM errors
   - Timeout errors

### Deployment Patterns

#### Container Deployment (Docker)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY .env .

# Download model at build time (optional)
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='gpt2')"

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "src/llm_api.py"]
```

Build and run:

```bash
docker build -t llm-api:latest .
docker run -p 5000:5000 --memory=2g llm-api:latest
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: llm-api
        image: llm-api:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MODEL_NAME
          value: "gpt2"
        - name: DEVICE
          value: "-1"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
```

## Testing and Validation

### Unit Testing

File: `tests/test_generation.py`

```python
import pytest
from src.llm_api import validate_generation_request

def test_validate_valid_request():
    """Test validation of valid request."""
    data = {
        "prompt": "Test prompt",
        "max_length": 50,
        "temperature": 0.7
    }
    error, status = validate_generation_request(data)
    assert error is None
    assert status is None

def test_validate_missing_prompt():
    """Test validation with missing prompt."""
    data = {"max_length": 50}
    error, status = validate_generation_request(data)
    assert error is not None
    assert status == 400

def test_validate_invalid_temperature():
    """Test validation with invalid temperature."""
    data = {
        "prompt": "Test",
        "temperature": 3.0  # Invalid: > 2.0
    }
    error, status = validate_generation_request(data)
    assert error is not None
    assert "temperature" in error.lower()
```

### Integration Testing

File: `tests/test_api.py`

```python
import pytest
from src.llm_api import app, initialize_model

@pytest.fixture
def client():
    """Create test client."""
    initialize_model()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_generate_endpoint(client):
    """Test text generation endpoint."""
    response = client.post('/generate', json={
        "prompt": "Test prompt",
        "max_length": 20,
        "temperature": 0.7
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'generated_text' in data
    assert 'inference_time_seconds' in data

def test_generate_invalid_request(client):
    """Test generation with invalid request."""
    response = client.post('/generate', json={
        "max_length": 20  # Missing prompt
    })
    assert response.status_code == 400
```

### Load Testing

```python
import concurrent.futures
import time

def load_test(url, num_requests=100, concurrency=10):
    """Simple load test."""

    def make_request():
        response = requests.post(f"{url}/generate", json={
            "prompt": "Test prompt",
            "max_length": 50
        })
        return response.status_code, response.elapsed.total_seconds()

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    total_time = time.time() - start_time
    latencies = [r[1] for r in results]

    print(f"Total requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Total time: {total_time:.2f}s")
    print(f"RPS: {num_requests/total_time:.2f}")
    print(f"Avg latency: {sum(latencies)/len(latencies):.3f}s")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.3f}s")
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Model Download Fails

**Symptoms**: Connection timeout, SSL errors

**Solutions**:
```bash
# Set custom cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Use HTTP instead of HTTPS (if behind proxy)
export HF_HUB_DISABLE_SSL_VERIFY=1

# Download model manually
python -c "from transformers import pipeline; pipeline('text-generation', model='gpt2')"
```

#### Issue 2: Out of Memory (OOM)

**Symptoms**: RuntimeError: CUDA out of memory, Killed

**Solutions**:
```python
# Use smaller model
MODEL_NAME = "distilgpt2"  # Instead of gpt2

# Reduce max_length
MAX_LENGTH_LIMIT = 100  # Instead of 200

# Use CPU instead of GPU (if GPU memory is limited)
DEVICE = -1

# Enable gradient checkpointing (if training)
model.gradient_checkpointing_enable()
```

#### Issue 3: Slow Inference

**Symptoms**: High latency, low throughput

**Solutions**:
```python
# Use GPU if available
DEVICE = 0  # Use first GPU

# Use smaller model
MODEL_NAME = "distilgpt2"

# Reduce max_length
max_length = 50  # Instead of 200

# Use model quantization
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2", load_in_8bit=True)
```

#### Issue 4: Port Already in Use

**Symptoms**: OSError: [Errno 48] Address already in use

**Solutions**:
```bash
# Find process using port
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port
export PORT=5001
python src/llm_api.py
```

## Performance Optimization

### Optimization Checklist

- [ ] Use GPU for inference (10-50x faster)
- [ ] Load model once at startup
- [ ] Implement request batching
- [ ] Use model quantization (INT8/INT4)
- [ ] Enable KV cache for generation
- [ ] Set appropriate max_length limits
- [ ] Implement response caching
- [ ] Use async request handling
- [ ] Profile and optimize bottlenecks
- [ ] Monitor resource utilization

### Advanced Optimization Techniques

#### 1. Flash Attention

```python
# Requires flash-attn library
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

#### 2. Compilation (PyTorch 2.0+)

```python
import torch

# Compile model for faster inference
model = torch.compile(model, mode="reduce-overhead")
```

#### 3. Continuous Batching

```python
# Implement request queue with dynamic batching
from queue import Queue
import threading

request_queue = Queue()

def batch_processor():
    """Process requests in batches."""
    while True:
        batch = []
        # Collect requests for up to 100ms
        timeout = time.time() + 0.1
        while time.time() < timeout and len(batch) < 8:
            try:
                batch.append(request_queue.get(timeout=0.01))
            except:
                break

        if batch:
            # Process batch together
            prompts = [req['prompt'] for req in batch]
            results = generator(prompts, max_length=50)
            # Return results to requests
```

## Advanced Topics

### Prompt Engineering

Improve output quality through better prompts:

```python
# Basic prompt
prompt = "Explain Docker"

# Better prompt (more context)
prompt = "Explain Docker in simple terms for a beginner:"

# Best prompt (with examples - few-shot)
prompt = """
Q: What is Kubernetes?
A: Kubernetes is a container orchestration platform.

Q: What is Docker?
A:"""
```

### Token-Level Control

Fine-grained control over generation:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Tokenize input
inputs = tokenizer("Hello", return_tensors="pt")

# Generate with token-level control
outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,  # Reduce repetition
    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
    early_stopping=True      # Stop at EOS token
)

# Decode output
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Streaming Responses

Stream tokens as they're generated:

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_generate(prompt, max_length=50):
    """Generate and stream tokens."""

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # Generate in separate thread
    generation_kwargs = {
        "input_ids": tokenizer(prompt, return_tensors="pt")['input_ids'],
        "max_length": max_length,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream tokens
    for new_text in streamer:
        yield new_text

    thread.join()

# Use with Flask
@app.route('/stream', methods=['POST'])
def stream():
    prompt = request.json['prompt']
    return Response(stream_generate(prompt), mimetype='text/plain')
```

## Summary

### Key Takeaways

1. **LLMs are resource-intensive**: Plan for 500MB-3GB+ memory per model
2. **Model loading is expensive**: Load once at startup, not per-request
3. **CPU inference is slow**: Use GPU for production workloads
4. **Parameters matter**: Temperature, max_length, top_k/top_p affect output
5. **Validation is critical**: Prevent resource exhaustion and errors
6. **Monitoring is essential**: Track latency, throughput, resource usage
7. **Optimization is key**: Use quantization, batching, caching
8. **Testing is crucial**: Unit, integration, and load testing

### Production Readiness Checklist

- [ ] Model loaded at startup
- [ ] Health check endpoint implemented
- [ ] Request validation in place
- [ ] Error handling comprehensive
- [ ] Logging structured and informative
- [ ] Monitoring and metrics enabled
- [ ] Resource limits configured
- [ ] Timeouts set appropriately
- [ ] Rate limiting implemented (if needed)
- [ ] Documentation complete
- [ ] Tests passing
- [ ] Load tested

### Next Steps

1. **Explore larger models**: Try Flan-T5, Mistral-7B, LLaMA-2
2. **Implement optimizations**: Quantization, batching, streaming
3. **Add features**: Multi-model support, caching, authentication
4. **Learn advanced serving**: vLLM, TGI, TensorRT-LLM
5. **Study distributed inference**: Model parallelism, pipeline parallelism

## Resources

- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **GPT-2 Model Card**: https://huggingface.co/gpt2
- **vLLM (Fast Inference)**: https://vllm.readthedocs.io/
- **Text Generation Inference**: https://github.com/huggingface/text-generation-inference
- **LLM Optimization Guide**: https://huggingface.co/docs/transformers/llm_tutorial

---

**Congratulations!** You now have a comprehensive understanding of LLM inference infrastructure. This knowledge forms the foundation for deploying production ML systems.
