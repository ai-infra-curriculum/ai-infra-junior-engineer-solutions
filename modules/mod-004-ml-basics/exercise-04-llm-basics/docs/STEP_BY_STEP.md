# Step-by-Step Implementation Guide: LLM Basics

## Overview

Get hands-on with Large Language Models (LLMs). Learn to run inference with pre-trained models, explore generation parameters, compare different models, build LLM APIs, and monitor resource usage.

**Time**: 3-4 hours | **Difficulty**: Intermediate
**Files**: `basic_generation.py`, `parameter_exploration.py`, `compare_models.py`, `llm_api.py`, `monitor_resources.py`

---

## Learning Objectives

✅ Run inference with HuggingFace Transformers
✅ Understand generation parameters (temperature, top-k, top-p)
✅ Compare different LLM architectures
✅ Build APIs for LLM serving
✅ Monitor GPU memory and performance
✅ Optimize LLM inference
✅ Handle long-context scenarios

---

## Quick Start

```bash
# Setup environment
bash scripts/setup.sh

# Basic generation
python src/basic_generation.py \
    --model-name "gpt2" \
    --prompt "Artificial intelligence is"

# Explore parameters
python src/parameter_exploration.py \
    --model-name "gpt2" \
    --prompt "Once upon a time" \
    --temperatures 0.7,1.0,1.5

# Compare models
python src/compare_models.py \
    --models "gpt2,distilgpt2,gpt2-medium" \
    --prompt "The future of AI"

# Start LLM API
python src/llm_api.py \
    --model-name "gpt2" \
    --port 8001

# Monitor resources
python src/monitor_resources.py \
    --interval 5
```

---

## Implementation Guide

### Phase 1: Basic Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0]))
```

### Phase 2: Parameter Exploration

**Temperature**: Controls randomness (0.1 = deterministic, 2.0 = creative)
**Top-k**: Samples from top-k tokens
**Top-p**: Nucleus sampling (cumulative probability)
**Repetition penalty**: Reduces repetitive text

### Phase 3: Model Comparison

**Metrics**: Inference speed, memory usage, output quality, perplexity

### Phase 4: LLM API

**Features**: Streaming responses, caching, batch processing

### Phase 5: Resource Monitoring

**Track**: GPU memory, inference latency, tokens/second

---

## Best Practices

- Use `torch.cuda.amp` for faster inference
- Implement KV-cache for efficiency
- Batch requests when possible
- Monitor GPU memory carefully
- Use quantization (int8, int4) for large models

---

## Optimization Tips

```python
# Use half precision
model = model.half()

# Enable attention optimization
model = model.to_bettertransformer()

# Quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=quantization_config
)
```

---

**LLM fundamentals complete!** 🤖
