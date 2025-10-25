# Exercise 04: LLM Basics - Solution

A complete, production-ready solution for running and serving Large Language Models (LLMs) using Hugging Face Transformers and Flask.

## Overview

This solution demonstrates how to:
- Load and run pre-trained LLMs (GPT-2)
- Understand and experiment with generation parameters
- Monitor resource usage (CPU, memory)
- Compare different model sizes
- Deploy an LLM as a REST API
- Handle errors and validate inputs
- Test LLM applications

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Run Basic Examples

```bash
# Basic text generation
python src/basic_generation.py

# Parameter exploration
python src/parameter_exploration.py

# Resource monitoring
python src/monitor_resources.py

# Model comparison
python src/compare_models.py
```

### 3. Start the API Server

```bash
# Using the run script
./scripts/run.sh

# Or manually
python src/llm_api.py
```

### 4. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Get API info
curl http://localhost:5000/info

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Machine learning is",
    "max_length": 50,
    "temperature": 0.7
  }'
```

### 5. Run Tests

```bash
./scripts/test.sh
```

## Project Structure

```
exercise-04-llm-basics/
├── README.md                    # This file - overview and quick start
├── STEP_BY_STEP.md             # Detailed implementation guide
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
├── .env                        # Environment configuration (created by setup)
├── src/                        # Source code
│   ├── __init__.py
│   ├── basic_generation.py     # Basic text generation demo
│   ├── parameter_exploration.py # Parameter experiments
│   ├── monitor_resources.py    # Resource monitoring utilities
│   ├── compare_models.py       # Model comparison tool
│   └── llm_api.py             # Flask API server
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_api.py            # API endpoint tests
│   └── test_generation.py     # Generation logic tests
└── scripts/                    # Utility scripts
    ├── setup.sh               # Environment setup
    ├── run.sh                 # Run the API server
    └── test.sh                # Run tests
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "gpt2",
  "device": "CPU",
  "cuda_available": false
}
```

### GET /info
Get API and model information.

**Response:**
```json
{
  "api_version": "1.0.0",
  "model": "gpt2",
  "device": "CPU",
  "max_length_limit": 200,
  "default_max_length": 50,
  "default_temperature": 0.7,
  "supported_parameters": {...}
}
```

### POST /generate
Generate text from a prompt.

**Request:**
```json
{
  "prompt": "Your text prompt",
  "max_length": 50,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 1.0
}
```

**Response:**
```json
{
  "success": true,
  "prompt": "Your text prompt",
  "generated_text": "Your text prompt and the generated continuation...",
  "inference_time_seconds": 0.123,
  "parameters": {
    "max_length": 50,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 1.0
  },
  "metadata": {
    "model": "gpt2",
    "device": "CPU"
  }
}
```

## Configuration

Configuration is managed via environment variables in the `.env` file:

```bash
# Model configuration
MODEL_NAME=gpt2              # Model to use (gpt2, distilgpt2, etc.)
DEVICE=-1                    # -1 for CPU, 0+ for GPU

# API limits
MAX_LENGTH_LIMIT=200         # Maximum allowed generation length
DEFAULT_MAX_LENGTH=50        # Default generation length
DEFAULT_TEMPERATURE=0.7      # Default temperature

# Server configuration
HOST=0.0.0.0                # Server host
PORT=5000                   # Server port
DEBUG=false                 # Debug mode
```

## Parameter Guide

### Temperature
Controls randomness in generation.
- **0.1-0.3**: Deterministic, focused
- **0.7-1.0**: Balanced (recommended)
- **1.0-2.0**: Creative, diverse

### Max Length
Total tokens (prompt + generation).
- Shorter: Faster, less complete
- Longer: Slower, more complete

### Top-K
Limits vocabulary to top K probable tokens.
- **10-30**: More focused
- **50**: Balanced (default)
- **100+**: More diverse

### Top-P (Nucleus Sampling)
Uses cumulative probability threshold.
- **0.9**: Focused sampling
- **0.95**: Balanced
- **1.0**: No filtering

## Testing

The solution includes comprehensive tests:

```bash
# Run all tests
./scripts/test.sh

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- API endpoint validation
- Parameter validation
- Error handling
- Generation logic
- Resource monitoring
- Model comparison

## Development

### Code Quality

The code follows best practices:
- Type hints for all functions
- Comprehensive docstrings
- Proper error handling
- Structured logging
- Environment-based configuration

### Adding New Features

1. Add implementation to `src/`
2. Add tests to `tests/`
3. Update documentation
4. Run tests: `./scripts/test.sh`

## Troubleshooting

### Model Download Issues
If model download fails:
```bash
# Set Hugging Face cache directory
export TRANSFORMERS_CACHE=~/.cache/huggingface
```

### Memory Issues
If you encounter OOM (Out of Memory):
- Use smaller model: `distilgpt2`
- Reduce `max_length`
- Close other applications

### Slow Inference
If generation is too slow:
- Expected on CPU (1-2 tokens/second)
- Use GPU for production (set `DEVICE=0`)
- Use smaller model
- Reduce `max_length`

### Port Already in Use
If port 5000 is busy:
```bash
# Change port in .env
PORT=5001

# Or export directly
export PORT=5001
python src/llm_api.py
```

## Resource Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 2GB available
- **Storage**: 1GB for model cache
- **Python**: 3.8+

### Recommended for Production
- **CPU**: 4+ cores or GPU
- **RAM**: 4-8GB
- **Storage**: 5GB+ for multiple models
- **Python**: 3.9+

## Model Sizes

| Model | Parameters | Memory | Speed (CPU) |
|-------|-----------|--------|-------------|
| distilgpt2 | 82M | ~350 MB | ~2-3 tok/s |
| gpt2 | 124M | ~500 MB | ~1-2 tok/s |
| gpt2-medium | 355M | ~1.5 GB | ~0.5-1 tok/s |
| gpt2-large | 774M | ~3 GB | ~0.2-0.5 tok/s |

## Learning Resources

- [STEP_BY_STEP.md](./STEP_BY_STEP.md) - Detailed implementation guide
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [GPT-2 Model Card](https://huggingface.co/gpt2)
- [Flask Documentation](https://flask.palletsprojects.com/)

## Production Deployment

For production deployment, consider:

1. **Use GPU acceleration** for better performance
2. **Implement request queuing** for concurrency
3. **Add rate limiting** to prevent abuse
4. **Set up monitoring** (Prometheus, Grafana)
5. **Use model quantization** to reduce memory
6. **Deploy behind load balancer** for scaling
7. **Implement caching** for common prompts
8. **Add authentication** for API access

## License

This solution is provided as educational material for the AI Infrastructure Junior Engineer course.

## Support

For questions or issues:
1. Check [STEP_BY_STEP.md](./STEP_BY_STEP.md) for detailed explanations
2. Review test files for usage examples
3. Check logs for error messages
4. Refer to the original exercise documentation
