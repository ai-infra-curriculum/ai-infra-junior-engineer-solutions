# Project 01: Simple Model API - Solution

A production-ready REST API for serving image classification predictions using pre-trained PyTorch models.

## Quick Start

```bash
# Using Docker (recommended)
docker-compose -f docker/docker-compose.yml up

# Or locally
pip install -r src/requirements.txt
python src/app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### GET /health
Health check endpoint
```bash
curl http://localhost:5000/health
```

### GET /info
Model information and metadata
```bash
curl http://localhost:5000/info
```

### POST /predict
Image classification
```bash
curl -X POST -F "file=@image.jpg" -F "top_k=5" http://localhost:5000/predict
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
MODEL_NAME=resnet50  # or mobilenet_v2
MODEL_DEVICE=cpu     # or cuda
API_PORT=5000
LOG_LEVEL=INFO
```

## Project Structure

```
project-01-simple-model-api/
├── src/
│   ├── app.py              # Flask application
│   ├── config.py           # Configuration management
│   ├── model_loader.py     # Model loading and inference
│   └── requirements.txt    # Python dependencies
├── tests/
│   └── test_app.py         # API tests
├── docker/
│   ├── Dockerfile          # Container definition
│   └── docker-compose.yml  # Docker Compose config
├── docs/
│   └── (documentation)
├── .env.example            # Environment template
├── README.md               # This file
└── SOLUTION_GUIDE.md       # Detailed implementation guide
```

## Features

- Pre-trained ResNet-50 and MobileNetV2 models
- RESTful API design
- Comprehensive error handling
- Docker containerization
- Health checks for Kubernetes
- Structured logging
- Input validation
- Production-ready with Gunicorn
- Comprehensive test suite
- Security best practices

## Documentation

See [SOLUTION_GUIDE.md](SOLUTION_GUIDE.md) for:
- Architecture details
- Implementation decisions
- Deployment guide
- Troubleshooting
- Performance benchmarks

## Requirements

- Python 3.11+
- PyTorch 2.1+
- Flask 3.0+
- Docker (for containerized deployment)

## License

Educational use only - AI Infrastructure Curriculum
