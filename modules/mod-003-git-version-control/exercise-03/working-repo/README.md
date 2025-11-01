# ML Image Classification API

A production-ready REST API for image classification using PyTorch pre-trained models. Built with FastAPI for high performance and easy deployment.

## Features

- **Fast and Scalable**: Built on FastAPI with async support
- **Pre-trained Models**: Uses PyTorch pre-trained models (ResNet, MobileNet, EfficientNet)
- **Image Processing**: Automatic image preprocessing and normalization
- **Batch Processing**: Support for batch predictions
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Configuration**: YAML-based configuration with environment overrides
- **Health Checks**: Built-in health check endpoints
- **API Documentation**: Auto-generated OpenAPI docs

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda
- (Optional) CUDA-capable GPU for faster inference

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-inference-api
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Running the API

Development mode with auto-reload:
```bash
python -m src.api.app
```

Or using uvicorn directly:
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8080
```

Production mode:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --workers 4
```

## Usage

### API Endpoints

#### Health Check
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Single Image Prediction
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

Response:
```json
{
  "success": true,
  "predictions": [
    {"class": "golden_retriever", "confidence": 0.92, "class_id": 207},
    {"class": "labrador", "confidence": 0.05, "class_id": 208},
    {"class": "german_shepherd", "confidence": 0.02, "class_id": 235}
  ],
  "top_prediction": "golden_retriever",
  "confidence": 0.92
}
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8080/predict/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

#### Get Available Classes
```bash
curl http://localhost:8080/classes
```

#### Model Information
```bash
curl http://localhost:8080/model/info
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Project Structure

```
ml-inference-api/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application
│   ├── models/
│   │   └── classifier.py       # Image classification model
│   ├── preprocessing/
│   │   └── image.py            # Image preprocessing
│   └── utils/
│       └── logging.py          # Logging configuration
├── configs/
│   ├── default.yaml            # Default configuration
│   └── production.yaml         # Production overrides
├── tests/
│   └── ...                     # Test files
├── scripts/
│   └── ...                     # Utility scripts
├── docs/
│   └── ...                     # Documentation
├── data/
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── .gitignore                  # Git ignore rules
├── .gitattributes              # Git attributes
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

Configuration is managed through YAML files in the `configs/` directory:

- `default.yaml`: Base configuration for all environments
- `production.yaml`: Production-specific overrides

Environment variables can override any config value:
```bash
export LOG_LEVEL=DEBUG
export MODEL_DEVICE=cuda
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
```

### Linting
```bash
flake8 src/
mypy src/
```

### Adding a New Model

1. Update `src/models/classifier.py` to support the new architecture
2. Add model-specific preprocessing if needed
3. Update configuration files
4. Test thoroughly

## Deployment

### Docker

Build image:
```bash
docker build -t ml-inference-api:latest .
```

Run container:
```bash
docker run -p 8080:8080 ml-inference-api:latest
```

### Kubernetes

Apply deployment:
```bash
kubectl apply -f k8s/deployment.yaml
```

### Environment Variables

Key environment variables for production:

- `ENV`: Environment name (production, staging, development)
- `LOG_LEVEL`: Logging level (INFO, WARNING, ERROR)
- `MODEL_DEVICE`: Device for inference (cpu, cuda)
- `API_KEY`: API authentication key
- `REDIS_HOST`: Redis host for caching

## Performance Optimization

### GPU Inference

To use GPU acceleration:
1. Install CUDA and cuDNN
2. Install GPU-enabled PyTorch
3. Set `MODEL_DEVICE=cuda` in config

### Batch Processing

For improved throughput, use batch predictions:
```python
# Process multiple images at once
result = await predict_batch(files=[file1, file2, file3])
```

### Caching

Enable Redis caching for frequently requested predictions:
```yaml
cache:
  enabled: true
  backend: redis
  ttl: 600
```

## Monitoring

### Prometheus Metrics

Enable Prometheus metrics:
```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
```

Metrics available at: http://localhost:9090/metrics

### Logging

Structured JSON logs are written to:
- Console: stdout
- File: `logs/app.log` (with rotation)

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Security

### Authentication

Enable API key authentication:
```yaml
security:
  auth:
    enabled: true
    type: bearer
```

Include API key in requests:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8080/predict
```

### HTTPS

For production, enable HTTPS:
```yaml
security:
  https:
    enabled: true
    cert_path: /path/to/cert.pem
    key_path: /path/to/key.pem
```

## Troubleshooting

### Model fails to load
- Check that PyTorch and torchvision are installed
- Verify CUDA installation if using GPU
- Check available disk space and memory

### Predictions are slow
- Enable GPU inference (`MODEL_DEVICE=cuda`)
- Increase batch size
- Enable caching for repeated requests

### Out of memory errors
- Reduce batch size
- Use a smaller model (e.g., mobilenet_v2 instead of resnet50)
- Adjust `gpu.memory_fraction` in config

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: <repository-url>/issues
- Documentation: <documentation-url>
- Email: support@example.com

## Acknowledgments

- FastAPI for the excellent web framework
- PyTorch team for pre-trained models
- The open-source ML community
