# ML Inference API - Team Repository

Production ML inference service for image classification.

## Contributing

We welcome contributions! Please follow our workflow:

1. Fork this repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request
5. Address review feedback

See CONTRIBUTING.md for detailed guidelines.

## Features

- Image classification with ResNet50
- REST API with FastAPI
- Prometheus metrics export
- Health check endpoints

## Setup

```bash
pip install -r requirements.txt
python -m src.api.app
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT License

## Model Monitoring

Track model performance with `ModelMetrics`:

```python
from src.monitoring.model_metrics import model_metrics

# Record predictions
model_metrics.add_prediction(
    prediction=0.95,
    truth=1.0,
    latency=0.05
)

# Get summary
summary = model_metrics.get_summary()
print(f"Accuracy: {summary['accuracy']}%")
print(f"Avg Latency: {summary['average_latency']}s")
```
