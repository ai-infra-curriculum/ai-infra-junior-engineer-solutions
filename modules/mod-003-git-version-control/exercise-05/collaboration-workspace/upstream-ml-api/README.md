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
