# Sentiment Classifier - ML Training Project

A production-ready sentiment analysis project demonstrating best practices for Python environment management, dependency handling, and ML project structure.

## Features

- ✅ Automated environment setup with `setup.sh`
- ✅ Virtual environment isolation
- ✅ Pinned production dependencies
- ✅ Environment variable configuration
- ✅ Comprehensive testing with pytest
- ✅ Type hints and documentation
- ✅ Professional project structure

## Quick Start

### Automated Setup (Recommended)

```bash
# Run the automated setup script
chmod +x setup.sh
./setup.sh

# Activate the environment
source venv/bin/activate

# Verify setup
python ../scripts/verify_setup.py
```

### Manual Setup

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements-dev.txt

# Create .env file
cp .env.example .env

# Edit .env with your settings
nano .env
```

## Project Structure

```
sentiment-classifier/
├── .gitignore                  # Git ignore patterns
├── README.md                   # This file
├── requirements.txt            # Production dependencies (pinned)
├── requirements-dev.txt        # Development dependencies
├── setup.sh                    # Automated setup script
├── .env.example                # Environment variable template
├── src/                       # Source code
│   ├── __init__.py
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py     # Data loading utilities
│       └── metrics.py         # Metrics computation
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_metrics.py
├── data/                      # Data directory (gitignored)
│   └── .gitkeep
├── models/                    # Model checkpoints (gitignored)
│   └── .gitkeep
└── configs/                   # Configuration files
    ├── training_config.yaml
    └── model_config.yaml
```

## Training

```bash
# Train the sentiment classifier
python src/train.py --config configs/training_config.yaml

# Evaluate the model
python src/evaluate.py --model-path models/sentiment_model.pth
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

Environment variables are loaded from `.env` file. Copy `.env.example` to `.env` and configure:

- `MODEL_NAME`: HuggingFace model identifier
- `DATA_PATH`: Path to training data
- `MODEL_OUTPUT_PATH`: Where to save trained models
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Optimizer learning rate
- `NUM_EPOCHS`: Number of training epochs
- `RANDOM_SEED`: Random seed for reproducibility

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
black src/ tests/
mypy src/

# Run tests
pytest tests/ -v
```

## Requirements

- Python 3.11+
- 4GB RAM minimum
- GPU recommended for training (optional)

## Dependencies

### Production Dependencies (requirements.txt)
- torch==2.1.0 - Deep learning framework
- transformers==4.35.0 - HuggingFace transformers
- pandas==2.1.0 - Data manipulation
- numpy==1.24.0 - Numerical computing
- scikit-learn==1.3.2 - ML utilities
- python-dotenv==1.0.0 - Environment variable management
- pyyaml==6.0.1 - YAML configuration
- tqdm==4.66.1 - Progress bars

### Development Dependencies (requirements-dev.txt)
- pytest==7.4.3 - Testing framework
- pytest-cov==4.1.0 - Test coverage
- black==23.11.0 - Code formatting
- mypy==1.7.1 - Type checking
- flake8==6.1.0 - Code linting

## Troubleshooting

### Issue: Permission denied when running setup.sh
```bash
chmod +x setup.sh
```

### Issue: Python version mismatch
```bash
# Install Python 3.11
# Ubuntu/Debian:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv

# Then create venv with specific version:
python3.11 -m venv venv
```

### Issue: Virtual environment not activating
```bash
# Use 'source' not 'sh':
source venv/bin/activate  # ✓ Correct

# Not:
sh venv/bin/activate      # ✗ Wrong
./venv/bin/activate       # ✗ Wrong
```

## License

MIT License - See LICENSE file for details

## Contributing

This is an educational project. Contributions welcome!

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Resources

- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Sentiment Analysis Tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification)

---

**Version:** 1.0.0
**Last Updated:** 2025-10-30
