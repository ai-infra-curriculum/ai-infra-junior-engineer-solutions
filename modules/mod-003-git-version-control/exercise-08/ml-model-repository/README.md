# ML Model Repository

Production-ready ML model repository with Git LFS and DVC integration.

## Repository Structure

```
ml-model-repository/
├── models/
│   ├── production/     # Production models (Git LFS)
│   └── experiments/    # Experimental models
├── configs/            # Model configurations
├── data/
│   └── sample/         # Sample data for testing
├── src/                # Source code
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── deployment/         # Deployment configs
├── docs/               # Documentation
├── MODELS.md           # Model registry
└── .gitattributes      # Git LFS configuration
```

## Quick Start

### Clone Repository

```bash
# Clone with all LFS files
git clone <repo-url>

# Clone without LFS files (faster)
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
cd ml-model-repository
git lfs pull --include="models/production/model-name-v1.0.0.onnx"
```

### Deploy Model

```bash
# Checkout specific model version
git checkout model-bert-v1.0.0

# Download model if not already present
git lfs pull --include="models/production/bert-*"

# Deploy
kubectl apply -f deployment/bert-classifier-v1.0.0.yaml
```

## Model Versioning

We use semantic versioning for models:

- **MAJOR**: Breaking changes (different input/output format)
- **MINOR**: Backward-compatible improvements
- **PATCH**: Bug fixes

## Documentation

- [Model Registry](MODELS.md) - All production models
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Development Workflow](docs/DEVELOPMENT.md)
