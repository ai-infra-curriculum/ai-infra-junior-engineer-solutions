# Model Registry

Production model versions with performance metrics and deployment information.

## Active Models

### BERT Sentiment Classifier

Binary sentiment classification for English text.

#### Version History

| Version | Date | Accuracy | F1 | Latency (p99) | Git Tag | Status | Notes |
|---------|------|----------|-----|---------------|---------|--------|-------|
| **1.1.0** | 2024-01-22 | **95.8%** | **95.1%** | **42ms** | model-bert-v1.1.0 | **✅ Production** | Recommended version |
| 1.0.0 | 2024-01-15 | 94.5% | 93.8% | 45ms | model-bert-v1.0.0 | 🟡 Supported | Legacy support |

#### Quick Start

```bash
# Deploy latest version (v1.1.0)
git checkout model-bert-v1.1.0
git lfs pull --include="models/production/bert-classifier-v1.1.0.onnx"
kubectl apply -f deployment/bert-classifier-v1.1.0.yaml
```

#### Version Details

**v1.1.0** (Current Production)
- **Improvements**: +1.3% accuracy, -3ms latency
- **Dataset**: sentiment-analysis v1.1 (75K samples)
- **Changes**: Data augmentation, improved preprocessing
- **Migration**: Drop-in replacement for v1.0.0
- **Deployment**: All environments

**v1.0.0** (Legacy)
- **Status**: Supported until 2024-03-01
- **Dataset**: sentiment-analysis v1.0 (50K samples)
- **Recommendation**: Migrate to v1.1.0

---

## Deployment Instructions

### Standard Deployment

```bash
# 1. Clone repository
git clone <repo-url>
cd ml-model-repository

# 2. Checkout model version
git checkout model-bert-v1.1.0

# 3. Download model files
git lfs pull --include="models/production/bert-classifier-v1.1.0.onnx"

# 4. Verify model
ls -lh models/production/bert-classifier-v1.1.0.onnx

# 5. Deploy
kubectl apply -f deployment/bert-classifier-v1.1.0.yaml
```

### Fast Clone (Without All LFS Files)

```bash
# Clone without LFS files (saves bandwidth)
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
cd ml-model-repository

# Checkout version
git checkout model-bert-v1.1.0

# Download only this model
git lfs pull --include="models/production/bert-classifier-v1.1.0.onnx"
```

---

## Rollback Procedures

### Rollback from v1.1.0 to v1.0.0

```bash
# 1. Checkout previous version
git checkout model-bert-v1.0.0

# 2. Download model
git lfs pull --include="models/production/bert-classifier-v1.0.0.onnx"

# 3. Deploy
kubectl apply -f deployment/bert-classifier-v1.0.0.yaml

# 4. Verify rollback
kubectl get pods -l model=bert-classifier
```

---

## Model Performance Tracking

### Accuracy Trends

```
v1.0.0: 94.5% ████████████████████████▌
v1.1.0: 95.8% ██████████████████████████
```

### Latency Trends (p99)

```
v1.0.0: 45ms ████████████████████████▌
v1.1.0: 42ms ███████████████████████
```

---

## Version Selection Guide

| Use Case | Recommended Version | Reason |
|----------|---------------------|---------|
| New deployments | v1.1.0 | Best performance |
| Production updates | v1.1.0 | Backward compatible |
| Legacy systems | v1.0.0 | If upgrade testing pending |
| Development/testing | v1.1.0 | Latest features |

---

## Model Metadata

All models include:

- **Model file**: ONNX format (optimized for inference)
- **Metadata**: YAML with training details, metrics, deployment specs
- **Config**: Production configuration
- **Git tag**: For version pinning
- **Documentation**: In `docs/models/`

---

## Adding New Models

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Training new versions
- Model validation requirements
- Metadata templates
- Release process
