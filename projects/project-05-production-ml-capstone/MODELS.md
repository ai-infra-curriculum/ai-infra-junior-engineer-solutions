# Model Registry - Production ML System

This document tracks all model versions deployed in the production ML system. Based on Module 003 Exercise 08 best practices for model versioning and registry management.

## Quick Navigation

- [Current Production Model](#current-production-model)
- [Version History](#version-history)
- [Deployment Instructions](#deployment-instructions)
- [Rollback Procedures](#rollback-procedures)
- [Model Versioning Guidelines](#model-versioning-guidelines)

---

## Current Production Model

**Model**: `ml-classifier-v2.1.0`
**Status**: ✅ Production
**Deployed**: 2024-01-25
**Git Tag**: `model-v2.1.0`
**Accuracy**: 96.2%
**Latency (P95)**: 38ms

### Quick Deploy
```bash
git checkout model-v2.1.0
kubectl set image deployment/model-api model=gcr.io/project/model-api:v2.1.0
```

---

## Version History

### Summary Table

| Version | Date | Accuracy | F1 Score | Latency P95 | Model Size | Git Tag | Status |
|---------|------|----------|----------|-------------|------------|---------|--------|
| **2.1.0** | 2024-01-25 | **96.2%** | **95.8%** | **38ms** | 245MB | model-v2.1.0 | ✅ **Production** |
| 2.0.0 | 2024-01-15 | 95.5% | 94.9% | 42ms | 250MB | model-v2.0.0 | 🟡 Supported |
| 1.2.0 | 2024-01-05 | 94.8% | 94.2% | 45ms | 248MB | model-v1.2.0 | 🟡 Supported |
| 1.1.0 | 2023-12-20 | 93.9% | 93.3% | 47ms | 240MB | model-v1.1.0 | 🔴 Deprecated |
| 1.0.0 | 2023-12-10 | 92.5% | 91.8% | 50ms | 235MB | model-v1.0.0 | 🔴 Deprecated |

### Performance Trend

```
Accuracy Improvement Over Time
96.2% ┤     ╭─
95.5% ┤   ╭─╯
94.8% ┤ ╭─╯
93.9% ┤╭╯
92.5% ┼╯
      └─────────────────────
      v1.0  v1.1  v1.2  v2.0  v2.1
```

---

## Model Details

### Version 2.1.0 (Current Production) ✅

**Release Date**: 2024-01-25
**Git Tag**: `model-v2.1.0`
**Git Commit**: `a1b2c3d`

#### Model Information
- **Architecture**: ResNet-50 with attention layers
- **Framework**: PyTorch 2.1.0
- **Format**: ONNX (optimized for inference)
- **Model Size**: 245 MB
- **Parameters**: 26.5M

#### Training Details
- **Dataset**: ImageNet-1K v2024.1
- **Training Samples**: 1.2M images
- **Validation Samples**: 50K images
- **Test Samples**: 100K images
- **Training Time**: 72 hours on 8x A100 GPUs
- **Experiment ID**: `exp-2024-01-20-resnet50-attention`

#### Hyperparameters
```yaml
training:
  epochs: 100
  batch_size: 512
  learning_rate: 0.0001
  optimizer: adamw
  weight_decay: 0.01
  lr_scheduler: cosine_annealing
  warmup_epochs: 10

augmentation:
  mixup_alpha: 0.2
  cutmix_alpha: 1.0
  random_erase_prob: 0.25
```

#### Performance Metrics
```yaml
accuracy:
  train: 0.982
  validation: 0.965
  test: 0.962

f1_score:
  macro: 0.958
  weighted: 0.960

precision: 0.964
recall: 0.961

confusion_matrix_summary:
  true_positives: 96200
  false_positives: 1800
  false_negatives: 2000
```

#### Inference Performance
```yaml
latency:
  p50: 25ms
  p95: 38ms
  p99: 52ms

throughput:
  requests_per_second: 450
  batch_size_1: 350 RPS
  batch_size_8: 1200 RPS

resource_usage:
  memory: 2.5 GB
  gpu_memory: 4.0 GB (with GPU)
  cpu_cores: 4
```

#### Improvements Over v2.0.0
- ✅ **+0.7%** accuracy improvement
- ✅ **+0.9%** F1 score improvement
- ✅ **-4ms** latency improvement (P95)
- ✅ **-5MB** model size reduction
- ✅ Better handling of edge cases
- ✅ Improved calibration on uncertainty estimates

#### Changes
1. Added attention mechanism to ResNet-50
2. Improved data augmentation with MixUp and CutMix
3. Extended training to 100 epochs with cosine annealing
4. Applied post-training quantization for inference
5. Optimized ONNX export with graph optimizations

#### Deployment Configuration
```yaml
deployment:
  replicas: 5
  resources:
    requests:
      cpu: 2
      memory: 3Gi
    limits:
      cpu: 4
      memory: 6Gi

  health_checks:
    liveness_probe:
      path: /health
      interval: 10s
    readiness_probe:
      path: /ready
      interval: 5s

  auto_scaling:
    min_replicas: 3
    max_replicas: 20
    target_cpu: 70%
    target_memory: 80%
```

#### Known Issues
- None reported in production

#### Backward Compatibility
- ✅ Fully compatible with v2.0.0 API
- ✅ Same input/output schema
- ✅ No breaking changes

---

### Version 2.0.0 (Supported) 🟡

**Release Date**: 2024-01-15
**Git Tag**: `model-v2.0.0`
**Status**: Supported (rollback available)

**Key Changes from v1.2.0**:
- Major architecture update: ResNet-50 → EfficientNet-B3
- Dataset upgrade: ImageNet-1K → ImageNet-21K subset
- **Breaking change**: New preprocessing requirements

**Performance**:
- Accuracy: 95.5%
- F1 Score: 94.9%
- Latency P95: 42ms

**Why Superseded**: v2.1.0 provides better accuracy and latency with same architecture

---

### Version 1.2.0 (Supported) 🟡

**Release Date**: 2024-01-05
**Git Tag**: `model-v1.2.0`
**Status**: Supported for legacy clients

**Key Changes from v1.1.0**:
- Improved training data quality
- Bug fix: Corrected preprocessing normalization
- Minor performance improvements

**Performance**:
- Accuracy: 94.8%
- F1 Score: 94.2%
- Latency P95: 45ms

**Support Until**: 2024-03-01

---

### Version 1.1.0 (Deprecated) 🔴

**Release Date**: 2023-12-20
**Git Tag**: `model-v1.1.0`
**Status**: Deprecated (not recommended)

**Deprecation Reason**: Superseded by v2.x with significant improvements

---

### Version 1.0.0 (Deprecated) 🔴

**Release Date**: 2023-12-10
**Git Tag**: `model-v1.0.0`
**Status**: Deprecated

**Initial Release**: Baseline model for production ML system

---

## Deployment Instructions

### Prerequisites
```bash
# Required tools
- git
- git-lfs
- kubectl
- helm (optional)
- Docker
```

### Deploy Latest Model

```bash
# 1. Clone repository
git clone https://github.com/org/production-ml-system.git
cd production-ml-system

# 2. Initialize Git LFS
git lfs install
git lfs pull

# 3. Checkout specific model version
git checkout model-v2.1.0

# 4. Verify model files
ls -lh models/production/
git lfs ls-files

# 5. Build Docker image
docker build -t model-api:v2.1.0 -f docker/Dockerfile .

# 6. Push to registry
docker tag model-api:v2.1.0 gcr.io/project/model-api:v2.1.0
docker push gcr.io/project/model-api:v2.1.0

# 7. Deploy to Kubernetes
kubectl set image deployment/model-api \
  model=gcr.io/project/model-api:v2.1.0 \
  -n production

# 8. Verify deployment
kubectl rollout status deployment/model-api -n production
kubectl get pods -n production -l app=model-api
```

### Canary Deployment

```bash
# 1. Deploy canary version (10% traffic)
kubectl apply -f kubernetes/overlays/canary/

# 2. Monitor metrics for 15 minutes
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Check Grafana dashboard: Model Performance

# 3. If metrics are good, promote to full deployment
kubectl patch deployment model-api \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"model","image":"gcr.io/project/model-api:v2.1.0"}]}}}}' \
  -n production

# 4. Cleanup canary
kubectl delete -f kubernetes/overlays/canary/
```

### Blue-Green Deployment

```bash
# 1. Deploy green version
kubectl apply -f kubernetes/overlays/green/

# 2. Verify green deployment
kubectl get pods -n production -l version=green

# 3. Switch traffic to green
kubectl patch service model-api \
  -p '{"spec":{"selector":{"version":"green"}}}' \
  -n production

# 4. Monitor for issues
# If successful, delete blue deployment after 24 hours
kubectl delete -f kubernetes/overlays/blue/
```

---

## Rollback Procedures

### Quick Rollback (Kubernetes)

```bash
# Rollback to previous version
kubectl rollout undo deployment/model-api -n production

# Rollback to specific revision
kubectl rollout undo deployment/model-api --to-revision=3 -n production

# Verify rollback
kubectl rollout status deployment/model-api -n production
```

### Git-Based Rollback

```bash
# 1. Identify target version
git tag -l "model-v*"

# 2. Checkout previous version
git checkout model-v2.0.0

# 3. Rebuild and redeploy
docker build -t model-api:v2.0.0 -f docker/Dockerfile .
docker push gcr.io/project/model-api:v2.0.0

kubectl set image deployment/model-api \
  model=gcr.io/project/model-api:v2.0.0 \
  -n production
```

### Emergency Rollback Checklist

- [ ] Identify issue (latency spike, accuracy drop, errors)
- [ ] Check current version: `kubectl describe deployment model-api -n production`
- [ ] Review last working version in MODELS.md
- [ ] Execute rollback (Kubernetes or Git method)
- [ ] Verify metrics return to normal
- [ ] Document incident in postmortem
- [ ] Create hotfix branch for investigation

---

## Model Versioning Guidelines

### Semantic Versioning

We follow semantic versioning for models: `MAJOR.MINOR.PATCH`

#### MAJOR Version (X.0.0)
Increment when:
- ✅ Breaking changes to API or input/output schema
- ✅ Different model architecture
- ✅ Incompatible preprocessing requirements
- ✅ Significant behavior changes

**Example**: v1.2.0 → v2.0.0 (ResNet-50 → EfficientNet-B3)

#### MINOR Version (X.Y.0)
Increment when:
- ✅ Backward-compatible improvements
- ✅ Accuracy or performance improvements
- ✅ New features added
- ✅ Dataset updates (compatible)

**Example**: v2.0.0 → v2.1.0 (Added attention mechanism)

#### PATCH Version (X.Y.Z)
Increment when:
- ✅ Bug fixes
- ✅ Small optimizations
- ✅ Documentation updates
- ✅ No functional changes

**Example**: v2.1.0 → v2.1.1 (Fixed preprocessing bug)

### Tagging Models

```bash
# Create annotated tag
git tag -a model-v2.1.0 -m "Model Release v2.1.0

ResNet-50 with attention mechanism
- Accuracy: 96.2%
- F1 Score: 95.8%
- Latency P95: 38ms

Training:
- Experiment: exp-2024-01-20-resnet50-attention
- Dataset: ImageNet-1K v2024.1
- 100 epochs, 8x A100 GPUs

Improvements:
- +0.7% accuracy vs v2.0.0
- -4ms latency vs v2.0.0
- Added attention mechanism
- Improved data augmentation"

# Push tag
git push origin model-v2.1.0
```

### Model Metadata Requirements

Every model release must include:

1. **Model File** (tracked with Git LFS):
   - `models/production/model-v{version}.onnx`

2. **Metadata File** (YAML):
   - `models/production/model-v{version}.yaml`

3. **Git Tag**:
   - Format: `model-v{version}`
   - Annotated with release notes

4. **Registry Entry**:
   - Updated in this MODELS.md file
   - Performance metrics documented
   - Deployment instructions provided

### Release Checklist

Before releasing a new model version:

- [ ] Model training completed and validated
- [ ] Experiment results documented in MLflow
- [ ] Model file exported and tested locally
- [ ] Performance metrics meet or exceed previous version
- [ ] Unit tests pass with new model
- [ ] Integration tests pass in staging
- [ ] Load tests show acceptable performance
- [ ] Model metadata file created
- [ ] Git commit with conventional commit message
- [ ] Git tag created with detailed release notes
- [ ] MODELS.md updated with new version
- [ ] Deployment guide reviewed
- [ ] Canary deployment tested
- [ ] Production deployment approved
- [ ] Monitoring dashboards updated
- [ ] Team notified of new release

---

## Model Selection Guide

### Choosing the Right Model Version

| Use Case | Recommended Version | Reason |
|----------|-------------------|---------|
| **New Production** | v2.1.0 | Best accuracy and latency |
| **Existing v2.x Clients** | v2.1.0 | Backward compatible upgrade |
| **Legacy v1.x Clients** | v1.2.0 | Maintained for compatibility |
| **High Throughput** | v2.1.0 | Optimized inference performance |
| **Low Latency** | v2.1.0 | 38ms P95 latency |
| **Rollback Target** | v2.0.0 | Stable, well-tested |

---

## Monitoring & Alerts

### Key Metrics to Monitor

```yaml
# Model Performance Alerts
- name: ModelAccuracyDrop
  condition: accuracy < 0.95
  severity: critical
  action: Consider rollback

- name: ModelLatencyHigh
  condition: latency_p95 > 50ms
  severity: warning
  action: Investigate performance

- name: ModelErrorRateHigh
  condition: error_rate > 0.01
  severity: critical
  action: Immediate rollback

# Resource Alerts
- name: ModelMemoryHigh
  condition: memory_usage > 5GB
  severity: warning

- name: ModelCPUHigh
  condition: cpu_usage > 90%
  severity: warning
```

### Grafana Dashboards

- **Model Performance**: Real-time accuracy, latency, throughput
- **Model Comparison**: Compare metrics across versions
- **Resource Usage**: CPU, memory, GPU utilization
- **Error Analysis**: Error rates, types, patterns

---

## Data Version Control

Models are tied to specific dataset versions. See DVC configuration:

```bash
# Check dataset version for model
cat models/production/model-v2.1.0.yaml | grep dataset_version

# Pull specific dataset version
dvc pull data/imagenet_v2024.1.dvc
```

---

## Related Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Git workflow and collaboration guidelines
- [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) - PR guidelines
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Detailed deployment procedures
- [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md) - Training best practices

---

## Support & Feedback

For questions about model versions or deployment:
- **Slack**: #ml-platform
- **Email**: ml-platform@company.com
- **Issues**: GitHub Issues

---

**Last Updated**: 2024-01-25
**Maintained By**: ML Platform Team
**Review Schedule**: Monthly
