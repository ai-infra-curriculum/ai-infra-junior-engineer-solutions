# Step-by-Step Implementation Guide: End-to-End ML Pipeline

## Overview

Build complete end-to-end ML pipelines! Integrate training, testing, building, deployment, and monitoring into automated workflows for production ML systems.

**Time**: 3-4 hours | **Difficulty**: Advanced

---

## Learning Objectives

✅ Design complete ML pipelines
✅ Orchestrate multi-stage workflows
✅ Implement data validation
✅ Automate model training and deployment
✅ Integrate monitoring and alerting
✅ Implement pipeline versioning
✅ Handle pipeline failures gracefully

---

## Complete Pipeline Workflow

```.github/workflows/ml-pipeline.yml
name: Complete ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'data/**'
      - 'src/**'
      - 'models/**'
  workflow_dispatch:

jobs:
  validate-data:
    name: Data Validation
    runs-on: ubuntu-latest
    outputs:
      data_version: ${{ steps.version.outputs.version }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install pandas great-expectations

      - name: Validate data schema
        run: python scripts/validate_data.py

      - name: Check data quality
        run: python scripts/data_quality_checks.py

      - name: Generate data version
        id: version
        run: echo "version=$(date +%Y%m%d-%H%M%S)" >> $GITHUB_OUTPUT

  train-model:
    name: Model Training
    needs: validate-data
    runs-on: ubuntu-latest
    outputs:
      model_uri: ${{ steps.train.outputs.model_uri }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python with GPU support
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install torch mlflow scikit-learn

      - name: Train model
        id: train
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          DATA_VERSION: ${{ needs.validate-data.outputs.data_version }}
        run: |
          MODEL_URI=$(python scripts/train.py --data-version $DATA_VERSION)
          echo "model_uri=$MODEL_URI" >> $GITHUB_OUTPUT

      - name: Run model tests
        run: pytest tests/model/

  validate-model:
    name: Model Validation
    needs: train-model
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Performance validation
        env:
          MODEL_URI: ${{ needs.train-model.outputs.model_uri }}
        run: python scripts/validate_performance.py --model-uri $MODEL_URI

      - name: Bias detection
        run: python scripts/check_bias.py --model-uri $MODEL_URI

      - name: Generate model card
        run: python scripts/generate_model_card.py --model-uri $MODEL_URI

  build-image:
    name: Build Docker Image
    needs: [train-model, validate-model]
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}

    steps:
      - uses: actions/checkout@v4

      - name: Download model
        env:
          MODEL_URI: ${{ needs.train-model.outputs.model_uri }}
        run: python scripts/download_model.py --model-uri $MODEL_URI --output models/

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ steps.meta.outputs.tags }}

  deploy-staging:
    name: Deploy to Staging
    needs: build-image
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}

      - name: Deploy to staging
        run: |
          kubectl set image deployment/ml-api \
            api=${{ needs.build-image.outputs.image_tag }} \
            -n staging

      - name: Wait for rollout
        run: kubectl rollout status deployment/ml-api -n staging

      - name: Run smoke tests
        run: ./scripts/smoke_tests.sh staging

  integration-tests:
    name: Integration Tests
    needs: deploy-staging
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Run integration tests
        env:
          API_URL: ${{ secrets.STAGING_API_URL }}
        run: pytest tests/integration/ -v

      - name: Load testing
        run: |
          pip install locust
          locust -f tests/load/locustfile.py \
            --headless \
            --users 100 \
            --spawn-rate 10 \
            --run-time 5m \
            --host ${{ secrets.STAGING_API_URL }}

  deploy-production:
    name: Deploy to Production
    needs: [deploy-staging, integration-tests]
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}

      - name: Canary deployment
        run: |
          # Deploy canary with 10% traffic
          kubectl apply -f k8s/canary-deployment.yaml
          kubectl patch ingress ml-api -p '{
            "metadata": {"annotations": {
              "nginx.ingress.kubernetes.io/canary": "true",
              "nginx.ingress.kubernetes.io/canary-weight": "10"
            }}
          }'

      - name: Monitor canary
        run: |
          sleep 300  # Monitor for 5 minutes
          python scripts/check_canary_metrics.py

      - name: Promote or rollback
        run: |
          if [ -f /tmp/canary_success ]; then
            kubectl set image deployment/ml-api \
              api=${{ needs.build-image.outputs.image_tag }} \
              -n production
            kubectl delete -f k8s/canary-deployment.yaml
          else
            kubectl delete -f k8s/canary-deployment.yaml
            exit 1
          fi

  post-deployment:
    name: Post-Deployment Tasks
    needs: deploy-production
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Register model in production
        env:
          MODEL_URI: ${{ needs.train-model.outputs.model_uri }}
        run: |
          python scripts/promote_model.py \
            --model-uri $MODEL_URI \
            --stage Production

      - name: Update documentation
        run: python scripts/update_docs.py

      - name: Send notifications
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "🚀 New model deployed to production!\nModel URI: ${{ needs.train-model.outputs.model_uri }}\nImage: ${{ needs.build-image.outputs.image_tag }}"
            }
```

---

## Pipeline Monitoring

### Monitoring Workflow

```.github/workflows/monitor-pipeline.yml
name: Pipeline Monitoring

on:
  schedule:
    - cron: '0 * * * *'  # Every hour

jobs:
  monitor:
    runs-on: ubuntu-latest

    steps:
      - name: Check model performance
        run: python scripts/check_model_drift.py

      - name: Check data quality
        run: python scripts/monitor_data_quality.py

      - name: Alert on issues
        if: failure()
        run: python scripts/send_alerts.py
```

---

## Best Practices

✅ Validate data before training
✅ Test models before deployment
✅ Use canary deployments
✅ Implement automated rollback
✅ Monitor all pipeline stages
✅ Version everything (data, code, models)
✅ Implement proper error handling
✅ Use environment-specific configs
✅ Test in staging first
✅ Automate documentation updates

---

**Complete ML Pipeline mastered!** 🎯

**Congratulations!** You've completed the CI/CD Basics module!
