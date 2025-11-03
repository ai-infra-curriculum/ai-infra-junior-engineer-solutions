# ADR: Real-Time Fraud Detection System

**Date**: 2025-10-30
**Status**: Accepted
**Decision Makers**: ML Infrastructure + Security Team
**Project**: TechShop Fraud Detection

---

## Context

### Business Requirements

TechShop processes $50M in transactions monthly and faces increasing fraud attempts. The finance team requires automated fraud detection to minimize losses while maintaining excellent user experience.

**Success Metrics**:
- **Recall**: Catch 95%+ of fraudulent transactions
- **Precision**: Keep false positives below 1% (minimize declined legitimate transactions)
- **Latency**: Make decision within **300ms** of transaction

**Scale**:
- 10,000 transactions/second at peak
- 0.1% fraud rate (10 fraudulent per 10,000 transactions)
- $50 average transaction value
- $200 average fraud loss per fraudulent transaction
- 24/7 critical service (99.99% uptime required)

**Business Impact**:
```
Current State (No ML):
- Manual review of flagged transactions
- 50% recall (half of fraud goes undetected)
- $3M annual fraud loss

Target State (With ML):
- Automated real-time detection
- 95% recall (catch 95% of fraud)
- Reduce fraud loss to $300K annually
- ROI: $2.7M savings - $180K ML infrastructure cost = $2.52M net benefit
```

### Critical Constraints

1. **False Positives Are Costly**
   - Declined legitimate transaction frustrates customers
   - Customer may leave and never return
   - Each false positive costs $50 in lost transaction + reputation damage

2. **Real-Time Requirement**
   - Must decide during transaction authorization
   - Cannot delay customer beyond 300ms
   - No opportunity for manual review before decision

3. **Extreme Class Imbalance**
   - 99.9% legitimate vs 0.1% fraudulent
   - Standard ML models fail with such imbalance
   - Cannot naively optimize accuracy (99.9% accuracy by predicting all legitimate!)

4. **Adversarial Environment**
   - Fraudsters actively adapt to detection
   - Model must be updated frequently
   - Need to detect novel fraud patterns

5. **Regulatory Requirements**
   - Must explain decisions (model interpretability)
   - Audit trail for all decisions
   - Comply with PCI-DSS, GDPR

---

## Decision

### High-Level Architecture

We chose a **multi-stage real-time detection system** with hybrid modeling:

```
Transaction Flow:

[Transaction] → [API Gateway] → [Stage 1: Rules Engine (Fast Path)]
                                        ↓
                                    50% auto-approve
                                        ↓
                                [Stage 2: ML Model (Moderate)]
                                        ↓
                                    45% auto-approve/decline
                                        ↓
                                [Stage 3: Deep Analysis (Slow Path)]
                                        ↓
                                    5% human review

Supporting Infrastructure:

[Feature Store]          [Model Serving]           [Fraud Analyst Feedback]
   (Redis)           (Ensemble: XGBoost + NN)           (Retraining Loop)
      ↓                        ↓                              ↓
[Streaming Pipeline]    [Multi-Model Inference]     [Online Learning]
 (Kafka + Flink)         (Load Balanced)            (Hourly Updates)
```

### Key Architectural Decisions

#### 1. Multi-Stage Detection Pipeline

**Choice**: Three-stage funnel with increasing complexity

**Stage 1: Fast Rules Engine** (< 10ms)
```python
# Handle obvious cases instantly
fast_rules = {
    "auto_approve": [
        "amount < $10",
        "user_txn_count_30d > 50",
        "merchant_verified == True",
        "same_card_same_merchant_last_24h == True"
    ],
    "auto_decline": [
        "amount > $5000 AND first_transaction",
        "user_on_blocklist",
        "merchant_on_blocklist",
        "ip_from_known_fraud_network"
    ]
}

# ~50% of transactions handled here
# Cost: $0.0001 per transaction
```

**Stage 2: ML Model** (< 200ms)
```python
# Ensemble model for nuanced decisions
ensemble = {
    "xgboost": weight=0.6,    # Handles tabular features well
    "neural_net": weight=0.4   # Captures complex patterns
}

# Score transactions that passed Stage 1
# If confidence > 0.95: auto-approve
# If confidence < 0.05: auto-decline
# Else: proceed to Stage 3

# ~45% of transactions handled here
# Cost: $0.005 per transaction
```

**Stage 3: Deep Analysis** (< 500ms, async)
```python
# For uncertain cases (5% of transactions)
deep_analysis = {
    "graph_analysis": check_transaction_network,
    "behavioral_anomaly": compare_to_user_profile,
    "merchant_risk": assess_merchant_history,
    "external_signals": query_fraud_bureaus
}

# Human review if still uncertain
# Cost: $0.02 per transaction + analyst time
```

**Rationale**:
- **Latency**: Fast path keeps 50% of transactions under 10ms
- **Cost**: Avoid expensive ML for obvious cases
- **Accuracy**: Focus ML compute on hard cases
- **Throughput**: Distribute load across stages

#### 2. Handling Extreme Class Imbalance

**Challenge**: 0.1% fraud rate means 999 legitimate for every 1 fraud

**Our Approach**: Multi-pronged strategy

**A. Class Weights in Loss Function**
```python
# Penalize false negatives heavily
class_weights = {
    0: 1.0,      # Legitimate (no penalty)
    1: 999.0     # Fraud (999x penalty)
}

# Use focal loss to focus on hard examples
loss = FocalLoss(alpha=0.25, gamma=2.0)
```

**B. SMOTE for Training Data**
```python
# Oversample minority class (fraud) synthetically
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(sampling_strategy=0.3).fit_resample(X, y)

# New ratio: 70% legitimate, 30% fraud (for training only)
```

**C. Ensemble with Diverse Models**
```python
ensemble = [
    XGBoost(scale_pos_weight=999),   # Handles class imbalance internally
    RandomForest(class_weight='balanced'),
    NeuralNet(focal_loss),
    IsolationForest()                 # Anomaly detection (unsupervised)
]
```

**D. Threshold Tuning**
```python
# Don't use 0.5 threshold - optimize for business metrics
optimal_threshold = optimize_threshold(
    metric="f1_score",
    weight_recall=0.7,    # Prioritize catching fraud
    weight_precision=0.3
)
# Result: threshold = 0.15 (lower threshold to catch more fraud)
```

**E. Anomaly Detection for Novel Fraud**
```python
# Detect new fraud patterns not seen in training
isolation_forest = IsolationForest(contamination=0.001)

if ensemble_score < threshold OR isolation_forest.predict(X) == -1:
    flag_as_potential_fraud()
```

#### 3. Real-Time Feature Engineering (< 50ms budget)

**Challenge**: Compute features in real-time without violating latency budget

**Feature Categories**:

**Precomputed Features (Redis lookup, ~5ms)**:
```python
user_features_precomputed = {
    "user_account_age_days": 547,
    "total_transactions_30d": 45,
    "total_amount_30d": 2340.50,
    "avg_transaction_amount": 52.01,
    "unique_merchants_30d": 18,
    "declined_transactions_30d": 0,
    "user_risk_score": 0.12,         # ML-derived score
    "favorite_merchant_categories": ["grocery", "gas"],
}

item_features_precomputed = {
    "merchant_age_days": 1825,
    "merchant_fraud_rate_30d": 0.08,
    "merchant_avg_transaction": 65.30,
    "merchant_category": "electronics",
    "merchant_risk_score": 0.25,
}

# Updated daily via batch pipeline
# Stored in Redis for fast lookup
```

**Real-Time Features (from transaction, ~10ms)**:
```python
transaction_features_realtime = {
    "transaction_amount": 125.50,
    "transaction_currency": "USD",
    "transaction_time_utc": "2025-10-30T14:23:45Z",
    "transaction_day_of_week": 3,        # Wednesday
    "transaction_hour": 14,
    "is_weekend": False,
    "is_night_transaction": False,       # 10 PM - 6 AM
}

device_features_realtime = {
    "device_ip": "192.168.1.1",
    "device_fingerprint": "abc123...",
    "user_agent": "Mozilla/5.0...",
    "is_vpn": False,
    "ip_country": "US",
    "ip_isp": "Comcast",
}
```

**Streaming Aggregation Features (Flink, ~20ms)**:
```python
# Computed from recent event stream
streaming_features = {
    "transactions_last_5min": 0,
    "transactions_last_1hour": 2,
    "amount_last_1hour": 95.30,
    "unique_merchants_last_1hour": 2,
    "different_locations_last_1hour": 1,
    "velocity_score": 0.3,               # Spending rate
}

# Maintained in-memory by Flink
# Queried via Feature Service API
```

**Feature Latency Budget**:
```yaml
total_feature_latency: 50ms

breakdown:
  redis_lookup: 8ms
  streaming_api_call: 12ms
  feature_computation: 15ms
  feature_encoding: 10ms
  buffer: 5ms
```

**Fallback Strategy**:
```python
# If feature service times out, use cached/default values
def get_features_with_fallback(transaction_id, timeout_ms=50):
    try:
        features = feature_service.get(transaction_id, timeout=timeout_ms)
    except TimeoutError:
        # Use stale features from cache
        features = redis.get(f"features:{transaction_id}:last_known")
        features["_fallback_used"] = True
        log_warning(f"Feature timeout for {transaction_id}")

    return features

# Fallback rate target: < 0.1%
```

#### 4. Model Serving Infrastructure

**Choice**: Kubernetes + TensorFlow Serving + Load Balancing

**Architecture**:
```yaml
model_serving:
  platform: Kubernetes (EKS)
  replicas: 20 (min) - 100 (max)
  instance_type: c5.2xlarge (8 vCPU, 16GB RAM)

  models:
    xgboost_model:
      framework: XGBoost
      version: v2.3.1
      latency_p99: 15ms
      replicas: 12

    neural_net_model:
      framework: TensorFlow Serving
      version: v1.5.0
      latency_p99: 40ms
      replicas: 8

    ensemble_aggregator:
      framework: Custom (Python + FastAPI)
      version: v1.2.0
      latency_p99: 8ms
      replicas: 20

  autoscaling:
    metric: request_queue_depth
    target_queue: 10 requests
    scale_up_threshold: 15 requests
    scale_down_threshold: 5 requests
    cooldown: 60 seconds

  circuit_breaker:
    failure_threshold: 5 errors in 30 seconds
    timeout: 200ms
    fallback: rules_engine_only
```

**Load Balancing Strategy**:
```python
# Weighted round-robin with health checks
load_balancer_config = {
    "algorithm": "weighted_round_robin",
    "weights": {
        "xgboost_model": 0.6,     # More weight to faster model
        "neural_net_model": 0.4
    },
    "health_check": {
        "interval": 5,             # seconds
        "timeout": 2,              # seconds
        "unhealthy_threshold": 2   # consecutive failures
    }
}

# If one model is down, route all traffic to healthy model
```

**Performance Optimization**:
```python
# Batching for efficiency
batching_config = {
    "max_batch_size": 32,
    "batch_timeout_ms": 5,         # Wait max 5ms for batch to fill
    "enable_dynamic_batching": True
}

# Model optimization
optimization = {
    "xgboost": "treelite_compilation",    # Compile to C for speed
    "neural_net": "tensorrt_fp16",        # Half precision, 2x speedup
    "quantization": "int8"                # 4x smaller model size
}
```

#### 5. Training and Online Learning

**Challenge**: Fraudsters adapt quickly; model must stay current

**Our Approach**: Hybrid offline + online learning

**Offline Retraining (Daily)**:
```yaml
daily_retrain:
  schedule: 2 AM UTC daily
  data: Last 90 days of transactions
  compute: 4x p3.2xlarge (GPU instances)
  duration: 120 minutes

  steps:
    1. data_extraction:
        source: Data warehouse
        filter: Labeled transactions (fraud analysts + automated)

    2. feature_engineering:
        apply: Same transformations as serving
        validation: Check feature distribution drift

    3. model_training:
        xgboost: 60 minutes
        neural_net: 45 minutes

    4. validation:
        holdout_test: Last 7 days
        metrics: Precision, recall, F1, AUC
        thresholds: Recall > 0.93, Precision > 0.98

    5. deployment:
        strategy: Canary (5% → 25% → 100%)
        rollback_trigger: Performance degradation > 2%
```

**Online Learning (Hourly)**:
```yaml
online_learning:
  trigger: Fraud analyst feedback
  frequency: Every hour
  method: Incremental model update

  process:
    1. collect_feedback:
        source: Fraud analyst reviews
        volume: ~50 new labels per hour

    2. update_model:
        method: Online gradient descent
        learning_rate: 0.001 (conservative)
        update_weight: 0.1 (blend with existing model)

    3. validate:
        test: On recent unlabeled data
        monitor: Prediction drift

    4. deploy:
        strategy: Shadow mode first (compare predictions)
        rollout: If metrics OK, deploy to 10% traffic
```

**Continuous Monitoring**:
```python
# Detect model degradation in real-time
model_monitoring = {
    "prediction_distribution": {
        "baseline": "Historical distribution",
        "alert_if": "KL divergence > 0.5"
    },
    "feature_drift": {
        "baseline": "Training data distribution",
        "alert_if": "Any feature drift > 3 std dev"
    },
    "performance_metrics": {
        "precision": "alert if < 0.98",
        "recall": "alert if < 0.93",
        "f1_score": "alert if < 0.95"
    }
}
```

---

## Rationale

### Why This Architecture?

#### 1. Meets Strict Latency Requirements

**Target**: <300ms (P99)
**Achieved**: ~250ms (P99) with 90th percentile at 180ms

**Latency Breakdown**:
```
Component                     Time (ms)
------------------------------------------
API Gateway                   10
Authentication/Authorization  15
Feature Lookup (Redis)        8
Streaming Features (Flink)    12
Feature Computation           15
Model Inference (Ensemble)    120
  ├─ XGBoost                  (15ms)
  ├─ Neural Net               (40ms)
  └─ Ensemble Aggregation     (8ms)
Business Rules Logic          20
Response Generation           10
Network Overhead              25
Buffer                        15
------------------------------------------
Total P99                     250ms
```

**How We Achieved This**:
1. **Multi-stage pipeline**: 50% of transactions decided in < 10ms (rules only)
2. **Aggressive caching**: Pre-computed features reduce lookup time
3. **Model optimization**: TensorRT, quantization, compiled XGBoost
4. **Parallel execution**: XGBoost and Neural Net run concurrently
5. **Async processing**: Non-critical analysis happens post-response

#### 2. Balances Precision and Recall

**Target**: 95% recall, <1% false positive rate
**Achieved**: 96.2% recall, 0.7% false positive rate

**How We Achieved This**:
```python
# Optimized decision threshold
threshold_optimization_results = {
    "threshold_0.5": {
        "recall": 0.88,
        "precision": 0.92,
        "f1": 0.90,
        "false_positive_rate": 0.15%,
        "business_cost": "$50K/month"
    },
    "threshold_0.15": {  # ✅ Chosen threshold
        "recall": 0.962,
        "precision": 0.988,
        "f1": 0.975,
        "false_positive_rate": 0.7%,
        "business_cost": "$35K/month"
    },
    "threshold_0.05": {
        "recall": 0.98,
        "precision": 0.82,
        "f1": 0.89,
        "false_positive_rate": 2.5%,
        "business_cost": "$125K/month"  # Too many false positives
    }
}

# Multi-stage approach further improves precision
stage_results = {
    "rules_engine": {
        "coverage": "50% of transactions",
        "precision": "99.8%",
        "recall": "N/A (not fraud detection)"
    },
    "ml_ensemble": {
        "coverage": "45% of transactions",
        "precision": "98.8%",
        "recall": "96.2%"
    },
    "human_review": {
        "coverage": "5% of transactions",
        "precision": "100%",
        "recall": "100%"
    }
}
```

#### 3. Cost-Effective at Scale

**Monthly Infrastructure Cost**: ~$15,000

**Breakdown**:
```yaml
training:
  daily_training_gpu: $2,400/month (80 hours × $1/hour × 30 days)
  online_learning_cpu: $300/month (continuous small updates)
  experiment_tracking: $150/month

serving:
  kubernetes_cluster: $200/month (EKS control plane)
  model_serving_c5_2xlarge: $6,000/month (20 instances × $0.40/hr × 730h)
  autoscaling_buffer: $2,000/month (peak traffic instances)
  feature_service: $1,500/month (API for streaming features)

streaming_pipeline:
  kafka_cluster: $800/month (3 brokers)
  flink_processing: $1,200/month (stream processing)

storage:
  transaction_logs_s3: $200/month
  model_artifacts: $50/month
  feature_store_redis: $1,200/month (large cache for features)

monitoring:
  datadog_cloudwatch: $500/month
  fraud_analyst_dashboard: $150/month

networking:
  data_transfer: $300/month
  load_balancer: $250/month

total: $15,000/month
```

**Cost per Transaction**: $0.0043
**ROI**:
```
Fraud Prevented: $2.7M/year (95% of $3M total fraud)
Infrastructure Cost: $180K/year
Net Benefit: $2.52M/year
ROI: 1400%
```

#### 4. Handles Extreme Scale

**Current**: 10,000 transactions/second
**Peak**: 15,000 transactions/second (Black Friday)
**Capacity**: 25,000 transactions/second (with auto-scaling)

**Scaling Strategy**:
```yaml
horizontal_scaling:
  model_serving: 20 → 100 instances (5x)
  feature_service: 10 → 50 instances (5x)
  kafka_brokers: 3 → 9 (3x)

vertical_scaling:
  redis_cluster: 3 nodes → 12 nodes (4x)
  database_connections: 100 → 500 (5x)

caching_optimization:
  feature_cache_hit_rate: 95%
  prediction_cache: Popular merchants/amounts
  ttl: 5 minutes for hot paths
```

---

## Consequences

### Positive Consequences

#### ✅ Business Impact

1. **Fraud Reduction**: From $3M to $300K annual loss (90% reduction)
2. **Improved Customer Experience**: <1% false positives vs 5% manual review
3. **Faster Decisions**: 300ms vs 24-hour manual review
4. **Scalable**: Can handle 10x growth without architecture changes

#### ✅ Technical Benefits

1. **Low Latency**: P99 <300ms achieved consistently
2. **High Accuracy**: 96% recall, 99% precision
3. **Fault Tolerant**: Multi-stage with fallbacks
4. **Observable**: Comprehensive dashboards and alerts
5. **Maintainable**: Clear separation of concerns

#### ✅ Operational Excellence

1. **24/7 Monitoring**: Real-time alerts for model degradation
2. **Online Learning**: Adapts to new fraud patterns hourly
3. **Explainable Decisions**: SHAP values for analyst review
4. **Audit Trail**: Every decision logged for compliance

### Negative Consequences

#### ⚠️ Complexity

1. **Multi-Component System**: Rules + ML + Streaming + Feature Store
   - **Mitigation**: Comprehensive documentation, runbooks
   - **Impact**: Requires 3-4 person team

2. **Real-Time Requirements**: Strict latency SLAs
   - **Mitigation**: Redundancy, fallbacks, extensive testing
   - **Impact**: Higher operational burden

3. **Adversarial Environment**: Fraudsters constantly adapt
   - **Mitigation**: Online learning, frequent retraining
   - **Impact**: Ongoing model maintenance required

#### ⚠️ Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Model Degradation** | Medium | High | Continuous monitoring, auto-rollback |
| **Feature Service Outage** | Low | High | Fallback to cached features |
| **False Positive Spike** | Low | Medium | Shadow mode testing, gradual rollout |
| **Novel Fraud Pattern** | High | Medium | Anomaly detection, human review |
| **Latency Violation** | Low | High | Auto-scaling, circuit breakers |

---

## Implementation Roadmap

### Phase 1: MVP (Weeks 1-6)

**Goal**: Basic fraud detection working

```yaml
weeks_1-2:
  - Set up data pipeline (Kafka, Flink)
  - Deploy Feature Store (Redis)
  - Implement rules engine

weeks_3-4:
  - Train baseline XGBoost model
  - Deploy model serving infrastructure
  - Implement A/B testing framework

weeks_5-6:
  - Shadow mode (log predictions, don't block)
  - Validate predictions against manual review
  - Deploy to 10% of traffic
```

### Phase 2: Production (Weeks 7-12)

**Goal**: Full production deployment

```yaml
weeks_7-8:
  - Add neural network to ensemble
  - Optimize thresholds based on business metrics
  - Expand to 50% traffic

weeks_9-10:
  - Implement online learning pipeline
  - Build fraud analyst feedback loop
  - 100% traffic rollout

weeks_11-12:
  - Performance optimization
  - Cost optimization review
  - Documentation and training
```

---

## Monitoring and Success Metrics

### Model Performance (Primary)

```yaml
fraud_detection_metrics:
  recall:
    target: 95%
    current: 96.2%
    alert_threshold: < 93%

  precision:
    target: 98%
    current: 98.8%
    alert_threshold: < 97%

  f1_score:
    target: 96.5%
    current: 97.5%
    alert_threshold: < 95%

  false_positive_rate:
    target: < 1%
    current: 0.7%
    alert_threshold: > 1.5%
```

### Business Metrics

```yaml
business_impact:
  fraud_prevented_monthly:
    baseline: $0 (no ML)
    current: $225K
    target: $250K

  false_positive_cost_monthly:
    baseline: $150K (manual review)
    current: $35K (0.7% FP rate)
    target: < $50K

  customer_experience:
    avg_decision_time: 250ms
    target: < 300ms
    manual_review_rate: 5%
    target: < 10%
```

### System Performance

```yaml
technical_metrics:
  latency:
    p50: 120ms
    p95: 200ms
    p99: 250ms
    alert: p99 > 300ms

  throughput:
    average: 6,000 req/sec
    peak: 15,000 req/sec
    capacity: 25,000 req/sec

  availability:
    target: 99.99% (52 minutes/year)
    current: 99.97%

  error_rate:
    target: < 0.01%
    current: 0.005%
```

---

**Document Owner**: ML Infrastructure + Security Team
**Last Updated**: 2025-10-30
**Next Review**: 2025-11-15 (Bi-weekly review due to adversarial environment)
**Version**: 1.0
