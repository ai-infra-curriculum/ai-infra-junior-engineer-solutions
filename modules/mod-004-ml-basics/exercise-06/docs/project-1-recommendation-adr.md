# ADR: Product Recommendation System ML Infrastructure

**Date**: 2025-10-30
**Status**: Accepted
**Decision Makers**: ML Infrastructure Team
**Project**: TechShop Product Recommendations

---

## Context

### Business Requirements

TechShop, an e-commerce company with 10 million monthly active users, needs a product recommendation system to increase customer engagement and revenue.

**Success Metrics**:
- Increase click-through rate (CTR) by 15%
- Increase average order value (AOV) by 10%
- Serve recommendations with **<100ms latency** (P99)

**Scale Requirements**:
- 500,000 daily active users
- 100,000 products in catalog
- **1,000 recommendations requests/second** at peak
- 24/7 availability (99.9% uptime target)

**Data Available**:
- User browsing events (10M events/day)
- Purchase history (50K transactions/day)
- Product catalog with metadata
- User demographics

### Technical Constraints

1. **Latency**: Must serve recommendations in <100ms (P99)
2. **Throughput**: Handle 1,000 req/sec (86.4M requests/day)
3. **Freshness**: Recommendations should reflect recent behavior
4. **Cost**: Infrastructure budget of ~$10K/month
5. **Team**: 2 ML engineers, 1 platform engineer

### Problem Statement

Design an end-to-end ML infrastructure that:
1. Processes user events and generates recommendations
2. Serves recommendations with <100ms latency
3. Scales to handle peak traffic
4. Enables rapid experimentation and model iteration
5. Stays within cost budget

---

## Decision

### High-Level Architecture

We chose a **hybrid batch + real-time architecture** with the following components:

```
Data Layer:
  User Events → Kafka → Data Warehouse (S3/Parquet)
                  ↓
            Stream Processor (Flink)
                  ↓
           Feature Store (Feast)
              /        \
         Offline     Online
        (S3/BQ)   (Redis Cache)

Training Layer:
  Offline Features → Model Training (Daily)
                           ↓
                    Model Registry (MLflow)
                           ↓
                   Validation Pipeline
                           ↓
                   Versioned Model Store (S3)

Serving Layer:
  [API Gateway] → [Recommendation Service] → [Model Inference]
                         ↓                          ↓
                   [Redis Cache]            [Feature Service]
                         ↓                          ↓
                  Precomputed              Real-time Features
                 Recommendations           (User Context)
```

### Key Architectural Decisions

#### 1. Model Architecture: Two-Tower Neural Network with Hybrid Filtering

**Choice**: Two-tower model combining collaborative filtering with content-based features

**Rationale**:
- **Scalability**: Separate user and item embeddings allow pre-computation
- **Cold-start handling**: Content features help with new users/items
- **Performance**: Inner product similarity is fast (sub-millisecond)
- **Flexibility**: Can incorporate rich features (text, images, metadata)

**Alternatives Considered**:
| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Matrix Factorization** | Simple, fast | Limited expressiveness, cold-start issues | ❌ Too basic |
| **Deep Learning (NCF)** | Powerful, SOTA accuracy | Slow inference, requires GPU | ❌ Latency concerns |
| **Two-Tower** | Balanced accuracy + speed | More complex than MF | ✅ **Chosen** |
| **Transformer-based** | Best accuracy | Prohibitively slow, expensive | ❌ Over-engineered |

**Implementation Details**:
- User tower: 256-dim embedding from demographics + behavior
- Item tower: 256-dim embedding from product features
- Similarity: Cosine similarity on embeddings
- Top-K retrieval: Approximate Nearest Neighbors (FAISS)

#### 2. Inference Strategy: Batch Pre-computation + Real-time Scoring

**Choice**: Hybrid approach - pre-compute candidates offline, apply real-time signals online

**Architecture**:
```
Offline (Daily Batch):
  1. Generate user embeddings for all users
  2. Generate item embeddings for all products
  3. For each user, pre-compute top 500 candidates
  4. Store in Redis (key: user_id, value: [product_ids])

Online (Request Time):
  1. Retrieve pre-computed candidates from Redis (~5ms)
  2. Apply real-time signals (recent clicks, cart items) (~10ms)
  3. Re-rank top 500 → top 100 using lightweight model (~20ms)
  4. Diversify and apply business rules (~5ms)
  5. Return top 20 recommendations (~5ms)

Total latency: ~45ms (well under 100ms budget)
```

**Rationale**:
- **Latency**: Pre-computation keeps online inference fast
- **Freshness**: Real-time signals provide personalization
- **Cost**: Batch processing is 60% cheaper than real-time
- **Flexibility**: Can update candidates daily without retraining

**Alternatives Considered**:
| Approach | Latency | Cost | Freshness | Decision |
|----------|---------|------|-----------|----------|
| **Pure Real-time** | 150-300ms | $25K/mo | Excellent | ❌ Latency violation |
| **Pure Batch** | <10ms | $4K/mo | Poor (daily) | ❌ Stale recommendations |
| **Hybrid** | <50ms | $8.5K/mo | Good (hourly) | ✅ **Chosen** |

#### 3. Feature Store: Feast with Redis + S3

**Choice**: Feast for feature management with Redis online store and S3 offline store

**Rationale**:
- **Dual serving**: Same features for training and serving (prevents skew)
- **Low latency**: Redis provides sub-10ms feature lookup
- **Cost-effective**: S3 for historical features ($0.02/GB/month)
- **Open source**: No vendor lock-in, active community

**Feature Categories**:
```python
user_features = {
    "precomputed": [
        "user_embedding_256d",          # From trained model
        "total_purchases_30d",          # Aggregated offline
        "avg_order_value",              # Aggregated offline
        "favorite_categories",          # Top-3 categories
        "browsing_velocity"             # Events per day
    ],
    "realtime": [
        "current_session_category",     # From current browsing
        "cart_items",                   # Active cart contents
        "last_click_timestamp",         # Recency signal
    ]
}

item_features = {
    "precomputed": [
        "item_embedding_256d",          # From trained model
        "category",                     # Product category
        "price_bucket",                 # Binned price
        "popularity_score",             # Aggregated clicks
        "avg_rating"                    # User ratings
    ],
    "realtime": [
        "current_inventory",            # Stock level
        "trending_score_1h",            # Recent popularity
    ]
}
```

**Storage Strategy**:
- **Online (Redis)**:
  - User features: TTL 1 hour, lazy update
  - Item features: TTL 6 hours, proactive refresh
  - Total size: ~50GB (500K users × 100KB/user)
  - Cost: ~$500/month (Redis ElastiCache)

- **Offline (S3)**:
  - Historical features in Parquet format
  - Partitioned by date for efficient training
  - Total size: ~2TB (1 year of history)
  - Cost: ~$40/month

#### 4. Training Infrastructure: Kubernetes on AWS EKS

**Choice**: Kubernetes-based training pipeline with spot instances

**Rationale**:
- **Cost optimization**: Spot instances reduce training cost by 60%
- **Flexibility**: Can scale up/down based on workload
- **Portability**: Not locked into specific cloud vendor
- **Tooling**: Integrates with MLflow, Feast, Airflow

**Training Pipeline**:
```yaml
schedule: Daily at 2 AM UTC
trigger: Airflow DAG

steps:
  1. feature_extraction:
      input: Last 90 days of events
      output: Feature dataset (Parquet)
      compute: 4x r5.2xlarge (spot)
      duration: 60 minutes
      cost: $8/run

  2. model_training:
      input: Feature dataset
      output: Model checkpoint
      compute: 2x p3.2xlarge (spot) with GPUs
      duration: 120 minutes
      cost: $25/run

  3. model_validation:
      metrics: AUC, NDCG@20, CTR simulation
      thresholds: AUC > 0.80, NDCG > 0.65
      duration: 15 minutes
      cost: $2/run

  4. model_deployment:
      if: Metrics pass thresholds
      action: Update model in serving
      strategy: Canary (5% → 25% → 100%)

total_cost_per_run: $35
monthly_cost: $1,050 (30 runs)
```

**Model Versioning**:
- Store in S3 with semantic versioning (v1.2.3)
- Track experiments in MLflow (hyperparameters, metrics)
- Keep last 10 versions for rollback
- Archive old models to Glacier after 90 days

#### 5. Deployment Platform: Kubernetes with Horizontal Pod Autoscaling

**Choice**: EKS (Kubernetes on AWS) with HPA and Redis caching

**Rationale**:
- **Auto-scaling**: Handle traffic spikes automatically
- **High availability**: Multi-AZ deployment
- **Cost efficiency**: Scale to zero during low traffic
- **Developer experience**: Familiar tooling (kubectl, Helm)

**Serving Infrastructure**:
```yaml
service: recommendation-api
replicas:
  min: 10
  max: 50
  target_cpu: 70%

instance_type: c5.xlarge (4 vCPU, 8GB RAM)

routing:
  load_balancer: ALB with SSL termination
  path: /api/v1/recommendations
  method: GET
  params: user_id, context (optional)

caching_layer:
  tool: Redis Cluster (3 nodes)
  size: r5.large (2 vCPU, 13GB RAM per node)
  strategy:
    - Cache precomputed candidates (TTL 1 hour)
    - Cache popular user requests (TTL 10 minutes)
    - LRU eviction policy
  cache_hit_rate: 85% (monitored)

performance:
  p50_latency: 25ms
  p95_latency: 60ms
  p99_latency: 90ms
  throughput: 2,500 req/sec (with caching)
```

---

## Rationale

### Why This Architecture?

#### 1. Meets Latency Requirements

**Target**: <100ms (P99)
**Achieved**: ~90ms (P99)

**Latency Breakdown**:
```
Component               Time (ms)
---------------------------------
API Gateway             5
Redis cache lookup      8
Feature enrichment      12
Model inference         20
Re-ranking              15
Post-processing         10
Response serialization  5
Network overhead        15
---------------------------------
Total P99               90ms
```

**Key Optimizations**:
- Pre-computed candidates eliminate expensive ANN search at request time
- Redis caching reduces feature lookup from 50ms to 8ms
- Lightweight re-ranking model (small neural net) instead of full model
- Batch processing for offline computation

#### 2. Cost-Effective

**Monthly Infrastructure Cost**: ~$8,500

**Breakdown**:
```yaml
training:
  model_training: $1,050/month (30 runs × $35/run)
  experiment_tracking_mlflow: $200/month (t3.medium)

serving:
  kubernetes_cluster_eks: $150/month (control plane)
  compute_c5xlarge_10_instances: $3,600/month ($0.17/hr × 10 × 730h)
  autoscaling_buffer: $1,200/month (peak traffic)
  redis_cluster: $1,200/month (3 × r5.large)

storage:
  s3_features_models: $50/month (2TB)
  backups_logs: $100/month

monitoring:
  cloudwatch_datadog: $400/month

networking:
  data_transfer: $350/month
  load_balancer: $200/month

total: $8,500/month
```

**Cost per Request**: $0.003 (well under industry benchmark of $0.01)

**Cost Optimization Strategies**:
- Spot instances for training (60% savings)
- Auto-scaling during off-peak hours
- Efficient caching reduces compute needs
- Batch processing cheaper than real-time

#### 3. Enables Rapid Iteration

**Experiment Turnaround**:
- Feature engineering → Training → Evaluation: 3-4 hours
- A/B test setup: 30 minutes
- Model deployment: 15 minutes (canary) → 2 hours (full rollout)

**MLOps Capabilities**:
- Feature store enables feature reuse across models
- MLflow tracks all experiments
- Automated validation pipeline prevents bad deployments
- Canary deployment reduces risk

#### 4. Scales to Future Growth

**Current**: 1,000 req/sec
**Capacity**: 2,500 req/sec (2.5x headroom)
**Max**: 5,000 req/sec (with auto-scaling)

**Scaling Strategies**:
- Horizontal pod autoscaling (10 → 50 instances)
- Redis cluster can scale to 20 nodes if needed
- S3 handles unlimited storage
- Batch pipeline can process 10x more data

---

## Consequences

### Positive Consequences

#### ✅ Technical Benefits

1. **Low Latency**: P99 < 100ms achieved consistently
2. **High Throughput**: Can handle 2,500 req/sec (2.5x requirement)
3. **Fault Tolerant**: Multi-AZ deployment, circuit breakers, fallbacks
4. **Observable**: Comprehensive metrics, dashboards, alerts
5. **Maintainable**: Clear separation of concerns, standard tools

#### ✅ Business Benefits

1. **Cost-Effective**: $8.5K/month vs $25K for pure real-time
2. **Fast Iteration**: Ship new models in hours, not days
3. **Scalable**: Can grow to 10x traffic without architecture changes
4. **Reliable**: 99.9% uptime SLA achievable

#### ✅ Team Benefits

1. **Familiar Tools**: Kubernetes, Python, standard ML stack
2. **Good Documentation**: Architecture decision records, runbooks
3. **Automated Pipelines**: Less manual work, fewer errors
4. **Monitoring**: Easy to debug issues

### Negative Consequences

#### ⚠️ Technical Trade-offs

1. **Complexity**: More moving parts than simpler batch-only approach
   - **Mitigation**: Comprehensive documentation, runbooks
   - **Impact**: Manageable with 3-person team

2. **Training-Serving Skew Risk**: Features must match between training and serving
   - **Mitigation**: Feature store ensures consistency
   - **Impact**: Low risk with proper testing

3. **Cold Start**: New users get generic recommendations initially
   - **Mitigation**: Content-based features help, accept as limitation
   - **Impact**: Affects <5% of users (new signups)

4. **Staleness**: Pre-computed recommendations updated daily
   - **Mitigation**: Real-time signals provide some freshness
   - **Impact**: Acceptable trade-off for latency and cost

#### ⚠️ Operational Challenges

1. **Multi-Component System**: Requires monitoring multiple services
   - **Mitigation**: Unified observability (DataDog/CloudWatch)
   - **Impact**: On-call engineers need broad knowledge

2. **Kubernetes Expertise**: Team needs K8s skills
   - **Mitigation**: Training, managed EKS reduces burden
   - **Impact**: Learning curve for 1-2 months

3. **Cache Invalidation**: Must coordinate batch updates with cache
   - **Mitigation**: TTL-based expiration, automated invalidation
   - **Impact**: Occasional stale data (< 1 hour old)

### Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Model degrades over time** | Medium | High | Daily retraining, performance monitoring |
| **Redis cache failure** | Low | Medium | Multi-node cluster, fallback to database |
| **Batch pipeline failure** | Low | Medium | Retry logic, alerting, previous day's data |
| **Traffic spike (3x)** | Medium | Medium | Auto-scaling, load testing, rate limiting |
| **Training-serving skew** | Low | High | Feature store, integration tests |

---

## Alternatives Not Chosen

### Alternative 1: Pure Real-Time Inference

**Approach**: Compute recommendations on every request

**Pros**:
- Maximum freshness
- Simpler architecture (no batch layer)
- Easier to reason about

**Cons**:
- **Latency**: 200-300ms (violates requirement)
- **Cost**: $25K/month (3x over budget)
- **Complexity**: Need GPU instances for inference

**Why Not Chosen**: Violates latency requirement and exceeds budget

### Alternative 2: Pure Batch (Daily Pre-computation)

**Approach**: Pre-compute all recommendations once per day

**Pros**:
- Very low latency (<10ms)
- Lowest cost ($4K/month)
- Simplest architecture

**Cons**:
- **Stale**: Recommendations 24 hours old
- **No personalization**: Can't incorporate real-time signals
- **Poor user experience**: Doesn't adapt to session behavior

**Why Not Chosen**: Insufficient personalization hurts business metrics

### Alternative 3: Content-Based Filtering Only

**Approach**: Recommend based on product similarity only

**Pros**:
- No cold-start problem
- Explainable recommendations
- Simple implementation

**Cons**:
- **Lower accuracy**: Ignores collaborative signals
- **Echo chamber**: Only recommends similar items
- **Misses hidden patterns**: Can't discover surprising relevance

**Why Not Chosen**: Lower business impact (CTR, AOV gains)

### Alternative 4: Third-Party SaaS (e.g., Algolia Recommend)

**Approach**: Use managed recommendation service

**Pros**:
- Fastest time to market
- No infrastructure management
- Expert support

**Cons**:
- **Cost**: $15K/month (1.8x more expensive)
- **Less control**: Can't customize algorithms
- **Vendor lock-in**: Hard to migrate away
- **Data privacy**: Sensitive data leaves our control

**Why Not Chosen**: Cost and control concerns

---

## Implementation Roadmap

### Phase 1: MVP (Weeks 1-4)

**Goal**: Basic recommendations working

```yaml
week_1:
  - Set up Kubernetes cluster (EKS)
  - Deploy MLflow for experiment tracking
  - Set up data pipeline (Kafka → S3)

week_2:
  - Implement matrix factorization baseline
  - Create offline evaluation framework
  - Set up feature store (Feast)

week_3:
  - Train two-tower model
  - Deploy model serving API
  - Implement Redis caching

week_4:
  - Load testing and optimization
  - Deploy to staging
  - Run A/B test (5% traffic)
```

**Deliverables**:
- Basic recommendations live for 5% of users
- A/B test showing 5%+ CTR improvement
- Monitoring dashboards operational

### Phase 2: Optimization (Weeks 5-8)

**Goal**: Hit latency and accuracy targets

```yaml
week_5:
  - Optimize model architecture
  - Implement batch pre-computation
  - Add real-time signals

week_6:
  - Latency optimization (target P99 < 100ms)
  - Increase A/B test to 25%
  - Add business rule layer

week_7:
  - Implement re-ranking model
  - Add diversity and freshness filters
  - Expand to 50% traffic

week_8:
  - Full rollout (100% traffic)
  - Performance tuning
  - Documentation and runbooks
```

**Deliverables**:
- 100% traffic on new system
- CTR +15%, AOV +10% (business goals achieved)
- P99 latency < 100ms consistently

### Phase 3: Advanced Features (Weeks 9-12)

**Goal**: Continuous improvement

```yaml
week_9:
  - Implement multi-armed bandit for exploration
  - Add contextual features (time of day, device)

week_10:
  - Build experimentation platform
  - Implement online learning pipeline

week_11:
  - Add cold-start improvements
  - Implement cross-sell/upsell logic

week_12:
  - Cost optimization review
  - Team training and knowledge transfer
```

---

## Monitoring and Success Metrics

### Business Metrics (Primary)

```yaml
objective: Increase engagement and revenue

metrics:
  click_through_rate:
    target: +15% improvement
    baseline: 2.5%
    goal: 2.875%
    measurement: Daily A/B test analysis

  average_order_value:
    target: +10% improvement
    baseline: $65
    goal: $71.50
    measurement: 7-day rolling average

  conversion_rate:
    target: Maintain or improve (>= 0%)
    baseline: 3.2%
    threshold: Must not drop below 3.0%
```

### Technical Metrics (Guardrails)

```yaml
latency:
  p50: < 30ms
  p95: < 60ms
  p99: < 100ms
  alert_threshold: p99 > 120ms

throughput:
  current: 600 req/sec average
  peak: 1,200 req/sec
  capacity: 2,500 req/sec
  alert_threshold: > 2,000 req/sec sustained

availability:
  target: 99.9% (43 minutes downtime/month)
  measurement: Uptime checks every 1 minute
  alert_threshold: 3 consecutive failures

error_rate:
  target: < 0.1%
  measurement: Failed requests / total requests
  alert_threshold: > 1% for 5 minutes

cache_hit_rate:
  target: > 80%
  measurement: Redis cache hits / total lookups
  alert_threshold: < 70% for 10 minutes
```

### Model Quality Metrics

```yaml
offline_metrics:
  auc: > 0.80 (area under ROC curve)
  ndcg_at_20: > 0.65 (ranking quality)
  coverage: > 70% of catalog recommended

online_metrics:
  ctr: Click-through rate on recommendations
  conversion: Purchases from recommendations
  diversity: Unique items recommended (Shannon entropy)
  freshness: Avg age of recommended items
```

---

## Appendix

### Technology Stack Summary

```yaml
languages:
  primary: Python 3.11
  infrastructure: YAML, HCL (Terraform)

ml_frameworks:
  training: PyTorch 2.0
  serving: TorchServe
  feature_engineering: Pandas, PySpark

infrastructure:
  orchestration: Kubernetes (EKS)
  compute: AWS EC2 (c5.xlarge, p3.2xlarge)
  storage: S3, EBS
  caching: Redis (ElastiCache)
  database: PostgreSQL (RDS)

data_pipeline:
  streaming: Apache Kafka
  processing: Apache Flink
  batch: Apache Spark
  orchestration: Apache Airflow

ml_ops:
  experiment_tracking: MLflow
  feature_store: Feast
  model_registry: MLflow + S3
  monitoring: Prometheus, Grafana, DataDog

ci_cd:
  version_control: Git, GitHub
  ci: GitHub Actions
  cd: ArgoCD (GitOps)
  infrastructure: Terraform
```

### Team Responsibilities

```yaml
ml_engineer_1:
  - Model development and experimentation
  - Feature engineering
  - Model evaluation and A/B testing

ml_engineer_2:
  - Training pipeline development
  - Feature store maintenance
  - Model monitoring and debugging

platform_engineer:
  - Infrastructure provisioning (Terraform)
  - Kubernetes cluster management
  - CI/CD pipeline maintenance
  - Monitoring and alerting setup
```

### Decision Log

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2025-10-20 | Two-tower architecture | Balance accuracy and latency | ✅ Approved |
| 2025-10-21 | Hybrid batch+real-time | Meet latency requirements | ✅ Approved |
| 2025-10-22 | Feast for feature store | Prevent training-serving skew | ✅ Approved |
| 2025-10-23 | Kubernetes on EKS | Flexibility and auto-scaling | ✅ Approved |
| 2025-10-24 | Redis for caching | Sub-10ms lookups | ✅ Approved |

---

**Document Owner**: ML Infrastructure Team
**Last Updated**: 2025-10-30
**Next Review**: 2025-11-30 (Monthly review)
**Version**: 1.0
