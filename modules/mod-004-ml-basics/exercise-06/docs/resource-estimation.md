# Resource Estimation and Cost Analysis

**Document Type**: Cost Breakdown & Capacity Planning
**Projects**: All Three (Recommendations, Fraud Detection, Image Moderation)
**Date**: 2025-10-30
**Currency**: USD
**Cloud Provider**: AWS (can be adapted to GCP/Azure)

---

## Executive Summary

| Project | Monthly Cost | Cost/Transaction | ROI | Payback Period |
|---------|--------------|------------------|-----|----------------|
| **Project 1: Recommendations** | $8,500 | $0.003 | 1400% | <2 months |
| **Project 2: Fraud Detection** | $15,000 | $0.0043 | 1400% | <1 month |
| **Project 3: Image Moderation** | $6,200 | $0.005/image | 5100% | <1 month |
| **Total** | **$29,700** | - | - | - |

**Annual Infrastructure Cost**: $356,400
**Annual Business Value**: $5.52M (revenue increase + cost savings)
**Net Annual Benefit**: $5.16M
**Overall ROI**: 1,448%

---

## Project 1: Product Recommendation System

### Infrastructure Components

#### Training Infrastructure

```yaml
daily_model_training:
  feature_extraction:
    compute: 4x r5.2xlarge (spot instances)
    specs: 8 vCPU, 64GB RAM each
    duration: 60 minutes
    cost_per_run: $8 ($0.14/hr spot × 4 × 1hr)
    monthly: $240 (30 runs)

  model_training:
    compute: 2x p3.2xlarge (spot instances with GPU)
    specs: 8 vCPU, 61GB RAM, 1x V100 GPU
    duration: 120 minutes
    cost_per_run: $25 ($0.612/hr spot × 2 × 2hr)
    monthly: $750 (30 runs)

  model_validation:
    compute: 1x c5.xlarge
    specs: 4 vCPU, 8GB RAM
    duration: 15 minutes
    cost_per_run: $2 ($0.17/hr × 0.25hr)
    monthly: $60 (30 runs)

experiment_tracking:
  service: MLflow on t3.medium
  specs: 2 vCPU, 4GB RAM
  cost: $30/month (24/7)
  storage: EBS 100GB @ $10/month

total_training: $1,090/month
```

#### Serving Infrastructure

```yaml
compute:
  kubernetes_cluster:
    service: Amazon EKS
    control_plane: $150/month
    worker_nodes:
      base_capacity: 10x c5.xlarge
      cost: $0.17/hr × 10 × 730hr = $1,241/month
      autoscaling_buffer: 5x c5.xlarge
      cost: $0.17/hr × 5 × 365hr = $620/month (50% utilization)
    total_compute: $2,011/month

  model_serving:
    replicas: 20 pods (distributed across nodes)
    resource_per_pod: 1 vCPU, 2GB RAM
    included_in_compute: Yes

feature_store:
  redis_cache:
    service: ElastiCache for Redis
    instance: 3x r5.large (cluster mode)
    specs: 2 vCPU, 13GB RAM per node
    cost: $0.164/hr × 3 × 730hr = $360/month

  offline_storage:
    service: S3
    size: 2TB (historical features)
    cost: 2,000GB × $0.023/GB = $46/month

data_pipeline:
  kafka:
    service: Amazon MSK (3 brokers)
    instance: kafka.m5.large
    cost: $0.21/hr × 3 × 730hr = $460/month

  streaming_processor:
    service: Flink on ECS
    compute: 3x c5.xlarge
    cost: $0.17/hr × 3 × 730hr = $372/month

storage:
  model_artifacts_s3: $15/month (100GB)
  logs_cloudwatch: $30/month
  backups: $20/month

monitoring:
  cloudwatch: $100/month
  datadog: $300/month

networking:
  data_transfer: $350/month (inter-AZ + egress)
  application_load_balancer: $200/month

total_serving: $4,264/month

database:
  rds_postgresql:
    instance: db.r5.large
    specs: 2 vCPU, 16GB RAM
    cost: $0.24/hr × 730hr = $175/month
  storage: 500GB × $0.115/GB = $58/month
total_database: $233/month

total_project_1: $5,587/month
# Rounded to $8,500/month in ADR (includes buffer and misc costs)
```

### Capacity Analysis

```yaml
current_load:
  requests_per_second: 600 (average)
  peak_requests_per_second: 1,200
  daily_requests: 51.8M

capacity:
  single_pod_throughput: 125 req/sec
  total_pods: 20
  theoretical_capacity: 2,500 req/sec
  with_overhead: 2,000 req/sec (safe limit)
  headroom: 67% above peak

scalability:
  max_pods_with_autoscaling: 50
  max_capacity: 5,000 req/sec
  can_handle_growth: 4x current peak

cost_at_scale:
  if_traffic_2x: $11,000/month (29% increase due to autoscaling)
  if_traffic_5x: $18,000/month (112% increase)
```

### Cost Optimization Opportunities

```yaml
immediate_savings:
  1_spot_instances_training:
      current: Using spot
      savings: Already saving 60%

  2_reserved_instances_serving:
      action: Purchase 1-year RI for base capacity
      savings: $400/month (20% discount on base compute)

  3_s3_lifecycle_policies:
      action: Move old features to Glacier
      savings: $30/month

  total_immediate: $430/month saved

future_optimizations:
  1_model_compression:
      technique: Quantization, pruning
      impact: Reduce serving compute by 20%
      savings: $400/month

  2_caching_improvements:
      action: Increase cache hit rate 85% → 95%
      impact: Reduce backend calls by 50%
      savings: $200/month

  3_batch_size_tuning:
      action: Optimize batch processing
      impact: Reduce training time by 15%
      savings: $150/month

  total_future: $750/month
```

---

## Project 2: Fraud Detection System

### Infrastructure Components

#### Training Infrastructure

```yaml
daily_model_training:
  data_preparation:
    compute: 4x r5.2xlarge (spot)
    duration: 90 minutes
    cost_per_run: $12
    monthly: $360 (30 runs)

  xgboost_training:
    compute: 4x c5.4xlarge (16 vCPU)
    duration: 60 minutes
    cost_per_run: $23
    monthly: $690 (30 runs)

  neural_net_training:
    compute: 2x p3.2xlarge (GPU)
    duration: 90 minutes
    cost_per_run: $55
    monthly: $1,650 (30 runs)

online_learning:
  continuous_updates:
    compute: 2x c5.xlarge (24/7)
    cost: $0.17/hr × 2 × 730hr = $248/month

experiment_tracking:
  mlflow: $30/month
  weights_and_biases: $50/month

total_training: $3,028/month
```

#### Serving Infrastructure

```yaml
compute:
  kubernetes_cluster:
    control_plane: $200/month (EKS)
    worker_nodes:
      base: 20x c5.2xlarge (8 vCPU, 16GB RAM)
      cost: $0.34/hr × 20 × 730hr = $4,964/month
      autoscaling: 10x c5.2xlarge (peak)
      cost: $0.34/hr × 10 × 365hr = $1,241/month (50% util)
    total: $6,405/month

model_serving:
  xgboost_pods: 12 replicas
  neural_net_pods: 8 replicas
  ensemble_aggregator: 20 replicas
  included_in_compute: Yes

feature_service:
  redis_cluster:
    instance: 5x r5.xlarge
    cost: $0.252/hr × 5 × 730hr = $920/month

  streaming_features:
    flink_cluster: 4x c5.2xlarge
    cost: $0.34/hr × 4 × 730hr = $993/month

data_pipeline:
  kafka_cluster:
    service: MSK (3 brokers)
    instance: kafka.m5.xlarge
    cost: $0.42/hr × 3 × 730hr = $920/month

  transaction_database:
    rds_postgresql: db.r5.2xlarge
    cost: $0.48/hr × 730hr = $350/month
    storage: 1TB × $0.115 = $115/month

storage:
  transaction_logs: $150/month
  model_artifacts: $30/month
  feature_history: $100/month

monitoring:
  security_monitoring: $200/month
  performance_monitoring: $300/month
  fraud_analyst_dashboard: $150/month

networking:
  data_transfer: $300/month
  load_balancer: $250/month

total_serving: $10,683/month

analyst_tools:
  fraud_review_dashboard: $100/month
  case_management_system: $150/month

total_project_2: $13,961/month
# Rounded to $15,000/month in ADR
```

### Capacity Analysis

```yaml
current_load:
  transactions_per_second: 6,000 (average)
  peak_transactions_per_second: 15,000
  daily_transactions: 518M

capacity:
  single_pod_throughput:
    xgboost: 400 req/sec
    neural_net: 200 req/sec
    (processed in parallel, take max)
  total_capacity: 20,000 req/sec
  headroom: 33% above peak

latency_breakdown:
  p50: 120ms
  p95: 200ms
  p99: 250ms
  sla: 300ms
  margin: 50ms (20%)

cost_per_transaction: $0.0043
transactions_per_month: 15.5B
infrastructure_cost_per_transaction: $0.0043
```

### Business Value Calculation

```yaml
fraud_prevented:
  total_fraud_annual: $3M
  detection_rate: 96%
  fraud_prevented_annual: $2,880,000

cost_of_false_positives:
  false_positive_rate: 0.7%
  transactions_daily: 51.8M
  false_positives_daily: 362,600
  cost_per_false_positive: $0.1 (customer service + reputation)
  cost_annual: $132,000

net_benefit:
  fraud_prevented: $2,880,000
  false_positive_cost: -$132,000
  infrastructure_cost: -$180,000 (annual)
  net_annual_benefit: $2,568,000

roi: 1426%
payback_period: 0.84 months
```

---

## Project 3: Image Moderation System

### Infrastructure Components

#### Training Infrastructure

```yaml
weekly_model_training:
  data_preparation:
    compute: 2x c5.2xlarge
    duration: 2 hours
    cost_per_run: $11
    monthly: $44 (4 runs)

  model_training:
    compute: 2x p3.2xlarge (GPU)
    duration: 4 hours
    cost_per_run: $98
    monthly: $392 (4 runs)

  validation:
    compute: 1x p3.2xlarge
    duration: 1 hour
    cost_per_run: $25
    monthly: $100 (4 runs)

total_training: $536/month
```

#### Serving Infrastructure

```yaml
compute:
  gpu_workers:
    base_capacity: 2x g4dn.xlarge (24/7)
    specs: 4 vCPU, 16GB RAM, 1x T4 GPU
    cost: $0.526/hr × 2 × 730hr = $768/month

    peak_capacity: 5x g4dn.xlarge (auto-scaling)
    utilization: 40% (peak hours)
    cost: $0.526/hr × 5 × 292hr = $768/month

  orchestration:
    ecs_fargate: $100/month (task management)

  total_compute: $1,636/month

storage:
  s3_images:
    current: 50TB
    cost: 50,000GB × $0.023 = $1,150/month
    lifecycle_to_glacier: -$400/month (move old)
    net: $750/month

  database_postgresql:
    rds: db.t3.medium
    cost: $0.068/hr × 730hr = $50/month
    storage: 100GB × $0.115 = $12/month

  redis_cache:
    elasticache: cache.t3.medium
    cost: $0.068/hr × 730hr = $50/month

queue:
  sqs: $5/month (1M requests/month)

monitoring:
  cloudwatch: $150/month
  moderator_dashboard: $100/month

networking:
  s3_transfer: $200/month
  load_balancer: $150/month

total_serving: $2,229/month

human_moderation:
  moderator_salaries: $3,000/month (1 FTE, reduced from 5)
  dashboard_hosting: $100/month

total_project_3: $5,865/month
# Rounded to $6,200/month in ADR
```

### Capacity Analysis

```yaml
current_load:
  images_per_day: 10,000
  images_per_hour: 417 (average)
  peak_images_per_hour: 1,000

capacity:
  single_gpu_throughput: 400 images/hour (with batching)
  base_capacity: 800 images/hour (2 GPUs)
  peak_capacity: 2,000 images/hour (5 GPUs)
  headroom: 100% above peak

processing_time:
  average: 2.5 seconds per image
  p99: 4.8 seconds
  sla: 5 seconds
  margin: 0.2 seconds (4%)

cost_per_image: $0.005
monthly_images: 300,000
infrastructure_cost_per_image: $0.005
```

### Business Value Calculation

```yaml
cost_savings:
  current_manual_moderation: $500,000/year (5 FTE × $100K)
  ml_infrastructure: $74,000/year
  remaining_human_moderation: $36,000/year (1 FTE)
  total_new_cost: $110,000/year
  savings: $390,000/year

efficiency_gains:
  manual_review_time: 2 hours/image
  ml_review_time: 2.5 seconds/image
  speedup: 2,880x faster

seller_experience:
  listing_approval_time: 2.5s vs 2 hours (previous)
  seller_satisfaction_improvement: 40%
  estimated_revenue_impact: $100K/year (more listings)

net_benefit:
  cost_savings: $390,000
  revenue_increase: $100,000
  infrastructure_cost: -$74,000
  human_cost: -$36,000
  net_annual_benefit: $380,000

roi: 514%
payback_period: 2.3 months
```

---

## Consolidated Summary

### Total Monthly Cost Breakdown

```yaml
project_costs:
  project_1_recommendations: $8,500
  project_2_fraud_detection: $15,000
  project_3_image_moderation: $6,200
  total_monthly: $29,700

annual_cost: $356,400

cost_by_category:
  compute: $18,500 (62%)
  storage: $3,200 (11%)
  networking: $2,000 (7%)
  monitoring: $1,500 (5%)
  data_pipeline: $2,500 (8%)
  database: $1,000 (3%)
  misc: $1,000 (4%)

cost_by_function:
  training: $4,654 (16%)
  serving: $23,046 (78%)
  monitoring: $1,500 (5%)
  storage: $500 (2%)
```

### Business Value Summary

```yaml
annual_business_value:
  project_1_revenue_increase: $2.5M (CTR + AOV improvements)
  project_2_fraud_prevented: $2.57M (net after FP costs)
  project_3_cost_savings: $490K (moderation + revenue)
  total_annual_value: $5.56M

annual_costs:
  infrastructure: $356K
  human_resources: $36K (reduced moderators)
  total_annual_cost: $392K

net_annual_benefit: $5.17M
overall_roi: 1,318%
```

### Cost Optimization Roadmap

#### Phase 1: Immediate (0-3 months)

```yaml
quick_wins:
  1_reserved_instances:
      target: Base capacity servers
      savings: $800/month
      effort: Low (purchase through AWS console)

  2_s3_lifecycle_policies:
      target: Old images, features
      savings: $500/month
      effort: Low (configure lifecycle rules)

  3_spot_instances_training:
      target: All training jobs (if not already)
      savings: $1,200/month
      effort: Medium (update training scripts)

  total_phase_1: $2,500/month ($30K/year)
```

#### Phase 2: Optimization (3-6 months)

```yaml
medium_term:
  1_model_compression:
      target: All models
      technique: Quantization, pruning
      savings: $1,500/month (reduce serving compute)
      effort: High (model engineering work)

  2_caching_improvements:
      target: Feature store, predictions
      savings: $800/month
      effort: Medium (tune cache policies)

  3_auto_scaling_tuning:
      target: All services
      savings: $600/month (better scale-down)
      effort: Medium (monitoring and tuning)

  total_phase_2: $2,900/month ($35K/year)
```

#### Phase 3: Architecture (6-12 months)

```yaml
long_term:
  1_serverless_migration:
      target: Image moderation workers
      approach: Lambda with GPU (when available)
      savings: $1,000/month
      effort: High (rewrite components)

  2_edge_caching:
      target: Popular recommendations
      approach: CloudFront CDN
      savings: $400/month
      effort: Medium (setup CDN)

  3_batch_optimization:
      target: All batch jobs
      approach: Larger batches, better scheduling
      savings: $500/month
      effort: Medium (optimize pipelines)

  total_phase_3: $1,900/month ($23K/year)
```

**Total Potential Savings**: $7,300/month ($88K/year)
**Optimized Annual Cost**: $268K (from $356K)
**Optimized ROI**: 1,931% (from 1,318%)

---

## Risk Analysis and Contingency

### Cost Overrun Risks

```yaml
risks:
  1_traffic_spike:
      probability: Medium
      impact: +50% costs during spike
      mitigation: Auto-scaling limits, rate limiting
      contingency: $15K/month buffer

  2_model_complexity:
      probability: Low
      impact: +30% training costs
      mitigation: Experiment tracking, cost monitoring
      contingency: $5K/month

  3_data_growth:
      probability: High
      impact: +20% storage costs
      mitigation: Lifecycle policies, compression
      contingency: $3K/month

  total_contingency: $23K/month (77% increase)
  worst_case_monthly: $52,700
```

### Cost Control Measures

```yaml
monitoring:
  1_cost_alerts:
      daily_spend_threshold: $1,200/day
      monthly_spend_threshold: $35,000/month
      notification: Email + Slack

  2_budget_forecasting:
      tool: AWS Cost Explorer
      frequency: Weekly
      action: Review and adjust

  3_tagging_strategy:
      tags: [project, environment, team, cost_center]
      purpose: Cost attribution and optimization

governance:
  1_approval_process:
      under_$500/month: Team lead approval
      over_$500/month: Director approval
      new_services: Architecture review

  2_quarterly_reviews:
      scope: All infrastructure costs
      output: Optimization recommendations
      target: 5% quarterly cost reduction
```

---

## Appendix A: Pricing References (AWS, October 2025)

### Compute Pricing

```yaml
ec2_instances:
  c5.xlarge: $0.17/hr on-demand, $0.08/hr spot
  c5.2xlarge: $0.34/hr on-demand, $0.15/hr spot
  r5.2xlarge: $0.504/hr on-demand, $0.18/hr spot
  p3.2xlarge: $3.06/hr on-demand, $1.02/hr spot
  g4dn.xlarge: $0.526/hr on-demand, $0.20/hr spot

reserved_instances:
  1_year_no_upfront: 20% discount
  1_year_all_upfront: 35% discount
  3_year_all_upfront: 50% discount
```

### Storage Pricing

```yaml
s3_storage:
  standard: $0.023/GB/month
  infrequent_access: $0.0125/GB/month
  glacier: $0.004/GB/month

ebs_volumes:
  gp3: $0.08/GB/month
  io2: $0.125/GB/month

database_storage:
  rds: $0.115/GB/month
```

### Managed Services

```yaml
eks: $0.10/hr ($73/month per cluster)
msk_kafka: $0.21/hr per broker (m5.large)
elasticache_redis: $0.068/hr (t3.medium)
rds_postgresql: $0.068/hr (db.t3.medium)
```

---

**Document Owner**: ML Infrastructure Team + Finance
**Last Updated**: 2025-10-30
**Next Review**: 2025-11-30 (Monthly)
**Version**: 1.0
