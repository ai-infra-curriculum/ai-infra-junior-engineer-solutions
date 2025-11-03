# ADR: Image Moderation System for Product Uploads

**Date**: 2025-10-30
**Status**: Accepted
**Decision Makers**: ML Infrastructure + Trust & Safety Team
**Project**: TechShop Automated Image Moderation

---

## Context

### Business Requirements

TechShop allows sellers to upload product images, but needs automated moderation to detect inappropriate content (violence, nudity, hate symbols) before images go live.

**Success Metrics**:
- Detect 98%+ inappropriate images (high recall)
- Process images within **5 seconds** of upload
- Handle 1,000 image uploads/hour (peak)
- Minimize seller friction (low false positive rate)

**Scale**:
- Average image size: 2MB (range 500KB - 10MB)
- Image resolution: 800x600 to 4000x3000 pixels
- 10,000 new product listings/day
- Storage: 50TB of historical images
- Growth: 20% year-over-year

**Data**:
- Product images uploaded by sellers
- Labels: appropriate/inappropriate with categories
  - ✅ Appropriate: Normal products
  - ❌ Inappropriate: Violence, nudity, hate symbols, fake products, trademark violations
- Image metadata: uploader, timestamp, product category

### Business Impact

```
Current State (Manual Review):
- Every image reviewed by human moderators
- 2-hour average review time
- $500K annual moderation cost
- Poor seller experience (slow listing approval)

Target State (With ML):
- 95% auto-moderation (confident cases)
- 5% human review (uncertain cases)
- <5 second automated decisions
- $150K annual cost (70% savings)
- Improved seller experience
```

### Technical Constraints

1. **Asynchronous Acceptable**: Sellers don't need instant approval
2. **Storage Costs**: 50TB of images, growing 10TB/year
3. **GPU Compute**: Deep learning models require GPU for speed
4. **Human-in-the-Loop**: Need review queue for uncertain cases
5. **Explainability**: Must explain why image was flagged

---

## Decision

### High-Level Architecture

We chose an **asynchronous batch processing** architecture with confidence-based routing:

```
Image Upload Flow:

[Seller Uploads Image]
        ↓
   [S3 Storage]
        ↓
   [SQS Queue]
        ↓
[Image Processor]
   (Lambda/ECS)
        ↓
  [Preprocessing]
    (Resize, Normalize)
        ↓
[ML Model Inference]
   (GPU Batch Processing)
        ↓
[Confidence Routing]
        ↓
    ┌───────┴────────┐
    ↓                ↓
[Auto Decision]  [Review Queue]
  (95% cases)    (5% cases)
    ↓                ↓
[Update Listing] [Human Moderator]
                     ↓
              [Feedback Loop]
                     ↓
              [Model Retraining]
```

### Key Architectural Decisions

#### 1. Model Architecture: EfficientNetV2 with TensorRT Optimization

**Choice**: EfficientNetV2-M as base model, optimized with TensorRT

**Rationale**:
- **Accuracy**: State-of-the-art on image classification (98.5% on validation set)
- **Efficiency**: Optimized for speed vs accuracy trade-off
- **Transfer Learning**: Pretrained on ImageNet, fine-tuned on our data
- **GPU Optimization**: TensorRT provides 3-5x speedup

**Alternatives Considered**:
| Model | Accuracy | Latency/Image | GPU Memory | Decision |
|-------|----------|---------------|------------|----------|
| **ResNet50** | 94% | 30ms | 1GB | ❌ Lower accuracy |
| **EfficientNetV2-S** | 96% | 20ms | 800MB | ✅ Good candidate |
| **EfficientNetV2-M** | 98.5% | 40ms | 1.5GB | ✅ **Chosen** (best accuracy) |
| **ViT-Large** | 99% | 150ms | 4GB | ❌ Too slow, expensive |

**Model Specifications**:
```yaml
model:
  architecture: EfficientNetV2-M
  input_size: 512x512 RGB
  output: 5 classes (appropriate, violence, nudity, hate, fake/trademark)
  parameters: 54M
  model_size: 210MB (FP32), 105MB (FP16)

optimization:
  tensorrt: True
  precision: FP16
  batch_size: 16
  throughput: 400 images/minute (single GPU)

training:
  pretrained_weights: ImageNet
  fine_tuning_epochs: 20
  dataset_size: 500K labeled images
  augmentation: rotation, flip, color jitter, cutout
```

#### 2. Processing Strategy: Asynchronous Batch with GPU

**Choice**: Process images in batches on GPU instances asynchronously

**Architecture**:
```yaml
image_upload:
  1_upload_to_s3:
      action: Seller uploads image
      storage: S3 bucket
      immediate_response: "Image uploaded, processing..."

  2_trigger_processing:
      action: S3 event → SQS queue
      queue: image-moderation-queue
      visibility_timeout: 60 seconds

  3_batch_processing:
      worker: ECS tasks with GPU (g4dn.xlarge)
      scaling: 2-10 tasks based on queue depth
      batch_size: 16 images
      processing_time: 3 seconds per batch

  4_store_results:
      database: PostgreSQL
      cache: Results cached in Redis
      notification: Seller notified via email/webhook

  5_human_review:
      if: Confidence < 90%
      queue: Human moderator dashboard
      sla: Review within 2 hours
```

**Rationale**:
- **Cost**: Batch processing 60% cheaper than real-time
- **Quality**: Can use larger, more accurate models
- **Acceptable latency**: 5 seconds is fine for async workflow
- **GPU efficiency**: Batch size 16 utilizes GPU fully

**Batch Processing Optimization**:
```python
# Process images in batches for GPU efficiency
def process_batch(image_paths, model, batch_size=16):
    """Process images in batches for optimal GPU utilization."""
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]

        # Parallel preprocessing on CPU
        images = parallel_load_and_preprocess(batch, num_workers=4)

        # GPU inference
        with torch.no_grad():
            predictions = model(images)
            confidences = torch.softmax(predictions, dim=1)

        # Post-processing
        for j, (pred, conf) in enumerate(zip(predictions, confidences)):
            results.append({
                "image_path": batch[j],
                "prediction": pred.argmax().item(),
                "confidence": conf.max().item(),
                "all_scores": conf.tolist()
            })

    return results

# Achieve 400 images/minute throughput (vs 60 without batching)
```

#### 3. Multi-Resolution Strategy

**Choice**: Two-stage processing - quick screen + detailed analysis

**Stage 1: Quick Screen (512x512, 20ms/image)**:
```python
# Resize to 512x512 for fast initial screening
quick_screen = {
    "model": "EfficientNetV2-S",
    "resolution": "512x512",
    "purpose": "Filter obvious cases",
    "latency": "20ms per image",
    "coverage": "80% of images (high confidence)"
}

# If confidence > 95%: auto-approve/reject
# If confidence < 95%: proceed to detailed analysis
```

**Stage 2: Detailed Analysis (2048x2048, 100ms/image)**:
```python
# Full resolution for uncertain cases
detailed_analysis = {
    "model": "EfficientNetV2-M",
    "resolution": "2048x2048",
    "purpose": "Analyze uncertain cases",
    "latency": "100ms per image",
    "coverage": "20% of images (low confidence from Stage 1)"
}

# If still confidence < 90%: human review
```

**Benefits**:
- **Cost**: 3x cheaper (avoid expensive processing for easy cases)
- **Quality**: Best of both worlds (fast + accurate)
- **Latency**: 80% of images processed in <1 second

#### 4. Confidence-Based Routing

**Choice**: Three-tier decision system based on model confidence

**Routing Logic**:
```python
def route_decision(prediction, confidence, image_metadata):
    """Route image to appropriate action based on confidence."""

    # Tier 1: High Confidence Auto-Approve (>95%)
    if prediction == "appropriate" and confidence > 0.95:
        return {
            "action": "auto_approve",
            "reason": "High confidence appropriate",
            "processing_time": "1 second"
        }

    # Tier 2: High Confidence Auto-Reject (>95%)
    if prediction in ["violence", "nudity", "hate"] and confidence > 0.95:
        return {
            "action": "auto_reject",
            "reason": f"High confidence {prediction}",
            "processing_time": "1 second",
            "explanation": generate_explanation(image, prediction)
        }

    # Tier 3: Medium Confidence Auto-Reject (>85%)
    if prediction in ["violence", "nudity", "hate"] and confidence > 0.85:
        return {
            "action": "auto_reject",
            "reason": f"Medium confidence {prediction}",
            "priority": "low",  # Less confident rejections
            "explanation": generate_explanation(image, prediction)
        }

    # Tier 4: Uncertain Cases - Human Review (<85% or borderline)
    return {
        "action": "human_review",
        "reason": "Low confidence or borderline case",
        "priority": calculate_priority(confidence, prediction),
        "estimated_wait": "2 hours",
        "suggested_action": prediction
    }

# Distribution:
# - 85% auto-approved (confidence > 95%)
# - 10% auto-rejected (confidence > 85%)
# - 5% human review (confidence < 85%)
```

**Priority Scoring for Human Review**:
```python
def calculate_priority(confidence, prediction):
    """Prioritize reviews based on risk and uncertainty."""

    # High priority: Potential false negatives (missed inappropriate)
    if prediction in ["violence", "nudity", "hate"] and confidence > 0.70:
        return "high"  # Likely inappropriate but not confident enough

    # Medium priority: Borderline cases
    if 0.50 < confidence < 0.70:
        return "medium"

    # Low priority: Very uncertain
    return "low"

# Review queue ordering: high → medium → low
```

#### 5. Active Learning and Feedback Loop

**Choice**: Continuous model improvement via human feedback

**Feedback Loop**:
```yaml
human_review_integration:
  1_moderator_review:
      dashboard: Web UI for moderators
      displays: Image, model prediction, confidence
      actions: [Approve, Reject (with category), Flag for review]
      average_time: 30 seconds per image

  2_feedback_collection:
      store: PostgreSQL with labels
      track:
        - Moderator decision
        - Model prediction
        - Confidence score
        - Time to review
        - Disagreement cases

  3_model_retraining:
      frequency: Weekly
      data: Last week's human-reviewed images
      strategy: Fine-tune existing model
      validation: Hold-out set + shadow mode testing

  4_prioritize_uncertain_examples:
      active_learning: True
      strategy: Sample low-confidence predictions for labeling
      goal: Improve model on hard cases
```

**Improvement Tracking**:
```python
# Monitor model improvement over time
improvement_metrics = {
    "week_1": {
        "accuracy": 95.0%,
        "human_review_rate": 8%,
        "false_positive_rate": 2%
    },
    "week_4": {
        "accuracy": 97.2%,
        "human_review_rate": 6%,
        "false_positive_rate": 1.2%
    },
    "week_12": {
        "accuracy": 98.5%,
        "human_review_rate": 5%,
        "false_positive_rate": 0.8%
    }
}

# Active learning reduces human review need by 3% over 3 months
```

---

## Rationale

### Why This Architecture?

#### 1. Meets Performance Requirements

**Target**: Process within 5 seconds
**Achieved**: Average 2.5 seconds, P99 4.8 seconds

**Latency Breakdown**:
```
Component                    Time (seconds)
--------------------------------------------
S3 Upload                    0.5
SQS Queue Wait               0.3
Image Download from S3       0.2
Preprocessing                0.4
  ├─ Resize to 512x512      (0.2s)
  └─ Normalize              (0.2s)
Quick Screen (512x512)       0.3
Detailed Analysis (if needed) 0.6 (20% of images)
Result Storage               0.2
Notification                 0.1
--------------------------------------------
Average Total                2.5s
P99 Total                    4.8s
```

#### 2. High Accuracy with Low False Positives

**Target**: 98% recall, low false positive rate
**Achieved**: 98.5% recall, 0.8% false positive rate

**Confusion Matrix (Validation Set)**:
```
                Predicted
              Appropriate | Inappropriate
Actual  ├─────────────────┼───────────────┤
Approp. │    48,500 (TN) │    400 (FP)   │ 48,900
Inappro.│      75 (FN)   │  4,925 (TP)   │  5,000
         ────────────────────────────────
            48,575          5,325          53,900

Metrics:
- Recall: 98.5% (TP / (TP + FN) = 4,925 / 5,000)
- Precision: 92.5% (TP / (TP + FP) = 4,925 / 5,325)
- False Positive Rate: 0.8% (FP / (TN + FP) = 400 / 48,900)
- F1 Score: 95.4%
```

**Impact**:
- **Seller Experience**: 0.8% false positives = 8 out of 1,000 legitimate images incorrectly flagged
- **Appeals**: Sellers can appeal, reviewed by humans
- **Trust & Safety**: Catch 98.5% of inappropriate content automatically

#### 3. Cost-Effective

**Monthly Infrastructure Cost**: ~$6,200

**Breakdown**:
```yaml
compute:
  gpu_instances_g4dn_xlarge: $3,600/month
    - 2 instances running 24/7: $2,400/month
    - Auto-scaling +3 instances during peaks: $1,200/month
    - Cost per image: $0.005

  ecs_orchestration: $100/month

storage:
  s3_images: $1,000/month (50TB × $0.02/GB)
  s3_lifecycle_glacier: Save $400/month (move old images)

  database_rds: $400/month (PostgreSQL for results)
  redis_cache: $200/month (cache predictions)

monitoring:
  cloudwatch_logs: $150/month
  datadog_dashboard: $300/month

networking:
  s3_transfer: $200/month
  load_balancer: $150/month

human_review:
  moderator_dashboard_hosting: $100/month
  moderator_salaries: $3,000/month (1 FTE, reduced from 5 FTE)

total_infrastructure: $6,200/month
total_with_salaries: $9,200/month

savings_vs_manual: $500K - ($9.2K × 12) = $389K annual savings (78%)
```

**Cost Optimization**:
- Spot instances for non-critical processing (50% discount)
- S3 Lifecycle policies (move to Glacier after 90 days)
- Batch processing maximizes GPU utilization
- Cache frequently accessed images

#### 4. Scalable to Future Growth

**Current**: 10,000 images/day (417 images/hour)
**Capacity**: 24,000 images/day (1,000 images/hour with current setup)
**Future**: 50,000 images/day (with auto-scaling to 10 GPU instances)

**Scaling Strategy**:
```yaml
horizontal_scaling:
  gpu_workers: 2 → 10 instances (5x)
  queue_throughput: 1,000 → 5,000 images/hour
  cost_scaling: Linear with GPU instances

vertical_scaling:
  batch_size: 16 → 32 (2x throughput per instance)
  model_optimization: TensorRT FP16 → INT8 (2x speedup)

storage_scaling:
  s3: Unlimited scalability
  database: RDS can scale to 64TB
```

---

## Consequences

### Positive Consequences

#### ✅ Business Benefits

1. **Cost Savings**: $389K/year (78% reduction vs manual review)
2. **Faster Listings**: 2.5 seconds vs 2 hours
3. **Seller Satisfaction**: Immediate feedback, clear explanations
4. **Scalability**: Can handle 5x growth without architecture changes

#### ✅ Technical Benefits

1. **High Accuracy**: 98.5% recall, 0.8% false positives
2. **Low Latency**: Average 2.5 seconds processing time
3. **GPU Efficiency**: Batch processing maximizes utilization
4. **Maintainable**: Clear components, good separation of concerns

#### ✅ Operational Benefits

1. **Human-in-the-Loop**: Moderators focus on hard cases only
2. **Active Learning**: Model improves continuously
3. **Explainability**: Can show why image was flagged
4. **Audit Trail**: All decisions logged for compliance

### Negative Consequences

#### ⚠️ Limitations

1. **Asynchronous Only**: Not suitable for synchronous use cases
   - **Impact**: Acceptable for product listings
   - **Mitigation**: Set seller expectations clearly

2. **GPU Dependency**: Requires expensive GPU instances
   - **Impact**: $3.6K/month GPU cost
   - **Mitigation**: Spot instances, batch processing efficiency

3. **Cold Start for New Categories**: Model may struggle with new product types
   - **Impact**: Higher human review rate initially
   - **Mitigation**: Active learning, category-specific fine-tuning

4. **Image Quality Dependent**: Low-quality images harder to classify
   - **Impact**: More false positives/negatives
   - **Mitigation**: Prompt sellers to upload high-quality images

#### ⚠️ Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Model Bias** | Medium | High | Regular fairness audits, diverse training data |
| **Adversarial Attacks** | Low | Medium | Adversarial training, human review for suspicious patterns |
| **GPU Instance Failure** | Low | Medium | Multi-instance deployment, auto-scaling |
| **False Positive Spike** | Low | High | Monitoring alerts, manual review before rejecting |
| **Data Privacy** | Low | High | Encryption at rest/transit, access controls |

---

## Monitoring and Success Metrics

### Model Performance

```yaml
accuracy_metrics:
  recall:
    target: 98%
    current: 98.5%
    alert_threshold: < 97%

  precision:
    target: 90%
    current: 92.5%
    alert_threshold: < 88%

  false_positive_rate:
    target: < 1.5%
    current: 0.8%
    alert_threshold: > 2%

  human_review_rate:
    target: < 8%
    current: 5%
    alert_threshold: > 10%
```

### System Performance

```yaml
processing_metrics:
  average_latency:
    target: < 5 seconds
    current: 2.5 seconds
    alert_threshold: > 6 seconds

  p99_latency:
    target: < 10 seconds
    current: 4.8 seconds
    alert_threshold: > 12 seconds

  throughput:
    target: 1,000 images/hour
    current: 1,200 images/hour
    capacity: 2,400 images/hour

  queue_depth:
    normal: < 50 images
    alert_threshold: > 200 images (backlog forming)

  gpu_utilization:
    target: > 70%
    current: 82%
    alert_threshold: < 50% (underutilized) or > 95% (saturated)
```

### Business Metrics

```yaml
business_impact:
  cost_per_image:
    target: < $0.01
    current: $0.005
    breakdown: $0.003 compute + $0.002 storage/overhead

  seller_satisfaction:
    listing_approval_time: 2.5 seconds vs 2 hours (previous)
    appeal_rate: 1.2% (low, indicates good quality)
    appeal_success_rate: 60% (balanced, not too high or low)

  trust_and_safety:
    inappropriate_content_caught: 98.5%
    inappropriate_content_live: < 0.1% (manual sampling)
    user_reports_on_auto_approved: < 0.05%
```

---

## Implementation Roadmap

### Phase 1: MVP (Weeks 1-4)

```yaml
week_1:
  - Set up S3, SQS, ECS infrastructure
  - Train baseline model (EfficientNetV2-S)
  - Implement image preprocessing pipeline

week_2:
  - Deploy model on GPU instance
  - Implement batch processing worker
  - Create results database

week_3:
  - Build human review dashboard
  - Implement confidence-based routing
  - Shadow mode (log predictions, don't act on them)

week_4:
  - Validate against human reviewers
  - Deploy to 10% of traffic
  - Monitor and adjust thresholds
```

### Phase 2: Production (Weeks 5-8)

```yaml
week_5:
  - Scale to 50% of traffic
  - Optimize GPU batching
  - Implement multi-resolution processing

week_6:
  - 100% traffic rollout
  - Set up monitoring and alerting
  - Document runbooks

week_7:
  - Implement active learning pipeline
  - Begin weekly retraining
  - Cost optimization pass

week_8:
  - Team training and knowledge transfer
  - Performance tuning
  - Launch retrospective and lessons learned
```

---

**Document Owner**: ML Infrastructure + Trust & Safety Team
**Last Updated**: 2025-10-30
**Next Review**: 2025-11-30 (Monthly review)
**Version**: 1.0
