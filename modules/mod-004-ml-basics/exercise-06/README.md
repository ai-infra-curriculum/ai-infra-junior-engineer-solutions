# Exercise 06: ML System Design and Workflow Planning - Solution

**Module:** ML Basics
**Difficulty:** Intermediate to Advanced
**Estimated Time:** 3-4 hours
**Completion Date:** 2025-10-30

## Table of Contents

1. [Overview](#overview)
2. [Solution Approach](#solution-approach)
3. [Projects Summary](#projects-summary)
4. [Key Design Principles](#key-design-principles)
5. [Files and Documents](#files-and-documents)
6. [Learning Outcomes Achieved](#learning-outcomes-achieved)
7. [How to Use This Solution](#how-to-use-this-solution)

---

## Overview

This solution provides comprehensive ML system designs for three production scenarios:

1. **Product Recommendation System** - Hybrid collaborative+content filtering
2. **Fraud Detection System** - Real-time anomaly detection with class imbalance handling
3. **Image Moderation System** - Asynchronous deep learning-based classification

Each project includes complete architecture documentation, trade-off analysis, resource estimation, and deployment strategies.

### Solution Philosophy

The solutions demonstrate **production-ready ML infrastructure design** with emphasis on:

- **Pragmatic Trade-offs**: Balancing accuracy, latency, and cost
- **Scalability**: Handling millions of requests per day
- **Reliability**: Comprehensive monitoring, alerting, and rollback strategies
- **Cost Efficiency**: Optimizing compute and storage costs
- **Operational Excellence**: Clear deployment procedures and incident response

---

## Solution Approach

### Design Methodology

Each project follows a structured design process:

```
1. Problem Analysis → 2. Architecture Design → 3. Resource Planning → 4. Deployment Strategy
        ↓                       ↓                       ↓                      ↓
   Requirements         Infrastructure          Cost Estimation        Monitoring &
   Trade-offs          Component Selection      Resource Sizing        Rollback Plans
```

### Key Design Decisions Framework

For each project, we systematically addressed:

**Functional Requirements**:
- What ML problem type? (classification, ranking, etc.)
- What model architecture?
- What features are needed?

**Non-Functional Requirements**:
- Latency targets (100ms vs 300ms vs 5 seconds)
- Throughput requirements (1K/sec vs 10K/sec)
- Availability and reliability expectations

**Trade-off Analysis**:
- Real-time vs batch inference
- Accuracy vs latency vs cost
- Freshness vs computational cost
- Simple vs complex architecture

**Operational Considerations**:
- Deployment strategy (canary, blue-green, A/B)
- Monitoring and alerting thresholds
- Rollback procedures and incident response
- Cost optimization strategies

---

## Projects Summary

### Project 1: Product Recommendation System

**Complexity:** Moderate
**Key Challenge:** Balancing recommendation freshness with sub-100ms latency

**Solution Highlights**:
- **Model**: Hybrid two-tower architecture (user + item embeddings) with collaborative filtering
- **Inference Strategy**: Batch pre-computation with real-time scoring
- **Data Pipeline**: Lambda architecture (batch + streaming)
- **Deployment**: Kubernetes with HPA, Redis caching
- **Estimated Cost**: $8,500/month

**Key Trade-off**:
- Chose hybrid batch+real-time over pure real-time for cost efficiency (60% cost reduction)
- Pre-compute candidate recommendations offline, apply real-time personalization signals

**Innovation**:
- Multi-stage retrieval (candidate generation → ranking → re-ranking)
- Feature store with online/offline split
- Incremental model updates (daily lightweight retrain)

### Project 2: Fraud Detection System

**Complexity:** High
**Key Challenge:** 99.9% class imbalance + real-time <300ms latency + minimizing false positives

**Solution Highlights**:
- **Model**: Ensemble (XGBoost + Neural Net) with class weights and focal loss
- **Inference Strategy**: Pure real-time with aggressive caching
- **Data Pipeline**: Stream processing (Kafka + Flink)
- **Deployment**: Multi-region with circuit breakers and fallbacks
- **Estimated Cost**: $15,000/month

**Key Trade-off**:
- Precision over recall (minimize false positives)
- Multi-stage detection (fast rules → ML model → deep analysis)
- Graceful degradation when model unavailable

**Innovation**:
- Streaming feature computation with 5-minute windows
- Confidence-based routing (high confidence auto-approve/decline, medium → deep analysis)
- Continuous online learning from fraud analyst feedback

### Project 3: Image Moderation System

**Complexity:** Moderate-High
**Key Challenge:** Large images (2MB) + GPU optimization + human-in-the-loop

**Solution Highlights**:
- **Model**: EfficientNetV2 with TensorRT optimization
- **Inference Strategy**: Asynchronous batch processing
- **Data Pipeline**: S3 → SQS → Lambda/ECS batch workers
- **Deployment**: GPU-optimized EC2 instances with auto-scaling
- **Estimated Cost**: $6,200/month

**Key Trade-off**:
- Asynchronous over synchronous (sellers don't wait)
- GPU batching (process 16 images together) for 3x cost savings
- Confidence thresholds for human review (90%+ auto-action)

**Innovation**:
- Multi-resolution processing (quick 512x512 screen, detailed 2048x2048 if needed)
- Active learning pipeline (human feedback → model improvement)
- Progressive rollout by product category

---

## Key Design Principles

### 1. Latency Budget Allocation

Every millisecond matters. Break down latency requirements:

```
Example (Fraud Detection 300ms total):
- Network/API Gateway: 20ms
- Feature lookup (Redis): 30ms
- Feature computation: 50ms
- Model inference: 120ms
- Business logic: 40ms
- Response serialization: 20ms
- Buffer: 20ms
-------------------------
Total: 300ms
```

### 2. Multi-Stage Inference

Don't use expensive models for easy cases:

```
Stage 1: Simple rules (50% of traffic, <5ms)
   ↓ (pass through)
Stage 2: Lightweight model (40% of traffic, <50ms)
   ↓ (uncertain cases)
Stage 3: Deep model (10% of traffic, <200ms)
```

### 3. Feature Store Pattern

Separate feature computation from model serving:

```
Offline Features         Online Features
(pre-computed)          (real-time)
      ↓                       ↓
    Redis ← Query ← Feature Service → Model
      ↓
   (cached)
```

### 4. Defense in Depth

Multiple layers of protection:

```
Request → Rate Limiting → Input Validation → Model → Output Validation → Response
              ↓                ↓                ↓            ↓
          Protect API      Prevent bad     Timeout      Sanity checks
                          features      & fallback
```

### 5. Cost Optimization Strategies

**Compute**:
- Batch processing where possible (3-5x cheaper)
- Auto-scaling based on queue depth, not just CPU
- Spot/preemptible instances for training (60% savings)

**Storage**:
- Lifecycle policies (S3 Standard → Glacier after 90 days)
- Compression (Parquet with Snappy)
- Data pruning (keep 1% sample for analysis)

**Caching**:
- Cache predictions for popular items/users
- TTL based on staleness tolerance
- Multi-level caching (L1: in-memory, L2: Redis, L3: database)

---

## Files and Documents

### Core Documentation

```
exercise-06/
├── README.md                              # This file
├── docs/
│   ├── project-1-recommendation-adr.md   # Architecture Decision Record
│   ├── project-2-fraud-detection-adr.md  # Architecture Decision Record
│   ├── project-3-image-moderation-adr.md # Architecture Decision Record
│   ├── project-1-design.md               # Detailed design document
│   ├── project-2-design.md               # Detailed design document
│   ├── project-3-design.md               # Detailed design document
│   ├── resource-estimation.md            # Cost breakdown for all projects
│   ├── deployment-runbook-p1.md          # Deployment procedures
│   ├── deployment-runbook-p2.md          # Deployment procedures
│   └── deployment-runbook-p3.md          # Deployment procedures
├── diagrams/
│   ├── project-1-architecture.md         # System architecture diagrams
│   ├── project-2-architecture.md         # System architecture diagrams
│   └── project-3-architecture.md         # System architecture diagrams
└── templates/
    ├── adr-template.md                   # ADR template for future use
    ├── design-doc-template.md            # Design document template
    └── runbook-template.md               # Runbook template
```

### Document Purposes

**Architecture Decision Records (ADRs)**:
- High-level architectural choices
- Alternatives considered and why rejected
- Consequences (pros/cons/risks)
- Target audience: Technical leads, architects

**Design Documents**:
- Detailed technical specifications
- Component interactions and data flows
- Implementation details and algorithms
- Target audience: Engineers implementing the system

**Deployment Runbooks**:
- Step-by-step operational procedures
- Monitoring setup and alert thresholds
- Incident response and rollback procedures
- Target audience: SREs, DevOps engineers

**Resource Estimation**:
- Infrastructure cost breakdown
- Capacity planning calculations
- Cost optimization opportunities
- Target audience: Engineering managers, finance

---

## Learning Outcomes Achieved

### ✅ System Design Skills

- [x] Designed 3 end-to-end ML workflows from requirements to deployment
- [x] Made informed trade-offs between competing objectives
- [x] Planned scalable data pipelines with validation
- [x] Architected deployment strategies for different use cases
- [x] Estimated resource requirements accurately
- [x] Applied production ML best practices

### ✅ Technical Depth

**Problem Analysis**:
- Identified appropriate ML problem types (ranking, binary classification, multi-class)
- Selected suitable model architectures based on constraints
- Designed feature engineering pipelines

**Infrastructure Design**:
- Chose deployment platforms (Kubernetes, ECS, Lambda)
- Designed caching strategies (Redis, application-level)
- Planned auto-scaling policies

**Operational Excellence**:
- Defined comprehensive monitoring metrics
- Established alerting thresholds
- Created rollback procedures
- Designed A/B testing strategies

### ✅ Business Acumen

- Balanced technical excellence with cost constraints
- Prioritized metrics aligned with business goals
- Communicated trade-offs clearly to stakeholders
- Estimated TCO (Total Cost of Ownership)

---

## How to Use This Solution

### For Learning

**1. Study One Project at a Time**

Start with Project 1 (Recommendations) as it's the most straightforward:
- Read the ADR first to understand high-level decisions
- Study the detailed design document
- Review the architecture diagrams
- Examine the deployment runbook

**2. Compare Alternative Approaches**

For each decision, ask:
- Why was this approach chosen over alternatives?
- What would change if requirements were different?
- How would this scale to 10x traffic?

**3. Practice Design Thinking**

Before reading the solution:
- Try designing it yourself
- Document your assumptions
- Compare with the provided solution
- Identify gaps in your design

### For Job Interviews

**System Design Interview Preparation**:

These documents demonstrate how to:
- Structure your thinking (requirements → architecture → trade-offs)
- Communicate design decisions clearly
- Consider operational concerns (monitoring, deployment)
- Estimate costs and resources

**Practice Exercise**:
1. Pick one project
2. Set 45-minute timer
3. Design on whiteboard/paper without looking at solution
4. Compare your design with solution
5. Note what you missed

### For Real-World Projects

**Adaptation Guide**:

Use these as templates:
1. Copy the ADR template
2. Fill in your project's requirements
3. Adapt the design patterns that fit
4. Modify monitoring and deployment strategies

**What to Customize**:
- Specific tools/vendors (AWS vs GCP vs Azure)
- Cost calculations (use your cloud pricing)
- Compliance requirements (GDPR, HIPAA, etc.)
- Team size and expertise

---

## Key Takeaways

### 1. No Perfect Solution

Every design involves trade-offs:
- Accuracy ↔ Latency
- Cost ↔ Performance
- Simplicity ↔ Flexibility
- Freshness ↔ Efficiency

**The key is making intentional, documented trade-offs.**

### 2. Start Simple, Scale Gradually

Don't over-engineer:
- Begin with batch inference, add real-time if needed
- Use managed services before building custom solutions
- Monitor first, optimize later

### 3. Operational Excellence Matters

50% of ML system design is monitoring and operations:
- Comprehensive metrics and alerts
- Clear rollback procedures
- Runbooks for common issues
- On-call playbooks

### 4. Cost is a Feature

Resource optimization isn't optional:
- Training costs (GPU hours)
- Serving costs (compute, storage, network)
- Opportunity cost (complexity tax)

**Rule of thumb**: Aim for <$0.01 per prediction at scale.

### 5. Learn from Production

Best practices come from failures:
- Cascade failures (circuit breakers)
- Training-serving skew (feature consistency)
- Model degradation (monitoring)
- Incident response (runbooks)

---

## Additional Resources

### Books

- **"Designing Machine Learning Systems" by Chip Huyen** - Comprehensive ML system design
- **"Machine Learning Design Patterns" by Valliappa Lakshmanan et al.** - Practical patterns
- **"Reliable Machine Learning" by Cathy Chen et al.** - Production ML reliability

### Blogs and Papers

- **Eugene Yan's Blog** (eugeneyan.com) - Real-world ML system designs
- **Netflix Tech Blog** - Large-scale recommendation systems
- **Uber Engineering Blog** - Real-time ML systems
- **Google Research Papers** - TFX, Vertex AI architectures

### Tools and Frameworks

- **Feature Stores**: Feast, Tecton, AWS SageMaker Feature Store
- **Model Serving**: TensorFlow Serving, TorchServe, KServe, Seldon Core
- **Experiment Tracking**: MLflow, Weights & Biases, Neptune
- **Model Monitoring**: Evidently AI, Fiddler, Arize

---

## Conclusion

This solution demonstrates production-grade ML system design across three distinct use cases. The key is not memorizing specific architectures, but understanding the systematic design process:

1. **Analyze requirements** (functional + non-functional)
2. **Evaluate alternatives** (with explicit trade-offs)
3. **Design for operations** (monitoring, deployment, incidents)
4. **Estimate costs** (compute, storage, opportunity)
5. **Document decisions** (for future you and your team)

Remember: **The best ML system is one that ships, scales, and doesn't wake you up at 3 AM.**

---

**Solution created for AI Infrastructure Junior Engineer Learning Curriculum**
**Last updated: 2025-10-30**
