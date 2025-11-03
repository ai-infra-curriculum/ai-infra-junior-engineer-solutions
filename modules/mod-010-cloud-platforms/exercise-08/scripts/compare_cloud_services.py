#!/usr/bin/env python3
"""
Cloud Service Comparison Tool

Compare equivalent services across AWS, GCP, and Azure for ML infrastructure.
Helps identify best-of-breed options and cost differences.

Usage:
    python compare_cloud_services.py --category ml
    python compare_cloud_services.py --category compute
    python compare_cloud_services.py --category storage
    python compare_cloud_services.py --all
"""

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CloudService:
    """Cloud service details."""
    provider: str
    service_name: str
    pricing: str
    strengths: List[str]
    weaknesses: List[str]
    ml_fit_score: int  # 1-10
    ease_of_use: int  # 1-10
    maturity: int  # 1-10


@dataclass
class ServiceCategory:
    """Category of cloud services with comparisons."""
    name: str
    description: str
    aws_service: CloudService
    gcp_service: CloudService
    azure_service: CloudService
    recommendation: str


# ============================================================================
# ML Platform Services
# ============================================================================

ML_PLATFORM_CATEGORY = ServiceCategory(
    name="Managed ML Platform",
    description="End-to-end ML platforms for training, deployment, and management",
    aws_service=CloudService(
        provider="AWS",
        service_name="SageMaker",
        pricing="$0.269/hour (ml.m5.xlarge training), $0.065/hour (ml.t3.medium inference)",
        strengths=[
            "Most mature ML platform (launched 2017)",
            "Excellent integration with AWS ecosystem",
            "SageMaker Studio notebooks",
            "Built-in algorithms and AutoML",
            "SageMaker Pipelines for MLOps",
            "Multi-model endpoints"
        ],
        weaknesses=[
            "Can be expensive at scale",
            "Learning curve for full platform",
            "Limited custom container support initially",
            "Vendor lock-in concerns"
        ],
        ml_fit_score=9,
        ease_of_use=7,
        maturity=10
    ),
    gcp_service=CloudService(
        provider="GCP",
        service_name="Vertex AI",
        pricing="$0.227/hour (n1-standard-4 training), $0.056/hour (n1-standard-2 inference)",
        strengths=[
            "Unified platform (merged AI Platform + AutoML)",
            "Best TPU integration for large models",
            "Excellent for TensorFlow workloads",
            "Feature Store built-in",
            "Great BigQuery integration",
            "Vertex AI Pipelines (Kubeflow-based)"
        ],
        weaknesses=[
            "Newer platform (2021 unification)",
            "Less mature than SageMaker",
            "Smaller ecosystem",
            "Documentation gaps"
        ],
        ml_fit_score=8,
        ease_of_use=8,
        maturity=7
    ),
    azure_service=CloudService(
        provider="Azure",
        service_name="Azure Machine Learning",
        pricing="$0.230/hour (Standard_D4s_v3 training), $0.058/hour (Standard_D2s_v3 inference)",
        strengths=[
            "Strong enterprise integration",
            "Excellent MLOps features",
            "Good hybrid cloud support",
            "Designer for no-code ML",
            "Responsible AI dashboard",
            "Strong AutoML capabilities"
        ],
        weaknesses=[
            "UI can be confusing",
            "Slower innovation pace",
            "Less popular in ML community",
            "Limited GPU instance types"
        ],
        ml_fit_score=7,
        ease_of_use=6,
        maturity=8
    ),
    recommendation="""
    BEST FOR:
      • AWS SageMaker: Most enterprises, comprehensive features, largest ecosystem
      • GCP Vertex AI: Large model training (TPUs), TensorFlow-heavy workloads
      • Azure ML: Microsoft-centric enterprises, hybrid cloud requirements

    MULTI-CLOUD STRATEGY:
      • Use GCP for training (TPUs) + AWS for inference (broader coverage)
      • Requires careful data pipeline design to avoid egress costs
    """
)


# ============================================================================
# Compute Services
# ============================================================================

COMPUTE_CATEGORY = ServiceCategory(
    name="GPU Compute Instances",
    description="GPU instances for ML training and inference",
    aws_service=CloudService(
        provider="AWS",
        service_name="EC2 P4/P3 Instances",
        pricing="$3.06/hour (p3.2xlarge, V100), $32.77/hour (p4d.24xlarge, 8x A100)",
        strengths=[
            "Widest instance selection",
            "Best spot instance availability",
            "EFA for distributed training",
            "Elastic Fabric Adapter",
            "Global availability"
        ],
        weaknesses=[
            "Can be expensive",
            "Quota limits for large instances",
            "Instance availability varies by region"
        ],
        ml_fit_score=9,
        ease_of_use=8,
        maturity=10
    ),
    gcp_service=CloudService(
        provider="GCP",
        service_name="Compute Engine + TPUs",
        pricing="$2.48/hour (n1-highmem-4 + V100), $4.50/hour (TPU v3), $8.00/hour (TPU v4)",
        strengths=[
            "TPU access (unique advantage)",
            "Best for large transformer models",
            "Preemptible GPUs at 70% discount",
            "Custom machine types",
            "Fast local SSD"
        ],
        weaknesses=[
            "Fewer GPU options than AWS",
            "TPUs require code changes",
            "Regional availability limited"
        ],
        ml_fit_score=9,
        ease_of_use=7,
        maturity=9
    ),
    azure_service=CloudService(
        provider="Azure",
        service_name="NC/ND Series VMs",
        pricing="$3.06/hour (NC6s_v3, V100), $27.20/hour (ND96asr_v4, 8x A100)",
        strengths=[
            "InfiniBand for HPC",
            "Good for Azure-native apps",
            "Enterprise support",
            "Hybrid cloud integration"
        ],
        weaknesses=[
            "Smallest GPU selection",
            "Limited spot/preemptible options",
            "Availability issues",
            "Higher pricing"
        ],
        ml_fit_score=6,
        ease_of_use=7,
        maturity=7
    ),
    recommendation="""
    BEST FOR:
      • AWS EC2: General-purpose ML training, widest selection
      • GCP Compute + TPUs: Large language models, TensorFlow at scale
      • Azure VMs: Existing Azure infrastructure, enterprise requirements

    COST OPTIMIZATION:
      • Use GCP preemptible GPUs (70% discount) for fault-tolerant training
      • Use AWS spot instances for batch inference
      • Reserve instances for production workloads (40-60% discount)
    """
)


# ============================================================================
# Storage Services
# ============================================================================

STORAGE_CATEGORY = ServiceCategory(
    name="Object Storage",
    description="Scalable object storage for datasets and models",
    aws_service=CloudService(
        provider="AWS",
        service_name="S3",
        pricing="$0.023/GB/month (Standard), $0.0125/GB/month (Intelligent-Tiering)",
        strengths=[
            "Industry standard, most mature",
            "Excellent performance",
            "Rich storage classes",
            "S3 Select for in-place querying",
            "Strong consistency",
            "Best ecosystem integration"
        ],
        weaknesses=[
            "Egress costs ($0.09/GB)",
            "Can be expensive at petabyte scale",
            "Complex pricing model"
        ],
        ml_fit_score=10,
        ease_of_use=9,
        maturity=10
    ),
    gcp_service=CloudService(
        provider="GCP",
        service_name="Cloud Storage",
        pricing="$0.020/GB/month (Standard), $0.010/GB/month (Nearline)",
        strengths=[
            "Slightly cheaper than S3",
            "Excellent BigQuery integration",
            "Simple pricing model",
            "Free egress to Google services",
            "Object lifecycle management"
        ],
        weaknesses=[
            "Smaller ecosystem",
            "Fewer integrations",
            "Egress costs ($0.08-0.12/GB)"
        ],
        ml_fit_score=9,
        ease_of_use=9,
        maturity=9
    ),
    azure_service=CloudService(
        provider="Azure",
        service_name="Blob Storage",
        pricing="$0.0184/GB/month (Hot), $0.010/GB/month (Cool)",
        strengths=[
            "Lowest storage cost",
            "Good for hybrid scenarios",
            "Azure Data Lake integration",
            "Hierarchical namespace option"
        ],
        weaknesses=[
            "Slowest performance",
            "Limited ML tooling integration",
            "Complex access tiers"
        ],
        ml_fit_score=7,
        ease_of_use=7,
        maturity=8
    ),
    recommendation="""
    BEST FOR:
      • AWS S3: Default choice, best ecosystem, maximum compatibility
      • GCP Cloud Storage: BigQuery integration, slightly lower cost
      • Azure Blob: Microsoft-centric infrastructure, lowest storage cost

    MULTI-CLOUD CONSIDERATION:
      • Data egress costs ($0.08-0.12/GB) can make multi-cloud prohibitive
      • For 100TB dataset: $8-12k to move between clouds
      • Keep training data in same cloud as compute
    """
)


# ============================================================================
# Container Orchestration
# ============================================================================

KUBERNETES_CATEGORY = ServiceCategory(
    name="Managed Kubernetes",
    description="Managed Kubernetes for containerized ML workloads",
    aws_service=CloudService(
        provider="AWS",
        service_name="EKS (Elastic Kubernetes Service)",
        pricing="$0.10/hour control plane + EC2 costs, $0.10/hour/vCPU (Fargate)",
        strengths=[
            "Most mature AWS K8s offering",
            "Good integration with AWS services",
            "EKS Anywhere for hybrid",
            "Fargate for serverless pods",
            "Strong security (IRSA)"
        ],
        weaknesses=[
            "Control plane costs ($72/month)",
            "Slower K8s version updates",
            "More complex than competitors",
            "NAT gateway costs"
        ],
        ml_fit_score=8,
        ease_of_use=6,
        maturity=9
    ),
    gcp_service=CloudService(
        provider="GCP",
        service_name="GKE (Google Kubernetes Engine)",
        pricing="$0.10/hour control plane (free for Autopilot), + Compute costs",
        strengths=[
            "Best K8s experience (Google created K8s)",
            "Autopilot mode (fully managed)",
            "Fastest K8s version updates",
            "Great observability integration",
            "Simple networking"
        ],
        weaknesses=[
            "Less AWS service integration",
            "Smaller ecosystem",
            "Regional limitations"
        ],
        ml_fit_score=9,
        ease_of_use=9,
        maturity=10
    ),
    azure_service=CloudService(
        provider="Azure",
        service_name="AKS (Azure Kubernetes Service)",
        pricing="Free control plane + VM costs",
        strengths=[
            "Free control plane",
            "Good Azure integration",
            "Virtual nodes (serverless)",
            "Strong enterprise features",
            "Hybrid cloud support"
        ],
        weaknesses=[
            "Less mature than GKE",
            "Complex networking",
            "Documentation quality varies"
        ],
        ml_fit_score=7,
        ease_of_use=7,
        maturity=8
    ),
    recommendation="""
    BEST FOR:
      • GKE: Best pure K8s experience, cloud-agnostic workloads
      • EKS: AWS-centric infrastructure, need AWS service integration
      • AKS: Azure-native apps, free control plane attractive

    CLOUD-AGNOSTIC STRATEGY:
      • Use GKE for development (best K8s experience)
      • Deploy to any cloud using Terraform + Helm
      • Avoid cloud-specific K8s features (stay portable)
    """
)


# ============================================================================
# Model Serving
# ============================================================================

MODEL_SERVING_CATEGORY = ServiceCategory(
    name="Model Serving & Inference",
    description="Services for deploying and serving ML models",
    aws_service=CloudService(
        provider="AWS",
        service_name="SageMaker Inference",
        pricing="$0.05-0.25/hour (CPU instances), $0.50-4.00/hour (GPU instances)",
        strengths=[
            "Multi-model endpoints (cost efficient)",
            "Auto-scaling built-in",
            "A/B testing support",
            "Batch transform for offline inference",
            "Model Monitor for drift detection"
        ],
        weaknesses=[
            "Expensive for high-traffic APIs",
            "Cold start latency",
            "Vendor lock-in"
        ],
        ml_fit_score=8,
        ease_of_use=7,
        maturity=9
    ),
    gcp_service=CloudService(
        provider="GCP",
        service_name="Vertex AI Endpoints",
        pricing="$0.04-0.22/hour (CPU), $0.45-3.50/hour (GPU)",
        strengths=[
            "Simple deployment from notebooks",
            "Good auto-scaling",
            "Online prediction API",
            "Batch prediction support",
            "Model versioning"
        ],
        weaknesses=[
            "Limited advanced features",
            "Less flexible than SageMaker",
            "Newer service"
        ],
        ml_fit_score=7,
        ease_of_use=8,
        maturity=7
    ),
    azure_service=CloudService(
        provider="Azure",
        service_name="Azure ML Endpoints",
        pricing="$0.05-0.24/hour (CPU), $0.48-3.80/hour (GPU)",
        strengths=[
            "Real-time and batch endpoints",
            "Good monitoring",
            "Managed online endpoints",
            "Integration with Azure services"
        ],
        weaknesses=[
            "Less mature than competitors",
            "Complex configuration",
            "Limited documentation"
        ],
        ml_fit_score=6,
        ease_of_use=6,
        maturity=6
    ),
    recommendation="""
    BEST FOR:
      • AWS SageMaker: Multi-model endpoints, advanced MLOps features
      • GCP Vertex AI: Simple deployment, ease of use
      • Azure ML: Azure-native applications

    ALTERNATIVE: Self-Hosted on Kubernetes
      • BentoML, Seldon Core, KServe for cloud-agnostic serving
      • More operational overhead but maximum flexibility
      • Better for multi-cloud portability
    """
)


# ============================================================================
# Database Services
# ============================================================================

DATABASE_CATEGORY = ServiceCategory(
    name="Managed Databases",
    description="Managed database services for ML metadata and features",
    aws_service=CloudService(
        provider="AWS",
        service_name="RDS / DynamoDB",
        pricing="RDS: $0.017/hour (db.t3.micro), DynamoDB: $0.25/GB/month + $1.25/million writes",
        strengths=[
            "Widest database selection",
            "Excellent Aurora performance",
            "DynamoDB for NoSQL",
            "Good backup/restore",
            "Multi-AZ for HA"
        ],
        weaknesses=[
            "Can be expensive",
            "RDS maintenance windows",
            "DynamoDB pricing complexity"
        ],
        ml_fit_score=8,
        ease_of_use=8,
        maturity=10
    ),
    gcp_service=CloudService(
        provider="GCP",
        service_name="Cloud SQL / Firestore",
        pricing="Cloud SQL: $0.0150/hour (db-f1-micro), Firestore: $0.18/GB/month + $0.06/100k reads",
        strengths=[
            "Slightly cheaper than AWS",
            "Excellent BigQuery integration",
            "Firestore for real-time",
            "Automatic backups",
            "Regional HA"
        ],
        weaknesses=[
            "Fewer database options",
            "Less mature than RDS",
            "Limited instance types"
        ],
        ml_fit_score=7,
        ease_of_use=8,
        maturity=8
    ),
    azure_service=CloudService(
        provider="Azure",
        service_name="Azure Database / Cosmos DB",
        pricing="Azure DB: $0.018/hour (Basic), Cosmos DB: $0.25/GB/month + $0.25/million RUs",
        strengths=[
            "Cosmos DB global distribution",
            "Good for hybrid scenarios",
            "Multiple APIs (SQL, MongoDB, Cassandra)",
            "Strong consistency options"
        ],
        weaknesses=[
            "Cosmos DB can be very expensive",
            "Complex pricing (RU model)",
            "Limited ML integrations"
        ],
        ml_fit_score=6,
        ease_of_use=6,
        maturity=8
    ),
    recommendation="""
    BEST FOR:
      • AWS RDS/Aurora: Default choice, best performance, widest selection
      • GCP Cloud SQL: BigQuery integration, slightly lower cost
      • Azure Cosmos DB: Global distribution requirements

    ML-SPECIFIC CONSIDERATION:
      • Feature stores often need high read throughput
      • Consider Redis/ElastiCache for real-time features
      • PostgreSQL with pgvector for vector similarity search
    """
)


# ============================================================================
# Display Functions
# ============================================================================

CATEGORIES = {
    "ml": ML_PLATFORM_CATEGORY,
    "compute": COMPUTE_CATEGORY,
    "storage": STORAGE_CATEGORY,
    "kubernetes": KUBERNETES_CATEGORY,
    "serving": MODEL_SERVING_CATEGORY,
    "database": DATABASE_CATEGORY
}


def print_category_comparison(category: ServiceCategory):
    """Print detailed comparison for a service category."""
    print()
    print("=" * 80)
    print(f"  {category.name}")
    print("=" * 80)
    print(f"\n{category.description}\n")

    services = [
        category.aws_service,
        category.gcp_service,
        category.azure_service
    ]

    for service in services:
        print(f"{'=' * 80}")
        print(f"{service.provider}: {service.service_name}")
        print(f"{'=' * 80}")
        print(f"\nPricing: {service.pricing}\n")

        print("Strengths:")
        for strength in service.strengths:
            print(f"  ✓ {strength}")

        print("\nWeaknesses:")
        for weakness in service.weaknesses:
            print(f"  ✗ {weakness}")

        print(f"\nScores:")
        print(f"  • ML Fit:     {service.ml_fit_score}/10  {'█' * service.ml_fit_score}")
        print(f"  • Ease of Use: {service.ease_of_use}/10  {'█' * service.ease_of_use}")
        print(f"  • Maturity:    {service.maturity}/10  {'█' * service.maturity}")
        print()

    print("=" * 80)
    print("  RECOMMENDATION")
    print("=" * 80)
    print(category.recommendation)
    print()


def print_summary_table():
    """Print summary comparison table across all categories."""
    print()
    print("=" * 80)
    print("  Cloud Service Comparison Summary")
    print("=" * 80)
    print()

    print(f"{'Category':<20} {'AWS (Best)':<25} {'GCP (Best)':<25} {'Azure (Best)':<25}")
    print("-" * 95)

    comparisons = [
        ("ML Platform", "SageMaker (9/10)", "Vertex AI (8/10)", "Azure ML (7/10)"),
        ("GPU Compute", "EC2 P4 (9/10)", "TPU v4 (9/10)", "ND Series (6/10)"),
        ("Object Storage", "S3 (10/10)", "Cloud Storage (9/10)", "Blob (7/10)"),
        ("Kubernetes", "EKS (8/10)", "GKE (9/10)", "AKS (7/10)"),
        ("Model Serving", "SageMaker (8/10)", "Vertex AI (7/10)", "Azure ML (6/10)"),
        ("Databases", "RDS/Aurora (8/10)", "Cloud SQL (7/10)", "Cosmos DB (6/10)")
    ]

    for category, aws, gcp, azure in comparisons:
        print(f"{category:<20} {aws:<25} {gcp:<25} {azure:<25}")

    print()
    print("=" * 80)
    print("  Key Takeaways")
    print("=" * 80)
    print()
    print("1. AWS leads in breadth and maturity (most services, largest ecosystem)")
    print("2. GCP excels in specific areas (TPUs, BigQuery, Kubernetes)")
    print("3. Azure is strong for Microsoft-centric enterprises")
    print()
    print("Multi-Cloud Best-of-Breed Opportunities:")
    print("  • Train on GCP (TPUs) → Deploy on AWS (broader regions)")
    print("  • Use GCP BigQuery for data warehouse → AWS for everything else")
    print("  • Kubernetes on GKE for best experience → deploy anywhere")
    print()
    print("⚠️  WARNING: Each cross-cloud integration adds operational complexity")
    print("              and data transfer costs. Carefully evaluate ROI.")
    print()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare cloud services across AWS, GCP, and Azure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Service Categories:
  ml         - Managed ML platforms (SageMaker, Vertex AI, Azure ML)
  compute    - GPU compute instances
  storage    - Object storage services
  kubernetes - Managed Kubernetes
  serving    - Model serving and inference
  database   - Managed databases

Examples:
  # Compare ML platforms
  %(prog)s --category ml

  # Compare all categories
  %(prog)s --all

  # Show summary table
  %(prog)s --summary
        """
    )

    parser.add_argument(
        "--category",
        choices=["ml", "compute", "storage", "kubernetes", "serving", "database"],
        help="Service category to compare"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all category comparisons"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary table only"
    )

    args = parser.parse_args()

    if args.summary:
        print_summary_table()
        return 0

    if args.all:
        for category_key in ["ml", "compute", "storage", "kubernetes", "serving", "database"]:
            print_category_comparison(CATEGORIES[category_key])
            print("\n")
        return 0

    if args.category:
        category = CATEGORIES[args.category]
        print_category_comparison(category)
        return 0

    # Default: show summary
    print_summary_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
