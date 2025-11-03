#!/usr/bin/env python3
"""
Cloud Storage Cost Calculator

Calculate and compare storage costs across AWS S3, GCP Cloud Storage, and Azure Blob.
Critical for understanding data transfer costs in multi-cloud architectures.

Usage:
    python storage_cost_calculator.py --storage 100 --egress 10 --provider aws
    python storage_cost_calculator.py --storage 1000 --egress 500 --compare
    python storage_cost_calculator.py --ml-scenario training
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Dict


@dataclass
class StoragePricing:
    """Storage pricing for a cloud provider."""
    provider: str
    storage_per_gb: float  # $/GB/month
    egress_per_gb: float   # $/GB
    api_requests_per_1000: float  # $/1000 requests
    region: str


# ============================================================================
# Pricing Data (as of 2024, US regions)
# ============================================================================

AWS_S3_PRICING = StoragePricing(
    provider="AWS S3",
    storage_per_gb=0.023,  # Standard storage
    egress_per_gb=0.09,    # Data transfer out to internet
    api_requests_per_1000=0.0004,  # GET requests
    region="us-east-1"
)

GCP_CLOUD_STORAGE_PRICING = StoragePricing(
    provider="GCP Cloud Storage",
    storage_per_gb=0.020,  # Standard storage
    egress_per_gb=0.12,    # Data transfer out to internet (>10TB/month)
    api_requests_per_1000=0.0004,  # Class A operations
    region="us-central1"
)

AZURE_BLOB_PRICING = StoragePricing(
    provider="Azure Blob Storage",
    storage_per_gb=0.0184,  # Hot tier
    egress_per_gb=0.087,    # Data transfer out (first 10TB)
    api_requests_per_1000=0.0044,  # Write operations
    region="East US"
)


@dataclass
class StorageCost:
    """Cost breakdown for storage."""
    provider: str
    storage_cost: float
    egress_cost: float
    api_cost: float
    total_monthly: float
    total_annual: float


def calculate_storage_cost(
    storage_gb: float,
    egress_gb: float,
    api_requests: int,
    pricing: StoragePricing
) -> StorageCost:
    """
    Calculate monthly storage costs.

    Args:
        storage_gb: Total storage in GB
        egress_gb: Data egress per month in GB
        api_requests: Number of API requests per month
        pricing: Provider pricing structure

    Returns:
        StorageCost with breakdown
    """
    storage_cost = storage_gb * pricing.storage_per_gb
    egress_cost = egress_gb * pricing.egress_per_gb
    api_cost = (api_requests / 1000) * pricing.api_requests_per_1000

    total_monthly = storage_cost + egress_cost + api_cost
    total_annual = total_monthly * 12

    return StorageCost(
        provider=pricing.provider,
        storage_cost=storage_cost,
        egress_cost=egress_cost,
        api_cost=api_cost,
        total_monthly=total_monthly,
        total_annual=total_annual
    )


def print_cost_breakdown(cost: StorageCost):
    """Print detailed cost breakdown."""
    print()
    print("=" * 70)
    print(f"  {cost.provider}")
    print("=" * 70)
    print()
    print(f"Storage Cost:        ${cost.storage_cost:>12,.2f}/month")
    print(f"Data Egress Cost:    ${cost.egress_cost:>12,.2f}/month")
    print(f"API Request Cost:    ${cost.api_cost:>12,.2f}/month")
    print("-" * 70)
    print(f"Total Monthly:       ${cost.total_monthly:>12,.2f}")
    print(f"Total Annual:        ${cost.total_annual:>12,.2f}")
    print()


def compare_all_providers(storage_gb: float, egress_gb: float, api_requests: int):
    """Compare costs across all three providers."""
    print()
    print("=" * 70)
    print("  Cloud Storage Cost Comparison")
    print("=" * 70)
    print()
    print(f"Storage:        {storage_gb:>10,.0f} GB")
    print(f"Data Egress:    {egress_gb:>10,.0f} GB/month")
    print(f"API Requests:   {api_requests:>10,} requests/month")
    print()

    # Calculate costs for all providers
    aws_cost = calculate_storage_cost(storage_gb, egress_gb, api_requests, AWS_S3_PRICING)
    gcp_cost = calculate_storage_cost(storage_gb, egress_gb, api_requests, GCP_CLOUD_STORAGE_PRICING)
    azure_cost = calculate_storage_cost(storage_gb, egress_gb, api_requests, AZURE_BLOB_PRICING)

    costs = [aws_cost, gcp_cost, azure_cost]

    # Print comparison table
    print(f"{'Provider':<25} {'Storage':<15} {'Egress':<15} {'API':<12} {'Monthly Total':<15}")
    print("-" * 82)

    for cost in costs:
        print(f"{cost.provider:<25} ${cost.storage_cost:>12,.2f}  "
              f"${cost.egress_cost:>12,.2f}  ${cost.api_cost:>9,.2f}  "
              f"${cost.total_monthly:>12,.2f}")

    print()

    # Find cheapest
    cheapest = min(costs, key=lambda c: c.total_monthly)
    most_expensive = max(costs, key=lambda c: c.total_monthly)

    print(f"✓ Cheapest: {cheapest.provider} at ${cheapest.total_monthly:,.2f}/month")
    print(f"✗ Most Expensive: {most_expensive.provider} at ${most_expensive.total_monthly:,.2f}/month")

    difference = most_expensive.total_monthly - cheapest.total_monthly
    pct_diff = (difference / cheapest.total_monthly) * 100

    print(f"\nCost Difference: ${difference:,.2f}/month ({pct_diff:.1f}% more expensive)")
    print(f"Annual Difference: ${difference * 12:,.2f}")
    print()

    # Show egress as % of total
    print("=" * 70)
    print("  Cost Breakdown Analysis")
    print("=" * 70)
    print()

    for cost in costs:
        egress_pct = (cost.egress_cost / cost.total_monthly) * 100
        storage_pct = (cost.storage_cost / cost.total_monthly) * 100

        print(f"{cost.provider}:")
        print(f"  • Storage: {storage_pct:>5.1f}% (${cost.storage_cost:,.2f})")
        print(f"  • Egress:  {egress_pct:>5.1f}% (${cost.egress_cost:,.2f})")

        if egress_pct > 50:
            print(f"  ⚠️  WARNING: Egress costs exceed storage costs!")
        print()


def analyze_multi_cloud_egress():
    """Analyze cross-cloud data transfer costs."""
    print()
    print("=" * 70)
    print("  Multi-Cloud Data Transfer Cost Analysis")
    print("=" * 70)
    print()

    print("Cross-Cloud Data Transfer Pricing:")
    print("  • AWS → GCP:    $0.09/GB (from AWS)")
    print("  • GCP → AWS:    $0.12/GB (from GCP)")
    print("  • AWS → Azure:  $0.09/GB (from AWS)")
    print("  • Azure → AWS:  $0.087/GB (from Azure)")
    print()

    # Example scenarios
    print("Example: Transfer 10TB Dataset Between Clouds")
    print("-" * 70)

    dataset_sizes_gb = [1000, 10000, 100000]  # 1TB, 10TB, 100TB

    print(f"{'Dataset Size':<20} {'AWS→GCP':<15} {'GCP→AWS':<15} {'Round Trip':<15}")
    print("-" * 70)

    for size_gb in dataset_sizes_gb:
        aws_to_gcp = size_gb * 0.09
        gcp_to_aws = size_gb * 0.12
        round_trip = aws_to_gcp + gcp_to_aws

        size_label = f"{size_gb / 1000:.0f}TB" if size_gb >= 1000 else f"{size_gb}GB"

        print(f"{size_label:<20} ${aws_to_gcp:>12,.0f}  ${gcp_to_aws:>12,.0f}  ${round_trip:>12,.0f}")

    print()
    print("Key Insights:")
    print("  • Transferring 100TB costs $9,000 (one-way)")
    print("  • At ML scale, this can exceed compute costs")
    print("  • Keep data and compute in same cloud")
    print("  • Multi-cloud data sync is EXPENSIVE")
    print()

    print("=" * 70)
    print("  ML Training Data Transfer Example")
    print("=" * 70)
    print()

    print("Scenario: Train on GCP, Deploy on AWS")
    print()
    print("Dataset: 50TB training data")
    print("Model: 5GB model file")
    print("Deployment: 100 inference endpoints (AWS regions)")
    print()

    training_data_transfer = 50000 * 0.09  # AWS→GCP for initial copy
    model_sync_cost = 5 * 0.12 * 100  # GCP→AWS for each endpoint

    print(f"One-time data transfer (AWS→GCP): ${training_data_transfer:,.0f}")
    print(f"Model deployment (GCP→AWS×100):   ${model_sync_cost:,.0f}")
    print(f"Total Transfer Cost:              ${training_data_transfer + model_sync_cost:,.0f}")
    print()
    print("⚠️  This is why most companies keep training & inference in same cloud")
    print()


def ml_scenario_analysis(scenario: str):
    """Analyze storage costs for common ML scenarios."""
    scenarios = {
        "training": {
            "name": "ML Model Training",
            "description": "Large dataset storage, frequent access during training",
            "storage_gb": 50000,  # 50TB dataset
            "egress_gb": 0,       # No egress (training in same cloud)
            "api_requests": 10000000  # Frequent reads during training
        },
        "inference": {
            "name": "Model Inference (Multi-Region)",
            "description": "Model artifacts replicated to multiple regions",
            "storage_gb": 500,    # 500GB models
            "egress_gb": 2000,    # Serving traffic to clients
            "api_requests": 50000000  # High request rate
        },
        "data_lake": {
            "name": "ML Data Lake",
            "description": "Archived datasets, infrequent access",
            "storage_gb": 500000,  # 500TB archived data
            "egress_gb": 100,      # Occasional analysis
            "api_requests": 100000  # Low request rate
        },
        "multi_cloud": {
            "name": "Multi-Cloud ML Pipeline",
            "description": "Training on GCP, inference on AWS (cross-cloud sync)",
            "storage_gb": 50000,   # 50TB on each cloud
            "egress_gb": 10000,    # 10TB/month cross-cloud sync
            "api_requests": 5000000
        }
    }

    if scenario not in scenarios:
        print(f"Error: Unknown scenario '{scenario}'")
        return

    config = scenarios[scenario]

    print()
    print("=" * 70)
    print(f"  Scenario: {config['name']}")
    print("=" * 70)
    print(f"\n{config['description']}\n")

    compare_all_providers(
        storage_gb=config["storage_gb"],
        egress_gb=config["egress_gb"],
        api_requests=config["api_requests"]
    )

    # Scenario-specific insights
    if scenario == "multi_cloud":
        print("=" * 70)
        print("  Multi-Cloud Impact")
        print("=" * 70)
        print()
        print("⚠️  This scenario shows the hidden cost of multi-cloud:")
        print()
        egress_cost = config["egress_gb"] * 0.09
        print(f"  • Cross-cloud data transfer: ${egress_cost:,.0f}/month")
        print(f"  • Annual egress cost: ${egress_cost * 12:,.0f}")
        print()
        print("  For many companies, this egress cost alone makes multi-cloud")
        print("  uneconomical for data-intensive ML workloads.")
        print()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and compare cloud storage costs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate AWS S3 cost for 100GB storage, 10GB egress
  %(prog)s --storage 100 --egress 10 --provider aws

  # Compare all providers
  %(prog)s --storage 1000 --egress 500 --compare

  # Analyze multi-cloud data transfer costs
  %(prog)s --multi-cloud-egress

  # ML scenario analysis
  %(prog)s --ml-scenario training
  %(prog)s --ml-scenario inference
  %(prog)s --ml-scenario data_lake
  %(prog)s --ml-scenario multi_cloud
        """
    )

    parser.add_argument(
        "--storage",
        type=float,
        help="Total storage in GB"
    )
    parser.add_argument(
        "--egress",
        type=float,
        default=0,
        help="Data egress per month in GB"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=1000000,
        help="API requests per month (default: 1 million)"
    )
    parser.add_argument(
        "--provider",
        choices=["aws", "gcp", "azure"],
        help="Calculate for specific provider"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all three providers"
    )
    parser.add_argument(
        "--multi-cloud-egress",
        action="store_true",
        help="Analyze cross-cloud data transfer costs"
    )
    parser.add_argument(
        "--ml-scenario",
        choices=["training", "inference", "data_lake", "multi_cloud"],
        help="Analyze specific ML scenario"
    )

    args = parser.parse_args()

    if args.multi_cloud_egress:
        analyze_multi_cloud_egress()
        return 0

    if args.ml_scenario:
        ml_scenario_analysis(args.ml_scenario)
        return 0

    if not args.storage:
        parser.print_help()
        return 1

    if args.compare:
        compare_all_providers(args.storage, args.egress, args.requests)
        return 0

    if args.provider:
        pricing_map = {
            "aws": AWS_S3_PRICING,
            "gcp": GCP_CLOUD_STORAGE_PRICING,
            "azure": AZURE_BLOB_PRICING
        }
        pricing = pricing_map[args.provider]
        cost = calculate_storage_cost(args.storage, args.egress, args.requests, pricing)
        print_cost_breakdown(cost)
        return 0

    # Default: compare all
    compare_all_providers(args.storage, args.egress, args.requests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
