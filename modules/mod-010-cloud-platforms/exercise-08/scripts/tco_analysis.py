#!/usr/bin/env python3
"""
Total Cost of Ownership (TCO) Analysis for Multi-Cloud

Calculate 3-year TCO comparing single-cloud vs multi-cloud architectures.
Includes infrastructure costs, operational overhead, and hidden costs.

Usage:
    python tco_analysis.py --scenario baseline
    python tco_analysis.py --scenario active_passive
    python tco_analysis.py --scenario best_of_breed
    python tco_analysis.py --compare
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CostBreakdown:
    """Cost breakdown for a specific category."""
    compute: float
    storage: float
    networking: float
    managed_services: float
    data_transfer: float
    operational_overhead: float
    tooling: float
    training: float
    compliance: float

    @property
    def total(self) -> float:
        """Calculate total cost."""
        return (
            self.compute + self.storage + self.networking +
            self.managed_services + self.data_transfer +
            self.operational_overhead + self.tooling +
            self.training + self.compliance
        )


@dataclass
class TCOScenario:
    """TCO scenario with cost breakdown."""
    name: str
    description: str
    year1_costs: CostBreakdown
    year2_costs: CostBreakdown
    year3_costs: CostBreakdown

    @property
    def total_3_year(self) -> float:
        """Calculate 3-year total."""
        return self.year1_costs.total + self.year2_costs.total + self.year3_costs.total

    @property
    def annual_average(self) -> float:
        """Calculate annual average cost."""
        return self.total_3_year / 3


# ============================================================================
# Scenario Definitions (in thousands of dollars)
# ============================================================================

# Baseline: Single Cloud (AWS)
BASELINE_SCENARIO = TCOScenario(
    name="Single Cloud (AWS)",
    description="Standard single-cloud deployment on AWS",
    year1_costs=CostBreakdown(
        compute=300,           # EC2, EKS, Fargate
        storage=80,            # S3, EBS, EFS
        networking=40,         # VPC, Load Balancers
        managed_services=120,  # RDS, ElastiCache, SageMaker
        data_transfer=30,      # Egress within region
        operational_overhead=200,  # 2 cloud engineers * $100k
        tooling=50,            # Monitoring, security, CI/CD
        training=20,           # AWS certifications
        compliance=60          # SOC2, pen testing
    ),
    year2_costs=CostBreakdown(
        compute=360,           # 20% growth
        storage=100,
        networking=50,
        managed_services=150,
        data_transfer=40,
        operational_overhead=220,  # Salary increases
        tooling=55,
        training=10,           # Reduced (team trained)
        compliance=65
    ),
    year3_costs=CostBreakdown(
        compute=430,           # 20% growth
        storage=120,
        networking=60,
        managed_services=180,
        data_transfer=50,
        operational_overhead=240,
        tooling=60,
        training=10,
        compliance=70
    )
)

# Active-Passive DR (AWS + GCP)
ACTIVE_PASSIVE_SCENARIO = TCOScenario(
    name="Active-Passive DR (AWS + GCP)",
    description="Primary on AWS, DR environment on GCP",
    year1_costs=CostBreakdown(
        compute=450,           # AWS primary + GCP standby
        storage=140,           # Replicated data
        networking=80,         # Cross-cloud connectivity
        managed_services=160,  # Dual cloud services
        data_transfer=100,     # Cross-cloud replication
        operational_overhead=350,  # 3.5 engineers (multi-cloud complexity)
        tooling=80,            # Multi-cloud monitoring
        training=60,           # GCP + multi-cloud patterns
        compliance=90          # Dual compliance audits
    ),
    year2_costs=CostBreakdown(
        compute=540,
        storage=170,
        networking=100,
        managed_services=200,
        data_transfer=130,     # Increased replication
        operational_overhead=380,
        tooling=90,
        training=30,           # Reduced after initial training
        compliance=95
    ),
    year3_costs=CostBreakdown(
        compute=650,
        storage=210,
        networking=120,
        managed_services=240,
        data_transfer=160,
        operational_overhead=420,
        tooling=100,
        training=30,
        compliance=100
    )
)

# Best-of-Breed Multi-Cloud (AWS + GCP + Azure)
BEST_OF_BREED_SCENARIO = TCOScenario(
    name="Best-of-Breed (AWS + GCP + Azure)",
    description="Workloads distributed across all three clouds",
    year1_costs=CostBreakdown(
        compute=420,           # Optimized per cloud
        storage=160,           # Data in multiple clouds
        networking=150,        # High cross-cloud traffic
        managed_services=200,  # Multiple providers
        data_transfer=200,     # Significant egress costs
        operational_overhead=500,  # 5 engineers (high complexity)
        tooling=120,           # Unified observability
        training=100,          # All three clouds
        compliance=120         # Multi-cloud compliance
    ),
    year2_costs=CostBreakdown(
        compute=480,
        storage=200,
        networking=180,
        managed_services=240,
        data_transfer=250,     # Growing egress costs
        operational_overhead=550,
        tooling=130,
        training=40,
        compliance=125
    ),
    year3_costs=CostBreakdown(
        compute=560,
        storage=240,
        networking=220,
        managed_services=280,
        data_transfer=300,
        operational_overhead=600,
        tooling=140,
        training=40,
        compliance=130
    )
)

# Cloud-Agnostic Kubernetes
CLOUD_AGNOSTIC_SCENARIO = TCOScenario(
    name="Cloud-Agnostic Kubernetes",
    description="Portable workloads, can run on any cloud",
    year1_costs=CostBreakdown(
        compute=380,           # Similar to single cloud
        storage=100,
        networking=60,
        managed_services=80,   # Reduced (avoid cloud-specific services)
        data_transfer=40,
        operational_overhead=400,  # 4 engineers (K8s expertise)
        tooling=100,           # Self-managed K8s tools
        training=80,           # K8s + multi-cloud
        compliance=70
    ),
    year2_costs=CostBreakdown(
        compute=460,
        storage=120,
        networking=75,
        managed_services=100,
        data_transfer=50,
        operational_overhead=440,
        tooling=110,
        training=30,
        compliance=75
    ),
    year3_costs=CostBreakdown(
        compute=550,
        storage=150,
        networking=90,
        managed_services=120,
        data_transfer=60,
        operational_overhead=480,
        tooling=120,
        training=30,
        compliance=80
    )
)


SCENARIOS = {
    "baseline": BASELINE_SCENARIO,
    "active_passive": ACTIVE_PASSIVE_SCENARIO,
    "best_of_breed": BEST_OF_BREED_SCENARIO,
    "cloud_agnostic": CLOUD_AGNOSTIC_SCENARIO
}


# ============================================================================
# Analysis Functions
# ============================================================================

def print_scenario_details(scenario: TCOScenario):
    """Print detailed cost breakdown for a scenario."""
    print()
    print("=" * 80)
    print(f"  {scenario.name}")
    print("=" * 80)
    print(f"\n{scenario.description}\n")

    print(f"{'Category':<25} {'Year 1':<12} {'Year 2':<12} {'Year 3':<12} {'3-Yr Total':<12}")
    print("-" * 80)

    categories = [
        ("Compute", "compute"),
        ("Storage", "storage"),
        ("Networking", "networking"),
        ("Managed Services", "managed_services"),
        ("Data Transfer", "data_transfer"),
        ("Operational Overhead", "operational_overhead"),
        ("Tooling", "tooling"),
        ("Training", "training"),
        ("Compliance", "compliance")
    ]

    for label, attr in categories:
        y1 = getattr(scenario.year1_costs, attr)
        y2 = getattr(scenario.year2_costs, attr)
        y3 = getattr(scenario.year3_costs, attr)
        total = y1 + y2 + y3
        print(f"{label:<25} ${y1:>9,.0f}k  ${y2:>9,.0f}k  ${y3:>9,.0f}k  ${total:>9,.0f}k")

    print("-" * 80)
    print(f"{'TOTAL':<25} ${scenario.year1_costs.total:>9,.0f}k  "
          f"${scenario.year2_costs.total:>9,.0f}k  "
          f"${scenario.year3_costs.total:>9,.0f}k  "
          f"${scenario.total_3_year:>9,.0f}k")
    print()
    print(f"Annual Average: ${scenario.annual_average:,.0f}k")
    print()


def compare_scenarios():
    """Compare all scenarios side by side."""
    print()
    print("=" * 80)
    print("  TCO Comparison: Single-Cloud vs Multi-Cloud (3-Year Total)")
    print("=" * 80)
    print()

    baseline = BASELINE_SCENARIO

    scenarios_list = [
        BASELINE_SCENARIO,
        ACTIVE_PASSIVE_SCENARIO,
        CLOUD_AGNOSTIC_SCENARIO,
        BEST_OF_BREED_SCENARIO
    ]

    # Sort by total cost
    scenarios_list.sort(key=lambda s: s.total_3_year)

    print(f"{'Architecture':<35} {'3-Yr TCO':<15} {'vs Baseline':<15} {'$/Month':<12}")
    print("-" * 80)

    for scenario in scenarios_list:
        cost_diff = scenario.total_3_year - baseline.total_3_year
        cost_diff_pct = (cost_diff / baseline.total_3_year * 100) if scenario != baseline else 0
        monthly = scenario.total_3_year / 36

        if scenario == baseline:
            comparison = "(baseline)"
        elif cost_diff > 0:
            comparison = f"+${cost_diff:,.0f}k (+{cost_diff_pct:.1f}%)"
        else:
            comparison = f"-${abs(cost_diff):,.0f}k ({cost_diff_pct:.1f}%)"

        print(f"{scenario.name:<35} ${scenario.total_3_year:>12,.0f}k  "
              f"{comparison:<15} ${monthly:>9,.0f}k")

    print()
    print("=" * 80)
    print("  Key Findings")
    print("=" * 80)
    print()

    # Calculate differences
    ap_diff = ACTIVE_PASSIVE_SCENARIO.total_3_year - baseline.total_3_year
    bb_diff = BEST_OF_BREED_SCENARIO.total_3_year - baseline.total_3_year
    ca_diff = CLOUD_AGNOSTIC_SCENARIO.total_3_year - baseline.total_3_year

    print(f"1. Active-Passive DR costs {ap_diff / baseline.total_3_year * 100:.0f}% more")
    print(f"   • Justification: Regulatory compliance, business continuity")
    print(f"   • Break-even: If downtime cost > ${ap_diff / 3:,.0f}k/year")
    print()

    print(f"2. Best-of-Breed costs {bb_diff / baseline.total_3_year * 100:.0f}% more")
    print(f"   • Justification: Cost optimization at massive scale")
    print(f"   • Break-even: Need ${bb_diff:,.0f}k in savings over 3 years")
    print(f"   • Realistic only for companies spending $5M+/year on cloud")
    print()

    print(f"3. Cloud-Agnostic costs {ca_diff / baseline.total_3_year * 100:.0f}% more")
    print(f"   • Justification: Portability requirements, customer deployments")
    print(f"   • Trade-off: Higher ops cost, but strategic flexibility")
    print()

    print("4. Hidden cost drivers:")

    # Operational overhead comparison
    baseline_ops = baseline.year1_costs.operational_overhead + baseline.year2_costs.operational_overhead + baseline.year3_costs.operational_overhead
    ap_ops = ACTIVE_PASSIVE_SCENARIO.year1_costs.operational_overhead + ACTIVE_PASSIVE_SCENARIO.year2_costs.operational_overhead + ACTIVE_PASSIVE_SCENARIO.year3_costs.operational_overhead
    ops_diff = ap_ops - baseline_ops

    print(f"   • Operational overhead: +${ops_diff:,.0f}k over 3 years")
    print(f"     (Equivalent to {ops_diff / 300:.1f} additional engineers)")

    # Data transfer comparison
    baseline_transfer = baseline.year1_costs.data_transfer + baseline.year2_costs.data_transfer + baseline.year3_costs.data_transfer
    bb_transfer = BEST_OF_BREED_SCENARIO.year1_costs.data_transfer + BEST_OF_BREED_SCENARIO.year2_costs.data_transfer + BEST_OF_BREED_SCENARIO.year3_costs.data_transfer
    transfer_diff = bb_transfer - baseline_transfer

    print(f"   • Cross-cloud data transfer: +${transfer_diff:,.0f}k")
    print(f"     (Can exceed compute costs at scale)")

    print()


def calculate_breakeven_savings():
    """Calculate required cloud savings to justify multi-cloud complexity."""
    print()
    print("=" * 80)
    print("  Break-Even Analysis: When Does Multi-Cloud Pay Off?")
    print("=" * 80)
    print()

    baseline = BASELINE_SCENARIO.total_3_year
    best_of_breed = BEST_OF_BREED_SCENARIO.total_3_year

    # Additional cost of multi-cloud
    additional_cost = best_of_breed - baseline

    print(f"Additional 3-year cost of multi-cloud: ${additional_cost:,.0f}k")
    print()

    print("To break even, you must save more than this through:")
    print("  1. Workload optimization (right cloud for each workload)")
    print("  2. Spot instance arbitrage")
    print("  3. Reserved instance optimization")
    print("  4. Negotiated discounts (cloud competition)")
    print()

    # Required savings by year
    print("Required Annual Savings by Current Spend:")
    print()
    print(f"{'Current Cloud Spend':<25} {'Required Savings':<20} {'% Reduction Needed'}")
    print("-" * 70)

    annual_additional = additional_cost / 3

    for current_spend in [1000, 2000, 5000, 10000]:
        required_pct = (annual_additional / current_spend) * 100
        print(f"${current_spend:>6,.0f}k/year {'':<12} ${annual_additional:>6,.0f}k/year {'':<8} {required_pct:>6.1f}%")

    print()
    print("Reality Check:")
    print("  • Companies spending <$2M/year: Multi-cloud rarely justifiable")
    print("  • Companies spending $2-5M/year: Possible but challenging")
    print("  • Companies spending >$5M/year: More realistic ROI opportunity")
    print()
    print("Key insight: Operational overhead is FIXED, infrastructure savings are VARIABLE")
    print("             Multi-cloud economics only work at large scale")
    print()


def show_hidden_costs():
    """Highlight often-overlooked costs in multi-cloud."""
    print()
    print("=" * 80)
    print("  Hidden Costs in Multi-Cloud Architectures")
    print("=" * 80)
    print()

    print("1. DATA EGRESS COSTS (often underestimated)")
    print("   • AWS → GCP: $0.09/GB")
    print("   • Transferring 10TB/month: $900/month = $10.8k/year")
    print("   • At scale (100TB/month): $108k/year")
    print("   • ⚠️  Can exceed compute costs for data-intensive ML pipelines")
    print()

    print("2. OPERATIONAL OVERHEAD")
    print("   • Single cloud: 2-3 engineers can manage infrastructure")
    print("   • Multi-cloud: 4-6 engineers needed")
    print("   • Additional cost: $200-400k/year in salaries")
    print("   • Plus: Slower velocity, more context switching")
    print()

    print("3. TOOLING COMPLEXITY")
    print("   • Unified observability: Datadog multi-cloud = $50-100k/year")
    print("   • Security: Prisma Cloud/Wiz multi-cloud = $50-80k/year")
    print("   • Cost management: CloudHealth/Cloudability = $20-40k/year")
    print("   • CI/CD: Jenkins/GitLab multi-cloud runners = $20-30k/year")
    print("   • Total: $140-250k/year vs $50-80k single cloud")
    print()

    print("4. TRAINING & ONBOARDING")
    print("   • AWS certifications: $300/person")
    print("   • GCP certifications: $200/person")
    print("   • Azure certifications: $165/person")
    print("   • 10-person team, all 3 clouds: $6,650")
    print("   • Plus: 2-3 months reduced productivity during ramp-up")
    print()

    print("5. OPPORTUNITY COST")
    print("   • Time spent managing multi-cloud complexity")
    print("   • NOT spent building features")
    print("   • Estimate: 20-30% engineering capacity for multi-cloud team")
    print("   • 5 engineers * 30% * $150k = $225k/year opportunity cost")
    print()

    print("6. COMPLIANCE & AUDIT")
    print("   • Single cloud SOC2 audit: $30-60k/year")
    print("   • Multi-cloud SOC2 audit: $60-100k/year")
    print("   • Pen testing: +50% cost for multi-cloud scope")
    print()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Total Cost of Ownership (TCO) analysis for multi-cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  baseline        - Single cloud (AWS) baseline
  active_passive  - Active-Passive DR (AWS + GCP)
  best_of_breed   - Best-of-breed multi-cloud (AWS + GCP + Azure)
  cloud_agnostic  - Cloud-agnostic Kubernetes

Examples:
  # Detailed view of single scenario
  %(prog)s --scenario baseline

  # Compare all scenarios
  %(prog)s --compare

  # Calculate break-even point
  %(prog)s --breakeven

  # Show hidden costs
  %(prog)s --hidden-costs
        """
    )

    parser.add_argument(
        "--scenario",
        choices=["baseline", "active_passive", "best_of_breed", "cloud_agnostic"],
        help="Show detailed breakdown for specific scenario"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all scenarios"
    )
    parser.add_argument(
        "--breakeven",
        action="store_true",
        help="Calculate break-even savings required"
    )
    parser.add_argument(
        "--hidden-costs",
        action="store_true",
        help="Show often-overlooked costs"
    )

    args = parser.parse_args()

    if args.scenario:
        scenario = SCENARIOS[args.scenario]
        print_scenario_details(scenario)
        return 0

    if args.compare:
        compare_scenarios()
        return 0

    if args.breakeven:
        calculate_breakeven_savings()
        return 0

    if args.hidden_costs:
        show_hidden_costs()
        return 0

    # Default: show comparison
    compare_scenarios()
    return 0


if __name__ == "__main__":
    sys.exit(main())
