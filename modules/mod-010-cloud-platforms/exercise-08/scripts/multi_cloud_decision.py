#!/usr/bin/env python3
"""
Multi-Cloud Decision Framework

Evaluates whether multi-cloud makes sense for your organization
based on specific priorities and constraints.

Usage:
    python multi_cloud_decision.py --scenario startup
    python multi_cloud_decision.py --scenario enterprise
    python multi_cloud_decision.py --scenario platform
    python multi_cloud_decision.py --custom
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Dict


@dataclass
class CloudStrategy:
    """Cloud strategy with evaluation metrics."""
    name: str
    complexity: int  # 1-10 (10 = most complex)
    cost_optimization: int  # 1-10 (10 = best cost optimization)
    risk_mitigation: int  # 1-10 (10 = best risk mitigation)
    vendor_lock_in: int  # 1-10 (10 = no lock-in)
    operational_overhead: int  # 1-10 (10 = highest overhead)
    time_to_market: int  # 1-10 (10 = fastest)

    def score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted score based on priorities."""
        return (
            self.complexity * weights.get("complexity", 0) +
            self.cost_optimization * weights.get("cost", 0) +
            self.risk_mitigation * weights.get("risk", 0) +
            self.vendor_lock_in * weights.get("portability", 0) +
            self.operational_overhead * weights.get("operations", 0) +
            self.time_to_market * weights.get("speed", 0)
        )


# ============================================================================
# Strategy Definitions
# ============================================================================

STRATEGIES = {
    "single_cloud": CloudStrategy(
        name="Single Cloud (AWS)",
        complexity=2,
        cost_optimization=5,
        risk_mitigation=3,
        vendor_lock_in=1,
        operational_overhead=3,
        time_to_market=9
    ),
    "active_passive_dr": CloudStrategy(
        name="Active-Passive DR (AWS + GCP)",
        complexity=6,
        cost_optimization=4,
        risk_mitigation=9,
        vendor_lock_in=6,
        operational_overhead=7,
        time_to_market=5
    ),
    "best_of_breed": CloudStrategy(
        name="Best-of-Breed (AWS + GCP + Azure)",
        complexity=9,
        cost_optimization=8,
        risk_mitigation=6,
        vendor_lock_in=8,
        operational_overhead=10,
        time_to_market=3
    ),
    "cloud_agnostic_k8s": CloudStrategy(
        name="Cloud-Agnostic Kubernetes",
        complexity=8,
        cost_optimization=6,
        risk_mitigation=7,
        vendor_lock_in=10,
        operational_overhead=8,
        time_to_market=4
    )
}


# ============================================================================
# Scenario Definitions
# ============================================================================

SCENARIOS = {
    "startup": {
        "name": "Startup (Focus on Speed)",
        "description": "Small team, fast iteration, minimize complexity",
        "weights": {
            "complexity": -3,      # Avoid complexity
            "cost": 2,             # Cost matters but not critical
            "risk": 1,             # Risk tolerance high
            "portability": 1,      # Not a priority
            "operations": -4,      # Minimize ops overhead
            "speed": 5             # Speed is everything
        },
        "context": {
            "team_size": 10,
            "cloud_expertise": "Medium",
            "revenue_stage": "Pre-revenue or early",
            "compliance": "Low"
        }
    },
    "enterprise": {
        "name": "Enterprise (Risk Mitigation)",
        "description": "Large team, compliance requirements, DR needed",
        "weights": {
            "complexity": -1,      # Can handle complexity
            "cost": 2,             # Cost conscious
            "risk": 5,             # Risk mitigation critical
            "portability": 3,      # Want options
            "operations": -2,      # Have ops team
            "speed": 2             # Stability over speed
        },
        "context": {
            "team_size": 100,
            "cloud_expertise": "High",
            "revenue_stage": "$100M+ ARR",
            "compliance": "High (SOC2, ISO)"
        }
    },
    "platform": {
        "name": "Platform Company (Portability)",
        "description": "Deploy to customer clouds, avoid lock-in",
        "weights": {
            "complexity": -2,      # Accept complexity
            "cost": 2,             # Optimize where possible
            "risk": 2,             # Moderate risk tolerance
            "portability": 6,      # Critical requirement
            "operations": -2,      # Have strong ops team
            "speed": 3             # Balanced
        },
        "context": {
            "team_size": 50,
            "cloud_expertise": "Very High",
            "revenue_stage": "$10-100M ARR",
            "compliance": "Medium"
        }
    },
    "cost_optimized": {
        "name": "Cost-Optimized (Mature ML Company)",
        "description": "Large scale, proven workloads, optimize costs",
        "weights": {
            "complexity": -1,      # Can manage
            "cost": 5,             # Primary driver
            "risk": 3,             # Moderate risk tolerance
            "portability": 4,      # Want flexibility
            "operations": -3,      # Have experienced team
            "speed": 2             # Stability preferred
        },
        "context": {
            "team_size": 200,
            "cloud_expertise": "Expert",
            "revenue_stage": "$100M+ ARR",
            "compliance": "High"
        }
    }
}


def evaluate_strategies(scenario_name: str):
    """Evaluate all strategies for a given scenario."""
    scenario = SCENARIOS[scenario_name]

    print("=" * 70)
    print(f"  {scenario['name']}")
    print("=" * 70)
    print(f"\n{scenario['description']}\n")
    print("Context:")
    for key, value in scenario["context"].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")

    print("\nPriorities:")
    weights = scenario["weights"]
    for key, value in weights.items():
        direction = "↑" if value > 0 else "↓"
        importance = abs(value)
        print(f"  {direction} {key.title()}: {importance}/5")

    print("\n" + "-" * 70)
    print("Strategy Evaluation:")
    print("-" * 70)
    print()

    scores = {}
    for key, strategy in STRATEGIES.items():
        score = strategy.score(weights)
        scores[key] = score

        print(f"{strategy.name}:")
        print(f"  Complexity:         {strategy.complexity}/10")
        print(f"  Cost Optimization:  {strategy.cost_optimization}/10")
        print(f"  Risk Mitigation:    {strategy.risk_mitigation}/10")
        print(f"  Vendor Independence: {strategy.vendor_lock_in}/10")
        print(f"  Operational Overhead: {strategy.operational_overhead}/10")
        print(f"  Time to Market:     {strategy.time_to_market}/10")
        print(f"  WEIGHTED SCORE:     {score:.1f}")
        print()

    # Find winner
    winner_key = max(scores, key=scores.get)
    winner = STRATEGIES[winner_key]

    print("=" * 70)
    print(f"  ✓ RECOMMENDED: {winner.name}")
    print("=" * 70)
    print()

    # Add reasoning
    if scenario_name == "startup":
        print("Reasoning:")
        print("  • Small teams should minimize operational complexity")
        print("  • Focus resources on product, not infrastructure")
        print("  • Can migrate to multi-cloud later if needed")
        print("  • AWS has largest ecosystem for hiring")
        print()
        print("Next Steps:")
        print("  1. Choose AWS (or GCP if team has expertise)")
        print("  2. Use Kubernetes for future portability")
        print("  3. Revisit decision at 50+ engineers")

    elif scenario_name == "enterprise":
        print("Reasoning:")
        print("  • Regulatory requirements justify DR investment")
        print("  • Team size can handle operational complexity")
        print("  • Different clouds provide uncorrelated failures")
        print("  • Can justify 30% cost premium for risk mitigation")
        print()
        print("Next Steps:")
        print("  1. Implement Active-Passive DR architecture")
        print("  2. Choose AWS primary + GCP/Azure secondary")
        print("  3. Test failover procedures monthly")
        print("  4. Budget for 3-5 dedicated cloud engineers")

    elif scenario_name == "platform":
        print("Reasoning:")
        print("  • Portability is business requirement")
        print("  • Kubernetes provides cloud abstraction")
        print("  • Can deploy to customer environments")
        print("  • Team has expertise to manage complexity")
        print()
        print("Next Steps:")
        print("  1. Build on cloud-agnostic Kubernetes")
        print("  2. Use Terraform for infrastructure as code")
        print("  3. Avoid cloud-specific managed services")
        print("  4. Establish multi-cloud CI/CD pipeline")

    elif scenario_name == "cost_optimized":
        print("Reasoning:")
        print("  • Large scale justifies cost optimization effort")
        print("  • Proven workloads can move between clouds")
        print("  • Team expertise can manage complexity")
        print("  • Must validate savings exceed operational costs")
        print()
        print("Next Steps:")
        print("  1. Start with pilot: Move training to GCP (TPUs)")
        print("  2. Measure actual costs for 3 months")
        print("  3. Expand only if savings >20%")
        print("  4. Monitor egress costs closely")

    print()


def quick_assessment():
    """Quick interactive assessment."""
    print("=" * 70)
    print("  Quick Multi-Cloud Assessment")
    print("=" * 70)
    print()

    # Ask questions
    print("Answer the following questions (1-10 scale):\n")

    try:
        team_size = int(input("1. Team size (number of engineers): "))
        has_dr = input("2. Do you have DR/compliance requirements? (y/n): ").lower() == 'y'
        expertise = int(input("3. Cloud expertise level (1-10): "))
        cost_pressure = int(input("4. Cost optimization pressure (1-10): "))
        portability = int(input("5. Need for portability (1-10): "))

        print("\n" + "=" * 70)
        print("  Assessment Results")
        print("=" * 70)
        print()

        score = 0
        reasons = []

        # Scoring logic
        if team_size > 100:
            score += 3
            reasons.append("✓ Large team can handle complexity")
        elif team_size > 50:
            score += 2
            reasons.append("✓ Medium team, multi-cloud possible")
        else:
            reasons.append("✗ Small team - complexity overhead high")

        if has_dr:
            score += 4
            reasons.append("✓ DR requirement justifies multi-cloud")

        if expertise >= 8:
            score += 3
            reasons.append("✓ High expertise reduces risk")
        elif expertise >= 5:
            score += 1
            reasons.append("⚠ Medium expertise - proceed carefully")
        else:
            reasons.append("✗ Low expertise - not recommended")

        if cost_pressure >= 8:
            score += 2
            reasons.append("✓ High cost pressure may justify effort")

        if portability >= 8:
            score += 3
            reasons.append("✓ Portability requirement is clear")

        # Print results
        for reason in reasons:
            print(f"  {reason}")

        print()
        print(f"Total Score: {score}/15")
        print()

        if score >= 10:
            print("RECOMMENDATION: ✅ Multi-cloud makes sense")
            print()
            print("Suggested approach: Active-Passive DR or Best-of-Breed")
            print("Next step: Complete full TCO analysis")
        elif score >= 6:
            print("RECOMMENDATION: ⚠️  Consider multi-cloud carefully")
            print()
            print("Suggested approach: Start with pilot project")
            print("Next step: Evaluate one workload on second cloud")
        else:
            print("RECOMMENDATION: ❌ Stay single-cloud")
            print()
            print("Rationale: Costs likely exceed benefits")
            print("Next step: Master one cloud first, revisit in 12 months")

        print()

    except (ValueError, KeyboardInterrupt):
        print("\nAssessment cancelled.")
        return


def main():
    parser = argparse.ArgumentParser(
        description="Multi-cloud decision framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  startup       - Small team, focus on speed
  enterprise    - Large team, compliance/DR requirements
  platform      - Need to deploy to customer clouds
  cost_optimized - Mature ML company optimizing costs

Examples:
  %(prog)s --scenario startup
  %(prog)s --scenario enterprise
  %(prog)s --custom  # Interactive assessment
        """
    )

    parser.add_argument(
        "--scenario",
        choices=["startup", "enterprise", "platform", "cost_optimized"],
        help="Predefined scenario to evaluate"
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Run interactive assessment"
    )

    args = parser.parse_args()

    if args.custom:
        quick_assessment()
        return 0

    if not args.scenario:
        parser.print_help()
        return 1

    evaluate_strategies(args.scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
