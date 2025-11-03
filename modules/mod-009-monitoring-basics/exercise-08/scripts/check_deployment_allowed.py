#!/usr/bin/env python3
"""
Error Budget Policy Enforcement

Automated policy checks to determine if deployments are allowed
based on current error budget status.

Usage:
    python check_deployment_allowed.py --service model-api --budget 0.85 --type feature
    python check_deployment_allowed.py --service model-api --budget 0.05 --type bugfix
    python check_deployment_allowed.py --service model-api --budget -0.10 --type security
"""

import argparse
import sys
from enum import Enum
from dataclasses import dataclass
from typing import List


class BudgetZone(Enum):
    """Error budget zones with associated policies."""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class PolicyAction:
    """Policy actions for a given budget zone."""
    zone: BudgetZone
    can_deploy: bool
    can_new_features: bool
    approval_required: str
    notification_required: List[str]
    restrictions: List[str]
    recommendations: List[str]


# ============================================================================
# Policy Definitions
# ============================================================================

GREEN_POLICY = PolicyAction(
    zone=BudgetZone.GREEN,
    can_deploy=True,
    can_new_features=True,
    approval_required="standard",
    notification_required=[],
    restrictions=[],
    recommendations=[
        "Normal feature development pace",
        "Good time for experiments and refactoring",
        "Can take calculated risks",
        "Deploy during business hours acceptable"
    ]
)

YELLOW_POLICY = PolicyAction(
    zone=BudgetZone.YELLOW,
    can_deploy=True,
    can_new_features=True,
    approval_required="senior_engineer",
    notification_required=["engineering_manager"],
    restrictions=[
        "No risky architectural changes",
        "Increased testing requirements",
        "Deploy outside business hours preferred"
    ],
    recommendations=[
        "Elevated caution",
        "Prioritize stability over new features",
        "Monitor error budget closely"
    ]
)

ORANGE_POLICY = PolicyAction(
    zone=BudgetZone.ORANGE,
    can_deploy=True,
    can_new_features=False,
    approval_required="engineering_manager",
    notification_required=["engineering_manager", "vp_engineering"],
    restrictions=[
        "NO new features",
        "Bug fixes and critical changes only",
        "Comprehensive testing mandatory",
        "Deploy only during low-traffic windows"
    ],
    recommendations=[
        "Focus on reliability",
        "Defer all non-critical work",
        "Plan reliability improvements"
    ]
)

RED_POLICY = PolicyAction(
    zone=BudgetZone.RED,
    can_deploy=False,
    can_new_features=False,
    approval_required="vp_engineering",
    notification_required=["engineering_manager", "vp_engineering", "ceo"],
    restrictions=[
        "COMPLETE FEATURE FREEZE",
        "No deploys except critical fixes",
        "All hands on reliability",
        "Incident post-mortem required"
    ],
    recommendations=[
        "Immediate root cause analysis",
        "Stop all feature work",
        "Focus entirely on reliability",
        "Executive team alignment needed"
    ]
)


# ============================================================================
# Policy Functions
# ============================================================================

def get_policy_for_budget(budget_remaining: float) -> PolicyAction:
    """
    Determine policy actions based on remaining error budget.

    Args:
        budget_remaining: Remaining error budget (0.0 to 1.0)
                         Can be negative if budget exhausted

    Returns:
        PolicyAction with restrictions and requirements
    """
    if budget_remaining >= 0.75:
        return GREEN_POLICY
    elif budget_remaining >= 0.25:
        return YELLOW_POLICY
    elif budget_remaining >= 0:
        return ORANGE_POLICY
    else:  # < 0
        return RED_POLICY


def get_zone_emoji(zone: BudgetZone) -> str:
    """Get emoji for visual zone indication."""
    return {
        BudgetZone.GREEN: "🟢",
        BudgetZone.YELLOW: "🟡",
        BudgetZone.ORANGE: "🟠",
        BudgetZone.RED: "🔴"
    }[zone]


def check_deployment_allowed(
    service: str,
    budget_remaining: float,
    change_type: str
) -> tuple[bool, str]:
    """
    Check if deployment is allowed based on error budget policy.

    Args:
        service: Service name
        budget_remaining: Current error budget remaining (-1 to 1)
        change_type: "feature", "bugfix", "security", "refactor"

    Returns:
        (allowed, reason) tuple
    """
    policy = get_policy_for_budget(budget_remaining)

    print("=" * 70)
    print(f"  Deployment Policy Check")
    print("=" * 70)
    print(f"Service:               {service}")
    print(f"Error Budget:          {budget_remaining * 100:+.1f}%")
    print(f"Zone:                  {get_zone_emoji(policy.zone)} {policy.zone.value.upper()}")
    print(f"Change Type:           {change_type}")
    print()

    # Security patches always allowed (exemption)
    if change_type == "security":
        print("✅ ALLOWED")
        print()
        print("Reason: Security patches exempt from error budget policy")
        print("Approval: Expedited review process")
        print("Notifications: Security team + Engineering manager")
        print()
        return True, "Security exemption"

    # Check zone-specific restrictions
    if policy.zone == BudgetZone.RED:
        print("❌ BLOCKED")
        print()
        print("Reason: FEATURE FREEZE in effect - error budget exhausted")
        print(f"Approval Required: {policy.approval_required}")
        print()
        print("⚠️  CRITICAL: Error budget exhausted")
        print("Only emergency fixes allowed with VP approval")
        print()
        if policy.restrictions:
            print("Active Restrictions:")
            for restriction in policy.restrictions:
                print(f"  • {restriction}")
        print()
        return False, "Feature freeze - error budget exhausted"

    if policy.zone == BudgetZone.ORANGE and change_type == "feature":
        print("❌ BLOCKED")
        print()
        print("Reason: No new features allowed in ORANGE zone")
        print("Allowed: Bug fixes and critical changes only")
        print(f"Approval Required: {policy.approval_required}")
        print()
        if policy.restrictions:
            print("Active Restrictions:")
            for restriction in policy.restrictions:
                print(f"  • {restriction}")
        print()
        return False, "New features not allowed in ORANGE zone"

    # Deployment allowed
    print("✅ ALLOWED")
    print()
    print(f"Approval Required: {policy.approval_required}")

    if policy.notification_required:
        print(f"Notifications: {', '.join(policy.notification_required)}")

    if policy.restrictions:
        print()
        print("Active Restrictions:")
        for restriction in policy.restrictions:
            print(f"  • {restriction}")

    if policy.recommendations:
        print()
        print("Recommendations:")
        for rec in policy.recommendations:
            print(f"  • {rec}")

    print()
    return True, f"Allowed in {policy.zone.value} zone"


def get_approval_contact(approval_level: str) -> str:
    """Get contact information for approval level."""
    contacts = {
        "standard": "Team lead or senior engineer",
        "senior_engineer": "Senior engineer + code review",
        "engineering_manager": "Engineering Manager approval required",
        "vp_engineering": "VP Engineering approval required (escalate immediately)"
    }
    return contacts.get(approval_level, "Unknown")


def print_policy_summary():
    """Print summary of error budget policy zones."""
    print()
    print("=" * 70)
    print("  Error Budget Policy Summary")
    print("=" * 70)
    print()
    print("Zone Thresholds:")
    print("  🟢 GREEN:  100% - 75% budget remaining")
    print("  🟡 YELLOW:  75% - 25% budget remaining")
    print("  🟠 ORANGE:  25% -  0% budget remaining")
    print("  🔴 RED:      <0% (budget exhausted)")
    print()
    print("Policy Actions by Zone:")
    print()

    for name, policy in [
        ("GREEN", GREEN_POLICY),
        ("YELLOW", YELLOW_POLICY),
        ("ORANGE", ORANGE_POLICY),
        ("RED", RED_POLICY)
    ]:
        emoji = get_zone_emoji(policy.zone)
        print(f"{emoji} {name}:")
        print(f"  • Deployments: {'✅ Allowed' if policy.can_deploy else '❌ Blocked'}")
        print(f"  • New Features: {'✅ Allowed' if policy.can_new_features else '❌ Blocked'}")
        print(f"  • Approval: {policy.approval_required}")
        if policy.restrictions:
            print(f"  • Restrictions: {len(policy.restrictions)} active")
        print()

    print("Exemptions:")
    print("  • Security patches (always allowed)")
    print("  • Data loss prevention (always allowed)")
    print("  • Legal/compliance requirements (case-by-case)")
    print()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check if deployment is allowed based on error budget policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if feature deployment allowed with good budget
  %(prog)s --service model-api --budget 0.85 --type feature

  # Check with low budget
  %(prog)s --service model-api --budget 0.15 --type feature

  # Check with exhausted budget
  %(prog)s --service model-api --budget -0.05 --type bugfix

  # Security patch (always allowed)
  %(prog)s --service model-api --budget -0.10 --type security

  # Show policy summary
  %(prog)s --show-policy
        """
    )

    parser.add_argument(
        "--service",
        help="Service name (e.g., model-api, feature-store)"
    )
    parser.add_argument(
        "--budget",
        type=float,
        help="Error budget remaining as decimal (e.g., 0.85 for 85%%)"
    )
    parser.add_argument(
        "--type",
        choices=["feature", "bugfix", "security", "refactor"],
        help="Type of change being deployed"
    )
    parser.add_argument(
        "--show-policy",
        action="store_true",
        help="Show policy summary and exit"
    )

    args = parser.parse_args()

    if args.show_policy:
        print_policy_summary()
        return 0

    if not all([args.service, args.budget is not None, args.type]):
        parser.print_help()
        return 1

    allowed, reason = check_deployment_allowed(
        args.service,
        args.budget,
        args.type
    )

    # Exit code: 0 = allowed, 1 = blocked
    return 0 if allowed else 1


if __name__ == "__main__":
    sys.exit(main())
