#!/usr/bin/env python3
"""
Error Budget Calculator

Calculate error budgets for different SLO targets and time periods.
Helps teams understand the implications of SLO choices.

Usage:
    python calculate_error_budget.py
    python calculate_error_budget.py --slo 0.999 --days 30
    python calculate_error_budget.py --compare
"""

import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class ErrorBudget:
    """Error budget calculation results."""
    slo_target: float
    slo_percent: str
    time_period_days: int
    allowed_error_rate: float
    allowed_error_percent: str
    total_minutes: int
    error_budget_minutes: float
    error_budget_hours: float
    error_budget_seconds: float
    availability_class: str


def calculate_error_budget(slo_target: float, time_period_days: int = 30) -> ErrorBudget:
    """
    Calculate error budget given an SLO target.

    Args:
        slo_target: Target availability (e.g., 0.999 for 99.9%)
        time_period_days: Time period for budget (default 30 days)

    Returns:
        ErrorBudget with all calculated values
    """
    total_minutes = time_period_days * 24 * 60
    allowed_error_rate = 1 - slo_target
    error_budget_minutes = total_minutes * allowed_error_rate
    error_budget_hours = error_budget_minutes / 60
    error_budget_seconds = error_budget_minutes * 60

    # Classify availability level
    if slo_target >= 0.9999:
        availability_class = "Four Nines (Mission Critical)"
    elif slo_target >= 0.999:
        availability_class = "Three Nines (High Availability)"
    elif slo_target >= 0.99:
        availability_class = "Two Nines (Standard)"
    else:
        availability_class = "Below Two Nines (Basic)"

    return ErrorBudget(
        slo_target=slo_target,
        slo_percent=f"{slo_target * 100}%",
        time_period_days=time_period_days,
        allowed_error_rate=allowed_error_rate,
        allowed_error_percent=f"{allowed_error_rate * 100}%",
        total_minutes=total_minutes,
        error_budget_minutes=error_budget_minutes,
        error_budget_hours=error_budget_hours,
        error_budget_seconds=error_budget_seconds,
        availability_class=availability_class
    )


def print_budget_details(budget: ErrorBudget):
    """Print detailed error budget information."""
    print()
    print("=" * 70)
    print(f"  Error Budget: {budget.slo_percent}")
    print("=" * 70)
    print()
    print(f"SLO Target:            {budget.slo_percent}")
    print(f"Allowed Error Rate:    {budget.allowed_error_percent}")
    print(f"Time Period:           {budget.time_period_days} days")
    print(f"Availability Class:    {budget.availability_class}")
    print()
    print("Error Budget Breakdown:")
    print(f"  • {budget.error_budget_seconds:.2f} seconds")
    print(f"  • {budget.error_budget_minutes:.2f} minutes")
    print(f"  • {budget.error_budget_hours:.2f} hours")
    print()

    # Contextualize the numbers
    if budget.error_budget_hours < 0.1:
        print("⚠️  VERY STRICT: Less than 6 minutes downtime allowed")
        print("   Requires exceptional engineering practices")
    elif budget.error_budget_hours < 1:
        print("⚠️  STRICT: Less than 1 hour downtime allowed")
        print("   Requires robust monitoring and quick response")
    elif budget.error_budget_hours < 10:
        print("✓ MODERATE: Several hours downtime budget")
        print("   Typical for production services")
    else:
        print("✓ LENIENT: Multiple hours downtime budget")
        print("   Suitable for non-critical services")
    print()


def compare_slo_targets():
    """Compare error budgets across different SLO targets."""
    print()
    print("=" * 70)
    print("  SLO Target Comparison (30-day window)")
    print("=" * 70)
    print()

    targets = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9999, 0.99999]

    print(f"{'SLO':<10} {'Error %':<10} {'Downtime':<20} {'Class':<25}")
    print("-" * 70)

    for target in targets:
        budget = calculate_error_budget(target, 30)

        # Format downtime nicely
        if budget.error_budget_hours >= 24:
            downtime_str = f"{budget.error_budget_hours / 24:.1f} days"
        elif budget.error_budget_hours >= 1:
            downtime_str = f"{budget.error_budget_hours:.1f} hours"
        else:
            downtime_str = f"{budget.error_budget_minutes:.1f} minutes"

        print(f"{budget.slo_percent:<10} {budget.allowed_error_percent:<10} "
              f"{downtime_str:<20} {budget.availability_class:<25}")

    print()
    print("Key Insights:")
    print("  • Going from 99% to 99.9% = 10x stricter (7.2h → 43min)")
    print("  • Going from 99.9% to 99.99% = 10x stricter (43min → 4.3min)")
    print("  • Each additional '9' requires ~10x more investment")
    print()


def calculate_required_slo(max_downtime_minutes: float, time_period_days: int = 30) -> float:
    """
    Calculate required SLO to stay within max downtime budget.

    Args:
        max_downtime_minutes: Maximum allowed downtime in minutes
        time_period_days: Time period

    Returns:
        Required SLO target
    """
    total_minutes = time_period_days * 24 * 60
    allowed_error_rate = max_downtime_minutes / total_minutes
    slo_target = 1 - allowed_error_rate
    return slo_target


def calculate_burn_rate_impact():
    """Calculate how burn rates affect budget consumption."""
    print()
    print("=" * 70)
    print("  Burn Rate Impact (99.9% SLO, 30-day budget)")
    print("=" * 70)
    print()

    budget = calculate_error_budget(0.999, 30)

    print(f"Total Error Budget: {budget.error_budget_minutes:.1f} minutes")
    print()

    burn_rates = [
        (1, "Sustainable", "30 days"),
        (2, "Moderate concern", "15 days"),
        (6, "Moderate burn alert", "5 days"),
        (14.4, "Fast burn alert (PAGE)", "2.08 days"),
        (100, "Critical incident", "7.2 hours")
    ]

    print(f"{'Burn Rate':<12} {'Severity':<25} {'Budget Exhausted In'}")
    print("-" * 70)

    for rate, severity, time_to_exhaust in burn_rates:
        print(f"{rate:<12} {severity:<25} {time_to_exhaust}")

    print()
    print("Burn Rate Definition:")
    print("  • 1x  = Consuming budget at exactly the sustainable rate")
    print("  • 2x  = Consuming budget twice as fast as sustainable")
    print("  • 14.4x = Google SRE 'fast burn' threshold (multi-window)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate error budgets for SLO targets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate budget for 99.9% SLO over 30 days
  %(prog)s --slo 0.999 --days 30

  # Compare different SLO targets
  %(prog)s --compare

  # Show burn rate impact
  %(prog)s --burn-rates

  # Calculate required SLO for max downtime
  %(prog)s --max-downtime 60 --days 30
        """
    )

    parser.add_argument(
        "--slo",
        type=float,
        help="SLO target (e.g., 0.999 for 99.9%%)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Time period in days (default: 30)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare error budgets across SLO targets"
    )
    parser.add_argument(
        "--burn-rates",
        action="store_true",
        help="Show burn rate impact analysis"
    )
    parser.add_argument(
        "--max-downtime",
        type=float,
        help="Calculate required SLO for max downtime (minutes)"
    )

    args = parser.parse_args()

    if args.compare:
        compare_slo_targets()
        return 0

    if args.burn_rates:
        calculate_burn_rate_impact()
        return 0

    if args.max_downtime:
        required_slo = calculate_required_slo(args.max_downtime, args.days)
        print()
        print("=" * 70)
        print("  Required SLO Calculation")
        print("=" * 70)
        print()
        print(f"Maximum Downtime:      {args.max_downtime:.1f} minutes")
        print(f"Time Period:           {args.days} days")
        print(f"Required SLO:          {required_slo * 100:.3f}%")
        print()

        # Show closest standard SLO
        standard_slos = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9999]
        closest_slo = min(standard_slos, key=lambda x: abs(x - required_slo))
        print(f"Closest Standard SLO:  {closest_slo * 100}%")

        if closest_slo > required_slo:
            print(f"                       (Stricter by {(closest_slo - required_slo) * 100:.3f}%)")
        elif closest_slo < required_slo:
            print(f"                       (More lenient by {(required_slo - closest_slo) * 100:.3f}%)")

        print()
        return 0

    if not args.slo:
        # Default: show comparison
        compare_slo_targets()
        print()
        print("For specific SLO calculation, use: --slo <target>")
        print("For burn rate analysis, use: --burn-rates")
        return 0

    budget = calculate_error_budget(args.slo, args.days)
    print_budget_details(budget)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
