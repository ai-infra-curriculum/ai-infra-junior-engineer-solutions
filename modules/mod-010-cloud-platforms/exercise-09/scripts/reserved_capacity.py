#!/usr/bin/env python3
"""
Reserved Instance and Savings Plan Analysis Tool

Analyze EC2 usage patterns and recommend Reserved Instance or Savings Plan purchases
to optimize costs.

Usage:
    python reserved_capacity.py --analyze --months 6
    python reserved_capacity.py --recommend --term 3year
    python reserved_capacity.py --utilization  # Check current RI/SP utilization
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict, Counter
import boto3


# RI discount rates (approximate)
RI_DISCOUNTS = {
    '1year': {
        'no-upfront': 0.35,
        'partial-upfront': 0.40,
        'all-upfront': 0.42
    },
    '3year': {
        'no-upfront': 0.55,
        'partial-upfront': 0.60,
        'all-upfront': 0.62
    }
}

# Savings Plans discount rates (approximate)
SP_DISCOUNTS = {
    '1year': 0.28,
    '3year': 0.50
}


def analyze_instance_usage(months: int = 6) -> Dict:
    """
    Analyze EC2 instance usage over time to identify stable workloads.

    Args:
        months: Number of months to analyze

    Returns:
        Dict with usage patterns
    """
    ec2 = boto3.client('ec2')

    print(f"Analyzing EC2 usage for last {months} months...")

    # Get all running and stopped instances
    response = ec2.describe_instances()

    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instances.append(instance)

    # Group by instance type and family
    instance_types = defaultdict(list)

    for instance in instances:
        instance_type = instance['InstanceType']
        instance_id = instance['InstanceId']
        state = instance['State']['Name']
        launch_time = instance['LaunchTime']

        # Get tags
        tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}

        # Calculate uptime in days
        uptime_days = (datetime.now(launch_time.tzinfo) - launch_time).days

        instance_types[instance_type].append({
            'instance_id': instance_id,
            'state': state,
            'launch_time': launch_time,
            'uptime_days': uptime_days,
            'environment': tags.get('environment', 'unknown'),
            'team': tags.get('team', 'unknown'),
            'project': tags.get('project', 'unknown')
        })

    return instance_types


def classify_workload_stability(instances: List[Dict], months: int) -> Dict:
    """
    Classify workload as stable, variable, or temporary based on uptime.

    Args:
        instances: List of instance dicts
        months: Analysis period in months

    Returns:
        Dict with classification
    """
    min_uptime_days = months * 30

    # Count instances by state
    running_count = sum(1 for i in instances if i['state'] == 'running')
    stopped_count = sum(1 for i in instances if i['state'] == 'stopped')

    # Check uptime stability
    stable_count = sum(1 for i in instances if i['uptime_days'] >= min_uptime_days)

    # Classification
    stability_ratio = stable_count / len(instances) if instances else 0

    if stability_ratio >= 0.8 and running_count >= 0.8 * len(instances):
        classification = 'STABLE'  # Good candidate for RIs
        confidence = 'HIGH'
    elif stability_ratio >= 0.5:
        classification = 'SEMI_STABLE'  # Consider RIs or Savings Plans
        confidence = 'MEDIUM'
    else:
        classification = 'VARIABLE'  # Use on-demand or Savings Plans
        confidence = 'LOW'

    return {
        'classification': classification,
        'confidence': confidence,
        'total_instances': len(instances),
        'running_instances': running_count,
        'stopped_instances': stopped_count,
        'stable_instances': stable_count,
        'stability_ratio': stability_ratio
    }


def recommend_reserved_instances(
    instance_types: Dict,
    months: int,
    term: str = '3year',
    payment_option: str = 'partial-upfront'
) -> List[Dict]:
    """
    Recommend Reserved Instance purchases based on usage analysis.

    Args:
        instance_types: Dict of instance type -> instances
        months: Analysis period
        term: RI term ('1year' or '3year')
        payment_option: Payment option

    Returns:
        List of RI recommendations
    """
    recommendations = []

    for instance_type, instances in instance_types.items():
        # Classify workload
        classification = classify_workload_stability(instances, months)

        # Only recommend RIs for stable workloads
        if classification['classification'] not in ['STABLE', 'SEMI_STABLE']:
            continue

        # Get pricing info (would need to call EC2 pricing API in reality)
        # Using approximate pricing here
        on_demand_hourly = get_on_demand_pricing(instance_type)

        if on_demand_hourly == 0:
            continue

        # Calculate RI pricing
        discount_rate = RI_DISCOUNTS[term][payment_option]
        ri_hourly = on_demand_hourly * (1 - discount_rate)

        # Recommend covering stable instances
        stable_count = classification['stable_instances']
        running_count = classification['running_instances']

        # Conservative recommendation: cover 70-80% of stable instances
        if classification['classification'] == 'STABLE':
            recommended_count = int(stable_count * 0.80)
        else:  # SEMI_STABLE
            recommended_count = int(stable_count * 0.70)

        if recommended_count == 0:
            continue

        # Calculate savings
        monthly_on_demand = on_demand_hourly * 730 * running_count
        monthly_ri = ri_hourly * 730 * recommended_count
        monthly_remaining_on_demand = on_demand_hourly * 730 * (running_count - recommended_count)
        monthly_total_with_ri = monthly_ri + monthly_remaining_on_demand
        monthly_savings = monthly_on_demand - monthly_total_with_ri

        # Calculate upfront cost (for partial-upfront)
        if payment_option == 'partial-upfront':
            if term == '1year':
                upfront_per_instance = on_demand_hourly * 730 * 12 * discount_rate * 0.5
            else:  # 3year
                upfront_per_instance = on_demand_hourly * 730 * 36 * discount_rate * 0.5
            total_upfront = upfront_per_instance * recommended_count
        elif payment_option == 'all-upfront':
            if term == '1year':
                upfront_per_instance = on_demand_hourly * 730 * 12 * (1 - discount_rate)
            else:
                upfront_per_instance = on_demand_hourly * 730 * 36 * (1 - discount_rate)
            total_upfront = upfront_per_instance * recommended_count
        else:  # no-upfront
            total_upfront = 0

        # Payback period (months)
        payback_months = total_upfront / monthly_savings if monthly_savings > 0 else float('inf')

        recommendations.append({
            'instance_type': instance_type,
            'current_count': running_count,
            'recommended_ri_count': recommended_count,
            'coverage_percentage': (recommended_count / running_count * 100) if running_count > 0 else 0,
            'classification': classification['classification'],
            'confidence': classification['confidence'],
            'term': term,
            'payment_option': payment_option,
            'on_demand_hourly': on_demand_hourly,
            'ri_hourly': ri_hourly,
            'monthly_on_demand_cost': monthly_on_demand,
            'monthly_ri_cost': monthly_total_with_ri,
            'monthly_savings': monthly_savings,
            'annual_savings': monthly_savings * 12,
            'upfront_cost': total_upfront,
            'payback_months': payback_months,
            'discount_percentage': discount_rate * 100
        })

    # Sort by monthly savings
    recommendations.sort(key=lambda x: x['monthly_savings'], reverse=True)

    return recommendations


def get_on_demand_pricing(instance_type: str) -> float:
    """Get on-demand hourly pricing for instance type (approximate)."""
    # Simplified pricing (us-east-1)
    pricing = {
        # General Purpose
        't3.micro': 0.0104,
        't3.small': 0.0208,
        't3.medium': 0.0416,
        't3.large': 0.0832,
        't3.xlarge': 0.1664,
        't3.2xlarge': 0.3328,
        'm5.large': 0.096,
        'm5.xlarge': 0.192,
        'm5.2xlarge': 0.384,
        'm5.4xlarge': 0.768,
        'm5.8xlarge': 1.536,
        'm5.12xlarge': 2.304,
        'm5.16xlarge': 3.072,
        'm5.24xlarge': 4.608,
        # Compute Optimized
        'c5.large': 0.085,
        'c5.xlarge': 0.17,
        'c5.2xlarge': 0.34,
        'c5.4xlarge': 0.68,
        'c5.9xlarge': 1.53,
        'c5.12xlarge': 2.04,
        'c5.18xlarge': 3.06,
        'c5.24xlarge': 4.08,
        # Memory Optimized
        'r5.large': 0.126,
        'r5.xlarge': 0.252,
        'r5.2xlarge': 0.504,
        'r5.4xlarge': 1.008,
        'r5.8xlarge': 2.016,
        'r5.12xlarge': 3.024,
        'r5.16xlarge': 4.032,
        'r5.24xlarge': 6.048,
        # GPU
        'p3.2xlarge': 3.06,
        'p3.8xlarge': 12.24,
        'p3.16xlarge': 24.48,
        'p3dn.24xlarge': 31.218,
        # Inference
        'g4dn.xlarge': 0.526,
        'g4dn.2xlarge': 0.752,
        'g4dn.4xlarge': 1.204,
        'g4dn.8xlarge': 2.176,
        'g4dn.12xlarge': 3.912,
        'g4dn.16xlarge': 4.352,
    }

    return pricing.get(instance_type, 0)


def check_current_ri_utilization() -> Dict:
    """Check current Reserved Instance utilization."""
    ec2 = boto3.client('ec2')
    ce = boto3.client('ce')

    print("Checking current Reserved Instance utilization...")

    # Get active RIs
    try:
        response = ec2.describe_reserved_instances(
            Filters=[{'Name': 'state', 'Values': ['active']}]
        )

        active_ris = response['ReservedInstances']

        # Get RI utilization from Cost Explorer
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)

        utilization_response = ce.get_reservation_utilization(
            TimePeriod={
                'Start': start_date.isoformat(),
                'End': end_date.isoformat()
            },
            Granularity='MONTHLY'
        )

        utilization_data = utilization_response['UtilizationsByTime']

        if utilization_data:
            total_reserved_hours = float(utilization_data[0]['Total']['TotalReservedHours'])
            total_actual_hours = float(utilization_data[0]['Total']['TotalActualHours'])
            utilization_percentage = float(utilization_data[0]['Total']['UtilizationPercentage'])

            return {
                'active_ri_count': len(active_ris),
                'reserved_hours': total_reserved_hours,
                'actual_hours': total_actual_hours,
                'utilization_percentage': utilization_percentage,
                'unused_hours': total_reserved_hours - total_actual_hours
            }
        else:
            return {
                'active_ri_count': len(active_ris),
                'utilization_percentage': 0
            }

    except Exception as e:
        print(f"Warning: Could not fetch RI utilization: {e}")
        return {}


def print_recommendations(recommendations: List[Dict]):
    """Print RI purchase recommendations."""
    print()
    print("=" * 80)
    print("  Reserved Instance Recommendations")
    print("=" * 80)
    print()

    if not recommendations:
        print("No RI recommendations at this time.")
        print("All workloads appear to be variable or short-lived.")
        print()
        return

    # Summary
    total_monthly_savings = sum(r['monthly_savings'] for r in recommendations)
    total_annual_savings = sum(r['annual_savings'] for r in recommendations)
    total_upfront = sum(r['upfront_cost'] for r in recommendations)
    total_ris = sum(r['recommended_ri_count'] for r in recommendations)

    print(f"Total RI recommendations: {total_ris} instances")
    print(f"Total monthly savings: ${total_monthly_savings:,.2f}")
    print(f"Total annual savings: ${total_annual_savings:,.2f}")
    print(f"Total upfront cost: ${total_upfront:,.2f}")
    if total_monthly_savings > 0:
        payback = total_upfront / total_monthly_savings
        print(f"Overall payback period: {payback:.1f} months")
    print()

    # Detailed recommendations
    print("Detailed Recommendations:")
    print("-" * 80)

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['instance_type']}")
        print(f"   Current: {rec['current_count']} instances on-demand")
        print(f"   Recommendation: Purchase {rec['recommended_ri_count']} Reserved Instances")
        print(f"   Coverage: {rec['coverage_percentage']:.1f}%")
        print(f"   Workload: {rec['classification']} (confidence: {rec['confidence']})")
        print()
        print(f"   Term: {rec['term']}, {rec['payment_option']}")
        print(f"   Discount: {rec['discount_percentage']:.1f}%")
        print()
        print(f"   Pricing:")
        print(f"     On-Demand: ${rec['on_demand_hourly']:.3f}/hour")
        print(f"     Reserved:  ${rec['ri_hourly']:.3f}/hour")
        print()
        print(f"   Costs:")
        print(f"     Current monthly (on-demand): ${rec['monthly_on_demand_cost']:,.2f}")
        print(f"     With RIs monthly:            ${rec['monthly_ri_cost']:,.2f}")
        print()
        print(f"   Savings:")
        print(f"     Monthly:  ${rec['monthly_savings']:,.2f}")
        print(f"     Annual:   ${rec['annual_savings']:,.2f}")
        print()
        print(f"   Upfront cost: ${rec['upfront_cost']:,.2f}")
        if rec['payback_months'] < float('inf'):
            print(f"   Payback period: {rec['payback_months']:.1f} months")
        print()


def print_ri_utilization(utilization: Dict):
    """Print current RI utilization."""
    print()
    print("=" * 80)
    print("  Current Reserved Instance Utilization")
    print("=" * 80)
    print()

    if not utilization:
        print("No Reserved Instances found or utilization data unavailable.")
        print()
        return

    print(f"Active Reserved Instances: {utilization.get('active_ri_count', 0)}")

    if 'utilization_percentage' in utilization:
        util_pct = utilization['utilization_percentage']
        print(f"Utilization rate: {util_pct:.1f}%")
        print(f"Reserved hours: {utilization['reserved_hours']:,.0f}")
        print(f"Actual hours: {utilization['actual_hours']:,.0f}")
        print(f"Unused hours: {utilization['unused_hours']:,.0f}")
        print()

        if util_pct < 90:
            print(f"⚠️  WARNING: RI utilization is below 90%")
            print(f"   You may be paying for unused Reserved Instances.")
            print(f"   Consider modifying instance types or purchasing flexible RIs.")
        elif util_pct >= 95:
            print(f"✓ Excellent RI utilization (>95%)")
            print(f"  Consider purchasing additional RIs for uncovered workloads.")
        else:
            print(f"✓ Good RI utilization (90-95%)")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze EC2 usage and recommend Reserved Instance purchases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze usage patterns
  %(prog)s --analyze --months 6

  # Get RI recommendations (3-year, partial upfront)
  %(prog)s --recommend --term 3year --payment partial-upfront

  # Check current RI utilization
  %(prog)s --utilization

  # Get recommendations and check utilization
  %(prog)s --analyze --recommend --utilization --months 6
        """
    )

    parser.add_argument('--analyze', action='store_true', help='Analyze EC2 usage patterns')
    parser.add_argument('--recommend', action='store_true', help='Generate RI purchase recommendations')
    parser.add_argument('--utilization', action='store_true', help='Check current RI utilization')

    parser.add_argument(
        '--months',
        type=int,
        default=6,
        help='Number of months to analyze (default: 6)'
    )
    parser.add_argument(
        '--term',
        choices=['1year', '3year'],
        default='3year',
        help='RI term (default: 3year)'
    )
    parser.add_argument(
        '--payment',
        choices=['no-upfront', 'partial-upfront', 'all-upfront'],
        default='partial-upfront',
        help='Payment option (default: partial-upfront)'
    )
    parser.add_argument(
        '--export',
        help='Export recommendations to JSON file'
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("  Reserved Instance and Savings Plan Analysis")
    print("=" * 80)
    print()

    try:
        # Check current RI utilization
        if args.utilization:
            utilization = check_current_ri_utilization()
            print_ri_utilization(utilization)

        # Analyze and recommend
        if args.analyze or args.recommend:
            instance_types = analyze_instance_usage(args.months)

            if not instance_types:
                print("No EC2 instances found")
                return 1

            print(f"Found {sum(len(instances) for instances in instance_types.values())} instances")
            print(f"Instance types: {len(instance_types)}")
            print()

            if args.recommend:
                recommendations = recommend_reserved_instances(
                    instance_types,
                    args.months,
                    args.term,
                    args.payment
                )

                print_recommendations(recommendations)

                # Export if requested
                if args.export:
                    with open(args.export, 'w') as f:
                        json.dump(recommendations, f, indent=2, default=str)
                    print(f"✓ Recommendations exported to {args.export}")
                    print()

        if not (args.analyze or args.recommend or args.utilization):
            parser.print_help()
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Note: This script requires AWS credentials with EC2 and Cost Explorer access")
        print("Run 'aws configure' to set up credentials")
        return 1


if __name__ == '__main__':
    sys.exit(main())
