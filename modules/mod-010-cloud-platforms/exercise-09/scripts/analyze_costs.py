#!/usr/bin/env python3
"""
AWS Cost Analysis Tool

Analyze AWS Cost Explorer data to understand spending patterns,
identify cost drivers, and detect anomalies.

Usage:
    python analyze_costs.py --months 6
    python analyze_costs.py --months 3 --service EC2
    python analyze_costs.py --export costs.json
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List
import boto3


def get_monthly_costs(start_date: str, end_date: str, group_by: str = 'SERVICE') -> Dict:
    """
    Get monthly costs from AWS Cost Explorer.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        group_by: Dimension to group by (SERVICE, LINKED_ACCOUNT, etc.)

    Returns:
        Dict with cost data
    """
    ce = boto3.client('ce')

    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,
            'End': end_date
        },
        Granularity='MONTHLY',
        Metrics=['UnblendedCost', 'UsageQuantity'],
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': group_by},
        ]
    )

    # Parse results
    costs_by_group = {}
    for result in response['ResultsByTime']:
        month = result['TimePeriod']['Start']
        for group in result['Groups']:
            group_name = group['Keys'][0]
            cost = float(group['Metrics']['UnblendedCost']['Amount'])
            usage = float(group['Metrics']['UsageQuantity']['Amount'])

            if group_name not in costs_by_group:
                costs_by_group[group_name] = {}

            costs_by_group[group_name][month] = {
                'cost': cost,
                'usage': usage
            }

    return costs_by_group


def get_daily_costs(start_date: str, end_date: str) -> Dict[str, float]:
    """Get daily costs for anomaly detection."""
    ce = boto3.client('ce')

    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,
            'End': end_date
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )

    daily_costs = {}
    for result in response['ResultsByTime']:
        date = result['TimePeriod']['Start']
        cost = float(result['Total']['UnblendedCost']['Amount'])
        daily_costs[date] = cost

    return daily_costs


def print_top_services(costs_by_service: Dict, top_n: int = 10):
    """Print top N services by cost."""
    # Calculate total cost per service
    service_totals = {}
    for service, monthly_costs in costs_by_service.items():
        service_totals[service] = sum(m['cost'] for m in monthly_costs.values())

    # Sort by total cost
    sorted_services = sorted(
        service_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print()
    print("=" * 70)
    print(f"  Top {top_n} Services by Cost")
    print("=" * 70)
    print()

    total_cost = sum(service_totals.values())

    for i, (service, cost) in enumerate(sorted_services[:top_n], 1):
        percentage = (cost / total_cost) * 100
        print(f"{i}. {service}")
        print(f"   Cost: ${cost:,.2f} ({percentage:.1f}%)")

        # Show month-over-month trend
        months = list(costs_by_service[service].keys())
        if len(months) >= 2:
            last_month = costs_by_service[service][months[-1]]['cost']
            prev_month = costs_by_service[service][months[-2]]['cost']
            change = ((last_month - prev_month) / prev_month) * 100

            trend = "↑" if change > 0 else "↓"
            print(f"   Trend: {trend} {abs(change):.1f}% vs previous month")

        print()


def analyze_cost_trends(costs_by_service: Dict):
    """Analyze cost trends and growth rates."""
    print()
    print("=" * 70)
    print("  Cost Trend Analysis")
    print("=" * 70)
    print()

    # Calculate total costs by month
    all_months = set()
    for service, monthly_costs in costs_by_service.items():
        all_months.update(monthly_costs.keys())

    all_months = sorted(all_months)
    monthly_totals = {}

    for month in all_months:
        total = sum(
            costs_by_service[service].get(month, {}).get('cost', 0)
            for service in costs_by_service.keys()
        )
        monthly_totals[month] = total

    # Print monthly breakdown
    print(f"{'Month':<15} {'Total Cost':<20} {'Change':<15}")
    print("-" * 50)

    prev_cost = None
    for month, cost in monthly_totals.items():
        if prev_cost is not None:
            change = ((cost - prev_cost) / prev_cost) * 100
            change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        else:
            change_str = "-"

        print(f"{month:<15} ${cost:>15,.2f}  {change_str:<15}")
        prev_cost = cost

    print()

    # Calculate average growth rate
    if len(monthly_totals) >= 2:
        costs_list = list(monthly_totals.values())
        growth_rates = []

        for i in range(1, len(costs_list)):
            growth = ((costs_list[i] - costs_list[i-1]) / costs_list[i-1]) * 100
            growth_rates.append(growth)

        avg_growth = sum(growth_rates) / len(growth_rates)

        print(f"Average monthly growth rate: {avg_growth:+.1f}%")
        print()

        # Project next month
        last_cost = costs_list[-1]
        projected_next_month = last_cost * (1 + avg_growth / 100)

        print(f"Current monthly cost: ${last_cost:,.2f}")
        print(f"Projected next month: ${projected_next_month:,.2f}")

        if avg_growth > 10:
            print()
            print(f"⚠️  WARNING: High growth rate ({avg_growth:+.1f}%/month)")
            print(f"   At this rate, costs will reach ${last_cost * (1.1 ** 6):,.2f} in 6 months")

        print()


def detect_cost_anomalies(daily_costs: Dict[str, float], threshold: float = 1.5):
    """
    Detect cost anomalies.

    Args:
        daily_costs: Dict of {date: cost}
        threshold: Spike detection threshold (1.5 = 50% above average)
    """
    costs = list(daily_costs.values())
    avg_cost = sum(costs) / len(costs)
    std_dev = (sum((c - avg_cost) ** 2 for c in costs) / len(costs)) ** 0.5

    print()
    print("=" * 70)
    print("  Cost Anomaly Detection")
    print("=" * 70)
    print()

    print(f"Average daily cost: ${avg_cost:,.2f}")
    print(f"Standard deviation: ${std_dev:,.2f}")
    print(f"Detection threshold: {threshold}x average")
    print()

    spikes = []
    for date, cost in daily_costs.items():
        if cost > (avg_cost * threshold):
            spike_percentage = ((cost / avg_cost) - 1) * 100
            spikes.append({
                'date': date,
                'cost': cost,
                'avg_cost': avg_cost,
                'spike_percentage': spike_percentage
            })

    if spikes:
        print(f"⚠️  Detected {len(spikes)} cost spikes:")
        print()

        for spike in sorted(spikes, key=lambda x: x['cost'], reverse=True)[:10]:
            print(f"Date: {spike['date']}")
            print(f"Cost: ${spike['cost']:,.2f}")
            print(f"Average: ${spike['avg_cost']:,.2f}")
            print(f"Spike: +{spike['spike_percentage']:.1f}%")
            print()
    else:
        print("✓ No significant cost spikes detected")
        print()


def export_cost_data(costs_by_service: Dict, filename: str):
    """Export cost data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(costs_by_service, f, indent=2, default=str)

    print(f"✓ Cost data exported to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AWS costs using Cost Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze last 6 months
  %(prog)s --months 6

  # Analyze specific service
  %(prog)s --months 3 --service EC2

  # Export data for further analysis
  %(prog)s --months 12 --export costs.json

  # Detect cost anomalies
  %(prog)s --months 1 --anomalies --threshold 1.3
        """
    )

    parser.add_argument(
        '--months',
        type=int,
        default=6,
        help='Number of months to analyze (default: 6)'
    )
    parser.add_argument(
        '--service',
        help='Filter by specific service (e.g., EC2, S3)'
    )
    parser.add_argument(
        '--export',
        help='Export cost data to JSON file'
    )
    parser.add_argument(
        '--anomalies',
        action='store_true',
        help='Detect cost anomalies'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1.5,
        help='Anomaly detection threshold (default: 1.5 = 50%% above average)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top services to show (default: 10)'
    )

    args = parser.parse_args()

    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=args.months * 30)

    print()
    print("=" * 70)
    print("  AWS Cost Analysis")
    print("=" * 70)
    print()
    print(f"Analysis Period: {start_date} to {end_date}")
    print(f"Duration: {args.months} months")
    print()

    try:
        # Get monthly costs
        print("Fetching cost data from AWS Cost Explorer...")
        costs_by_service = get_monthly_costs(
            start_date.isoformat(),
            end_date.isoformat()
        )

        # Filter by service if specified
        if args.service:
            costs_by_service = {
                k: v for k, v in costs_by_service.items()
                if args.service.lower() in k.lower()
            }

            if not costs_by_service:
                print(f"No data found for service: {args.service}")
                return 1

        # Print top services
        print_top_services(costs_by_service, args.top)

        # Analyze trends
        analyze_cost_trends(costs_by_service)

        # Detect anomalies if requested
        if args.anomalies:
            print("Fetching daily cost data for anomaly detection...")
            daily_costs = get_daily_costs(
                (end_date - timedelta(days=args.months * 30)).isoformat(),
                end_date.isoformat()
            )
            detect_cost_anomalies(daily_costs, args.threshold)

        # Export data if requested
        if args.export:
            export_cost_data(costs_by_service, args.export)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Note: This script requires AWS credentials with Cost Explorer access.")
        print("Run 'aws configure' to set up credentials.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
