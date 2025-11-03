#!/usr/bin/env python3
"""
FinOps Cost Dashboard Generator

Generate comprehensive cost dashboards with visualizations for AWS spending analysis.

Usage:
    python cost_dashboard.py --period monthly --export dashboard.html
    python cost_dashboard.py --period weekly --team ml-platform
    python cost_dashboard.py --custom --start 2024-01-01 --end 2024-06-30
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List
import boto3

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    print("Error: matplotlib is required for dashboard generation")
    print("Install with: pip install matplotlib")
    sys.exit(1)


def get_cost_data(start_date: str, end_date: str, granularity: str = 'DAILY') -> Dict:
    """Fetch cost data from AWS Cost Explorer."""
    ce = boto3.client('ce')

    # Get total costs
    response = ce.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity=granularity,
        Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )

    return response


def get_cost_by_team(start_date: str, end_date: str) -> Dict:
    """Get costs grouped by team tag."""
    ce = boto3.client('ce')

    response = ce.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity='MONTHLY',
        Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'TAG', 'Key': 'team'}]
    )

    return response


def parse_cost_data(response: Dict) -> tuple:
    """Parse AWS Cost Explorer response into time series data."""
    dates = []
    costs_by_service = {}

    for result in response['ResultsByTime']:
        date = datetime.strptime(result['TimePeriod']['Start'], '%Y-%m-%d')
        dates.append(date)

        for group in result['Groups']:
            service = group['Keys'][0]
            cost = float(group['Metrics']['UnblendedCost']['Amount'])

            if service not in costs_by_service:
                costs_by_service[service] = []

            costs_by_service[service].append(cost)

    # Fill missing values with 0
    for service, costs in costs_by_service.items():
        while len(costs) < len(dates):
            costs.append(0)

    return dates, costs_by_service


def create_cost_trend_chart(dates: List, costs_by_service: Dict, output_file: str):
    """Create cost trend line chart."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top 10 services by total cost
    service_totals = {
        service: sum(costs)
        for service, costs in costs_by_service.items()
    }

    top_services = sorted(
        service_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Plot top services
    for service, _ in top_services:
        costs = costs_by_service[service]
        ax.plot(dates, costs, marker='o', linewidth=2, label=service)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cost (USD)', fontsize=12)
    ax.set_title('AWS Cost Trend by Service (Top 10)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created cost trend chart: {output_file}")


def create_service_pie_chart(costs_by_service: Dict, output_file: str):
    """Create pie chart of costs by service."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate total cost per service
    service_totals = {
        service: sum(costs)
        for service, costs in costs_by_service.items()
    }

    # Sort by cost
    sorted_services = sorted(
        service_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top 10 + "Other"
    top_10 = sorted_services[:10]
    other_total = sum(cost for _, cost in sorted_services[10:])

    labels = [service for service, _ in top_10]
    sizes = [cost for _, cost in top_10]

    if other_total > 0:
        labels.append('Other')
        sizes.append(other_total)

    # Create pie chart
    colors = plt.cm.Set3(range(len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Cost Distribution by Service', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created service pie chart: {output_file}")


def create_stacked_area_chart(dates: List, costs_by_service: Dict, output_file: str):
    """Create stacked area chart showing cost composition over time."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top 10 services
    service_totals = {
        service: sum(costs)
        for service, costs in costs_by_service.items()
    }

    top_services = sorted(
        service_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Prepare data
    service_names = [service for service, _ in top_services]
    costs_matrix = [costs_by_service[service] for service in service_names]

    # Create stacked area chart
    ax.stackplot(
        dates,
        *costs_matrix,
        labels=service_names,
        alpha=0.8
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cost (USD)', fontsize=12)
    ax.set_title('Cost Composition Over Time (Top 10 Services)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created stacked area chart: {output_file}")


def create_daily_cost_bar_chart(dates: List, costs_by_service: Dict, output_file: str):
    """Create bar chart of daily total costs."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate daily totals
    daily_totals = []
    for i in range(len(dates)):
        daily_total = sum(costs[i] for costs in costs_by_service.values())
        daily_totals.append(daily_total)

    # Calculate average and highlight anomalies
    avg_cost = sum(daily_totals) / len(daily_totals)
    threshold = avg_cost * 1.5

    colors = ['red' if cost > threshold else 'steelblue' for cost in daily_totals]

    # Create bar chart
    ax.bar(dates, daily_totals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add average line
    ax.axhline(y=avg_cost, color='green', linestyle='--', linewidth=2, label=f'Average: ${avg_cost:.2f}')

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Anomaly Threshold: ${threshold:.2f}')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Cost (USD)', fontsize=12)
    ax.set_title('Daily Cost with Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created daily cost bar chart: {output_file}")


def create_team_comparison_chart(team_costs: Dict, output_file: str):
    """Create bar chart comparing costs by team."""
    fig, ax = plt.subplots(figsize=(12, 8))

    teams = list(team_costs.keys())
    costs = list(team_costs.values())

    # Sort by cost
    sorted_pairs = sorted(zip(teams, costs), key=lambda x: x[1], reverse=True)
    teams = [team for team, _ in sorted_pairs]
    costs = [cost for _, cost in sorted_pairs]

    colors = plt.cm.viridis(range(len(teams)))

    bars = ax.barh(teams, costs, color=colors, edgecolor='black', linewidth=1)

    # Add cost labels
    for bar, cost in zip(bars, costs):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f'  ${cost:,.0f}',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Cost (USD)', fontsize=12)
    ax.set_ylabel('Team', fontsize=12)
    ax.set_title('Cost by Team', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created team comparison chart: {output_file}")


def generate_summary_report(
    dates: List,
    costs_by_service: Dict,
    output_file: str
):
    """Generate text summary report."""
    total_cost = sum(sum(costs) for costs in costs_by_service.values())
    num_days = len(dates)
    avg_daily_cost = total_cost / num_days if num_days > 0 else 0

    # Top services
    service_totals = {
        service: sum(costs)
        for service, costs in costs_by_service.items()
    }

    top_services = sorted(
        service_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("  AWS Cost Dashboard - Summary Report")
    report.append("=" * 80)
    report.append("")
    report.append(f"Analysis Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    report.append(f"Duration: {num_days} days")
    report.append("")
    report.append(f"Total Cost: ${total_cost:,.2f}")
    report.append(f"Average Daily Cost: ${avg_daily_cost:,.2f}")
    report.append(f"Projected Monthly Cost: ${avg_daily_cost * 30:,.2f}")
    report.append("")
    report.append("Top 10 Services by Cost:")
    report.append("-" * 80)

    for i, (service, cost) in enumerate(top_services, 1):
        percentage = (cost / total_cost) * 100
        report.append(f"{i:2d}. {service:<40} ${cost:>12,.2f}  ({percentage:>5.1f}%)")

    report.append("")
    report.append("=" * 80)

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Created summary report: {output_file}")

    # Also print to console
    print()
    print('\n'.join(report))
    print()


def create_html_dashboard(
    dates: List,
    costs_by_service: Dict,
    charts: Dict[str, str],
    output_file: str
):
    """Create interactive HTML dashboard."""
    total_cost = sum(sum(costs) for costs in costs_by_service.values())
    num_days = len(dates)
    avg_daily_cost = total_cost / num_days if num_days > 0 else 0

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AWS Cost Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #232f3e;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #232f3e;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #232f3e;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AWS Cost Dashboard</h1>
        <p>Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}</p>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value">${total_cost:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average Daily Cost</div>
            <div class="metric-value">${avg_daily_cost:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Projected Monthly</div>
            <div class="metric-value">${avg_daily_cost * 30:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Number of Services</div>
            <div class="metric-value">{len(costs_by_service)}</div>
        </div>
    </div>

    <div class="chart-container">
        <div class="chart-title">Cost Trend by Service</div>
        <img src="{charts['trend']}" alt="Cost Trend">
    </div>

    <div class="chart-container">
        <div class="chart-title">Daily Cost with Anomaly Detection</div>
        <img src="{charts['daily']}" alt="Daily Cost">
    </div>

    <div class="chart-container">
        <div class="chart-title">Cost Composition Over Time</div>
        <img src="{charts['stacked']}" alt="Cost Composition">
    </div>

    <div class="chart-container">
        <div class="chart-title">Service Cost Distribution</div>
        <img src="{charts['pie']}" alt="Service Distribution">
    </div>

    <div class="footer">
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>AWS Cost Dashboard v1.0</p>
    </div>
</body>
</html>
    """

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✓ Created HTML dashboard: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate AWS cost dashboard with visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dashboard for last 30 days
  %(prog)s --period monthly --export dashboard.html

  # Generate weekly dashboard
  %(prog)s --period weekly

  # Custom date range
  %(prog)s --custom --start 2024-01-01 --end 2024-06-30

  # Dashboard for specific team
  %(prog)s --period monthly --team ml-platform
        """
    )

    parser.add_argument(
        '--period',
        choices=['weekly', 'monthly', 'quarterly'],
        help='Predefined period'
    )
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Use custom date range'
    )
    parser.add_argument(
        '--start',
        help='Start date (YYYY-MM-DD) for custom period'
    )
    parser.add_argument(
        '--end',
        help='End date (YYYY-MM-DD) for custom period'
    )
    parser.add_argument(
        '--team',
        help='Filter by team tag'
    )
    parser.add_argument(
        '--export',
        default='dashboard.html',
        help='Output HTML file (default: dashboard.html)'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory for charts (default: current directory)'
    )

    args = parser.parse_args()

    # Determine date range
    end_date = datetime.now().date()

    if args.custom:
        if not args.start or not args.end:
            print("Error: --start and --end required for --custom")
            return 1
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    elif args.period == 'weekly':
        start_date = end_date - timedelta(days=7)
    elif args.period == 'monthly':
        start_date = end_date - timedelta(days=30)
    elif args.period == 'quarterly':
        start_date = end_date - timedelta(days=90)
    else:
        print("Error: Specify --period or --custom")
        return 1

    print()
    print("=" * 80)
    print("  Generating AWS Cost Dashboard")
    print("=" * 80)
    print()
    print(f"Period: {start_date} to {end_date}")
    print()

    try:
        # Fetch data
        print("Fetching cost data from AWS Cost Explorer...")
        response = get_cost_data(
            start_date.isoformat(),
            end_date.isoformat(),
            granularity='DAILY'
        )

        # Parse data
        dates, costs_by_service = parse_cost_data(response)

        if not dates:
            print("No cost data available for the specified period")
            return 1

        print(f"Loaded {len(dates)} days of cost data")
        print(f"Found {len(costs_by_service)} services")
        print()

        # Generate charts
        print("Generating visualizations...")

        chart_files = {
            'trend': f"{args.output_dir}/cost_trend.png",
            'pie': f"{args.output_dir}/service_pie.png",
            'stacked': f"{args.output_dir}/cost_stacked.png",
            'daily': f"{args.output_dir}/daily_cost.png"
        }

        create_cost_trend_chart(dates, costs_by_service, chart_files['trend'])
        create_service_pie_chart(costs_by_service, chart_files['pie'])
        create_stacked_area_chart(dates, costs_by_service, chart_files['stacked'])
        create_daily_cost_bar_chart(dates, costs_by_service, chart_files['daily'])

        # Generate summary report
        summary_file = f"{args.output_dir}/cost_summary.txt"
        generate_summary_report(dates, costs_by_service, summary_file)

        # Generate HTML dashboard
        create_html_dashboard(dates, costs_by_service, chart_files, args.export)

        print()
        print("✓ Dashboard generation complete!")
        print(f"  Open {args.export} in your browser to view the dashboard")
        print()

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Note: This script requires:")
        print("1. AWS credentials with Cost Explorer read access")
        print("2. matplotlib installed (pip install matplotlib)")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
