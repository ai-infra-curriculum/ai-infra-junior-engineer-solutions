#!/usr/bin/env python3
"""
EC2 Instance Right-Sizing Tool

Analyze EC2 instance utilization and provide right-sizing recommendations
to reduce costs while maintaining performance.

Usage:
    python rightsize_instances.py --days 30
    python rightsize_instances.py --days 14 --percentile 99
    python rightsize_instances.py --export recommendations.json
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import boto3


# AWS pricing (approximate, us-east-1)
EC2_PRICING = {
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
    # Compute Optimized
    'c5.large': 0.085,
    'c5.xlarge': 0.17,
    'c5.2xlarge': 0.34,
    'c5.4xlarge': 0.68,
    'c5.9xlarge': 1.53,
    # Memory Optimized
    'r5.large': 0.126,
    'r5.xlarge': 0.252,
    'r5.2xlarge': 0.504,
    'r5.4xlarge': 1.008,
    # GPU
    'p3.2xlarge': 3.06,
    'p3.8xlarge': 12.24,
    'p3.16xlarge': 24.48,
}


def get_instance_metrics(
    instance_id: str,
    start_time: datetime,
    end_time: datetime,
    percentile: int = 95
) -> Dict:
    """
    Get CPU and memory utilization metrics for an instance.

    Args:
        instance_id: EC2 instance ID
        start_time: Start of analysis period
        end_time: End of analysis period
        percentile: Percentile to use for max calculations (default: 95)

    Returns:
        Dict with utilization metrics
    """
    cloudwatch = boto3.client('cloudwatch')

    # Get CPU utilization
    cpu_response = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,  # 1 hour
        Statistics=['Average', 'Maximum']
    )

    # Calculate CPU statistics
    cpu_datapoints = cpu_response['Datapoints']
    if not cpu_datapoints:
        return None

    cpu_averages = [dp['Average'] for dp in cpu_datapoints]
    cpu_maximums = [dp['Maximum'] for dp in cpu_datapoints]

    cpu_avg = sum(cpu_averages) / len(cpu_averages)
    cpu_max = max(cpu_maximums)
    cpu_p95 = sorted(cpu_maximums)[int(len(cpu_maximums) * (percentile / 100))]

    # Get memory utilization (requires CloudWatch agent)
    # Note: This requires CloudWatch agent installed on instance
    try:
        memory_response = cloudwatch.get_metric_statistics(
            Namespace='CWAgent',
            MetricName='mem_used_percent',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average', 'Maximum']
        )

        memory_datapoints = memory_response['Datapoints']
        if memory_datapoints:
            memory_averages = [dp['Average'] for dp in memory_datapoints]
            memory_maximums = [dp['Maximum'] for dp in memory_datapoints]

            memory_avg = sum(memory_averages) / len(memory_averages)
            memory_max = max(memory_maximums)
            memory_p95 = sorted(memory_maximums)[int(len(memory_maximums) * (percentile / 100))]
        else:
            # No memory metrics available
            memory_avg = None
            memory_max = None
            memory_p95 = None
    except Exception as e:
        print(f"Warning: Could not fetch memory metrics for {instance_id}: {e}")
        memory_avg = None
        memory_max = None
        memory_p95 = None

    # Get network utilization
    network_in = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='NetworkIn',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,
        Statistics=['Sum']
    )

    network_out = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='NetworkOut',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,
        Statistics=['Sum']
    )

    network_in_gb = sum([dp['Sum'] for dp in network_in['Datapoints']]) / (1024**3)
    network_out_gb = sum([dp['Sum'] for dp in network_out['Datapoints']]) / (1024**3)

    return {
        'cpu_avg': cpu_avg,
        'cpu_max': cpu_max,
        'cpu_p95': cpu_p95,
        'memory_avg': memory_avg,
        'memory_max': memory_max,
        'memory_p95': memory_p95,
        'network_in_gb': network_in_gb,
        'network_out_gb': network_out_gb,
        'datapoints': len(cpu_datapoints)
    }


def get_instance_family_and_size(instance_type: str) -> tuple:
    """Parse instance type into family and size."""
    parts = instance_type.split('.')
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def recommend_instance_type(
    current_type: str,
    cpu_p95: float,
    memory_p95: Optional[float],
    network_io_gb: float
) -> Optional[Dict]:
    """
    Recommend a more appropriate instance type based on utilization.

    Args:
        current_type: Current EC2 instance type
        cpu_p95: 95th percentile CPU utilization
        memory_p95: 95th percentile memory utilization (can be None)
        network_io_gb: Total network I/O in GB

    Returns:
        Dict with recommendation, or None if no change recommended
    """
    family, size = get_instance_family_and_size(current_type)
    if not family:
        return None

    # Instance size ordering
    sizes = ['nano', 'micro', 'small', 'medium', 'large', 'xlarge', '2xlarge', '4xlarge', '8xlarge', '12xlarge', '16xlarge', '24xlarge']

    try:
        current_size_idx = sizes.index(size)
    except ValueError:
        return None

    # Decision logic
    # Over-provisioned if CPU <30% and Memory <40% (if available)
    # Under-provisioned if CPU >70% or Memory >80%

    if cpu_p95 > 70 or (memory_p95 and memory_p95 > 80):
        # Under-provisioned: recommend larger instance
        if current_size_idx < len(sizes) - 1:
            recommended_size = sizes[current_size_idx + 1]
            recommended_type = f"{family}.{recommended_size}"
            action = "UPSIZE"
            reason = f"High utilization (CPU: {cpu_p95:.1f}%"
            if memory_p95:
                reason += f", Memory: {memory_p95:.1f}%"
            reason += ")"
        else:
            return None  # Already at max size

    elif cpu_p95 < 30 and (memory_p95 is None or memory_p95 < 40):
        # Over-provisioned: recommend smaller instance
        if current_size_idx > 0:
            recommended_size = sizes[current_size_idx - 1]
            recommended_type = f"{family}.{recommended_size}"
            action = "DOWNSIZE"
            reason = f"Low utilization (CPU: {cpu_p95:.1f}%"
            if memory_p95:
                reason += f", Memory: {memory_p95:.1f}%"
            reason += ")"
        else:
            return None  # Already at min size

    else:
        # Appropriately sized
        return None

    # Calculate savings
    current_cost = EC2_PRICING.get(current_type, 0)
    recommended_cost = EC2_PRICING.get(recommended_type, 0)

    if current_cost == 0 or recommended_cost == 0:
        # Pricing not available
        savings_hourly = 0
        savings_monthly = 0
    else:
        savings_hourly = current_cost - recommended_cost
        savings_monthly = savings_hourly * 730  # hours per month

    return {
        'action': action,
        'recommended_type': recommended_type,
        'current_cost_hourly': current_cost,
        'recommended_cost_hourly': recommended_cost,
        'savings_hourly': savings_hourly,
        'savings_monthly': savings_monthly,
        'reason': reason
    }


def analyze_instance(
    instance: Dict,
    days: int,
    percentile: int
) -> Optional[Dict]:
    """
    Analyze a single instance and provide recommendation.

    Args:
        instance: EC2 instance dict
        days: Number of days to analyze
        percentile: Percentile for max calculations

    Returns:
        Dict with analysis and recommendation
    """
    instance_id = instance['InstanceId']
    instance_type = instance['InstanceType']
    instance_state = instance['State']['Name']

    # Skip stopped instances
    if instance_state != 'running':
        return None

    # Get tags
    tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    print(f"Analyzing {instance_id} ({instance_type})...", end=' ')

    # Get metrics
    metrics = get_instance_metrics(instance_id, start_time, end_time, percentile)

    if not metrics:
        print("No metrics available")
        return None

    if metrics['datapoints'] < 24:
        print(f"Insufficient data ({metrics['datapoints']} datapoints)")
        return None

    print(f"CPU: {metrics['cpu_p95']:.1f}%", end='')
    if metrics['memory_p95']:
        print(f", Memory: {metrics['memory_p95']:.1f}%", end='')

    # Get recommendation
    network_io = metrics['network_in_gb'] + metrics['network_out_gb']
    recommendation = recommend_instance_type(
        instance_type,
        metrics['cpu_p95'],
        metrics['memory_p95'],
        network_io
    )

    if recommendation:
        print(f" → {recommendation['action']}: {recommendation['recommended_type']} (${recommendation['savings_monthly']:.2f}/month)")
    else:
        print(" → OK (appropriately sized)")

    return {
        'instance_id': instance_id,
        'instance_type': instance_type,
        'environment': tags.get('environment', 'unknown'),
        'team': tags.get('team', 'unknown'),
        'project': tags.get('project', 'unknown'),
        'name': tags.get('Name', ''),
        'metrics': metrics,
        'recommendation': recommendation
    }


def print_summary(results: List[Dict]):
    """Print summary of recommendations."""
    print()
    print("=" * 80)
    print("  Right-Sizing Summary")
    print("=" * 80)
    print()

    # Filter results with recommendations
    with_recommendations = [r for r in results if r['recommendation']]

    if not with_recommendations:
        print("✓ All instances are appropriately sized!")
        print()
        return

    # Calculate total savings
    total_savings_monthly = sum(
        r['recommendation']['savings_monthly']
        for r in with_recommendations
    )

    print(f"Instances analyzed: {len(results)}")
    print(f"Recommendations: {len(with_recommendations)}")
    print(f"Potential monthly savings: ${total_savings_monthly:,.2f}")
    print(f"Potential annual savings: ${total_savings_monthly * 12:,.2f}")
    print()

    # Group by action
    upsize = [r for r in with_recommendations if r['recommendation']['action'] == 'UPSIZE']
    downsize = [r for r in with_recommendations if r['recommendation']['action'] == 'DOWNSIZE']

    if downsize:
        print(f"DOWNSIZE Recommendations ({len(downsize)} instances):")
        print("-" * 80)

        # Sort by savings
        downsize_sorted = sorted(
            downsize,
            key=lambda x: x['recommendation']['savings_monthly'],
            reverse=True
        )

        for i, result in enumerate(downsize_sorted[:10], 1):  # Top 10
            rec = result['recommendation']
            metrics = result['metrics']

            print(f"{i}. {result['instance_id']} ({result['name']})")
            print(f"   Current: {result['instance_type']} (${rec['current_cost_hourly']:.3f}/hour)")
            print(f"   Recommended: {rec['recommended_type']} (${rec['recommended_cost_hourly']:.3f}/hour)")
            print(f"   Utilization: CPU p95={metrics['cpu_p95']:.1f}%", end='')
            if metrics['memory_p95']:
                print(f", Memory p95={metrics['memory_p95']:.1f}%", end='')
            print()
            print(f"   Savings: ${rec['savings_monthly']:.2f}/month")
            print(f"   Environment: {result['environment']}, Team: {result['team']}")
            print()

        if len(downsize_sorted) > 10:
            print(f"   ... and {len(downsize_sorted) - 10} more")
            print()

    if upsize:
        print()
        print(f"UPSIZE Recommendations ({len(upsize)} instances):")
        print("-" * 80)
        print("⚠️  These instances may be under-provisioned and could affect performance")
        print()

        for i, result in enumerate(upsize, 1):
            rec = result['recommendation']
            metrics = result['metrics']

            print(f"{i}. {result['instance_id']} ({result['name']})")
            print(f"   Current: {result['instance_type']}")
            print(f"   Recommended: {rec['recommended_type']}")
            print(f"   Utilization: CPU p95={metrics['cpu_p95']:.1f}%", end='')
            if metrics['memory_p95']:
                print(f", Memory p95={metrics['memory_p95']:.1f}%", end='')
            print()
            print(f"   Additional cost: ${abs(rec['savings_monthly']):.2f}/month")
            print(f"   Environment: {result['environment']}, Team: {result['team']}")
            print()


def export_recommendations(results: List[Dict], filename: str):
    """Export recommendations to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Recommendations exported to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze EC2 instance utilization and provide right-sizing recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze last 30 days with p95
  %(prog)s --days 30

  # More aggressive recommendations (p99)
  %(prog)s --days 14 --percentile 99

  # Export recommendations for automation
  %(prog)s --days 30 --export recommendations.json

  # Filter by environment
  %(prog)s --days 30 --environment production
        """
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to analyze (default: 30)'
    )
    parser.add_argument(
        '--percentile',
        type=int,
        default=95,
        choices=[90, 95, 99],
        help='Percentile for utilization calculation (default: 95)'
    )
    parser.add_argument(
        '--environment',
        help='Filter by environment tag (e.g., production, staging)'
    )
    parser.add_argument(
        '--team',
        help='Filter by team tag'
    )
    parser.add_argument(
        '--export',
        help='Export recommendations to JSON file'
    )
    parser.add_argument(
        '--min-savings',
        type=float,
        default=0,
        help='Only show recommendations with savings >= this amount ($/month)'
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("  EC2 Instance Right-Sizing Analysis")
    print("=" * 80)
    print()
    print(f"Analysis period: Last {args.days} days")
    print(f"Percentile: p{args.percentile}")
    print()

    try:
        ec2 = boto3.client('ec2')

        # Build filters
        filters = [{'Name': 'instance-state-name', 'Values': ['running']}]

        if args.environment:
            filters.append({'Name': 'tag:environment', 'Values': [args.environment]})

        if args.team:
            filters.append({'Name': 'tag:team', 'Values': [args.team]})

        # Get instances
        print("Fetching EC2 instances...")
        response = ec2.describe_instances(Filters=filters)

        instances = []
        for reservation in response['Reservations']:
            instances.extend(reservation['Instances'])

        print(f"Found {len(instances)} running instances")
        print()

        # Analyze each instance
        results = []
        for instance in instances:
            result = analyze_instance(instance, args.days, args.percentile)
            if result:
                results.append(result)

        # Filter by minimum savings
        if args.min_savings > 0:
            results = [
                r for r in results
                if r['recommendation'] and r['recommendation']['savings_monthly'] >= args.min_savings
            ]

        # Print summary
        print_summary(results)

        # Export if requested
        if args.export:
            export_recommendations(results, args.export)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Note: This script requires:")
        print("1. AWS credentials with EC2 and CloudWatch read access")
        print("2. CloudWatch agent installed on instances for memory metrics")
        print()
        print("Run 'aws configure' to set up credentials")
        return 1


if __name__ == '__main__':
    sys.exit(main())
