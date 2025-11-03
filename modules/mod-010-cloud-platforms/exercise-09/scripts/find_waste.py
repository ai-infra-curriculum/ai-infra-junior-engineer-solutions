#!/usr/bin/env python3
"""
Cloud Waste Detection Tool

Identify idle and underutilized cloud resources to reduce costs.

Usage:
    python find_waste.py --all
    python find_waste.py --ec2-only
    python find_waste.py --detailed --days 14
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Dict
import boto3


# Pricing data (us-east-1, approximate)
PRICING = {
    # EC2 on-demand hourly rates
    'ec2': {
        't3.nano': 0.0052, 't3.micro': 0.0104, 't3.small': 0.0208,
        't3.medium': 0.0416, 't3.large': 0.0832, 't3.xlarge': 0.1664,
        't3.2xlarge': 0.3328,
        'm5.large': 0.096, 'm5.xlarge': 0.192, 'm5.2xlarge': 0.384,
        'm5.4xlarge': 0.768, 'm5.8xlarge': 1.536,
        'c5.large': 0.085, 'c5.xlarge': 0.17, 'c5.2xlarge': 0.34,
        'r5.large': 0.126, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504,
        'p3.2xlarge': 3.06, 'p3.8xlarge': 12.24, 'p3.16xlarge': 24.48,
    },
    # EBS per GB-month
    'ebs': {
        'gp2': 0.10, 'gp3': 0.08, 'io1': 0.125, 'io2': 0.125,
        'st1': 0.045, 'sc1': 0.015, 'standard': 0.05
    },
    # EBS snapshots per GB-month
    'snapshot': 0.05,
    # Elastic IP (per hour if not attached)
    'eip': 0.005,
    # Load balancer per hour
    'elb': 0.025,
    'alb': 0.0225,
    'nlb': 0.0225,
}


def find_idle_ec2_instances(threshold_cpu: float = 5.0, days: int = 7) -> List[Dict]:
    """
    Find EC2 instances with low CPU utilization.

    Args:
        threshold_cpu: Max average CPU % to consider idle
        days: Days to analyze

    Returns:
        List of idle instances with details
    """
    ec2 = boto3.client('ec2')
    cloudwatch = boto3.client('cloudwatch')

    print(f"Scanning for idle EC2 instances (CPU <{threshold_cpu}% for {days} days)...")

    # Get all running instances
    response = ec2.describe_instances(
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
    )

    idle_instances = []

    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            instance_type = instance['InstanceType']

            # Get tags
            tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
            name = tags.get('Name', 'N/A')

            # Get CPU metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            try:
                metrics = cloudwatch.get_metric_statistics(
                    Namespace='AWS/EC2',
                    MetricName='CPUUtilization',
                    Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,  # 1 hour
                    Statistics=['Average']
                )

                if not metrics['Datapoints']:
                    continue

                avg_cpu = sum(d['Average'] for d in metrics['Datapoints']) / len(metrics['Datapoints'])

                if avg_cpu < threshold_cpu:
                    # Calculate monthly cost
                    cost_per_hour = PRICING['ec2'].get(instance_type, 0.10)
                    monthly_cost = cost_per_hour * 24 * 30

                    idle_instances.append({
                        'instance_id': instance_id,
                        'name': name,
                        'instance_type': instance_type,
                        'avg_cpu': avg_cpu,
                        'cost_per_hour': cost_per_hour,
                        'monthly_cost': monthly_cost,
                        'potential_savings': monthly_cost,
                        'environment': tags.get('environment', 'N/A'),
                        'team': tags.get('team', 'N/A')
                    })

            except Exception as e:
                print(f"Warning: Could not get metrics for {instance_id}: {e}")
                continue

    return idle_instances


def find_unattached_ebs_volumes() -> List[Dict]:
    """Find EBS volumes not attached to any instance."""
    ec2 = boto3.client('ec2')

    print("Scanning for unattached EBS volumes...")

    response = ec2.describe_volumes(
        Filters=[{'Name': 'status', 'Values': ['available']}]
    )

    unattached = []
    for volume in response['Volumes']:
        volume_id = volume['VolumeId']
        size = volume['Size']
        volume_type = volume['VolumeType']
        create_time = volume['CreateTime']

        # Get tags
        tags = {tag['Key']: tag['Value'] for tag in volume.get('Tags', [])}
        name = tags.get('Name', 'N/A')

        # Calculate age
        age_days = (datetime.now(create_time.tzinfo) - create_time).days

        # Cost per GB-month
        cost_per_gb = PRICING['ebs'].get(volume_type, 0.10)
        monthly_cost = size * cost_per_gb

        unattached.append({
            'volume_id': volume_id,
            'name': name,
            'size_gb': size,
            'volume_type': volume_type,
            'age_days': age_days,
            'cost_per_gb': cost_per_gb,
            'monthly_cost': monthly_cost,
            'potential_savings': monthly_cost
        })

    return unattached


def find_old_ebs_snapshots(threshold_days: int = 90) -> List[Dict]:
    """Find old EBS snapshots that could be deleted."""
    ec2 = boto3.client('ec2')

    print(f"Scanning for old EBS snapshots (>{threshold_days} days)...")

    response = ec2.describe_snapshots(OwnerIds=['self'])

    old_snapshots = []
    for snapshot in response['Snapshots']:
        snapshot_id = snapshot['SnapshotId']
        size = snapshot['VolumeSize']
        start_time = snapshot['StartTime']

        # Get tags
        tags = {tag['Key']: tag['Value'] for tag in snapshot.get('Tags', [])}
        name = tags.get('Name', 'N/A')

        # Calculate age
        age_days = (datetime.now(start_time.tzinfo) - start_time).days

        if age_days > threshold_days:
            monthly_cost = size * PRICING['snapshot']

            old_snapshots.append({
                'snapshot_id': snapshot_id,
                'name': name,
                'size_gb': size,
                'age_days': age_days,
                'monthly_cost': monthly_cost,
                'potential_savings': monthly_cost
            })

    return old_snapshots


def find_unused_elastic_ips() -> List[Dict]:
    """Find Elastic IPs not associated with any instance."""
    ec2 = boto3.client('ec2')

    print("Scanning for unused Elastic IPs...")

    response = ec2.describe_addresses()

    unused_eips = []
    for address in response['Addresses']:
        allocation_id = address.get('AllocationId', 'N/A')
        public_ip = address.get('PublicIp', 'N/A')

        # Check if associated
        if 'AssociationId' not in address:
            monthly_cost = PRICING['eip'] * 24 * 30

            unused_eips.append({
                'allocation_id': allocation_id,
                'public_ip': public_ip,
                'monthly_cost': monthly_cost,
                'potential_savings': monthly_cost
            })

    return unused_eips


def find_idle_rds_instances(threshold_connections: int = 5, days: int = 7) -> List[Dict]:
    """Find RDS instances with low connection count."""
    rds = boto3.client('rds')
    cloudwatch = boto3.client('cloudwatch')

    print(f"Scanning for idle RDS instances (<{threshold_connections} connections for {days} days)...")

    response = rds.describe_db_instances()

    idle_databases = []

    for db in response['DBInstances']:
        db_identifier = db['DBInstanceIdentifier']
        db_class = db['DBInstanceClass']
        engine = db['Engine']

        # Get connection metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        try:
            metrics = cloudwatch.get_metric_statistics(
                Namespace='AWS/RDS',
                MetricName='DatabaseConnections',
                Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_identifier}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average']
            )

            if not metrics['Datapoints']:
                continue

            avg_connections = sum(d['Average'] for d in metrics['Datapoints']) / len(metrics['Datapoints'])

            if avg_connections < threshold_connections:
                # Rough cost estimate (varies by instance class and engine)
                monthly_cost = 100  # Placeholder

                idle_databases.append({
                    'db_identifier': db_identifier,
                    'db_class': db_class,
                    'engine': engine,
                    'avg_connections': avg_connections,
                    'monthly_cost': monthly_cost,
                    'potential_savings': monthly_cost
                })

        except Exception as e:
            print(f"Warning: Could not get metrics for {db_identifier}: {e}")
            continue

    return idle_databases


def print_waste_report(
    idle_ec2: List[Dict],
    unattached_ebs: List[Dict],
    old_snapshots: List[Dict],
    unused_eips: List[Dict],
    idle_rds: List[Dict],
    detailed: bool = False
):
    """Print comprehensive waste detection report."""
    print()
    print("=" * 80)
    print("  Cloud Waste Detection Report")
    print("=" * 80)
    print()

    total_savings = 0

    # Idle EC2 Instances
    print("[1] Idle EC2 Instances")
    print("-" * 80)
    if idle_ec2:
        ec2_savings = sum(i['potential_savings'] for i in idle_ec2)
        total_savings += ec2_savings

        print(f"Found: {len(idle_ec2)} idle instances")
        print(f"Potential monthly savings: ${ec2_savings:,.2f}")
        print()

        if detailed:
            for instance in sorted(idle_ec2, key=lambda x: x['monthly_cost'], reverse=True)[:10]:
                print(f"  Instance: {instance['instance_id']} ({instance['name']})")
                print(f"  Type: {instance['instance_type']}")
                print(f"  Avg CPU: {instance['avg_cpu']:.1f}%")
                print(f"  Environment: {instance['environment']}")
                print(f"  Team: {instance['team']}")
                print(f"  Monthly cost: ${instance['monthly_cost']:.2f}")
                print()
    else:
        print("✓ No idle EC2 instances found")
        print()

    # Unattached EBS Volumes
    print("[2] Unattached EBS Volumes")
    print("-" * 80)
    if unattached_ebs:
        ebs_savings = sum(v['potential_savings'] for v in unattached_ebs)
        total_savings += ebs_savings

        total_size = sum(v['size_gb'] for v in unattached_ebs)
        print(f"Found: {len(unattached_ebs)} unattached volumes ({total_size:,} GB)")
        print(f"Potential monthly savings: ${ebs_savings:,.2f}")
        print()

        if detailed:
            for volume in sorted(unattached_ebs, key=lambda x: x['monthly_cost'], reverse=True)[:10]:
                print(f"  Volume: {volume['volume_id']} ({volume['name']})")
                print(f"  Size: {volume['size_gb']} GB ({volume['volume_type']})")
                print(f"  Age: {volume['age_days']} days")
                print(f"  Monthly cost: ${volume['monthly_cost']:.2f}")
                print()
    else:
        print("✓ No unattached EBS volumes found")
        print()

    # Old EBS Snapshots
    print("[3] Old EBS Snapshots")
    print("-" * 80)
    if old_snapshots:
        snapshot_savings = sum(s['potential_savings'] for s in old_snapshots)
        total_savings += snapshot_savings

        total_size = sum(s['size_gb'] for s in old_snapshots)
        print(f"Found: {len(old_snapshots)} old snapshots ({total_size:,} GB)")
        print(f"Potential monthly savings: ${snapshot_savings:,.2f}")
        print()

        if detailed:
            for snapshot in sorted(old_snapshots, key=lambda x: x['age_days'], reverse=True)[:10]:
                print(f"  Snapshot: {snapshot['snapshot_id']} ({snapshot['name']})")
                print(f"  Size: {snapshot['size_gb']} GB")
                print(f"  Age: {snapshot['age_days']} days")
                print(f"  Monthly cost: ${snapshot['monthly_cost']:.2f}")
                print()
    else:
        print("✓ No old snapshots found")
        print()

    # Unused Elastic IPs
    print("[4] Unused Elastic IPs")
    print("-" * 80)
    if unused_eips:
        eip_savings = sum(e['potential_savings'] for e in unused_eips)
        total_savings += eip_savings

        print(f"Found: {len(unused_eips)} unused Elastic IPs")
        print(f"Potential monthly savings: ${eip_savings:,.2f}")
        print()

        if detailed:
            for eip in unused_eips:
                print(f"  EIP: {eip['public_ip']} (Allocation: {eip['allocation_id']})")
                print(f"  Monthly cost: ${eip['monthly_cost']:.2f}")
                print()
    else:
        print("✓ No unused Elastic IPs found")
        print()

    # Idle RDS Instances
    print("[5] Idle RDS Instances")
    print("-" * 80)
    if idle_rds:
        rds_savings = sum(d['potential_savings'] for d in idle_rds)
        total_savings += rds_savings

        print(f"Found: {len(idle_rds)} idle RDS instances")
        print(f"Potential monthly savings: ${rds_savings:,.2f}")
        print()

        if detailed:
            for db in idle_rds:
                print(f"  Database: {db['db_identifier']}")
                print(f"  Class: {db['db_class']} ({db['engine']})")
                print(f"  Avg connections: {db['avg_connections']:.1f}")
                print(f"  Estimated monthly cost: ${db['monthly_cost']:.2f}")
                print()
    else:
        print("✓ No idle RDS instances found")
        print()

    # Summary
    print("=" * 80)
    print("  Summary")
    print("=" * 80)
    print()
    print(f"Total Potential Monthly Savings: ${total_savings:,.2f}")
    print(f"Annual Savings: ${total_savings * 12:,.2f}")
    print()

    # Recommendations
    print("Recommended Actions:")
    print("1. Review idle EC2 instances - consider termination or auto-shutdown")
    print("2. Delete unattached EBS volumes after backup verification")
    print("3. Implement snapshot retention policy (30-90 days)")
    print("4. Release unused Elastic IPs")
    print("5. Consider downgrading or stopping idle RDS instances")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Detect cloud waste and identify cost savings opportunities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all resource types
  %(prog)s --all --detailed

  # Scan only EC2 instances
  %(prog)s --ec2-only --days 14

  # Scan with custom thresholds
  %(prog)s --all --cpu-threshold 3.0 --snapshot-days 60
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Scan all resource types'
    )
    parser.add_argument(
        '--ec2-only',
        action='store_true',
        help='Scan only EC2 instances'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed information for each resource'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Days to analyze for CPU/connection metrics (default: 7)'
    )
    parser.add_argument(
        '--cpu-threshold',
        type=float,
        default=5.0,
        help='CPU threshold for idle detection (default: 5.0%%)'
    )
    parser.add_argument(
        '--snapshot-days',
        type=int,
        default=90,
        help='Age threshold for old snapshots in days (default: 90)'
    )

    args = parser.parse_args()

    if not (args.all or args.ec2_only):
        parser.print_help()
        return 1

    print()
    print("=" * 80)
    print("  Cloud Waste Detection")
    print("=" * 80)
    print()
    print(f"Analysis period: {args.days} days")
    print(f"CPU threshold: {args.cpu_threshold}%")
    print()

    try:
        # Scan resources
        idle_ec2 = find_idle_ec2_instances(args.cpu_threshold, args.days)

        if args.all:
            unattached_ebs = find_unattached_ebs_volumes()
            old_snapshots = find_old_ebs_snapshots(args.snapshot_days)
            unused_eips = find_unused_elastic_ips()
            idle_rds = find_idle_rds_instances(5, args.days)
        else:
            unattached_ebs = []
            old_snapshots = []
            unused_eips = []
            idle_rds = []

        # Print report
        print_waste_report(
            idle_ec2,
            unattached_ebs,
            old_snapshots,
            unused_eips,
            idle_rds,
            args.detailed
        )

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Note: This script requires AWS credentials with appropriate permissions.")
        print("Run 'aws configure' to set up credentials.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
