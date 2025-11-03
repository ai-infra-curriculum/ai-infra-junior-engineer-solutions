#!/usr/bin/env python3
"""
AWS Auto-Shutdown Lambda Function

Automatically start/stop EC2 instances and RDS databases based on schedules
to reduce costs for non-production environments.

This can be deployed as an AWS Lambda function triggered by EventBridge (CloudWatch Events).

Usage (Local Testing):
    python auto_shutdown.py --action stop --environment development
    python auto_shutdown.py --action start --environment staging
    python auto_shutdown.py --dry-run

Usage (Lambda Deployment):
    Deploy this file as Lambda function and trigger with EventBridge rules:
    - Shutdown: cron(0 20 * * ? *)  # 8 PM daily
    - Startup:  cron(0 8 ? * MON-FRI *)  # 8 AM weekdays
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict
import boto3


def get_instances_to_manage(
    environment: str = None,
    auto_shutdown_tag: str = None
) -> List[Dict]:
    """
    Get EC2 instances that should be managed by auto-shutdown.

    Args:
        environment: Filter by environment tag (development, staging, etc.)
        auto_shutdown_tag: Filter by auto-shutdown tag value

    Returns:
        List of instance dicts
    """
    ec2 = boto3.client('ec2')

    filters = []

    # Filter by auto-shutdown tag
    if auto_shutdown_tag:
        filters.append({
            'Name': 'tag:auto-shutdown',
            'Values': [auto_shutdown_tag]
        })
    else:
        # Default: any instance with auto-shutdown tag
        filters.append({
            'Name': 'tag-key',
            'Values': ['auto-shutdown']
        })

    # Filter by environment
    if environment:
        filters.append({
            'Name': 'tag:environment',
            'Values': [environment]
        })

    response = ec2.describe_instances(Filters=filters)

    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}

            instances.append({
                'instance_id': instance['InstanceId'],
                'instance_type': instance['InstanceType'],
                'state': instance['State']['Name'],
                'environment': tags.get('environment', 'unknown'),
                'auto_shutdown': tags.get('auto-shutdown', 'unknown'),
                'team': tags.get('team', 'unknown'),
                'name': tags.get('Name', ''),
                'launch_time': instance.get('LaunchTime')
            })

    return instances


def get_rds_instances_to_manage(
    environment: str = None
) -> List[Dict]:
    """
    Get RDS instances that should be managed by auto-shutdown.

    Args:
        environment: Filter by environment tag

    Returns:
        List of RDS instance dicts
    """
    rds = boto3.client('rds')

    try:
        response = rds.describe_db_instances()

        rds_instances = []
        for db_instance in response['DBInstances']:
            db_identifier = db_instance['DBInstanceIdentifier']

            # Get tags
            db_arn = db_instance['DBInstanceArn']
            tags_response = rds.list_tags_for_resource(ResourceName=db_arn)
            tags = {tag['Key']: tag['Value'] for tag in tags_response['TagList']}

            # Check if auto-shutdown is enabled
            auto_shutdown = tags.get('auto-shutdown', '')

            if auto_shutdown in ['enabled', 'nights', 'weekends', 'nights-weekends']:
                # Filter by environment if specified
                db_environment = tags.get('environment', 'unknown')

                if environment and db_environment != environment:
                    continue

                rds_instances.append({
                    'db_identifier': db_identifier,
                    'db_instance_class': db_instance['DBInstanceClass'],
                    'status': db_instance['DBInstanceStatus'],
                    'environment': db_environment,
                    'auto_shutdown': auto_shutdown,
                    'team': tags.get('team', 'unknown')
                })

        return rds_instances

    except Exception as e:
        print(f"Warning: Could not fetch RDS instances: {e}")
        return []


def should_shutdown(auto_shutdown_value: str, current_time: datetime) -> bool:
    """
    Determine if resource should be shut down based on schedule.

    Args:
        auto_shutdown_value: Value of auto-shutdown tag
        current_time: Current datetime

    Returns:
        True if should shutdown, False otherwise
    """
    hour = current_time.hour
    weekday = current_time.weekday()  # 0=Monday, 6=Sunday
    is_weekend = weekday >= 5

    if auto_shutdown_value == 'enabled':
        # Shutdown outside business hours (8 PM - 8 AM)
        return hour >= 20 or hour < 8

    elif auto_shutdown_value == 'nights':
        # Shutdown every night (8 PM - 8 AM)
        return hour >= 20 or hour < 8

    elif auto_shutdown_value == 'weekends':
        # Shutdown on weekends only
        return is_weekend

    elif auto_shutdown_value == 'nights-weekends':
        # Shutdown nights and weekends
        return (hour >= 20 or hour < 8) or is_weekend

    elif auto_shutdown_value == 'disabled':
        # Never shutdown
        return False

    else:
        # Unknown value, don't shutdown
        return False


def should_startup(auto_shutdown_value: str, current_time: datetime) -> bool:
    """
    Determine if resource should be started based on schedule.

    Args:
        auto_shutdown_value: Value of auto-shutdown tag
        current_time: Current datetime

    Returns:
        True if should startup, False otherwise
    """
    hour = current_time.hour
    weekday = current_time.weekday()
    is_weekday = weekday < 5
    is_weekend = weekday >= 5

    if auto_shutdown_value == 'enabled':
        # Startup during business hours (8 AM) on weekdays
        return hour >= 8 and hour < 20 and is_weekday

    elif auto_shutdown_value == 'nights':
        # Startup at 8 AM
        return hour >= 8 and hour < 20

    elif auto_shutdown_value == 'weekends':
        # Startup on weekdays
        return is_weekday

    elif auto_shutdown_value == 'nights-weekends':
        # Startup at 8 AM on weekdays
        return hour >= 8 and hour < 20 and is_weekday

    elif auto_shutdown_value == 'disabled':
        # Never startup (manual control)
        return False

    else:
        return False


def stop_ec2_instances(instance_ids: List[str], dry_run: bool = False) -> Dict:
    """Stop EC2 instances."""
    if not instance_ids:
        return {'stopped': 0, 'failed': 0}

    ec2 = boto3.client('ec2')

    if dry_run:
        print(f"[DRY RUN] Would stop {len(instance_ids)} instances: {instance_ids}")
        return {'stopped': len(instance_ids), 'failed': 0}

    try:
        response = ec2.stop_instances(InstanceIds=instance_ids)

        stopped = len(response['StoppingInstances'])
        failed = len(instance_ids) - stopped

        for instance in response['StoppingInstances']:
            print(f"✓ Stopped {instance['InstanceId']}: {instance['PreviousState']['Name']} → {instance['CurrentState']['Name']}")

        return {'stopped': stopped, 'failed': failed}

    except Exception as e:
        print(f"✗ Error stopping instances: {e}")
        return {'stopped': 0, 'failed': len(instance_ids)}


def start_ec2_instances(instance_ids: List[str], dry_run: bool = False) -> Dict:
    """Start EC2 instances."""
    if not instance_ids:
        return {'started': 0, 'failed': 0}

    ec2 = boto3.client('ec2')

    if dry_run:
        print(f"[DRY RUN] Would start {len(instance_ids)} instances: {instance_ids}")
        return {'started': len(instance_ids), 'failed': 0}

    try:
        response = ec2.start_instances(InstanceIds=instance_ids)

        started = len(response['StartingInstances'])
        failed = len(instance_ids) - started

        for instance in response['StartingInstances']:
            print(f"✓ Started {instance['InstanceId']}: {instance['PreviousState']['Name']} → {instance['CurrentState']['Name']}")

        return {'started': started, 'failed': failed}

    except Exception as e:
        print(f"✗ Error starting instances: {e}")
        return {'started': 0, 'failed': len(instance_ids)}


def stop_rds_instances(db_identifiers: List[str], dry_run: bool = False) -> Dict:
    """Stop RDS instances."""
    if not db_identifiers:
        return {'stopped': 0, 'failed': 0}

    rds = boto3.client('rds')

    stopped = 0
    failed = 0

    for db_identifier in db_identifiers:
        if dry_run:
            print(f"[DRY RUN] Would stop RDS instance: {db_identifier}")
            stopped += 1
            continue

        try:
            rds.stop_db_instance(DBInstanceIdentifier=db_identifier)
            print(f"✓ Stopped RDS instance: {db_identifier}")
            stopped += 1
        except Exception as e:
            print(f"✗ Failed to stop RDS instance {db_identifier}: {e}")
            failed += 1

    return {'stopped': stopped, 'failed': failed}


def start_rds_instances(db_identifiers: List[str], dry_run: bool = False) -> Dict:
    """Start RDS instances."""
    if not db_identifiers:
        return {'started': 0, 'failed': 0}

    rds = boto3.client('rds')

    started = 0
    failed = 0

    for db_identifier in db_identifiers:
        if dry_run:
            print(f"[DRY RUN] Would start RDS instance: {db_identifier}")
            started += 1
            continue

        try:
            rds.start_db_instance(DBInstanceIdentifier=db_identifier)
            print(f"✓ Started RDS instance: {db_identifier}")
            started += 1
        except Exception as e:
            print(f"✗ Failed to start RDS instance {db_identifier}: {e}")
            failed += 1

    return {'started': started, 'failed': failed}


def lambda_handler(event, context):
    """
    AWS Lambda handler for auto-shutdown.

    Expected event format:
    {
        "action": "stop" or "start",
        "environment": "development" (optional),
        "dry_run": false
    }
    """
    action = event.get('action', 'evaluate')  # stop, start, or evaluate
    environment = event.get('environment')
    dry_run = event.get('dry_run', False)

    current_time = datetime.utcnow()

    print(f"Auto-shutdown Lambda triggered at {current_time}")
    print(f"Action: {action}, Environment: {environment}, Dry run: {dry_run}")

    # Get instances to manage
    ec2_instances = get_instances_to_manage(environment)
    rds_instances = get_rds_instances_to_manage(environment)

    print(f"Found {len(ec2_instances)} EC2 instances with auto-shutdown enabled")
    print(f"Found {len(rds_instances)} RDS instances with auto-shutdown enabled")

    # Separate by action needed
    ec2_to_stop = []
    ec2_to_start = []
    rds_to_stop = []
    rds_to_start = []

    # Evaluate EC2 instances
    for instance in ec2_instances:
        auto_shutdown_value = instance['auto_shutdown']

        if action == 'stop':
            # Explicit stop action
            if instance['state'] == 'running':
                ec2_to_stop.append(instance['instance_id'])
        elif action == 'start':
            # Explicit start action
            if instance['state'] == 'stopped':
                ec2_to_start.append(instance['instance_id'])
        else:
            # Evaluate based on schedule
            if instance['state'] == 'running' and should_shutdown(auto_shutdown_value, current_time):
                ec2_to_stop.append(instance['instance_id'])
            elif instance['state'] == 'stopped' and should_startup(auto_shutdown_value, current_time):
                ec2_to_start.append(instance['instance_id'])

    # Evaluate RDS instances
    for db_instance in rds_instances:
        auto_shutdown_value = db_instance['auto_shutdown']

        if action == 'stop':
            if db_instance['status'] == 'available':
                rds_to_stop.append(db_instance['db_identifier'])
        elif action == 'start':
            if db_instance['status'] == 'stopped':
                rds_to_start.append(db_instance['db_identifier'])
        else:
            if db_instance['status'] == 'available' and should_shutdown(auto_shutdown_value, current_time):
                rds_to_stop.append(db_instance['db_identifier'])
            elif db_instance['status'] == 'stopped' and should_startup(auto_shutdown_value, current_time):
                rds_to_start.append(db_instance['db_identifier'])

    # Execute actions
    results = {
        'ec2_stopped': 0,
        'ec2_started': 0,
        'rds_stopped': 0,
        'rds_started': 0,
        'failed': 0
    }

    # Stop instances
    if ec2_to_stop:
        print(f"Stopping {len(ec2_to_stop)} EC2 instances...")
        stop_result = stop_ec2_instances(ec2_to_stop, dry_run)
        results['ec2_stopped'] = stop_result['stopped']
        results['failed'] += stop_result['failed']

    if rds_to_stop:
        print(f"Stopping {len(rds_to_stop)} RDS instances...")
        stop_result = stop_rds_instances(rds_to_stop, dry_run)
        results['rds_stopped'] = stop_result['stopped']
        results['failed'] += stop_result['failed']

    # Start instances
    if ec2_to_start:
        print(f"Starting {len(ec2_to_start)} EC2 instances...")
        start_result = start_ec2_instances(ec2_to_start, dry_run)
        results['ec2_started'] = start_result['started']
        results['failed'] += start_result['failed']

    if rds_to_start:
        print(f"Starting {len(rds_to_start)} RDS instances...")
        start_result = start_rds_instances(rds_to_start, dry_run)
        results['rds_started'] = start_result['started']
        results['failed'] += start_result['failed']

    # Summary
    print()
    print("=" * 60)
    print("Auto-Shutdown Summary")
    print("=" * 60)
    print(f"EC2 instances stopped: {results['ec2_stopped']}")
    print(f"EC2 instances started: {results['ec2_started']}")
    print(f"RDS instances stopped: {results['rds_stopped']}")
    print(f"RDS instances started: {results['rds_started']}")
    print(f"Failed actions: {results['failed']}")
    print()

    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }


def main():
    """CLI interface for testing."""
    parser = argparse.ArgumentParser(
        description="Auto-shutdown/startup EC2 and RDS instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate and execute based on current time
  %(prog)s --evaluate

  # Explicit stop (for testing)
  %(prog)s --action stop --environment development

  # Explicit start (for testing)
  %(prog)s --action start --environment staging

  # Dry run (preview only)
  %(prog)s --action stop --dry-run

Lambda Deployment:
  1. Deploy this file as Lambda function
  2. Set appropriate IAM role (EC2, RDS permissions)
  3. Create EventBridge rules:
     - Shutdown: cron(0 20 * * ? *)
     - Startup:  cron(0 8 ? * MON-FRI *)

  4. Configure event payload:
     {"action": "evaluate"}
        """
    )

    parser.add_argument(
        '--action',
        choices=['stop', 'start', 'evaluate'],
        default='evaluate',
        help='Action to perform (default: evaluate based on schedule)'
    )
    parser.add_argument(
        '--environment',
        help='Filter by environment tag (development, staging, etc.)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview actions without executing'
    )

    args = parser.parse_args()

    # Build event for lambda_handler
    event = {
        'action': args.action,
        'environment': args.environment,
        'dry_run': args.dry_run
    }

    # Mock Lambda context
    class MockContext:
        function_name = 'auto-shutdown'
        memory_limit_in_mb = 128
        invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:auto-shutdown'

    context = MockContext()

    # Execute
    result = lambda_handler(event, context)

    return 0 if result['statusCode'] == 200 else 1


if __name__ == '__main__':
    sys.exit(main())
