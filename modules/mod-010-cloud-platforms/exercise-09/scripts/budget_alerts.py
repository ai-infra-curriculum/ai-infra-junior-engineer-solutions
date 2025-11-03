#!/usr/bin/env python3
"""
AWS Budget and Alert Management Tool

Create and manage AWS Budgets with email/SNS notifications to control costs.

Usage:
    python budget_alerts.py --create --name "Monthly-Budget" --amount 50000
    python budget_alerts.py --list
    python budget_alerts.py --delete --name "Monthly-Budget"
"""

import argparse
import sys
import json
from typing import List, Dict
import boto3


def create_sns_topic_for_budget(budget_name: str, emails: List[str]) -> str:
    """
    Create SNS topic for budget alerts.

    Args:
        budget_name: Name of the budget
        emails: List of email addresses to notify

    Returns:
        SNS topic ARN
    """
    sns = boto3.client('sns')

    topic_name = f"budget-alerts-{budget_name.lower().replace(' ', '-')}"

    try:
        # Create SNS topic
        response = sns.create_topic(Name=topic_name)
        topic_arn = response['TopicArn']

        print(f"Created SNS topic: {topic_arn}")

        # Subscribe emails
        for email in emails:
            sns.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )
            print(f"Subscribed {email} to budget alerts (check email for confirmation)")

        return topic_arn

    except Exception as e:
        print(f"Error creating SNS topic: {e}")
        raise


def create_budget(
    budget_name: str,
    amount: float,
    emails: List[str],
    thresholds: List[int] = [80, 90, 100],
    time_unit: str = 'MONTHLY',
    cost_type: str = 'ACTUAL',
    filters: Dict = None
) -> Dict:
    """
    Create AWS Budget with alerts.

    Args:
        budget_name: Name of the budget
        amount: Budget amount in USD
        emails: List of email addresses for alerts
        thresholds: Alert thresholds as percentages (e.g., [80, 90, 100])
        time_unit: MONTHLY, QUARTERLY, or ANNUALLY
        cost_type: ACTUAL or FORECASTED
        filters: Optional cost filters (tags, services, etc.)

    Returns:
        Dict with budget details
    """
    budgets = boto3.client('budgets')

    # Get account ID
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']

    # Create SNS topic for notifications
    if emails:
        topic_arn = create_sns_topic_for_budget(budget_name, emails)
        subscribers = [
            {
                'SubscriptionType': 'SNS',
                'Address': topic_arn
            }
        ]
    else:
        subscribers = []

    # Also add email subscribers directly
    for email in emails:
        subscribers.append({
            'SubscriptionType': 'EMAIL',
            'Address': email
        })

    # Build budget configuration
    budget = {
        'BudgetName': budget_name,
        'BudgetLimit': {
            'Amount': str(amount),
            'Unit': 'USD'
        },
        'TimeUnit': time_unit,
        'BudgetType': 'COST',
        'CostTypes': {
            'IncludeTax': True,
            'IncludeSubscription': True,
            'UseBlended': False,
            'IncludeRefund': False,
            'IncludeCredit': False,
            'IncludeUpfront': True,
            'IncludeRecurring': True,
            'IncludeOtherSubscription': True,
            'IncludeSupport': True,
            'IncludeDiscount': True,
            'UseAmortized': False
        }
    }

    # Add cost filters if provided
    if filters:
        budget['CostFilters'] = filters

    # Create notifications for each threshold
    notifications_with_subscribers = []

    for threshold in sorted(thresholds):
        notification = {
            'Notification': {
                'NotificationType': cost_type,
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': float(threshold),
                'ThresholdType': 'PERCENTAGE',
                'NotificationState': 'ALARM'
            },
            'Subscribers': subscribers
        }
        notifications_with_subscribers.append(notification)

    try:
        # Create budget
        response = budgets.create_budget(
            AccountId=account_id,
            Budget=budget,
            NotificationsWithSubscribers=notifications_with_subscribers
        )

        print(f"✓ Created budget: {budget_name}")
        print(f"  Amount: ${amount:,.2f} {time_unit.lower()}")
        print(f"  Alert thresholds: {thresholds}%")
        print(f"  Notification type: {cost_type}")

        return {
            'budget_name': budget_name,
            'amount': amount,
            'time_unit': time_unit,
            'thresholds': thresholds,
            'subscribers': len(subscribers)
        }

    except budgets.exceptions.DuplicateRecordException:
        print(f"✗ Budget '{budget_name}' already exists")
        print("  Use --delete to remove it first, or choose a different name")
        return None

    except Exception as e:
        print(f"✗ Failed to create budget: {e}")
        return None


def list_budgets() -> List[Dict]:
    """List all AWS Budgets."""
    budgets_client = boto3.client('budgets')
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']

    try:
        response = budgets_client.describe_budgets(AccountId=account_id)

        budgets_list = []
        for budget in response['Budgets']:
            budgets_list.append({
                'name': budget['BudgetName'],
                'amount': float(budget['BudgetLimit']['Amount']),
                'unit': budget['BudgetLimit']['Unit'],
                'time_unit': budget['TimeUnit'],
                'budget_type': budget['BudgetType']
            })

        return budgets_list

    except Exception as e:
        print(f"Error listing budgets: {e}")
        return []


def get_budget_details(budget_name: str) -> Dict:
    """Get details of a specific budget including alerts."""
    budgets_client = boto3.client('budgets')
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']

    try:
        # Get budget
        budget_response = budgets_client.describe_budget(
            AccountId=account_id,
            BudgetName=budget_name
        )

        budget = budget_response['Budget']

        # Get notifications
        notifications_response = budgets_client.describe_notifications_for_budget(
            AccountId=account_id,
            BudgetName=budget_name
        )

        notifications = notifications_response.get('Notifications', [])

        return {
            'budget': budget,
            'notifications': notifications
        }

    except budgets_client.exceptions.NotFoundException:
        print(f"Budget '{budget_name}' not found")
        return None
    except Exception as e:
        print(f"Error getting budget details: {e}")
        return None


def delete_budget(budget_name: str) -> bool:
    """Delete an AWS Budget."""
    budgets_client = boto3.client('budgets')
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']

    try:
        budgets_client.delete_budget(
            AccountId=account_id,
            BudgetName=budget_name
        )

        print(f"✓ Deleted budget: {budget_name}")
        return True

    except budgets_client.exceptions.NotFoundException:
        print(f"✗ Budget '{budget_name}' not found")
        return False
    except Exception as e:
        print(f"✗ Failed to delete budget: {e}")
        return False


def create_team_budgets(teams: Dict[str, float], thresholds: List[int] = [80, 90, 100]):
    """
    Create budgets for multiple teams.

    Args:
        teams: Dict of team_name -> monthly_budget
        thresholds: Alert thresholds

    Example:
        teams = {
            'ml-platform': 25000,
            'data-science': 20000,
            'ml-research': 10000,
            'ml-ops': 5000
        }
    """
    print(f"Creating budgets for {len(teams)} teams...")
    print()

    for team_name, budget_amount in teams.items():
        budget_name = f"{team_name}-monthly-budget"

        # Create budget with team tag filter
        filters = {
            'TagKeys': ['team'],
            'Tags': {
                'team': [team_name]
            }
        }

        result = create_budget(
            budget_name=budget_name,
            amount=budget_amount,
            emails=[f"{team_name}@company.com"],
            thresholds=thresholds,
            filters=filters
        )

        if result:
            print(f"  ✓ {team_name}: ${budget_amount:,.2f}/month")
        else:
            print(f"  ✗ {team_name}: Failed")

        print()


def print_budget_summary(budgets_list: List[Dict]):
    """Print summary of all budgets."""
    print()
    print("=" * 80)
    print("  AWS Budgets Summary")
    print("=" * 80)
    print()

    if not budgets_list:
        print("No budgets found")
        print()
        return

    total_budget = sum(b['amount'] for b in budgets_list)

    print(f"Total budgets: {len(budgets_list)}")
    print(f"Total budget amount: ${total_budget:,.2f}")
    print()

    # Sort by amount
    budgets_sorted = sorted(budgets_list, key=lambda x: x['amount'], reverse=True)

    print(f"{'Budget Name':<40} {'Amount':<20} {'Period':<15}")
    print("-" * 80)

    for budget in budgets_sorted:
        print(f"{budget['name']:<40} ${budget['amount']:>15,.2f}  {budget['time_unit']:<15}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create and manage AWS Budgets with alerts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a monthly budget with alerts at 80%, 90%, 100%
  %(prog)s --create --name "ML-Platform-Monthly" --amount 50000 \\
    --email ml-platform@company.com --thresholds 80,90,100

  # Create budget with forecasted cost alerts
  %(prog)s --create --name "Overall-Monthly" --amount 100000 \\
    --email finops@company.com --cost-type FORECASTED

  # Create team budgets
  %(prog)s --create-team-budgets --teams-json teams.json

  # List all budgets
  %(prog)s --list

  # Get budget details
  %(prog)s --details --name "ML-Platform-Monthly"

  # Delete budget
  %(prog)s --delete --name "ML-Platform-Monthly"

Teams JSON format:
{
  "ml-platform": 25000,
  "data-science": 20000,
  "ml-research": 10000,
  "ml-ops": 5000
}
        """
    )

    parser.add_argument('--create', action='store_true', help='Create a new budget')
    parser.add_argument('--create-team-budgets', action='store_true', help='Create budgets for teams')
    parser.add_argument('--list', action='store_true', help='List all budgets')
    parser.add_argument('--details', action='store_true', help='Get budget details')
    parser.add_argument('--delete', action='store_true', help='Delete a budget')

    parser.add_argument('--name', help='Budget name')
    parser.add_argument('--amount', type=float, help='Budget amount in USD')
    parser.add_argument(
        '--email',
        action='append',
        help='Email address for alerts (can specify multiple times)'
    )
    parser.add_argument(
        '--thresholds',
        default='80,90,100',
        help='Alert thresholds as comma-separated percentages (default: 80,90,100)'
    )
    parser.add_argument(
        '--time-unit',
        choices=['MONTHLY', 'QUARTERLY', 'ANNUALLY'],
        default='MONTHLY',
        help='Budget time period (default: MONTHLY)'
    )
    parser.add_argument(
        '--cost-type',
        choices=['ACTUAL', 'FORECASTED'],
        default='ACTUAL',
        help='Cost type for alerts (default: ACTUAL)'
    )
    parser.add_argument(
        '--teams-json',
        help='JSON file with team budgets (for --create-team-budgets)'
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("  AWS Budget Management")
    print("=" * 80)
    print()

    try:
        # List budgets
        if args.list:
            budgets_list = list_budgets()
            print_budget_summary(budgets_list)

        # Get budget details
        if args.details:
            if not args.name:
                print("Error: --name required for --details")
                return 1

            details = get_budget_details(args.name)

            if details:
                budget = details['budget']
                notifications = details['notifications']

                print(f"Budget: {budget['BudgetName']}")
                print(f"Amount: ${float(budget['BudgetLimit']['Amount']):,.2f} {budget['BudgetLimit']['Unit']}")
                print(f"Period: {budget['TimeUnit']}")
                print()

                if notifications:
                    print(f"Notifications ({len(notifications)}):")
                    for notif in notifications:
                        print(f"  - {notif['NotificationType']} at {notif['Threshold']}% ({notif['ComparisonOperator']})")
                else:
                    print("No notifications configured")

                print()

        # Create budget
        if args.create:
            if not args.name or not args.amount:
                print("Error: --name and --amount required for --create")
                return 1

            emails = args.email or []
            thresholds = [int(t.strip()) for t in args.thresholds.split(',')]

            result = create_budget(
                budget_name=args.name,
                amount=args.amount,
                emails=emails,
                thresholds=thresholds,
                time_unit=args.time_unit,
                cost_type=args.cost_type
            )

            if not result:
                return 1

        # Create team budgets
        if args.create_team_budgets:
            if not args.teams_json:
                print("Error: --teams-json required for --create-team-budgets")
                return 1

            with open(args.teams_json) as f:
                teams = json.load(f)

            thresholds = [int(t.strip()) for t in args.thresholds.split(',')]
            create_team_budgets(teams, thresholds)

        # Delete budget
        if args.delete:
            if not args.name:
                print("Error: --name required for --delete")
                return 1

            success = delete_budget(args.name)
            if not success:
                return 1

        if not (args.list or args.details or args.create or args.create_team_budgets or args.delete):
            parser.print_help()
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Note: This script requires AWS credentials with Budgets and SNS access")
        print("Run 'aws configure' to set up credentials")
        return 1


if __name__ == '__main__':
    sys.exit(main())
