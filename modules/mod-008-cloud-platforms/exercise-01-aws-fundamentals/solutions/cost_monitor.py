#!/usr/bin/env python3
"""
AWS Cost Monitor

Track and monitor AWS costs for ML infrastructure with alerts and reporting.

Usage:
    python cost_monitor.py daily --days 30
    python cost_monitor.py monthly --months 6
    python cost_monitor.py forecast --days 30
    python cost_monitor.py by-service --start 2024-01-01 --end 2024-01-31
    python cost_monitor.py alert --threshold 100
"""

import boto3
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


class CostMonitor:
    """Monitor AWS costs for ML infrastructure"""

    def __init__(self, region: str = 'us-east-1'):
        """Initialize cost monitor"""
        self.ce = boto3.client('ce', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sns = boto3.client('sns', region_name=region)

    def get_daily_costs(self, days: int = 30) -> List[Dict]:
        """Get daily costs for the last N days"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        print(f"{Colors.BLUE}Getting daily costs from {start_date} to {end_date}...{Colors.END}\n")

        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': str(start_date),
                'End': str(end_date)
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost', 'UsageQuantity']
        )

        costs = []
        total_cost = 0.0

        print(f"{'Date':<15} {'Cost':<15}")
        print("=" * 30)

        for result in response['ResultsByTime']:
            date = result['TimePeriod']['Start']
            cost = float(result['Total']['UnblendedCost']['Amount'])
            costs.append({'date': date, 'cost': cost})
            total_cost += cost

            # Color code based on cost
            if cost > 100:
                cost_str = f"{Colors.RED}${cost:.2f}{Colors.END}"
            elif cost > 50:
                cost_str = f"{Colors.YELLOW}${cost:.2f}{Colors.END}"
            else:
                cost_str = f"{Colors.GREEN}${cost:.2f}{Colors.END}"

            print(f"{date:<15} {cost_str}")

        print("=" * 30)
        print(f"{'Total':<15} {Colors.BOLD}${total_cost:.2f}{Colors.END}")
        print(f"{'Average/day':<15} {Colors.BOLD}${total_cost/days:.2f}{Colors.END}\n")

        return costs

    def get_monthly_costs(self, months: int = 6) -> List[Dict]:
        """Get monthly costs for the last N months"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months * 30)

        print(f"{Colors.BLUE}Getting monthly costs from {start_date} to {end_date}...{Colors.END}\n")

        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': str(start_date),
                'End': str(end_date)
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost']
        )

        costs = []
        total_cost = 0.0

        print(f"{'Month':<15} {'Cost':<15}")
        print("=" * 30)

        for result in response['ResultsByTime']:
            start = result['TimePeriod']['Start']
            cost = float(result['Total']['UnblendedCost']['Amount'])
            costs.append({'month': start[:7], 'cost': cost})
            total_cost += cost

            # Color code based on cost
            if cost > 1000:
                cost_str = f"{Colors.RED}${cost:.2f}{Colors.END}"
            elif cost > 500:
                cost_str = f"{Colors.YELLOW}${cost:.2f}{Colors.END}"
            else:
                cost_str = f"{Colors.GREEN}${cost:.2f}{Colors.END}"

            print(f"{start[:7]:<15} {cost_str}")

        print("=" * 30)
        print(f"{'Total':<15} {Colors.BOLD}${total_cost:.2f}{Colors.END}")
        print(f"{'Average/month':<15} {Colors.BOLD}${total_cost/len(costs):.2f}{Colors.END}\n")

        return costs

    def get_costs_by_service(self, start_date: str, end_date: str) -> Dict[str, float]:
        """Get costs breakdown by service"""
        print(f"{Colors.BLUE}Getting costs by service from {start_date} to {end_date}...{Colors.END}\n")

        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'}
            ]
        )

        service_costs = {}
        total_cost = 0.0

        # Aggregate costs by service
        for result in response['ResultsByTime']:
            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])

                if service in service_costs:
                    service_costs[service] += cost
                else:
                    service_costs[service] = cost

                total_cost += cost

        # Sort by cost (descending)
        sorted_services = sorted(service_costs.items(), key=lambda x: x[1], reverse=True)

        print(f"{'Service':<40} {'Cost':<15} {'Percentage':<10}")
        print("=" * 70)

        for service, cost in sorted_services:
            percentage = (cost / total_cost) * 100 if total_cost > 0 else 0

            # Color code based on percentage
            if percentage > 30:
                cost_str = f"{Colors.RED}${cost:.2f}{Colors.END}"
                pct_str = f"{Colors.RED}{percentage:.1f}%{Colors.END}"
            elif percentage > 10:
                cost_str = f"{Colors.YELLOW}${cost:.2f}{Colors.END}"
                pct_str = f"{Colors.YELLOW}{percentage:.1f}%{Colors.END}"
            else:
                cost_str = f"{Colors.GREEN}${cost:.2f}{Colors.END}"
                pct_str = f"{Colors.GREEN}{percentage:.1f}%{Colors.END}"

            print(f"{service:<40} {cost_str:<25} {pct_str}")

        print("=" * 70)
        print(f"{'Total':<40} {Colors.BOLD}${total_cost:.2f}{Colors.END}\n")

        return service_costs

    def get_costs_by_tag(self, start_date: str, end_date: str, tag_key: str) -> Dict[str, float]:
        """Get costs breakdown by tag"""
        print(f"{Colors.BLUE}Getting costs by tag '{tag_key}' from {start_date} to {end_date}...{Colors.END}\n")

        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'TAG', 'Key': tag_key}
            ]
        )

        tag_costs = {}
        total_cost = 0.0

        for result in response['ResultsByTime']:
            for group in result['Groups']:
                tag_value = group['Keys'][0].split('$')[1] if '$' in group['Keys'][0] else group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])

                if tag_value in tag_costs:
                    tag_costs[tag_value] += cost
                else:
                    tag_costs[tag_value] = cost

                total_cost += cost

        # Sort by cost (descending)
        sorted_tags = sorted(tag_costs.items(), key=lambda x: x[1], reverse=True)

        print(f"{f'{tag_key}':<40} {'Cost':<15} {'Percentage':<10}")
        print("=" * 70)

        for tag_value, cost in sorted_tags:
            percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
            print(f"{tag_value:<40} ${cost:<14.2f} {percentage:.1f}%")

        print("=" * 70)
        print(f"{'Total':<40} {Colors.BOLD}${total_cost:.2f}{Colors.END}\n")

        return tag_costs

    def forecast_costs(self, days: int = 30) -> Dict:
        """Forecast costs for next N days"""
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=days)

        print(f"{Colors.BLUE}Forecasting costs for next {days} days...{Colors.END}\n")

        response = self.ce.get_cost_forecast(
            TimePeriod={
                'Start': str(start_date),
                'End': str(end_date)
            },
            Metric='UNBLENDED_COST',
            Granularity='MONTHLY'
        )

        forecast_cost = float(response['Total']['Amount'])

        # Get current month-to-date cost for comparison
        month_start = start_date.replace(day=1)
        mtd_response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': str(month_start),
                'End': str(start_date)
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost']
        )

        mtd_cost = float(mtd_response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount']) if mtd_response['ResultsByTime'] else 0

        print(f"{'Metric':<30} {'Value':<15}")
        print("=" * 50)
        print(f"{'Month-to-date cost':<30} ${mtd_cost:.2f}")
        print(f"{'Forecasted cost (next {days}d)':<30} ${forecast_cost:.2f}")
        print(f"{'Estimated month total':<30} ${mtd_cost + forecast_cost:.2f}\n")

        if mtd_cost + forecast_cost > 1000:
            print(f"{Colors.RED}⚠ Warning: Forecasted monthly cost exceeds $1000{Colors.END}\n")

        return {
            'mtd_cost': mtd_cost,
            'forecast': forecast_cost,
            'estimated_total': mtd_cost + forecast_cost
        }

    def create_cost_alert(self, threshold: float, email: str):
        """Create CloudWatch alarm for cost threshold"""
        print(f"{Colors.BLUE}Creating cost alert for threshold ${threshold}...{Colors.END}\n")

        # Create SNS topic
        try:
            topic_response = self.sns.create_topic(Name='cost-alerts')
            topic_arn = topic_response['TopicArn']
            print(f"{Colors.GREEN}✓ Created SNS topic: {topic_arn}{Colors.END}")

            # Subscribe email
            self.sns.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )
            print(f"{Colors.GREEN}✓ Subscribed {email} to alerts (check email for confirmation){Colors.END}")

        except self.sns.exceptions.TopicLimitExceededException:
            # Topic already exists
            topics = self.sns.list_topics()
            topic_arn = next(t['TopicArn'] for t in topics['Topics'] if 'cost-alerts' in t['TopicArn'])
            print(f"{Colors.YELLOW}SNS topic already exists: {topic_arn}{Colors.END}")

        # Create CloudWatch alarm
        self.cloudwatch.put_metric_alarm(
            AlarmName='MonthlyCloudCostAlert',
            AlarmDescription=f'Alert when monthly cost exceeds ${threshold}',
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            MetricName='EstimatedCharges',
            Namespace='AWS/Billing',
            Statistic='Maximum',
            Dimensions=[{'Name': 'Currency', 'Value': 'USD'}],
            Period=21600,  # 6 hours
            EvaluationPeriods=1,
            Threshold=threshold,
            ComparisonOperator='GreaterThanThreshold'
        )

        print(f"{Colors.GREEN}✓ Created CloudWatch alarm{Colors.END}")
        print(f"\nYou will receive email alerts when monthly cost exceeds ${threshold}\n")

    def get_savings_recommendations(self) -> List[Dict]:
        """Get cost savings recommendations"""
        print(f"{Colors.BLUE}Getting cost savings recommendations...{Colors.END}\n")

        # Use Cost Explorer recommendations API
        try:
            response = self.ce.get_rightsizing_recommendation(
                Service='AmazonEC2',
                Configuration={
                    'RecommendationTarget': 'SAME_INSTANCE_FAMILY',
                    'BenefitsConsidered': True
                }
            )

            recommendations = response.get('RightsizingRecommendations', [])

            if not recommendations:
                print(f"{Colors.GREEN}No rightsizing recommendations found{Colors.END}\n")
                return []

            print(f"{'Instance ID':<20} {'Current Type':<15} {'Recommended':<15} {'Monthly Savings':<15}")
            print("=" * 70)

            total_savings = 0.0
            for rec in recommendations:
                current_instance = rec['CurrentInstance']
                recommendation = rec.get('ModifyRecommendationDetail', {})

                instance_id = current_instance['ResourceId']
                current_type = current_instance['InstanceType']
                recommended_type = recommendation.get('TargetInstances', [{}])[0].get('InstanceType', 'N/A')
                savings = float(recommendation.get('EstimatedMonthlySavings', 0))

                total_savings += savings

                print(f"{instance_id:<20} {current_type:<15} {recommended_type:<15} ${savings:<14.2f}")

            print("=" * 70)
            print(f"{'Total potential savings':<50} {Colors.GREEN}${total_savings:.2f}{Colors.END}\n")

            return recommendations

        except Exception as e:
            print(f"{Colors.YELLOW}Could not get recommendations: {e}{Colors.END}\n")
            return []

    def generate_cost_report(self, start_date: str, end_date: str, output_file: str = 'cost_report.json'):
        """Generate comprehensive cost report"""
        print(f"{Colors.BLUE}Generating cost report from {start_date} to {end_date}...{Colors.END}\n")

        report = {
            'period': {'start': start_date, 'end': end_date},
            'generated_at': datetime.now().isoformat()
        }

        # Get costs by service
        service_costs = self.get_costs_by_service(start_date, end_date)
        report['by_service'] = service_costs

        # Get daily costs
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        days = (end - start).days

        if days <= 90:
            daily_costs = self.get_daily_costs(days)
            report['daily'] = daily_costs

        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"{Colors.GREEN}✓ Cost report saved to {output_file}{Colors.END}\n")

        return report

    def compare_costs(self, period1_start: str, period1_end: str, period2_start: str, period2_end: str):
        """Compare costs between two periods"""
        print(f"{Colors.BLUE}Comparing costs between periods...{Colors.END}")
        print(f"Period 1: {period1_start} to {period1_end}")
        print(f"Period 2: {period2_start} to {period2_end}\n")

        # Get costs for period 1
        response1 = self.ce.get_cost_and_usage(
            TimePeriod={'Start': period1_start, 'End': period1_end},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost']
        )

        # Get costs for period 2
        response2 = self.ce.get_cost_and_usage(
            TimePeriod={'Start': period2_start, 'End': period2_end},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost']
        )

        cost1 = sum(float(r['Total']['UnblendedCost']['Amount']) for r in response1['ResultsByTime'])
        cost2 = sum(float(r['Total']['UnblendedCost']['Amount']) for r in response2['ResultsByTime'])

        diff = cost2 - cost1
        pct_change = (diff / cost1 * 100) if cost1 > 0 else 0

        print(f"{'Metric':<30} {'Value':<15}")
        print("=" * 50)
        print(f"{'Period 1 cost':<30} ${cost1:.2f}")
        print(f"{'Period 2 cost':<30} ${cost2:.2f}")

        if diff > 0:
            print(f"{'Change':<30} {Colors.RED}+${diff:.2f} (+{pct_change:.1f}%){Colors.END}")
        else:
            print(f"{'Change':<30} {Colors.GREEN}${diff:.2f} ({pct_change:.1f}%){Colors.END}")


def main():
    parser = argparse.ArgumentParser(description='AWS Cost Monitor')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Daily costs
    daily_parser = subparsers.add_parser('daily', help='Get daily costs')
    daily_parser.add_argument('--days', type=int, default=30, help='Number of days')
    daily_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Monthly costs
    monthly_parser = subparsers.add_parser('monthly', help='Get monthly costs')
    monthly_parser.add_argument('--months', type=int, default=6, help='Number of months')
    monthly_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Costs by service
    service_parser = subparsers.add_parser('by-service', help='Get costs by service')
    service_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    service_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    service_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Forecast
    forecast_parser = subparsers.add_parser('forecast', help='Forecast costs')
    forecast_parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    forecast_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Create alert
    alert_parser = subparsers.add_parser('alert', help='Create cost alert')
    alert_parser.add_argument('--threshold', type=float, required=True, help='Cost threshold')
    alert_parser.add_argument('--email', required=True, help='Email for alerts')
    alert_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Recommendations
    rec_parser = subparsers.add_parser('recommendations', help='Get savings recommendations')
    rec_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Generate report
    report_parser = subparsers.add_parser('report', help='Generate cost report')
    report_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    report_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    report_parser.add_argument('--output', default='cost_report.json', help='Output file')
    report_parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    monitor = CostMonitor(region=args.region)

    if args.command == 'daily':
        monitor.get_daily_costs(args.days)

    elif args.command == 'monthly':
        monitor.get_monthly_costs(args.months)

    elif args.command == 'by-service':
        monitor.get_costs_by_service(args.start, args.end)

    elif args.command == 'forecast':
        monitor.forecast_costs(args.days)

    elif args.command == 'alert':
        monitor.create_cost_alert(args.threshold, args.email)

    elif args.command == 'recommendations':
        monitor.get_savings_recommendations()

    elif args.command == 'report':
        monitor.generate_cost_report(args.start, args.end, args.output)


if __name__ == '__main__':
    main()
