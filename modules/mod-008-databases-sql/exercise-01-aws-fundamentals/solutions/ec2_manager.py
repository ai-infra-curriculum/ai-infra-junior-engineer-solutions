#!/usr/bin/env python3
"""
EC2 Manager for ML Workloads

Automates EC2 instance management including launching GPU instances,
spot instance management, and lifecycle operations.

Usage:
    python ec2_manager.py launch --instance-type p3.2xlarge --spot
    python ec2_manager.py list
    python ec2_manager.py stop <instance-id>
    python ec2_manager.py start <instance-id>
    python ec2_manager.py terminate <instance-id>
    python ec2_manager.py ssh <instance-id>
"""

import boto3
import time
import argparse
import json
from typing import List, Dict, Optional
from datetime import datetime

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


class EC2Manager:
    """Manage EC2 instances for ML workloads"""

    # GPU instance types and their specs
    GPU_INSTANCES = {
        'p3.2xlarge': {'gpus': 1, 'gpu_type': 'V100', 'gpu_memory': '16GB', 'vcpus': 8, 'memory': '61GB'},
        'p3.8xlarge': {'gpus': 4, 'gpu_type': 'V100', 'gpu_memory': '64GB', 'vcpus': 32, 'memory': '244GB'},
        'p3.16xlarge': {'gpus': 8, 'gpu_type': 'V100', 'gpu_memory': '128GB', 'vcpus': 64, 'memory': '488GB'},
        'p4d.24xlarge': {'gpus': 8, 'gpu_type': 'A100', 'gpu_memory': '320GB', 'vcpus': 96, 'memory': '1152GB'},
        'g5.xlarge': {'gpus': 1, 'gpu_type': 'A10G', 'gpu_memory': '24GB', 'vcpus': 4, 'memory': '16GB'},
        'g5.2xlarge': {'gpus': 1, 'gpu_type': 'A10G', 'gpu_memory': '24GB', 'vcpus': 8, 'memory': '32GB'},
        'g5.4xlarge': {'gpus': 1, 'gpu_type': 'A10G', 'gpu_memory': '24GB', 'vcpus': 16, 'memory': '64GB'},
        'g4dn.xlarge': {'gpus': 1, 'gpu_type': 'T4', 'gpu_memory': '16GB', 'vcpus': 4, 'memory': '16GB'},
    }

    def __init__(self, region: str = 'us-east-1'):
        """Initialize EC2 manager"""
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)

    def get_deep_learning_ami(self) -> str:
        """Get the latest Deep Learning AMI"""
        print(f"{Colors.BLUE}Finding latest Deep Learning AMI...{Colors.END}")

        response = self.ec2.describe_images(
            Owners=['amazon'],
            Filters=[
                {'Name': 'name', 'Values': ['Deep Learning AMI GPU PyTorch*']},
                {'Name': 'state', 'Values': ['available']},
                {'Name': 'architecture', 'Values': ['x86_64']}
            ]
        )

        if not response['Images']:
            raise ValueError("No Deep Learning AMI found")

        # Sort by creation date and get the latest
        images = sorted(response['Images'], key=lambda x: x['CreationDate'], reverse=True)
        latest_ami = images[0]

        print(f"{Colors.GREEN}✓ Found AMI: {latest_ami['ImageId']} ({latest_ami['Name']}){Colors.END}")
        return latest_ami['ImageId']

    def create_security_group(self, name: str = 'ml-sg') -> str:
        """Create security group for ML instances"""
        try:
            # Check if security group already exists
            response = self.ec2.describe_security_groups(
                Filters=[{'Name': 'group-name', 'Values': [name]}]
            )

            if response['SecurityGroups']:
                sg_id = response['SecurityGroups'][0]['GroupId']
                print(f"{Colors.YELLOW}Security group '{name}' already exists: {sg_id}{Colors.END}")
                return sg_id

        except self.ec2.exceptions.ClientError:
            pass

        # Create new security group
        print(f"{Colors.BLUE}Creating security group '{name}'...{Colors.END}")

        response = self.ec2.create_security_group(
            GroupName=name,
            Description='Security group for ML instances'
        )

        sg_id = response['GroupId']

        # Allow SSH access
        self.ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8888,
                    'ToPort': 8888,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'Jupyter access'}]
                }
            ]
        )

        print(f"{Colors.GREEN}✓ Created security group: {sg_id}{Colors.END}")
        return sg_id

    def create_key_pair(self, name: str = 'ml-key') -> str:
        """Create key pair for SSH access"""
        try:
            # Check if key pair already exists
            response = self.ec2.describe_key_pairs(KeyNames=[name])
            print(f"{Colors.YELLOW}Key pair '{name}' already exists{Colors.END}")
            return name

        except self.ec2.exceptions.ClientError:
            pass

        # Create new key pair
        print(f"{Colors.BLUE}Creating key pair '{name}'...{Colors.END}")

        response = self.ec2.create_key_pair(KeyName=name)

        # Save private key
        key_file = f"{name}.pem"
        with open(key_file, 'w') as f:
            f.write(response['KeyMaterial'])

        import os
        os.chmod(key_file, 0o400)

        print(f"{Colors.GREEN}✓ Created key pair and saved to {key_file}{Colors.END}")
        return name

    def launch_instance(
        self,
        instance_type: str,
        name: str,
        spot: bool = False,
        max_price: Optional[float] = None,
        volume_size: int = 200,
        user_data: Optional[str] = None
    ) -> str:
        """Launch EC2 instance"""

        # Get AMI
        ami_id = self.get_deep_learning_ami()

        # Get or create security group
        sg_id = self.create_security_group()

        # Get or create key pair
        key_name = self.create_key_pair()

        # Prepare launch configuration
        launch_config = {
            'ImageId': ami_id,
            'InstanceType': instance_type,
            'KeyName': key_name,
            'SecurityGroupIds': [sg_id],
            'TagSpecifications': [{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': name},
                    {'Key': 'Type', 'Value': 'ML-Training'},
                    {'Key': 'CreatedBy', 'Value': 'ec2-manager'},
                    {'Key': 'LaunchedAt', 'Value': datetime.now().isoformat()}
                ]
            }],
            'BlockDeviceMappings': [{
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': volume_size,
                    'VolumeType': 'gp3',
                    'DeleteOnTermination': True
                }
            }]
        }

        if user_data:
            launch_config['UserData'] = user_data

        # Launch instance
        if spot:
            print(f"{Colors.BLUE}Launching spot instance {instance_type}...{Colors.END}")
            return self._launch_spot_instance(launch_config, max_price)
        else:
            print(f"{Colors.BLUE}Launching on-demand instance {instance_type}...{Colors.END}")
            return self._launch_on_demand_instance(launch_config)

    def _launch_on_demand_instance(self, config: Dict) -> str:
        """Launch on-demand instance"""
        response = self.ec2.run_instances(
            MinCount=1,
            MaxCount=1,
            **config
        )

        instance_id = response['Instances'][0]['InstanceId']
        print(f"{Colors.GREEN}✓ Launched instance: {instance_id}{Colors.END}")

        # Wait for instance to be running
        print(f"{Colors.BLUE}Waiting for instance to be running...{Colors.END}")
        waiter = self.ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])

        # Get instance details
        instance = self.get_instance_details(instance_id)
        print(f"{Colors.GREEN}✓ Instance is running{Colors.END}")
        print(f"  Public IP: {instance['PublicIpAddress']}")
        print(f"  Private IP: {instance['PrivateIpAddress']}")
        print(f"\n{Colors.CYAN}To connect:{Colors.END}")
        print(f"  ssh -i ml-key.pem ubuntu@{instance['PublicIpAddress']}")

        return instance_id

    def _launch_spot_instance(self, config: Dict, max_price: Optional[float]) -> str:
        """Launch spot instance"""
        spot_config = {
            'InstanceCount': 1,
            'Type': 'one-time',
            'LaunchSpecification': config
        }

        if max_price:
            spot_config['SpotPrice'] = str(max_price)

        response = self.ec2.request_spot_instances(**spot_config)

        request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        print(f"{Colors.GREEN}✓ Created spot request: {request_id}{Colors.END}")

        # Wait for spot request to be fulfilled
        print(f"{Colors.BLUE}Waiting for spot request to be fulfilled...{Colors.END}")

        while True:
            response = self.ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[request_id]
            )

            request = response['SpotInstanceRequests'][0]
            state = request['State']

            if state == 'active':
                instance_id = request['InstanceId']
                print(f"{Colors.GREEN}✓ Spot request fulfilled: {instance_id}{Colors.END}")
                return instance_id
            elif state == 'failed' or state == 'cancelled':
                raise Exception(f"Spot request failed: {request.get('Status', {}).get('Message')}")

            time.sleep(5)

    def list_instances(self, filters: Optional[List[Dict]] = None) -> List[Dict]:
        """List EC2 instances"""
        if filters is None:
            filters = [
                {'Name': 'tag:CreatedBy', 'Values': ['ec2-manager']},
                {'Name': 'instance-state-name', 'Values': ['pending', 'running', 'stopping', 'stopped']}
            ]

        response = self.ec2.describe_instances(Filters=filters)

        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append(instance)

        return instances

    def print_instances(self, instances: List[Dict]):
        """Print instances in a formatted table"""
        if not instances:
            print(f"{Colors.YELLOW}No instances found{Colors.END}")
            return

        print(f"\n{Colors.BOLD}{'ID':<20} {'Name':<20} {'Type':<15} {'State':<12} {'Public IP':<15} {'Launched':<20}{Colors.END}")
        print("=" * 120)

        for instance in instances:
            instance_id = instance['InstanceId']
            instance_type = instance['InstanceType']
            state = instance['State']['Name']
            public_ip = instance.get('PublicIpAddress', 'N/A')
            launch_time = instance['LaunchTime'].strftime('%Y-%m-%d %H:%M:%S')

            # Get name tag
            name = 'N/A'
            for tag in instance.get('Tags', []):
                if tag['Key'] == 'Name':
                    name = tag['Value']
                    break

            # Color code state
            if state == 'running':
                state_colored = f"{Colors.GREEN}{state}{Colors.END}"
            elif state == 'stopped':
                state_colored = f"{Colors.RED}{state}{Colors.END}"
            else:
                state_colored = f"{Colors.YELLOW}{state}{Colors.END}"

            # Highlight GPU instances
            if instance_type in self.GPU_INSTANCES:
                gpu_info = self.GPU_INSTANCES[instance_type]
                instance_type_colored = f"{Colors.CYAN}{instance_type} ({gpu_info['gpu_type']}){Colors.END}"
            else:
                instance_type_colored = instance_type

            print(f"{instance_id:<20} {name:<20} {instance_type_colored:<30} {state_colored:<25} {public_ip:<15} {launch_time:<20}")

    def get_instance_details(self, instance_id: str) -> Dict:
        """Get instance details"""
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        return response['Reservations'][0]['Instances'][0]

    def stop_instance(self, instance_id: str):
        """Stop instance"""
        print(f"{Colors.BLUE}Stopping instance {instance_id}...{Colors.END}")
        self.ec2.stop_instances(InstanceIds=[instance_id])

        waiter = self.ec2.get_waiter('instance_stopped')
        waiter.wait(InstanceIds=[instance_id])

        print(f"{Colors.GREEN}✓ Instance stopped{Colors.END}")

    def start_instance(self, instance_id: str):
        """Start instance"""
        print(f"{Colors.BLUE}Starting instance {instance_id}...{Colors.END}")
        self.ec2.start_instances(InstanceIds=[instance_id])

        waiter = self.ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])

        instance = self.get_instance_details(instance_id)
        print(f"{Colors.GREEN}✓ Instance started{Colors.END}")
        print(f"  Public IP: {instance['PublicIpAddress']}")

    def terminate_instance(self, instance_id: str):
        """Terminate instance"""
        print(f"{Colors.RED}Terminating instance {instance_id}...{Colors.END}")

        # Confirm termination
        response = input(f"{Colors.YELLOW}Are you sure you want to terminate this instance? (yes/no): {Colors.END}")
        if response.lower() != 'yes':
            print(f"{Colors.YELLOW}Termination cancelled{Colors.END}")
            return

        self.ec2.terminate_instances(InstanceIds=[instance_id])

        waiter = self.ec2.get_waiter('instance_terminated')
        waiter.wait(InstanceIds=[instance_id])

        print(f"{Colors.GREEN}✓ Instance terminated{Colors.END}")

    def get_ssh_command(self, instance_id: str) -> str:
        """Get SSH command for instance"""
        instance = self.get_instance_details(instance_id)
        public_ip = instance.get('PublicIpAddress')

        if not public_ip:
            raise ValueError("Instance does not have a public IP")

        return f"ssh -i ml-key.pem ubuntu@{public_ip}"

    def show_gpu_info(self):
        """Show available GPU instance types"""
        print(f"\n{Colors.BOLD}Available GPU Instance Types:{Colors.END}\n")
        print(f"{'Type':<15} {'GPUs':<6} {'GPU Type':<10} {'GPU Mem':<10} {'vCPUs':<8} {'RAM':<10}")
        print("=" * 70)

        for instance_type, specs in self.GPU_INSTANCES.items():
            print(f"{instance_type:<15} {specs['gpus']:<6} {specs['gpu_type']:<10} {specs['gpu_memory']:<10} {specs['vcpus']:<8} {specs['memory']:<10}")

    def get_spot_prices(self, instance_types: Optional[List[str]] = None) -> Dict[str, float]:
        """Get current spot prices"""
        if instance_types is None:
            instance_types = list(self.GPU_INSTANCES.keys())

        print(f"\n{Colors.BOLD}Current Spot Prices:{Colors.END}\n")

        prices = {}
        for instance_type in instance_types:
            response = self.ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                MaxResults=1,
                ProductDescriptions=['Linux/UNIX'],
                StartTime=datetime.now()
            )

            if response['SpotPriceHistory']:
                price = float(response['SpotPriceHistory'][0]['SpotPrice'])
                prices[instance_type] = price
                print(f"{instance_type:<15} ${price:.4f}/hr")

        return prices


def main():
    parser = argparse.ArgumentParser(description='EC2 Manager for ML Workloads')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Launch command
    launch_parser = subparsers.add_parser('launch', help='Launch EC2 instance')
    launch_parser.add_argument('--instance-type', required=True, help='Instance type (e.g., p3.2xlarge)')
    launch_parser.add_argument('--name', required=True, help='Instance name')
    launch_parser.add_argument('--spot', action='store_true', help='Use spot instance')
    launch_parser.add_argument('--max-price', type=float, help='Maximum spot price')
    launch_parser.add_argument('--volume-size', type=int, default=200, help='Root volume size in GB')
    launch_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # List command
    list_parser = subparsers.add_parser('list', help='List EC2 instances')
    list_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop instance')
    stop_parser.add_argument('instance_id', help='Instance ID')
    stop_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start instance')
    start_parser.add_argument('instance_id', help='Instance ID')
    start_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Terminate command
    terminate_parser = subparsers.add_parser('terminate', help='Terminate instance')
    terminate_parser.add_argument('instance_id', help='Instance ID')
    terminate_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # SSH command
    ssh_parser = subparsers.add_parser('ssh', help='Get SSH command')
    ssh_parser.add_argument('instance_id', help='Instance ID')
    ssh_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # GPU info command
    gpu_parser = subparsers.add_parser('gpu-info', help='Show GPU instance types')
    gpu_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Spot prices command
    spot_parser = subparsers.add_parser('spot-prices', help='Show spot prices')
    spot_parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = EC2Manager(region=args.region)

    if args.command == 'launch':
        manager.launch_instance(
            instance_type=args.instance_type,
            name=args.name,
            spot=args.spot,
            max_price=args.max_price,
            volume_size=args.volume_size
        )

    elif args.command == 'list':
        instances = manager.list_instances()
        manager.print_instances(instances)

    elif args.command == 'stop':
        manager.stop_instance(args.instance_id)

    elif args.command == 'start':
        manager.start_instance(args.instance_id)

    elif args.command == 'terminate':
        manager.terminate_instance(args.instance_id)

    elif args.command == 'ssh':
        ssh_cmd = manager.get_ssh_command(args.instance_id)
        print(f"\n{Colors.CYAN}SSH Command:{Colors.END}")
        print(ssh_cmd)

    elif args.command == 'gpu-info':
        manager.show_gpu_info()

    elif args.command == 'spot-prices':
        manager.get_spot_prices()


if __name__ == '__main__':
    main()
