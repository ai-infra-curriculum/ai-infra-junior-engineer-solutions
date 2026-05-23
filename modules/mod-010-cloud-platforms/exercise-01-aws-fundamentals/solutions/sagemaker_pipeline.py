#!/usr/bin/env python3
"""
SageMaker ML Pipeline

Complete end-to-end ML pipeline using AWS SageMaker including:
- Data preprocessing
- Model training
- Hyperparameter tuning
- Model deployment
- Batch inference

Usage:
    python sagemaker_pipeline.py train --config config.json
    python sagemaker_pipeline.py deploy --model-name my-model --version 1
    python sagemaker_pipeline.py predict --endpoint my-endpoint --data test.csv
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker import get_execution_role
import argparse
import json
from typing import Dict, Optional
import os

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


class SageMakerPipeline:
    """SageMaker ML Pipeline Manager"""

    def __init__(self, role: Optional[str] = None, region: str = 'us-east-1'):
        """Initialize SageMaker pipeline"""
        self.region = region
        self.session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
        self.bucket = self.session.default_bucket()

        # Get or use provided IAM role
        try:
            self.role = role if role else get_execution_role()
        except ValueError:
            # If not running in SageMaker, need to create or specify role
            self.role = self._get_or_create_role()

        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)

    def _get_or_create_role(self) -> str:
        """Get or create IAM role for SageMaker"""
        iam = boto3.client('iam')
        role_name = 'SageMakerExecutionRole'

        try:
            role = iam.get_role(RoleName=role_name)
            return role['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            print(f"{Colors.YELLOW}Creating SageMaker execution role...{Colors.END}")

            # Create role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }]
            }

            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description='Execution role for SageMaker'
            )

            # Attach policies
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )

            print(f"{Colors.GREEN}✓ Created role: {role['Role']['Arn']}{Colors.END}")
            return role['Role']['Arn']

    def upload_training_data(self, local_path: str, s3_prefix: str = 'training-data') -> str:
        """Upload training data to S3"""
        print(f"{Colors.BLUE}Uploading training data to S3...{Colors.END}")

        s3_uri = self.session.upload_data(
            path=local_path,
            bucket=self.bucket,
            key_prefix=s3_prefix
        )

        print(f"{Colors.GREEN}✓ Data uploaded to: {s3_uri}{Colors.END}")
        return s3_uri

    def train_model(
        self,
        entry_point: str,
        source_dir: str,
        hyperparameters: Dict,
        instance_type: str = 'ml.p3.2xlarge',
        instance_count: int = 1,
        training_data: str = None,
        job_name: Optional[str] = None
    ) -> PyTorch:
        """Train model using SageMaker"""
        print(f"{Colors.BLUE}Starting training job...{Colors.END}")
        print(f"  Instance type: {instance_type}")
        print(f"  Instance count: {instance_count}")
        print(f"  Hyperparameters: {hyperparameters}\n")

        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point=entry_point,
            source_dir=source_dir,
            role=self.role,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='2.0.0',
            py_version='py310',
            hyperparameters=hyperparameters,
            output_path=f's3://{self.bucket}/models',
            code_location=f's3://{self.bucket}/code',
            sagemaker_session=self.session,
            job_name=job_name,
            use_spot_instances=True,
            max_run=3600,  # 1 hour
            max_wait=7200,  # 2 hours (for spot)
        )

        # Start training
        if training_data:
            estimator.fit({'training': training_data}, wait=True)
        else:
            estimator.fit(wait=True)

        print(f"{Colors.GREEN}✓ Training completed{Colors.END}")
        print(f"  Model artifacts: {estimator.model_data}")
        print(f"  Training job name: {estimator.latest_training_job.name}\n")

        return estimator

    def hyperparameter_tuning(
        self,
        entry_point: str,
        source_dir: str,
        hyperparameter_ranges: Dict,
        static_hyperparameters: Dict,
        training_data: str,
        instance_type: str = 'ml.p3.2xlarge',
        max_jobs: int = 10,
        max_parallel_jobs: int = 2,
        objective_metric: str = 'validation:accuracy',
        objective_type: str = 'Maximize'
    ) -> HyperparameterTuner:
        """Run hyperparameter tuning"""
        print(f"{Colors.BLUE}Starting hyperparameter tuning...{Colors.END}")
        print(f"  Max jobs: {max_jobs}")
        print(f"  Max parallel jobs: {max_parallel_jobs}")
        print(f"  Objective: {objective_type} {objective_metric}\n")

        # Create estimator
        estimator = PyTorch(
            entry_point=entry_point,
            source_dir=source_dir,
            role=self.role,
            instance_type=instance_type,
            instance_count=1,
            framework_version='2.0.0',
            py_version='py310',
            hyperparameters=static_hyperparameters,
            output_path=f's3://{self.bucket}/tuning',
            sagemaker_session=self.session,
            use_spot_instances=True,
            max_run=3600,
            max_wait=7200
        )

        # Configure tuner
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name=objective_metric,
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[
                {'Name': 'validation:accuracy', 'Regex': 'Validation Accuracy: ([0-9\\.]+)'},
                {'Name': 'training:loss', 'Regex': 'Training Loss: ([0-9\\.]+)'}
            ],
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            objective_type=objective_type
        )

        # Start tuning
        tuner.fit({'training': training_data}, wait=True)

        # Get best training job
        best_job = tuner.best_training_job()
        print(f"{Colors.GREEN}✓ Hyperparameter tuning completed{Colors.END}")
        print(f"  Best job: {best_job}\n")

        return tuner

    def deploy_model(
        self,
        model_data: str,
        entry_point: str,
        source_dir: str,
        endpoint_name: str,
        instance_type: str = 'ml.m5.large',
        initial_instance_count: int = 1
    ) -> sagemaker.predictor.Predictor:
        """Deploy model to SageMaker endpoint"""
        print(f"{Colors.BLUE}Deploying model to endpoint '{endpoint_name}'...{Colors.END}")
        print(f"  Instance type: {instance_type}")
        print(f"  Instance count: {initial_instance_count}\n")

        # Create PyTorch model
        model = PyTorchModel(
            model_data=model_data,
            role=self.role,
            entry_point=entry_point,
            source_dir=source_dir,
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=self.session
        )

        # Deploy model
        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            wait=True
        )

        print(f"{Colors.GREEN}✓ Model deployed{Colors.END}")
        print(f"  Endpoint: {endpoint_name}\n")

        return predictor

    def update_endpoint(
        self,
        endpoint_name: str,
        new_model_data: str,
        entry_point: str,
        source_dir: str
    ):
        """Update existing endpoint with new model"""
        print(f"{Colors.BLUE}Updating endpoint '{endpoint_name}'...{Colors.END}")

        # Create new model
        model = PyTorchModel(
            model_data=new_model_data,
            role=self.role,
            entry_point=entry_point,
            source_dir=source_dir,
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=self.session
        )

        # Get current endpoint config
        endpoint_desc = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        current_config = endpoint_desc['EndpointConfigName']

        # Create new endpoint config
        new_config_name = f"{endpoint_name}-config-{int(time.time())}"

        # Deploy model with new config (this will update the endpoint)
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large',
            endpoint_name=endpoint_name,
            update_endpoint=True,
            wait=True
        )

        print(f"{Colors.GREEN}✓ Endpoint updated{Colors.END}\n")

        return predictor

    def batch_transform(
        self,
        model_data: str,
        entry_point: str,
        source_dir: str,
        input_data: str,
        output_path: str,
        instance_type: str = 'ml.m5.large',
        instance_count: int = 1
    ):
        """Run batch inference using SageMaker Batch Transform"""
        print(f"{Colors.BLUE}Starting batch transform job...{Colors.END}")

        # Create model
        model = PyTorchModel(
            model_data=model_data,
            role=self.role,
            entry_point=entry_point,
            source_dir=source_dir,
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=self.session
        )

        # Create transformer
        transformer = model.transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=output_path,
            assemble_with='Line',
            accept='application/json'
        )

        # Run transform
        transformer.transform(
            data=input_data,
            content_type='application/json',
            split_type='Line',
            wait=True
        )

        print(f"{Colors.GREEN}✓ Batch transform completed{Colors.END}")
        print(f"  Output: {output_path}\n")

    def list_training_jobs(self, max_results: int = 10):
        """List recent training jobs"""
        print(f"{Colors.BLUE}Recent training jobs:{Colors.END}\n")

        response = self.sagemaker_client.list_training_jobs(
            MaxResults=max_results,
            SortBy='CreationTime',
            SortOrder='Descending'
        )

        print(f"{'Name':<40} {'Status':<15} {'Instance':<15} {'Created':<20}")
        print("=" * 100)

        for job in response['TrainingJobSummaries']:
            name = job['TrainingJobName']
            status = job['TrainingJobStatus']
            instance = job.get('ResourceConfig', {}).get('InstanceType', 'N/A')
            created = job['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')

            # Color code status
            if status == 'Completed':
                status_colored = f"{Colors.GREEN}{status}{Colors.END}"
            elif status == 'Failed':
                status_colored = f"{Colors.RED}{status}{Colors.END}"
            elif status == 'InProgress':
                status_colored = f"{Colors.YELLOW}{status}{Colors.END}"
            else:
                status_colored = status

            print(f"{name:<40} {status_colored:<30} {instance:<15} {created:<20}")

    def list_endpoints(self):
        """List active endpoints"""
        print(f"{Colors.BLUE}Active endpoints:{Colors.END}\n")

        response = self.sagemaker_client.list_endpoints(
            SortBy='CreationTime',
            SortOrder='Descending'
        )

        print(f"{'Name':<40} {'Status':<15} {'Created':<20}")
        print("=" * 80)

        for endpoint in response['Endpoints']:
            name = endpoint['EndpointName']
            status = endpoint['EndpointStatus']
            created = endpoint['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')

            # Color code status
            if status == 'InService':
                status_colored = f"{Colors.GREEN}{status}{Colors.END}"
            elif status == 'Failed':
                status_colored = f"{Colors.RED}{status}{Colors.END}"
            else:
                status_colored = f"{Colors.YELLOW}{status}{Colors.END}"

            print(f"{name:<40} {status_colored:<30} {created:<20}")

    def delete_endpoint(self, endpoint_name: str):
        """Delete SageMaker endpoint"""
        print(f"{Colors.RED}Deleting endpoint '{endpoint_name}'...{Colors.END}")

        # Confirm deletion
        response = input(f"{Colors.YELLOW}Are you sure? (yes/no): {Colors.END}")
        if response.lower() != 'yes':
            print(f"{Colors.YELLOW}Deletion cancelled{Colors.END}")
            return

        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"{Colors.GREEN}✓ Endpoint deleted{Colors.END}")

    def get_training_metrics(self, job_name: str):
        """Get metrics from training job"""
        print(f"{Colors.BLUE}Getting metrics for job '{job_name}'...{Colors.END}\n")

        response = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)

        print(f"{'Metric':<30} {'Value':<15}")
        print("=" * 50)

        for metric in response.get('FinalMetricDataList', []):
            print(f"{metric['MetricName']:<30} {metric['Value']:<15.4f}")

        print(f"\n{Colors.CYAN}Training time:{Colors.END} {response.get('TrainingTimeInSeconds', 0)} seconds")
        print(f"{Colors.CYAN}Billable time:{Colors.END} {response.get('BillableTimeInSeconds', 0)} seconds\n")


def main():
    parser = argparse.ArgumentParser(description='SageMaker ML Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', required=True, help='Training configuration JSON')
    train_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model')
    deploy_parser.add_argument('--model-data', required=True, help='S3 URI of model data')
    deploy_parser.add_argument('--endpoint', required=True, help='Endpoint name')
    deploy_parser.add_argument('--instance-type', default='ml.m5.large', help='Instance type')
    deploy_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # List training jobs
    list_jobs_parser = subparsers.add_parser('list-jobs', help='List training jobs')
    list_jobs_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # List endpoints
    list_endpoints_parser = subparsers.add_parser('list-endpoints', help='List endpoints')
    list_endpoints_parser.add_argument('--region', default='us-east-1', help='AWS region')

    # Delete endpoint
    delete_parser = subparsers.add_parser('delete-endpoint', help='Delete endpoint')
    delete_parser.add_argument('endpoint', help='Endpoint name')
    delete_parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    pipeline = SageMakerPipeline(region=args.region)

    if args.command == 'train':
        with open(args.config, 'r') as f:
            config = json.load(f)

        pipeline.train_model(
            entry_point=config['entry_point'],
            source_dir=config['source_dir'],
            hyperparameters=config['hyperparameters'],
            instance_type=config.get('instance_type', 'ml.p3.2xlarge'),
            training_data=config.get('training_data')
        )

    elif args.command == 'deploy':
        pipeline.deploy_model(
            model_data=args.model_data,
            entry_point='inference.py',
            source_dir='src',
            endpoint_name=args.endpoint,
            instance_type=args.instance_type
        )

    elif args.command == 'list-jobs':
        pipeline.list_training_jobs()

    elif args.command == 'list-endpoints':
        pipeline.list_endpoints()

    elif args.command == 'delete-endpoint':
        pipeline.delete_endpoint(args.endpoint)


if __name__ == '__main__':
    import time
    main()
