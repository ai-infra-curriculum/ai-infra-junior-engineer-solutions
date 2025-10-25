"""
Unit tests for cloud abstraction layer
Tests cloud provider implementations and factory
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
sys.path.append('../../src/cloud-abstraction')

from cloud_provider import (
    CloudProvider,
    CloudFactory,
    AWSStorageClient,
    GCPStorageClient,
    AzureStorageClient,
    AWSModelClient,
    GCPModelClient,
    AzureModelClient,
    CloudMetrics
)


class TestCloudFactory:
    """Test cloud factory for creating clients"""

    def test_create_aws_storage_client(self):
        """Test 1: Create AWS storage client"""
        factory = CloudFactory()
        client = factory.create_storage_client(CloudProvider.AWS)
        assert isinstance(client, AWSStorageClient)
        assert client.bucket_name == "ml-platform-models-prod"

    def test_create_gcp_storage_client(self):
        """Test 2: Create GCP storage client"""
        factory = CloudFactory()
        client = factory.create_storage_client(CloudProvider.GCP)
        assert isinstance(client, GCPStorageClient)
        assert client.bucket_name == "ml-platform-models-prod"

    def test_create_azure_storage_client(self):
        """Test 3: Create Azure storage client"""
        factory = CloudFactory()
        client = factory.create_storage_client(CloudProvider.AZURE)
        assert isinstance(client, AzureStorageClient)
        assert client.container_name == "models"

    def test_create_aws_model_client(self):
        """Test 4: Create AWS model client"""
        factory = CloudFactory()
        client = factory.create_model_client(CloudProvider.AWS)
        assert isinstance(client, AWSModelClient)
        assert client.region == "us-east-1"

    def test_create_gcp_model_client(self):
        """Test 5: Create GCP model client"""
        factory = CloudFactory()
        client = factory.create_model_client(CloudProvider.GCP)
        assert isinstance(client, GCPModelClient)
        assert client.project_id == "ml-platform-prod"

    def test_create_azure_model_client(self):
        """Test 6: Create Azure model client"""
        factory = CloudFactory()
        client = factory.create_model_client(CloudProvider.AZURE)
        assert isinstance(client, AzureModelClient)
        assert client.workspace_name == "ml-platform-workspace"

    def test_invalid_cloud_provider_storage(self):
        """Test 7: Invalid cloud provider for storage"""
        factory = CloudFactory()
        with pytest.raises(ValueError):
            factory.create_storage_client("invalid")

    def test_invalid_cloud_provider_model(self):
        """Test 8: Invalid cloud provider for model"""
        factory = CloudFactory()
        with pytest.raises(ValueError):
            factory.create_model_client("invalid")


class TestAWSStorageClient:
    """Test AWS S3 storage client"""

    @pytest.mark.asyncio
    async def test_upload_file(self):
        """Test 9: Upload file to S3"""
        client = AWSStorageClient(bucket_name="test-bucket", region="us-east-1")
        result = await client.upload_file("/local/path", "remote/path")
        assert result == "s3://test-bucket/remote/path"

    @pytest.mark.asyncio
    async def test_file_exists(self):
        """Test 10: Check if file exists in S3"""
        client = AWSStorageClient(bucket_name="test-bucket", region="us-east-1")
        exists = await client.file_exists("test/path")
        assert exists is True

    @pytest.mark.asyncio
    async def test_list_files(self):
        """Test 11: List files in S3"""
        client = AWSStorageClient(bucket_name="test-bucket", region="us-east-1")
        files = await client.list_files("test/prefix")
        assert isinstance(files, list)


class TestGCPStorageClient:
    """Test GCP Cloud Storage client"""

    @pytest.mark.asyncio
    async def test_upload_file(self):
        """Test 12: Upload file to GCS"""
        client = GCPStorageClient(bucket_name="test-bucket", project_id="test-project")
        result = await client.upload_file("/local/path", "remote/path")
        assert result == "gs://test-bucket/remote/path"

    @pytest.mark.asyncio
    async def test_file_exists(self):
        """Test 13: Check if file exists in GCS"""
        client = GCPStorageClient(bucket_name="test-bucket", project_id="test-project")
        exists = await client.file_exists("test/path")
        assert exists is True

    @pytest.mark.asyncio
    async def test_list_files(self):
        """Test 14: List files in GCS"""
        client = GCPStorageClient(bucket_name="test-bucket", project_id="test-project")
        files = await client.list_files("test/prefix")
        assert isinstance(files, list)


class TestAzureStorageClient:
    """Test Azure Blob Storage client"""

    @pytest.mark.asyncio
    async def test_upload_file(self):
        """Test 15: Upload file to Azure Blob"""
        client = AzureStorageClient(account_name="test-account", container_name="test-container")
        result = await client.upload_file("/local/path", "remote/path")
        assert "test-account.blob.core.windows.net" in result

    @pytest.mark.asyncio
    async def test_file_exists(self):
        """Test 16: Check if file exists in Azure Blob"""
        client = AzureStorageClient(account_name="test-account", container_name="test-container")
        exists = await client.file_exists("test/path")
        assert exists is True

    @pytest.mark.asyncio
    async def test_list_files(self):
        """Test 17: List files in Azure Blob"""
        client = AzureStorageClient(account_name="test-account", container_name="test-container")
        files = await client.list_files("test/prefix")
        assert isinstance(files, list)


class TestAWSModelClient:
    """Test AWS SageMaker model client"""

    @pytest.mark.asyncio
    async def test_predict(self):
        """Test 18: Make prediction with AWS model"""
        client = AWSModelClient(region="us-east-1")
        result = await client.predict(
            model_name="test-model",
            model_version="v1.0.0",
            input_data={"feature1": 1.0}
        )
        assert "prediction" in result
        assert result["provider"] == "aws"

    @pytest.mark.asyncio
    async def test_deploy_model(self):
        """Test 19: Deploy model to AWS"""
        client = AWSModelClient(region="us-east-1")
        result = await client.deploy_model(
            model_name="test-model",
            model_version="v1.0.0"
        )
        assert result["status"] == "deployed"
        assert "endpoint" in result

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test 20: List AWS models"""
        client = AWSModelClient(region="us-east-1")
        models = await client.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test 21: AWS model health check"""
        client = AWSModelClient(region="us-east-1")
        is_healthy = await client.health_check()
        assert is_healthy is True


class TestGCPModelClient:
    """Test GCP AI Platform model client"""

    @pytest.mark.asyncio
    async def test_predict(self):
        """Test 22: Make prediction with GCP model"""
        client = GCPModelClient(project_id="test-project", region="us-central1")
        result = await client.predict(
            model_name="test-model",
            model_version="v1.0.0",
            input_data={"feature1": 1.0}
        )
        assert "prediction" in result
        assert result["provider"] == "gcp"

    @pytest.mark.asyncio
    async def test_deploy_model(self):
        """Test 23: Deploy model to GCP"""
        client = GCPModelClient(project_id="test-project", region="us-central1")
        result = await client.deploy_model(
            model_name="test-model",
            model_version="v1.0.0"
        )
        assert result["status"] == "deployed"
        assert "endpoint" in result

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test 24: List GCP models"""
        client = GCPModelClient(project_id="test-project", region="us-central1")
        models = await client.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test 25: GCP model health check"""
        client = GCPModelClient(project_id="test-project", region="us-central1")
        is_healthy = await client.health_check()
        assert is_healthy is True


class TestAzureModelClient:
    """Test Azure ML model client"""

    @pytest.mark.asyncio
    async def test_predict(self):
        """Test 26: Make prediction with Azure model"""
        client = AzureModelClient(workspace_name="test-ws", resource_group="test-rg")
        result = await client.predict(
            model_name="test-model",
            model_version="v1.0.0",
            input_data={"feature1": 1.0}
        )
        assert "prediction" in result
        assert result["provider"] == "azure"

    @pytest.mark.asyncio
    async def test_deploy_model(self):
        """Test 27: Deploy model to Azure"""
        client = AzureModelClient(workspace_name="test-ws", resource_group="test-rg")
        result = await client.deploy_model(
            model_name="test-model",
            model_version="v1.0.0"
        )
        assert result["status"] == "deployed"
        assert "endpoint" in result

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test 28: List Azure models"""
        client = AzureModelClient(workspace_name="test-ws", resource_group="test-rg")
        models = await client.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test 29: Azure model health check"""
        client = AzureModelClient(workspace_name="test-ws", resource_group="test-rg")
        is_healthy = await client.health_check()
        assert is_healthy is True


class TestCloudMetrics:
    """Test cloud metrics data class"""

    def test_create_cloud_metrics(self):
        """Test 30: Create cloud metrics object"""
        metrics = CloudMetrics(
            cloud_provider="aws",
            latency_p50=45.0,
            latency_p95=87.0,
            latency_p99=120.0,
            error_rate=0.001,
            requests_per_second=150.0,
            cost_per_1000_requests=0.05
        )
        assert metrics.cloud_provider == "aws"
        assert metrics.latency_p50 == 45.0
        assert metrics.error_rate == 0.001

    def test_metrics_attributes(self):
        """Test 31: Verify all metrics attributes"""
        metrics = CloudMetrics(
            cloud_provider="gcp",
            latency_p50=42.0,
            latency_p95=85.0,
            latency_p99=115.0,
            error_rate=0.0005,
            requests_per_second=200.0,
            cost_per_1000_requests=0.04
        )
        assert hasattr(metrics, 'cloud_provider')
        assert hasattr(metrics, 'latency_p50')
        assert hasattr(metrics, 'latency_p95')
        assert hasattr(metrics, 'latency_p99')
        assert hasattr(metrics, 'error_rate')
        assert hasattr(metrics, 'requests_per_second')
        assert hasattr(metrics, 'cost_per_1000_requests')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
