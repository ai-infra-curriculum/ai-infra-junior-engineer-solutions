"""Tests for GCP automation scripts."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gcp_automation import GCPMLInfraManager


@pytest.fixture
def mock_gcp_clients():
    """Mock GCP clients."""
    with patch('gcp_automation.storage.Client') as mock_storage, \
         patch('gcp_automation.bigquery.Client') as mock_bq, \
         patch('gcp_automation.aiplatform.init') as mock_ai_init:

        yield {
            'storage': mock_storage,
            'bigquery': mock_bq,
            'aiplatform': mock_ai_init
        }


class TestGCPMLInfraManager:
    """Test GCP ML infrastructure manager."""

    def test_init(self, mock_gcp_clients):
        """Test manager initialization."""
        manager = GCPMLInfraManager("test-project", "us-central1")

        assert manager.project_id == "test-project"
        assert manager.region == "us-central1"

    def test_upload_to_gcs(self, mock_gcp_clients):
        """Test GCS upload."""
        manager = GCPMLInfraManager("test-project")

        # Mock bucket and blob
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        manager.storage_client.bucket.return_value = mock_bucket

        manager.upload_to_gcs("test-bucket", "local_file.txt", "remote_file.txt")

        mock_bucket.blob.assert_called_once_with("remote_file.txt")
        mock_blob.upload_from_filename.assert_called_once_with("local_file.txt")

    def test_download_from_gcs(self, mock_gcp_clients):
        """Test GCS download."""
        manager = GCPMLInfraManager("test-project")

        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        manager.storage_client.bucket.return_value = mock_bucket

        manager.download_from_gcs("test-bucket", "remote_file.txt", "local_file.txt")

        mock_bucket.blob.assert_called_once_with("remote_file.txt")
        mock_blob.download_to_filename.assert_called_once_with("local_file.txt")

    def test_list_gcs_objects(self, mock_gcp_clients):
        """Test listing GCS objects."""
        manager = GCPMLInfraManager("test-project")

        mock_bucket = Mock()
        mock_blobs = [Mock(name=f"file{i}.txt") for i in range(3)]
        mock_bucket.list_blobs.return_value = mock_blobs
        manager.storage_client.bucket.return_value = mock_bucket

        objects = manager.list_gcs_objects("test-bucket", "prefix/")

        assert len(objects) == 3
        mock_bucket.list_blobs.assert_called_once_with(prefix="prefix/")

    @patch('gcp_automation.subprocess.run')
    def test_get_gke_credentials(self, mock_subprocess, mock_gcp_clients):
        """Test getting GKE credentials."""
        manager = GCPMLInfraManager("test-project", "us-central1")

        manager.get_gke_credentials("test-cluster")

        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "gcloud" in args
        assert "test-cluster" in args


# Additional test cases for comprehensive coverage
class TestGCPAutomationEdgeCases:
    """Test edge cases and error handling."""

    def test_manager_with_default_region(self, mock_gcp_clients):
        """Test manager with default region."""
        manager = GCPMLInfraManager("test-project")
        assert manager.region == "us-central1"

    def test_manager_with_custom_region(self, mock_gcp_clients):
        """Test manager with custom region."""
        manager = GCPMLInfraManager("test-project", "europe-west1")
        assert manager.region == "europe-west1"
