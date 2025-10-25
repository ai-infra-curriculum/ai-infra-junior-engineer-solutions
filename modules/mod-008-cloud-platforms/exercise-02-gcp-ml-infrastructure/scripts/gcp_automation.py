#!/usr/bin/env python3
"""GCP automation scripts for ML infrastructure."""

import argparse
import logging
from typing import List, Dict, Any, Optional
from google.cloud import storage, bigquery, aiplatform
from google.oauth2 import service_account
import subprocess
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCPMLInfraManager:
    """Manage GCP ML infrastructure programmatically."""

    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize GCP manager.

        Args:
            project_id: GCP project ID
            region: GCP region
        """
        self.project_id = project_id
        self.region = region
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        aiplatform.init(project=project_id, location=region)

        logger.info(f"Initialized GCP manager for project: {project_id}")

    def create_bucket(
        self,
        bucket_name: str,
        location: Optional[str] = None,
        storage_class: str = "STANDARD"
    ) -> storage.Bucket:
        """Create Cloud Storage bucket.

        Args:
            bucket_name: Name of bucket to create
            location: Bucket location (default: self.region)
            storage_class: Storage class (STANDARD, NEARLINE, COLDLINE)

        Returns:
            Created bucket object
        """
        location = location or self.region
        logger.info(f"Creating bucket: {bucket_name}")

        bucket = self.storage_client.bucket(bucket_name)
        bucket.storage_class = storage_class
        bucket = self.storage_client.create_bucket(bucket, location=location)

        logger.info(f"Bucket {bucket_name} created successfully")
        return bucket

    def upload_to_gcs(
        self,
        bucket_name: str,
        source_file: str,
        destination_blob: str
    ):
        """Upload file to Cloud Storage.

        Args:
            bucket_name: Target bucket name
            source_file: Local file path
            destination_blob: Destination blob name in GCS
        """
        logger.info(f"Uploading {source_file} to gs://{bucket_name}/{destination_blob}")

        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(source_file)

        logger.info("Upload complete")

    def download_from_gcs(
        self,
        bucket_name: str,
        source_blob: str,
        destination_file: str
    ):
        """Download file from Cloud Storage.

        Args:
            bucket_name: Source bucket name
            source_blob: Source blob name in GCS
            destination_file: Local destination file path
        """
        logger.info(f"Downloading gs://{bucket_name}/{source_blob} to {destination_file}")

        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        blob.download_to_filename(destination_file)

        logger.info("Download complete")

    def list_gcs_objects(self, bucket_name: str, prefix: str = "") -> List[str]:
        """List objects in Cloud Storage bucket.

        Args:
            bucket_name: Bucket name
            prefix: Filter by prefix

        Returns:
            List of blob names
        """
        bucket = self.storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]

    def create_bq_dataset(
        self,
        dataset_id: str,
        location: Optional[str] = None,
        description: str = ""
    ) -> bigquery.Dataset:
        """Create BigQuery dataset.

        Args:
            dataset_id: Dataset ID
            location: Dataset location
            description: Dataset description

        Returns:
            Created dataset
        """
        location = location or self.region
        logger.info(f"Creating BigQuery dataset: {dataset_id}")

        dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
        dataset.location = location
        dataset.description = description

        dataset = self.bq_client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset {dataset_id} created successfully")

        return dataset

    def load_bq_from_gcs(
        self,
        dataset_id: str,
        table_id: str,
        gcs_uri: str,
        schema: Optional[List[bigquery.SchemaField]] = None
    ):
        """Load data from GCS to BigQuery.

        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            gcs_uri: GCS URI (gs://bucket/path)
            schema: Table schema (optional, auto-detect if None)
        """
        logger.info(f"Loading {gcs_uri} to {dataset_id}.{table_id}")

        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        job_config = bigquery.LoadJobConfig()
        if schema:
            job_config.schema = schema
        else:
            job_config.autodetect = True

        job_config.source_format = bigquery.SourceFormat.CSV
        job_config.skip_leading_rows = 1
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

        load_job = self.bq_client.load_table_from_uri(
            gcs_uri,
            table_ref,
            job_config=job_config
        )

        load_job.result()  # Wait for job to complete
        logger.info("Data loaded successfully")

    def query_bq(self, query: str) -> List[Dict[str, Any]]:
        """Execute BigQuery query.

        Args:
            query: SQL query string

        Returns:
            List of result rows as dictionaries
        """
        logger.info("Executing BigQuery query")

        query_job = self.bq_client.query(query)
        results = query_job.result()

        rows = [dict(row) for row in results]
        logger.info(f"Query returned {len(rows)} rows")

        return rows

    def get_gke_credentials(self, cluster_name: str, zone: Optional[str] = None):
        """Get GKE cluster credentials.

        Args:
            cluster_name: GKE cluster name
            zone: Cluster zone (default: use region)
        """
        location = zone or self.region
        logger.info(f"Getting credentials for cluster: {cluster_name}")

        cmd = [
            "gcloud", "container", "clusters", "get-credentials",
            cluster_name,
            f"--region={location}",
            f"--project={self.project_id}"
        ]

        subprocess.run(cmd, check=True)
        logger.info("Credentials configured successfully")

    def deploy_vertex_ai_model(
        self,
        model_name: str,
        artifact_uri: str,
        serving_container_image_uri: str
    ):
        """Deploy model to Vertex AI.

        Args:
            model_name: Model name
            artifact_uri: GCS URI to model artifacts
            serving_container_image_uri: Container image for serving
        """
        logger.info(f"Deploying model to Vertex AI: {model_name}")

        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri
        )

        logger.info(f"Model deployed: {model.resource_name}")
        return model

    def create_vertex_ai_endpoint(
        self,
        endpoint_name: str,
        model: aiplatform.Model,
        machine_type: str = "n1-standard-4",
        min_replica_count: int = 1,
        max_replica_count: int = 3
    ):
        """Create Vertex AI endpoint and deploy model.

        Args:
            endpoint_name: Endpoint name
            model: Vertex AI model to deploy
            machine_type: Machine type for serving
            min_replica_count: Minimum replicas
            max_replica_count: Maximum replicas
        """
        logger.info(f"Creating Vertex AI endpoint: {endpoint_name}")

        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=endpoint_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count
        )

        logger.info(f"Endpoint created: {endpoint.resource_name}")
        return endpoint


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="GCP ML Infrastructure Automation")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload file to GCS")
    upload_parser.add_argument("--bucket", required=True, help="Bucket name")
    upload_parser.add_argument("--source", required=True, help="Source file")
    upload_parser.add_argument("--destination", required=True, help="Destination blob")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download file from GCS")
    download_parser.add_argument("--bucket", required=True, help="Bucket name")
    download_parser.add_argument("--source", required=True, help="Source blob")
    download_parser.add_argument("--destination", required=True, help="Destination file")

    # List command
    list_parser = subparsers.add_parser("list", help="List GCS objects")
    list_parser.add_argument("--bucket", required=True, help="Bucket name")
    list_parser.add_argument("--prefix", default="", help="Object prefix")

    # Query command
    query_parser = subparsers.add_parser("query", help="Execute BigQuery query")
    query_parser.add_argument("--sql", required=True, help="SQL query")

    # GKE credentials command
    gke_parser = subparsers.add_parser("gke-credentials", help="Get GKE credentials")
    gke_parser.add_argument("--cluster", required=True, help="Cluster name")

    args = parser.parse_args()

    manager = GCPMLInfraManager(args.project_id, args.region)

    if args.command == "upload":
        manager.upload_to_gcs(args.bucket, args.source, args.destination)
    elif args.command == "download":
        manager.download_from_gcs(args.bucket, args.source, args.destination)
    elif args.command == "list":
        objects = manager.list_gcs_objects(args.bucket, args.prefix)
        for obj in objects:
            print(obj)
    elif args.command == "query":
        results = manager.query_bq(args.sql)
        print(json.dumps(results, indent=2, default=str))
    elif args.command == "gke-credentials":
        manager.get_gke_credentials(args.cluster)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
