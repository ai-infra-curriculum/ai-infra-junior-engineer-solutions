# GCP Module Outputs

output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.ml_platform.name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.ml_platform.endpoint
}

output "gke_cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.ml_platform.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "gke_cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.ml_platform.location
}

output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.ml_platform.name
}

output "network_self_link" {
  description = "VPC network self link"
  value       = google_compute_network.ml_platform.self_link
}

output "gke_subnet_name" {
  description = "GKE subnet name"
  value       = google_compute_subnetwork.gke_subnet.name
}

output "gcs_ml_models_bucket" {
  description = "Cloud Storage bucket name for ML models"
  value       = google_storage_bucket.ml_models.name
}

output "gcs_ml_models_bucket_url" {
  description = "Cloud Storage bucket URL for ML models"
  value       = google_storage_bucket.ml_models.url
}

output "gcs_data_lake_bucket" {
  description = "Cloud Storage bucket name for data lake"
  value       = google_storage_bucket.data_lake.name
}

output "gcs_data_lake_bucket_url" {
  description = "Cloud Storage bucket URL for data lake"
  value       = google_storage_bucket.data_lake.url
}

output "bigquery_dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.ml_platform.dataset_id
}

output "cloudsql_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.ml_platform.connection_name
}

output "cloudsql_private_ip" {
  description = "Cloud SQL private IP address"
  value       = google_sql_database_instance.ml_platform.private_ip_address
}

output "redis_host" {
  description = "Redis host"
  value       = google_redis_instance.ml_platform.host
}

output "redis_port" {
  description = "Redis port"
  value       = google_redis_instance.ml_platform.port
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository"
  value       = google_artifact_registry_repository.ml_platform.name
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.ml_platform.email
}

output "project_id" {
  description = "GCP project ID"
  value       = var.gcp_project_id
}

output "region" {
  description = "GCP region"
  value       = var.gcp_region
}
