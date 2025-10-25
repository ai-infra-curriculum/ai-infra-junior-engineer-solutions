# GCP ML Infrastructure with Terraform
# This configuration sets up a complete ML infrastructure on GCP

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "ml-infra-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "aiplatform.googleapis.com",
    "storage-api.googleapis.com",
    "bigquery.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "serviceusage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "artifactregistry.googleapis.com"
  ])

  service            = each.value
  disable_on_destroy = false
}

# VPC Network for ML infrastructure
resource "google_compute_network" "ml_network" {
  name                    = "ml-network"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.required_apis]
}

# Subnet for ML workloads
resource "google_compute_subnetwork" "ml_subnet" {
  name          = "ml-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.ml_network.id

  secondary_ip_range {
    range_name    = "ml-pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "ml-services"
    ip_cidr_range = var.services_cidr
  }

  private_ip_google_access = true
}

# Firewall rules
resource "google_compute_firewall" "allow_internal" {
  name    = "ml-allow-internal"
  network = google_compute_network.ml_network.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [var.subnet_cidr, var.pods_cidr, var.services_cidr]
}

# Cloud Storage buckets
resource "google_storage_bucket" "ml_data" {
  name          = "${var.project_id}-ml-data"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_storage_bucket" "ml_models" {
  name          = "${var.project_id}-ml-models"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_storage_bucket" "ml_artifacts" {
  name          = "${var.project_id}-ml-artifacts"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  depends_on = [google_project_service.required_apis]
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "ml_images" {
  location      = var.region
  repository_id = "ml-images"
  description   = "Docker images for ML workloads"
  format        = "DOCKER"

  depends_on = [google_project_service.required_apis]
}

# BigQuery dataset for ML data
resource "google_bigquery_dataset" "ml_dataset" {
  dataset_id    = "ml_dataset"
  friendly_name = "ML Training and Inference Data"
  description   = "Dataset for ML training and inference"
  location      = var.region

  default_table_expiration_ms = 31536000000 # 365 days

  labels = {
    environment = var.environment
    purpose     = "ml-training"
  }

  depends_on = [google_project_service.required_apis]
}

# GKE cluster for ML workloads
resource "google_container_cluster" "ml_cluster" {
  name     = "ml-cluster"
  location = var.region

  # Use regional cluster for HA
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.ml_network.name
  subnetwork = google_compute_subnetwork.ml_subnet.name

  # IP allocation for GKE
  ip_allocation_policy {
    cluster_secondary_range_name  = "ml-pods"
    services_secondary_range_name = "ml-services"
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Enable features
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Logging and monitoring
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_compute_subnetwork.ml_subnet
  ]
}

# Node pool for general ML workloads
resource "google_container_node_pool" "ml_general_pool" {
  name       = "ml-general-pool"
  location   = var.region
  cluster    = google_container_cluster.ml_cluster.name
  node_count = var.general_node_count

  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }

  node_config {
    machine_type = "n1-standard-8"
    disk_size_gb = 100
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      workload = "ml-general"
    }

    taint {
      key    = "workload"
      value  = "ml"
      effect = "NO_SCHEDULE"
    }

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Node pool for GPU workloads
resource "google_container_node_pool" "ml_gpu_pool" {
  name       = "ml-gpu-pool"
  location   = var.region
  cluster    = google_container_cluster.ml_cluster.name
  node_count = 0 # Start with 0, scale when needed

  autoscaling {
    min_node_count = 0
    max_node_count = 4
  }

  node_config {
    machine_type = "n1-standard-8"
    disk_size_gb = 100

    # GPU accelerator
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      workload = "ml-gpu"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Vertex AI Workbench instance
resource "google_notebooks_instance" "ml_workbench" {
  name     = "ml-workbench"
  location = "${var.region}-a"

  machine_type = "n1-standard-4"

  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf-latest-cpu"
  }

  install_gpu_driver = false

  boot_disk_type    = "PD_SSD"
  boot_disk_size_gb = 100

  data_disk_type    = "PD_STANDARD"
  data_disk_size_gb = 200

  network = google_compute_network.ml_network.id
  subnet  = google_compute_subnetwork.ml_subnet.id

  metadata = {
    terraform = "true"
  }

  depends_on = [google_project_service.required_apis]
}

# Service account for ML workloads
resource "google_service_account" "ml_workload_sa" {
  account_id   = "ml-workload-sa"
  display_name = "ML Workload Service Account"
}

# IAM bindings for ML service account
resource "google_project_iam_member" "ml_sa_roles" {
  for_each = toset([
    "roles/storage.objectAdmin",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/aiplatform.user",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.ml_workload_sa.email}"
}

# Outputs
output "gke_cluster_name" {
  value = google_container_cluster.ml_cluster.name
}

output "gke_cluster_endpoint" {
  value     = google_container_cluster.ml_cluster.endpoint
  sensitive = true
}

output "ml_data_bucket" {
  value = google_storage_bucket.ml_data.name
}

output "ml_models_bucket" {
  value = google_storage_bucket.ml_models.name
}

output "artifact_registry_repo" {
  value = google_artifact_registry_repository.ml_images.name
}

output "bigquery_dataset" {
  value = google_bigquery_dataset.ml_dataset.dataset_id
}

output "ml_workload_sa_email" {
  value = google_service_account.ml_workload_sa.email
}
