# GCP Infrastructure Module
# Provisions GKE cluster, Cloud Storage, BigQuery, and supporting infrastructure

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region

  default_labels = {
    project     = "ml-platform"
    environment = var.environment
    managed_by  = "terraform"
    cloud       = "gcp"
  }
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "container.googleapis.com",           # GKE
    "compute.googleapis.com",             # Compute Engine
    "storage.googleapis.com",             # Cloud Storage
    "bigquery.googleapis.com",            # BigQuery
    "cloudfunctions.googleapis.com",      # Cloud Functions
    "cloudresourcemanager.googleapis.com", # Resource Manager
    "servicenetworking.googleapis.com",   # Service Networking
    "sql-component.googleapis.com",       # Cloud SQL
    "sqladmin.googleapis.com",            # Cloud SQL Admin
    "monitoring.googleapis.com",          # Cloud Monitoring
    "logging.googleapis.com",             # Cloud Logging
  ])

  service            = each.value
  disable_on_destroy = false
}

# VPC Network
resource "google_compute_network" "ml_platform" {
  name                    = "${var.project_name}-network-${var.environment}"
  auto_create_subnetworks = false
  routing_mode            = "GLOBAL"

  depends_on = [google_project_service.required_apis]
}

# Subnet for GKE cluster
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "${var.project_name}-gke-subnet-${var.environment}"
  ip_cidr_range = var.gke_subnet_cidr
  region        = var.gcp_region
  network       = google_compute_network.ml_platform.id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.gke_pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.gke_services_cidr
  }

  private_ip_google_access = true
}

# Cloud NAT for private GKE nodes
resource "google_compute_router" "ml_platform" {
  name    = "${var.project_name}-router-${var.environment}"
  region  = var.gcp_region
  network = google_compute_network.ml_platform.id
}

resource "google_compute_router_nat" "ml_platform" {
  name                               = "${var.project_name}-nat-${var.environment}"
  router                             = google_compute_router.ml_platform.name
  region                             = google_compute_router.ml_platform.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# GKE Cluster
resource "google_container_cluster" "ml_platform" {
  name     = "${var.project_name}-${var.environment}"
  location = var.gcp_region

  # Regional cluster for high availability
  network    = google_compute_network.ml_platform.name
  subnetwork = google_compute_subnetwork.gke_subnet.name

  # Start with a minimal node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  # Network configuration
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.gke_master_cidr
  }

  # Master authorized networks
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All networks"
    }
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }

  # Add-ons
  addons_config {
    http_load_balancing {
      disabled = false
    }

    horizontal_pod_autoscaling {
      disabled = false
    }

    network_policy_config {
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

  # Enable binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  # Monitoring configuration
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]

    managed_prometheus {
      enabled = true
    }
  }

  # Logging configuration
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  depends_on = [
    google_project_service.required_apis,
    google_compute_subnetwork.gke_subnet
  ]
}

# General purpose node pool
resource "google_container_node_pool" "general" {
  name     = "general"
  location = var.gcp_region
  cluster  = google_container_cluster.ml_platform.name

  initial_node_count = var.min_nodes

  autoscaling {
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }

  node_config {
    machine_type = "e2-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      role = "general"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ML workloads node pool (spot instances)
resource "google_container_node_pool" "ml_workloads" {
  name     = "ml-workloads"
  location = var.gcp_region
  cluster  = google_container_cluster.ml_platform.name

  initial_node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }

  node_config {
    machine_type = "c2-standard-8"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    spot         = true

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      role     = "ml-workloads"
      workload = "training"
    }

    taint {
      key    = "workload"
      value  = "ml"
      effect = "NO_SCHEDULE"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# GPU node pool for ML training
resource "google_container_node_pool" "gpu" {
  name     = "gpu"
  location = var.gcp_region
  cluster  = google_container_cluster.ml_platform.name

  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = 5
  }

  node_config {
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    spot         = true

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
      role = "gpu"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Cloud Storage buckets
resource "google_storage_bucket" "ml_models" {
  name          = "${var.project_name}-ml-models-${var.environment}-${var.gcp_project_id}"
  location      = var.gcp_region
  force_destroy = var.environment == "dev" ? true : false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 10
    }
    action {
      type = "Delete"
    }
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

  labels = {
    purpose     = "model-storage"
    environment = var.environment
  }
}

resource "google_storage_bucket" "data_lake" {
  name          = "${var.project_name}-data-lake-${var.environment}-${var.gcp_project_id}"
  location      = var.gcp_region
  force_destroy = var.environment == "dev" ? true : false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 180
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  labels = {
    purpose     = "data-storage"
    environment = var.environment
  }
}

# BigQuery dataset
resource "google_bigquery_dataset" "ml_platform" {
  dataset_id  = "${replace(var.project_name, "-", "_")}_${var.environment}"
  location    = var.gcp_region
  description = "ML Platform dataset for ${var.environment}"

  default_table_expiration_ms = 7776000000 # 90 days

  labels = {
    environment = var.environment
  }
}

# BigQuery table for predictions
resource "google_bigquery_table" "predictions" {
  dataset_id = google_bigquery_dataset.ml_platform.dataset_id
  table_id   = "predictions"

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["model_name", "model_version"]

  schema = jsonencode([
    {
      name        = "timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "Prediction timestamp"
    },
    {
      name        = "model_name"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Model name"
    },
    {
      name        = "model_version"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Model version"
    },
    {
      name        = "input_data"
      type        = "JSON"
      mode        = "REQUIRED"
      description = "Input data for prediction"
    },
    {
      name        = "prediction"
      type        = "JSON"
      mode        = "REQUIRED"
      description = "Model prediction output"
    },
    {
      name        = "latency_ms"
      type        = "FLOAT64"
      mode        = "REQUIRED"
      description = "Prediction latency in milliseconds"
    }
  ])

  labels = {
    purpose = "predictions"
  }
}

# Cloud SQL PostgreSQL instance
resource "google_sql_database_instance" "ml_platform" {
  name             = "${var.project_name}-db-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.gcp_region

  settings {
    tier              = var.db_tier
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"
    disk_size         = var.db_disk_size
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 7
        retention_unit   = "COUNT"
      }
    }

    maintenance_window {
      day          = 7
      hour         = 3
      update_track = "stable"
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.ml_platform.id
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }

  deletion_protection = var.environment == "prod" ? true : false

  depends_on = [google_project_service.required_apis]
}

resource "google_sql_database" "ml_platform" {
  name     = var.db_name
  instance = google_sql_database_instance.ml_platform.name
}

resource "google_sql_user" "ml_platform" {
  name     = var.db_username
  instance = google_sql_database_instance.ml_platform.name
  password = var.db_password
}

# Memorystore Redis instance
resource "google_redis_instance" "ml_platform" {
  name               = "${var.project_name}-redis-${var.environment}"
  tier               = var.environment == "prod" ? "STANDARD_HA" : "BASIC"
  memory_size_gb     = var.redis_memory_size_gb
  region             = var.gcp_region
  redis_version      = "REDIS_7_0"
  display_name       = "ML Platform Redis"
  authorized_network = google_compute_network.ml_platform.id

  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
      }
    }
  }

  labels = {
    environment = var.environment
  }

  depends_on = [google_project_service.required_apis]
}

# Artifact Registry for container images
resource "google_artifact_registry_repository" "ml_platform" {
  location      = var.gcp_region
  repository_id = "${var.project_name}-${var.environment}"
  description   = "Container registry for ML platform"
  format        = "DOCKER"

  cleanup_policies {
    id     = "keep-recent-versions"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }

  labels = {
    environment = var.environment
  }

  depends_on = [google_project_service.required_apis]
}

# Service Account for workload identity
resource "google_service_account" "ml_platform" {
  account_id   = "${var.project_name}-sa-${var.environment}"
  display_name = "ML Platform Service Account"
  description  = "Service account for ML platform workloads"
}

# IAM bindings for service account
resource "google_project_iam_member" "ml_platform_storage" {
  project = var.gcp_project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ml_platform.email}"
}

resource "google_project_iam_member" "ml_platform_bigquery" {
  project = var.gcp_project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.ml_platform.email}"
}

resource "google_project_iam_member" "ml_platform_cloudsql" {
  project = var.gcp_project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.ml_platform.email}"
}
