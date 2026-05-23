# GCP Module Variables

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "ml-platform"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod"
  }
}

variable "gke_subnet_cidr" {
  description = "CIDR block for GKE subnet"
  type        = string
  default     = "10.1.0.0/20"
}

variable "gke_pods_cidr" {
  description = "CIDR block for GKE pods"
  type        = string
  default     = "10.2.0.0/16"
}

variable "gke_services_cidr" {
  description = "CIDR block for GKE services"
  type        = string
  default     = "10.3.0.0/20"
}

variable "gke_master_cidr" {
  description = "CIDR block for GKE master"
  type        = string
  default     = "172.16.0.0/28"
}

variable "min_nodes" {
  description = "Minimum number of nodes in general node pool"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of nodes in general node pool"
  type        = number
  default     = 10
}

variable "db_tier" {
  description = "Cloud SQL tier"
  type        = string
  default     = "db-custom-2-7680"
}

variable "db_disk_size" {
  description = "Cloud SQL disk size in GB"
  type        = number
  default     = 100
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "mlplatform"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "mlplatform"
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_memory_size_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 5
}

variable "enable_vpn" {
  description = "Enable VPN connectivity to other clouds"
  type        = bool
  default     = true
}
