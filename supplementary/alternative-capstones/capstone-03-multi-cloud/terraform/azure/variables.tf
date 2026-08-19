# Azure Module Variables

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

variable "azure_location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

# --- Networking ---

# main.tf wraps each of these in a list ([var.x]), so they are scalars.
variable "vnet_address_space" {
  description = "Address space for the virtual network"
  type        = string
  default     = "10.1.0.0/16"
}

variable "aks_subnet_address_prefix" {
  description = "Address prefix for the AKS subnet"
  type        = string
  default     = "10.1.1.0/24"
}

variable "db_subnet_address_prefix" {
  description = "Address prefix for the database subnet"
  type        = string
  default     = "10.1.2.0/24"
}

# --- AKS ---

variable "kubernetes_version" {
  description = "Kubernetes version for the AKS cluster"
  type        = string
  default     = "1.27"
}

variable "aks_service_cidr" {
  description = "CIDR block for AKS services"
  type        = string
  default     = "10.2.0.0/24"
}

variable "aks_dns_service_ip" {
  description = "IP address within the service CIDR for the AKS DNS service"
  type        = string
  default     = "10.2.0.10"
}

variable "min_nodes" {
  description = "Minimum number of nodes in the AKS node pool"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of nodes in the AKS node pool"
  type        = number
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of nodes in the AKS node pool"
  type        = number
  default     = 3
}

# --- Database (Azure SQL) ---

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "mlplatform"
}

variable "sql_admin_username" {
  description = "Azure SQL administrator username"
  type        = string
  default     = "mlplatform"
  sensitive   = true
}

variable "sql_admin_password" {
  description = "Azure SQL administrator password"
  type        = string
  sensitive   = true
}

variable "sql_sku_name" {
  description = "SKU name for the Azure SQL database"
  type        = string
  default     = "GP_S_Gen5_2"
}

# --- Redis Cache ---

variable "redis_capacity" {
  description = "Redis cache capacity (0-6 for Basic/Standard, 1-5 for Premium)"
  type        = number
  default     = 1
}

variable "redis_family" {
  description = "Redis cache family (C for Basic/Standard, P for Premium)"
  type        = string
  default     = "C"
}

variable "redis_sku_name" {
  description = "Redis cache SKU (Basic, Standard, Premium)"
  type        = string
  default     = "Standard"
}
