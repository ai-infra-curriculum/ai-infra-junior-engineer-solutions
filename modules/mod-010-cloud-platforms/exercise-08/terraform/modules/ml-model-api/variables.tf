# Cloud-Agnostic ML Model API Module - Variables

# ============================================================================
# Basic Configuration
# ============================================================================

variable "service_name" {
  description = "Name of the service (will be used for all Kubernetes resources)"
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace to deploy into"
  type        = string
  default     = "default"
}

variable "create_namespace" {
  description = "Whether to create the namespace (set to false if it already exists)"
  type        = bool
  default     = false
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "cloud_provider" {
  description = "Cloud provider (aws, gcp, azure, on-prem) - used for metadata/monitoring"
  type        = string
}

# ============================================================================
# Container Configuration
# ============================================================================

variable "image" {
  description = "Container image (e.g., gcr.io/project/model-api:v1.0.0)"
  type        = string
}

variable "image_pull_policy" {
  description = "Image pull policy"
  type        = string
  default     = "IfNotPresent"
}

variable "container_port" {
  description = "Port the container listens on"
  type        = number
  default     = 8000
}

variable "model_version" {
  description = "ML model version being served"
  type        = string
  default     = "v1.0.0"
}

variable "workers" {
  description = "Number of worker processes (for uvicorn/gunicorn)"
  type        = number
  default     = 4
}

variable "log_level" {
  description = "Application log level"
  type        = string
  default     = "info"
}

# ============================================================================
# Deployment Configuration
# ============================================================================

variable "replicas" {
  description = "Number of pod replicas"
  type        = number
  default     = 3
}

variable "max_surge" {
  description = "Maximum number of pods that can be created above desired replicas during rolling update"
  type        = string
  default     = "25%"
}

variable "max_unavailable" {
  description = "Maximum number of pods that can be unavailable during rolling update"
  type        = string
  default     = "0"
}

variable "enable_pod_anti_affinity" {
  description = "Enable pod anti-affinity to spread pods across nodes"
  type        = bool
  default     = true
}

# ============================================================================
# Resource Limits
# ============================================================================

variable "cpu_request" {
  description = "CPU request (e.g., '500m', '1000m')"
  type        = string
  default     = "500m"
}

variable "cpu_limit" {
  description = "CPU limit (e.g., '1000m', '2000m')"
  type        = string
  default     = "1000m"
}

variable "memory_request" {
  description = "Memory request (e.g., '512Mi', '1Gi')"
  type        = string
  default     = "512Mi"
}

variable "memory_limit" {
  description = "Memory limit (e.g., '1Gi', '2Gi')"
  type        = string
  default     = "1Gi"
}

# ============================================================================
# Health Checks
# ============================================================================

variable "liveness_probe_path" {
  description = "Path for liveness probe"
  type        = string
  default     = "/health"
}

variable "liveness_probe_initial_delay" {
  description = "Initial delay for liveness probe (seconds)"
  type        = number
  default     = 30
}

variable "liveness_probe_period" {
  description = "Period for liveness probe (seconds)"
  type        = number
  default     = 10
}

variable "liveness_probe_timeout" {
  description = "Timeout for liveness probe (seconds)"
  type        = number
  default     = 5
}

variable "liveness_probe_failure_threshold" {
  description = "Failure threshold for liveness probe"
  type        = number
  default     = 3
}

variable "readiness_probe_path" {
  description = "Path for readiness probe"
  type        = string
  default     = "/ready"
}

variable "readiness_probe_initial_delay" {
  description = "Initial delay for readiness probe (seconds)"
  type        = number
  default     = 10
}

variable "readiness_probe_period" {
  description = "Period for readiness probe (seconds)"
  type        = number
  default     = 5
}

variable "readiness_probe_timeout" {
  description = "Timeout for readiness probe (seconds)"
  type        = number
  default     = 3
}

variable "readiness_probe_failure_threshold" {
  description = "Failure threshold for readiness probe"
  type        = number
  default     = 3
}

# ============================================================================
# Service Configuration
# ============================================================================

variable "service_type" {
  description = "Kubernetes service type (ClusterIP, NodePort, LoadBalancer)"
  type        = string
  default     = "ClusterIP"
}

variable "service_port" {
  description = "Service port"
  type        = number
  default     = 80
}

variable "service_annotations" {
  description = "Annotations for the service"
  type        = map(string)
  default     = {}
}

variable "service_session_affinity" {
  description = "Session affinity for the service"
  type        = string
  default     = "None"
}

# ============================================================================
# Service Account (for cloud IAM integration)
# ============================================================================

variable "service_account_annotations" {
  description = "Annotations for service account (e.g., for IRSA or Workload Identity)"
  type        = map(string)
  default     = {}

  # Example for AWS IRSA:
  # {
  #   "eks.amazonaws.com/role-arn" = "arn:aws:iam::123456789:role/model-api-role"
  # }

  # Example for GCP Workload Identity:
  # {
  #   "iam.gke.io/gcp-service-account" = "model-api@project.iam.gserviceaccount.com"
  # }
}

# ============================================================================
# Environment Variables
# ============================================================================

variable "extra_env_vars" {
  description = "Additional environment variables"
  type        = map(string)
  default     = {}
}

variable "secrets" {
  description = "Secret data (will be created as Kubernetes secret)"
  type        = map(string)
  default     = {}
  sensitive   = true
}

# ============================================================================
# Volumes
# ============================================================================

variable "volumes" {
  description = "Volumes to attach to pods"
  type = list(object({
    name       = string
    type       = string # emptyDir, persistentVolumeClaim
    medium     = optional(string)
    claim_name = optional(string)
  }))
  default = []
}

variable "volume_mounts" {
  description = "Volume mounts for the container"
  type = list(object({
    name       = string
    mount_path = string
    read_only  = bool
  }))
  default = []
}

# ============================================================================
# Horizontal Pod Autoscaler (HPA)
# ============================================================================

variable "enable_hpa" {
  description = "Enable Horizontal Pod Autoscaler"
  type        = bool
  default     = false
}

variable "hpa_min_replicas" {
  description = "Minimum number of replicas for HPA"
  type        = number
  default     = 2
}

variable "hpa_max_replicas" {
  description = "Maximum number of replicas for HPA"
  type        = number
  default     = 10
}

variable "hpa_cpu_target_percentage" {
  description = "Target CPU utilization percentage for HPA"
  type        = number
  default     = 70
}

variable "hpa_memory_target_percentage" {
  description = "Target memory utilization percentage for HPA"
  type        = number
  default     = 80
}

variable "hpa_custom_metrics" {
  description = "Custom metrics for HPA (e.g., request rate)"
  type = list(object({
    name         = string
    target_value = string
  }))
  default = []

  # Example:
  # [
  #   {
  #     name         = "http_requests_per_second"
  #     target_value = "1000"
  #   }
  # ]
}

variable "hpa_scale_down_stabilization" {
  description = "Stabilization window for scale down (seconds)"
  type        = number
  default     = 300
}

variable "hpa_scale_up_stabilization" {
  description = "Stabilization window for scale up (seconds)"
  type        = number
  default     = 0
}

# ============================================================================
# Pod Disruption Budget (PDB)
# ============================================================================

variable "enable_pdb" {
  description = "Enable Pod Disruption Budget"
  type        = bool
  default     = true
}

variable "pdb_min_available" {
  description = "Minimum available pods during disruptions"
  type        = string
  default     = "50%"
}

# ============================================================================
# Ingress
# ============================================================================

variable "enable_ingress" {
  description = "Enable Ingress resource"
  type        = bool
  default     = false
}

variable "ingress_class_name" {
  description = "Ingress class name (nginx, alb, etc.)"
  type        = string
  default     = "nginx"
}

variable "ingress_host" {
  description = "Hostname for ingress"
  type        = string
  default     = ""
}

variable "ingress_path" {
  description = "Path for ingress rule"
  type        = string
  default     = "/"
}

variable "ingress_annotations" {
  description = "Annotations for ingress"
  type        = map(string)
  default     = {}

  # Example for AWS ALB:
  # {
  #   "kubernetes.io/ingress.class"           = "alb"
  #   "alb.ingress.kubernetes.io/scheme"       = "internet-facing"
  #   "alb.ingress.kubernetes.io/target-type"  = "ip"
  # }

  # Example for NGINX + cert-manager:
  # {
  #   "kubernetes.io/ingress.class"               = "nginx"
  #   "cert-manager.io/cluster-issuer"            = "letsencrypt-prod"
  #   "nginx.ingress.kubernetes.io/ssl-redirect" = "true"
  # }
}

variable "ingress_tls_enabled" {
  description = "Enable TLS for ingress"
  type        = bool
  default     = false
}

variable "ingress_tls_secret_name" {
  description = "Secret name for TLS certificate"
  type        = string
  default     = ""
}
