# Cloud-Agnostic ML Model API Module
#
# This module deploys an ML model API to any Kubernetes cluster,
# whether running on AWS EKS, GCP GKE, Azure AKS, or on-premise.
#
# Key principle: Use only standard Kubernetes resources, no cloud-specific CRDs

terraform {
  required_version = ">= 1.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

# ============================================================================
# Namespace
# ============================================================================

resource "kubernetes_namespace" "app" {
  count = var.create_namespace ? 1 : 0

  metadata {
    name = var.namespace

    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "environment"                  = var.environment
    }
  }
}

# ============================================================================
# ConfigMap for application configuration
# ============================================================================

resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "${var.service_name}-config"
    namespace = var.namespace

    labels = {
      app = var.service_name
    }
  }

  data = {
    "ENVIRONMENT"     = var.environment
    "CLOUD_PROVIDER"  = var.cloud_provider
    "LOG_LEVEL"       = var.log_level
    "MODEL_VERSION"   = var.model_version
    "WORKERS"         = tostring(var.workers)
  }
}

# ============================================================================
# Secret for sensitive configuration (e.g., API keys, DB passwords)
# ============================================================================

resource "kubernetes_secret" "app_secrets" {
  metadata {
    name      = "${var.service_name}-secrets"
    namespace = var.namespace

    labels = {
      app = var.service_name
    }
  }

  data = var.secrets

  type = "Opaque"
}

# ============================================================================
# Deployment
# ============================================================================

resource "kubernetes_deployment" "app" {
  metadata {
    name      = var.service_name
    namespace = var.namespace

    labels = {
      app         = var.service_name
      version     = var.model_version
      environment = var.environment
    }
  }

  spec {
    replicas = var.replicas

    strategy {
      type = "RollingUpdate"

      rolling_update {
        max_surge       = var.max_surge
        max_unavailable = var.max_unavailable
      }
    }

    selector {
      match_labels = {
        app = var.service_name
      }
    }

    template {
      metadata {
        labels = {
          app         = var.service_name
          version     = var.model_version
          environment = var.environment
        }

        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "8000"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        # Service account for IRSA (AWS) or Workload Identity (GCP)
        service_account_name            = kubernetes_service_account.app.metadata[0].name
        automount_service_account_token = true

        # Pod anti-affinity for high availability
        dynamic "affinity" {
          for_each = var.enable_pod_anti_affinity ? [1] : []

          content {
            pod_anti_affinity {
              preferred_during_scheduling_ignored_during_execution {
                weight = 100

                pod_affinity_term {
                  label_selector {
                    match_expressions {
                      key      = "app"
                      operator = "In"
                      values   = [var.service_name]
                    }
                  }

                  topology_key = "kubernetes.io/hostname"
                }
              }
            }
          }
        }

        container {
          name              = var.service_name
          image             = var.image
          image_pull_policy = var.image_pull_policy

          port {
            name           = "http"
            container_port = var.container_port
            protocol       = "TCP"
          }

          # Environment variables from ConfigMap
          env_from {
            config_map_ref {
              name = kubernetes_config_map.app_config.metadata[0].name
            }
          }

          # Sensitive environment variables from Secret
          dynamic "env" {
            for_each = var.secrets

            content {
              name = env.key

              value_from {
                secret_key_ref {
                  name = kubernetes_secret.app_secrets.metadata[0].name
                  key  = env.key
                }
              }
            }
          }

          # Additional environment variables
          dynamic "env" {
            for_each = var.extra_env_vars

            content {
              name  = env.key
              value = env.value
            }
          }

          # Resource requests and limits
          resources {
            requests = {
              cpu    = var.cpu_request
              memory = var.memory_request
            }

            limits = {
              cpu    = var.cpu_limit
              memory = var.memory_limit
            }
          }

          # Liveness probe
          liveness_probe {
            http_get {
              path   = var.liveness_probe_path
              port   = var.container_port
              scheme = "HTTP"
            }

            initial_delay_seconds = var.liveness_probe_initial_delay
            period_seconds        = var.liveness_probe_period
            timeout_seconds       = var.liveness_probe_timeout
            failure_threshold     = var.liveness_probe_failure_threshold
          }

          # Readiness probe
          readiness_probe {
            http_get {
              path   = var.readiness_probe_path
              port   = var.container_port
              scheme = "HTTP"
            }

            initial_delay_seconds = var.readiness_probe_initial_delay
            period_seconds        = var.readiness_probe_period
            timeout_seconds       = var.readiness_probe_timeout
            failure_threshold     = var.readiness_probe_failure_threshold
          }

          # Volume mounts
          dynamic "volume_mount" {
            for_each = var.volume_mounts

            content {
              name       = volume_mount.value.name
              mount_path = volume_mount.value.mount_path
              read_only  = volume_mount.value.read_only
            }
          }
        }

        # Volumes
        dynamic "volume" {
          for_each = var.volumes

          content {
            name = volume.value.name

            dynamic "empty_dir" {
              for_each = volume.value.type == "emptyDir" ? [1] : []
              content {
                medium = lookup(volume.value, "medium", "")
              }
            }

            dynamic "persistent_volume_claim" {
              for_each = volume.value.type == "persistentVolumeClaim" ? [1] : []
              content {
                claim_name = volume.value.claim_name
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_config_map.app_config,
    kubernetes_secret.app_secrets
  ]
}

# ============================================================================
# Service
# ============================================================================

resource "kubernetes_service" "app" {
  metadata {
    name      = var.service_name
    namespace = var.namespace

    labels = {
      app = var.service_name
    }

    annotations = var.service_annotations
  }

  spec {
    type = var.service_type

    selector = {
      app = var.service_name
    }

    port {
      name        = "http"
      port        = var.service_port
      target_port = var.container_port
      protocol    = "TCP"
    }

    session_affinity = var.service_session_affinity
  }
}

# ============================================================================
# Service Account (for cloud IAM integration)
# ============================================================================

resource "kubernetes_service_account" "app" {
  metadata {
    name      = var.service_name
    namespace = var.namespace

    labels = {
      app = var.service_name
    }

    # Annotations for cloud provider IAM integration
    # AWS: IRSA (IAM Roles for Service Accounts)
    # GCP: Workload Identity
    # Azure: AAD Pod Identity
    annotations = var.service_account_annotations
  }
}

# ============================================================================
# Horizontal Pod Autoscaler
# ============================================================================

resource "kubernetes_horizontal_pod_autoscaler_v2" "app" {
  count = var.enable_hpa ? 1 : 0

  metadata {
    name      = var.service_name
    namespace = var.namespace
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.app.metadata[0].name
    }

    min_replicas = var.hpa_min_replicas
    max_replicas = var.hpa_max_replicas

    # CPU-based scaling
    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = var.hpa_cpu_target_percentage
        }
      }
    }

    # Memory-based scaling
    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = var.hpa_memory_target_percentage
        }
      }
    }

    # Custom metrics (e.g., request rate)
    dynamic "metric" {
      for_each = var.hpa_custom_metrics

      content {
        type = "Pods"
        pods {
          metric {
            name = metric.value.name
          }
          target {
            type          = "AverageValue"
            average_value = metric.value.target_value
          }
        }
      }
    }

    behavior {
      scale_down {
        stabilization_window_seconds = var.hpa_scale_down_stabilization
        select_policy                = "Min"

        policy {
          type          = "Percent"
          value         = 50
          period_seconds = 60
        }
      }

      scale_up {
        stabilization_window_seconds = var.hpa_scale_up_stabilization
        select_policy                = "Max"

        policy {
          type          = "Percent"
          value         = 100
          period_seconds = 30
        }
      }
    }
  }
}

# ============================================================================
# Pod Disruption Budget (for high availability)
# ============================================================================

resource "kubernetes_pod_disruption_budget_v1" "app" {
  count = var.enable_pdb ? 1 : 0

  metadata {
    name      = var.service_name
    namespace = var.namespace
  }

  spec {
    min_available = var.pdb_min_available

    selector {
      match_labels = {
        app = var.service_name
      }
    }
  }
}

# ============================================================================
# Ingress (optional, for external access)
# ============================================================================

resource "kubernetes_ingress_v1" "app" {
  count = var.enable_ingress ? 1 : 0

  metadata {
    name      = var.service_name
    namespace = var.namespace

    annotations = var.ingress_annotations
  }

  spec {
    ingress_class_name = var.ingress_class_name

    dynamic "tls" {
      for_each = var.ingress_tls_enabled ? [1] : []

      content {
        hosts       = [var.ingress_host]
        secret_name = var.ingress_tls_secret_name
      }
    }

    rule {
      host = var.ingress_host

      http {
        path {
          path      = var.ingress_path
          path_type = "Prefix"

          backend {
            service {
              name = kubernetes_service.app.metadata[0].name
              port {
                number = var.service_port
              }
            }
          }
        }
      }
    }
  }
}
