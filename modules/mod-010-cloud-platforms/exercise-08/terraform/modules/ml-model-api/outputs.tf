# Cloud-Agnostic ML Model API Module - Outputs

output "namespace" {
  description = "Namespace where the service is deployed"
  value       = var.namespace
}

output "service_name" {
  description = "Name of the deployed service"
  value       = var.service_name
}

output "deployment_name" {
  description = "Name of the Kubernetes deployment"
  value       = kubernetes_deployment.app.metadata[0].name
}

output "service_account_name" {
  description = "Name of the Kubernetes service account"
  value       = kubernetes_service_account.app.metadata[0].name
}

output "service_endpoint" {
  description = "Internal service endpoint (ClusterIP)"
  value       = "${kubernetes_service.app.metadata[0].name}.${var.namespace}.svc.cluster.local:${var.service_port}"
}

output "service_cluster_ip" {
  description = "ClusterIP of the service"
  value       = kubernetes_service.app.spec[0].cluster_ip
}

output "ingress_hostname" {
  description = "Ingress hostname (if enabled)"
  value       = var.enable_ingress ? var.ingress_host : null
}

output "model_version" {
  description = "ML model version being served"
  value       = var.model_version
}

output "replicas" {
  description = "Number of replicas configured"
  value       = var.replicas
}

output "hpa_enabled" {
  description = "Whether HPA is enabled"
  value       = var.enable_hpa
}

output "hpa_min_replicas" {
  description = "Minimum replicas for HPA (if enabled)"
  value       = var.enable_hpa ? var.hpa_min_replicas : null
}

output "hpa_max_replicas" {
  description = "Maximum replicas for HPA (if enabled)"
  value       = var.enable_hpa ? var.hpa_max_replicas : null
}

output "cloud_provider" {
  description = "Cloud provider this deployment is running on"
  value       = var.cloud_provider
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}
