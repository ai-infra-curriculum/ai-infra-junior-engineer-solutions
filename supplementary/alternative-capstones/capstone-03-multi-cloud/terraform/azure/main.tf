# Azure Infrastructure Module
# Provisions AKS cluster, Blob Storage, Azure SQL, and supporting infrastructure

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.75"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.45"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = var.environment == "prod" ? true : false
    }
  }
}

data "azurerm_client_config" "current" {}

# Resource Group
resource "azurerm_resource_group" "ml_platform" {
  name     = "${var.project_name}-rg-${var.environment}"
  location = var.azure_location

  tags = {
    Project     = "ml-platform"
    Environment = var.environment
    ManagedBy   = "terraform"
    Cloud       = "azure"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "ml_platform" {
  name                = "${var.project_name}-vnet-${var.environment}"
  location            = azurerm_resource_group.ml_platform.location
  resource_group_name = azurerm_resource_group.ml_platform.name
  address_space       = [var.vnet_address_space]

  tags = azurerm_resource_group.ml_platform.tags
}

# Subnet for AKS
resource "azurerm_subnet" "aks" {
  name                 = "${var.project_name}-aks-subnet-${var.environment}"
  resource_group_name  = azurerm_resource_group.ml_platform.name
  virtual_network_name = azurerm_virtual_network.ml_platform.name
  address_prefixes     = [var.aks_subnet_address_prefix]
}

# Subnet for Azure SQL
resource "azurerm_subnet" "database" {
  name                 = "${var.project_name}-db-subnet-${var.environment}"
  resource_group_name  = azurerm_resource_group.ml_platform.name
  virtual_network_name = azurerm_virtual_network.ml_platform.name
  address_prefixes     = [var.db_subnet_address_prefix]

  delegation {
    name = "sql-delegation"

    service_delegation {
      name = "Microsoft.Sql/managedInstances"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action",
        "Microsoft.Network/virtualNetworks/subnets/prepareNetworkPolicies/action"
      ]
    }
  }
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "ml_platform" {
  name                = "${var.project_name}-aks-${var.environment}"
  location            = azurerm_resource_group.ml_platform.location
  resource_group_name = azurerm_resource_group.ml_platform.name
  dns_prefix          = "${var.project_name}-${var.environment}"
  kubernetes_version  = var.kubernetes_version

  default_node_pool {
    name                = "general"
    node_count          = var.desired_nodes
    vm_size             = "Standard_D4s_v3"
    vnet_subnet_id      = azurerm_subnet.aks.id
    enable_auto_scaling = true
    min_count           = var.min_nodes
    max_count           = var.max_nodes
    os_disk_size_gb     = 100
    os_disk_type        = "Managed"

    node_labels = {
      role = "general"
    }

    upgrade_settings {
      max_surge = "10%"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "calico"
    load_balancer_sku = "standard"
    service_cidr      = var.aks_service_cidr
    dns_service_ip    = var.aks_dns_service_ip
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.ml_platform.id
  }

  microsoft_defender {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.ml_platform.id
  }

  azure_policy_enabled = true

  maintenance_window {
    allowed {
      day   = "Sunday"
      hours = [3, 4]
    }
  }

  tags = azurerm_resource_group.ml_platform.tags
}

# ML Workloads Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "ml_workloads" {
  name                  = "mlworkloads"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.ml_platform.id
  vm_size               = "Standard_F8s_v2"
  node_count            = 1

  enable_auto_scaling = true
  min_count           = 1
  max_count           = 10

  vnet_subnet_id  = azurerm_subnet.aks.id
  os_disk_size_gb = 100
  priority        = "Spot"
  eviction_policy = "Delete"
  spot_max_price  = -1

  node_labels = {
    role     = "ml-workloads"
    workload = "training"
  }

  node_taints = [
    "workload=ml:NoSchedule"
  ]

  tags = azurerm_resource_group.ml_platform.tags
}

# GPU Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.ml_platform.id
  vm_size               = "Standard_NC6s_v3"
  node_count            = 0

  enable_auto_scaling = true
  min_count           = 0
  max_count           = 5

  vnet_subnet_id  = azurerm_subnet.aks.id
  os_disk_size_gb = 100
  priority        = "Spot"
  eviction_policy = "Delete"
  spot_max_price  = -1

  node_labels = {
    role             = "gpu"
    "nvidia.com/gpu" = "true"
  }

  node_taints = [
    "nvidia.com/gpu=true:NoSchedule"
  ]

  tags = azurerm_resource_group.ml_platform.tags
}

# Storage Account for ML models
resource "azurerm_storage_account" "ml_models" {
  name                     = "${replace(var.project_name, "-", "")}mlmodels${var.environment}"
  resource_group_name      = azurerm_resource_group.ml_platform.name
  location                 = azurerm_resource_group.ml_platform.location
  account_tier             = "Standard"
  account_replication_type = var.environment == "prod" ? "GRS" : "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true

  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 30
    }

    container_delete_retention_policy {
      days = 30
    }
  }

  tags = azurerm_resource_group.ml_platform.tags
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.ml_models.name
  container_access_type = "private"
}

# Storage Account for data lake
resource "azurerm_storage_account" "data_lake" {
  name                     = "${replace(var.project_name, "-", "")}datalake${var.environment}"
  resource_group_name      = azurerm_resource_group.ml_platform.name
  location                 = azurerm_resource_group.ml_platform.location
  account_tier             = "Standard"
  account_replication_type = var.environment == "prod" ? "GRS" : "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true

  blob_properties {
    delete_retention_policy {
      days = 30
    }
  }

  tags = azurerm_resource_group.ml_platform.tags
}

resource "azurerm_storage_container" "raw_data" {
  name                  = "raw-data"
  storage_account_name  = azurerm_storage_account.data_lake.name
  container_access_type = "private"
}

# Azure SQL Server
resource "azurerm_mssql_server" "ml_platform" {
  name                         = "${var.project_name}-sql-${var.environment}"
  resource_group_name          = azurerm_resource_group.ml_platform.name
  location                     = azurerm_resource_group.ml_platform.location
  version                      = "12.0"
  administrator_login          = var.sql_admin_username
  administrator_login_password = var.sql_admin_password
  minimum_tls_version          = "1.2"

  azuread_administrator {
    login_username = "AzureAD Admin"
    object_id      = data.azurerm_client_config.current.object_id
  }

  tags = azurerm_resource_group.ml_platform.tags
}

resource "azurerm_mssql_database" "ml_platform" {
  name           = var.db_name
  server_id      = azurerm_mssql_server.ml_platform.id
  collation      = "SQL_Latin1_General_CP1_CI_AS"
  sku_name       = var.sql_sku_name
  zone_redundant = var.environment == "prod" ? true : false

  short_term_retention_policy {
    retention_days = 7
  }

  long_term_retention_policy {
    weekly_retention  = "P4W"
    monthly_retention = "P12M"
    yearly_retention  = "P5Y"
    week_of_year      = 1
  }

  tags = azurerm_resource_group.ml_platform.tags
}

# Azure Cache for Redis
resource "azurerm_redis_cache" "ml_platform" {
  name                = "${var.project_name}-redis-${var.environment}"
  location            = azurerm_resource_group.ml_platform.location
  resource_group_name = azurerm_resource_group.ml_platform.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  redis_configuration {
    maxmemory_policy = "allkeys-lru"
  }

  patch_schedule {
    day_of_week    = "Sunday"
    start_hour_utc = 3
  }

  tags = azurerm_resource_group.ml_platform.tags
}

# Container Registry
resource "azurerm_container_registry" "ml_platform" {
  name                = "${replace(var.project_name, "-", "")}acr${var.environment}"
  resource_group_name = azurerm_resource_group.ml_platform.name
  location            = azurerm_resource_group.ml_platform.location
  sku                 = var.environment == "prod" ? "Premium" : "Standard"
  admin_enabled       = false

  georeplications = var.environment == "prod" ? [
    {
      location                = "westus2"
      zone_redundancy_enabled = true
    }
  ] : []

  network_rule_set = var.environment == "prod" ? {
    default_action = "Deny"
    ip_rule = []
    virtual_network = [{
      action    = "Allow"
      subnet_id = azurerm_subnet.aks.id
    }]
  } : null

  tags = azurerm_resource_group.ml_platform.tags
}

# Grant AKS access to ACR
resource "azurerm_role_assignment" "aks_acr" {
  principal_id                     = azurerm_kubernetes_cluster.ml_platform.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.ml_platform.id
  skip_service_principal_aad_check = true
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "ml_platform" {
  name                = "${var.project_name}-logs-${var.environment}"
  location            = azurerm_resource_group.ml_platform.location
  resource_group_name = azurerm_resource_group.ml_platform.name
  sku                 = "PerGB2018"
  retention_in_days   = var.environment == "prod" ? 90 : 30

  tags = azurerm_resource_group.ml_platform.tags
}

# Application Insights
resource "azurerm_application_insights" "ml_platform" {
  name                = "${var.project_name}-appinsights-${var.environment}"
  location            = azurerm_resource_group.ml_platform.location
  resource_group_name = azurerm_resource_group.ml_platform.name
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.ml_platform.id

  tags = azurerm_resource_group.ml_platform.tags
}

# Key Vault for secrets management
resource "azurerm_key_vault" "ml_platform" {
  name                       = "${var.project_name}-kv-${var.environment}"
  location                   = azurerm_resource_group.ml_platform.location
  resource_group_name        = azurerm_resource_group.ml_platform.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 90
  purge_protection_enabled   = var.environment == "prod" ? true : false

  network_acls {
    bypass         = "AzureServices"
    default_action = "Deny"
    ip_rules       = []
    virtual_network_subnet_ids = [
      azurerm_subnet.aks.id
    ]
  }

  tags = azurerm_resource_group.ml_platform.tags
}

# Grant AKS access to Key Vault
resource "azurerm_key_vault_access_policy" "aks" {
  key_vault_id = azurerm_key_vault.ml_platform.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_kubernetes_cluster.ml_platform.kubelet_identity[0].object_id

  secret_permissions = [
    "Get",
    "List"
  ]
}
