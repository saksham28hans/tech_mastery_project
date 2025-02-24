# Terraform template for deploying Real-Time Fraud Detection System on Azure

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "fraud_detection" {
  name     = "fraud-detection-rg"
  location = "East US"
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "fraud-detection-aks"
  location            = azurerm_resource_group.fraud_detection.location
  resource_group_name = azurerm_resource_group.fraud_detection.name
  dns_prefix          = "fraud-detection"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D2s_v3"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_log_analytics_workspace" "monitor" {
  name                = "fraud-detection-monitor"
  location            = azurerm_resource_group.fraud_detection.location
  resource_group_name = azurerm_resource_group.fraud_detection.name
  sku                 = "PerGB2018"
}

resource "azurerm_application_insights" "app_insights" {
  name                = "fraud-detection-insights"
  location            = azurerm_resource_group.fraud_detection.location
  resource_group_name = azurerm_resource_group.fraud_detection.name
  application_type    = "web"
}

# Deploy MLFlow on AKS
resource "kubernetes_deployment" "mlflow" {
  metadata {
    name      = "mlflow"
    namespace = "default"
  }
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "mlflow"
      }
    }
    template {
      metadata {
        labels = {
          app = "mlflow"
        }
      }
      spec {
        container {
          image = "mlflow/mlflow"
          name  = "mlflow"
          port {
            container_port = 5000
          }
        }
      }
    }
  }
}

# Deploy Kubeflow on AKS
resource "kubernetes_namespace" "kubeflow" {
  metadata {
    name = "kubeflow"
  }
}

# Deploy ArgoCD on AKS
resource "kubernetes_namespace" "argocd" {
  metadata {
    name = "argocd"
  }
}

resource "kubernetes_deployment" "argocd" {
  metadata {
    name      = "argocd-server"
    namespace = kubernetes_namespace.argocd.metadata[0].name
  }
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "argocd-server"
      }
    }
    template {
      metadata {
        labels = {
          app = "argocd-server"
        }
      }
      spec {
        container {
          image = "argoproj/argocd"
          name  = "argocd-server"
          port {
            container_port = 8080
          }
        }
      }
    }
  }
}

output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.aks.name
}

output "monitor_workspace_id" {
  value = azurerm_log_analytics_workspace.monitor.id
}
