# ----------------------------------------------------------------------------
# Resource group
# ----------------------------------------------------------------------------
resource "azurerm_resource_group" "m5" {
  name     = var.resource_group_name
  location = var.location
  tags     = { project = "m5" }
}

# ----------------------------------------------------------------------------
# Networking
# ----------------------------------------------------------------------------
resource "azurerm_virtual_network" "m5" {
  name                = "vnet-m5"
  location            = azurerm_resource_group.m5.location
  resource_group_name = azurerm_resource_group.m5.name
  address_space       = ["10.40.0.0/16"]
}

resource "azurerm_subnet" "m5" {
  name                 = "snet-m5"
  resource_group_name  = azurerm_resource_group.m5.name
  virtual_network_name = azurerm_virtual_network.m5.name
  address_prefixes     = ["10.40.1.0/24"]
}

resource "azurerm_network_security_group" "train" {
  name                = "nsg-m5-train"
  location            = azurerm_resource_group.m5.location
  resource_group_name = azurerm_resource_group.m5.name

  security_rule {
    name                       = "ssh"
    priority                   = 1000
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefixes    = var.allowed_ssh_cidrs
    destination_address_prefix = "*"
  }
}

resource "azurerm_network_security_group" "serve" {
  name                = "nsg-m5-serve"
  location            = azurerm_resource_group.m5.location
  resource_group_name = azurerm_resource_group.m5.name

  security_rule {
    name                       = "ssh"
    priority                   = 1000
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefixes    = var.allowed_ssh_cidrs
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "fastapi"
    priority                   = 1010
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = tostring(var.serve_port)
    source_address_prefixes    = var.allowed_serve_cidrs
    destination_address_prefix = "*"
  }
}

# ----------------------------------------------------------------------------
# Storage account + blob container
# ----------------------------------------------------------------------------
resource "azurerm_storage_account" "m5" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.m5.name
  location                 = azurerm_resource_group.m5.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"
  tags                     = { project = "m5" }
}

resource "azurerm_storage_container" "artifact" {
  name                  = var.container_name
  storage_account_name  = azurerm_storage_account.m5.name
  container_access_type = "private"
}

# ----------------------------------------------------------------------------
# Public IPs + NICs
# ----------------------------------------------------------------------------
resource "azurerm_public_ip" "train" {
  count               = var.create_train ? 1 : 0
  name                = "pip-m5-train"
  location            = azurerm_resource_group.m5.location
  resource_group_name = azurerm_resource_group.m5.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_public_ip" "serve" {
  count               = var.create_serve ? 1 : 0
  name                = "pip-m5-serve"
  location            = azurerm_resource_group.m5.location
  resource_group_name = azurerm_resource_group.m5.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_network_interface" "train" {
  count               = var.create_train ? 1 : 0
  name                = "nic-m5-train"
  location            = azurerm_resource_group.m5.location
  resource_group_name = azurerm_resource_group.m5.name

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.m5.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.train[0].id
  }
}

resource "azurerm_network_interface" "serve" {
  count               = var.create_serve ? 1 : 0
  name                = "nic-m5-serve"
  location            = azurerm_resource_group.m5.location
  resource_group_name = azurerm_resource_group.m5.name

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.m5.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.serve[0].id
  }
}

resource "azurerm_network_interface_security_group_association" "train" {
  count                     = var.create_train ? 1 : 0
  network_interface_id      = azurerm_network_interface.train[0].id
  network_security_group_id = azurerm_network_security_group.train.id
}

resource "azurerm_network_interface_security_group_association" "serve" {
  count                     = var.create_serve ? 1 : 0
  network_interface_id      = azurerm_network_interface.serve[0].id
  network_security_group_id = azurerm_network_security_group.serve.id
}

# ----------------------------------------------------------------------------
# User-data templating
# ----------------------------------------------------------------------------
locals {
  user_data_template = "${path.module}/../../cloud-init/_user_data.sh.tftpl"
  artifact_uri       = "az://${var.container_name}/${var.artifact_prefix}"

  user_data_train = base64encode(templatefile(local.user_data_template, {
    role                  = "train"
    git_repo              = var.git_repo
    git_ref               = var.git_ref
    artifact_dest         = local.artifact_uri
    artifact_source       = ""
    last_n_days           = var.last_n_days
    n_series              = var.n_series
    horizon               = var.horizon
    shutdown_on_done      = var.shutdown_train_on_done ? "true" : "false"
    serve_port            = var.serve_port
    serve_api_key         = var.serve_api_key
    object_store_endpoint = ""
    aws_access_key_id     = ""
    aws_secret_access_key = ""
    aws_region            = ""
  }))

  user_data_serve = base64encode(templatefile(local.user_data_template, {
    role                  = "serve"
    git_repo              = var.git_repo
    git_ref               = var.git_ref
    artifact_dest         = ""
    artifact_source       = "${local.artifact_uri}/latest"
    last_n_days           = var.last_n_days
    n_series              = var.n_series
    horizon               = var.horizon
    shutdown_on_done      = "false"
    serve_port            = var.serve_port
    serve_api_key         = var.serve_api_key
    object_store_endpoint = ""
    aws_access_key_id     = ""
    aws_secret_access_key = ""
    aws_region            = ""
  }))
}

# ----------------------------------------------------------------------------
# VMs
# ----------------------------------------------------------------------------
resource "azurerm_linux_virtual_machine" "train" {
  count                 = var.create_train ? 1 : 0
  name                  = "vm-m5-train"
  resource_group_name   = azurerm_resource_group.m5.name
  location              = azurerm_resource_group.m5.location
  size                  = var.train_vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.train[0].id]
  custom_data           = local.user_data_train
  identity { type = "SystemAssigned" }
  tags                  = { project = "m5", role = "train" }

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 60
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "ubuntu-24_04-lts"
    sku       = "server"
    version   = "latest"
  }
}

resource "azurerm_linux_virtual_machine" "serve" {
  count                 = var.create_serve ? 1 : 0
  name                  = "vm-m5-serve"
  resource_group_name   = azurerm_resource_group.m5.name
  location              = azurerm_resource_group.m5.location
  size                  = var.serve_vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.serve[0].id]
  custom_data           = local.user_data_serve
  identity { type = "SystemAssigned" }
  tags                  = { project = "m5", role = "serve" }

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    disk_size_gb         = 30
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "ubuntu-24_04-lts"
    sku       = "server"
    version   = "latest"
  }
}

# ----------------------------------------------------------------------------
# Grant the VM-managed identities access to the storage account
# ----------------------------------------------------------------------------
resource "azurerm_role_assignment" "train_blob" {
  count                = var.create_train ? 1 : 0
  scope                = azurerm_storage_account.m5.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_linux_virtual_machine.train[0].identity[0].principal_id
}

resource "azurerm_role_assignment" "serve_blob" {
  count                = var.create_serve ? 1 : 0
  scope                = azurerm_storage_account.m5.id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_linux_virtual_machine.serve[0].identity[0].principal_id
}
