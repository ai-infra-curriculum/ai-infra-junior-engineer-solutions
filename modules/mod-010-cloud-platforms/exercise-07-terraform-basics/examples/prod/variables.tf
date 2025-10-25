# Variables for Production Environment

variable "aws_region" {
  description = "AWS region for production environment"
  type        = string
  default     = "us-east-1"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into instances (should be VPN or bastion)"
  type        = string
  # In production, this should be your VPN or bastion host CIDR, not 0.0.0.0/0
  default = "10.0.0.0/8"
}
