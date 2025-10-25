# Variables for Development Environment

variable "aws_region" {
  description = "AWS region for dev environment"
  type        = string
  default     = "us-east-1"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into instances"
  type        = string
  default     = "0.0.0.0/0" # Change this to your IP in production!
}
