# EC2 Configuration
# Creates EC2 instances for ML workloads with proper security and monitoring

# Security Group for ML Instance
resource "aws_security_group" "ml_instance" {
  name        = "${local.name_prefix}-ml-instance-sg"
  description = "Security group for ML training instance"
  vpc_id      = aws_vpc.main.id

  # SSH access
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # Jupyter Notebook access (if enabled)
  dynamic "ingress" {
    for_each = var.enable_jupyter ? [1] : []
    content {
      description = "Jupyter Notebook"
      from_port   = 8888
      to_port     = 8888
      protocol    = "tcp"
      cidr_blocks = [var.allowed_ssh_cidr]
    }
  }

  # TensorBoard access (optional)
  ingress {
    description = "TensorBoard"
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # Allow all outbound traffic
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-ml-sg"
    }
  )
}

# EC2 Instance for ML Training
resource "aws_instance" "ml_training" {
  ami           = data.aws_ami.amazon_linux_2.id
  instance_type = var.instance_type

  # Network configuration
  subnet_id                   = aws_subnet.public[0].id
  vpc_security_group_ids      = [aws_security_group.ml_instance.id]
  associate_public_ip_address = true

  # IAM instance profile for S3 access
  iam_instance_profile = aws_iam_instance_profile.ml_instance.name

  # Storage configuration
  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true

    tags = merge(
      local.common_tags,
      {
        Name = "${local.name_prefix}-ml-root-volume"
      }
    )
  }

  # User data script for initialization
  user_data = templatefile("${path.module}/user_data.sh", {
    environment        = var.environment
    datasets_bucket    = local.datasets_bucket_name
    models_bucket      = local.models_bucket_name
    enable_jupyter     = var.enable_jupyter
    project_name       = var.project_name
  })

  # Metadata options for security
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  # Enable detailed monitoring
  monitoring = var.enable_cloudwatch_alarms

  tags = merge(
    local.common_tags,
    {
      Name         = "${local.name_prefix}-ml-training"
      Purpose      = "ML Training"
      WorkloadType = var.ml_workload_type
    }
  )
}

# CloudWatch Alarm for High CPU Usage
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-ml-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = var.cpu_alarm_threshold
  alarm_description   = "This metric monitors EC2 CPU utilization"
  alarm_actions       = []

  dimensions = {
    InstanceId = aws_instance.ml_training.id
  }

  tags = local.common_tags
}

# CloudWatch Alarm for Status Check Failed
resource "aws_cloudwatch_metric_alarm" "status_check_failed" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-ml-status-check"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "This metric monitors EC2 status check failures"
  alarm_actions       = []

  dimensions = {
    InstanceId = aws_instance.ml_training.id
  }

  tags = local.common_tags
}

# CloudWatch Log Group for EC2 instance logs
resource "aws_cloudwatch_log_group" "ml_instance" {
  name              = "/aws/ec2/${local.name_prefix}-ml-training"
  retention_in_days = 7

  tags = local.common_tags
}
