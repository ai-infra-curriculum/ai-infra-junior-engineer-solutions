#!/bin/bash
# User Data Script for ML Training Instance
# This script runs on instance first boot to install and configure ML tools

set -e

# Variables passed from Terraform
ENVIRONMENT="${environment}"
DATASETS_BUCKET="${datasets_bucket}"
MODELS_BUCKET="${models_bucket}"
ENABLE_JUPYTER="${enable_jupyter}"
PROJECT_NAME="${project_name}"

# Log output
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "Starting user data script execution..."
echo "Environment: $ENVIRONMENT"
echo "Project: $PROJECT_NAME"

# Update system packages
echo "Updating system packages..."
yum update -y

# Install essential packages
echo "Installing essential packages..."
yum install -y \
    python3 \
    python3-pip \
    python3-devel \
    git \
    htop \
    tmux \
    vim \
    wget \
    unzip \
    gcc \
    gcc-c++ \
    make

# Upgrade pip
echo "Upgrading pip..."
pip3 install --upgrade pip setuptools wheel

# Install AWS CLI v2
echo "Installing AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Install Python ML libraries
echo "Installing Python ML libraries..."
pip3 install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    notebook \
    torch \
    torchvision \
    tensorflow \
    boto3 \
    awscli \
    ipython

# Configure Jupyter Notebook (if enabled)
if [ "$ENABLE_JUPYTER" = "true" ]; then
    echo "Configuring Jupyter Notebook..."

    # Create Jupyter directory
    mkdir -p /home/ec2-user/.jupyter
    mkdir -p /home/ec2-user/notebooks

    # Generate Jupyter config
    cat > /home/ec2-user/.jupyter/jupyter_notebook_config.py << 'EOF'
# Jupyter Notebook Configuration
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.allow_root = False
c.NotebookApp.notebook_dir = '/home/ec2-user/notebooks'
EOF

    # Create systemd service for Jupyter
    cat > /etc/systemd/system/jupyter.service << EOF
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
ExecStart=/usr/local/bin/jupyter notebook
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Set proper ownership
    chown -R ec2-user:ec2-user /home/ec2-user/.jupyter
    chown -R ec2-user:ec2-user /home/ec2-user/notebooks

    # Enable and start Jupyter service
    systemctl daemon-reload
    systemctl enable jupyter
    systemctl start jupyter

    echo "Jupyter Notebook configured and started"
fi

# Create sample Python script for testing S3 access
cat > /home/ec2-user/test_s3_access.py << EOF
#!/usr/bin/env python3
"""
Test script to verify S3 access from EC2 instance
"""
import boto3
import sys

def test_s3_access():
    """Test S3 bucket access"""
    s3_client = boto3.client('s3')

    buckets = {
        'datasets': '${datasets_bucket}',
        'models': '${models_bucket}'
    }

    print("Testing S3 access...")
    for name, bucket in buckets.items():
        try:
            response = s3_client.head_bucket(Bucket=bucket)
            print(f"✓ Successfully accessed {name} bucket: {bucket}")
        except Exception as e:
            print(f"✗ Failed to access {name} bucket: {bucket}")
            print(f"  Error: {str(e)}")
            return False

    print("\nAll S3 buckets are accessible!")
    return True

if __name__ == '__main__':
    success = test_s3_access()
    sys.exit(0 if success else 1)
EOF

chmod +x /home/ec2-user/test_s3_access.py
chown ec2-user:ec2-user /home/ec2-user/test_s3_access.py

# Create a sample notebook
if [ "$ENABLE_JUPYTER" = "true" ]; then
    cat > /home/ec2-user/notebooks/Welcome.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Welcome to Your ML Training Environment\n",
    "\n",
    "This Jupyter Notebook is running on your EC2 instance with access to S3 buckets for datasets and models.\n",
    "\n",
    "## Quick Start\n",
    "\n",
    "1. Test your Python environment\n",
    "2. Verify S3 access\n",
    "3. Start building your ML models!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test imports\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import boto3\n",
    "\n",
    "print(\"All imports successful!\")\n",
    "print(f\"NumPy version: {np.__version__}\")\n",
    "print(f\"Pandas version: {pd.__version__}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test S3 access\n",
    "s3 = boto3.client('s3')\n",
    "response = s3.list_buckets()\n",
    "print(\"Your S3 buckets:\")\n",
    "for bucket in response['Buckets']:\n",
    "    print(f\"  - {bucket['Name']}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
    chown ec2-user:ec2-user /home/ec2-user/notebooks/Welcome.ipynb
fi

# Create README file
cat > /home/ec2-user/README.md << EOF
# ML Training Environment

This EC2 instance is configured for ML training workloads.

## Environment Details
- Environment: $ENVIRONMENT
- Project: $PROJECT_NAME
- Datasets Bucket: $DATASETS_BUCKET
- Models Bucket: $MODELS_BUCKET

## Installed Software
- Python 3
- AWS CLI v2
- Machine Learning Libraries:
  - NumPy, Pandas, Scikit-learn
  - PyTorch, TensorFlow
  - Jupyter, JupyterLab

## Quick Start

### Test S3 Access
\`\`\`bash
python3 /home/ec2-user/test_s3_access.py
\`\`\`

### Access Jupyter Notebook
Open in browser: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8888

### Upload Data to S3
\`\`\`bash
aws s3 cp local-file.csv s3://$DATASETS_BUCKET/
\`\`\`

### Download Data from S3
\`\`\`bash
aws s3 cp s3://$DATASETS_BUCKET/data.csv ./
\`\`\`

## Useful Commands
- Check Jupyter status: \`sudo systemctl status jupyter\`
- View Jupyter logs: \`sudo journalctl -u jupyter -f\`
- List S3 buckets: \`aws s3 ls\`
EOF

chown ec2-user:ec2-user /home/ec2-user/README.md

# Create completion marker
echo "User data script completed successfully at $(date)" > /var/log/user-data-complete
echo "Setup complete! Instance is ready for ML workloads."
