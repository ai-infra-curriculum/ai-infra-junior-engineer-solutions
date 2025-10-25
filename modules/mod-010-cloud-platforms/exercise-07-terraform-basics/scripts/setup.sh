#!/bin/bash
# setup.sh - Install and configure Terraform
# This script installs Terraform and verifies the installation

set -e

TERRAFORM_VERSION="1.6.0"
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_NC='\033[0m' # No Color

echo "========================================"
echo "Terraform Setup Script"
echo "========================================"
echo ""

# Function to print colored output
print_status() {
    echo -e "${COLOR_GREEN}[+]${COLOR_NC} $1"
}

print_error() {
    echo -e "${COLOR_RED}[!]${COLOR_NC} $1"
}

print_warning() {
    echo -e "${COLOR_YELLOW}[*]${COLOR_NC} $1"
}

# Check if Terraform is already installed
if command -v terraform &> /dev/null; then
    INSTALLED_VERSION=$(terraform version | head -n 1 | awk '{print $2}' | sed 's/v//')
    print_status "Terraform is already installed (version: $INSTALLED_VERSION)"

    if [ "$INSTALLED_VERSION" = "$TERRAFORM_VERSION" ]; then
        print_status "Version matches the required version!"
    else
        print_warning "Installed version differs from recommended version ($TERRAFORM_VERSION)"
        read -p "Do you want to reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
else
    print_status "Terraform not found. Installing..."
fi

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$ARCH" in
    x86_64)
        ARCH="amd64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    *)
        print_error "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

print_status "Detected OS: $OS, Architecture: $ARCH"

# Install Terraform
case "$OS" in
    darwin)
        print_status "Installing Terraform on macOS..."
        if command -v brew &> /dev/null; then
            brew tap hashicorp/tap
            brew install hashicorp/tap/terraform
        else
            print_error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        ;;

    linux)
        print_status "Installing Terraform on Linux..."

        # Download Terraform
        TERRAFORM_URL="https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_${OS}_${ARCH}.zip"
        print_status "Downloading from: $TERRAFORM_URL"

        wget -q "$TERRAFORM_URL" -O /tmp/terraform.zip

        # Extract and install
        unzip -q /tmp/terraform.zip -d /tmp/
        sudo mv /tmp/terraform /usr/local/bin/
        sudo chmod +x /usr/local/bin/terraform

        # Cleanup
        rm /tmp/terraform.zip

        print_status "Terraform installed successfully"
        ;;

    *)
        print_error "Unsupported OS: $OS"
        print_warning "Please install Terraform manually from: https://www.terraform.io/downloads"
        exit 1
        ;;
esac

# Verify installation
echo ""
print_status "Verifying Terraform installation..."
if terraform version; then
    print_status "Terraform installed successfully!"
else
    print_error "Terraform installation failed"
    exit 1
fi

echo ""
print_status "Checking AWS CLI..."
if command -v aws &> /dev/null; then
    print_status "AWS CLI is installed ($(aws --version))"
else
    print_warning "AWS CLI not found. You'll need it to authenticate with AWS."
    print_warning "Install from: https://aws.amazon.com/cli/"
fi

echo ""
print_status "Checking AWS credentials..."
if aws sts get-caller-identity &> /dev/null; then
    print_status "AWS credentials are configured!"
    aws sts get-caller-identity
else
    print_warning "AWS credentials not configured or invalid"
    print_warning "Run: aws configure"
fi

echo ""
print_status "Setup complete! You can now use Terraform."
echo ""
echo "Next steps:"
echo "  1. Configure AWS credentials: aws configure"
echo "  2. Navigate to terraform directory: cd terraform/"
echo "  3. Initialize Terraform: ./scripts/init.sh"
echo "  4. Create infrastructure: ./scripts/apply.sh"
echo ""
