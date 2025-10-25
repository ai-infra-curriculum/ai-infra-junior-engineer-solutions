#!/bin/bash
#
# AI Infrastructure Development Environment Setup Script
#
# This script automates the installation of required tools for AI/ML infrastructure development.
# Supports: macOS (Intel & Apple Silicon), Ubuntu/Debian, RHEL/CentOS/Fedora
#
# Usage:
#   ./install_dependencies.sh              # Install all dependencies
#   ./install_dependencies.sh --minimal    # Install only essential tools
#   ./install_dependencies.sh --check      # Check what would be installed
#
# Author: AI Infrastructure Curriculum Team
# License: MIT

set -e  # Exit on error

#===============================================================================
# CONFIGURATION
#===============================================================================

VERSION="1.0.0"
SCRIPT_NAME=$(basename "$0")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Installation flags
INSTALL_PYTHON=true
INSTALL_DOCKER=true
INSTALL_KUBECTL=true
INSTALL_TERRAFORM=false
INSTALL_CLOUD_TOOLS=false
INSTALL_ML_LIBS=true
DRY_RUN=false

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================

print_header() {
    echo -e "\n${BOLD}${CYAN}================================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    elif [[ -f /etc/redhat-release ]]; then
        echo "redhat"
    else
        echo "unknown"
    fi
}

detect_arch() {
    local arch=$(uname -m)
    case $arch in
        x86_64)
            echo "amd64"
            ;;
        aarch64|arm64)
            echo "arm64"
            ;;
        *)
            echo "$arch"
            ;;
    esac
}

check_sudo() {
    if [[ $EUID -eq 0 ]]; then
        SUDO=""
    elif command_exists sudo; then
        SUDO="sudo"
    else
        print_error "This script requires sudo privileges"
        exit 1
    fi
}

#===============================================================================
# INSTALLATION FUNCTIONS
#===============================================================================

install_homebrew() {
    if ! command_exists brew; then
        print_info "Installing Homebrew..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Homebrew"
            return
        fi
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        print_success "Homebrew installed"
    else
        print_success "Homebrew already installed"
    fi
}

install_python_macos() {
    if ! command_exists python3; then
        print_info "Installing Python 3..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Python via Homebrew"
            return
        fi
        brew install python@3.11
        print_success "Python 3.11 installed"
    else
        local version=$(python3 --version | awk '{print $2}')
        print_success "Python $version already installed"
    fi
}

install_python_debian() {
    if ! command_exists python3.11; then
        print_info "Installing Python 3.11..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Python 3.11"
            return
        fi
        $SUDO apt-get update
        $SUDO apt-get install -y software-properties-common
        $SUDO add-apt-repository -y ppa:deadsnakes/ppa
        $SUDO apt-get update
        $SUDO apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
        print_success "Python 3.11 installed"
    else
        print_success "Python 3.11 already installed"
    fi
}

install_python_redhat() {
    if ! command_exists python3.11; then
        print_info "Installing Python 3.11..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Python 3.11"
            return
        fi
        $SUDO yum install -y python311 python311-devel python311-pip
        print_success "Python 3.11 installed"
    else
        print_success "Python 3.11 already installed"
    fi
}

install_docker_macos() {
    if ! command_exists docker; then
        print_warning "Please install Docker Desktop manually:"
        print_info "Visit: https://www.docker.com/products/docker-desktop"
        print_info "Or use: brew install --cask docker"
    else
        print_success "Docker already installed"
    fi
}

install_docker_debian() {
    if ! command_exists docker; then
        print_info "Installing Docker..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Docker"
            return
        fi

        # Remove old versions
        $SUDO apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

        # Install dependencies
        $SUDO apt-get update
        $SUDO apt-get install -y \
            ca-certificates \
            curl \
            gnupg \
            lsb-release

        # Add Docker's GPG key
        $SUDO mkdir -p /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg

        # Set up repository
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          $(lsb_release -cs) stable" | $SUDO tee /etc/apt/sources.list.d/docker.list > /dev/null

        # Install Docker
        $SUDO apt-get update
        $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

        # Add user to docker group
        $SUDO usermod -aG docker $USER

        print_success "Docker installed"
        print_warning "Please log out and back in for docker group membership to take effect"
    else
        print_success "Docker already installed"
    fi
}

install_docker_redhat() {
    if ! command_exists docker; then
        print_info "Installing Docker..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Docker"
            return
        fi

        $SUDO yum install -y yum-utils
        $SUDO yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        $SUDO yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        $SUDO systemctl start docker
        $SUDO systemctl enable docker
        $SUDO usermod -aG docker $USER

        print_success "Docker installed"
        print_warning "Please log out and back in for docker group membership to take effect"
    else
        print_success "Docker already installed"
    fi
}

install_kubectl() {
    if ! command_exists kubectl; then
        print_info "Installing kubectl..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install kubectl"
            return
        fi

        local os=$(detect_os)
        local arch=$(detect_arch)

        case $os in
            macos)
                brew install kubectl
                ;;
            debian)
                curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/$arch/kubectl"
                $SUDO install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
                rm kubectl
                ;;
            redhat)
                cat <<EOF | $SUDO tee /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://packages.cloud.google.com/yum/repos/kubernetes-el7-\$basearch
enabled=1
gpgcheck=1
gpgkey=https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF
                $SUDO yum install -y kubectl
                ;;
        esac

        print_success "kubectl installed"
    else
        print_success "kubectl already installed"
    fi
}

install_terraform() {
    if ! command_exists terraform; then
        print_info "Installing Terraform..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Terraform"
            return
        fi

        local os=$(detect_os)

        case $os in
            macos)
                brew tap hashicorp/tap
                brew install hashicorp/tap/terraform
                ;;
            debian|redhat)
                wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | $SUDO tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
                echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | $SUDO tee /etc/apt/sources.list.d/hashicorp.list
                $SUDO apt update && $SUDO apt install -y terraform
                ;;
        esac

        print_success "Terraform installed"
    else
        print_success "Terraform already installed"
    fi
}

install_aws_cli() {
    if ! command_exists aws; then
        print_info "Installing AWS CLI..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install AWS CLI"
            return
        fi

        local os=$(detect_os)
        local arch=$(detect_arch)

        case $os in
            macos)
                curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
                $SUDO installer -pkg AWSCLIV2.pkg -target /
                rm AWSCLIV2.pkg
                ;;
            *)
                curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
                unzip awscliv2.zip
                $SUDO ./aws/install
                rm -rf aws awscliv2.zip
                ;;
        esac

        print_success "AWS CLI installed"
    else
        print_success "AWS CLI already installed"
    fi
}

install_gcloud() {
    if ! command_exists gcloud; then
        print_info "Installing Google Cloud SDK..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Google Cloud SDK"
            return
        fi

        local os=$(detect_os)

        case $os in
            macos)
                brew install --cask google-cloud-sdk
                ;;
            debian)
                echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | $SUDO tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
                curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | $SUDO apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
                $SUDO apt-get update && $SUDO apt-get install -y google-cloud-sdk
                ;;
            redhat)
                $SUDO tee -a /etc/yum.repos.d/google-cloud-sdk.repo << EOM
[google-cloud-sdk]
name=Google Cloud SDK
baseurl=https://packages.cloud.google.com/yum/repos/cloud-sdk-el7-x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://packages.cloud.google.com/yum/doc/yum-key.gpg
       https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOM
                $SUDO yum install -y google-cloud-sdk
                ;;
        esac

        print_success "Google Cloud SDK installed"
    else
        print_success "Google Cloud SDK already installed"
    fi
}

install_git() {
    if ! command_exists git; then
        print_info "Installing Git..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install Git"
            return
        fi

        local os=$(detect_os)

        case $os in
            macos)
                brew install git
                ;;
            debian)
                $SUDO apt-get update
                $SUDO apt-get install -y git
                ;;
            redhat)
                $SUDO yum install -y git
                ;;
        esac

        print_success "Git installed"
    else
        print_success "Git already installed"
    fi
}

configure_git() {
    if command_exists git; then
        if ! git config user.name >/dev/null 2>&1; then
            print_info "Configuring Git..."
            read -p "Enter your name for Git: " git_name
            read -p "Enter your email for Git: " git_email

            if [[ $DRY_RUN == true ]]; then
                print_info "[DRY RUN] Would configure Git with name: $git_name, email: $git_email"
                return
            fi

            git config --global user.name "$git_name"
            git config --global user.email "$git_email"
            print_success "Git configured"
        else
            print_success "Git already configured"
        fi
    fi
}

install_vscode() {
    if ! command_exists code; then
        print_info "Installing VS Code..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install VS Code"
            return
        fi

        local os=$(detect_os)

        case $os in
            macos)
                brew install --cask visual-studio-code
                ;;
            debian)
                wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
                $SUDO install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
                $SUDO sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
                rm -f packages.microsoft.gpg
                $SUDO apt update
                $SUDO apt install -y code
                ;;
            redhat)
                $SUDO rpm --import https://packages.microsoft.com/keys/microsoft.asc
                $SUDO sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'
                $SUDO yum install -y code
                ;;
        esac

        print_success "VS Code installed"
    else
        print_success "VS Code already installed"
    fi
}

install_ml_libraries() {
    if command_exists pip3; then
        print_info "Installing ML libraries..."
        if [[ $DRY_RUN == true ]]; then
            print_info "[DRY RUN] Would install: numpy pandas scikit-learn jupyter matplotlib seaborn"
            return
        fi

        pip3 install --user numpy pandas scikit-learn jupyter matplotlib seaborn
        print_success "ML libraries installed"
    else
        print_warning "pip3 not found, skipping ML libraries"
    fi
}

#===============================================================================
# MAIN INSTALLATION LOGIC
#===============================================================================

main() {
    print_header "AI Infrastructure Development Environment Setup v${VERSION}"

    # Detect OS
    local os=$(detect_os)
    print_info "Detected OS: $os ($(uname -m))"

    if [[ $os == "unknown" ]]; then
        print_error "Unsupported operating system"
        exit 1
    fi

    # Check for sudo if needed
    if [[ $os != "macos" ]]; then
        check_sudo
    fi

    # Install package manager for macOS
    if [[ $os == "macos" ]]; then
        install_homebrew
    fi

    # Install Git first (needed for other tools)
    print_header "Installing Version Control"
    install_git
    configure_git

    # Install Python
    if [[ $INSTALL_PYTHON == true ]]; then
        print_header "Installing Python"
        case $os in
            macos)
                install_python_macos
                ;;
            debian)
                install_python_debian
                ;;
            redhat)
                install_python_redhat
                ;;
        esac
    fi

    # Install Docker
    if [[ $INSTALL_DOCKER == true ]]; then
        print_header "Installing Docker"
        case $os in
            macos)
                install_docker_macos
                ;;
            debian)
                install_docker_debian
                ;;
            redhat)
                install_docker_redhat
                ;;
        esac
    fi

    # Install Kubernetes tools
    if [[ $INSTALL_KUBECTL == true ]]; then
        print_header "Installing Kubernetes Tools"
        install_kubectl
    fi

    # Install Terraform
    if [[ $INSTALL_TERRAFORM == true ]]; then
        print_header "Installing Terraform"
        install_terraform
    fi

    # Install Cloud CLI tools
    if [[ $INSTALL_CLOUD_TOOLS == true ]]; then
        print_header "Installing Cloud CLI Tools"
        install_aws_cli
        install_gcloud
    fi

    # Install VS Code
    print_header "Installing Development Tools"
    install_vscode

    # Install ML libraries
    if [[ $INSTALL_ML_LIBS == true ]]; then
        print_header "Installing ML Libraries"
        install_ml_libraries
    fi

    # Final summary
    print_header "Installation Complete!"
    print_success "All components have been installed successfully"
    print_info "Run './check_environment.py' to verify your setup"

    if [[ $os != "macos" ]] && command_exists docker; then
        print_warning "Please log out and back in for docker group membership to take effect"
    fi
}

#===============================================================================
# COMMAND LINE ARGUMENT PARSING
#===============================================================================

show_help() {
    cat << EOF
AI Infrastructure Development Environment Setup Script

Usage: $SCRIPT_NAME [OPTIONS]

Options:
    -h, --help              Show this help message
    --minimal               Install only essential tools (Python, Git, Docker)
    --full                  Install all tools including Terraform and cloud CLIs
    --check                 Show what would be installed (dry run)
    --no-python             Skip Python installation
    --no-docker             Skip Docker installation
    --no-kubectl            Skip kubectl installation
    --cloud-tools           Install AWS CLI and Google Cloud SDK
    --ml-libs               Install ML libraries (numpy, pandas, etc.)
    --version               Show script version

Examples:
    $SCRIPT_NAME                    # Standard installation
    $SCRIPT_NAME --minimal          # Install only essentials
    $SCRIPT_NAME --full             # Install everything
    $SCRIPT_NAME --check            # Dry run to see what would be installed

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --minimal)
            INSTALL_TERRAFORM=false
            INSTALL_CLOUD_TOOLS=false
            shift
            ;;
        --full)
            INSTALL_TERRAFORM=true
            INSTALL_CLOUD_TOOLS=true
            shift
            ;;
        --check)
            DRY_RUN=true
            print_info "DRY RUN MODE - No changes will be made"
            shift
            ;;
        --no-python)
            INSTALL_PYTHON=false
            shift
            ;;
        --no-docker)
            INSTALL_DOCKER=false
            shift
            ;;
        --no-kubectl)
            INSTALL_KUBECTL=false
            shift
            ;;
        --cloud-tools)
            INSTALL_CLOUD_TOOLS=true
            shift
            ;;
        --ml-libs)
            INSTALL_ML_LIBS=true
            shift
            ;;
        --version)
            echo "$SCRIPT_NAME version $VERSION"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main installation
main
