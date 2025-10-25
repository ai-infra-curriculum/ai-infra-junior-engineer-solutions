#!/bin/bash
#
# Automated Development Environment Setup for macOS
# AI Infrastructure Junior Engineer Curriculum
#
# This script installs and configures:
# - Homebrew package manager
# - iTerm2 terminal
# - Zsh + Oh My Zsh
# - Git
# - Python 3.11 with pyenv
# - VS Code
# - Docker Desktop
# - AWS CLI, gcloud
# - Productivity tools
#
# Usage: ./setup-macos.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo ""
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_error "This script is for macOS only!"
    exit 1
fi

print_header "AI Infrastructure Dev Environment Setup - macOS"

log_info "This script will install development tools for AI Infrastructure."
log_info "You may be prompted for your password (sudo access required)."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Setup cancelled."
    exit 0
fi

# ============================================================
# Step 1: Install Homebrew
# ============================================================
print_header "Step 1: Installing Homebrew"

if command -v brew &> /dev/null; then
    log_success "Homebrew already installed"
    brew --version
else
    log_info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        # Intel
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi

    log_success "Homebrew installed"
fi

# Update Homebrew
log_info "Updating Homebrew..."
brew update

# ============================================================
# Step 2: Install iTerm2
# ============================================================
print_header "Step 2: Installing iTerm2"

if brew list --cask iterm2 &> /dev/null; then
    log_success "iTerm2 already installed"
else
    log_info "Installing iTerm2..."
    brew install --cask iterm2
    log_success "iTerm2 installed"
    log_info "You can switch to iTerm2 after setup completes"
fi

# ============================================================
# Step 3: Install Zsh + Oh My Zsh
# ============================================================
print_header "Step 3: Installing Zsh + Oh My Zsh"

if command -v zsh &> /dev/null; then
    log_success "Zsh already installed: $(zsh --version)"
else
    log_info "Installing Zsh..."
    brew install zsh
    log_success "Zsh installed"
fi

# Set Zsh as default shell
if [[ "$SHELL" != *"zsh"* ]]; then
    log_info "Setting Zsh as default shell..."
    chsh -s $(which zsh)
    log_success "Default shell set to Zsh (takes effect on next login)"
fi

# Install Oh My Zsh
if [ -d "$HOME/.oh-my-zsh" ]; then
    log_success "Oh My Zsh already installed"
else
    log_info "Installing Oh My Zsh..."
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
    log_success "Oh My Zsh installed"
fi

# Install useful Oh My Zsh plugins
log_info "Installing Zsh plugins..."
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions 2>/dev/null || true
git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting 2>/dev/null || true
log_success "Zsh plugins installed"

# ============================================================
# Step 4: Install Git
# ============================================================
print_header "Step 4: Installing Git"

if command -v git &> /dev/null; then
    log_success "Git already installed: $(git --version)"
else
    log_info "Installing Git..."
    brew install git
    log_success "Git installed"
fi

# ============================================================
# Step 5: Install Python with pyenv
# ============================================================
print_header "Step 5: Installing Python with pyenv"

if command -v pyenv &> /dev/null; then
    log_success "pyenv already installed: $(pyenv --version)"
else
    log_info "Installing pyenv..."
    brew install pyenv

    # Add pyenv to shell
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc

    # Load pyenv in current session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    log_success "pyenv installed"
fi

# Install Python 3.11
log_info "Installing Python 3.11..."
if pyenv versions | grep -q "3.11"; then
    log_success "Python 3.11 already installed"
else
    pyenv install 3.11.7
    pyenv global 3.11.7
    log_success "Python 3.11.7 installed and set as global"
fi

# Install pip packages
log_info "Installing essential Python packages..."
pip install --upgrade pip
pip install virtualenv poetry black flake8 mypy pytest ipython
log_success "Essential Python packages installed"

# ============================================================
# Step 6: Install VS Code
# ============================================================
print_header "Step 6: Installing VS Code"

if command -v code &> /dev/null; then
    log_success "VS Code already installed: $(code --version | head -n1)"
else
    log_info "Installing VS Code..."
    brew install --cask visual-studio-code
    log_success "VS Code installed"
fi

# ============================================================
# Step 7: Install Docker Desktop
# ============================================================
print_header "Step 7: Installing Docker Desktop"

if command -v docker &> /dev/null; then
    log_success "Docker already installed: $(docker --version)"
else
    log_info "Installing Docker Desktop..."
    brew install --cask docker
    log_success "Docker Desktop installed"
    log_warning "Please open Docker Desktop from Applications to complete setup"
fi

# ============================================================
# Step 8: Install Cloud CLI Tools
# ============================================================
print_header "Step 8: Installing Cloud CLI Tools"

# AWS CLI
if command -v aws &> /dev/null; then
    log_success "AWS CLI already installed: $(aws --version)"
else
    log_info "Installing AWS CLI v2..."
    brew install awscli
    log_success "AWS CLI installed"
fi

# Google Cloud SDK
if command -v gcloud &> /dev/null; then
    log_success "gcloud already installed: $(gcloud --version | head -n1)"
else
    log_info "Installing Google Cloud SDK..."
    brew install --cask google-cloud-sdk
    log_success "gcloud installed"
fi

# ============================================================
# Step 9: Install Productivity Tools
# ============================================================
print_header "Step 9: Installing Productivity Tools"

TOOLS=(
    "tmux:Terminal multiplexer"
    "fzf:Fuzzy finder"
    "ripgrep:Fast grep"
    "jq:JSON processor"
    "htop:System monitor"
    "tree:Directory tree"
    "wget:File downloader"
    "curl:Data transfer tool"
    "bat:Better cat"
    "exa:Better ls"
)

for tool in "${TOOLS[@]}"; do
    IFS=':' read -r name description <<< "$tool"
    if command -v $name &> /dev/null; then
        log_success "$description ($name) already installed"
    else
        log_info "Installing $description ($name)..."
        brew install $name
    fi
done

log_success "Productivity tools installed"

# ============================================================
# Step 10: Install kubectl (Kubernetes CLI)
# ============================================================
print_header "Step 10: Installing kubectl"

if command -v kubectl &> /dev/null; then
    log_success "kubectl already installed: $(kubectl version --client --short)"
else
    log_info "Installing kubectl..."
    brew install kubectl
    log_success "kubectl installed"
fi

# ============================================================
# Step 11: Install Terraform
# ============================================================
print_header "Step 11: Installing Terraform"

if command -v terraform &> /dev/null; then
    log_success "Terraform already installed: $(terraform --version | head -n1)"
else
    log_info "Installing Terraform..."
    brew tap hashicorp/tap
    brew install hashicorp/tap/terraform
    log_success "Terraform installed"
fi

# ============================================================
# Final Steps
# ============================================================
print_header "Setup Complete!"

echo ""
log_success "Development environment setup complete!"
echo ""
echo "✅ Installed Tools:"
echo "   - Homebrew"
echo "   - iTerm2"
echo "   - Zsh + Oh My Zsh"
echo "   - Git"
echo "   - Python 3.11 (pyenv)"
echo "   - VS Code"
echo "   - Docker Desktop"
echo "   - AWS CLI"
echo "   - gcloud"
echo "   - kubectl"
echo "   - Terraform"
echo "   - Productivity tools (tmux, fzf, ripgrep, etc.)"
echo ""
echo "📝 Next Steps:"
echo "   1. Restart your terminal (or run: source ~/.zshrc)"
echo "   2. Open Docker Desktop and complete initial setup"
echo "   3. Configure Git: git config --global user.name 'Your Name'"
echo "   4. Configure Git: git config --global user.email 'your@email.com'"
echo "   5. Configure AWS CLI: aws configure"
echo "   6. Run validation: python scripts/validate_setup.py"
echo "   7. Install VS Code extensions (see README.md)"
echo ""
echo "🎉 Happy coding!"
echo ""
