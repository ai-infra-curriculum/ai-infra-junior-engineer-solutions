# Step-by-Step Implementation Guide: Development Environment Setup

## Overview

Set up your professional AI/ML infrastructure development environment! Learn IDE configuration, Python environments, Docker, Kubernetes tools, and essential productivity tools.

**Time**: 2-3 hours | **Difficulty**: Beginner

---

## Learning Objectives

✅ Configure a professional IDE
✅ Set up Python virtual environments
✅ Install Docker and Kubernetes tools
✅ Configure Git and SSH keys
✅ Install essential CLI tools
✅ Set up development workflow
✅ Configure shell and terminal

---

## System Requirements

### Recommended Specifications
- **OS**: Linux (Ubuntu 22.04+), macOS 12+, or Windows 11 with WSL2
- **CPU**: 4+ cores
- **RAM**: 16GB+ (32GB recommended for ML work)
- **Storage**: 50GB+ free space
- **Internet**: Stable connection for downloads

### Supported Operating Systems

**Linux (Recommended)**
- Ubuntu 22.04 LTS or later
- Debian 11+
- Fedora 36+
- Arch Linux

**macOS**
- macOS Monterey (12.0) or later
- Apple Silicon (M1/M2) or Intel supported

**Windows**
- Windows 11 with WSL2 (Ubuntu 22.04)
- Docker Desktop for Windows
- Windows Terminal

---

## Phase 1: IDE Setup

### Visual Studio Code (Recommended)

```bash
# Linux (Ubuntu/Debian)
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code

# macOS
brew install --cask visual-studio-code

# Windows
# Download from https://code.visualstudio.com/
```

### Essential VS Code Extensions

```bash
# Install via command line
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
code --install-extension eamodio.gitlens
code --install-extension github.copilot
code --install-extension esbenp.prettier-vscode
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode-remote.remote-containers
```

### VS Code Settings

```json
// settings.json
{
  "editor.formatOnSave": true,
  "editor.rulers": [80, 120],
  "editor.tabSize": 4,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "docker.showStartPage": false,
  "kubernetes.kubectlPath": "/usr/local/bin/kubectl"
}
```

### Alternative IDEs

**PyCharm Professional**
```bash
# Linux
sudo snap install pycharm-professional --classic

# macOS
brew install --cask pycharm
```

**Vim/Neovim** (Advanced)
```bash
# Install Neovim
sudo apt install neovim  # Linux
brew install neovim      # macOS

# Install vim-plug
sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
```

---

## Phase 2: Python Environment

### Install Python 3.10+

```bash
# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# macOS
brew install python@3.10

# Verify installation
python3 --version
pip3 --version
```

### Virtual Environment Setup

```bash
# Install virtualenv
pip3 install virtualenv

# Create project directory
mkdir -p ~/projects/ml-infra
cd ~/projects/ml-infra

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Poetry (Modern Alternative)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Create new project
poetry new ml-project
cd ml-project

# Install dependencies
poetry add numpy pandas scikit-learn torch

# Activate environment
poetry shell
```

### pyenv (Python Version Management)

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python versions
pyenv install 3.10.12
pyenv install 3.11.5

# Set global version
pyenv global 3.11.5

# Set local version for project
cd ~/projects/ml-infra
pyenv local 3.10.12
```

### Essential Python Packages

```bash
# Development tools
pip install \
  black \
  flake8 \
  pylint \
  mypy \
  pytest \
  pytest-cov \
  ipython \
  jupyterlab

# ML/Infrastructure packages
pip install \
  numpy \
  pandas \
  scikit-learn \
  torch \
  requests \
  pyyaml \
  python-dotenv \
  structlog

# Save requirements
pip freeze > requirements.txt
```

---

## Phase 3: Docker Setup

### Install Docker

```bash
# Linux (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# macOS
brew install --cask docker

# Start Docker Desktop
open -a Docker

# Verify installation
docker --version
docker run hello-world
```

### Docker Configuration

```bash
# Create daemon configuration
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-address-pools": [
    {
      "base": "172.17.0.0/16",
      "size": 24
    }
  ]
}
EOF

# Restart Docker
sudo systemctl restart docker
```

### Docker Compose

```bash
# Linux
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# macOS (included with Docker Desktop)
docker-compose --version

# Verify
docker-compose version
```

---

## Phase 4: Kubernetes Tools

### kubectl

```bash
# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# macOS
brew install kubectl

# Verify
kubectl version --client
```

### minikube (Local Kubernetes)

```bash
# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# macOS
brew install minikube

# Start cluster
minikube start --cpus=4 --memory=8192 --driver=docker

# Verify
kubectl get nodes
```

### Helm

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
helm version

# Add common repos
helm repo add stable https://charts.helm.sh/stable
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

### k9s (Kubernetes CLI UI)

```bash
# Linux
curl -sS https://webinstall.dev/k9s | bash

# macOS
brew install k9s

# Launch
k9s
```

### kubectx and kubens

```bash
# macOS
brew install kubectx

# Linux
sudo git clone https://github.com/ahmetb/kubectx /opt/kubectx
sudo ln -s /opt/kubectx/kubectx /usr/local/bin/kubectx
sudo ln -s /opt/kubectx/kubens /usr/local/bin/kubens

# Usage
kubectx          # List contexts
kubectx minikube # Switch context
kubens default   # Switch namespace
```

---

## Phase 5: Git and Version Control

### Install Git

```bash
# Linux
sudo apt install git

# macOS
brew install git

# Configure
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
git config --global core.editor "code --wait"
```

### SSH Key Setup

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Start ssh-agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

### Git Configuration

```bash
# .gitconfig
[user]
    name = Your Name
    email = your.email@example.com
[core]
    editor = code --wait
    autocrlf = input
[init]
    defaultBranch = main
[pull]
    rebase = false
[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    lg = log --oneline --graph --decorate --all
```

### GitHub CLI

```bash
# Install gh
# macOS
brew install gh

# Linux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate
gh auth login

# Usage
gh repo create my-project --public
gh pr create --title "Feature" --body "Description"
```

---

## Phase 6: Essential CLI Tools

### Modern CLI Replacements

```bash
# bat (better cat)
brew install bat  # macOS
sudo apt install bat  # Linux

# exa (better ls)
brew install exa  # macOS
sudo apt install exa  # Linux

# ripgrep (better grep)
brew install ripgrep  # macOS
sudo apt install ripgrep  # Linux

# fd (better find)
brew install fd  # macOS
sudo apt install fd-find  # Linux

# jq (JSON processor)
brew install jq  # macOS
sudo apt install jq  # Linux

# httpie (better curl)
brew install httpie  # macOS
sudo apt install httpie  # Linux
```

### Cloud CLIs

```bash
# AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Monitoring Tools

```bash
# htop (system monitor)
sudo apt install htop

# nvtop (GPU monitor)
sudo apt install nvtop

# ctop (container monitor)
sudo wget https://github.com/bcicen/ctop/releases/download/v0.7.7/ctop-0.7.7-linux-amd64 -O /usr/local/bin/ctop
sudo chmod +x /usr/local/bin/ctop
```

---

## Phase 7: Shell Configuration

### Zsh + Oh My Zsh

```bash
# Install Zsh
sudo apt install zsh  # Linux
brew install zsh      # macOS

# Set as default shell
chsh -s $(which zsh)

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install plugins
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

### .zshrc Configuration

```bash
# ~/.zshrc
export ZSH="$HOME/.oh-my-zsh"

ZSH_THEME="robbyrussell"

plugins=(
  git
  docker
  kubectl
  python
  zsh-autosuggestions
  zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

# Custom aliases
alias k="kubectl"
alias d="docker"
alias dc="docker-compose"
alias ll="exa -l --git"
alias cat="bat"

# Python
export PATH="$HOME/.local/bin:$PATH"
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Kubernetes
export KUBECONFIG=~/.kube/config
source <(kubectl completion zsh)

# Functions
kexec() {
  kubectl exec -it "$1" -- /bin/bash
}

klogs() {
  kubectl logs -f "$1"
}
```

### Starship Prompt (Alternative)

```bash
# Install
curl -sS https://starship.rs/install.sh | sh

# Add to .zshrc or .bashrc
echo 'eval "$(starship init zsh)"' >> ~/.zshrc

# Configure
mkdir -p ~/.config
cat > ~/.config/starship.toml <<EOF
[kubernetes]
disabled = false
format = '[$symbol$context( \($namespace\))]($style) '

[python]
format = '[${symbol}${pyenv_prefix}(${version} )(\($virtualenv\) )]($style)'

[git_branch]
format = '[$symbol$branch]($style) '
EOF
```

---

## Phase 8: Development Workflow

### Project Template

```bash
# Create project structure
mkdir -p ~/projects/ml-project/{src,tests,data,models,notebooks,scripts,docs}

# Project structure
ml-project/
├── .git/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── src/
│   ├── __init__.py
│   ├── models/
│   ├── utils/
│   └── api/
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   └── test_utils.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── scripts/
└── docs/
```

### .gitignore

```bash
# .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/

# Data
data/raw/*
data/processed/*
*.csv
*.parquet

# Models
models/*.pth
models/*.h5
*.ckpt

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp

# Environment
.env
.env.local

# Docker
.dockerignore

# Logs
*.log
logs/
```

### Makefile

```makefile
# Makefile
.PHONY: install test lint format clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ tests/
	pylint src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

docker-build:
	docker build -t ml-project:latest .

docker-run:
	docker run -p 8000:8000 ml-project:latest

k8s-deploy:
	kubectl apply -f k8s/
```

---

## Phase 9: Productivity Tools

### Tmux (Terminal Multiplexer)

```bash
# Install
sudo apt install tmux  # Linux
brew install tmux      # macOS

# Basic config
cat > ~/.tmux.conf <<EOF
# Remap prefix
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

# Split panes
bind | split-window -h
bind - split-window -v

# Enable mouse
set -g mouse on

# Status bar
set -g status-position bottom
set -g status-bg colour234
set -g status-fg colour137
EOF
```

### VS Code Remote Development

```bash
# Install Remote SSH extension
code --install-extension ms-vscode-remote.remote-ssh

# Configure SSH host
# ~/.ssh/config
Host gpu-server
    HostName 192.168.1.100
    User ml-user
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

---

## Phase 10: Verification

### Environment Check Script

```bash
#!/bin/bash
# check_env.sh

echo "=== Development Environment Check ==="

# Python
echo -n "Python: "
python3 --version

# Pip
echo -n "pip: "
pip3 --version

# Docker
echo -n "Docker: "
docker --version

# Docker Compose
echo -n "Docker Compose: "
docker-compose --version

# kubectl
echo -n "kubectl: "
kubectl version --client --short

# Helm
echo -n "Helm: "
helm version --short

# Git
echo -n "Git: "
git --version

# AWS CLI
echo -n "AWS CLI: "
aws --version

# gcloud
echo -n "gcloud: "
gcloud --version | head -1

# Check minikube
echo -n "Minikube: "
minikube status | grep host

echo "=== Environment check complete ==="
```

---

## Best Practices

✅ Use virtual environments for all Python projects
✅ Keep tools and packages updated regularly
✅ Version control your configuration files
✅ Use SSH keys instead of passwords
✅ Configure shell aliases for common commands
✅ Automate environment setup with scripts
✅ Document your setup process
✅ Use containerized development when possible
✅ Back up your configuration files
✅ Learn keyboard shortcuts for your IDE

---

## Troubleshooting

### Docker Permission Denied
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### kubectl Context Not Set
```bash
kubectl config get-contexts
kubectl config use-context minikube
```

### Python Package Conflicts
```bash
# Create fresh environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

**Development Environment Setup complete!** 🛠️

**Next Exercise**: Version Control with Git
