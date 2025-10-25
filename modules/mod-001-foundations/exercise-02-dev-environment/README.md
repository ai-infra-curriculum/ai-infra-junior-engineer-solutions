# Exercise 02: Development Environment Setup

## Overview

Set up a complete, production-grade development environment with all the tools and configurations you'll need for AI Infrastructure Engineering. This exercise ensures you have a consistent, efficient, and reproducible development setup.

## Learning Objectives

- ✅ Install and configure a modern terminal (Zsh, Oh My Zsh)
- ✅ Set up VS Code or PyCharm with essential extensions
- ✅ Manage Python versions and virtual environments
- ✅ Install Docker Desktop and verify functionality
- ✅ Configure cloud CLI tools (AWS, GCP, Azure)
- ✅ Create a dotfiles repository for reproducibility
- ✅ Install essential productivity and development tools

## Prerequisites

**Hardware:**
- Modern computer (2015+)
- 8GB+ RAM (16GB recommended)
- 20GB+ free disk space
- Internet connection

**Access:**
- Administrator/sudo privileges
- Ability to install software

**Accounts to Create:**
- GitHub account (free)
- Docker Hub account (free)
- AWS account (optional, free tier)

---

## Tools We'll Install

### Core Tools
- **Terminal:** iTerm2 (Mac) or Windows Terminal (Windows) or GNOME Terminal (Linux)
- **Shell:** Zsh with Oh My Zsh
- **Text Editor:** VS Code or PyCharm Community
- **Version Control:** Git
- **Python:** Python 3.11+ with pyenv
- **Container Runtime:** Docker Desktop
- **Package Managers:** Homebrew (Mac), apt (Linux), Chocolatey (Windows)

### Cloud CLI Tools
- **AWS CLI** (v2)
- **gcloud** (Google Cloud SDK)
- **Azure CLI** (optional)

### Productivity Tools
- **tmux** - Terminal multiplexer
- **fzf** - Fuzzy file finder
- **ripgrep** - Fast grep alternative
- **jq** - JSON processor
- **htop** - System monitor
- **tree** - Directory visualization

---

## Installation Guides by Operating System

Choose your operating system:

### macOS
See: [docs/setup-macos.md](docs/setup-macos.md)

**Summary:**
1. Install Homebrew
2. Install iTerm2
3. Install Zsh + Oh My Zsh
4. Install VS Code
5. Install Python with pyenv
6. Install Docker Desktop
7. Install cloud CLIs
8. Configure dotfiles

**Estimated Time:** 2-3 hours

---

### Linux (Ubuntu/Debian)
See: [docs/setup-linux.md](docs/setup-linux.md)

**Summary:**
1. Update system packages
2. Install Zsh + Oh My Zsh
3. Install VS Code
4. Install Python with pyenv
5. Install Docker
6. Install cloud CLIs
7. Configure dotfiles

**Estimated Time:** 2-3 hours

---

### Windows (WSL2)
See: [docs/setup-windows.md](docs/setup-windows.md)

**Summary:**
1. Enable WSL2
2. Install Ubuntu on WSL2
3. Install Windows Terminal
4. Follow Linux setup within WSL2
5. Install Docker Desktop for Windows
6. Configure VS Code for WSL2

**Estimated Time:** 3-4 hours

---

## Quick Start (Automated Setup)

We provide automated setup scripts for each OS:

### macOS

```bash
# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions/main/modules/mod-001-foundations/exercise-02-dev-environment/scripts/setup-macos.sh | bash
```

### Linux

```bash
# Download and run setup script
wget -qO- https://raw.githubusercontent.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions/main/modules/mod-001-foundations/exercise-02-dev-environment/scripts/setup-linux.sh | bash
```

### Windows (WSL2)

```powershell
# Run in PowerShell as Administrator
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions/main/modules/mod-001-foundations/exercise-02-dev-environment/scripts/setup-windows.ps1" -OutFile "setup-windows.ps1"
.\setup-windows.ps1
```

**Note:** Review scripts before running to understand what they install.

---

## Manual Setup Guide

If you prefer manual installation, follow [STEP_BY_STEP.md](STEP_BY_STEP.md) for detailed instructions.

---

## VS Code Extensions

Essential extensions to install:

### Python Development
- **Python** (Microsoft) - Language support
- **Pylance** - Fast language server
- **Python Docstring Generator** - Auto-generate docstrings
- **autoDocstring** - Generate docstrings from function signatures

### Docker & Kubernetes
- **Docker** (Microsoft) - Container management
- **Kubernetes** (Microsoft) - K8s support
- **YAML** - YAML language support

### Git & Version Control
- **GitLens** - Git supercharged
- **Git Graph** - Visualize git history
- **GitHub Pull Requests** - Review PRs in VS Code

### Productivity
- **Remote - SSH** - Edit files on remote servers
- **Remote - Containers** - Develop inside containers
- **Remote - WSL** - Windows Subsystem for Linux support
- **Live Share** - Collaborative editing
- **Code Spell Checker** - Spell check your code

### Infrastructure as Code
- **HashiCorp Terraform** - Terraform language support
- **AWS Toolkit** - AWS integration
- **Azure Tools** - Azure integration

### Quality & Formatting
- **Black Formatter** - Python code formatter
- **isort** - Sort Python imports
- **Flake8** - Python linter
- **Prettier** - Multi-language formatter
- **EditorConfig** - Consistent coding styles

**Installation:**
```bash
# Install all recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
code --install-extension eamodio.gitlens
# ... (see scripts/install-vscode-extensions.sh for complete list)
```

---

## Python Environment Management

### Using pyenv

**Install pyenv:**
```bash
# macOS/Linux
curl https://pyenv.run | bash

# Add to ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

**Install Python versions:**
```bash
# Install Python 3.11
pyenv install 3.11.7

# Set as global default
pyenv global 3.11.7

# Verify
python --version  # Should show 3.11.7
```

### Using virtualenv

**Create virtual environment:**
```bash
# Create venv for a project
python -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Deactivate
deactivate
```

### Using Poetry (Recommended)

**Install Poetry:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Usage:**
```bash
# Create new project
poetry new myproject
cd myproject

# Install dependencies
poetry install

# Add packages
poetry add fastapi uvicorn

# Run in virtual environment
poetry run python main.py
```

---

## Docker Setup

### Verify Installation

```bash
# Check Docker version
docker --version

# Check Docker Compose
docker compose version

# Run hello-world
docker run hello-world

# Check running containers
docker ps
```

### Docker Configuration

**Increase resources (Docker Desktop):**
- Memory: 4GB minimum, 8GB recommended
- CPUs: 2 minimum, 4 recommended
- Disk: 20GB minimum

**Enable Kubernetes (optional):**
- Docker Desktop → Settings → Kubernetes → Enable

---

## Dotfiles Repository

Create a dotfiles repository to version control your configurations:

### Structure

```
dotfiles/
├── .zshrc
├── .gitconfig
├── .vimrc
├── .tmux.conf
├── vscode/
│   ├── settings.json
│   └── keybindings.json
├── install.sh
└── README.md
```

### Setup

```bash
# Create dotfiles repo
mkdir ~/dotfiles
cd ~/dotfiles
git init

# Copy current configs
cp ~/.zshrc .zshrc
cp ~/.gitconfig .gitconfig
cp ~/.vimrc .vimrc

# Create install script
cat > install.sh << 'EOF'
#!/bin/bash
# Link dotfiles
ln -sf ~/dotfiles/.zshrc ~/.zshrc
ln -sf ~/dotfiles/.gitconfig ~/.gitconfig
ln -sf ~/dotfiles/.vimrc ~/.vimrc
echo "Dotfiles linked!"
EOF

chmod +x install.sh

# Commit and push
git add .
git commit -m "Initial dotfiles"
git remote add origin https://github.com/yourusername/dotfiles.git
git push -u origin main
```

See [dotfiles/](dotfiles/) for example configurations.

---

## Validation

### Run Validation Script

```bash
python scripts/validate_setup.py
```

**Expected output:**
```
============================================================
Development Environment Validation
============================================================

Checking Core Tools...
✅ Zsh installed: /bin/zsh (version 5.9)
✅ Oh My Zsh installed
✅ Git installed: 2.42.0
✅ VS Code installed: 1.85.0

Checking Python Environment...
✅ Python installed: 3.11.7
✅ pyenv installed: 2.3.35
✅ pip installed: 23.3.2
✅ poetry installed: 1.7.1

Checking Docker...
✅ Docker installed: 24.0.7
✅ Docker Compose installed: 2.23.3
✅ Docker running
✅ Docker hello-world works

Checking Cloud CLI Tools...
✅ AWS CLI installed: aws-cli/2.15.0
✅ gcloud installed: 459.0.0
⚠️  Azure CLI not installed (optional)

Checking Productivity Tools...
✅ tmux installed: 3.3a
✅ fzf installed: 0.45.0
✅ ripgrep installed: 14.0.3
✅ jq installed: 1.7

============================================================
Validation Summary
============================================================

✅ All required tools installed!
⚠️  1 optional tool missing (Azure CLI)

🎉 Development environment ready!

Next Steps:
1. Configure your dotfiles
2. Install VS Code extensions
3. Test Python environment
4. Move to Exercise 03
```

---

## Troubleshooting

### Common Issues

#### "zsh: command not found: brew"
**Solution:**
```bash
# Reinstall Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

#### "pyenv: command not found"
**Solution:**
```bash
# Add to ~/.zshrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Reload
source ~/.zshrc
```

#### "Docker daemon not running"
**Solution:**
- macOS: Open Docker Desktop application
- Linux: `sudo systemctl start docker`
- Windows: Start Docker Desktop

#### "Permission denied" when running Docker
**Solution:**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended)
sudo docker ps
```

#### VS Code extensions not installing
**Solution:**
```bash
# Check VS Code CLI
code --version

# Reinstall specific extension
code --install-extension ms-python.python --force
```

See [docs/troubleshooting.md](docs/troubleshooting.md) for more solutions.

---

## Post-Setup Configuration

### Configure Git

```bash
# Set your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default branch name
git config --global init.defaultBranch main

# Enable color output
git config --global color.ui auto

# Set default editor
git config --global core.editor "code --wait"

# Configure aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
```

### Configure AWS CLI

```bash
# Configure credentials
aws configure

# Test
aws s3 ls
```

### Configure SSH Keys

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Start ssh-agent
eval "$(ssh-agent -s)"

# Add key
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH Keys → New SSH Key
```

---

## Cheat Sheet

Create a quick reference: [docs/cheat-sheet.md](docs/cheat-sheet.md)

**Essential Commands:**
```bash
# Python
python --version
pip install package-name
python -m venv venv

# Docker
docker ps
docker images
docker run -it ubuntu bash
docker compose up -d

# Git
git status
git add .
git commit -m "message"
git push

# Kubernetes
kubectl get pods
kubectl describe pod <name>
kubectl logs <pod-name>
```

---

## Deliverables

By the end of this exercise, you should have:

✅ **Fully configured development environment**
- Terminal (Zsh + Oh My Zsh)
- Code editor (VS Code with extensions)
- Python environment (pyenv + virtualenv/poetry)
- Docker Desktop running

✅ **Cloud CLI tools installed and configured**
- AWS CLI
- gcloud (optional)
- Azure CLI (optional)

✅ **Dotfiles repository**
- Version controlled configurations
- Install script for reproducibility
- Pushed to GitHub

✅ **Validation passed**
- All required tools installed
- All tools functioning correctly
- Quick reference cheat sheet created

---

## Time Breakdown

**Estimated Total:** 4-6 hours

- Initial setup and installations: 2-3 hours
- Configuration and customization: 1-2 hours
- Creating dotfiles repository: 30 minutes
- Validation and testing: 30 minutes

---

## Next Steps

After completing this exercise:

1. **Exercise 03: Version Control with Git** - Master Git workflows
2. Test your environment with a small project
3. Customize your dotfiles further
4. Explore productivity tools (tmux, fzf)

---

## Additional Resources

### Official Documentation
- [Oh My Zsh](https://ohmyz.sh/)
- [VS Code Docs](https://code.visualstudio.com/docs)
- [Docker Docs](https://docs.docker.com/)
- [pyenv Documentation](https://github.com/pyenv/pyenv)
- [Poetry Documentation](https://python-poetry.org/docs/)

### Community Resources
- [Awesome Dotfiles](https://github.com/webpro/awesome-dotfiles)
- [Awesome Dev Environment](https://github.com/jondot/awesome-devenv)
- [Awesome VSCode](https://github.com/viatsko/awesome-vscode)

### Video Tutorials
- [Setting up a Dev Environment](https://www.youtube.com/results?search_query=developer+environment+setup)
- [VS Code Tips and Tricks](https://www.youtube.com/results?search_query=vscode+tips+and+tricks)

---

**Ready to get your development environment set up? Let's go! 🚀**
