# AI Infrastructure Engineer - Zsh Configuration
# Copy this to ~/.zshrc

# Path to oh-my-zsh installation
export ZSH="$HOME/.oh-my-zsh"

# Theme
ZSH_THEME="robbyrussell"  # or "agnoster", "powerlevel10k/powerlevel10k"

# Plugins
plugins=(
    git
    docker
    kubectl
    python
    pip
    terraform
    aws
    zsh-autosuggestions
    zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

# ============================================================
# User Configuration
# ============================================================

# Python (pyenv)
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Homebrew (Apple Silicon)
if [[ $(uname -m) == "arm64" ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# ============================================================
# Aliases
# ============================================================

# General
alias zshconfig="code ~/.zshrc"
alias zshreload="source ~/.zshrc"

# Navigation
alias ..="cd .."
alias ...="cd ../.."
alias ....="cd ../../.."

# Better defaults
alias ls="exa --icons"  # if exa installed, otherwise use "ls -G"
alias ll="exa -l --icons"
alias la="exa -la --icons"
alias cat="bat"  # if bat installed

# Git
alias gs="git status"
alias ga="git add"
alias gc="git commit"
alias gp="git push"
alias gl="git log --oneline --graph --decorate"
alias gd="git diff"
alias gco="git checkout"
alias gb="git branch"

# Docker
alias d="docker"
alias dc="docker compose"
alias dps="docker ps"
alias di="docker images"
alias dcu="docker compose up -d"
alias dcd="docker compose down"
alias dcl="docker compose logs -f"

# Kubernetes
alias k="kubectl"
alias kgp="kubectl get pods"
alias kgs="kubectl get services"
alias kgd="kubectl get deployments"
alias kdp="kubectl describe pod"
alias kl="kubectl logs"
alias kex="kubectl exec -it"

# Python
alias py="python"
alias pip="pip3"
alias venv="python -m venv venv"
alias activate="source venv/bin/activate"

# Terraform
alias tf="terraform"
alias tfi="terraform init"
alias tfp="terraform plan"
alias tfa="terraform apply"
alias tfd="terraform destroy"

# AWS
alias awswhoami="aws sts get-caller-identity"

# ============================================================
# Functions
# ============================================================

# Create and activate virtual environment
mkvenv() {
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
}

# Create directory and cd into it
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# Find and kill process by port
killport() {
    lsof -ti:$1 | xargs kill -9
}

# Git commit with message
gcm() {
    git commit -m "$1"
}

# Docker cleanup
dclean() {
    docker system prune -af
    docker volume prune -f
}

# Show Docker container IPs
dip() {
    docker inspect -f '{{.Name}} - {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker ps -aq)
}

# ============================================================
# Environment Variables
# ============================================================

# Editor
export EDITOR="code --wait"

# History
export HISTSIZE=10000
export SAVEHIST=10000
export HISTFILE=~/.zsh_history

# Colors
export CLICOLOR=1
export LSCOLORS=ExGxBxDxCxEgEdxbxgxcxd

# ============================================================
# Custom Prompt (optional)
# ============================================================

# Show Git branch in prompt
autoload -Uz vcs_info
precmd_vcs_info() { vcs_info }
precmd_functions+=( precmd_vcs_info )
setopt prompt_subst
zstyle ':vcs_info:git:*' formats '%F{green}(%b)%f '
RPROMPT='${vcs_info_msg_0_}'

# ============================================================
# FZF (Fuzzy Finder)
# ============================================================

# FZF key bindings and fuzzy completion
if command -v fzf &> /dev/null; then
    [ -f ~/.fzf.zsh ] && source ~/.fzf.zsh
fi

# ============================================================
# Custom Settings
# ============================================================

# Add your custom settings below this line

