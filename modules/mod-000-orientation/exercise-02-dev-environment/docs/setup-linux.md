# Linux Development Environment Setup

Target audience: Ubuntu 22.04 LTS (Jammy) or 24.04 LTS (Noble) on x86_64 or arm64. Most steps translate to Debian 12, Pop!_OS, and Linux Mint with the same `apt` commands. If you are on Fedora/RHEL/Arch, the structure is the same but you need to translate package names — see the notes at the end.

Estimated time: 90-180 minutes if no surprises.

---

## 0. Before you start

```bash
lsb_release -a
uname -m              # x86_64 or aarch64
df -h /               # need at least 30 GB free
```

Update everything first. Do not skip this — apt is reasonable about not breaking things but only when the index is fresh:

```bash
sudo apt update
sudo apt -y upgrade
sudo apt -y install \
  build-essential curl wget git unzip \
  ca-certificates gnupg lsb-release \
  software-properties-common \
  apt-transport-https \
  jq tree htop ripgrep fzf tmux
```

`build-essential` is what pyenv will need to compile Python from source.

---

## 1. Shell

Ubuntu ships bash. zsh is optional but recommended because Oh My Zsh and the plugin ecosystem are genuinely useful for daily development.

If you want to stay on bash, skip to Section 2. The rest of the guide works either way.

### zsh + Oh My Zsh

```bash
sudo apt -y install zsh
chsh -s "$(which zsh)"          # changes your default shell on next login
```

Log out and back in (or `exec zsh` for the current terminal), then install Oh My Zsh:

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

Recommended plugins (clone into the custom dir, not the core repo):

```bash
ZSH_CUSTOM=${ZSH_CUSTOM:-~/.oh-my-zsh/custom}
git clone https://github.com/zsh-users/zsh-autosuggestions      $ZSH_CUSTOM/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting  $ZSH_CUSTOM/plugins/zsh-syntax-highlighting
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git $ZSH_CUSTOM/themes/powerlevel10k
```

Edit `~/.zshrc`:

```zsh
ZSH_THEME="powerlevel10k/powerlevel10k"
plugins=(git docker kubectl python pip fzf zsh-autosuggestions zsh-syntax-highlighting)
source $ZSH/oh-my-zsh.sh
```

Reload and walk through the powerlevel10k wizard once: `exec zsh`.

### Staying on bash

If you stay on bash, install `bash-completion` and skip the rest of this section:

```bash
sudo apt -y install bash-completion
```

Whenever this guide says "add to `~/.zshrc`", add to `~/.bashrc` instead.

---

## 2. Python with pyenv

Do not use `apt install python3` for development environments. It works, but it is owned by the distribution and you cannot install arbitrary versions. Use `pyenv`.

Install the build dependencies that pyenv will need to compile Python:

```bash
sudo apt -y install \
  make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev
```

Install pyenv:

```bash
curl https://pyenv.run | bash
```

Add to your shell rc (`~/.zshrc` or `~/.bashrc`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Reload and install Python:

```bash
exec $SHELL
pyenv install 3.11.9
pyenv install 3.12.6
pyenv global 3.11.9

python --version          # Python 3.11.9
which python              # ~/.pyenv/shims/python
```

Compilation takes 3-6 minutes. If it fails, the error message usually tells you exactly which dev package is missing.

### Virtual environments

Stdlib:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Or pyenv-virtualenv (lets you `pyenv local <envname>` and have it auto-activate):

```bash
pyenv virtualenv 3.11.9 ai-infra
pyenv activate ai-infra
```

Or Poetry for projects with serious dependency graphs:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
poetry config virtualenvs.in-project true
```

---

## 3. Docker Engine (not Docker Desktop)

Docker Desktop for Linux exists but is heavier than necessary. Most Linux engineers run Docker Engine directly — it is faster, integrates with systemd, and does not require a VM.

Remove any old or distro-shipped Docker bits first:

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt remove -y "$pkg" || true
done
```

Add Docker's official apt repo:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt update
sudo apt -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Add yourself to the `docker` group so you can run docker without sudo:

```bash
sudo usermod -aG docker $USER
newgrp docker          # apply group in current shell, OR fully log out + back in
```

Verify:

```bash
docker --version
docker compose version
docker run --rm hello-world
docker info | grep -i 'storage driver'    # expect overlay2
```

Recommended `/etc/docker/daemon.json` (create with `sudo`):

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "default-address-pools": [
    { "base": "172.30.0.0/16", "size": 24 }
  ],
  "live-restore": true
}
```

```bash
sudo systemctl restart docker
```

### GPU passthrough (optional)

If you have an NVIDIA GPU and want containers to see it:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt -y install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## 4. VS Code

```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /tmp/packages.microsoft.gpg
sudo install -D -o root -g root -m 644 /tmp/packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" \
  | sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null
sudo apt update
sudo apt -y install code
```

Verify:

```bash
code --version
```

Install the curated extension set:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
code --install-extension redhat.vscode-yaml
code --install-extension hashicorp.terraform
code --install-extension eamodio.gitlens
code --install-extension github.vscode-pull-request-github
code --install-extension ms-vscode-remote.remote-containers
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension tamasfe.even-better-toml
```

`~/.config/Code/User/settings.json` baseline:

```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit",
    "source.fixAll": "explicit"
  },
  "python.defaultInterpreterPath": "~/.pyenv/shims/python",
  "[python]": { "editor.defaultFormatter": "ms-python.black-formatter" },
  "terminal.integrated.defaultProfile.linux": "zsh"
}
```

---

## 5. Git configuration

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.editor "code --wait"
git config --global push.autoSetupRemote true
git config --global rerere.enabled true
```

SSH key for GitHub:

```bash
ssh-keygen -t ed25519 -C "you@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Paste the printed key into https://github.com/settings/keys -> New SSH key, then:

```bash
ssh -T git@github.com
```

Optional but recommended:

```bash
sudo apt -y install gh
gh auth login
```

---

## 6. Cloud CLIs

```bash
# AWS
curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o awscliv2.zip
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip
aws --version

# Google Cloud
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
sudo apt update && sudo apt -y install google-cloud-cli
gcloud version

# Azure
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az version
```

---

## 7. Kubernetes CLI tooling

```bash
# kubectl — install latest stable
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/$(dpkg --print-architecture)/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
kubectl version --client

# helm
curl https://baltocdn.com/helm/signing.asc | sudo gpg --dearmor -o /usr/share/keyrings/helm.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" \
  | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt update && sudo apt -y install helm

# k9s — TUI for k8s
curl -L https://github.com/derailed/k9s/releases/latest/download/k9s_Linux_amd64.tar.gz | sudo tar xz -C /usr/local/bin k9s
```

For local clusters:

```bash
# kind
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
sudo install kind /usr/local/bin/kind && rm kind

# minikube (alternative)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-$(dpkg --print-architecture).deb
sudo dpkg -i minikube-linux-*.deb && rm minikube-linux-*.deb
```

---

## 8. Productivity tools

You already installed `jq`, `tree`, `htop`, `ripgrep`, `fzf`, `tmux` in Section 0. Add a few modern replacements:

```bash
sudo apt -y install bat zoxide
# 'bat' is installed as 'batcat' on Ubuntu — alias it
mkdir -p ~/.local/bin
ln -sf /usr/bin/batcat ~/.local/bin/bat

# eza (modern ls). Install via cargo or apt PPA:
sudo apt -y install eza  # 24.04+; on 22.04 use cargo install eza
```

Add to your rc:

```bash
export PATH="$HOME/.local/bin:$PATH"
alias cat=bat
alias ls=eza
alias ll='eza -l --git'
eval "$(zoxide init zsh)"  # or bash
```

---

## 9. Verification

```bash
#!/usr/bin/env bash
set -u
fail=0
check() {
  if "$@" >/dev/null 2>&1; then
    printf "  OK   %s\n" "$*"
  else
    printf "  FAIL %s\n" "$*"
    fail=$((fail + 1))
  fi
}
echo "Verifying dev environment..."
check git --version
check python --version
check pyenv --version
check docker --version
check docker info
check code --version
check aws --version
check kubectl version --client
check helm version
check gh --version
exit $fail
```

Save as `~/scripts/verify-dev-env.sh`, `chmod +x`, run. Zero failures means proceed.

---

## 10. Notes for other distros

### Fedora / RHEL / Rocky

- `dnf` instead of `apt`
- Docker repo: `https://download.docker.com/linux/centos/docker-ce.repo`
- Build deps for pyenv: `sudo dnf install gcc zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel tk-devel libffi-devel xz-devel`
- VS Code rpm repo: `https://packages.microsoft.com/yumrepos/vscode`

### Arch / Manjaro

- `pacman -S` instead of `apt`
- Docker is just `pacman -S docker docker-compose docker-buildx`
- Pyenv build deps: `base-devel openssl zlib xz tk`
- VS Code via AUR (`yay -S visual-studio-code-bin`)

---

## What we deliberately skipped

- `snap` Docker / `snap` code — both have known issues with file watchers and group permissions. Use the apt or `.deb` paths above.
- `docker-desktop` package on Linux — it works, but creates an extra VM you don't need.
- System `python3-pip` — pip is owned by pyenv-managed Pythons; system pip will eventually conflict.

Continue with [STEP_BY_STEP.md](../STEP_BY_STEP.md) for project setup, or [troubleshooting.md](./troubleshooting.md) if anything above failed.
