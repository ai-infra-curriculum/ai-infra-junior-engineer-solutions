# STEP_BY_STEP — Exercise 02 Dev Environment

This is the consolidated walkthrough for setting up your dev environment on **macOS**, **Linux (Ubuntu)**, or **Windows 11 (WSL2 + Ubuntu)**. It is opinionated — there is a right answer in this curriculum and this file tells you exactly what to do.

There are verification gates after each phase. Do not move past a gate that fails. If something breaks, jump to [docs/troubleshooting.md](docs/troubleshooting.md), fix the issue, then re-run the verification command.

Total time budget: 3-4 hours for the full path, less if you've done some of this before.

---

## Conventions used in this file

- Commands prefixed `(macOS)`, `(Linux)`, `(Windows-WSL)` mean only that OS. Unprefixed commands work on all three.
- `>>` next to a line means "expected output".
- "Verification gate" sections are sanity checks. Run them. Don't skip them.

---

## Phase 0 — Pick your OS path

Choose one and follow the matching docs file. Come back here after each phase verification.

| OS | Detailed guide |
|---|---|
| macOS 13+ | [docs/setup-macos.md](docs/setup-macos.md) |
| Ubuntu 22.04 / 24.04 | [docs/setup-linux.md](docs/setup-linux.md) |
| Windows 11 (WSL2) | [docs/setup-windows.md](docs/setup-windows.md) |

If you don't yet have a preference: WSL2 is the slowest path; native Linux is the fastest; macOS is the most polished. The curriculum supports all three.

---

## Phase 1 — Base OS package manager and shell

### 1.1 Package manager

- **(macOS)** Install Homebrew. See setup-macos.md Section 2.
- **(Linux)** `sudo apt update && sudo apt -y upgrade && sudo apt -y install build-essential curl wget git unzip ca-certificates gnupg software-properties-common jq tree htop ripgrep fzf tmux zsh`
- **(Windows-WSL)** Install WSL2 + Ubuntu 22.04: `wsl --install -d Ubuntu-22.04` in elevated PowerShell, reboot, set username/password. From here on, all commands run inside the Ubuntu terminal.

### 1.2 Shell

We standardize on **zsh + Oh My Zsh + powerlevel10k** because the prompt makes context (git branch, kube context, venv) visible — and that prevents whole classes of "what cluster am I on?" mistakes later.

```bash
# Linux only — set zsh as default (macOS is already zsh)
chsh -s "$(which zsh)"

# All OSes
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

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

Reload: `exec zsh`. Walk through the powerlevel10k wizard the first time it appears.

### Verification gate 1

```bash
echo $SHELL                              # >> /bin/zsh or /usr/bin/zsh
zsh --version                            # >> zsh 5.9 or later
brew --version 2>/dev/null || apt --version 2>/dev/null   # one or the other
```

All three should print sane values. If `brew` says "command not found" on macOS, fix it before moving on — see [troubleshooting.md #1](docs/troubleshooting.md#1-homebrew-not-on-path).

---

## Phase 2 — Python with pyenv

We use pyenv because it is the only Python version manager that:

1. Compiles from source so you get exactly the version you ask for.
2. Lets you set per-project versions (`.python-version`).
3. Doesn't get owned by the OS upgrade cycle.

### 2.1 Install build deps

- **(macOS)** `brew install pyenv pyenv-virtualenv openssl readline sqlite3 xz zlib tcl-tk`
- **(Linux / Windows-WSL)**
  ```bash
  sudo apt -y install make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
  curl https://pyenv.run | bash
  ```

### 2.2 Wire pyenv into shell

Add to `~/.zshrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

`exec zsh`.

### 2.3 Install Python

```bash
pyenv install 3.11.9
pyenv install 3.12.6
pyenv global 3.11.9
```

3.11 is the default because every project in this curriculum has a 3.11 lockfile. 3.12 is there because some projects opt in.

### 2.4 First venv

```bash
mkdir -p ~/code/ai-infra && cd ~/code/ai-infra
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
```

### Verification gate 2

```bash
which python                              # >> ~/.venv/bin/python  (active venv)
python --version                          # >> Python 3.11.9
pip --version                             # >> recent pip with python 3.11
python -c "import ssl, sqlite3, lzma, ctypes; print('ok')"  # >> ok
deactivate
```

The `import ssl, sqlite3, lzma, ctypes` check matters — these are the four modules that silently fail if pyenv couldn't find the dev libraries. If it errors, see [troubleshooting.md #3](docs/troubleshooting.md#3-pyenv-install-fails-on-macos-sonomasequoia) or [#4](docs/troubleshooting.md#4-pyenv-install-fails-on-ubuntu).

---

## Phase 3 — Docker

### 3.1 Install

- **(macOS)** Either:
  - Docker Desktop: `brew install --cask docker`, then launch `/Applications/Docker.app` once.
  - Colima (recommended for engineers): `brew install colima docker docker-compose && colima start --cpu 4 --memory 8 --disk 60 --vm-type vz --vz-rosetta`
- **(Linux)** Docker Engine via Docker's apt repo. See [setup-linux.md Section 3](docs/setup-linux.md#3-docker-engine-not-docker-desktop). Don't forget `sudo usermod -aG docker $USER && newgrp docker`.
- **(Windows-WSL)** Docker Desktop with WSL2 backend. In Docker Desktop Settings -> Resources -> WSL Integration, enable for Ubuntu-22.04.

### 3.2 Configure resources

You need at least 4 CPUs and 8 GB RAM available to Docker for the later exercises (Ray clusters, vLLM, multi-container compose stacks).

- Docker Desktop: Settings -> Resources -> Advanced.
- Colima: re-run `colima start --cpu 4 --memory 8 ...`.
- Linux native: no limits — Docker uses the host directly.
- Windows-WSL: edit `%UserProfile%\.wslconfig` then `wsl --shutdown`.

### Verification gate 3

```bash
docker --version                          # >> Docker version 24.x or later
docker compose version                    # >> Docker Compose version v2.x
docker info | grep -E 'CPUs|Memory'       # >> CPUs >= 4, Memory >= 7.5GiB
docker run --rm hello-world               # >> "Hello from Docker!"
docker run --rm -it alpine sh -c 'apk add --no-cache curl && curl -sI https://github.com | head -1'
                                          # >> HTTP/2 200
```

If `docker info` shows fewer CPUs or less memory than expected, your container runtime is throttled — fix Phase 3.2 before moving on. If `permission denied`, see [troubleshooting.md #7](docs/troubleshooting.md#7-docker-permission-denied).

---

## Phase 4 — Git and SSH for GitHub

### 4.1 Configure git

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.editor "code --wait"
git config --global core.autocrlf input
git config --global push.autoSetupRemote true
git config --global rerere.enabled true
```

### 4.2 SSH key

```bash
ssh-keygen -t ed25519 -C "you@example.com"
# Default location, set a passphrase you'll remember.

eval "$(ssh-agent -s)"

# macOS only — let the agent persist to Keychain
if [[ "$(uname)" == "Darwin" ]]; then
  cat >> ~/.ssh/config <<'EOF'
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
EOF
  ssh-add --apple-use-keychain ~/.ssh/id_ed25519
else
  ssh-add ~/.ssh/id_ed25519
fi

# Print pubkey to paste into GitHub
cat ~/.ssh/id_ed25519.pub
```

Open https://github.com/settings/keys -> "New SSH key", paste, save.

### 4.3 GitHub CLI (optional but recommended)

```bash
# macOS: brew install gh
# Linux/WSL: sudo apt -y install gh   (after adding the gh apt repo)
gh auth login   # choose SSH, follow prompts
```

### Verification gate 4

```bash
ssh -T git@github.com
# >> Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.

git config --get user.email               # >> the email you set
git config --get core.autocrlf            # >> input
gh auth status 2>/dev/null || true        # if gh installed, should show 'Logged in'
```

If SSH says "Permission denied (publickey)", see [troubleshooting.md #16](docs/troubleshooting.md#16-github-publickey-denied).

---

## Phase 5 — VS Code

### 5.1 Install

- **(macOS)** `brew install --cask visual-studio-code`. Then `Cmd+Shift+P` -> "Shell Command: Install 'code' command in PATH".
- **(Linux)** Microsoft apt repo, then `sudo apt -y install code`. See [setup-linux.md Section 4](docs/setup-linux.md#4-vs-code).
- **(Windows-WSL)** Install VS Code on the **Windows side**, not inside WSL. Then `code --install-extension ms-vscode-remote.remote-wsl` from PowerShell.

### 5.2 Install extensions

Run this in the right context:
- macOS / Linux: from a normal terminal.
- Windows-WSL: from the Ubuntu terminal, after `code .` has opened a folder in WSL once (so the WSL server is installed).

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

### 5.3 Baseline settings.json

Open Settings (`Cmd+,` / `Ctrl+,`), click the JSON icon, paste:

```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit",
    "source.fixAll": "explicit"
  },
  "python.defaultInterpreterPath": "~/.pyenv/shims/python",
  "[python]": { "editor.defaultFormatter": "ms-python.black-formatter" },
  "files.exclude": {
    "**/.pytest_cache": true,
    "**/__pycache__": true,
    "**/.mypy_cache": true,
    "**/.ruff_cache": true
  },
  "terminal.integrated.defaultProfile.osx": "zsh",
  "terminal.integrated.defaultProfile.linux": "zsh"
}
```

### Verification gate 5

```bash
code --version                            # >> 1.85+ on three lines
code --list-extensions | wc -l            # >> at least 13
```

Open VS Code, create a `hello.py`, type `def foo():` and hit Enter — Pylance should show type hints. If not, [troubleshooting.md #12](docs/troubleshooting.md#12-vs-code-uses-the-wrong-python-interpreter).

---

## Phase 6 — Cloud CLIs and Kubernetes tooling

Install what your track needs. Most modules use AWS as the primary cloud, with optional GCP/Azure exercises.

### 6.1 AWS CLI

- **(macOS)** `brew install awscli`
- **(Linux / Windows-WSL)**
  ```bash
  curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o awscliv2.zip
  unzip -q awscliv2.zip && sudo ./aws/install && rm -rf aws awscliv2.zip
  ```
- Optional: `aws configure` only if you have AWS credentials. The curriculum gives you free-tier setup instructions in later exercises.

### 6.2 kubectl + helm + kind

```bash
# kubectl (Linux/WSL)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && rm kubectl

# kubectl (macOS)
brew install kubectl

# helm
# macOS: brew install helm
# Linux: see setup-linux.md Section 7

# kind for local clusters
# macOS: brew install kind
# Linux: curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64 && sudo install kind /usr/local/bin/kind
```

### 6.3 Optional: gcloud, az

Only install if you intend to do GCP or Azure exercises.

- **(macOS)** `brew install --cask google-cloud-sdk` / `brew install azure-cli`
- **(Linux / Windows-WSL)** Apt repos for both — see [setup-linux.md Section 6](docs/setup-linux.md#6-cloud-clis).

### Verification gate 6

```bash
aws --version                             # >> aws-cli/2.x
kubectl version --client                  # >> Client Version: v1.28+
helm version                              # >> v3.x
kind --version 2>/dev/null && echo "kind installed"
```

---

## Phase 7 — Productivity tools

Install the modern CLI replacements that will save you minutes every day:

```bash
# macOS
brew install bat eza zoxide uv

# Linux / WSL Ubuntu 22.04 (some packages via cargo or apt)
sudo apt -y install bat zoxide
mkdir -p ~/.local/bin && ln -sf /usr/bin/batcat ~/.local/bin/bat
# eza on 24.04: sudo apt -y install eza
# uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add to `~/.zshrc`:

```zsh
export PATH="$HOME/.local/bin:$PATH"
alias cat=bat
alias ls=eza
alias ll='eza -lah --git'
eval "$(zoxide init zsh)"
```

`exec zsh`.

---

## Phase 8 — Create a dotfiles repo

This is what makes Phase 1-7 reproducible on a new machine. Don't skip it.

```bash
mkdir -p ~/dotfiles && cd ~/dotfiles
git init

# Copy the configs you've curated
cp ~/.zshrc        .zshrc
cp ~/.gitconfig    .gitconfig 2>/dev/null || true
mkdir -p vscode
cp ~/Library/Application\ Support/Code/User/settings.json vscode/settings.json 2>/dev/null \
  || cp ~/.config/Code/User/settings.json vscode/settings.json 2>/dev/null || true

cat > install.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ln -sf "$SCRIPT_DIR/.zshrc"     ~/.zshrc
ln -sf "$SCRIPT_DIR/.gitconfig" ~/.gitconfig
if [[ "$(uname)" == "Darwin" ]]; then
  VSCODE_DIR=~/Library/Application\ Support/Code/User
else
  VSCODE_DIR=~/.config/Code/User
fi
mkdir -p "$VSCODE_DIR"
ln -sf "$SCRIPT_DIR/vscode/settings.json" "$VSCODE_DIR/settings.json"
echo "Dotfiles installed. Restart your shell."
EOF
chmod +x install.sh

cat > README.md <<'EOF'
# dotfiles

Personal configuration. Run `./install.sh` on a fresh machine.
EOF

git add -A
git commit -m "feat: initial dotfiles"
# Create the repo on GitHub via gh:
gh repo create dotfiles --private --source=. --remote=origin --push
```

### Verification gate 8

```bash
gh repo view dotfiles --json url --jq .url
# >> https://github.com/<you>/dotfiles
```

You should be able to wipe `~/dotfiles`, `git clone` it back, and run `./install.sh` cleanly.

---

## Phase 9 — Final environment validation

Save and run this script as `~/.local/bin/verify-dev-env`:

```bash
#!/usr/bin/env bash
set -u
fail=0
warn=0
hr() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf "  \033[32mOK\033[0m   %s\n" "$1"; }
bad()  { printf "  \033[31mFAIL\033[0m %s\n" "$1"; fail=$((fail+1)); }
maybe(){ printf "  \033[33mWARN\033[0m %s\n" "$1"; warn=$((warn+1)); }
check() {
  if "$@" >/dev/null 2>&1; then ok "$*"; else bad "$*"; fi
}

hr "Core"
check git --version
check zsh --version
check curl --version

hr "Python"
check python --version
check pyenv --version
check pip --version
python -c "import ssl, sqlite3, lzma, ctypes" 2>/dev/null && ok "stdlib (ssl,sqlite3,lzma,ctypes)" || bad "stdlib modules missing — see troubleshooting #3/#4"

hr "Docker"
check docker --version
check docker compose version
check docker info
check docker run --rm hello-world

hr "VS Code"
check code --version
n=$(code --list-extensions 2>/dev/null | wc -l)
[[ $n -ge 13 ]] && ok "VS Code extensions ($n installed)" || maybe "Only $n VS Code extensions — expected 13+"

hr "Git/SSH/GitHub"
check git config --get user.email
ssh -T git@github.com 2>&1 | grep -q "successfully authenticated" && ok "ssh github.com authenticated" || bad "ssh github.com NOT authenticated"

hr "Cloud + K8s"
check aws --version
check kubectl version --client
check helm version
command -v kind >/dev/null && ok "kind installed" || maybe "kind not installed (needed for K8s exercises)"
command -v gcloud >/dev/null && ok "gcloud installed" || maybe "gcloud not installed (optional)"
command -v az >/dev/null && ok "az installed" || maybe "az not installed (optional)"

hr "Productivity"
check bat --version
check eza --version
check rg --version
check jq --version
check fzf --version
check tmux -V

hr "Summary"
echo "  $fail failures, $warn warnings"
exit $fail
```

```bash
chmod +x ~/.local/bin/verify-dev-env
verify-dev-env
```

**Pass criteria**: 0 failures. Warnings are OK if you've intentionally skipped optional tooling.

---

## Phase 10 — Smoke test with a real workload

Final gate: prove you can clone, build, and run something end-to-end.

```bash
cd ~/code
git clone https://github.com/docker/awesome-compose.git
cd awesome-compose/fastapi
docker compose up -d
sleep 5
curl http://localhost:8000/        # >> {"Hello":"World"}
docker compose down
```

If that works on macOS/Linux/WSL with your dev tools, you are done. Move on to **Exercise 03: Version Control with Git**.

---

## Troubleshooting & next steps

- Stuck? [docs/troubleshooting.md](docs/troubleshooting.md) — 25 common issues, symptom → fix.
- Need a quick command reference for daily work? [docs/cheat-sheet.md](docs/cheat-sheet.md).
- Specific OS gotchas: [setup-macos.md](docs/setup-macos.md), [setup-linux.md](docs/setup-linux.md), [setup-windows.md](docs/setup-windows.md).

When you finish this exercise, commit a screenshot of `verify-dev-env` output to your `dotfiles` repo's README so you have a record of the day everything worked.
