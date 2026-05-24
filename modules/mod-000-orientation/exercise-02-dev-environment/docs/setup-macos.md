# macOS Development Environment Setup

Target audience: macOS 13 (Ventura), 14 (Sonoma), or 15 (Sequoia) on Apple Silicon (M1/M2/M3/M4) or Intel.

Estimated time: 90-150 minutes if no surprises.

---

## 0. Before you start

Verify your macOS version and architecture:

```bash
sw_vers
uname -m         # arm64 on Apple Silicon, x86_64 on Intel
```

A few things this guide assumes:

- You can `sudo`.
- You have at least 30 GB free on `/`.
- You have a working internet connection that does not block Homebrew or GitHub.

If you are on Apple Silicon, Homebrew lives at `/opt/homebrew`. On Intel, it lives at `/usr/local`. The commands below pick the right path automatically through `brew shellenv`.

---

## 1. Apple Command Line Tools

Xcode CLT supplies `git`, `cc`, `make`, and the headers that pyenv and Homebrew need to compile packages.

```bash
xcode-select --install
```

A GUI prompt appears. Accept and let it finish (5-15 min). Then verify:

```bash
xcode-select -p
# /Library/Developer/CommandLineTools
clang --version
```

If you already have full Xcode, the line above may print the Xcode developer directory instead. That is fine.

---

## 2. Homebrew

Homebrew is the package manager we will use for everything we don't get from Python or Docker directly.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon, Homebrew prints two lines you must add to your shell rc (it does not append them itself):

```bash
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

On Intel:

```bash
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/usr/local/bin/brew shellenv)"
```

Verify:

```bash
brew --version       # Homebrew 4.x.x
brew doctor          # should print "Your system is ready to brew."
```

If `brew doctor` flags warnings, read them. Most are about stray formulas from old installs and can be ignored if everything below works.

Recommended baseline:

```bash
brew update
brew install git wget jq tree htop ripgrep fzf gh tmux
```

Do not yet install Docker via brew cask, see Section 5 for the Colima alternative.

---

## 3. Terminal

You can stay on Apple's Terminal.app, but iTerm2 or Warp are better for daily development.

Pick one:

```bash
# iTerm2 — battle-tested, scriptable, free
brew install --cask iterm2

# Warp — modern, command palette, AI features, free for individual use
brew install --cask warp
```

After installing, open the new terminal once and let it become the default by going to Settings -> General. From this point on, run commands inside iTerm2 or Warp, not Terminal.app, so your environment is consistent.

Recommended iTerm2 settings:

- Profiles -> Text -> Font: JetBrains Mono Nerd Font 14pt (install with `brew install --cask font-jetbrains-mono-nerd-font`)
- Profiles -> Terminal -> Scrollback Lines: 100000
- Profiles -> Keys -> Key Mappings -> Presets: Natural Text Editing

---

## 4. Shell — zsh + plugins

macOS already ships zsh as the default since Catalina. Verify:

```bash
echo $SHELL          # /bin/zsh
zsh --version        # 5.9 or higher
```

Install Oh My Zsh:

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

Then add three plugins that are genuinely useful (not the kitchen sink):

```bash
brew install zsh-autosuggestions zsh-syntax-highlighting

# enable powerlevel10k prompt (optional but recommended)
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
  ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

Edit `~/.zshrc`:

```zsh
ZSH_THEME="powerlevel10k/powerlevel10k"

plugins=(
  git
  docker
  kubectl
  python
  pip
  fzf
)

source $ZSH/oh-my-zsh.sh

source $(brew --prefix)/share/zsh-autosuggestions/zsh-autosuggestions.zsh
source $(brew --prefix)/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
```

Open a new tab. powerlevel10k will run a configuration wizard the first time. Pick lean style if unsure.

---

## 5. Docker — Docker Desktop vs Colima

Docker Desktop is the simplest path but it requires accepting Docker's commercial license if you work at a 250+ employee company. Colima is a fully open-source drop-in that uses Lima VM under the hood.

### Option A: Docker Desktop (default)

```bash
brew install --cask docker
```

Open `/Applications/Docker.app` once so the daemon starts and installs CLI tooling. Then:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

Recommended Docker Desktop settings:

- Resources -> Advanced: 4 CPUs, 8 GB RAM, 4 GB swap, 60 GB disk
- General: enable "Use Virtualization framework" and "VirtioFS" (faster file sync)
- Kubernetes: leave off until you actually need it (it adds memory pressure)

### Option B: Colima (recommended for engineers who want lighter, scriptable Docker)

```bash
brew install colima docker docker-compose
```

Configure and start:

```bash
colima start --cpu 4 --memory 8 --disk 60 --vm-type vz --vz-rosetta
```

Notes:

- `--vm-type vz` uses Apple's Virtualization.framework instead of QEMU. Faster, lower idle CPU.
- `--vz-rosetta` lets you run amd64 images on Apple Silicon transparently. Required for many ML images that don't yet ship arm64 layers.
- To autostart on login: `brew services start colima` (this is `colima start` with default settings — re-run `colima start` with your flags after to persist them, then `colima stop`/`colima start` will reuse).

Verify:

```bash
docker context ls    # colima should be the default
docker run --rm hello-world
```

To stop Colima: `colima stop`. To delete entirely: `colima delete`.

If you ever need to switch between Docker Desktop and Colima, do `docker context use desktop-linux` or `docker context use colima`.

---

## 6. Python with pyenv

Do not use system Python (`/usr/bin/python3`) for development. It is owned by Apple and breaks badly when you `pip install` into it. Use `pyenv`.

Install pyenv and Python build dependencies:

```bash
brew install pyenv pyenv-virtualenv openssl readline sqlite3 xz zlib tcl-tk
```

Add to `~/.zshrc` (above any prompt init):

```zsh
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Reload:

```bash
exec zsh
```

Install Python:

```bash
pyenv install 3.11.9
pyenv install 3.12.6
pyenv global 3.11.9

python --version       # Python 3.11.9
which python           # ~/.pyenv/shims/python
```

Per-project pinning:

```bash
cd myproject
pyenv local 3.12.6        # writes .python-version
```

Virtual environments via pyenv-virtualenv:

```bash
pyenv virtualenv 3.11.9 ai-infra
pyenv activate ai-infra
pip install --upgrade pip
```

Or use stdlib `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

For dependency-heavy projects, install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
poetry config virtualenvs.in-project true   # keep .venv inside the project
```

---

## 7. VS Code + extensions

```bash
brew install --cask visual-studio-code
```

After install, run `code --version` to confirm the `code` CLI works. If it does not, open VS Code, press `Cmd+Shift+P`, and run "Shell Command: Install 'code' command in PATH".

Install the curated extensions for this curriculum:

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
code --install-extension streetsidesoftware.code-spell-checker
```

Recommended `~/Library/Application Support/Code/User/settings.json` additions:

```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit",
    "source.fixAll": "explicit"
  },
  "python.defaultInterpreterPath": "~/.pyenv/shims/python",
  "python.terminal.activateEnvironment": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "files.exclude": {
    "**/.pytest_cache": true,
    "**/__pycache__": true,
    "**/.mypy_cache": true,
    "**/.ruff_cache": true
  },
  "terminal.integrated.defaultProfile.osx": "zsh"
}
```

---

## 8. Git configuration

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

Set up SSH for GitHub:

```bash
ssh-keygen -t ed25519 -C "you@example.com"
# Press enter for default location. Add a passphrase you'll remember.

eval "$(ssh-agent -s)"

cat > ~/.ssh/config <<'EOF'
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
EOF

ssh-add --apple-use-keychain ~/.ssh/id_ed25519
pbcopy < ~/.ssh/id_ed25519.pub
```

Paste the key into https://github.com/settings/keys -> New SSH key, then verify:

```bash
ssh -T git@github.com
# Hi <username>! You've successfully authenticated...
```

Optional: install `gh` CLI for easier PR work:

```bash
brew install gh
gh auth login
```

---

## 9. Cloud CLIs

You will not need all three for every exercise. Install what your track requires.

```bash
# AWS
brew install awscli
aws --version
aws configure          # only if you have credentials yet

# Google Cloud
brew install --cask google-cloud-sdk
gcloud version
# gcloud init           # only if you have a GCP project yet

# Azure
brew install azure-cli
az version
# az login              # only if you have an Azure tenant yet
```

Optional but useful:

```bash
brew install kubectl helm kubectx k9s
```

`kubectx` and `k9s` are not strictly required by the curriculum, but you will use them constantly once you have a cluster.

---

## 10. Verification

Save the following as `~/scripts/verify-dev-env.sh` and run it:

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
check brew --version
check git --version
check python --version
check pyenv --version
check docker --version
check docker info
check code --version
check aws --version
check gh --version
check kubectl version --client
exit $fail
```

```bash
chmod +x ~/scripts/verify-dev-env.sh
~/scripts/verify-dev-env.sh
```

Zero failures means you are ready. Any failure: cross-reference the section above or jump to [troubleshooting.md](./troubleshooting.md).

---

## 11. Optional but recommended

- `brew install --cask rectangle` — window tiling
- `brew install --cask raycast` — spotlight replacement with clipboard history
- `brew install bat eza zoxide` — better cat/ls/cd
- `brew install --cask cleanmymac` only if you need it; it is not free
- `brew install --cask stats` — menu bar system monitor

Add to `~/.zshrc` if you installed the modern CLI replacements:

```zsh
alias cat=bat
alias ls=eza
alias ll='eza -l --git'
eval "$(zoxide init zsh)"
```

---

## What changed vs the README quick start

- We default Python install via `pyenv`, not Homebrew formula `python@3.11`. Brew's Python upgrades unpredictably and breaks venvs.
- We list Colima as a peer to Docker Desktop, not a footnote.
- We use SSH for GitHub by default, not HTTPS + token.
- We pin specific extension IDs so you can re-run the install on a fresh machine.

Next: continue with [STEP_BY_STEP.md](../STEP_BY_STEP.md) Section 4 (project setup) once verification passes.
