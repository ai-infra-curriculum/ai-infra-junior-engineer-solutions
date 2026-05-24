# Troubleshooting — Dev Environment Setup

This file is the consolidated index of the issues juniors actually hit while doing Exercise 02. Each entry is **symptom → cause → fix**, in that order, with the actual commands to run.

If you don't see your issue here, search the exact error text on GitHub Issues for the tool (`pyenv`, `docker`, `wsl`, etc.) before posting in the curriculum's discussion channel.

---

## Index

1. [Homebrew not on PATH](#1-homebrew-not-on-path)
2. [`brew install` fails with SSL or proxy errors](#2-brew-install-fails-with-ssl-or-proxy-errors)
3. [`pyenv install` fails on macOS Sonoma/Sequoia](#3-pyenv-install-fails-on-macos-sonomasequoia)
4. [`pyenv install` fails on Ubuntu](#4-pyenv-install-fails-on-ubuntu)
5. [`python` still points at system Python after pyenv install](#5-python-still-points-at-system-python-after-pyenv-install)
6. [`pip install` fails inside a venv with permission errors](#6-pip-install-fails-inside-a-venv-with-permission-errors)
7. [`docker: permission denied while trying to connect to the Docker daemon socket`](#7-docker-permission-denied)
8. [Docker Desktop "starting…" forever / hangs on boot](#8-docker-desktop-stuck-starting)
9. [Docker on Linux: `Cannot connect to the Docker daemon`](#9-docker-on-linux-cannot-connect-to-daemon)
10. [Docker slow to mount Mac/Windows files into containers](#10-docker-slow-bind-mount)
11. [Conflicting Python versions: brew python vs pyenv python](#11-conflicting-python-versions)
12. [VS Code uses the wrong Python interpreter](#12-vs-code-uses-the-wrong-python-interpreter)
13. [VS Code extensions install but features don't work](#13-vs-code-extensions-not-loading)
14. [VS Code Remote-WSL: server fails to start](#14-vs-code-remote-wsl-server-fails)
15. [Git push asks for password every time](#15-git-push-asks-for-password)
16. [`ssh -T git@github.com` returns permission denied (publickey)](#16-github-publickey-denied)
17. [Line ending hell — CRLF vs LF](#17-line-endings)
18. [WSL2 won't install / virtualization not enabled](#18-wsl-virtualization)
19. [WSL2 uses too much RAM](#19-wsl-ram)
20. [`kubectl` works but `kubectx` / completion doesn't](#20-kubectl-completion)
21. [`aws configure` succeeds but every command says "Unable to locate credentials"](#21-aws-creds-not-loaded)
22. [Apple Silicon: `bad CPU type in executable` or `Bad arch` errors](#22-apple-silicon-arch)
23. [Apple Silicon: ML libs require Rosetta or fail to install](#23-apple-silicon-rosetta)
24. [Slow `pip install` / hangs on resolving dependencies](#24-pip-slow)
25. [Disk fills up unexpectedly](#25-disk-full)

---

## 1. Homebrew not on PATH

**Symptom**: After installing Homebrew, `brew --version` says `command not found`.

**Cause**: Brew's installer does not write to your shell rc on Apple Silicon. It only prints the lines you need to add.

**Fix** (Apple Silicon):

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Intel Mac: replace `/opt/homebrew/bin/brew` with `/usr/local/bin/brew`.

Reopen the terminal. `brew --version` should now work.

---

## 2. `brew install` fails with SSL or proxy errors

**Symptom**: `curl: (60) SSL certificate problem: unable to get local issuer certificate` or 403 from raw.githubusercontent.com.

**Cause**: Corporate VPN or proxy is intercepting TLS or blocking GitHub.

**Fix**:

```bash
# If you have a corporate cert bundle, point Homebrew at it:
export HOMEBREW_CURL_PATH=/usr/bin/curl
export SSL_CERT_FILE=/path/to/corp-bundle.pem

# Or temporarily disable VPN to bootstrap, then re-enable.
```

If your company maintains an internal Homebrew mirror, use `HOMEBREW_ARTIFACT_DOMAIN=https://mirror.example.com brew install ...`.

---

## 3. `pyenv install` fails on macOS Sonoma/Sequoia

**Symptom**: `BUILD FAILED (OS X 14.x using python-build ...)` with `ModuleNotFoundError: No module named '_ssl'` or `zlib not available`.

**Cause**: Pyenv compiles Python from source. Headers from Homebrew openssl, readline, zlib, etc. are not on the search paths.

**Fix**:

```bash
brew install openssl readline sqlite3 xz zlib tcl-tk

export LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix sqlite)/lib"
export CPPFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix readline)/include -I$(brew --prefix zlib)/include -I$(brew --prefix sqlite)/include"
export PKG_CONFIG_PATH="$(brew --prefix openssl)/lib/pkgconfig:$(brew --prefix readline)/lib/pkgconfig:$(brew --prefix zlib)/lib/pkgconfig:$(brew --prefix sqlite)/lib/pkgconfig"

pyenv install 3.11.9
```

If still failing, dump the build log: `cat /var/folders/.../python-build.*.log` (path is printed). The bottom usually names the missing dependency.

---

## 4. `pyenv install` fails on Ubuntu

**Symptom**: `ModuleNotFoundError: No module named '_ctypes'` or `_lzma` after `pyenv install`.

**Cause**: Missing apt build dependencies.

**Fix**:

```bash
sudo apt -y install make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncurses-dev xz-utils \
  tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

pyenv uninstall 3.11.9
pyenv install 3.11.9
```

---

## 5. `python` still points at system Python after pyenv install

**Symptom**: `python --version` shows 3.9.x even after `pyenv install 3.11.9 && pyenv global 3.11.9`.

**Cause**: Pyenv shims dir is not first on PATH, or `eval "$(pyenv init -)"` was not added to the shell rc.

**Fix**:

```bash
# ~/.zshrc or ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload: `exec $SHELL`. Then `which python` should be `~/.pyenv/shims/python`.

---

## 6. `pip install` fails inside a venv with permission errors

**Symptom**: `Could not install packages due to an OSError: [Errno 13] Permission denied: '/usr/lib/python3/dist-packages/...'`.

**Cause**: You are not actually in your venv. `pip` is trying to install into the system site-packages.

**Fix**:

```bash
which python      # should show /path/to/.venv/bin/python
which pip         # same prefix

source .venv/bin/activate    # re-activate
pip install <package>
```

If it still tries the system path, your venv was created with the wrong Python. Recreate it:

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
```

---

## 7. Docker: permission denied

**Symptom**: `docker ps` returns `permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock`.

**Cause**: Your user is not in the `docker` group.

**Fix (Linux)**:

```bash
sudo usermod -aG docker $USER
newgrp docker         # apply in current shell

# Fully log out + back in for permanent effect.
docker ps
```

**Fix (macOS)**: Should not happen with Docker Desktop or Colima — they expose the socket via a user-owned channel. If it does, Docker Desktop did not finish setup. Quit it and restart.

Never `sudo docker` as a workaround — you'll create root-owned files inside your repo.

---

## 8. Docker Desktop stuck "starting…"

**Symptom**: Docker Desktop tray icon spins forever. CLI says `Cannot connect to the Docker daemon`.

**Cause**: Usually a corrupt `~/Library/Group Containers/group.com.docker` (macOS) or VHDX (Windows) from an unclean shutdown.

**Fix (macOS)**:

```bash
osascript -e 'quit app "Docker"'
rm -rf ~/Library/Group\ Containers/group.com.docker/Data/vms
open -a Docker
```

If still stuck, "Reset to factory defaults" from the bug icon in Docker Desktop settings (you'll lose all images/containers).

**Fix (Windows)**: PowerShell:

```powershell
wsl --shutdown
Restart-Service com.docker.service -Force
```

If still stuck, uninstall Docker Desktop, reboot, reinstall.

---

## 9. Docker on Linux: Cannot connect to daemon

**Symptom**: `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?`

**Cause**: dockerd is not running.

**Fix**:

```bash
sudo systemctl status docker
sudo systemctl start docker
sudo systemctl enable docker     # auto-start on boot

# If status shows it failed:
sudo journalctl -u docker --no-pager -n 100
```

Common root cause: `/etc/docker/daemon.json` has a syntax error. Validate with `jq . /etc/docker/daemon.json`.

---

## 10. Docker slow bind mount

**Symptom**: Volume mounts (`-v $(pwd):/app`) feel painfully slow inside containers — `pip install` takes 10x longer than on the host.

**Cause** (macOS / Windows): Bind mounts cross the VM filesystem boundary. Each file syscall is translated.

**Fix**:

- macOS Docker Desktop: enable VirtioFS in Settings -> General. Restart.
- macOS Colima: it uses 9p by default; rerun `colima start --mount-type virtiofs ...` (requires `--vm-type vz`).
- Windows: ensure your project is inside the WSL filesystem (`~`), not `/mnt/c`.

For frequent small-file workloads (Node modules, Python venvs), keep them inside named volumes instead of bind mounts:

```yaml
volumes:
  - .:/app
  - node_modules:/app/node_modules
```

---

## 11. Conflicting Python versions

**Symptom**: `python3` is one version, `python3.11` is another, `pyenv version` says a third.

**Cause**: Brew installed a `python@3.x` formula in parallel with pyenv. Or apt's `python3` shadows pyenv's shim because `/usr/bin` is before `~/.pyenv/shims` in PATH.

**Fix**:

```bash
# Audit
which -a python python3 python3.11

# Remove brew's python (only if you don't intentionally need it)
brew uninstall python@3.11
brew autoremove

# Confirm pyenv shims come first
echo $PATH | tr ':' '\n' | head -5
# Top entry should include .pyenv/shims
```

If `/usr/bin` is winning, your `eval "$(pyenv init -)"` is missing or runs too late. Move it above any Oh My Zsh sourcing.

---

## 12. VS Code uses the wrong Python interpreter

**Symptom**: VS Code shows squigglies under imports that are installed, or runs the wrong Python when you click Run.

**Cause**: VS Code's Python extension picked up a different interpreter than your venv.

**Fix**:

1. `Cmd+Shift+P` -> "Python: Select Interpreter"
2. Choose the one inside your project's `.venv/bin/python` (path starts with your project folder).
3. Or pin in `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

If your interpreter doesn't appear in the list, click "Enter interpreter path..." and browse to it manually. Then restart VS Code so Pylance reindexes.

---

## 13. VS Code extensions not loading

**Symptom**: You installed `ms-python.python` but there's no syntax highlighting / IntelliSense.

**Cause** (most common): You installed the extension into the Windows VS Code while your project is opened in WSL — extensions must be installed into the WSL server.

**Fix**:

1. In VS Code, open the Extensions panel.
2. Find the extension. It will show a button "Install in WSL: Ubuntu".
3. Click it.

For Remote-Containers, the same applies — extensions install separately into each remote.

You can mark extensions to install in all remotes automatically by adding to user settings:

```json
{
  "remote.extensionKind": {
    "ms-python.python": ["workspace"]
  }
}
```

---

## 14. VS Code Remote-WSL: server fails to start

**Symptom**: VS Code shows "Starting VS Code Server" then fails with "Could not establish connection".

**Cause**: Antivirus blocking the server binary, or a stale server install.

**Fix**:

In WSL Ubuntu:

```bash
rm -rf ~/.vscode-server
```

Reopen VS Code from Windows; it will reinstall the server. If that doesn't work:

- Disable Windows Defender real-time protection temporarily and retry.
- Run `wsl --shutdown` from PowerShell, reopen Ubuntu.
- Confirm Ubuntu has internet: `curl -I https://update.code.visualstudio.com/`.

---

## 15. Git push asks for password every time

**Symptom**: Every `git push` over HTTPS prompts for a password.

**Cause**: No credential helper configured. GitHub disallows password auth since 2021 anyway — you need a Personal Access Token or SSH.

**Fix (recommended)**: switch to SSH.

```bash
git remote -v
# origin  https://github.com/user/repo.git (push)
git remote set-url origin git@github.com:user/repo.git
```

Then set up SSH key (see [setup-macos.md Section 8](./setup-macos.md#8-git-configuration) or the matching Linux/Windows section).

**Fix (HTTPS + PAT)**:

```bash
# macOS — use Keychain
git config --global credential.helper osxkeychain

# Linux
git config --global credential.helper "cache --timeout=86400"
# Or install Git Credential Manager:
# https://github.com/git-ecosystem/git-credential-manager
```

Generate a PAT at https://github.com/settings/tokens (classic, with `repo` scope). On first push, enter the PAT as the password — the helper will cache it.

---

## 16. GitHub publickey denied

**Symptom**: `ssh -T git@github.com` returns `Permission denied (publickey).`

**Causes & fixes**, in order of likelihood:

1. **Key not added to GitHub.** `cat ~/.ssh/id_ed25519.pub` and paste at https://github.com/settings/keys.
2. **Wrong identity sent.** Force the right key:
   ```bash
   ssh -T -i ~/.ssh/id_ed25519 git@github.com
   ```
   If that works, your SSH agent has a different default. Add to `~/.ssh/config`:
   ```
   Host github.com
     User git
     IdentityFile ~/.ssh/id_ed25519
     IdentitiesOnly yes
   ```
3. **Agent not running**: `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519`.
4. **Permissions wrong** (Linux): `chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519 && chmod 644 ~/.ssh/id_ed25519.pub`.

---

## 17. Line endings

**Symptom**: After cloning a repo on Windows, every file shows as modified in `git diff`. CI complains "shell script not executable" or "syntax error near unexpected token `$'\r''".

**Cause**: Git normalized LF to CRLF on checkout, or your editor saved CRLF into a `.sh` file.

**Fix** (all platforms):

```bash
git config --global core.autocrlf input       # Linux/macOS
git config --global core.autocrlf true        # Windows native (NOT WSL)
# Inside WSL, treat it as Linux:
git config --global core.autocrlf input
```

Add a `.gitattributes` to the repo:

```
* text=auto eol=lf
*.sh text eol=lf
*.ps1 text eol=crlf
*.bat text eol=crlf
*.png binary
*.jpg binary
```

Re-normalize an existing repo:

```bash
git add --renormalize .
git commit -m "Normalize line endings"
```

---

## 18. WSL virtualization

**Symptom**: `wsl --install` returns "Error: 0x80370102 The virtual machine could not be started because a required feature is not installed."

**Cause**: VT-x / SVM (CPU virtualization) is disabled in BIOS, or Hyper-V is incompatible with another hypervisor (VirtualBox older than 6.0).

**Fix**:

1. Reboot into BIOS/UEFI. Enable "Intel VT-x" or "AMD SVM" or "Virtualization Technology".
2. In Windows:
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```
3. Reboot.
4. If you have an old VirtualBox installed, upgrade to 6.0+ or uninstall it.

---

## 19. WSL RAM

**Symptom**: Windows feels slow. Task Manager shows `Vmmem` consuming 20+ GB.

**Cause**: WSL2 by default allows itself up to 50% of host RAM. Once allocated, it does not release back to Windows.

**Fix**: Create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

Then in PowerShell: `wsl --shutdown`. Reopen Ubuntu. Memory cap now applies.

To release memory mid-session without shutting down:

```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

---

## 20. kubectl completion

**Symptom**: `kubectl get po<tab>` doesn't autocomplete to `pods`.

**Fix**:

```bash
# zsh
echo 'source <(kubectl completion zsh)' >> ~/.zshrc
echo 'compdef __start_kubectl k' >> ~/.zshrc    # if you alias k=kubectl

# bash
echo 'source <(kubectl completion bash)' >> ~/.bashrc

# Install kubectx for fast context/namespace switching
brew install kubectx          # macOS / Linuxbrew
sudo apt -y install kubectx   # Ubuntu 24.04
```

`exec $SHELL` to reload.

---

## 21. AWS creds not loaded

**Symptom**: You ran `aws configure` successfully, but every `aws s3 ls` says `Unable to locate credentials`.

**Causes & fixes**:

1. **AWS_PROFILE points elsewhere.** `echo $AWS_PROFILE` — if set, that's the profile being used. Either `unset AWS_PROFILE` or `aws configure --profile $AWS_PROFILE`.
2. **AWS_ACCESS_KEY_ID env var is empty but set.** Env vars beat config files. `unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN`.
3. **Wrong file location.** Confirm `cat ~/.aws/credentials` shows your keys.
4. **SSO session expired.** If you use `aws sso login`, the cache in `~/.aws/sso/cache` may be stale: `aws sso login --profile <p>` again.

---

## 22. Apple Silicon arch

**Symptom**: `zsh: bad CPU type in executable: some-binary`.

**Cause**: A binary built for Intel was downloaded on Apple Silicon.

**Fix**: Install Rosetta once:

```bash
softwareupdate --install-rosetta --agree-to-license
```

Then either re-download an arm64 build, or run the existing binary under Rosetta:

```bash
arch -x86_64 some-binary
```

For Homebrew formulas, prefer arm64 (`brew install`) over `arch -x86_64 brew install` unless a formula is genuinely Intel-only.

---

## 23. Apple Silicon Rosetta

**Symptom**: `pip install vllm` (or any heavy ML lib) fails with `Could not find a version that satisfies the requirement`.

**Cause**: No arm64 wheel exists yet. The library only ships Linux x86_64 or macOS Intel.

**Fix options**:

1. **Use Linux in Docker** (preferred for ML):
   ```bash
   docker run --rm -it --platform linux/amd64 python:3.11 bash
   pip install vllm
   ```
   On Colima you need `--vz-rosetta` enabled (see [setup-macos.md](./setup-macos.md#5-docker)).
2. **Use an arm64-compatible alternative**: many libs have arm64 wheels now (`torch`, `transformers`, `langchain`). Check PyPI's "Download files" tab.
3. **Run pyenv under Rosetta** for a separate x86_64 Python:
   ```bash
   arch -x86_64 zsh
   pyenv install 3.11.9
   ```
   This is heavy — only do it if Docker won't work.

---

## 24. pip slow

**Symptom**: `pip install -r requirements.txt` takes 5+ minutes resolving.

**Causes & fixes**:

1. **Old pip.** `pip install --upgrade pip` — the new resolver is dramatically faster.
2. **Conflicting constraints.** Use `pip install --use-deprecated=legacy-resolver` only to confirm the issue is resolver work, then fix the conflict properly.
3. **Network slow.** Use a mirror:
   ```bash
   pip install --index-url https://pypi.org/simple/ <pkg>
   ```
   Inside China use Tsinghua: `https://pypi.tuna.tsinghua.edu.cn/simple/`.
4. **Wheel building from source.** Watch for "Building wheel for X" — install the matching apt/brew dev package. Often missing: `libpq-dev` (psycopg), `libffi-dev`.

For sustained speed, switch to `uv`:

```bash
pip install uv
uv pip install -r requirements.txt   # 10-100x faster
```

---

## 25. Disk full

**Symptom**: `No space left on device`.

**Causes and fixes**:

```bash
# Find the biggest culprits
df -h
du -sh ~/* | sort -h | tail -10
du -sh ~/.cache/* | sort -h | tail -10

# Docker is usually the worst offender
docker system df
docker system prune -a --volumes      # removes ALL stopped containers, unused images, networks, volumes

# pyenv keeps every installed Python
ls ~/.pyenv/versions
pyenv uninstall 3.10.0

# pip cache
pip cache purge

# brew old versions
brew cleanup -s
brew autoremove

# macOS: shrink the Docker VM after pruning
# Docker Desktop -> Troubleshoot -> Reset to factory defaults (last resort)
```

On Windows, Docker's WSL VHDX does not shrink automatically:

```powershell
wsl --shutdown
# In an elevated PowerShell:
diskpart
> select vdisk file="C:\Users\<you>\AppData\Local\Docker\wsl\data\ext4.vhdx"
> attach vdisk readonly
> compact vdisk
> detach vdisk
> exit
```

---

## When to ask for help

If you've tried the relevant section above and the issue persists for more than 30 minutes:

1. Capture the exact command and full error: `command 2>&1 | tee error.log`.
2. Include OS / arch / version, tool version, and what you've already tried.
3. Post in the curriculum's discussion channel with that bundle.

Do **not** post screenshots of terminal text. Paste the text directly so others can grep it.
