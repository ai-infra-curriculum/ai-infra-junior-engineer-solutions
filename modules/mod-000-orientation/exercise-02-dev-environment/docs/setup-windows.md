# Windows Development Environment Setup

Target audience: Windows 11 (build 22000+) or Windows 10 (build 19044+, Pro / Home / Enterprise). We will run all our development inside WSL2 + Ubuntu. The Windows side is only for Docker Desktop, VS Code, and Windows Terminal.

Estimated time: 2-4 hours including downloads.

If you are willing to dual-boot or reformat to Linux instead, follow [setup-linux.md](./setup-linux.md) and you'll have a smoother experience. If you must stay on Windows, this guide is the path.

---

## 0. Why WSL2 and not native Windows

Most AI infra tooling assumes Linux. Native Windows works for some of it, but you will hit recurring friction:

- Many Python ML libraries (vLLM, deepspeed, triton) only ship Linux wheels.
- Bash scripts in this curriculum will not run in PowerShell without rewriting.
- Container build behavior is different.

WSL2 runs a real Linux kernel inside Hyper-V. You get a Linux filesystem, package manager, and process model. Files on `\\wsl$\Ubuntu\...` are accessible from Windows; files on `C:\Users\you\...` are accessible from Linux via `/mnt/c/Users/you/...`.

The rule: **keep your code inside the WSL filesystem (`~`)**, not under `/mnt/c`. Cross-filesystem I/O is 10-50x slower.

---

## 1. Prerequisites check

Open PowerShell as Administrator and run:

```powershell
# Windows version
winver
# Need 11 (any), or 10 build 19044+

# Virtualization enabled in BIOS?
systeminfo | Select-String "Hyper-V"
# Should show "A hypervisor has been detected" OR all four hypervisor requirements as Yes.
```

If virtualization is not enabled, reboot into BIOS/UEFI and turn it on (Intel: VT-x; AMD: SVM). Without it, WSL2 cannot run.

---

## 2. Install WSL2 + Ubuntu

In an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

This single command:

1. Enables the WSL feature.
2. Enables the Virtual Machine Platform feature.
3. Downloads the WSL2 kernel.
4. Installs Ubuntu 22.04 from the Store.

Reboot when prompted. After reboot, an Ubuntu window opens automatically and asks for a new UNIX username and password. Pick something simple (e.g., `dev` / a strong password). **This is unrelated to your Windows username.**

Verify:

```powershell
wsl --list --verbose
# NAME        STATE       VERSION
# Ubuntu-22.04 Running    2
wsl --version
```

Confirm you are on WSL **2**, not 1. If you see VERSION 1, run:

```powershell
wsl --set-version Ubuntu-22.04 2
wsl --set-default-version 2
```

Set Ubuntu as the default distro:

```powershell
wsl --set-default Ubuntu-22.04
```

From now on, `wsl` with no arguments drops you into Ubuntu.

---

## 3. Windows Terminal

Windows Terminal is preinstalled on Windows 11 and available from the Store on Windows 10. Confirm:

```powershell
wt --version
```

Open Windows Terminal, click the dropdown arrow next to the `+` tab, choose **Settings**, then under **Startup** set "Default profile" to Ubuntu-22.04. Reopen Terminal — new tabs now open straight into WSL.

Recommended settings (in the JSON via "Open JSON file"):

```json
{
  "defaultProfile": "{your-ubuntu-guid}",
  "copyOnSelect": true,
  "copyFormatting": "none",
  "profiles": {
    "defaults": {
      "fontFace": "JetBrainsMono NF",
      "fontSize": 12,
      "useAcrylic": false,
      "scrollbarState": "hidden"
    }
  }
}
```

Install JetBrainsMono NF: download from https://www.nerdfonts.com/font-downloads and double-click the `.ttf` files to install for the current user.

---

## 4. Update Ubuntu inside WSL

Open Ubuntu and bring it current:

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install \
  build-essential curl wget git unzip \
  ca-certificates gnupg lsb-release \
  software-properties-common apt-transport-https \
  jq tree htop ripgrep fzf tmux zsh
```

From this point forward, **all the apt commands in [setup-linux.md](./setup-linux.md) apply.** Follow Sections 1 (shell), 2 (pyenv + Python), 4 (VS Code in WSL — skip, see below), 5 (git), 6 (cloud CLIs), 7 (kubectl/helm), 8 (productivity), and 9 (verification).

This guide only describes the Windows-specific bits. Refer back to setup-linux.md for the Linux work.

---

## 5. VS Code on Windows with Remote-WSL

Do **not** install VS Code inside WSL. Install it on Windows and use the Remote-WSL extension. This gives you a fast UI on Windows while running the language servers, debugger, and integrated terminal inside Linux.

1. Download VS Code: https://code.visualstudio.com/Download (User Installer, 64-bit)
2. During install, check:
   - "Add to PATH" (default)
   - "Register Code as an editor for supported file types"

After install, open PowerShell:

```powershell
code --version
```

Install the WSL bridge from PowerShell:

```powershell
code --install-extension ms-vscode-remote.remote-wsl
```

Now from Ubuntu (inside Terminal):

```bash
cd ~
mkdir -p ai-infra && cd ai-infra
code .
```

VS Code launches, downloads its server into your WSL distro the first time (15-60 seconds), and opens the folder. The bottom-left corner should show `WSL: Ubuntu-22.04`. If it does not, you are editing files via Windows and will hit performance issues.

Install the curated extensions from the WSL-attached VS Code window (they install into the WSL server, not Windows):

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
code --install-extension tamasfe.even-better-toml
```

Settings to add in the WSL settings.json:

```json
{
  "editor.formatOnSave": true,
  "python.defaultInterpreterPath": "~/.pyenv/shims/python",
  "[python]": { "editor.defaultFormatter": "ms-python.black-formatter" },
  "terminal.integrated.defaultProfile.linux": "zsh",
  "remote.WSL.fileWatcher.polling": false
}
```

---

## 6. Docker Desktop with WSL2 backend

Inside WSL you could install `docker-ce` directly (as in setup-linux.md Section 3), but on Windows the canonical choice is Docker Desktop with the WSL2 backend. It:

- Runs the Docker daemon in a lightweight VM, not inside Ubuntu.
- Exposes the daemon to all your WSL distros over the same socket.
- Gives you a GUI for resource limits, image pruning, and extensions.

Caveat: Docker Desktop requires a paid subscription for companies with 250+ employees or >$10M revenue. Free for personal, education, small business, and OSS.

### Install

1. Download from https://www.docker.com/products/docker-desktop/
2. Run the installer. Accept the default option **Use WSL 2 instead of Hyper-V (recommended)**.
3. After install completes, sign out and back in.

### Configure for WSL

Open Docker Desktop -> Settings -> Resources -> WSL Integration:

- Enable integration with default WSL distro.
- Enable integration with `Ubuntu-22.04`.

Apply & Restart. Now from Ubuntu:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

If `docker` is not found in Ubuntu, the integration toggle did not stick — reopen the WSL Integration screen and re-toggle.

### Resource limits

Edit `%UserProfile%\.wslconfig` (create if missing). Note this is a Windows-side file:

```ini
[wsl2]
memory=8GB
processors=4
swap=4GB
localhostForwarding=true
```

Then `wsl --shutdown` from PowerShell, reopen Ubuntu. Without this file, WSL can grow to use 50% of your RAM, which Docker pulls from.

---

## 7. Git on Windows + WSL

You want git configured in both places. The simplest pattern:

- Use git inside WSL for project work.
- Use Git Credential Manager (Windows) as the credential helper so SSH/HTTPS auth survives reboots and doesn't get stuck.

### Inside WSL

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.editor "code --wait"
git config --global core.autocrlf input
git config --global push.autoSetupRemote true
```

The `core.autocrlf=input` setting is critical on Windows-adjacent setups — it keeps line endings as LF in the repo, normalizing any CRLF you accidentally introduce.

Wire the Windows Git Credential Manager into WSL so HTTPS clones don't keep prompting:

```bash
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager.exe"
```

(Path may vary; verify with `ls "/mnt/c/Program Files/Git/mingw64/bin/" | grep credential`. If you don't have Git for Windows installed, install it from https://git-scm.com/download/win first.)

### SSH keys

Generate inside WSL (preferred), the agent runs inside the distro:

```bash
ssh-keygen -t ed25519 -C "you@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Paste to https://github.com/settings/keys, then `ssh -T git@github.com`.

---

## 8. Cloud CLIs (inside WSL)

Follow [setup-linux.md Section 6](./setup-linux.md). All `aws`, `gcloud`, `az` commands work identically.

For Azure specifically, if `az login` opens a browser via WSL but Edge cannot reach localhost, run `az login --use-device-code` instead.

---

## 9. Kubernetes (inside WSL)

Follow [setup-linux.md Section 7](./setup-linux.md). One Windows-specific gotcha: kubectl uses `~/.kube/config`. If you install Docker Desktop's Kubernetes feature, it writes to `C:\Users\you\.kube\config`, not your WSL home. Either:

- Disable Docker Desktop Kubernetes and use `kind`/`minikube` inside WSL.
- Or symlink: `ln -sf /mnt/c/Users/$WIN_USER/.kube/config ~/.kube/config`

---

## 10. Verification

Run this from inside Ubuntu (WSL):

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
echo "Verifying WSL dev environment..."
check git --version
check python --version
check pyenv --version
check docker --version
check docker info
check code --version
check aws --version
check kubectl version --client
exit $fail
```

If `docker info` fails: Docker Desktop is not running, or WSL integration is off.
If `code` fails: open VS Code on Windows once, then run `code .` from WSL.

---

## 11. Performance tips specific to WSL

- **Keep code in `~`, not `/mnt/c`.** `/mnt/c` mounts NTFS through a translation layer and is slow.
- **Use `wsl --shutdown` between heavy sessions** to free memory; restarting takes 1-2 seconds.
- **Disable Windows Defender real-time scanning** for your WSL VHDX file (`%LocalAppData%\Packages\CanonicalGroupLimited.Ubuntu*`) and your project directories. Defender will scan every Linux file event by default — measurable slowdown on `pip install`.
- **Use the WSL git, not Git for Windows**, for repos inside `~`. Mixing is fine but the WSL one is faster on Linux filesystems.

---

## 12. What we deliberately skipped

- **Cygwin / Git Bash as primary shell** — both predate WSL2 and are strictly worse for this workflow.
- **Powershell-based Python dev** — works but you lose 60% of the curriculum's bash scripts.
- **Chocolatey / winget for everything** — fine for Windows-side tooling, but Ubuntu inside WSL handles all the Linux-side software more reliably.
- **Docker Desktop's Kubernetes** — adds 2 GB RAM overhead for a single-node cluster that `kind` does better with 200 MB.

Continue with [STEP_BY_STEP.md](../STEP_BY_STEP.md) Section 4 once verification passes. If anything failed, see [troubleshooting.md](./troubleshooting.md) — there is a WSL-specific section.
