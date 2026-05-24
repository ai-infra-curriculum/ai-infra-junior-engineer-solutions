# Dev Environment Cheat Sheet

Quick reference for the commands you'll actually use after Exercise 02. Keep this open in a tab.

---

## Shell

| Action | Command |
|---|---|
| Reload rc file | `exec $SHELL` |
| Print PATH (one per line) | `echo $PATH \| tr ':' '\n'` |
| Find a command's location | `which -a foo` |
| Time a command | `time command` |
| Run last command with sudo | `sudo !!` |
| Repeat last arg | `cmd !$` |
| Search history fuzzy | `Ctrl-R` (with fzf) |

---

## Git

```bash
# Setup
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global push.autoSetupRemote true

# Daily flow
git status
git add path/to/file        # stage
git add -p                  # stage interactively, hunk by hunk
git commit -m "feat: ..."
git push

# Branches
git switch main             # checkout existing
git switch -c feature/x     # checkout new
git branch -d feature/x     # delete merged
git branch -D feature/x     # delete force

# Sync with main
git fetch origin
git rebase origin/main      # preferred for personal branches
git merge origin/main       # preferred for shared branches

# Inspect
git log --oneline -20
git log --graph --oneline --all
git diff                    # unstaged
git diff --staged           # staged
git diff main...HEAD        # all changes since branching from main
git show <sha>

# Undo
git restore path/to/file              # discard unstaged
git restore --staged path/to/file     # unstage
git commit --amend                    # fix last commit message
git reset --soft HEAD~1               # undo last commit, keep changes staged
git reset --mixed HEAD~1              # undo + unstage (default)
git reset --hard HEAD~1               # NUKE last commit AND changes

# Stash
git stash                   # save uncommitted work
git stash pop               # restore + remove from stash
git stash list

# Remotes
git remote -v
git remote add origin git@github.com:user/repo.git
git remote set-url origin git@github.com:user/repo.git
```

### GitHub CLI

```bash
gh auth login
gh repo clone user/repo
gh pr create --fill          # PR from current branch
gh pr list
gh pr view <num> --web
gh pr checkout <num>
gh pr checks
gh issue create
gh run watch                 # watch latest CI run
```

---

## Python (pyenv + venv + pip)

```bash
# pyenv
pyenv install --list | grep -E '^\s*3\.(11|12|13)'   # available versions
pyenv install 3.11.9
pyenv global 3.11.9                # default Python for shells
pyenv local 3.12.6                 # writes .python-version to cwd
pyenv versions
pyenv uninstall 3.10.0

# venv
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
deactivate

# pip
pip install <pkg>
pip install -r requirements.txt
pip install -e .                   # editable, current package
pip install --upgrade pip
pip freeze > requirements.txt
pip list --outdated
pip show <pkg>
pip cache purge

# uv (10-100x faster than pip)
uv venv
uv pip install -r requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# poetry
poetry init
poetry add fastapi uvicorn
poetry add --group dev pytest ruff
poetry install
poetry shell
poetry run pytest
poetry lock --no-update
```

---

## Docker

```bash
# Daily
docker ps                      # running containers
docker ps -a                   # all, incl. stopped
docker images
docker pull image:tag
docker run --rm -it image bash
docker run -d --name foo -p 8080:80 image
docker logs -f foo
docker exec -it foo sh
docker stop foo
docker rm foo
docker rmi image:tag

# Build
docker build -t myapp:dev .
docker build -t myapp:dev -f Dockerfile.prod .
docker build --target builder -t myapp:builder .
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:multi --push .

# Compose
docker compose up -d
docker compose up --build
docker compose logs -f api
docker compose down
docker compose down -v         # also remove named volumes
docker compose ps
docker compose exec api bash

# Inspection
docker inspect foo
docker stats                   # live resource usage
docker network ls
docker volume ls
docker system df               # disk usage by docker

# Cleanup
docker system prune            # stopped containers + dangling images
docker system prune -a         # also unused images
docker system prune -a --volumes  # NUCLEAR — also volumes
docker image prune
docker volume prune
docker builder prune

# GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi
```

---

## Kubernetes (kubectl basics)

```bash
# Context / config
kubectl config get-contexts
kubectl config use-context my-cluster
kubectl config current-context
kubectl config set-context --current --namespace=my-ns

# Listing
kubectl get pods
kubectl get pods -A                       # all namespaces
kubectl get pods -o wide                  # incl. node, IP
kubectl get pods -w                       # watch
kubectl get pods -l app=foo               # by label
kubectl get all                           # everything in current ns
kubectl get nodes
kubectl get svc,deploy,po

# Describe / inspect
kubectl describe pod my-pod
kubectl get pod my-pod -o yaml
kubectl get pod my-pod -o jsonpath='{.status.podIP}'

# Logs
kubectl logs my-pod
kubectl logs -f my-pod                    # follow
kubectl logs -f my-pod -c sidecar         # specific container
kubectl logs --previous my-pod            # previous crash
kubectl logs -l app=foo --tail=100        # by label

# Exec
kubectl exec -it my-pod -- bash
kubectl exec my-pod -- env

# Port forward
kubectl port-forward svc/grafana 3000:3000

# Apply / create / delete
kubectl apply -f manifest.yaml
kubectl apply -k overlays/dev/            # kustomize
kubectl create -f manifest.yaml
kubectl delete -f manifest.yaml
kubectl delete pod my-pod --grace-period=0 --force

# Rollouts
kubectl rollout status deploy/api
kubectl rollout history deploy/api
kubectl rollout undo deploy/api
kubectl rollout restart deploy/api
kubectl scale deploy/api --replicas=3

# Debug
kubectl get events --sort-by=.lastTimestamp
kubectl top pods
kubectl top nodes
kubectl debug -it my-pod --image=busybox --target=app
```

### Helm

```bash
helm repo add <name> <url>
helm repo update
helm search repo <term>
helm install <release> <chart>
helm upgrade --install <release> <chart> -f values.yaml
helm uninstall <release>
helm list
helm status <release>
helm get values <release>
helm template <release> <chart> > rendered.yaml
helm rollback <release> <revision>
```

---

## Cloud CLIs

### AWS

```bash
aws configure                 # interactive setup
aws configure --profile work  # named profile
aws sts get-caller-identity   # who am I

aws s3 ls
aws s3 ls s3://bucket/
aws s3 cp file s3://bucket/path/
aws s3 sync ./dir s3://bucket/path/

aws ec2 describe-instances
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <acct>.dkr.ecr.us-west-2.amazonaws.com
aws eks update-kubeconfig --name my-cluster --region us-west-2

aws logs tail /aws/lambda/foo --follow
```

### gcloud

```bash
gcloud auth login
gcloud auth application-default login
gcloud config list
gcloud config set project my-project

gcloud projects list
gcloud compute instances list
gcloud container clusters get-credentials my-cluster --region us-central1

gcloud storage ls gs://bucket/
gcloud storage cp file gs://bucket/path/
```

### Azure

```bash
az login
az account list
az account set --subscription <id>

az aks get-credentials --resource-group my-rg --name my-cluster
az acr login --name myregistry
az storage blob list --account-name myacct --container-name foo
```

---

## File / text utilities

```bash
# ripgrep — fast grep
rg pattern
rg -i pattern              # case-insensitive
rg -t py pattern           # only .py files
rg -l pattern              # filenames only
rg -A 3 -B 1 pattern       # context lines

# fd — fast find
fd pattern
fd -e py                   # only .py files
fd -t f -x rm              # find files and remove

# fzf
ls | fzf
git log --oneline | fzf
# In zsh: Ctrl-R for history, Ctrl-T for files, Alt-C for cd

# jq
cat file.json | jq .
jq '.users[].name' file.json
jq -r '.items[] | "\(.id) \(.name)"' file.json
curl ... | jq '.data | length'

# bat
bat file.py
bat --plain file.py        # no decorations
bat file.json -l json

# eza
eza
eza -l
eza -la --git
eza -T -L 2                # tree, 2 levels

# tmux
tmux                       # new session
tmux ls
tmux attach -t 0
tmux kill-session -t 0
# Inside tmux:
#   Ctrl-b c   new window
#   Ctrl-b ,   rename window
#   Ctrl-b %   split vertical
#   Ctrl-b "   split horizontal
#   Ctrl-b d   detach
```

---

## SSH

```bash
ssh-keygen -t ed25519 -C "you@example.com"
ssh-add ~/.ssh/id_ed25519
ssh -i ~/.ssh/id_ed25519 user@host

# Copy local file to remote
scp file user@host:/path/

# Copy remote to local
scp user@host:/path/file ./

# Sync directory
rsync -avz --progress local/ user@host:remote/

# Port forward (remote port 8080 → local 8080)
ssh -L 8080:localhost:8080 user@host
```

`~/.ssh/config` example:

```
Host gpu-box
  HostName 10.0.0.5
  User dev
  IdentityFile ~/.ssh/id_ed25519
  ForwardAgent yes
```

---

## VS Code

| Action | Shortcut (macOS / others) |
|---|---|
| Command palette | `Cmd+Shift+P` / `Ctrl+Shift+P` |
| Quick file open | `Cmd+P` / `Ctrl+P` |
| Find in files | `Cmd+Shift+F` / `Ctrl+Shift+F` |
| Toggle terminal | `Ctrl+\`` |
| Split editor | `Cmd+\` / `Ctrl+\` |
| Go to definition | `F12` |
| Peek definition | `Opt+F12` / `Alt+F12` |
| Rename symbol | `F2` |
| Format document | `Shift+Opt+F` / `Shift+Alt+F` |
| Toggle sidebar | `Cmd+B` / `Ctrl+B` |
| Switch interpreter (Python) | `Cmd+Shift+P` -> "Python: Select Interpreter" |

```bash
code .                     # open cwd
code -d file1 file2        # diff
code -r ~/other-project    # reuse window
code --install-extension <id>
code --list-extensions
```

---

## Quick env diagnostics

```bash
# What shell am I in
echo $SHELL && $SHELL --version

# Which Python and pip
which python && python --version
which pip && pip --version

# Active venv?
echo $VIRTUAL_ENV

# Docker reachable
docker info | head -5

# Cluster reachable
kubectl cluster-info

# Disk
df -h
du -sh ~/* 2>/dev/null | sort -h | tail -10
```

---

## Survival aliases (add to your rc)

```bash
alias ll='eza -lah --git'
alias g='git'
alias gs='git status -sb'
alias gd='git diff'
alias gco='git checkout'
alias gp='git push'
alias gl='git log --oneline --graph --decorate -20'
alias k='kubectl'
alias kga='kubectl get all -A'
alias d='docker'
alias dc='docker compose'
alias dps='docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
alias py='python'
alias venv='source .venv/bin/activate'
alias serve='python -m http.server 8000'
alias ports='lsof -i -P -n | grep LISTEN'
alias myip='curl -s ifconfig.me'
```

That's the daily working set. Anything more specialized lives in the relevant exercise's docs.
