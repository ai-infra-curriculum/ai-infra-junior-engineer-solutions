# Useful Tools for AI Infrastructure Engineering

A comprehensive collection of tools, utilities, and software for AI/ML infrastructure work. Organized by category with installation instructions, usage examples, and comparisons.

## Table of Contents

- [Container Tools](#container-tools)
- [Kubernetes Tools](#kubernetes-tools)
- [Python Development Tools](#python-development-tools)
- [Cloud CLI Tools](#cloud-cli-tools)
- [Infrastructure as Code](#infrastructure-as-code)
- [CI/CD Tools](#cicd-tools)
- [Monitoring & Observability](#monitoring--observability)
- [ML Development Tools](#ml-development-tools)
- [Version Control Tools](#version-control-tools)
- [Terminal & Shell Tools](#terminal--shell-tools)
- [Networking Tools](#networking-tools)
- [Security Tools](#security-tools)
- [IDE & Code Editors](#ide--code-editors)
- [Database Tools](#database-tools)
- [Testing Tools](#testing-tools)

---

## Container Tools

### Docker

**Description**: Industry-standard container platform for building, shipping, and running applications.

**Cost**: Free (Community Edition) / Paid (Enterprise)

**Installation**:
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# macOS
brew install --cask docker

# Verify installation
docker --version
```

**Basic Usage**:
```bash
# Build an image
docker build -t myapp:latest .

# Run a container
docker run -d -p 8080:80 myapp:latest

# List containers
docker ps

# View logs
docker logs <container_id>
```

**Links**:
- Official Site: https://www.docker.com/
- Documentation: https://docs.docker.com/
- Docker Hub: https://hub.docker.com/

### Docker Compose

**Description**: Tool for defining and running multi-container Docker applications.

**Cost**: Free

**Installation**:
```bash
# Linux
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify
docker-compose --version
```

**Basic Usage**:
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

**Links**: https://docs.docker.com/compose/

### Podman

**Description**: Daemonless container engine, Docker alternative with rootless containers.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y podman

# Fedora/RHEL
sudo dnf install -y podman
```

**Basic Usage**:
```bash
# Run container (Docker-compatible syntax)
podman run -d -p 8080:80 nginx

# Build image
podman build -t myapp .
```

**Links**: https://podman.io/

**Comparison**: Podman vs Docker
- Podman: Daemonless, rootless by default, Docker-compatible CLI
- Docker: Daemon-based, more ecosystem support, wider adoption
- Use Podman for: Security-focused environments, rootless containers
- Use Docker for: Broad ecosystem, enterprise support

### Buildah

**Description**: Tool for building OCI container images without Docker daemon.

**Cost**: Free (Open Source)

**Installation**:
```bash
sudo apt-get install -y buildah
```

**Basic Usage**:
```bash
# Create working container
container=$(buildah from ubuntu:20.04)

# Run commands
buildah run $container apt-get update

# Commit image
buildah commit $container myimage
```

**Links**: https://buildah.io/

### dive

**Description**: Tool for exploring Docker image layers and reducing image size.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Linux
wget https://github.com/wagoodman/dive/releases/download/v0.11.0/dive_0.11.0_linux_amd64.deb
sudo apt install ./dive_0.11.0_linux_amd64.deb

# macOS
brew install dive
```

**Basic Usage**:
```bash
# Analyze an image
dive myapp:latest
```

**Links**: https://github.com/wagoodman/dive

### Trivy

**Description**: Comprehensive security scanner for containers and other artifacts.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Using script
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# macOS
brew install trivy
```

**Basic Usage**:
```bash
# Scan Docker image
trivy image python:3.9

# Scan filesystem
trivy fs /path/to/project
```

**Links**: https://aquasecurity.github.io/trivy/

---

## Kubernetes Tools

### kubectl

**Description**: Official Kubernetes command-line tool.

**Cost**: Free

**Installation**:
```bash
# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# macOS
brew install kubectl

# Verify
kubectl version --client
```

**Basic Usage**:
```bash
# Get cluster info
kubectl cluster-info

# Get pods
kubectl get pods -A

# Apply manifest
kubectl apply -f deployment.yaml

# Port forward
kubectl port-forward pod/mypod 8080:80
```

**Links**: https://kubernetes.io/docs/reference/kubectl/

### Helm

**Description**: Package manager for Kubernetes, manages charts (pre-configured Kubernetes resources).

**Cost**: Free (Open Source)

**Installation**:
```bash
# Using script
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# macOS
brew install helm
```

**Basic Usage**:
```bash
# Add repo
helm repo add stable https://charts.helm.sh/stable

# Install chart
helm install myrelease stable/mysql

# List releases
helm list

# Upgrade release
helm upgrade myrelease stable/mysql
```

**Links**: https://helm.sh/

### k9s

**Description**: Terminal-based UI for managing Kubernetes clusters.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Linux
curl -sS https://webinstall.dev/k9s | bash

# macOS
brew install k9s
```

**Basic Usage**:
```bash
# Launch k9s
k9s

# Navigate with arrow keys
# ':' to enter command mode
# '/' to search
```

**Links**: https://k9scli.io/

### kubectx & kubens

**Description**: Fast way to switch between Kubernetes contexts and namespaces.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install kubectx

# Linux
sudo git clone https://github.com/ahmetb/kubectx /opt/kubectx
sudo ln -s /opt/kubectx/kubectx /usr/local/bin/kubectx
sudo ln -s /opt/kubectx/kubens /usr/local/bin/kubens
```

**Basic Usage**:
```bash
# Switch context
kubectx my-cluster

# Switch namespace
kubens my-namespace

# List contexts
kubectx
```

**Links**: https://github.com/ahmetb/kubectx

### stern

**Description**: Multi-pod and container log tailing for Kubernetes.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install stern

# Linux
curl -LO https://github.com/stern/stern/releases/download/v1.25.0/stern_1.25.0_linux_amd64.tar.gz
tar xzf stern_1.25.0_linux_amd64.tar.gz
sudo mv stern /usr/local/bin/
```

**Basic Usage**:
```bash
# Tail logs from all pods matching pattern
stern my-app

# Tail from specific namespace
stern --namespace production my-app
```

**Links**: https://github.com/stern/stern

### Kustomize

**Description**: Template-free Kubernetes configuration customization.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Already included in kubectl 1.14+
kubectl kustomize --help

# Or install standalone
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
```

**Basic Usage**:
```bash
# Build kustomization
kubectl kustomize ./overlays/production

# Apply kustomization
kubectl apply -k ./overlays/production
```

**Links**: https://kustomize.io/

### Lens

**Description**: Kubernetes IDE - powerful GUI for managing clusters.

**Cost**: Free (Community Edition) / Paid (Pro)

**Installation**:
```bash
# macOS
brew install --cask lens

# Windows: Download from website
# Linux: AppImage available
```

**Features**:
- Multi-cluster management
- Real-time metrics
- Terminal access
- Resource editing
- Log streaming

**Links**: https://k8slens.dev/

### Minikube

**Description**: Local Kubernetes cluster for development.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

**Basic Usage**:
```bash
# Start cluster
minikube start

# Stop cluster
minikube stop

# Access dashboard
minikube dashboard
```

**Links**: https://minikube.sigs.k8s.io/

### kind (Kubernetes IN Docker)

**Description**: Tool for running local Kubernetes clusters using Docker containers.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install kind

# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

**Basic Usage**:
```bash
# Create cluster
kind create cluster

# Create multi-node cluster
kind create cluster --config cluster.yaml

# Delete cluster
kind delete cluster
```

**Links**: https://kind.sigs.k8s.io/

**Comparison**: Minikube vs kind vs k3s
- Minikube: Full-featured, VM or container-based, best for learning
- kind: Fast, Docker-based, great for CI/CD testing
- k3s: Lightweight, production-capable, edge computing friendly

---

## Python Development Tools

### pyenv

**Description**: Simple Python version management tool.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Linux/macOS
curl https://pyenv.run | bash

# Add to shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

**Basic Usage**:
```bash
# Install Python version
pyenv install 3.10.0

# Set global version
pyenv global 3.10.0

# List installed versions
pyenv versions
```

**Links**: https://github.com/pyenv/pyenv

### Poetry

**Description**: Modern Python dependency management and packaging tool.

**Cost**: Free (Open Source)

**Installation**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Basic Usage**:
```bash
# Create new project
poetry new myproject

# Add dependency
poetry add requests

# Install dependencies
poetry install

# Run script
poetry run python script.py
```

**Links**: https://python-poetry.org/

### pipenv

**Description**: Python dependency management that combines pip and virtualenv.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install pipenv
```

**Basic Usage**:
```bash
# Create environment and install packages
pipenv install requests

# Activate environment
pipenv shell

# Run command in environment
pipenv run python script.py
```

**Links**: https://pipenv.pypa.io/

**Comparison**: Poetry vs pipenv vs pip
- Poetry: Modern, fast, lockfile support, best overall
- pipenv: Mature, Pipfile format, good for smaller projects
- pip + venv: Standard library, works everywhere, manual management

### Black

**Description**: Uncompromising Python code formatter.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install black
```

**Basic Usage**:
```bash
# Format file
black script.py

# Format directory
black src/

# Check without modifying
black --check src/
```

**Links**: https://github.com/psf/black

### pylint

**Description**: Python static code analysis tool.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install pylint
```

**Basic Usage**:
```bash
# Lint file
pylint mymodule.py

# Lint directory
pylint src/

# Generate config
pylint --generate-rcfile > .pylintrc
```

**Links**: https://pylint.org/

### mypy

**Description**: Static type checker for Python.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install mypy
```

**Basic Usage**:
```bash
# Type check file
mypy script.py

# Type check package
mypy src/

# Strict mode
mypy --strict script.py
```

**Links**: https://mypy.readthedocs.io/

### IPython

**Description**: Enhanced interactive Python shell.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install ipython
```

**Basic Usage**:
```bash
# Launch IPython
ipython

# Magic commands
%timeit function()
%run script.py
```

**Links**: https://ipython.org/

### Jupyter

**Description**: Web-based interactive development environment for notebooks.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install jupyter

# Or use JupyterLab
pip install jupyterlab
```

**Basic Usage**:
```bash
# Start Jupyter Notebook
jupyter notebook

# Start JupyterLab
jupyter lab
```

**Links**: https://jupyter.org/

---

## Cloud CLI Tools

### AWS CLI

**Description**: Official command-line interface for Amazon Web Services.

**Cost**: Free

**Installation**:
```bash
# Linux/macOS
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify
aws --version
```

**Basic Usage**:
```bash
# Configure credentials
aws configure

# List S3 buckets
aws s3 ls

# Describe EC2 instances
aws ec2 describe-instances

# Copy to S3
aws s3 cp file.txt s3://mybucket/
```

**Links**: https://aws.amazon.com/cli/

### gcloud CLI

**Description**: Command-line interface for Google Cloud Platform.

**Cost**: Free

**Installation**:
```bash
# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# macOS
brew install --cask google-cloud-sdk

# Initialize
gcloud init
```

**Basic Usage**:
```bash
# List projects
gcloud projects list

# Set project
gcloud config set project PROJECT_ID

# List compute instances
gcloud compute instances list

# Deploy to App Engine
gcloud app deploy
```

**Links**: https://cloud.google.com/sdk/gcloud

### Azure CLI

**Description**: Command-line interface for Microsoft Azure.

**Cost**: Free

**Installation**:
```bash
# Linux (one-line install)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# macOS
brew install azure-cli

# Verify
az --version
```

**Basic Usage**:
```bash
# Login
az login

# List resource groups
az group list

# Create VM
az vm create --resource-group myRG --name myVM --image UbuntuLTS

# List storage accounts
az storage account list
```

**Links**: https://docs.microsoft.com/en-us/cli/azure/

### eksctl

**Description**: CLI tool for creating and managing Amazon EKS clusters.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew tap weaveworks/tap
brew install weaveworks/tap/eksctl

# Linux
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
```

**Basic Usage**:
```bash
# Create cluster
eksctl create cluster --name my-cluster

# Delete cluster
eksctl delete cluster --name my-cluster

# List clusters
eksctl get cluster
```

**Links**: https://eksctl.io/

---

## Infrastructure as Code

### Terraform

**Description**: Infrastructure as Code tool for building, changing, and versioning infrastructure.

**Cost**: Free (Open Source) / Paid (Cloud/Enterprise)

**Installation**:
```bash
# macOS
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip
unzip terraform_1.5.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
```

**Basic Usage**:
```bash
# Initialize
terraform init

# Plan changes
terraform plan

# Apply changes
terraform apply

# Destroy infrastructure
terraform destroy
```

**Links**: https://www.terraform.io/

### Terragrunt

**Description**: Terraform wrapper that provides extra tools for keeping configurations DRY.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install terragrunt

# Linux
wget https://github.com/gruntwork-io/terragrunt/releases/download/v0.48.0/terragrunt_linux_amd64
chmod +x terragrunt_linux_amd64
sudo mv terragrunt_linux_amd64 /usr/local/bin/terragrunt
```

**Basic Usage**:
```bash
# Run terraform commands through terragrunt
terragrunt plan
terragrunt apply

# Run across multiple modules
terragrunt run-all plan
```

**Links**: https://terragrunt.gruntwork.io/

### Pulumi

**Description**: Modern Infrastructure as Code using real programming languages.

**Cost**: Free (Community) / Paid (Team/Enterprise)

**Installation**:
```bash
# macOS/Linux
curl -fsSL https://get.pulumi.com | sh

# Verify
pulumi version
```

**Basic Usage**:
```bash
# Create new project
pulumi new aws-python

# Preview changes
pulumi preview

# Deploy
pulumi up

# Destroy
pulumi destroy
```

**Links**: https://www.pulumi.com/

### Ansible

**Description**: IT automation tool for configuration management and application deployment.

**Cost**: Free (Open Source) / Paid (Tower/AWX)

**Installation**:
```bash
# Python pip
pip install ansible

# macOS
brew install ansible

# Verify
ansible --version
```

**Basic Usage**:
```bash
# Run playbook
ansible-playbook playbook.yml

# Run ad-hoc command
ansible all -m ping

# Check syntax
ansible-playbook playbook.yml --syntax-check
```

**Links**: https://www.ansible.com/

**Comparison**: Terraform vs Pulumi vs Ansible
- Terraform: Declarative, HCL language, vast provider ecosystem
- Pulumi: Use real programming languages (Python, TypeScript, etc.)
- Ansible: Procedural, great for configuration management
- Use Terraform for: Standard IaC, multi-cloud, large ecosystem
- Use Pulumi for: Complex logic, existing language skills
- Use Ansible for: Configuration management, app deployment

---

## CI/CD Tools

### Jenkins

**Description**: Open source automation server for CI/CD.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Using Docker
docker run -p 8080:8080 -p 50000:50000 jenkins/jenkins:lts

# Ubuntu/Debian
wget -q -O - https://pkg.jenkins.io/debian/jenkins.io.key | sudo apt-key add -
sudo sh -c 'echo deb http://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
sudo apt-get update
sudo apt-get install jenkins
```

**Links**: https://www.jenkins.io/

### GitHub Actions

**Description**: CI/CD platform integrated with GitHub repositories.

**Cost**: Free (public repos, limited minutes for private) / Paid (additional minutes)

**Basic Usage**: Create `.github/workflows/ci.yml`:
```yaml
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest
```

**Links**: https://github.com/features/actions

### GitLab CI/CD

**Description**: Built-in CI/CD in GitLab.

**Cost**: Free (GitLab.com) / Paid (Premium features)

**Basic Usage**: Create `.gitlab-ci.yml`:
```yaml
test:
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pytest
```

**Links**: https://docs.gitlab.com/ee/ci/

### ArgoCD

**Description**: Declarative GitOps continuous delivery tool for Kubernetes.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Install in Kubernetes
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Install CLI
brew install argocd
```

**Basic Usage**:
```bash
# Login
argocd login <ARGOCD_SERVER>

# Create app
argocd app create myapp \
  --repo https://github.com/myorg/myrepo \
  --path manifests \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace default

# Sync app
argocd app sync myapp
```

**Links**: https://argo-cd.readthedocs.io/

### CircleCI CLI

**Description**: Command-line tool for CircleCI.

**Cost**: Free tier available / Paid plans

**Installation**:
```bash
# macOS
brew install circleci

# Linux
curl -fLSs https://raw.githubusercontent.com/CircleCI-Public/circleci-cli/master/install.sh | bash
```

**Basic Usage**:
```bash
# Validate config
circleci config validate

# Run job locally
circleci local execute --job test
```

**Links**: https://circleci.com/docs/local-cli/

---

## Monitoring & Observability

### Prometheus

**Description**: Open-source monitoring and alerting toolkit.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Using Docker
docker run -p 9090:9090 prom/prometheus

# Or download binary
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64
./prometheus
```

**Basic Usage**:
- Access UI at http://localhost:9090
- Configure targets in `prometheus.yml`
- Query metrics using PromQL

**Links**: https://prometheus.io/

### Grafana

**Description**: Open-source platform for monitoring and observability.

**Cost**: Free (Open Source) / Paid (Cloud/Enterprise)

**Installation**:
```bash
# Using Docker
docker run -d -p 3000:3000 grafana/grafana

# Ubuntu/Debian
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
sudo apt-get update
sudo apt-get install grafana
```

**Links**: https://grafana.com/

### Datadog Agent

**Description**: Monitoring agent for Datadog platform.

**Cost**: Paid (with free tier)

**Installation**:
```bash
# One-line install
DD_API_KEY=<your-api-key> bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"
```

**Links**: https://www.datadoghq.com/

### Jaeger

**Description**: Open-source distributed tracing system.

**Cost**: Free (Open Source)

**Installation**:
```bash
# All-in-one Docker image
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 14268:14268 \
  jaegertracing/all-in-one:latest
```

**Links**: https://www.jaegertracing.io/

### OpenTelemetry Collector

**Description**: Vendor-agnostic way to receive, process and export telemetry data.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Download collector
curl -LO https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.80.0/otelcol_0.80.0_linux_amd64.tar.gz
tar -xvf otelcol_0.80.0_linux_amd64.tar.gz
```

**Links**: https://opentelemetry.io/

### ctop

**Description**: Top-like interface for container metrics.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Linux
sudo wget https://github.com/bcicen/ctop/releases/download/v0.7.7/ctop-0.7.7-linux-amd64 -O /usr/local/bin/ctop
sudo chmod +x /usr/local/bin/ctop

# macOS
brew install ctop
```

**Basic Usage**:
```bash
# Launch ctop
ctop
```

**Links**: https://github.com/bcicen/ctop

---

## ML Development Tools

### MLflow

**Description**: Open-source platform for the ML lifecycle.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install mlflow
```

**Basic Usage**:
```bash
# Start UI
mlflow ui

# Run experiment
mlflow run . -P alpha=0.5

# Serve model
mlflow models serve -m runs:/<RUN_ID>/model
```

**Links**: https://mlflow.org/

### DVC (Data Version Control)

**Description**: Git for data science - manage data, models, and experiments.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install dvc

# Or with extras
pip install dvc[all]
```

**Basic Usage**:
```bash
# Initialize
dvc init

# Track data
dvc add data/dataset.csv

# Push to remote
dvc push

# Pull data
dvc pull
```

**Links**: https://dvc.org/

### Weights & Biases (wandb)

**Description**: Experiment tracking, model management, and collaboration platform.

**Cost**: Free (individual/academic) / Paid (teams)

**Installation**:
```bash
pip install wandb
```

**Basic Usage**:
```python
import wandb

# Initialize run
wandb.init(project="my-project")

# Log metrics
wandb.log({"accuracy": 0.95})
```

**Links**: https://wandb.ai/

### TensorBoard

**Description**: TensorFlow's visualization toolkit.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install tensorboard
```

**Basic Usage**:
```bash
# Start TensorBoard
tensorboard --logdir=logs/

# Access at http://localhost:6006
```

**Links**: https://www.tensorflow.org/tensorboard

### Optuna

**Description**: Hyperparameter optimization framework.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install optuna
```

**Basic Usage**:
```python
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

**Links**: https://optuna.org/

### Ray

**Description**: Framework for scaling Python and ML applications.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install ray[default]
```

**Basic Usage**:
```python
import ray
ray.init()

@ray.remote
def task():
    return 1

result = ray.get(task.remote())
```

**Links**: https://www.ray.io/

---

## Version Control Tools

### Git

**Description**: Distributed version control system.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install git

# macOS
brew install git

# Configure
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

**Basic Usage**:
```bash
# Clone repository
git clone https://github.com/user/repo.git

# Stage changes
git add .

# Commit
git commit -m "Message"

# Push
git push origin main
```

**Links**: https://git-scm.com/

### gh (GitHub CLI)

**Description**: GitHub on the command line.

**Cost**: Free

**Installation**:
```bash
# macOS
brew install gh

# Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

**Basic Usage**:
```bash
# Authenticate
gh auth login

# Create repo
gh repo create

# Create PR
gh pr create

# View issues
gh issue list
```

**Links**: https://cli.github.com/

### GitLab CLI (glab)

**Description**: GitLab CLI tool.

**Cost**: Free

**Installation**:
```bash
# macOS
brew install glab

# Linux
curl -sL https://j.mp/glab-i | sudo bash
```

**Basic Usage**:
```bash
# Clone project
glab repo clone group/project

# Create MR
glab mr create

# List issues
glab issue list
```

**Links**: https://gitlab.com/gitlab-org/cli

### pre-commit

**Description**: Framework for managing git pre-commit hooks.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install pre-commit
```

**Basic Usage**:
```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

**Links**: https://pre-commit.com/

---

## Terminal & Shell Tools

### Oh My Zsh

**Description**: Framework for managing Zsh configuration.

**Cost**: Free (Open Source)

**Installation**:
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

**Links**: https://ohmyz.sh/

### tmux

**Description**: Terminal multiplexer for managing multiple terminal sessions.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install tmux

# macOS
brew install tmux
```

**Basic Usage**:
```bash
# Start session
tmux

# Split pane horizontally
Ctrl+b then "

# Split pane vertically
Ctrl+b then %

# Detach
Ctrl+b then d

# Reattach
tmux attach
```

**Links**: https://github.com/tmux/tmux

### fzf

**Description**: Command-line fuzzy finder.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install fzf

# Linux
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install
```

**Basic Usage**:
```bash
# Search files
vim $(fzf)

# Search command history
Ctrl+R

# Search directories
cd $(find * -type d | fzf)
```

**Links**: https://github.com/junegunn/fzf

### bat

**Description**: Cat clone with syntax highlighting and Git integration.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install bat

# Ubuntu/Debian
sudo apt install bat
```

**Basic Usage**:
```bash
# View file
bat file.py

# View with line numbers
bat -n file.py
```

**Links**: https://github.com/sharkdp/bat

### exa

**Description**: Modern replacement for ls.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install exa

# Ubuntu/Debian
sudo apt install exa
```

**Basic Usage**:
```bash
# List files
exa

# Tree view
exa --tree

# Long format with git status
exa -l --git
```

**Links**: https://the.exa.website/

### ripgrep (rg)

**Description**: Fast line-oriented search tool.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install ripgrep

# Ubuntu/Debian
sudo apt-get install ripgrep
```

**Basic Usage**:
```bash
# Search for pattern
rg "pattern"

# Search in specific files
rg "pattern" -g "*.py"

# Case insensitive
rg -i "pattern"
```

**Links**: https://github.com/BurntSushi/ripgrep

### htop

**Description**: Interactive process viewer.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install htop

# macOS
brew install htop
```

**Basic Usage**:
```bash
# Launch htop
htop
```

**Links**: https://htop.dev/

---

## Networking Tools

### curl

**Description**: Command-line tool for transferring data with URLs.

**Cost**: Free (Open Source)

**Installation**: Usually pre-installed on most systems

**Basic Usage**:
```bash
# GET request
curl https://api.example.com

# POST request
curl -X POST -d '{"key":"value"}' https://api.example.com

# Save to file
curl -o file.txt https://example.com/file.txt
```

**Links**: https://curl.se/

### httpie

**Description**: User-friendly HTTP client.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Python pip
pip install httpie

# macOS
brew install httpie
```

**Basic Usage**:
```bash
# GET request
http GET https://api.example.com

# POST with JSON
http POST https://api.example.com key=value

# Headers
http GET https://api.example.com Authorization:"Bearer token"
```

**Links**: https://httpie.io/

### nmap

**Description**: Network exploration and security auditing tool.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install nmap

# macOS
brew install nmap
```

**Basic Usage**:
```bash
# Scan host
nmap 192.168.1.1

# Scan ports
nmap -p 80,443 example.com

# Service detection
nmap -sV example.com
```

**Links**: https://nmap.org/

### netcat (nc)

**Description**: Networking utility for reading/writing network connections.

**Cost**: Free (Open Source)

**Installation**: Usually pre-installed

**Basic Usage**:
```bash
# Listen on port
nc -l 8080

# Connect to host
nc example.com 80

# Port scanning
nc -zv example.com 20-30
```

### tcpdump

**Description**: Packet analyzer for network traffic.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install tcpdump

# macOS (usually pre-installed)
```

**Basic Usage**:
```bash
# Capture on interface
sudo tcpdump -i eth0

# Save to file
sudo tcpdump -w capture.pcap

# Filter by port
sudo tcpdump port 80
```

**Links**: https://www.tcpdump.org/

---

## Security Tools

### Vault

**Description**: Secrets management tool by HashiCorp.

**Cost**: Free (Open Source) / Paid (Enterprise)

**Installation**:
```bash
# macOS
brew tap hashicorp/tap
brew install hashicorp/tap/vault

# Linux
wget https://releases.hashicorp.com/vault/1.14.0/vault_1.14.0_linux_amd64.zip
unzip vault_1.14.0_linux_amd64.zip
sudo mv vault /usr/local/bin/
```

**Basic Usage**:
```bash
# Start dev server
vault server -dev

# Set secret
vault kv put secret/myapp password=secret

# Get secret
vault kv get secret/myapp
```

**Links**: https://www.vaultproject.io/

### SOPS

**Description**: Editor of encrypted files supporting various formats.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install sops

# Linux
wget https://github.com/mozilla/sops/releases/download/v3.7.3/sops-v3.7.3.linux
chmod +x sops-v3.7.3.linux
sudo mv sops-v3.7.3.linux /usr/local/bin/sops
```

**Basic Usage**:
```bash
# Encrypt file
sops -e secrets.yaml > secrets.enc.yaml

# Edit encrypted file
sops secrets.enc.yaml

# Decrypt
sops -d secrets.enc.yaml
```

**Links**: https://github.com/mozilla/sops

### Checkov

**Description**: Static code analysis tool for infrastructure as code.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install checkov
```

**Basic Usage**:
```bash
# Scan directory
checkov -d /path/to/iac

# Scan specific framework
checkov -d . --framework terraform

# Output as JSON
checkov -d . -o json
```

**Links**: https://www.checkov.io/

### tfsec

**Description**: Security scanner for Terraform code.

**Cost**: Free (Open Source)

**Installation**:
```bash
# macOS
brew install tfsec

# Linux
curl -s https://raw.githubusercontent.com/aquasecurity/tfsec/master/scripts/install_linux.sh | bash
```

**Basic Usage**:
```bash
# Scan current directory
tfsec .

# Scan with specific checks
tfsec --minimum-severity HIGH .
```

**Links**: https://aquasecurity.github.io/tfsec/

---

## IDE & Code Editors

### Visual Studio Code

**Description**: Popular open-source code editor by Microsoft.

**Cost**: Free

**Installation**:
```bash
# macOS
brew install --cask visual-studio-code

# Ubuntu/Debian
sudo snap install --classic code
```

**Essential Extensions for AI Infrastructure**:
- Python
- Docker
- Kubernetes
- Remote - SSH
- GitLens
- Terraform
- YAML

**Links**: https://code.visualstudio.com/

### PyCharm

**Description**: Python IDE by JetBrains.

**Cost**: Free (Community) / Paid (Professional)

**Installation**:
```bash
# macOS
brew install --cask pycharm-ce

# Or download from website
```

**Links**: https://www.jetbrains.com/pycharm/

### Vim/Neovim

**Description**: Highly configurable text editor.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Neovim on macOS
brew install neovim

# Ubuntu/Debian
sudo apt-get install neovim
```

**Links**:
- Vim: https://www.vim.org/
- Neovim: https://neovim.io/

---

## Database Tools

### DBeaver

**Description**: Universal database tool.

**Cost**: Free (Community) / Paid (Enterprise)

**Installation**:
```bash
# macOS
brew install --cask dbeaver-community

# Linux: Download from website
```

**Links**: https://dbeaver.io/

### pgcli

**Description**: Postgres CLI with auto-completion.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install pgcli
```

**Basic Usage**:
```bash
pgcli postgresql://user:password@localhost:5432/database
```

**Links**: https://www.pgcli.com/

### Redis CLI

**Description**: Command-line interface for Redis.

**Cost**: Free (Open Source)

**Installation**:
```bash
# Comes with Redis
brew install redis  # macOS
sudo apt-get install redis-tools  # Ubuntu
```

**Basic Usage**:
```bash
redis-cli
> SET key value
> GET key
```

**Links**: https://redis.io/

---

## Testing Tools

### pytest

**Description**: Python testing framework.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install pytest
```

**Basic Usage**:
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=myapp tests/

# Run specific test
pytest tests/test_module.py::test_function
```

**Links**: https://pytest.org/

### Locust

**Description**: Load testing tool written in Python.

**Cost**: Free (Open Source)

**Installation**:
```bash
pip install locust
```

**Basic Usage**:
```bash
# Start Locust
locust -f locustfile.py

# Access web interface at http://localhost:8089
```

**Links**: https://locust.io/

### k6

**Description**: Modern load testing tool.

**Cost**: Free (Open Source) / Paid (Cloud)

**Installation**:
```bash
# macOS
brew install k6

# Linux
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

**Basic Usage**:
```bash
# Run test
k6 run script.js

# Run with virtual users
k6 run --vus 10 --duration 30s script.js
```

**Links**: https://k6.io/

---

## Comparison Tables

### Container Platforms
| Tool | Best For | Learning Curve | Production Ready |
|------|----------|----------------|------------------|
| Docker | General use, wide ecosystem | Low | Yes |
| Podman | Rootless, daemonless | Medium | Yes |
| containerd | Kubernetes runtime | High | Yes |

### Kubernetes Development
| Tool | Purpose | Type | Best For |
|------|---------|------|----------|
| Minikube | Local cluster | Full cluster | Learning, feature testing |
| kind | Local cluster | Lightweight | CI/CD, testing |
| k3s | Lightweight K8s | Minimal | Edge, production |
| k9s | Management | CLI UI | Daily operations |
| Lens | Management | GUI | Visual cluster management |

### Infrastructure as Code
| Tool | Language | Cloud Support | Learning Curve |
|------|----------|---------------|----------------|
| Terraform | HCL | Multi-cloud | Medium |
| Pulumi | Real languages | Multi-cloud | Medium-High |
| Ansible | YAML | Agnostic | Low-Medium |
| CloudFormation | JSON/YAML | AWS only | Medium |

### Monitoring Solutions
| Tool | Type | Cost | Best For |
|------|------|------|----------|
| Prometheus | Metrics | Free | Time-series metrics |
| Grafana | Visualization | Free/Paid | Dashboards |
| Datadog | Full platform | Paid | Enterprise monitoring |
| New Relic | APM | Paid | Application performance |
| ELK Stack | Logs | Free/Paid | Log aggregation |

---

## Quick Start Guides

### Setting Up a Development Environment

```bash
# Install essential tools
brew install docker kubectl helm terraform git python@3.10

# Install Python tools
pip install poetry black pylint mypy pytest

# Install cloud CLIs
brew install awscli google-cloud-sdk azure-cli

# Install monitoring tools
brew install k9s stern
```

### Docker Development Workflow

```bash
# Build image
docker build -t myapp:dev .

# Run with volume mount for development
docker run -v $(pwd):/app -p 8080:8080 myapp:dev

# Debug running container
docker exec -it <container> /bin/bash

# View logs
docker logs -f <container>
```

### Kubernetes Development Workflow

```bash
# Start local cluster
minikube start

# Apply manifests
kubectl apply -f k8s/

# Port forward for local access
kubectl port-forward svc/myapp 8080:80

# Stream logs
stern myapp

# Interactive debugging
kubectl exec -it <pod> -- /bin/bash
```

---

## Tool Integration Examples

### Docker + Python Development

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - .:/app
    environment:
      - DEBUG=true
```

### Kubernetes + Helm Deployment

```bash
# Create Helm chart
helm create myapp

# Install with custom values
helm install myapp ./myapp -f values-prod.yaml

# Upgrade deployment
helm upgrade myapp ./myapp

# Rollback if needed
helm rollback myapp
```

### Terraform + Cloud Provider

```hcl
# main.tf
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "MyWebServer"
  }
}
```

```bash
# Deploy
terraform init
terraform plan
terraform apply
```

---

## Recommended Tool Combinations

### Minimal Setup (Getting Started)
- Docker
- Visual Studio Code
- Git
- Python + pip
- kubectl + Minikube

### Intermediate Setup
- Docker + Docker Compose
- Kubernetes (kubectl, Helm, k9s)
- Terraform or Pulumi
- GitHub Actions or GitLab CI
- Prometheus + Grafana

### Advanced Setup
- Container platform (Docker/Podman)
- Kubernetes (kubectl, Helm, ArgoCD, Lens)
- IaC (Terraform + Terragrunt)
- CI/CD (Jenkins/GitHub Actions + ArgoCD)
- Monitoring (Prometheus, Grafana, Jaeger)
- ML Tools (MLflow, DVC, Weights & Biases)
- Cloud CLIs (AWS/GCP/Azure)

---

## Resources for Learning Tools

- **Official Documentation**: Always start here
- **YouTube Channels**: TechWorld with Nana, DevOps Toolkit, Cloud Native Foundation
- **Practice Platforms**: Katacoda, Play with Docker, Play with Kubernetes
- **Community Forums**: Reddit r/devops, r/kubernetes, Stack Overflow
- **Books**: See additional-reading.md for tool-specific books

---

Last Updated: October 2025
