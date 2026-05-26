# Exercise 05: Linux Package Management — Solution

## What the exercise asked for

Practice the package-management tools every Linux-on-ML
operator needs: apt/dnf for OS packages, pip/uv/poetry for
Python, conda for environments, and Docker for containerized
runtimes.

## OS package management

### Debian / Ubuntu (apt)

```bash
# Update package indices first
sudo apt update

# Install
sudo apt install -y build-essential python3-pip nvidia-cuda-toolkit

# Search
apt search "cuda"

# Show details
apt show pytorch

# Held packages (don't auto-upgrade — useful for pinning driver versions)
sudo apt-mark hold nvidia-driver-550

# Clean cache (in containers especially, after install)
sudo apt clean && rm -rf /var/lib/apt/lists/*
```

### Red Hat / Fedora (dnf / yum)

```bash
sudo dnf install gcc python3-devel cuda-toolkit
sudo dnf search cuda
sudo dnf upgrade
```

For ML platforms running on RHEL-derived distros (CentOS Stream,
Rocky, Alma), the package set is usually narrower than Debian
— budget for additional builds from source.

## Python package management

### pip (the baseline)

```bash
# Install from PyPI
pip install torch torchvision

# From a requirements file
pip install -r requirements.txt

# From a specific version
pip install "torch==2.1.2"

# Editable install (for your own code)
pip install -e .

# Show what's installed
pip list
pip show torch

# Freeze for reproducibility
pip freeze > requirements-lock.txt
```

### uv (the new fast standard)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a venv
uv venv .venv

# Install from a requirements file
uv pip install -r requirements.txt

# Sync exactly what's in the lockfile
uv pip sync requirements-lock.txt
```

`uv` is 10-100× faster than pip. For ML platforms with large
dependency trees, the time savings compound.

### poetry (for projects with lockfiles)

```bash
# Initialize a project
poetry init

# Add a dep
poetry add torch

# Add a dev-only dep
poetry add --group dev pytest

# Install
poetry install

# Lock without installing
poetry lock --no-update
```

Poetry shines for projects with both runtime + dev + test
deps that need to be locked together.

## Virtual environments

```bash
# Built-in: venv
python3 -m venv .venv
source .venv/bin/activate

# uv (preferred for ML)
uv venv .venv && source .venv/bin/activate

# conda (especially for non-Python deps like CUDA)
conda create -n ml-env python=3.11
conda activate ml-env
conda install pytorch torchvision -c pytorch
```

When to use conda vs. pip-based:

- **conda**: when you need non-Python dependencies that
  pip doesn't manage well (CUDA, MKL, system libraries).
  Older but still common in research workflows.
- **uv / pip + venv**: when your deps are all Python or
  you're using Docker for the system layer. Faster, more
  reproducible, smaller footprint.

## ML-specific tips

- **Pin transitive deps**: `pip freeze > requirements-lock.txt`
  catches the actual versions installed, not just direct deps.
- **Separate dev / test / runtime deps**: pyproject.toml with
  groups, or separate requirements files.
- **GPU stack alignment**: PyTorch + CUDA + driver versions
  must match. Check the [PyTorch install matrix](https://pytorch.org/get-started/locally/)
  before installing.
- **Avoid `--upgrade` in production**: it's how you get
  surprise breakage.

## Common mistakes

- Installing globally with `sudo pip install` (breaks system
  Python, gets clobbered on OS updates).
- Not pinning versions; "works on my machine" syndrome.
- Mixing conda + pip in the same env without care.
- Forgetting to activate the venv (installs go to the wrong
  place).
- Using `apt install python3-numpy` when the project needs
  a specific version (apt versions lag).

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-002-linux-essentials/exercises/exercise-05-package-mgmt.md`
- Docker is the production-level answer:
  `mod-005-docker-containers/`.
