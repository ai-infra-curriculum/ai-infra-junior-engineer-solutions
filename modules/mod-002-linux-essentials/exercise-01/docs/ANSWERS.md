# Exercise 01: Navigation and File System - Reflection Questions

## Question 1: Explain the difference between absolute and relative paths. When would you use each?

### Answer

**Absolute Paths** and **Relative Paths** are two fundamental ways to specify file locations in Linux.

### Absolute Paths

An **absolute path** specifies the complete path from the root directory (`/`) to a file or directory.

**Characteristics**:
- Always starts with `/` (root) or `~` (home directory)
- Provides complete location information
- Works from any current directory
- Always unambiguous

**Examples**:
```bash
/home/user/projects/ml-classifier/data/raw/dataset.csv
/usr/local/bin/python3
/etc/nginx/nginx.conf
~/projects/ml-classifier/models/production/model.pkl
```

**Structure Breakdown**:
```
/home/user/projects/ml-classifier/data/raw/dataset.csv
│  │    │    │         │            │    │   └─ File
│  │    │    │         │            │    └─── Directory
│  │    │    │         │            └──────── Directory
│  │    │    │         └─────────────────────  Directory
│  │    │    └───────────────────────────────  Directory
│  │    └────────────────────────────────────  Directory
│  └─────────────────────────────────────────  Directory
└────────────────────────────────────────────  Root
```

### Relative Paths

A **relative path** specifies a location relative to the current working directory.

**Characteristics**:
- Never starts with `/`
- Depends on current working directory
- Shorter and more convenient for nearby files
- Uses `.` (current directory) and `..` (parent directory)

**Examples**:
```bash
# If current directory is /home/user/projects/ml-classifier
data/raw/dataset.csv              # File in subdirectory
../another-project/README.md      # File in sibling directory
../../shared-data/imagenet.csv    # File two levels up and down
./train.sh                        # File in current directory
```

**Navigation Symbols**:
- `.` - Current directory
- `..` - Parent directory (one level up)
- `../..` - Two levels up
- `~` - Home directory (shortcut for absolute path)
- `-` - Previous directory

### When to Use Absolute Paths

**1. Scripts and Automation**
```bash
#!/bin/bash
# Absolute paths ensure script works from any location
DATA_PATH="/home/user/ml-projects/data/raw"
MODEL_PATH="/home/user/ml-projects/models/production"

python3 /home/user/ml-projects/src/train.py \
    --data "$DATA_PATH" \
    --output "$MODEL_PATH"
```

**2. Configuration Files**
```yaml
# config.yaml
data:
  train: /mnt/datasets/imagenet/train
  val: /mnt/datasets/imagenet/val
models:
  checkpoint_dir: /models/checkpoints
  production_dir: /models/production
```

**3. Symbolic Links to Shared Resources**
```bash
# Link to shared dataset that shouldn't move
ln -s /mnt/shared-storage/datasets/imagenet ~/projects/ml-classifier/data/imagenet
```

**4. System Commands and Binaries**
```bash
# Always use absolute path for system binaries in cron jobs
0 2 * * * /usr/bin/python3 /home/user/scripts/backup.py
```

**5. Collaborative Environments**
```bash
# When team members have different directory structures
# But agree on absolute paths for shared resources
/opt/ml-infrastructure/datasets/
/opt/ml-infrastructure/models/
```

### When to Use Relative Paths

**1. Project-Internal Navigation**
```bash
# Working within a project
cd ~/projects/ml-classifier

# Navigate to data
cd data/raw                    # Not cd ~/projects/ml-classifier/data/raw

# Navigate to models from data
cd ../../models/checkpoints    # Not cd ~/projects/ml-classifier/models/checkpoints
```

**2. Portable Scripts**
```bash
#!/bin/bash
# Script that works regardless of installation location

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use relative paths from script location
DATA_DIR="$SCRIPT_DIR/../data"
SRC_DIR="$SCRIPT_DIR/../src"

python3 "$SRC_DIR/train.py" --data "$DATA_DIR/train.csv"
```

**3. Git Repositories**
```bash
# .gitignore with relative paths
data/raw/*
models/checkpoints/*
__pycache__/

# Not:
# /home/user/project/data/raw/*  # Would only work for one user
```

**4. Documentation**
```markdown
# README.md with relative paths
To train the model:
```bash
cd src/training
python train.py --config ../../configs/train.yaml
```
```

**5. Copying or Moving Project Directories**
```bash
# Project using relative paths can be moved anywhere
cp -r ~/projects/ml-classifier /tmp/
cd /tmp/ml-classifier
./scripts/train.sh  # Still works!

# Project using absolute paths would break
# ./scripts/train.sh  # Error: /home/user/... not found
```

### Comparison Table

| Aspect | Absolute Path | Relative Path |
|--------|--------------|---------------|
| **Starting Point** | Root (`/`) or home (`~`) | Current directory |
| **Example** | `/home/user/project/data` | `../data` |
| **Length** | Usually longer | Usually shorter |
| **Portability** | Not portable (hardcoded) | Portable (adaptable) |
| **Reliability** | Always works (if path exists) | Depends on current directory |
| **Best For** | System config, cron jobs | Project navigation, Git repos |
| **Safety** | Can't accidentally affect wrong directory | Risk if current directory is wrong |

### Real-World ML Infrastructure Example

```bash
# Project structure
ml-classifier/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
└── scripts/
    └── train.sh

# train.sh using BOTH types appropriately
#!/bin/bash

# Absolute path for shared dataset (doesn't change)
SHARED_DATA="/mnt/datasets/imagenet"

# Relative paths for project structure (portable)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src/training"
OUTPUT_DIR="$PROJECT_ROOT/models/checkpoints"

# Use absolute path for shared resource
# Use relative structure for project organization
python3 "$SRC_DIR/train.py" \
    --data "$SHARED_DATA/train" \
    --output "$OUTPUT_DIR"
```

### Best Practices

**1. Use Absolute Paths For**:
- External dependencies
- System resources
- Shared storage
- Production deployments
- Cron jobs and scheduled tasks

**2. Use Relative Paths For**:
- Project-internal navigation
- Git-tracked files
- Portable scripts
- Documentation
- Development workflows

**3. Convert Between Them**:
```bash
# Get absolute path from relative
RELATIVE_PATH="../data"
ABSOLUTE_PATH=$(cd "$RELATIVE_PATH" && pwd)
echo "$ABSOLUTE_PATH"  # /home/user/projects/ml-classifier/data

# Use relative from absolute
CURRENT=$(pwd)
TARGET="/home/user/projects/ml-classifier/models"
RELATIVE=$(realpath --relative-to="$CURRENT" "$TARGET")
echo "$RELATIVE"  # ../models (if in src/)
```

**4. Defensive Programming**:
```bash
#!/bin/bash
# Always verify directory before operations

TARGET_DIR="/path/to/important/data"

# Check if absolute path exists
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: $TARGET_DIR does not exist"
    exit 1
fi

# Verify you're in correct directory for relative paths
if [[ $(basename "$PWD") != "ml-classifier" ]]; then
    echo "Error: Must run from ml-classifier directory"
    exit 1
fi
```

### Summary

- **Absolute paths** provide certainty and work from anywhere, but are less portable
- **Relative paths** are portable and concise, but require awareness of current location
- **Use absolute paths** for system resources, configuration, and automation
- **Use relative paths** for project-internal navigation and portable scripts
- **Best practice**: Combine both - absolute for external dependencies, relative for internal structure

---

## Question 2: What is the purpose of the `/home`, `/etc`, `/var`, and `/usr` directories in Linux?

### Answer

The Linux Filesystem Hierarchy Standard (FHS) defines a structured organization for the filesystem. Understanding these key directories is essential for ML infrastructure engineers.

## /home - User Home Directories

**Purpose**: Personal directories for system users

**Contains**:
- User personal files
- User configurations (dotfiles)
- User projects and data
- User-specific application settings

**Structure**:
```
/home/
├── alice/
│   ├── .bashrc
│   ├── .ssh/
│   ├── projects/
│   │   └── ml-classifier/
│   └── datasets/
├── bob/
│   ├── .vimrc
│   └── experiments/
└── mluser/
    ├── models/
    ├── data/
    └── notebooks/
```

**ML Infrastructure Examples**:
```bash
# User project directory
/home/mluser/projects/image-classification/

# User datasets
/home/mluser/datasets/imagenet/

# User trained models
/home/mluser/models/production/

# User virtual environments
/home/mluser/.virtualenvs/ml-project/

# User Jupyter notebooks
/home/mluser/notebooks/experiments/
```

**Key Characteristics**:
- Each user has their own subdirectory: `/home/username`
- Users have full control over their home directory
- Default location when user logs in
- Separated from system files for security
- Can be on separate partition for easy backups

**Common User Configurations** (Hidden Files):
```bash
~/.bashrc          # Bash shell configuration
~/.bash_profile    # Bash login configuration
~/.ssh/            # SSH keys and config
~/.gitconfig       # Git configuration
~/.vimrc           # Vim editor configuration
~/.jupyter/        # Jupyter configuration
~/.aws/            # AWS credentials
~/.docker/         # Docker configuration
```

**Permissions**:
```bash
$ ls -ld /home/mluser
drwxr-xr-x 25 mluser mluser 4096 Jan 31 14:30 /home/mluser
# drwxr-xr-x: Owner has rwx, group and others have r-x
```

## /etc - System Configuration Files

**Purpose**: System-wide configuration files and scripts

**Contains**:
- System configuration files
- Application configurations
- Service configurations
- Network settings
- User authentication information

**Structure**:
```
/etc/
├── hostname           # System hostname
├── hosts              # Host name to IP mappings
├── passwd             # User account information
├── group              # Group information
├── ssh/
│   └── sshd_config   # SSH server configuration
├── nginx/
│   └── nginx.conf    # Nginx web server config
├── systemd/
│   └── system/       # Service configurations
├── apt/              # Package manager config
├── environment       # System-wide environment variables
└── docker/
    └── daemon.json   # Docker daemon configuration
```

**ML Infrastructure Examples**:
```bash
# Nginx configuration for ML model serving
/etc/nginx/sites-available/ml-api

# Docker daemon configuration
/etc/docker/daemon.json

# System-wide Python package locations
/etc/pip.conf

# Jupyter Hub configuration
/etc/jupyterhub/jupyterhub_config.py

# CUDA configuration
/etc/ld.so.conf.d/cuda.conf

# Environment variables for ML
/etc/environment
```

**Example Configuration Files**:

```bash
# /etc/hosts - Map hostnames to IPs
127.0.0.1   localhost
192.168.1.10 ml-server-1
192.168.1.11 ml-server-2

# /etc/environment - System-wide environment
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CUDA_HOME="/usr/local/cuda"
LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# /etc/nginx/sites-available/ml-api
server {
    listen 80;
    server_name api.ml-project.com;

    location / {
        proxy_pass http://localhost:5000;
    }
}
```

**Key Characteristics**:
- Typically text-based configuration files
- Requires root/sudo access to modify
- System-wide effects (all users)
- Backed up frequently
- Version controlled (with sensitive data removed)

**Best Practices for ML Infrastructure**:
```bash
# Always backup before modifying
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Use configuration management (Ansible, Chef, Puppet)
# Track changes in Git (without secrets)
# Use environment-specific configs (dev, staging, prod)
```

## /var - Variable Data Files

**Purpose**: Files that change frequently during system operation

**Contains**:
- Log files
- Temporary files
- Spool files (print, mail)
- Cache data
- Application runtime data

**Structure**:
```
/var/
├── log/              # System and application logs
│   ├── syslog       # System logs
│   ├── auth.log     # Authentication logs
│   ├── nginx/       # Nginx logs
│   └── apt/         # Package manager logs
├── cache/           # Application cache
│   ├── apt/        # Package cache
│   └── pip/        # Python package cache
├── spool/          # Spooled files
│   ├── cron/       # Cron job output
│   └── mail/       # Email queue
├── tmp/            # Temporary files (survives reboots)
├── lib/            # Variable state information
│   ├── docker/     # Docker images and containers
│   └── postgresql/ # Database files
└── run/            # Runtime data (cleared on reboot)
```

**ML Infrastructure Examples**:
```bash
# Application logs
/var/log/ml-api/access.log
/var/log/ml-api/error.log
/var/log/training/train_20250131.log

# Docker data
/var/lib/docker/containers/
/var/lib/docker/volumes/

# Database data
/var/lib/postgresql/12/main/
/var/lib/mongodb/

# Temporary model files
/var/tmp/model-cache/

# Training job outputs
/var/spool/ml-jobs/output/

# Nginx logs
/var/log/nginx/access.log
/var/log/nginx/error.log
```

**Log File Examples**:

```bash
# /var/log/ml-api/training.log
2025-01-31 14:30:15 INFO Starting training job job_12345
2025-01-31 14:30:20 INFO Loading data from /data/train.csv
2025-01-31 14:30:25 INFO Epoch 1/10 - Loss: 0.5234
2025-01-31 14:35:30 INFO Epoch 10/10 - Loss: 0.1245
2025-01-31 14:35:35 INFO Model saved to /models/model_final.pkl

# /var/log/nginx/ml-api-access.log
192.168.1.100 - - [31/Jan/2025:14:30:00 +0000] "POST /predict HTTP/1.1" 200 1024
192.168.1.101 - - [31/Jan/2025:14:30:01 +0000] "POST /predict HTTP/1.1" 200 987
```

**Key Characteristics**:
- Files grow over time (logs, caches)
- Requires regular cleanup/rotation
- Often on separate partition (prevent filling root)
- High I/O activity
- Logs are critical for debugging

**Log Rotation Example**:
```bash
# /etc/logrotate.d/ml-api
/var/log/ml-api/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 mluser mluser
    sharedscripts
    postrotate
        systemctl reload ml-api
    endscript
}
```

**Monitoring Disk Usage**:
```bash
# Check /var disk usage
df -h /var

# Find large files in /var
du -h /var | sort -rh | head -10

# Clean old logs
sudo find /var/log -name "*.log" -mtime +30 -delete
```

## /usr - User System Resources

**Purpose**: Read-only user data and applications

**Contains**:
- System binaries and programs
- Libraries
- Documentation
- Header files
- Shared data

**Structure**:
```
/usr/
├── bin/              # User commands
│   ├── python3
│   ├── docker
│   ├── git
│   └── vim
├── sbin/             # System administration commands
│   └── nginx
├── lib/              # Libraries for /usr/bin and /usr/sbin
│   ├── python3.8/
│   └── x86_64-linux-gnu/
├── local/            # Locally installed software
│   ├── bin/
│   ├── lib/
│   └── include/
├── share/            # Architecture-independent data
│   ├── doc/         # Documentation
│   ├── man/         # Manual pages
│   └── applications/
└── include/          # Header files
    ├── python3.8/
    └── cuda.h
```

**ML Infrastructure Examples**:
```bash
# Python binaries
/usr/bin/python3
/usr/bin/pip3

# CUDA libraries
/usr/local/cuda/bin/nvcc
/usr/local/cuda/lib64/libcudart.so

# Custom ML tools
/usr/local/bin/ml-train
/usr/local/bin/ml-serve

# Python libraries
/usr/lib/python3.8/
/usr/local/lib/python3.8/dist-packages/

# Shared ML models
/usr/share/ml-models/pretrained/

# Development headers
/usr/include/python3.8/
/usr/include/cudnn.h
```

**Key Subdirectories**:

### /usr/bin - User Binaries
```bash
$ ls /usr/bin | grep -E 'python|docker|git'
docker
git
python3
python3.8
```

### /usr/lib - Libraries
```bash
$ ls /usr/lib/python3.8/
collections/
json/
os.py
sys.py
```

### /usr/local - Locally Installed Software
```bash
# Manually installed software goes here
/usr/local/bin/custom-ml-tool
/usr/local/lib/libcustom.so

# Often used for:
# - Software compiled from source
# - Third-party applications
# - Custom organization tools
```

### /usr/share - Shared Data
```bash
/usr/share/doc/python3.8/          # Documentation
/usr/share/man/man1/python3.gz     # Manual pages
```

**Key Characteristics**:
- Shareable between systems
- Read-only in production
- Large directory (applications + libraries)
- Rarely modified during runtime
- Can be mounted read-only for security

## Comparison Table

| Directory | Purpose | Writable by Users | Changes Frequently | Example Contents |
|-----------|---------|-------------------|-------------------|------------------|
| `/home` | User personal files | Yes (own dir) | Yes | Projects, datasets, configs |
| `/etc` | System configuration | No (sudo only) | Occasionally | nginx.conf, ssh_config |
| `/var` | Variable data | No (sudo only) | Yes | Logs, cache, temp files |
| `/usr` | User system resources | No (sudo only) | Rarely | Binaries, libraries |

## ML Infrastructure Best Practices

### 1. Separate Data Storage
```bash
# Don't store large datasets in /home
# Use separate mounted storage
/mnt/datasets/imagenet/
/mnt/models/production/

# Link to user directory
ln -s /mnt/datasets ~/datasets
```

### 2. Configuration Management
```bash
# Track /etc configs in Git (without secrets)
/etc/nginx/sites-available/
/etc/systemd/system/ml-api.service

# Use configuration management tools
ansible-playbook -i inventory setup-ml-server.yml
```

### 3. Log Monitoring
```bash
# Monitor /var/log for issues
tail -f /var/log/ml-api/training.log

# Set up log aggregation
# Ship logs to centralized logging (ELK, Splunk)
```

### 4. Disk Space Management
```bash
# Monitor critical directories
df -h /home /var /usr

# Clean up regularly
# - /var/log: Old logs
# - /var/cache: Old caches
# - /tmp: Temporary files
```

### 5. Application Deployment
```bash
# Install system-wide applications to /usr/local
sudo cp ml-train /usr/local/bin/
sudo chmod +x /usr/local/bin/ml-train

# User-specific applications in ~/bin
mkdir -p ~/bin
cp my-script.sh ~/bin/
export PATH="$HOME/bin:$PATH"
```

## Complete Example: ML API Deployment

```bash
# 1. Application binary
/usr/local/bin/ml-api

# 2. Configuration
/etc/ml-api/config.yaml
/etc/nginx/sites-available/ml-api
/etc/systemd/system/ml-api.service

# 3. Variable data
/var/log/ml-api/access.log
/var/log/ml-api/error.log
/var/lib/ml-api/cache/

# 4. User data
/home/mluser/models/
/home/mluser/experiments/

# 5. Shared storage
/mnt/datasets/  (mounted external storage)
/mnt/models/    (mounted external storage)
```

### Summary

- **`/home`**: User personal files and projects - full user control
- **`/etc`**: System-wide configuration files - requires sudo
- **`/var`**: Frequently changing data (logs, cache) - grows over time
- **`/usr`**: System binaries and libraries - rarely modified

Understanding these directories is crucial for:
- Proper application deployment
- System configuration
- Log monitoring and debugging
- Resource management
- Security and permissions

---

## Question 3: How can symbolic links help manage large datasets in ML projects?

### Answer

Symbolic links (symlinks) are a powerful tool for managing large datasets in ML projects. They provide a way to reference files without duplicating data, which is critical when working with large datasets that can range from gigabytes to terabytes.

## What Are Symbolic Links?

A **symbolic link** is a special file that contains a reference (path) to another file or directory.

**Analogy**: Think of a symlink as a shortcut on Windows or an alias on macOS - it points to another location without copying the actual data.

**Syntax**:
```bash
ln -s SOURCE_PATH LINK_PATH
```

**Example**:
```bash
# Create symlink to large dataset
ln -s /mnt/storage/imagenet ~/projects/ml-classifier/data/imagenet

# Now you can access the dataset as if it's in your project
cd ~/projects/ml-classifier/data
ls -l imagenet  # Shows it's a symlink
cat imagenet/labels.txt  # Accesses the original file
```

**Visual Representation**:
```
Project Structure          Actual Storage
================           ==============
~/projects/ml-classifier/
├── data/
│   ├── imagenet/ ───────→ /mnt/storage/imagenet/ (50GB)
│   │   (symlink)          ├── train/
│   │                      ├── val/
│   │                      └── labels.txt
│   └── processed/
└── models/

Disk Usage:
- Project: ~100MB (code only)
- Symlink: ~4KB (just the link)
- Original dataset: 50GB (unchanged)
Total: ~100MB instead of ~50GB
```

## Benefits for ML Projects

### 1. Disk Space Savings

**Problem**: ML datasets are huge and duplicating them wastes disk space.

```bash
# Without symlinks - wasting space
/home/user/project1/data/imagenet/  # 150GB
/home/user/project2/data/imagenet/  # 150GB (duplicate!)
/home/user/project3/data/imagenet/  # 150GB (duplicate!)
# Total: 450GB for the same data

# With symlinks - efficient storage
/mnt/shared-storage/datasets/imagenet/  # 150GB (only copy)
/home/user/project1/data/imagenet -> /mnt/shared-storage/datasets/imagenet/  # 4KB
/home/user/project2/data/imagenet -> /mnt/shared-storage/datasets/imagenet/  # 4KB
/home/user/project3/data/imagenet -> /mnt/shared-storage/datasets/imagenet/  # 4KB
# Total: 150GB + 12KB (97% space savings!)
```

**Real-World Example**:
```bash
# Dataset repository
/data/datasets/
├── imagenet/          # 150GB
├── coco/             # 25GB
├── cityscapes/       # 11GB
├── ade20k/           # 4GB
└── openimages/       # 500GB

# Project 1: Image classification
cd ~/projects/image-classification
ln -s /data/datasets/imagenet data/imagenet
ln -s /data/datasets/coco data/coco

# Project 2: Object detection
cd ~/projects/object-detection
ln -s /data/datasets/coco data/coco
ln -s /data/datasets/openimages data/openimages

# Project 3: Segmentation
cd ~/projects/segmentation
ln -s /data/datasets/cityscapes data/cityscapes
ln -s /data/datasets/ade20k data/ade20k

# Result: Multiple projects share datasets without duplication
```

### 2. Centralized Data Management

**Benefit**: Single source of truth for datasets

```bash
# Central dataset repository
/data/datasets/
├── imagenet/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── metadata/
├── README.md
├── CHANGELOG.md
└── scripts/
    ├── download.sh
    └── validate.sh

# All projects link to central repository
/home/alice/projects/resnet/data/imagenet -> /data/datasets/imagenet
/home/bob/projects/efficientnet/data/imagenet -> /data/datasets/imagenet
/home/charlie/projects/vit/data/imagenet -> /data/datasets/imagenet

# Benefits:
# 1. Update dataset once, all projects see changes
# 2. Consistency across all projects
# 3. Easy version control for datasets
# 4. Single backup point
```

**Update Propagation**:
```bash
# Update dataset in central location
cd /data/datasets/imagenet
./scripts/add-new-classes.sh

# All linked projects automatically see the updates
cd /home/alice/projects/resnet/data/imagenet
ls  # Shows new classes without any action
```

### 3. Easy Dataset Switching

**Benefit**: Switch between dataset versions without moving files

```bash
# Multiple dataset versions
/data/datasets/
├── imagenet-2012/
├── imagenet-2017/
└── imagenet-2021/

# Project links to current version
cd ~/projects/ml-classifier
ln -s /data/datasets/imagenet-2021 data/imagenet

# Switch to different version (for comparison)
rm data/imagenet
ln -s /data/datasets/imagenet-2017 data/imagenet

# Or use environment-specific links
ln -s /data/datasets/imagenet-2021 data/imagenet-prod
ln -s /data/datasets/imagenet-2017 data/imagenet-test

# Training script can switch based on environment
python train.py --data data/imagenet-${ENV}
```

### 4. Separate Storage Mounts

**Benefit**: Keep datasets on fast/large storage, project code on SSD

```bash
# System setup
/home/user/           # SSD (fast, limited space)
/mnt/hdd-storage/    # HDD (slow, large capacity)
/mnt/ssd-cache/      # SSD (fast, limited space)

# Project structure on SSD
~/projects/ml-classifier/
├── src/              # Code (small, needs fast access)
├── models/           # Models (medium, needs fast access)
└── data/
    ├── processed/    # Processed data on SSD (fast access)
    └── raw/ -> /mnt/hdd-storage/datasets/imagenet/  # Raw data on HDD

# Benefits:
# - Code runs fast (SSD)
# - Processed data cached on SSD
# - Raw data on cheap HDD storage
# - Flexible storage allocation
```

**Multi-Tier Storage Strategy**:
```bash
# Tier 1: NVMe SSD (fastest, most expensive)
/mnt/nvme-cache/
├── active-experiments/  # Currently training
└── preprocessed-data/   # Frequently accessed

# Tier 2: SATA SSD (fast, moderate cost)
/home/user/projects/     # Project code and small files

# Tier 3: HDD (slow, cheapest)
/mnt/hdd-storage/
├── raw-datasets/        # Original datasets
└── archived-models/     # Old model checkpoints

# Tier 4: Network storage (slowest, unlimited)
/mnt/nfs-storage/
└── backups/

# Project uses symlinks to access appropriate tier
cd ~/projects/ml-classifier
ln -s /mnt/nvme-cache/preprocessed-data data/processed
ln -s /mnt/hdd-storage/raw-datasets/imagenet data/raw
```

### 5. Collaborative Workflows

**Benefit**: Team members share datasets without duplication

```bash
# Shared team dataset repository
/data/team-datasets/
├── imagenet/
├── coco/
└── custom-dataset/

# Each team member links to shared datasets
# Alice
ln -s /data/team-datasets/imagenet ~/alice-project/data/imagenet

# Bob
ln -s /data/team-datasets/imagenet ~/bob-project/data/imagenet

# Charlie
ln -s /data/team-datasets/imagenet ~/charlie-project/data/imagenet

# Advantages:
# - No data duplication across team
# - Consistent dataset version
# - Reduced download bandwidth
# - Faster onboarding (no long downloads)
```

### 6. Git-Friendly Project Structure

**Benefit**: Track project structure without large data files

```bash
# Project structure in Git
ml-classifier/
├── .gitignore
├── data/
│   ├── .gitkeep           # Track directory structure
│   └── imagenet/ -> /data/datasets/imagenet/  # Symlink (tracked)
├── models/
└── src/

# .gitignore
data/*           # Ignore data contents
!data/.gitkeep   # But track directory
!data/README.md  # And documentation

# Clone repository
git clone https://github.com/user/ml-classifier.git
cd ml-classifier

# Setup script creates symlinks
./scripts/setup-data.sh

# setup-data.sh
#!/bin/bash
SHARED_DATA="/data/datasets"
ln -s "$SHARED_DATA/imagenet" data/imagenet
ln -s "$SHARED_DATA/coco" data/coco
```

### 7. Backup Efficiency

**Benefit**: Exclude large datasets from backups

```bash
# Backup project without datasets
tar --exclude='data/*' -czf project-backup.tar.gz ml-classifier/

# Or use backup script
#!/bin/bash
rsync -av \
  --exclude='data/' \
  --exclude='models/checkpoints/' \
  ~/projects/ml-classifier/ \
  /backup/ml-classifier/

# Datasets backed up separately (once)
rsync -av /data/datasets/ /backup/datasets/

# Benefits:
# - Faster backups
# - Less backup storage
# - Backup datasets once, not per-project
```

## Practical Examples

### Example 1: Multi-Project Dataset Sharing

```bash
# Setup shared dataset repository
sudo mkdir -p /data/datasets
sudo chown $USER:$USER /data/datasets

# Download ImageNet once
cd /data/datasets
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
tar -xf ILSVRC2012_img_train.tar
mv ILSVRC2012 imagenet

# Project 1: ResNet training
cd ~/projects/resnet-training
mkdir -p data
ln -s /data/datasets/imagenet data/imagenet

# Project 2: EfficientNet comparison
cd ~/projects/efficientnet-comparison
mkdir -p data
ln -s /data/datasets/imagenet data/imagenet

# Project 3: Transfer learning
cd ~/projects/transfer-learning
mkdir -p data
ln -s /data/datasets/imagenet data/imagenet

# All projects use same dataset (150GB saved!)
```

### Example 2: Dataset Version Management

```bash
# Maintain multiple dataset versions
/data/datasets/
├── imagenet-v1/       # Original version
├── imagenet-v2/       # Updated labels
└── imagenet-v3/       # Additional classes

# Project uses symbolic link to switch versions
cd ~/projects/ml-classifier
ln -s /data/datasets/imagenet-v3 data/imagenet

# Experiment with different version
rm data/imagenet
ln -s /data/datasets/imagenet-v2 data/imagenet
python train.py

# Easy A/B testing between versions
for version in v1 v2 v3; do
    rm data/imagenet
    ln -s /data/datasets/imagenet-$version data/imagenet
    python train.py --experiment imagenet-$version
done
```

### Example 3: Preprocessing Pipeline

```bash
# Raw data on slow HDD
/mnt/hdd/datasets/imagenet-raw/  # 150GB

# Preprocessed data on fast SSD
/mnt/ssd/datasets/imagenet-processed/  # 50GB (compressed)

# Project structure
~/projects/ml-classifier/
├── data/
│   ├── raw/ -> /mnt/hdd/datasets/imagenet-raw/
│   └── processed/ -> /mnt/ssd/datasets/imagenet-processed/
└── scripts/
    └── preprocess.py

# Preprocessing script reads from HDD, writes to SSD
python scripts/preprocess.py \
    --input data/raw \
    --output data/processed

# Training reads from fast SSD
python train.py --data data/processed
```

### Example 4: Cloud Storage Integration

```bash
# Mount cloud storage (S3, Google Cloud Storage)
mkdir -p /mnt/s3-datasets
s3fs my-ml-datasets /mnt/s3-datasets \
    -o passwd_file=~/.s3fs_credentials \
    -o url=https://s3.amazonaws.com

# Link to cloud storage
cd ~/projects/ml-classifier
ln -s /mnt/s3-datasets/imagenet data/imagenet

# Benefits:
# - No local storage needed
# - Access datasets from anywhere
# - Automatic scaling
# - Pay-per-use storage
```

## Best Practices

### 1. Document Symlink Structure

```bash
# data/README.md
# Dataset Links

This project uses symbolic links to access shared datasets.

## Required Datasets

- `imagenet`: ImageNet 2012 classification dataset
  - Expected location: `/data/datasets/imagenet`
  - Size: ~150GB
  - Create link: `ln -s /data/datasets/imagenet data/imagenet`

- `coco`: COCO 2017 object detection dataset
  - Expected location: `/data/datasets/coco`
  - Size: ~25GB
  - Create link: `ln -s /data/datasets/coco data/coco`

## Setup Script

Run `./scripts/setup-data.sh` to create all required symlinks.
```

### 2. Automated Setup Script

```bash
#!/bin/bash
# scripts/setup-data.sh

set -e

SHARED_DATA="/data/datasets"

# Check if shared data exists
if [[ ! -d "$SHARED_DATA" ]]; then
    echo "Error: Shared dataset directory $SHARED_DATA not found"
    exit 1
fi

# Create data directory
mkdir -p data

# Create symlinks
datasets=("imagenet" "coco" "cityscapes")

for dataset in "${datasets[@]}"; do
    source="$SHARED_DATA/$dataset"
    target="data/$dataset"

    if [[ ! -d "$source" ]]; then
        echo "Warning: Dataset $dataset not found at $source"
        continue
    fi

    if [[ -L "$target" ]]; then
        echo "Link already exists: $target"
    else
        ln -s "$source" "$target"
        echo "Created link: $target -> $source"
    fi
done

echo "Dataset setup complete!"
```

### 3. Verify Symlink Integrity

```bash
#!/bin/bash
# scripts/verify-data.sh

# Check if symlinks are valid
for link in data/*; do
    if [[ -L "$link" ]]; then
        target=$(readlink "$link")
        if [[ -e "$target" ]]; then
            echo "✓ $link -> $target (OK)"
        else
            echo "✗ $link -> $target (BROKEN)"
        fi
    fi
done
```

### 4. Handle Broken Symlinks

```bash
# Find broken symlinks
find data/ -xtype l

# Remove broken symlinks
find data/ -xtype l -delete

# Fix broken symlink
rm data/imagenet  # Remove broken link
ln -s /new/path/to/imagenet data/imagenet  # Create new link
```

## Common Pitfalls and Solutions

### Pitfall 1: Symlinks in Archives

```bash
# Problem: tar archives symlink as link (not content)
tar -czf backup.tar.gz ml-classifier/

# Solution: Dereference symlinks (-h flag)
tar -czfh backup.tar.gz ml-classifier/
# This archives the actual data (makes archive large)

# Better: Exclude symlinked data
tar --exclude='data/*' -czf backup.tar.gz ml-classifier/
```

### Pitfall 2: Moving Linked Projects

```bash
# Problem: Absolute symlinks break when project moves
/home/alice/project/data/imagenet -> /data/datasets/imagenet  # Absolute

# Moving project breaks link
mv /home/alice/project /home/bob/project
ls /home/bob/project/data/imagenet  # Still points to /data/datasets/imagenet (OK)

# But if dataset path changes...
mv /data/datasets /data/shared-datasets
ls /home/bob/project/data/imagenet  # BROKEN

# Solution: Use relative paths where possible
# Or document external dependencies
```

### Pitfall 3: Git Tracking Symlinks

```bash
# Problem: Git tracks symlink itself, not target
git add data/imagenet  # Adds symlink (not data)

# On clone, symlink may be broken
git clone repo.git
cd repo
ls data/imagenet  # May not exist on new machine

# Solution: Document setup in README
# Use setup script to create symlinks after clone
```

## Summary

Symbolic links are essential for efficient ML dataset management:

**Key Benefits**:
1. **Space savings**: No data duplication
2. **Centralization**: Single source of truth
3. **Flexibility**: Easy version switching
4. **Performance**: Optimize storage tiers
5. **Collaboration**: Share datasets across team
6. **Git-friendly**: Track structure, not data
7. **Backup efficiency**: Backup once, use everywhere

**Best Practices**:
- Document symlink structure in README
- Create automated setup scripts
- Verify symlink integrity before training
- Use absolute paths for stable references
- Exclude symlinked data from backups
- Plan storage hierarchy for performance

**Common Pattern**:
```bash
# One copy of data
/data/datasets/imagenet/  # 150GB

# Many projects linking to it
~/project-1/data/imagenet -> /data/datasets/imagenet  # 4KB
~/project-2/data/imagenet -> /data/datasets/imagenet  # 4KB
~/project-3/data/imagenet -> /data/datasets/imagenet  # 4KB
# Total: 150GB instead of 450GB (67% savings)
```

Symbolic links transform how we manage large ML datasets, making multi-project workflows practical and efficient.

---

## Question 4: Why is it important to separate data, models, and source code in ML projects?

### Answer

Separating data, models, and source code is a fundamental organizational principle for professional ML projects. This separation provides numerous benefits for development, deployment, collaboration, and maintenance.

## Core Principle: Separation of Concerns

**Definition**: Different types of assets should be organized separately based on their characteristics, lifecycle, and management requirements.

**Three Main Categories**:
1. **Source Code**: Logic, algorithms, scripts
2. **Data**: Training datasets, validation sets, test sets
3. **Models**: Trained model artifacts, checkpoints, weights

## Why Separate? The Reasons

### 1. Different Lifecycles and Update Frequencies

**Source Code**:
- Changes frequently during development
- Updated with bug fixes, features
- Version controlled with Git
- Small size (KB to MB)

**Data**:
- Changes infrequently (new collection, updates)
- Large size (GB to TB)
- Not practical for Git
- Requires special versioning (DVC, LakeFS)

**Models**:
- Generated during training
- Medium to large size (MB to GB)
- Versioned with model registry
- Multiple versions coexist

**Example Timeline**:
```
Week 1:
  Code: 20 commits
  Data: 0 changes
  Models: 5 new versions

Week 2:
  Code: 15 commits
  Data: 0 changes
  Models: 8 new versions

Month 2:
  Code: 100 commits
  Data: 1 update (new collection)
  Models: 50 new versions
```

If everything was mixed together, every code commit would involve GB of data!

### 2. Version Control Efficiency

**Git is Optimized for Code**:
- Text files
- Line-by-line diffs
- Small file sizes
- Frequent updates

**Git is NOT Optimized for Data/Models**:
- Binary files
- No meaningful diffs
- Large file sizes
- Can bloat repository

**Example - Bad Approach**:
```bash
# DON'T DO THIS
git add data/imagenet/  # 150GB added to Git
git commit -m "Added dataset"
# Git repository now 150GB
# Clone takes hours
# Every developer needs 150GB locally
```

**Example - Good Approach**:
```bash
# Separate storage
Code: Git (100MB)
Data: S3/DVC (150GB, versioned separately)
Models: Model registry (5GB, versioned separately)

# Clone repository (fast)
git clone repo.git  # Only 100MB

# Download data separately (when needed)
dvc pull  # Downloads only required datasets

# Download model (when needed)
mlflow models download model-uri
```

### 3. Storage Optimization

**Different storage requirements**:

**Source Code**:
- Small size
- Needs fast access
- On SSD
- Backed up frequently
- Git remote (GitHub, GitLab)

**Data**:
- Large size
- Can be on slower storage
- On HDD or network storage
- Backed up less frequently
- S3, Google Cloud Storage, Azure Blob

**Models**:
- Medium to large size
- Frequently accessed during serving
- On SSD for inference
- Archived old versions
- Model registry (MLflow, Weights & Biases)

**Cost Comparison**:
```
Option 1: Everything on fast SSD
  - 150GB dataset on SSD: $150/month
  - 5GB models on SSD: $5/month
  - 100MB code on SSD: $0.10/month
  Total: $155.10/month

Option 2: Separated by type
  - 150GB dataset on HDD: $15/month (10x cheaper)
  - 5GB models on SSD: $5/month (fast access)
  - 100MB code on SSD: $0.10/month (version control)
  Total: $20.10/month (87% savings)
```

### 4. Collaboration and Sharing

**Scenario**: Team of 10 ML engineers

**Without Separation**:
```bash
# Each engineer clones everything
Developer 1: 150GB (code + data + models)
Developer 2: 150GB
...
Developer 10: 150GB
Total: 1,500GB storage
Total: 10x download bandwidth
```

**With Separation**:
```bash
# Each engineer clones code
Developer 1: 100MB code
Developer 2: 100MB code
...
Developer 10: 100MB code
Total: 1GB storage

# Everyone shares same data/models
Shared data: 150GB (accessed via network or symlinks)
Shared models: 5GB (accessed via model registry)
Total: 155GB storage (90% savings)
```

### 5. Access Control and Security

**Different security requirements**:

**Source Code**:
- Intellectual property
- May be open source
- Shared with developers
- Version controlled on GitHub

**Data**:
- May contain PII (Personally Identifiable Information)
- Subject to regulations (GDPR, HIPAA)
- Restricted access
- Encrypted storage

**Models**:
- Proprietary algorithms
- Trade secrets
- May contain data leakage
- Access control required

**Example Access Matrix**:
```
                 Code    Data    Models
Developers       RW      R       R
Data Scientists  R       RW      RW
ML Ops           R       R       RW
Auditors         R       R       R
External         R       -       -
```

**Implementation**:
```bash
# Code: Public GitHub repository
https://github.com/company/ml-project

# Data: Private S3 bucket with encryption
s3://company-ml-data/datasets/
  - Encrypted at rest
  - Access logs enabled
  - IAM roles for access control

# Models: Private model registry
https://ml-models.company.com/
  - Authentication required
  - Audit trail
  - Version control
```

### 6. CI/CD and Deployment Efficiency

**Continuous Integration**:
```bash
# Without separation
git push
# Triggers CI: Downloads 150GB repo
# Runs tests
# Takes 2 hours (mostly download time)

# With separation
git push  # Only code (100MB)
# Triggers CI: Downloads 100MB
# Uses cached/mocked data for tests
# Runs tests
# Takes 10 minutes (20x faster)
```

**Deployment Pipeline**:
```bash
# Separate deployments for different assets

# Code deployment (frequent)
git push
# CI/CD deploys code to servers
# Takes 5 minutes

# Model deployment (occasional)
mlflow models deploy --model-uri models:/production/v1.2
# Deploys model artifact only
# Takes 10 minutes

# Data update (rare)
dvc push
# Updates data version
# Background sync to servers
# Takes 2 hours (doesn't block code deployment)
```

### 7. Reproducibility and Experiment Tracking

**Separated assets enable better tracking**:

```python
# experiment-config.yaml
experiment:
  name: "resnet50-imagenet"
  code:
    repo: "https://github.com/company/ml-project"
    commit: "a3f2d1b"  # Exact code version
  data:
    source: "s3://ml-data/imagenet-2021/"
    version: "v2.1"  # Exact data version
  model:
    architecture: "resnet50"
    checkpoint: "models://resnet50/v3.4"  # Exact model version

# Can reproduce experiment months later
# Fetch exact code version from Git
# Fetch exact data version from DVC
# Fetch exact model checkpoint from registry
```

**Experiment Comparison**:
```bash
# Compare experiments with different data versions
Experiment A:
  Code: commit abc123
  Data: imagenet-v1
  Model: 92.5% accuracy

Experiment B:
  Code: commit abc123 (same)
  Data: imagenet-v2 (updated labels)
  Model: 94.1% accuracy (improvement!)

# Separation makes it clear the improvement is from data, not code
```

### 8. Backup and Disaster Recovery

**Different backup strategies**:

**Code**:
- Backed up automatically (Git)
- Multiple remotes (GitHub, GitLab, Bitbucket)
- Instant recovery (git clone)
- Backup size: Small

**Data**:
- Backed up periodically
- S3 versioning enabled
- Replicated across regions
- Backup size: Large
- Recovery time: Hours to days

**Models**:
- Backed up after training
- Model registry with versioning
- Archived old versions
- Backup size: Medium
- Recovery time: Minutes to hours

**Disaster Recovery Plan**:
```bash
# Scenario: Server crashed, need to restore

# Step 1: Restore code (fast)
git clone https://github.com/company/ml-project
cd ml-project
# Time: 1 minute

# Step 2: Restore models (medium)
mlflow models download --model-uri models://production/latest
# Time: 10 minutes

# Step 3: Restore data (slow, but not urgent for serving)
dvc pull
# Time: 2 hours (background process)

# Production serving restored in 11 minutes
# Data restoration continues in background
```

### 9. Development Workflow Efficiency

**Local Development**:
```bash
# Developer workflow with separation

# 1. Clone code repository (fast)
git clone https://github.com/company/ml-project
cd ml-project

# 2. Link to shared data (instant)
ln -s /mnt/shared-data/imagenet data/imagenet

# 3. Download only needed models
mlflow models download --model-uri models://resnet50/v1.0

# 4. Start developing
vim src/training/train.py

# 5. Test with sample data
pytest tests/ --sample-data

# 6. Commit code changes
git add src/training/train.py
git commit -m "Improved training loop"
git push

# Benefits:
# - Started working in 5 minutes
# - No need to download 150GB dataset
# - Can use shared or sampled data for testing
# - Commits are fast (only code)
```

### 10. Scalability

**As project grows**:

**Code**:
- Grows slowly (maybe 10x over project lifetime)
- 100MB → 1GB

**Data**:
- Grows rapidly (new data collection)
- 150GB → 1.5TB (10x in a year)

**Models**:
- Many experiments, versions
- 5GB → 500GB (100 model versions)

**Without Separation**:
```bash
# Single repository
Year 1: 155GB
Year 2: 2,001.5GB (2TB)
# Git repository unusable
# Clone impossible
# Collaboration broken
```

**With Separation**:
```bash
# Separated assets
Code repo: 1GB (manageable)
Data storage: 1.5TB (on appropriate storage)
Model registry: 500GB (on model-specific storage)

# Each system scales independently
# Code repo remains usable
# Data uses scalable object storage
# Models use dedicated registry
```

## Standard ML Project Structure

```
ml-project/
├── README.md
├── .gitignore              # Excludes data/ and models/
├── requirements.txt
├── setup.py
│
├── data/                   # DATA (not in Git)
│   ├── .gitkeep           # Track directory structure
│   ├── README.md          # How to get data
│   ├── raw/               # Original, immutable data
│   ├── processed/         # Cleaned, transformed data
│   └── external/          # External datasets
│
├── models/                # MODELS (not in Git)
│   ├── .gitkeep
│   ├── README.md          # How to get models
│   ├── checkpoints/       # Training checkpoints
│   └── production/        # Production-ready models
│
├── src/                   # CODE (in Git)
│   ├── __init__.py
│   ├── preprocessing/     # Data preprocessing
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation scripts
│   └── utils/             # Utility functions
│
├── tests/                 # CODE (in Git)
│   ├── test_preprocessing.py
│   └── test_training.py
│
├── notebooks/             # CODE (in Git)
│   ├── exploratory/       # EDA notebooks
│   └── reports/           # Final analysis
│
├── configs/               # CODE (in Git)
│   ├── train_config.yaml
│   └── model_config.yaml
│
└── scripts/               # CODE (in Git)
    ├── download_data.sh
    └── train.sh
```

**.gitignore**:
```gitignore
# Exclude data
data/raw/*
data/processed/*
data/external/*
!data/.gitkeep
!data/README.md

# Exclude models
models/checkpoints/*
models/production/*
!models/.gitkeep
!models/README.md

# Include code
src/
tests/
notebooks/
configs/
scripts/
```

**data/README.md**:
```markdown
# Data

This directory contains datasets used for training and evaluation.

## Getting Data

### ImageNet 2012
```bash
# Download from official source
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
tar -xf ILSVRC2012_img_train.tar -C data/raw/imagenet/
```

### Using DVC
```bash
# Pull data from remote storage
dvc pull data/raw/imagenet.dvc
```

## Data Structure

- `raw/`: Original, immutable data
- `processed/`: Cleaned and transformed data ready for training
- `external/`: External datasets and references
```

## Real-World Example

**Project**: Image classification with ResNet50

**Bad Approach** (Everything Mixed):
```bash
ml-classifier-bad/
├── train.py
├── evaluate.py
├── imagenet-train/      # 150GB in Git!
├── imagenet-val/        # 10GB in Git!
├── model_epoch_1.pth    # 100MB in Git!
├── model_epoch_2.pth    # 100MB in Git!
├── ...
├── model_epoch_90.pth   # 100MB in Git!
└── config.yaml

# Git repository: 169GB
# Clone time: 6 hours
# Every commit involves GB of data
# Can't share with team (too large)
```

**Good Approach** (Separated):
```bash
# Code repository (Git)
ml-classifier/
├── src/
│   ├── train.py
│   └── evaluate.py
├── configs/
│   └── config.yaml
└── requirements.txt

# Size: 10MB
# Clone time: 5 seconds

# Data storage (S3 + DVC)
s3://ml-data/imagenet-2012/
├── train/      # 150GB
└── val/        # 10GB

# Version controlled with DVC
# Shared across team
# Downloaded on-demand

# Model registry (MLflow)
models://resnet50/
├── v1.0/      # Epoch 30, 91.2% acc
├── v2.0/      # Epoch 60, 92.5% acc
└── v3.0/      # Epoch 90, 93.1% acc (production)

# Each version: 100MB
# Versioned and tracked
# Easy rollback
```

## Best Practices

### 1. Document Data and Model Access

**README.md**:
```markdown
# ML Classifier Project

## Setup

### Get Code
```bash
git clone https://github.com/company/ml-classifier
cd ml-classifier
```

### Get Data
```bash
# Option 1: Download from S3
aws s3 sync s3://company-ml-data/imagenet data/raw/imagenet

# Option 2: Use DVC
dvc pull

# Option 3: Use symlink to shared storage
ln -s /mnt/shared-data/imagenet data/raw/imagenet
```

### Get Models
```bash
# Download production model
mlflow models download \
    --model-uri models://resnet50/production \
    --dst models/production/resnet50.pth
```
```

### 2. Use Data Version Control (DVC)

```bash
# Initialize DVC
dvc init

# Track data
dvc add data/raw/imagenet
git add data/raw/imagenet.dvc .gitignore
git commit -m "Track ImageNet dataset"

# Configure remote storage
dvc remote add -d storage s3://company-ml-data

# Push data to remote
dvc push

# Team members can pull
dvc pull
```

### 3. Use Model Registry

```python
# Log model to registry during training
import mlflow

with mlflow.start_run():
    # Train model
    model = train(data)

    # Log parameters
    mlflow.log_params(config)

    # Log metrics
    mlflow.log_metrics({"accuracy": accuracy, "loss": loss})

    # Log model
    mlflow.pytorch.log_model(model, "model")

# Promote to production
mlflow models transition-to-production --model-name resnet50 --version 3
```

### 4. Automate Setup

**scripts/setup.sh**:
```bash
#!/bin/bash
set -e

echo "Setting up ML project..."

# Create data directories
mkdir -p data/raw data/processed data/external

# Link to shared data
if [[ -d /mnt/shared-data ]]; then
    ln -sf /mnt/shared-data/imagenet data/raw/imagenet
    echo "✓ Linked to shared data"
fi

# Pull data with DVC
if command -v dvc &>/dev/null; then
    dvc pull
    echo "✓ Pulled data with DVC"
fi

# Download models
if command -v mlflow &>/dev/null; then
    mlflow models download \
        --model-uri models://resnet50/production \
        --dst models/production/
    echo "✓ Downloaded production model"
fi

echo "Setup complete!"
```

## Summary

Separating data, models, and source code is critical for:

1. **Version Control Efficiency**: Keep Git fast and usable
2. **Storage Optimization**: Use appropriate storage for each asset type
3. **Collaboration**: Share code without duplicating large files
4. **Access Control**: Different security requirements
5. **CI/CD Performance**: Fast, focused deployments
6. **Reproducibility**: Track versions independently
7. **Backup Strategy**: Different backup requirements
8. **Development Workflow**: Start working quickly
9. **Scalability**: Each asset scales independently
10. **Cost Efficiency**: Optimize costs per asset type

**Key Principle**: Treat data, models, and code as first-class citizens with their own lifecycles, storage, and versioning strategies.

---

## Question 5: What commands would you use to find all Python files modified in the last 7 days?

### Answer

There are multiple approaches to finding Python files modified in the last 7 days, each with different capabilities and use cases.

## Method 1: Using `find` (Most Common)

### Basic Syntax
```bash
find . -name "*.py" -mtime -7
```

**Explanation**:
- `find`: Search for files
- `.`: Start from current directory
- `-name "*.py"`: Match files ending with .py
- `-mtime -7`: Modified within last 7 days

**Output Example**:
```
./src/training/train.py
./src/evaluation/evaluate.py
./tests/test_metrics.py
./scripts/preprocess.py
```

### Detailed Variations

**1. Search specific directory**:
```bash
find /home/user/projects -name "*.py" -mtime -7
```

**2. Case-insensitive search** (matches .py, .PY, .Py):
```bash
find . -iname "*.py" -mtime -7
```

**3. Only search for files** (exclude directories):
```bash
find . -type f -name "*.py" -mtime -7
```

**4. Show detailed information** (size, permissions, timestamps):
```bash
find . -type f -name "*.py" -mtime -7 -ls
```

**Output**:
```
1234567   24 -rw-r--r--   1 user user   24576 Jan 31 14:30 ./src/training/train.py
1234568   12 -rw-r--r--   1 user user   12288 Jan 30 09:15 ./tests/test_metrics.py
```

**5. Show with human-readable timestamps**:
```bash
find . -type f -name "*.py" -mtime -7 -printf "%TY-%Tm-%Td %TH:%TM %p\n"
```

**Output**:
```
2025-01-31 14:30 ./src/training/train.py
2025-01-30 09:15 ./tests/test_metrics.py
2025-01-29 16:45 ./scripts/preprocess.py
```

### Time Options Explained

**`-mtime` (Modified Time)**:
```bash
# Last 7 days
find . -name "*.py" -mtime -7

# Exactly 7 days ago
find . -name "*.py" -mtime 7

# More than 7 days ago
find . -name "*.py" -mtime +7
```

**`-mmin` (Modified Minutes)**:
```bash
# Last 60 minutes (1 hour)
find . -name "*.py" -mmin -60

# Last 24 hours
find . -name "*.py" -mmin -1440
```

**`-newer` (Modified After Specific File)**:
```bash
# Files modified after reference file
touch reference_date.txt
find . -name "*.py" -newer reference_date.txt
```

**`-newermt` (Modified After Specific Date)**:
```bash
# Files modified after specific date/time
find . -name "*.py" -newermt "2025-01-24"
find . -name "*.py" -newermt "2025-01-24 14:30:00"
find . -name "*.py" -newermt "7 days ago"
find . -name "*.py" -newermt "1 week ago"
```

### Actions on Found Files

**1. List with `ls` format**:
```bash
find . -name "*.py" -mtime -7 -exec ls -lh {} \;
```

**2. Count lines of code**:
```bash
find . -name "*.py" -mtime -7 -exec wc -l {} +
```

**Output**:
```
   324 ./src/training/train.py
   187 ./tests/test_metrics.py
    95 ./scripts/preprocess.py
   606 total
```

**3. Search for specific content**:
```bash
find . -name "*.py" -mtime -7 -exec grep -l "def train" {} \;
```

**4. Copy to backup directory**:
```bash
find . -name "*.py" -mtime -7 -exec cp {} /backup/recent-changes/ \;
```

**5. Create archive**:
```bash
find . -name "*.py" -mtime -7 -print0 | tar -czf recent-changes.tar.gz --null -T -
```

## Method 2: Using `ls` with Time Sorting

### Basic Syntax
```bash
ls -lt --time-style=long-iso **/*.py | awk -v date=$(date -d '7 days ago' +%Y-%m-%d) '$6 >= date'
```

**More Practical Approach**:
```bash
# Show all Python files sorted by modification time
ls -lt $(find . -name "*.py")

# Filter with grep and date
find . -name "*.py" -mtime -7 -exec ls -lt {} +
```

## Method 3: Using `stat` for Precise Timestamps

```bash
find . -name "*.py" -type f -exec stat --format='%Y %n' {} + | \
    awk -v cutoff=$(date -d '7 days ago' +%s) '$1 >= cutoff {print $2}'
```

**Explanation**:
- `stat --format='%Y %n'`: Timestamp and filename
- `date -d '7 days ago' +%s`: Cutoff timestamp
- `awk`: Filter files newer than cutoff

## Method 4: Using `git` (For Git Repositories)

### Modified Files in Git

```bash
# Files modified in last 7 days (committed)
git log --since="7 days ago" --name-only --pretty=format: | \
    grep "\.py$" | sort -u
```

**Output**:
```
src/training/train.py
tests/test_metrics.py
src/evaluation/evaluate.py
```

### Including Uncommitted Changes

```bash
# Recently committed files
git log --since="7 days ago" --name-only --pretty=format: | grep "\.py$" | sort -u

# Plus uncommitted changes
git status --short | awk '{print $2}' | grep "\.py$"
```

### With Author Filter

```bash
# Files modified by specific author in last 7 days
git log --since="7 days ago" --author="John Doe" --name-only --pretty=format: | \
    grep "\.py$" | sort -u
```

## Method 5: Combining with Other Filters

### Size Filter

```bash
# Python files modified in last 7 days, larger than 10KB
find . -name "*.py" -mtime -7 -size +10k
```

### Exclude Directories

```bash
# Exclude __pycache__ and virtual environments
find . -name "*.py" -mtime -7 \
    -not -path "*/__pycache__/*" \
    -not -path "*/venv/*" \
    -not -path "*/.venv/*"
```

### Multiple Extensions

```bash
# Python and Jupyter notebook files
find . \( -name "*.py" -o -name "*.ipynb" \) -mtime -7
```

### Permission Filter

```bash
# Executable Python files modified in last 7 days
find . -name "*.py" -mtime -7 -executable
```

## Practical ML Workflow Examples

### Example 1: Review Recent Code Changes

```bash
#!/bin/bash
# Show Python files modified in last 7 days with details

echo "Python files modified in last 7 days:"
echo "======================================"
echo ""

find . -type f -name "*.py" -mtime -7 \
    -not -path "*/__pycache__/*" \
    -not -path "*/venv/*" \
    -exec ls -lh --time-style=long-iso {} \; | \
    sort -k6,7 -r | \
    awk '{printf "%-20s %-10s %s %s %s\n", $7, $5, $6, $7, $8}'
```

**Output**:
```
train.py             24K        2025-01-31 14:30
test_metrics.py      12K        2025-01-30 09:15
preprocess.py        8K         2025-01-29 16:45
```

### Example 2: Code Review Assistant

```bash
#!/bin/bash
# Generate report of recent changes

DAYS=7
OUTPUT="recent_changes_report.md"

echo "# Code Changes Report (Last $DAYS Days)" > "$OUTPUT"
echo "" >> "$OUTPUT"
echo "Generated: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

echo "## Modified Python Files" >> "$OUTPUT"
echo "" >> "$OUTPUT"

find . -type f -name "*.py" -mtime -$DAYS \
    -not -path "*/__pycache__/*" \
    -not -path "*/venv/*" | \
while read file; do
    echo "### $file" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    echo "- **Modified**: $(stat -c %y "$file")" >> "$OUTPUT"
    echo "- **Size**: $(du -h "$file" | cut -f1)" >> "$OUTPUT"
    echo "- **Lines**: $(wc -l < "$file")" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

echo "Report generated: $OUTPUT"
```

### Example 3: Find Files for Testing

```bash
# Find recently modified test files
find tests/ -name "test_*.py" -mtime -7

# Find recently modified source files and their corresponding tests
for file in $(find src/ -name "*.py" -mtime -7); do
    basename=$(basename "$file" .py)
    test_file="tests/test_${basename}.py"
    if [[ -f "$test_file" ]]; then
        echo "Source: $file -> Test: $test_file"
    else
        echo "Source: $file -> Test: MISSING"
    fi
done
```

### Example 4: Backup Recent Changes

```bash
#!/bin/bash
# Backup Python files modified in last 7 days

BACKUP_DIR="backup_$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

echo "Backing up recently modified Python files..."

find . -type f -name "*.py" -mtime -7 \
    -not -path "*/__pycache__/*" \
    -not -path "*/venv/*" | \
while read file; do
    # Preserve directory structure
    dir=$(dirname "$file")
    mkdir -p "$BACKUP_DIR/$dir"
    cp "$file" "$BACKUP_DIR/$file"
    echo "Backed up: $file"
done

# Create archive
tar -czf "${BACKUP_DIR}.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: ${BACKUP_DIR}.tar.gz"
```

### Example 5: CI/CD Integration

```bash
# Run tests only on files modified in last 7 days
MODIFIED_FILES=$(find src/ tests/ -name "*.py" -mtime -7)

if [[ -n "$MODIFIED_FILES" ]]; then
    echo "Running tests on recently modified files:"
    echo "$MODIFIED_FILES"

    # Run pytest on modified files
    pytest $MODIFIED_FILES -v
else
    echo "No Python files modified in last 7 days"
fi
```

## Performance Considerations

### Large Codebases

**Slow**:
```bash
# Searches entire filesystem (can take minutes)
find / -name "*.py" -mtime -7
```

**Fast**:
```bash
# Restrict search scope
find ~/projects/ml-classifier -name "*.py" -mtime -7

# Or use locate (pre-indexed)
locate "*.py" | while read file; do
    if [[ $(find "$file" -mtime -7 2>/dev/null) ]]; then
        echo "$file"
    fi
done
```

### Excluding Large Directories

```bash
# Exclude node_modules, venv, build directories
find . -name "*.py" -mtime -7 \
    -not -path "*/node_modules/*" \
    -not -path "*/venv/*" \
    -not -path "*/.venv/*" \
    -not -path "*/build/*" \
    -not -path "*/dist/*" \
    -not -path "*/__pycache__/*"
```

## Combining with Other Tools

### With `grep` for Content Search

```bash
# Find Python files modified in last 7 days containing "train"
find . -name "*.py" -mtime -7 -exec grep -l "def train" {} \;
```

### With `wc` for Statistics

```bash
# Count lines in recently modified files
find . -name "*.py" -mtime -7 -exec wc -l {} + | sort -rn
```

### With `diff` for Comparison

```bash
# Compare with backed up versions
find . -name "*.py" -mtime -7 | while read file; do
    if [[ -f "backup/$file" ]]; then
        echo "Changes in $file:"
        diff "backup/$file" "$file"
    fi
done
```

## Common Use Cases Summary

| Use Case | Command |
|----------|---------|
| Basic search | `find . -name "*.py" -mtime -7` |
| With details | `find . -name "*.py" -mtime -7 -ls` |
| Exclude dirs | `find . -name "*.py" -mtime -7 -not -path "*/venv/*"` |
| Count lines | `find . -name "*.py" -mtime -7 -exec wc -l {} +` |
| Git changes | `git log --since="7 days ago" --name-only --pretty=format: | grep "\.py$"` |
| Last hour | `find . -name "*.py" -mmin -60` |
| Specific date | `find . -name "*.py" -newermt "2025-01-24"` |
| Create backup | `find . -name "*.py" -mtime -7 | tar -czf backup.tar.gz -T -` |

## Best Practices

1. **Always specify file type** to avoid matching directories:
   ```bash
   find . -type f -name "*.py" -mtime -7
   ```

2. **Exclude cache directories** to reduce noise:
   ```bash
   find . -name "*.py" -mtime -7 -not -path "*/__pycache__/*"
   ```

3. **Use `-print0` and `xargs -0` for filenames with spaces**:
   ```bash
   find . -name "*.py" -mtime -7 -print0 | xargs -0 ls -lh
   ```

4. **Test with `-maxdepth` first** for large directories:
   ```bash
   find . -maxdepth 2 -name "*.py" -mtime -7
   ```

5. **Combine with Git** for committed files:
   ```bash
   git log --since="7 days ago" --name-only --pretty=format: | grep "\.py$"
   ```

## Summary

**Most Common Command**:
```bash
find . -type f -name "*.py" -mtime -7
```

**Production-Ready Command** (with exclusions):
```bash
find . -type f -name "*.py" -mtime -7 \
    -not -path "*/__pycache__/*" \
    -not -path "*/venv/*" \
    -not -path "*/.venv/*" \
    -not -path "*/node_modules/*"
```

**With Detailed Output**:
```bash
find . -type f -name "*.py" -mtime -7 \
    -not -path "*/__pycache__/*" \
    -exec ls -lh --time-style=long-iso {} \; | \
    sort -k6,7 -r
```

The `find` command provides the most flexible and powerful way to search for files based on modification time, with extensive options for filtering, formatting, and taking actions on the results.

---

*Exercise 01: Navigation and File System - Completed*
