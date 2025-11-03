# Exercise 01: Navigation and File System Mastery - Solution

## Overview

This solution demonstrates comprehensive Linux file system navigation and file operations for ML infrastructure projects. It provides production-ready shell scripts for creating, managing, and maintaining ML project directory structures.

## Learning Objectives Covered

- ✅ Master Linux directory navigation (cd, pwd, ls)
- ✅ Understand absolute vs relative paths
- ✅ Create and organize ML project directory structures
- ✅ Perform file operations (cp, mv, rm, mkdir)
- ✅ Use find and locate commands for file discovery
- ✅ Create and manage symbolic links
- ✅ Implement backup and cleanup scripts
- ✅ Apply Linux filesystem best practices for ML projects

## Solution Structure

```
exercise-01/
├── README.md                          # This file - solution overview
├── IMPLEMENTATION_GUIDE.md            # Step-by-step implementation guide
├── scripts/                           # Shell scripts for ML project management
│   ├── create_ml_project.sh          # Automated ML project structure creation
│   ├── project_stats.sh              # Project statistics and analysis
│   ├── backup_project.sh             # Intelligent backup (excludes large files)
│   ├── cleanup.sh                    # Clean temporary and cache files
│   └── navigate_examples.sh          # Interactive navigation examples
├── examples/                          # Example project structures
│   └── ml-image-classifier/          # Sample ML project created by scripts
└── docs/
    └── ANSWERS.md                    # Reflection question answers
```

## Key Features

### 1. Automated ML Project Creation
- Creates standard ML project directory structure
- Includes all necessary subdirectories (data, models, notebooks, etc.)
- Generates initial README and .gitignore files
- Follows industry best practices

### 2. Project Statistics Script
- Counts files and directories
- Calculates total disk usage
- Analyzes file types and distributions
- Provides detailed project metrics

### 3. Intelligent Backup System
- Backs up project while excluding large data files
- Creates timestamped archives
- Preserves directory structure
- Optimized for ML projects (excludes models, datasets, checkpoints)

### 4. Cleanup Automation
- Removes Python cache files (__pycache__, .pyc)
- Cleans Jupyter notebook checkpoints
- Removes temporary files and logs
- Safe operation with confirmation prompts

### 5. Navigation Examples
- Interactive demonstration of cd, pwd, ls
- Absolute vs relative path examples
- Symbolic link creation and usage
- Find and locate command demonstrations

## Quick Start

### Create a New ML Project

```bash
cd scripts
./create_ml_project.sh my-ml-project
```

This creates a complete ML project structure:
```
my-ml-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── checkpoints/
│   └── production/
├── notebooks/
│   ├── exploratory/
│   └── reports/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tests/
├── configs/
├── scripts/
├── docs/
├── logs/
├── README.md
└── .gitignore
```

### Get Project Statistics

```bash
./project_stats.sh ../examples/ml-image-classifier
```

Output:
```
=== Project Statistics ===
Project: ml-image-classifier
Total Directories: 18
Total Files: 25
Total Size: 156K

=== Directory Breakdown ===
data/: 8 files (48K)
models/: 0 files (0K)
src/: 12 files (72K)
notebooks/: 3 files (24K)
tests/: 2 files (12K)

=== File Type Analysis ===
.py: 15 files
.ipynb: 3 files
.md: 5 files
.txt: 2 files
```

### Create Backup

```bash
./backup_project.sh ../examples/ml-image-classifier
```

Creates timestamped backup:
```
ml-image-classifier_backup_20250131_143022.tar.gz
```

### Clean Project

```bash
./cleanup.sh ../examples/ml-image-classifier
```

Removes:
- `__pycache__` directories
- `.pyc` files
- `.ipynb_checkpoints`
- Temporary files
- Log files

## Linux Commands Demonstrated

### Navigation Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `cd` | Change directory | `cd /home/user/projects` |
| `cd ~` | Go to home directory | `cd ~` |
| `cd -` | Go to previous directory | `cd -` |
| `pwd` | Print working directory | `pwd` |
| `ls` | List directory contents | `ls -lah` |
| `ls -l` | Long format listing | `ls -l models/` |
| `ls -a` | Show hidden files | `ls -a` |
| `ls -h` | Human-readable sizes | `ls -lh` |

### File Operations

| Command | Purpose | Example |
|---------|---------|---------|
| `mkdir` | Create directory | `mkdir -p data/raw` |
| `mkdir -p` | Create parent directories | `mkdir -p src/utils` |
| `cp` | Copy files | `cp config.yaml config.bak` |
| `cp -r` | Copy directories recursively | `cp -r models/ models_backup/` |
| `mv` | Move or rename files | `mv old.py new.py` |
| `rm` | Remove files | `rm temp.txt` |
| `rm -r` | Remove directories | `rm -r __pycache__` |
| `rm -f` | Force remove | `rm -f *.log` |

### File Finding

| Command | Purpose | Example |
|---------|---------|---------|
| `find` | Search for files | `find . -name "*.py"` |
| `find -type f` | Find files only | `find . -type f -name "test_*"` |
| `find -type d` | Find directories only | `find . -type d -name "__pycache__"` |
| `find -size` | Find by size | `find . -size +100M` |
| `find -mtime` | Find by modification time | `find . -mtime -7` |
| `locate` | Fast file search | `locate model.pkl` |

### Symbolic Links

| Command | Purpose | Example |
|---------|---------|---------|
| `ln -s` | Create symbolic link | `ln -s /data/large-dataset ./data/` |
| `ls -l` | View link target | `ls -l data/` |
| `readlink` | Show link target | `readlink data/large-dataset` |

### Disk Usage

| Command | Purpose | Example |
|---------|---------|---------|
| `du` | Disk usage | `du -h models/` |
| `du -sh` | Summary in human-readable | `du -sh data/` |
| `df` | Filesystem disk space | `df -h` |

## Path Types

### Absolute Paths
- Start from root directory `/`
- Always work regardless of current directory
- Examples:
  - `/home/user/projects/ml-classifier`
  - `/usr/local/bin/python3`
  - `/etc/nginx/nginx.conf`

### Relative Paths
- Relative to current working directory
- Use `.` (current directory) and `..` (parent directory)
- Examples:
  - `./scripts/train.py` (current dir)
  - `../data/processed/` (parent dir)
  - `../../models/production/` (two levels up)

## ML Project Directory Best Practices

### 1. Data Directory Organization
```
data/
├── raw/           # Original, immutable data
├── processed/     # Cleaned, transformed data
└── external/      # External datasets and references
```

### 2. Model Management
```
models/
├── checkpoints/   # Training checkpoints
└── production/    # Production-ready models
```

### 3. Source Code Structure
```
src/
├── preprocessing/  # Data preprocessing modules
├── training/       # Training scripts
├── evaluation/     # Evaluation and metrics
└── utils/          # Utility functions
```

### 4. Version Control
- Use `.gitignore` to exclude large files
- Exclude `data/`, `models/`, `*.pyc`, `__pycache__`
- Track code, configs, and documentation only

### 5. Symbolic Links for Large Data
```bash
# Instead of copying large datasets
ln -s /mnt/storage/large-dataset ./data/large-dataset

# Benefits:
# - Saves disk space
# - No data duplication
# - Easy to update source data
```

## Common ML Project Navigation Patterns

### Pattern 1: Organize by Pipeline Stage
```bash
cd ~/projects/ml-classifier
cd data/raw                    # Start with raw data
cd ../processed                # Move to processed data
cd ../../src/preprocessing     # Go to preprocessing code
cd ../training                 # Move to training code
cd ../../models/checkpoints    # Check training checkpoints
```

### Pattern 2: Quick Access with Aliases
```bash
# Add to ~/.bashrc
alias cdproj='cd ~/projects/ml-classifier'
alias cddata='cd ~/projects/ml-classifier/data'
alias cdmodels='cd ~/projects/ml-classifier/models'

# Usage
cdproj      # Jump to project root
cddata      # Jump to data directory
```

### Pattern 3: Using Find for Model Management
```bash
# Find all model checkpoints
find models/checkpoints -name "*.ckpt"

# Find models modified in last 7 days
find models -type f -mtime -7

# Find large model files (>100MB)
find models -type f -size +100M

# Remove old checkpoints (older than 30 days)
find models/checkpoints -type f -mtime +30 -delete
```

## Script Usage Details

### create_ml_project.sh

**Purpose**: Create standardized ML project structure

**Usage**:
```bash
./create_ml_project.sh PROJECT_NAME
```

**Features**:
- Creates complete directory structure
- Generates initial README.md
- Creates .gitignore with ML-specific exclusions
- Sets proper directory permissions
- Validates project name

**Example**:
```bash
./create_ml_project.sh image-segmentation
cd image-segmentation
ls -la
```

### project_stats.sh

**Purpose**: Analyze project structure and metrics

**Usage**:
```bash
./project_stats.sh PROJECT_PATH
```

**Output**:
- Total directories and files
- Disk usage by directory
- File type distribution
- Largest files and directories
- Recent modifications

**Example**:
```bash
./project_stats.sh ~/projects/ml-classifier
```

### backup_project.sh

**Purpose**: Create intelligent backups excluding large files

**Usage**:
```bash
./backup_project.sh PROJECT_PATH [OUTPUT_DIR]
```

**Features**:
- Excludes data/, models/, __pycache__
- Creates timestamped archives
- Compresses using gzip
- Shows backup size and location
- Validates backup integrity

**Example**:
```bash
./backup_project.sh ~/projects/ml-classifier ~/backups
# Creates: ml-classifier_backup_20250131_143022.tar.gz
```

### cleanup.sh

**Purpose**: Remove temporary and cache files

**Usage**:
```bash
./cleanup.sh PROJECT_PATH
```

**Removes**:
- Python cache: `__pycache__/`, `*.pyc`, `*.pyo`
- Jupyter checkpoints: `.ipynb_checkpoints/`
- Temporary files: `*.tmp`, `*.log`
- Build artifacts: `build/`, `dist/`, `*.egg-info`

**Example**:
```bash
./cleanup.sh ~/projects/ml-classifier
```

### navigate_examples.sh

**Purpose**: Interactive navigation demonstrations

**Usage**:
```bash
./navigate_examples.sh
```

**Demonstrates**:
- Absolute vs relative paths
- cd, pwd, ls commands
- Symbolic link creation
- find and locate usage
- Common navigation patterns

## Testing the Solution

### 1. Create Test Project
```bash
cd scripts
./create_ml_project.sh test-project
```

### 2. Verify Structure
```bash
cd test-project
tree -L 2  # or use ls -R if tree not available
```

### 3. Get Statistics
```bash
cd ../scripts
./project_stats.sh test-project
```

### 4. Create Backup
```bash
./backup_project.sh test-project
ls -lh *.tar.gz
```

### 5. Add Some Files
```bash
cd test-project
echo "print('test')" > src/test.py
touch models/checkpoints/model_epoch_10.ckpt
```

### 6. Run Cleanup
```bash
cd ../scripts
./cleanup.sh test-project
```

## Integration with ML Workflow

### Typical ML Project Workflow

1. **Project Initialization**
   ```bash
   ./create_ml_project.sh customer-churn-prediction
   cd customer-churn-prediction
   ```

2. **Data Preparation**
   ```bash
   cd data/raw
   # Download or copy raw data
   cp /mnt/datasets/churn.csv .
   cd ../processed
   # Process data (run preprocessing scripts)
   ```

3. **Model Development**
   ```bash
   cd ../../notebooks/exploratory
   jupyter notebook eda.ipynb
   # Explore data

   cd ../../src/training
   python train.py
   ```

4. **Regular Backups**
   ```bash
   cd ../../scripts
   ./backup_project.sh .. ~/backups
   ```

5. **Periodic Cleanup**
   ```bash
   ./cleanup.sh ..
   ```

6. **Model Management**
   ```bash
   cd ../models/checkpoints
   # Find best model
   find . -name "*.ckpt" -mtime -7
   # Copy to production
   cp model_epoch_50.ckpt ../production/
   ```

## Performance Considerations

### 1. Large Data Handling
- Use symbolic links for shared datasets
- Keep raw data on separate storage
- Use `.gitignore` to exclude large files

### 2. Efficient Navigation
- Use tab completion for speed
- Create aliases for frequent paths
- Use `cd -` to toggle between directories

### 3. Fast File Finding
- Use `locate` for quick searches (requires updatedb)
- Use `find` with specific paths (not from root)
- Use wildcards strategically

### 4. Disk Space Management
- Regular cleanup of cache and temporary files
- Monitor disk usage with `du -sh`
- Archive old experiments

## Common Issues and Solutions

### Issue 1: Permission Denied
```bash
# Problem
cd /root/project
# bash: cd: /root/project: Permission denied

# Solution
# Check permissions
ls -ld /root/project
# Use sudo if necessary
sudo ls /root/project
```

### Issue 2: Directory Not Found
```bash
# Problem
cd projets/ml-classifier  # Typo
# bash: cd: projets/ml-classifier: No such file or directory

# Solution
# Use tab completion to avoid typos
cd proj<TAB>
# Or verify path exists
ls -d proj*
```

### Issue 3: Accidental File Deletion
```bash
# Problem
rm -rf *  # Ran in wrong directory

# Prevention
# Always verify current directory
pwd
# Use -i flag for confirmation
alias rm='rm -i'
# Create backups before bulk operations
```

### Issue 4: Symbolic Link Broken
```bash
# Problem
ls -l data/dataset
# lrwxrwxrwx 1 user user 26 Jan 31 14:30 data/dataset -> /mnt/storage/dataset (broken)

# Solution
# Remove broken link
rm data/dataset
# Create new link with correct path
ln -s /mnt/new-storage/dataset data/dataset
```

## Best Practices Summary

1. **Always verify paths before destructive operations**
   ```bash
   pwd  # Verify current directory
   ls   # Verify contents
   rm -i file.txt  # Use interactive mode
   ```

2. **Use relative paths for portability**
   ```bash
   # Good (portable)
   cd ../data/processed

   # Bad (hardcoded)
   cd /home/user/projects/data/processed
   ```

3. **Create regular backups**
   ```bash
   # Automate with cron
   0 2 * * * /path/to/backup_project.sh ~/projects/ml-classifier ~/backups
   ```

4. **Keep projects organized**
   - Use standard directory structure
   - Document in README
   - Version control code only (not data)
   - Use symbolic links for shared resources

5. **Monitor disk usage**
   ```bash
   du -sh ~/projects/*
   df -h
   ```

## Time to Complete

- **Reading specifications**: 15 minutes
- **Creating project structure**: 10 minutes
- **Writing shell scripts**: 30 minutes
- **Testing and validation**: 15 minutes
- **Total**: 60-75 minutes

## Skills Acquired

- ✅ Linux directory navigation proficiency
- ✅ File system operations mastery
- ✅ Shell scripting for automation
- ✅ ML project organization best practices
- ✅ Backup and cleanup strategies
- ✅ Symbolic link management
- ✅ File searching and filtering
- ✅ Disk usage monitoring

## Next Steps

- Complete Exercise 02: File Permissions and Access Control
- Learn process management (Exercise 03)
- Master shell scripting (Exercise 04)
- Explore package management (Exercise 05)

## Resources

- [Linux Directory Structure](https://www.pathname.com/fhs/)
- [Bash Guide for Beginners](https://tldp.org/LDP/Bash-Beginners-Guide/html/)
- [ML Project Structure Best Practices](https://drivendata.github.io/cookiecutter-data-science/)
- [GNU Coreutils Manual](https://www.gnu.org/software/coreutils/manual/)

## Conclusion

This solution provides production-ready tools for Linux file system navigation and ML project management. The shell scripts automate common tasks while demonstrating best practices for organizing and maintaining ML infrastructure projects.

**Key Achievement**: Complete implementation of ML project management scripts with automated creation, statistics, backup, and cleanup capabilities.

---

**Exercise 01: Navigation and File System - ✅ COMPLETE**
