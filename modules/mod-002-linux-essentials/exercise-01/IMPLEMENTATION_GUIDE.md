# Exercise 01: Navigation and File System - Implementation Guide

## Overview

This guide provides step-by-step instructions for completing Exercise 01: Navigation and File System Mastery. Follow these steps to learn Linux filesystem navigation and file operations for ML infrastructure projects.

## Time Required

**Total**: 60-75 minutes
- Part 1-3: 15 minutes (basic navigation)
- Part 4-6: 20 minutes (file operations)
- Part 7-9: 20 minutes (advanced commands)
- Part 10: 10-15 minutes (testing and validation)

## Prerequisites

- Linux or macOS terminal access
- Basic command line familiarity
- Text editor (vim, nano, or any editor)

## Part 1: Basic Navigation Commands (10 minutes)

### Step 1.1: Understanding Your Location

Open a terminal and practice these commands:

```bash
# Show current directory
pwd

# List contents
ls

# List with details
ls -l

# List with hidden files
ls -la

# List with human-readable sizes
ls -lh
```

**Exercise**: Navigate to your home directory and list all files

```bash
cd ~
pwd
ls -la
```

### Step 1.2: Creating a Test Environment

Create a temporary workspace:

```bash
# Create test directory
mkdir -p ~/linux-practice
cd ~/linux-practice
pwd

# Verify you're in the right place
```

**Expected output**: `/home/yourusername/linux-practice`

### Step 1.3: Practice Absolute vs Relative Paths

```bash
# Absolute path (starts from root /)
cd /home
pwd

# Relative path (from current directory)
cd ..
pwd

# Back to practice directory (absolute)
cd ~/linux-practice

# Create subdirectories
mkdir -p project/src/utils
mkdir -p project/data

# Navigate using relative path
cd project/src
pwd

# Go up one level
cd ..
pwd

# Go up two levels
cd ../..
pwd
```

**Key Concepts**:
- Absolute paths: Start with `/` or `~`
- Relative paths: Start with `.` (current) or `..` (parent)
- `~` represents home directory

## Part 2: File Operations (10 minutes)

### Step 2.1: Creating Files

```bash
cd ~/linux-practice

# Create empty files
touch file1.txt
touch file2.txt file3.txt

# Create file with content
echo "Hello Linux!" > greeting.txt

# View file content
cat greeting.txt

# Create multiple files at once
touch test{1..5}.txt
ls -l
```

### Step 2.2: Copying Files

```bash
# Copy single file
cp greeting.txt greeting_backup.txt

# Copy to different directory
cp greeting.txt project/

# Copy multiple files
cp file*.txt project/

# Copy directory recursively
cp -r project project_backup

# Verify
ls -R
```

### Step 2.3: Moving and Renaming Files

```bash
# Move file
mv greeting_backup.txt project/

# Rename file
mv greeting.txt hello.txt

# Move and rename simultaneously
mv hello.txt project/welcome.txt

# Move directory
mv project_backup old_project

# Verify
ls -R
```

### Step 2.4: Deleting Files (Use with Caution!)

```bash
# Remove single file
rm test1.txt

# Remove multiple files
rm test2.txt test3.txt

# Remove with confirmation
rm -i test4.txt

# Remove directory (empty)
rmdir empty_dir

# Remove directory with contents (DANGEROUS!)
rm -r old_project

# Safer: interactive removal
rm -ri old_project
```

**Warning**: `rm -rf` permanently deletes files without confirmation. Always verify your current directory first!

## Part 3: Creating ML Project Structure (15 minutes)

### Step 3.1: Create Project Using Script

```bash
cd ~/linux-practice

# Download or use the create_ml_project.sh script
# Make it executable
chmod +x create_ml_project.sh

# Create project
./create_ml_project.sh ml-classifier

# Verify structure
cd ml-classifier
ls -la
tree -L 2  # or ls -R if tree not available
```

### Step 3.2: Manually Create ML Project (Alternative)

If you want to create the structure manually:

```bash
cd ~/linux-practice
mkdir -p ml-manual-project/{data/{raw,processed,external},models/{checkpoints,production},notebooks/{exploratory,reports},src/{preprocessing,training,evaluation,utils},tests,configs,scripts,docs,logs}

cd ml-manual-project
tree -L 2
```

### Step 3.3: Add Initial Files

```bash
cd ~/linux-practice/ml-classifier

# Create README
cat > README.md << 'EOF'
# ML Classifier Project

## Overview
This is a machine learning classification project.

## Structure
- data/: Dataset files
- models/: Trained models
- notebooks/: Jupyter notebooks
- src/: Source code
- tests/: Unit tests
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
data/raw/*
models/*
.ipynb_checkpoints
EOF

# Create sample Python file
cat > src/training/train.py << 'EOF'
"""Training script for ML classifier."""

def train_model(data_path, config):
    """Train machine learning model."""
    print(f"Training model with data from {data_path}")
    # Training logic here
    pass

if __name__ == "__main__":
    train_model("data/processed/train.csv", {})
EOF
```

## Part 4: Finding Files (10 minutes)

### Step 4.1: Using find Command

```bash
cd ~/linux-practice/ml-classifier

# Find all Python files
find . -name "*.py"

# Find only files (not directories)
find . -type f -name "*.py"

# Find directories
find . -type d -name "src"

# Find files modified in last 10 minutes
find . -type f -mmin -10

# Find files larger than 1KB
find . -type f -size +1k

# Find and execute command
find . -name "*.py" -exec wc -l {} \;

# Find empty directories
find . -type d -empty
```

### Step 4.2: Using grep to Search File Contents

```bash
# Search for text in files
grep "import" src/**/*.py

# Search recursively
grep -r "def " src/

# Search case-insensitive
grep -ri "train" .

# Show line numbers
grep -n "import" src/training/train.py

# Search with context (show surrounding lines)
grep -C 2 "def " src/training/train.py
```

### Step 4.3: Combining find and grep

```bash
# Find Python files and search for "train" in them
find . -name "*.py" -exec grep -l "train" {} \;

# More efficient using xargs
find . -name "*.py" | xargs grep "def "
```

## Part 5: Symbolic Links (10 minutes)

### Step 5.1: Creating Symbolic Links

```bash
cd ~/linux-practice/ml-classifier

# Create a large dataset simulation
mkdir -p ~/shared-datasets
echo "large,dataset,file" > ~/shared-datasets/imagenet.csv

# Create symbolic link to shared dataset
ln -s ~/shared-datasets/imagenet.csv data/raw/imagenet-link.csv

# Verify link
ls -l data/raw/
readlink data/raw/imagenet-link.csv
```

### Step 5.2: Working with Symbolic Links

```bash
# Access file through link
cat data/raw/imagenet-link.csv

# Links don't duplicate data
du -h ~/shared-datasets/imagenet.csv
du -h data/raw/imagenet-link.csv

# Remove link (doesn't affect original)
rm data/raw/imagenet-link.csv
cat ~/shared-datasets/imagenet.csv  # Still exists
```

### Step 5.3: Practical ML Use Case

```bash
# Link to frequently used dataset location
ln -s ~/shared-datasets data/shared

# Link to model repository
mkdir -p ~/model-zoo
ln -s ~/model-zoo models/pretrained

# Verify
ls -l data/
ls -l models/
```

## Part 6: Testing the Scripts (10 minutes)

### Step 6.1: Test Project Creation Script

```bash
cd ~/linux-practice

# Create new test project
./create_ml_project.sh test-project-1

# Verify structure
cd test-project-1
ls -R
cat README.md
cat .gitignore
```

### Step 6.2: Test Project Statistics Script

```bash
cd ~/linux-practice

# Generate statistics
./project_stats.sh ml-classifier

# Generate for another project
./project_stats.sh test-project-1
```

Expected output:
- Total directories and files count
- Disk usage by directory
- File type distribution
- Largest files and directories

### Step 6.3: Test Backup Script

```bash
cd ~/linux-practice

# Create backup
./backup_project.sh ml-classifier

# Verify backup exists
ls -lh *.tar.gz

# List contents without extracting
tar -tzf ml-classifier_backup_*.tar.gz

# Test restore
mkdir restore-test
cd restore-test
tar -xzf ../ml-classifier_backup_*.tar.gz
ls -R
```

### Step 6.4: Test Cleanup Script

```bash
cd ~/linux-practice/ml-classifier

# Create some cache files
mkdir -p src/__pycache__
touch src/__pycache__/train.cpython-38.pyc
touch temp.log
mkdir -p notebooks/.ipynb_checkpoints

# Dry run (see what would be deleted)
cd ~/linux-practice
./cleanup.sh ml-classifier --dry-run

# Actually clean
./cleanup.sh ml-classifier -y

# Verify cleanup
find ml-classifier -name "__pycache__"
find ml-classifier -name "*.log"
```

## Part 7: Advanced Navigation (5 minutes)

### Step 7.1: Navigation Shortcuts

```bash
# Go to home directory
cd ~
pwd

# Go to previous directory
cd -
pwd

# Use directory stack
pushd ~/linux-practice/ml-classifier/data
pushd ~/linux-practice/ml-classifier/models
pushd ~/linux-practice/ml-classifier/src

# View stack
dirs

# Pop directories
popd
pwd
popd
pwd
```

### Step 7.2: Efficient Directory Navigation

```bash
# Create aliases (add to ~/.bashrc for permanent)
alias cdml='cd ~/linux-practice/ml-classifier'
alias cddata='cd ~/linux-practice/ml-classifier/data'
alias cdmodels='cd ~/linux-practice/ml-classifier/models'

# Use aliases
cdml
pwd
cddata
pwd
```

## Part 8: Disk Usage Analysis (5 minutes)

### Step 8.1: Check Disk Usage

```bash
cd ~/linux-practice

# Check total disk usage
du -sh ml-classifier

# Check usage by subdirectory
du -h ml-classifier

# Summarize one level deep
du -h --max-depth=1 ml-classifier

# Sort by size
du -h ml-classifier | sort -rh | head -10
```

### Step 8.2: Find Large Files

```bash
# Find files larger than 1MB
find ml-classifier -type f -size +1M

# List largest files
find ml-classifier -type f -exec du -h {} + | sort -rh | head -10

# Check filesystem usage
df -h
```

## Part 9: Practical ML Workflow (10 minutes)

### Step 9.1: Typical Development Workflow

```bash
cd ~/linux-practice/ml-classifier

# 1. Check project status
pwd
ls -l

# 2. Navigate to data directory
cd data/raw
ls -lh

# 3. Process data (simulated)
echo "processed" > ../processed/train.csv

# 4. Navigate to training code
cd ../../src/training
cat train.py

# 5. Run training (simulated)
python3 train.py

# 6. Check model outputs
cd ../../models/checkpoints
touch model_epoch_1.ckpt
ls -lh

# 7. Return to project root
cd ../..
pwd

# 8. Create backup
cd ~/linux-practice
./backup_project.sh ml-classifier
```

### Step 9.2: File Management Workflow

```bash
cd ~/linux-practice/ml-classifier

# Find all Python files
find . -name "*.py"

# Count lines of code
find . -name "*.py" -exec wc -l {} + | tail -1

# Find recent modifications
find . -type f -mtime -1

# Search for specific functions
grep -r "def " src/

# Clean up cache files
cd ~/linux-practice
./cleanup.sh ml-classifier -y
```

## Part 10: Validation and Testing (10 minutes)

### Step 10.1: Verify All Scripts Work

```bash
cd ~/linux-practice

# Test 1: Create project
./create_ml_project.sh validation-project
cd validation-project
ls -R

# Test 2: Generate statistics
cd ..
./project_stats.sh validation-project

# Test 3: Create backup
./backup_project.sh validation-project
ls -lh *.tar.gz

# Test 4: Add cache files
cd validation-project
mkdir -p src/__pycache__
touch src/__pycache__/test.pyc

# Test 5: Clean up
cd ..
./cleanup.sh validation-project -y
```

### Step 10.2: Verify Navigation Skills

Complete these exercises:

```bash
cd ~/linux-practice

# Exercise 1: Navigate to validation-project/src/training
cd validation-project/src/training
pwd  # Should show .../validation-project/src/training

# Exercise 2: Go to models directory using relative path
cd ../../models
pwd  # Should show .../validation-project/models

# Exercise 3: Return to linux-practice using absolute path
cd ~/linux-practice
pwd

# Exercise 4: Find all directories named "src"
find . -type d -name "src"

# Exercise 5: Create symbolic link
ln -s validation-project/data shared-data
ls -l shared-data
```

### Step 10.3: Performance Check

```bash
# Check script execution time
time ./create_ml_project.sh perf-test
# Should complete in < 5 seconds

time ./project_stats.sh perf-test
# Should complete in < 10 seconds

time ./backup_project.sh perf-test
# Should complete in < 15 seconds

time ./cleanup.sh perf-test -y
# Should complete in < 5 seconds
```

## Part 11: Cleanup (5 minutes)

### Step 11.1: Remove Practice Files

```bash
# Remove all practice files
cd ~
rm -rf linux-practice

# Verify removal
ls -d linux-practice 2>/dev/null || echo "Cleanup successful"
```

### Step 11.2: Optional - Preserve Scripts

If you want to keep the scripts for future use:

```bash
# Copy scripts to home directory
mkdir -p ~/bin
cp linux-practice/scripts/*.sh ~/bin/

# Add to PATH (add to ~/.bashrc for permanent)
export PATH="$HOME/bin:$PATH"

# Now you can use scripts from anywhere
create_ml_project.sh my-new-project
```

## Common Issues and Solutions

### Issue 1: Permission Denied

```bash
# Problem
./script.sh
# bash: ./script.sh: Permission denied

# Solution
chmod +x script.sh
./script.sh
```

### Issue 2: Directory Not Found

```bash
# Problem
cd non-existent-dir
# bash: cd: non-existent-dir: No such file or directory

# Solution
# Check current directory
pwd
ls

# Verify path exists
ls -d non-existent-dir

# Create if needed
mkdir -p non-existent-dir
```

### Issue 3: find Command Too Slow

```bash
# Problem: find searches entire filesystem

# Solution: Specify starting directory
find ~/linux-practice -name "*.py"  # Instead of find / -name "*.py"

# Use locate for faster searches (requires updatedb)
locate "*.py"
```

### Issue 4: Accidental Deletion

```bash
# Prevention: Use -i flag
rm -i file.txt

# Create alias (add to ~/.bashrc)
alias rm='rm -i'

# For directories
alias rm='rm -ri'
```

## Best Practices Learned

1. **Always verify your location before destructive operations**
   ```bash
   pwd  # Check where you are
   ls   # Check what's here
   rm -i file.txt  # Use interactive mode
   ```

2. **Use relative paths for portability**
   ```bash
   cd ../data  # Good
   cd /home/user/project/data  # Less portable
   ```

3. **Create aliases for frequent operations**
   ```bash
   alias cdproj='cd ~/projects/ml-classifier'
   alias ll='ls -lah'
   ```

4. **Use tab completion to avoid typos**
   - Type `cd pro` and press TAB
   - Bash completes to `cd projects/`

5. **Regular backups are essential**
   ```bash
   ./backup_project.sh my-project ~/backups
   ```

## Next Steps

1. ✅ Complete Exercise 02: File Permissions and Access Control
2. ✅ Practice these commands daily
3. ✅ Create your own shell aliases
4. ✅ Explore advanced find options
5. ✅ Learn vim or emacs for efficient editing

## Summary

You've learned:
- ✅ Linux directory navigation (cd, pwd, ls)
- ✅ Absolute vs relative paths
- ✅ File operations (cp, mv, rm, mkdir)
- ✅ Finding files (find, locate)
- ✅ Symbolic links for efficient data management
- ✅ Shell scripting for automation
- ✅ ML project organization best practices
- ✅ Backup and cleanup strategies

**Time Investment**: 60-75 minutes
**Skills Acquired**: Core Linux navigation and file operations
**Ready For**: Exercise 02 (File Permissions) and real-world ML projects

---

**Congratulations on completing Exercise 01!** 🎉
