# Implementation Guide: File Permissions and Access Control for ML Teams

## Overview

This comprehensive implementation guide walks you through mastering Linux file permissions and access control in the context of multi-user ML infrastructure environments. You'll learn to secure datasets, protect sensitive credentials, enable team collaboration, and implement industry-standard security practices for AI/ML projects.

**What You'll Learn:**
- Linux permission model (user, group, other) and permission bits
- Configuring permissions with chmod (numeric and symbolic notation)
- Managing file ownership with chown and chgrp
- Implementing Access Control Lists (ACLs) for fine-grained permissions
- Setting default permissions with umask
- Applying security best practices for ML infrastructure
- Creating secure shared directories for ML team collaboration

**Estimated Time:** 90-120 minutes
**Difficulty:** Intermediate

## Prerequisites

Before starting this exercise, ensure you have:
- [ ] Completed Exercise 01 (Navigation and File Operations)
- [ ] Completed Lecture 03: Permissions and Security
- [ ] Access to a Linux system (VM, WSL, or native Linux)
- [ ] Basic understanding of Linux command line
- [ ] At least 2GB free disk space
- [ ] Terminal emulator installed

**Recommended Setup:**
```bash
# Verify you have the necessary tools
which chmod chown ls stat find

# Check available disk space
df -h ~

# Verify you're in a suitable working directory
pwd
```

## Learning Path Structure

This guide is organized into progressive phases:

1. **Phase 1**: Understanding Permissions (30 minutes)
2. **Phase 2**: Configuring Permissions (45 minutes)
3. **Phase 3**: Advanced Permissions with ACLs (30 minutes)
4. **Phase 4**: Security Best Practices (30 minutes)
5. **Phase 5**: Production Implementation (15 minutes)

---

# Phase 1: Understanding Permissions (30 minutes)

## Step 1.1: Create Your Lab Environment

First, set up a dedicated workspace for this exercise.

```bash
# Create and enter the lab directory
mkdir -p ~/ml-permissions-lab
cd ~/ml-permissions-lab

# Verify you're in the right place
pwd
# Expected: /home/username/ml-permissions-lab
```

**Expected Output:**
```
/home/yourusername/ml-permissions-lab
```

**Validation:**
- [ ] Lab directory created successfully
- [ ] Currently in the lab directory
- [ ] Have write permissions in this location

## Step 1.2: Create Initial Test Files and Directories

Create a variety of files and directories to examine permissions.

```bash
cd ~/ml-permissions-lab

# Create various file types that you'll find in ML projects
touch dataset.csv model.h5 training_script.py config.yaml
touch api_keys.yaml credentials.json

# Create directory structure
mkdir shared_models team_data private_keys

# List everything with permissions
ls -l
```

**Expected Output:**
```
-rw-r--r-- 1 username username    0 Nov 01 10:00 api_keys.yaml
-rw-r--r-- 1 username username    0 Nov 01 10:00 config.yaml
-rw-r--r-- 1 username username    0 Nov 01 10:00 credentials.json
-rw-r--r-- 1 username username    0 Nov 01 10:00 dataset.csv
-rw-r--r-- 1 username username    0 Nov 01 10:00 model.h5
drwxr-xr-x 2 username username 4096 Nov 01 10:00 private_keys
drwxr-xr-x 2 username username 4096 Nov 01 10:00 shared_models
drwxr-xr-x 2 username username 4096 Nov 01 10:00 team_data
-rw-r--r-- 1 username username    0 Nov 01 10:00 training_script.py
```

**Key Observations:**
- Files start with `-`, directories with `d`
- Files have `rw-r--r--` (644) permissions by default
- Directories have `rwxr-xr-x` (755) permissions by default
- Each item shows owner and group

**Validation:**
- [ ] All files created successfully
- [ ] All directories created successfully
- [ ] Can see permission strings in `ls -l` output

## Step 1.3: Decode Permission Strings

Learn to read and understand permission strings.

```bash
# Examine a specific file in detail
ls -l dataset.csv

# Break down the permission string
echo "Let's decode: -rw-r--r--"
echo ""
echo "Position 1: '-' = regular file (d=directory, l=link)"
echo "Positions 2-4: 'rw-' = owner can read and write"
echo "Positions 5-7: 'r--' = group can only read"
echo "Positions 8-10: 'r--' = others can only read"
```

**Create a Visual Reference:**

```bash
cat > permission_decoder.txt << 'EOF'
PERMISSION STRING DECODER
=========================

Example: -rwxr-xr--

Position  Permission  Who      Meaning
--------  ----------  -------  -------
    1        -        Type     Regular file
    2        r        Owner    Read permission
    3        w        Owner    Write permission
    4        x        Owner    Execute permission
    5        r        Group    Read permission
    6        -        Group    No write
    7        x        Group    Execute permission
    8        r        Others   Read permission
    9        -        Others   No write
   10        -        Others   No execute

FILE TYPES:
-  Regular file
d  Directory
l  Symbolic link
c  Character device
b  Block device
p  Named pipe (FIFO)
s  Socket

PERMISSION MEANINGS:
r (read)    - Can view file contents / list directory
w (write)   - Can modify file / create/delete files in directory
x (execute) - Can run file / access directory contents
EOF

cat permission_decoder.txt
```

**Expected Output:**
```
PERMISSION STRING DECODER
=========================
[... full reference displayed ...]
```

**Validation:**
- [ ] Understand each position in permission string
- [ ] Can identify file types
- [ ] Know what r, w, x mean for files and directories

## Step 1.4: Understand Ownership

Examine file ownership and your user context.

```bash
# Check who you are
whoami

# See all your groups
groups

# Detailed user information
id

# Examine ownership of a file
ls -l dataset.csv | awk '{print "Owner:", $3, "Group:", $4}'

# See numeric user and group IDs
stat -c 'User: %U (%u) Group: %G (%g)' dataset.csv
```

**Expected Output:**
```
yourusername
yourusername sudo docker

uid=1000(yourusername) gid=1000(yourusername) groups=1000(yourusername),4(adm),27(sudo),999(docker)

Owner: yourusername Group: yourusername

User: yourusername (1000) Group: yourusername (1000)
```

**Key Concepts:**
- **Owner (User)**: The person who created the file
- **Group**: A collection of users who share access
- **Others**: Everyone else on the system

**Validation:**
- [ ] Know your username
- [ ] Can list your groups
- [ ] Understand UID and GID concepts
- [ ] Can identify file owner and group

## Step 1.5: Master Numeric Permission Notation

Learn to calculate permissions using the numeric (octal) system.

```bash
cat > numeric_permissions.txt << 'EOF'
NUMERIC PERMISSION CALCULATION
==============================

Each permission has a numeric value:
r (read)    = 4
w (write)   = 2
x (execute) = 1

Combine them by adding:
---  = 0 (no permissions)
--x  = 1 (execute only)
-w-  = 2 (write only)
-wx  = 3 (write + execute)
r--  = 4 (read only)
r-x  = 5 (read + execute)
rw-  = 6 (read + write)
rwx  = 7 (all permissions)

COMPLETE PERMISSION EXAMPLES:
=============================

644 = rw-r--r--
  Owner:  6 = rw-  (read + write)
  Group:  4 = r--  (read only)
  Others: 4 = r--  (read only)
  Use: Standard file permissions

755 = rwxr-xr-x
  Owner:  7 = rwx  (full control)
  Group:  5 = r-x  (read + execute)
  Others: 5 = r-x  (read + execute)
  Use: Executable files, directories

600 = rw-------
  Owner:  6 = rw-  (read + write)
  Group:  0 = ---  (no access)
  Others: 0 = ---  (no access)
  Use: Private files (credentials, keys)

700 = rwx------
  Owner:  7 = rwx  (full control)
  Group:  0 = ---  (no access)
  Others: 0 = ---  (no access)
  Use: Private directories

775 = rwxrwxr-x
  Owner:  7 = rwx  (full control)
  Group:  7 = rwx  (full control)
  Others: 5 = r-x  (read + execute)
  Use: Team collaborative directories

664 = rw-rw-r--
  Owner:  6 = rw-  (read + write)
  Group:  6 = rw-  (read + write)
  Others: 4 = r--  (read only)
  Use: Team editable files
EOF

cat numeric_permissions.txt
```

**Practice Exercise:**

Calculate the numeric permissions for these symbolic permissions:

```bash
echo "Practice Problems:"
echo "1. rwxrwxrwx = ?"
echo "2. rw-rw-rw- = ?"
echo "3. r-xr-xr-x = ?"
echo "4. rwx------ = ?"
echo "5. rw-r----- = ?"
echo "6. rwxr-x--- = ?"
echo ""
echo "Try to calculate them yourself, then check answers below!"
echo ""

# Give yourself time to think...

cat << 'EOF'

ANSWERS:
========
1. rwxrwxrwx = 777 (7+7+7: all permissions for all)
2. rw-rw-rw- = 666 (6+6+6: read/write for all, no execute)
3. r-xr-xr-x = 555 (5+5+5: read/execute for all)
4. rwx------ = 700 (7+0+0: owner only, full access)
5. rw-r----- = 640 (6+4+0: owner rw, group r, others none)
6. rwxr-x--- = 750 (7+5+0: owner full, group rx, others none)

CALCULATION EXAMPLE for 640:
  rw- = 4+2+0 = 6
  r-- = 4+0+0 = 4
  --- = 0+0+0 = 0
  Result: 640
EOF
```

**Validation:**
- [ ] Can calculate numeric permissions from symbolic
- [ ] Understand the 4-2-1 system
- [ ] Know common permission patterns (644, 755, 600, 700)

## Step 1.6: Check Current umask

Understand how umask affects default permissions for new files.

```bash
# Check current umask (numeric)
umask

# Check umask in symbolic notation
umask -S

# Create a reference for umask
cat > umask_explanation.txt << 'EOF'
UMASK EXPLAINED
===============

umask SUBTRACTS from default permissions:
- Files default:       666 (rw-rw-rw-)
- Directories default: 777 (rwxrwxrwx)

Common umask values:

umask 0022 (Standard/Default)
-----------------------------
Files:     666 - 022 = 644 (rw-r--r--)
Dirs:      777 - 022 = 755 (rwxr-xr-x)
Use case:  General single-user work

umask 0002 (Collaborative)
--------------------------
Files:     666 - 002 = 664 (rw-rw-r--)
Dirs:      777 - 002 = 775 (rwxrwxr-x)
Use case:  Team collaboration, shared projects

umask 0077 (Secure/Private)
---------------------------
Files:     666 - 077 = 600 (rw-------)
Dirs:      777 - 077 = 700 (rwx------)
Use case:  Sensitive files, credentials

CALCULATION EXAMPLE (umask 0027):
---------------------------------
User  digit: 7 - 0 = 7 (rwx)
Group digit: 7 - 2 = 5 (r-x)
Other digit: 7 - 7 = 0 (---)

For files:  666 - 027 = 640 (rw-r-----)
For dirs:   777 - 027 = 750 (rwxr-x---)
EOF

cat umask_explanation.txt
```

**Test umask in Action:**

```bash
# Save current umask
ORIGINAL_UMASK=$(umask)

# Test with different umask values
echo "Testing umask effects..."

# Standard umask (0022)
umask 0022
touch test_022_file
mkdir test_022_dir
ls -ld test_022_file test_022_dir

# Collaborative umask (0002)
umask 0002
touch test_002_file
mkdir test_002_dir
ls -ld test_002_file test_002_dir

# Secure umask (0077)
umask 0077
touch test_077_file
mkdir test_077_dir
ls -ld test_077_file test_077_dir

# Compare results
echo ""
echo "Comparison:"
ls -l test_* | awk '{print $1, $9}'

# Restore original umask
umask $ORIGINAL_UMASK
echo ""
echo "Restored original umask: $(umask)"
```

**Expected Output:**
```
-rw-r--r-- test_022_file
drwxr-xr-x test_022_dir
-rw-rw-r-- test_002_file
drwxrwxr-x test_002_dir
-rw------- test_077_file
drwx------ test_077_dir
```

**Validation:**
- [ ] Understand how umask affects new files
- [ ] Know the difference between 0022, 0002, and 0077
- [ ] Can calculate resulting permissions from umask

---

# Phase 2: Configuring Permissions (45 minutes)

## Step 2.1: Change Permissions with Numeric chmod

Practice setting exact permissions using numeric notation.

```bash
cd ~/ml-permissions-lab
mkdir numeric_practice
cd numeric_practice

# Create test files
touch private_model.h5 shared_dataset.csv team_script.py public_readme.md

# Set private file (owner only, read/write)
chmod 600 private_model.h5
ls -l private_model.h5
echo "Expected: -rw-------"

# Set shared dataset (owner rw, group rw, other read)
chmod 664 shared_dataset.csv
ls -l shared_dataset.csv
echo "Expected: -rw-rw-r--"

# Set executable script (owner rwx, group rx, other rx)
chmod 755 team_script.py
ls -l team_script.py
echo "Expected: -rwxr-xr-x"

# Set public readable file
chmod 644 public_readme.md
ls -l public_readme.md
echo "Expected: -rw-r--r--"

# Create directories with proper permissions
mkdir secure_models shared_experiments

# Private directory (owner only)
chmod 700 secure_models
ls -ld secure_models
echo "Expected: drwx------"

# Shared directory (group collaborative)
chmod 775 shared_experiments
ls -ld shared_experiments
echo "Expected: drwxrwxr-x"
```

**Verification Script:**

```bash
# Verify all permissions are correct
echo "=== Permission Verification ==="

check_permission() {
    local file=$1
    local expected=$2
    local actual=$(stat -c '%a' "$file")

    if [ "$actual" = "$expected" ]; then
        echo "✓ $file: $actual (correct)"
    else
        echo "✗ $file: $actual (expected $expected)"
    fi
}

check_permission private_model.h5 600
check_permission shared_dataset.csv 664
check_permission team_script.py 755
check_permission public_readme.md 644
check_permission secure_models 700
check_permission shared_experiments 775
```

**Expected Output:**
```
=== Permission Verification ===
✓ private_model.h5: 600 (correct)
✓ shared_dataset.csv: 664 (correct)
✓ team_script.py: 755 (correct)
✓ public_readme.md: 644 (correct)
✓ secure_models: 700 (correct)
✓ shared_experiments: 775 (correct)
```

**Validation:**
- [ ] Can set exact permissions with numeric chmod
- [ ] Understand 600, 644, 664, 755, 700, 775 meanings
- [ ] Can verify permissions with stat command

## Step 2.2: Change Permissions with Symbolic chmod

Learn the more flexible symbolic notation for chmod.

```bash
cd ~/ml-permissions-lab
mkdir symbolic_practice
cd symbolic_practice

# Create files for symbolic mode practice
touch model_v1.h5 data_prep.py results.csv notebook.ipynb

# Add execute permission for owner
chmod u+x data_prep.py
ls -l data_prep.py
echo "Before: -rw-r--r--, After: -rwxr--r--"

# Add write permission for group
chmod g+w results.csv
ls -l results.csv
echo "Before: -rw-r--r--, After: -rw-rw-r--"

# Remove read permission for others
chmod o-r model_v1.h5
ls -l model_v1.h5
echo "Before: -rw-r--r--, After: -rw-r-----"

# Set exact permissions (overwrites existing)
chmod u=rwx,g=rx,o=r data_prep.py
ls -l data_prep.py
echo "Result: -rwxr-xr--"

# Multiple changes at once
chmod u+x,g+w,o-r notebook.ipynb
ls -l notebook.ipynb

# Add execute to all (user, group, others)
touch analyze.sh
chmod a+x analyze.sh
ls -l analyze.sh
echo "a = all (user, group, other)"
```

**Create Symbolic Mode Reference:**

```bash
cat > symbolic_reference.txt << 'EOF'
SYMBOLIC MODE REFERENCE
=======================

WHO (User classes):
-------------------
u = user/owner
g = group
o = others
a = all (ugo)

OPERATION:
----------
+ = add permission
- = remove permission
= = set exact permission (overwrites)

PERMISSION:
-----------
r = read
w = write
x = execute

EXAMPLES:
---------
chmod u+x file          Add execute for owner
chmod g-w file          Remove write for group
chmod o-r file          Remove read for others
chmod a+r file          Add read for all
chmod u=rwx file        Set owner to rwx exactly
chmod go=rx file        Set group and other to r-x
chmod u+x,g+x file      Add execute for owner and group
chmod a-w file          Remove write for all
chmod ug+rw file        Add read/write for owner and group

RECURSIVE:
----------
chmod -R u+x dir/       Apply recursively to directory

SPECIAL PATTERNS:
-----------------
chmod +x file           Add execute to existing permissions
chmod -R go-w dir/      Remove group/other write recursively
EOF

cat symbolic_reference.txt
```

**Practice Exercise:**

```bash
# Create a practice file
touch practice_file.txt

echo "Starting permissions:"
ls -l practice_file.txt

# Follow these steps and observe the changes:
echo ""
echo "Step 1: Add execute for owner"
chmod u+x practice_file.txt
ls -l practice_file.txt

echo ""
echo "Step 2: Add write for group"
chmod g+w practice_file.txt
ls -l practice_file.txt

echo ""
echo "Step 3: Remove all permissions for others"
chmod o= practice_file.txt
ls -l practice_file.txt

echo ""
echo "Step 4: Set owner to full access"
chmod u=rwx practice_file.txt
ls -l practice_file.txt

echo ""
echo "Final permissions:"
stat -c '%a %n' practice_file.txt
```

**Validation:**
- [ ] Can add permissions with +
- [ ] Can remove permissions with -
- [ ] Can set exact permissions with =
- [ ] Understand u, g, o, a notation
- [ ] Can combine multiple operations

## Step 2.3: Recursive Permission Changes

Learn to modify permissions for entire directory trees.

```bash
cd ~/ml-permissions-lab
mkdir -p recursive_test/project/{data,models,scripts,configs}

# Create files in the tree
touch recursive_test/project/data/train.csv
touch recursive_test/project/data/test.csv
touch recursive_test/project/models/model.h5
touch recursive_test/project/scripts/train.sh
touch recursive_test/project/scripts/deploy.sh
touch recursive_test/project/configs/config.yaml

# View initial structure
echo "Before recursive changes:"
find recursive_test -ls

# Apply recursive changes
# Make all directories accessible
chmod -R 755 recursive_test/project

# Make all shell scripts executable
find recursive_test/project/scripts -type f -name "*.sh" -exec chmod +x {} \;

# Set data files to read-only
find recursive_test/project/data -type f -name "*.csv" -exec chmod 444 {} \;

# Set collaborative permissions for models directory
chmod 775 recursive_test/project/models

echo ""
echo "After recursive changes:"
find recursive_test -ls
```

**Validation:**
- [ ] Can use chmod -R for recursive changes
- [ ] Can use find with -exec for selective changes
- [ ] Understand when to use recursive vs. selective

## Step 2.4: Real-World ML Permission Scenarios

Implement realistic permission structures for ML projects.

```bash
cd ~/ml-permissions-lab
mkdir ml_project_team
cd ml_project_team

echo "=== Scenario 1: Shared Dataset Directory ==="
# Multiple data scientists need read access, only owner can modify

mkdir -p datasets/{raw,processed}

# Raw data: immutable (owner rw, group and others read)
chmod 755 datasets/raw
touch datasets/raw/train_images.tar
chmod 644 datasets/raw/train_images.tar

echo "Raw dataset permissions:"
ls -ld datasets/raw
ls -l datasets/raw/train_images.tar

# Processed data: team can write
chmod 775 datasets/processed
touch datasets/processed/augmented_images.npz
chmod 664 datasets/processed/augmented_images.npz

echo "Processed dataset permissions:"
ls -ld datasets/processed
ls -l datasets/processed/augmented_images.npz

echo ""
echo "=== Scenario 2: Model Repository ==="
# Team can read models, only ML engineers can write

mkdir -p models/{checkpoints,production}

# Checkpoints: Team collaborative
chmod 775 models/checkpoints
touch models/checkpoints/epoch_050.h5
chmod 664 models/checkpoints/epoch_050.h5

# Production: Restricted
chmod 755 models/production
touch models/production/v1.2.3.h5
chmod 444 models/production/v1.2.3.h5  # Read-only

echo "Model repository permissions:"
ls -ld models/checkpoints models/production
ls -l models/checkpoints/epoch_050.h5
ls -l models/production/v1.2.3.h5

echo ""
echo "=== Scenario 3: Training Scripts ==="
# Everyone can read and execute, only developers can modify

mkdir -p scripts
touch scripts/train_model.py scripts/evaluate.py scripts/deploy.sh

chmod 755 scripts/*.py
chmod 755 scripts/*.sh

echo "Script permissions:"
ls -l scripts/

echo ""
echo "=== Scenario 4: Sensitive Configuration ==="
# Only owner should access

mkdir -p configs/secrets
chmod 700 configs/secrets

touch configs/secrets/api_keys.yaml configs/secrets/db_credentials.json
chmod 600 configs/secrets/*

echo "Secrets directory:"
ls -ld configs/secrets
echo "Secret files (note: might not be able to list them):"
ls -l configs/secrets/ 2>&1 || echo "Directory is properly secured"

echo ""
echo "=== Scenario 5: Log Files ==="
# System writes, team reads

mkdir -p logs
chmod 755 logs

touch logs/training_2024-11-01.log
chmod 644 logs/training_2024-11-01.log

echo "Log directory and files:"
ls -ld logs
ls -l logs/

echo ""
echo "=== Scenario 6: Collaborative Notebooks ==="
# Team can edit together

mkdir -p notebooks
chmod 775 notebooks

touch notebooks/exploration.ipynb notebooks/training_analysis.ipynb
chmod 664 notebooks/*.ipynb

echo "Notebook directory and files:"
ls -ld notebooks
ls -l notebooks/
```

**Verification Checklist:**

```bash
# Create automated verification script
cat > verify_ml_scenarios.sh << 'EOF'
#!/bin/bash

echo "=== ML Project Permission Verification ==="

check_permission() {
    local path=$1
    local expected=$2
    local actual=$(stat -c '%a' "$path" 2>/dev/null)

    if [ "$actual" = "$expected" ]; then
        echo "✓ $path: $actual"
    else
        echo "✗ $path: $actual (expected $expected)"
    fi
}

# Datasets
check_permission "datasets/raw" 755
check_permission "datasets/raw/train_images.tar" 644
check_permission "datasets/processed" 775
check_permission "datasets/processed/augmented_images.npz" 664

# Models
check_permission "models/checkpoints" 775
check_permission "models/checkpoints/epoch_050.h5" 664
check_permission "models/production" 755
check_permission "models/production/v1.2.3.h5" 444

# Scripts
check_permission "scripts/train_model.py" 755

# Secrets
check_permission "configs/secrets" 700
check_permission "configs/secrets/api_keys.yaml" 600

# Logs
check_permission "logs" 755
check_permission "logs/training_2024-11-01.log" 644

# Notebooks
check_permission "notebooks" 775
check_permission "notebooks/exploration.ipynb" 664

echo ""
echo "Verification complete!"
EOF

chmod +x verify_ml_scenarios.sh
./verify_ml_scenarios.sh
```

**Expected Output:**
```
=== ML Project Permission Verification ===
✓ datasets/raw: 755
✓ datasets/raw/train_images.tar: 644
✓ datasets/processed: 775
✓ datasets/processed/augmented_images.npz: 664
✓ models/checkpoints: 775
✓ models/checkpoints/epoch_050.h5: 664
✓ models/production: 755
✓ models/production/v1.2.3.h5: 444
✓ scripts/train_model.py: 755
✓ configs/secrets: 700
✓ configs/secrets/api_keys.yaml: 600
✓ logs: 755
✓ logs/training_2024-11-01.log: 644
✓ notebooks: 775
✓ notebooks/exploration.ipynb: 664

Verification complete!
```

**Validation:**
- [ ] Raw data is readable but not writable by team
- [ ] Processed data is collaborative
- [ ] Production models are read-only
- [ ] Secrets are private (700/600)
- [ ] Scripts are executable
- [ ] Notebooks are team-editable

---

# Phase 3: Advanced Permissions with ACLs (30 minutes)

## Step 3.1: Introduction to Access Control Lists (ACLs)

ACLs provide fine-grained permission control beyond the traditional user/group/other model.

```bash
cd ~/ml-permissions-lab
mkdir acl_practice
cd acl_practice

# Check if ACL is supported on your filesystem
echo "Checking ACL support..."
getfacl --version
df -T .

# Create a test file
touch model_registry.h5
```

**Expected Output:**
```
getfacl 2.3.1
Filesystem     Type  ...
/dev/sda1      ext4  ...
```

**Key Concepts:**
- ACLs extend traditional permissions
- Allow per-user and per-group permissions
- Don't replace traditional permissions, supplement them
- Require filesystem support (most modern Linux filesystems support ACLs)

## Step 3.2: View and Understand ACLs

```bash
# Create a shared model directory
mkdir shared_model_registry
touch shared_model_registry/model_v1.h5

# View default ACLs
echo "Default ACL (no special ACLs set):"
getfacl shared_model_registry/model_v1.h5

# The output shows:
# - file: filename
# - owner: username
# - group: groupname
# - user::permissions (owner permissions)
# - group::permissions (group permissions)
# - other::permissions (others permissions)
```

**Expected Output:**
```
# file: shared_model_registry/model_v1.h5
# owner: username
# group: username
user::rw-
group::r--
other::r--
```

**Understanding ACL Output:**
- `user::rw-` - Owner has read/write
- `group::r--` - Group has read
- `other::r--` - Others have read
- Additional lines would show ACL entries

## Step 3.3: Set User-Specific ACLs

Learn to grant specific permissions to individual users.

```bash
# Note: These examples show the syntax
# In a real environment, you'd replace 'alice' and 'bob' with actual usernames

# Create ACL example script
cat > acl_examples.sh << 'EOF'
#!/bin/bash
# ACL Configuration Examples for ML Infrastructure

# These are example commands - they require:
# 1. Multiple users on the system
# 2. sudo access for some operations
# 3. ACL-enabled filesystem

echo "=== ACL Examples for ML Projects ==="
echo ""

# Example 1: Give specific user read access to model
echo "Example 1: User-specific model access"
echo "setfacl -m u:alice:r model_v1.h5"
echo "  Grants user 'alice' read permission to model_v1.h5"
echo ""

# Example 2: Give group full access
echo "Example 2: Group access to shared directory"
echo "setfacl -m g:mlteam:rwx shared_models/"
echo "  Grants 'mlteam' group full access to shared_models/"
echo ""

# Example 3: Set default ACL for directory
echo "Example 3: Default ACL for new files"
echo "setfacl -d -m g:datateam:rw datasets/processed/"
echo "  New files in datasets/processed/ inherit group permissions"
echo ""

# Example 4: Give specific user execute access
echo "Example 4: User-specific script execution"
echo "setfacl -m u:bob:rx scripts/train.py"
echo "  Allows 'bob' to read and execute train.py"
echo ""

# Example 5: Complex scenario
echo "Example 5: Multiple ACL entries"
echo "setfacl -m u:alice:rw,u:bob:r,g:mlops:rwx production_model.h5"
echo "  alice: read/write, bob: read, mlops group: full access"
echo ""

# View ACL
echo "View ACLs:"
echo "getfacl filename"
echo ""

# Remove specific ACL
echo "Remove user ACL:"
echo "setfacl -x u:alice filename"
echo ""

# Remove all ACLs
echo "Remove all ACLs:"
echo "setfacl -b filename"
echo ""

# Copy ACLs
echo "Copy ACL from one file to another:"
echo "getfacl file1 | setfacl --set-file=- file2"
EOF

chmod +x acl_examples.sh
./acl_examples.sh
```

## Step 3.4: Set Default ACLs for Directories

Default ACLs are inherited by new files created in a directory.

```bash
# Create example of default ACLs
cat > default_acl_demo.sh << 'EOF'
#!/bin/bash

echo "=== Default ACL Demonstration ==="
echo ""
echo "Default ACLs apply to new files/directories created within a directory"
echo ""

# Create a collaborative dataset directory
mkdir -p collaborative_datasets

echo "Setting default ACL on collaborative_datasets/"
echo "Command: setfacl -d -m g:mlteam:rw collaborative_datasets/"
echo ""
echo "This means:"
echo "  - All new files created in collaborative_datasets/"
echo "  - Will automatically grant 'mlteam' group read/write permissions"
echo "  - Ensures team collaboration without manual permission setting"
echo ""

# Show the concept
echo "Example workflow:"
echo "1. Admin sets default ACL: setfacl -d -m g:mlteam:rw datasets/"
echo "2. User creates file:      touch datasets/new_data.csv"
echo "3. File automatically has: mlteam group with rw permissions"
echo ""

# Real-world scenarios
cat << 'SCENARIOS'
REAL-WORLD SCENARIOS:
====================

Scenario 1: Data Engineering Team
----------------------------------
setfacl -d -m g:dataeng:rw /shared/datasets/raw/
setfacl -d -m g:dataeng:rw /shared/datasets/processed/

Result: All new datasets automatically accessible to data engineering team

Scenario 2: ML Model Repository
--------------------------------
setfacl -d -m u:ml_engineer:rwx /models/experiments/
setfacl -d -m g:datascience:rx /models/experiments/

Result: ML engineer can modify, data science team can read/execute

Scenario 3: Secure Logs
------------------------
setfacl -d -m u:logservice:rw /var/log/ml-training/
setfacl -d -m g:admins:r /var/log/ml-training/

Result: Log service writes, admins can read

SCENARIOS
EOF

chmod +x default_acl_demo.sh
./default_acl_demo.sh
```

## Step 3.5: ACL Reference Guide

Create a comprehensive ACL command reference.

```bash
cat > acl_reference.txt << 'EOF'
ACCESS CONTROL LISTS (ACL) COMMAND REFERENCE
============================================

VIEW ACLs:
----------
getfacl file                    View ACLs for file
getfacl -R directory            View ACLs recursively
getfacl -t file                 Tabular format
getfacl -c file                 Skip comments in output

SET USER ACLs:
--------------
setfacl -m u:username:rwx file  Set user permissions
setfacl -m u:alice:rw file      Give alice read/write
setfacl -m u:bob:r file         Give bob read-only

SET GROUP ACLs:
---------------
setfacl -m g:groupname:rw file  Set group permissions
setfacl -m g:mlteam:rwx dir/    Give mlteam full access
setfacl -m g:readonly:r file    Give readonly group read access

SET MULTIPLE ACLs:
------------------
setfacl -m u:alice:rw,u:bob:r,g:team:rx file
  Sets ACLs for alice, bob, and team group

DEFAULT ACLs (for directories):
-------------------------------
setfacl -d -m g:group:rw dir/   New files inherit group permissions
setfacl -d -m u:user:rx dir/    New files inherit user permissions

Default ACLs only affect NEW files/directories created inside

MODIFY ACLs:
------------
setfacl -m u:alice:rwx file     Modify alice's permissions
setfacl -M aclfile.txt file     Read ACLs from file

REMOVE ACLs:
------------
setfacl -x u:user file          Remove user ACL
setfacl -x g:group file         Remove group ACL
setfacl -b file                 Remove all ACLs
setfacl -k directory            Remove default ACLs

COPY ACLs:
----------
getfacl file1 | setfacl --set-file=- file2
  Copy all ACLs from file1 to file2

BACKUP/RESTORE ACLs:
--------------------
getfacl -R /project > acl_backup.txt
setfacl --restore=acl_backup.txt

RECURSIVE OPERATIONS:
---------------------
setfacl -R -m g:team:rw project/
  Apply ACL recursively to all files/directories

MASK (Maximum Effective Permissions):
--------------------------------------
setfacl -m m::rx file           Set mask to r-x
  Mask limits the maximum effective permissions for group and ACL entries

ACL NOTATION IN ls -l:
----------------------
-rw-rw-r--+  The '+' indicates ACLs are set
drwxrwxr-x+  Directory with ACLs

PRACTICAL EXAMPLES:
-------------------

1. Shared Model Directory:
   setfacl -m g:mlteam:rwx models/shared/
   setfacl -d -m g:mlteam:rw models/shared/

2. Research Collaboration:
   setfacl -m u:researcher1:rwx,u:researcher2:rx experiments/exp_001/

3. Production Model Access:
   setfacl -m u:deploy_service:r,g:sre:rx models/production/v1.0/

4. Log File Access:
   setfacl -m u:monitoring:r,g:devops:r logs/training.log

5. Sensitive Data:
   setfacl -m u:data_engineer:rw,m::r sensitive_data/
   (mask limits maximum to read, even though data_engineer has rw)

CHECKING EFFECTIVE PERMISSIONS:
--------------------------------
getfacl file | grep -A 1 "effective"
  Shows actual effective permissions considering mask

TROUBLESHOOTING:
----------------
- "Operation not supported": Filesystem doesn't support ACLs
  Solution: Remount with acl option or use different filesystem

- ACLs not inherited: Default ACLs not set
  Solution: Use -d flag when setting ACLs on directories

- Unexpected permissions: Check mask
  Solution: getfacl file and look for mask entry
EOF

cat acl_reference.txt
```

**Validation:**
- [ ] Understand what ACLs are and when to use them
- [ ] Know basic ACL commands (getfacl, setfacl)
- [ ] Understand default ACLs for directories
- [ ] Can read ACL notation
- [ ] Know when ACLs are better than traditional permissions

---

# Phase 4: Security Best Practices (30 minutes)

## Step 4.1: Implement Least Privilege Principle

Create security zones with appropriate access levels.

```bash
cd ~/ml-permissions-lab
mkdir security_patterns
cd security_patterns

# Implement security zones
mkdir -p {public,shared,restricted,private}

echo "=== Creating Security Zones ==="

# Zone 1: Public (everyone can read)
chmod 755 public
touch public/readme.md public/sample_data.csv public/documentation.pdf
chmod 644 public/*

echo "Public zone (755/644):"
ls -ld public
ls -l public/

# Zone 2: Shared (team collaboration)
chmod 775 shared
touch shared/experiment.ipynb shared/results.csv shared/analysis.py
chmod 664 shared/*

echo ""
echo "Shared zone (775/664):"
ls -ld shared
ls -l shared/

# Zone 3: Restricted (limited access)
chmod 750 restricted
touch restricted/prod_model.h5 restricted/customer_data.csv
chmod 640 restricted/*

echo ""
echo "Restricted zone (750/640):"
ls -ld restricted
ls -l restricted/

# Zone 4: Private (owner only)
chmod 700 private
touch private/api_keys.yaml private/db_password.txt private/ssh_key.pem
chmod 600 private/*

echo ""
echo "Private zone (700/600):"
ls -ld private
ls -l private/ 2>&1 || echo "(Only owner can list)"

# Create visual summary
cat > security_zones.txt << 'EOF'
SECURITY ZONES FOR ML PROJECTS
===============================

Zone: PUBLIC (755/644)
----------------------
Purpose: Documentation, public datasets, examples
Who: Everyone can read, only owner can modify
Directories: 755 (rwxr-xr-x)
Files: 644 (rw-r--r--)
Examples:
  - README.md
  - public_datasets/
  - documentation/
  - sample_code/

Zone: SHARED (775/664)
----------------------
Purpose: Team collaboration, experiments, notebooks
Who: Team can read and write, others can read
Directories: 775 (rwxrwxr-x)
Files: 664 (rw-rw-r--)
Examples:
  - notebooks/
  - experiments/
  - collaborative_datasets/
  - team_scripts/

Zone: RESTRICTED (750/640)
--------------------------
Purpose: Production models, sensitive data
Who: Owner full access, group read, others none
Directories: 750 (rwxr-x---)
Files: 640 (rw-r-----)
Examples:
  - production_models/
  - customer_data/
  - internal_reports/
  - staging_area/

Zone: PRIVATE (700/600)
-----------------------
Purpose: Credentials, API keys, secrets
Who: Owner only
Directories: 700 (rwx------)
Files: 600 (rw-------)
Examples:
  - api_keys/
  - credentials/
  - ssh_keys/
  - database_passwords/

DECISION TREE:
--------------
Is data public?
  YES → PUBLIC zone (755/644)
  NO  → Is team collaboration needed?
          YES → SHARED zone (775/664)
          NO  → Does anyone else need read access?
                  YES → RESTRICTED zone (750/640)
                  NO  → PRIVATE zone (700/600)
EOF

cat security_zones.txt
```

**Validation:**
- [ ] Four security zones created with appropriate permissions
- [ ] Understand least privilege principle
- [ ] Can determine appropriate zone for different file types

## Step 4.2: Secure File Creation

Learn to create files with secure permissions from the start.

```bash
cd ~/ml-permissions-lab/security_patterns

# Create secure file creation function
cat > secure_file_creation.sh << 'EOF'
#!/bin/bash
# Secure file creation patterns

create_secure_file() {
    local filename=$1
    local content=${2:-""}

    # Save current umask
    old_umask=$(umask)

    # Set restrictive umask (owner only)
    umask 0077

    # Create file (will have 600 permissions)
    echo "$content" > "$filename"

    # Restore previous umask
    umask $old_umask

    echo "Created secure file: $filename"
    ls -l "$filename"
}

create_team_file() {
    local filename=$1
    local content=${2:-""}

    old_umask=$(umask)
    umask 0002  # Collaborative

    echo "$content" > "$filename"

    umask $old_umask

    echo "Created team file: $filename"
    ls -l "$filename"
}

create_readonly_file() {
    local filename=$1
    local content=${2:-""}

    old_umask=$(umask)
    umask 0022  # Standard

    echo "$content" > "$filename"
    chmod 444 "$filename"  # Make read-only

    umask $old_umask

    echo "Created read-only file: $filename"
    ls -l "$filename"
}

# Demonstrations
echo "=== Secure File Creation Patterns ==="
echo ""

echo "1. Creating secure credential file:"
create_secure_file "credentials.yaml" "api_key: secret_key_123"

echo ""
echo "2. Creating team-editable file:"
create_team_file "team_config.yaml" "setting: value"

echo ""
echo "3. Creating read-only production file:"
create_readonly_file "production_model_v1.h5" "model_data"

echo ""
echo "Verification:"
ls -l credentials.yaml team_config.yaml production_model_v1.h5
EOF

chmod +x secure_file_creation.sh
./secure_file_creation.sh
```

**Expected Output:**
```
=== Secure File Creation Patterns ===

1. Creating secure credential file:
Created secure file: credentials.yaml
-rw------- 1 username username ... credentials.yaml

2. Creating team-editable file:
Created team file: team_config.yaml
-rw-rw-r-- 1 username username ... team_config.yaml

3. Creating read-only production file:
Created read-only production file: production_model_v1.h5
-r--r--r-- 1 username username ... production_model_v1.h5
```

**Validation:**
- [ ] Can create files with specific permissions using umask
- [ ] Understand temporary umask changes
- [ ] Know when to use different creation patterns

## Step 4.3: Permission Auditing

Use the audit script from the solution to check for security issues.

```bash
cd ~/ml-permissions-lab

# Copy the audit script from solutions
# For this guide, we'll create a simplified version

cat > audit_permissions_simple.sh << 'EOF'
#!/bin/bash
# Simplified permission audit script

PROJECT_ROOT="${1:-.}"

echo "=========================================="
echo " Permission Security Audit"
echo "=========================================="
echo "Project: $(basename "$PROJECT_ROOT")"
echo "Path: $PROJECT_ROOT"
echo ""

ISSUES=0
WARNINGS=0

# Check 1: World-writable files (CRITICAL)
echo "[1] Checking for world-writable files..."
WW_FILES=$(find "$PROJECT_ROOT" -type f -perm -002 2>/dev/null)
if [ -n "$WW_FILES" ]; then
    echo "✗ CRITICAL: Found world-writable files:"
    echo "$WW_FILES"
    ((ISSUES++))
else
    echo "✓ No world-writable files"
fi
echo ""

# Check 2: World-writable directories (CRITICAL)
echo "[2] Checking for world-writable directories..."
WW_DIRS=$(find "$PROJECT_ROOT" -type d -perm -002 2>/dev/null)
if [ -n "$WW_DIRS" ]; then
    echo "✗ CRITICAL: Found world-writable directories:"
    echo "$WW_DIRS"
    ((ISSUES++))
else
    echo "✓ No world-writable directories"
fi
echo ""

# Check 3: Sensitive files with wrong permissions
echo "[3] Checking sensitive files..."
SENSITIVE=$(find "$PROJECT_ROOT" -type f \( -name "*secret*" -o -name "*password*" -o -name "*key*" -o -name "*.pem" \) -perm -044 2>/dev/null)
if [ -n "$SENSITIVE" ]; then
    echo "✗ WARNING: Sensitive files readable by others:"
    echo "$SENSITIVE"
    ((WARNINGS++))
else
    echo "✓ Sensitive files properly protected"
fi
echo ""

# Check 4: 777 permissions
echo "[4] Checking for overly permissive files (777)..."
PERM_777=$(find "$PROJECT_ROOT" -type f -perm 777 2>/dev/null)
if [ -n "$PERM_777" ]; then
    echo "✗ WARNING: Files with 777 permissions:"
    echo "$PERM_777"
    ((WARNINGS++))
else
    echo "✓ No files with 777 permissions"
fi
echo ""

# Summary
echo "=========================================="
echo " Summary"
echo "=========================================="
if [ $ISSUES -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ PASSED: No security issues found"
    exit 0
elif [ $ISSUES -gt 0 ]; then
    echo "✗ FAILED: $ISSUES critical issues, $WARNINGS warnings"
    exit 1
else
    echo "⚠ WARNING: $WARNINGS warnings found"
    exit 0
fi
EOF

chmod +x audit_permissions_simple.sh

# Run audit on security_patterns directory
echo "Running audit..."
./audit_permissions_simple.sh security_patterns
```

**Create Common Issues and Fixes:**

```bash
cat > common_security_issues.md << 'EOF'
# Common Security Issues and Solutions

## Issue 1: World-Writable Files

**Symptom:**
```
-rw-rw-rw- 1 user user 1234 Nov 01 10:00 dataset.csv
```

**Problem:** Anyone on the system can modify the file

**Solution:**
```bash
chmod 644 dataset.csv  # Standard file
# or
chmod 664 dataset.csv  # Team-writable
```

**Prevention:**
```bash
# Set appropriate umask
umask 0022  # or 0002 for collaboration
```

## Issue 2: Secrets Readable by Others

**Symptom:**
```
-rw-r--r-- 1 user user 100 Nov 01 10:00 api_keys.yaml
```

**Problem:** Sensitive file readable by all users

**Solution:**
```bash
chmod 600 api_keys.yaml
# Also secure the directory
chmod 700 $(dirname api_keys.yaml)
```

**Prevention:**
```bash
# Use restrictive umask for secrets
(umask 0077 && cat > api_keys.yaml << 'EOF'
key: secret
EOF
)
```

## Issue 3: Production Models Are Writable

**Symptom:**
```
-rw-rw-r-- 1 user user 5M Nov 01 10:00 production_v1.h5
```

**Problem:** Production model can be accidentally modified

**Solution:**
```bash
chmod 444 production_v1.h5  # Read-only for all
```

**Prevention:**
```bash
# Make immutable after deployment
chmod 444 models/production/*
```

## Issue 4: Scripts Not Executable

**Symptom:**
```
-rw-r--r-- 1 user user 200 Nov 01 10:00 train.sh
bash: ./train.sh: Permission denied
```

**Problem:** Script lacks execute permission

**Solution:**
```bash
chmod +x train.sh
# or
chmod 755 train.sh
```

**Prevention:**
```bash
# Make all scripts executable during setup
find scripts/ -name "*.sh" -exec chmod +x {} \;
```

## Issue 5: Directory Not Accessible

**Symptom:**
```
drw-r--r-- 2 user user 4096 Nov 01 10:00 models
cd: models: Permission denied
```

**Problem:** Directory missing execute permission

**Solution:**
```bash
chmod +x models
# or
chmod 755 models
```

**Key Rule:** Directories always need execute (x) to access contents

## Issue 6: Team Can't Collaborate

**Symptom:**
```
drwxr-xr-x 2 user user 4096 Nov 01 10:00 shared
-rw-r--r-- 1 user user 100 Nov 01 10:00 shared/file.txt
```

**Problem:** Directory allows access but files aren't group-writable

**Solution:**
```bash
chmod 775 shared          # Allow group access
chmod 664 shared/file.txt # Allow group write
```

**Prevention:**
```bash
# Set collaborative umask
umask 0002
# Set default ACLs
setfacl -d -m g:team:rw shared/
```

## Security Checklist

Run this before deploying:

```bash
# Check for security issues
echo "Security Checklist:"

# No world-writable files
echo -n "World-writable files: "
find . -type f -perm -002 | wc -l

# No world-writable directories
echo -n "World-writable directories: "
find . -type d -perm -002 | wc -l

# Secrets are private
echo -n "Public sensitive files: "
find . -type f \( -name "*secret*" -o -name "*password*" -o -name "*.pem" \) -perm -044 | wc -l

# Scripts are executable
echo -n "Non-executable scripts: "
find . -name "*.sh" ! -executable | wc -l

echo ""
echo "All counts should be 0 for passing security"
```
EOF

cat common_security_issues.md
```

**Validation:**
- [ ] Can identify security issues
- [ ] Know how to fix common problems
- [ ] Understand prevention strategies

## Step 4.4: Automated Permission Fixing

Use the fix script to automatically correct common issues.

```bash
cd ~/ml-permissions-lab

# Create a simplified fix script
cat > fix_permissions_simple.sh << 'EOF'
#!/bin/bash
# Automated permission fixing

PROJECT_ROOT="${1:-.}"

[[ ! -d "$PROJECT_ROOT" ]] && echo "Error: Directory not found" && exit 1

echo "Fixing permissions in: $PROJECT_ROOT"
echo ""

# Fix directories (755)
echo "Setting directory permissions to 755..."
find "$PROJECT_ROOT" -type d -exec chmod 755 {} \; 2>/dev/null
echo "✓ Directories fixed"

# Fix regular files (644)
echo "Setting file permissions to 644..."
find "$PROJECT_ROOT" -type f -exec chmod 644 {} \; 2>/dev/null
echo "✓ Files fixed"

# Make scripts executable (755)
echo "Making scripts executable..."
find "$PROJECT_ROOT" -type f -name "*.sh" -exec chmod 755 {} \; 2>/dev/null
find "$PROJECT_ROOT" -type f -name "*.py" -path "*/scripts/*" -exec chmod 755 {} \; 2>/dev/null
echo "✓ Scripts made executable"

# Secure credential files (600)
echo "Securing sensitive files..."
find "$PROJECT_ROOT" -type f \( -name "*secret*" -o -name "*password*" -o -name "*key*" -o -name "credentials.*" -o -name "*.pem" \) -exec chmod 600 {} \; 2>/dev/null
echo "✓ Sensitive files secured"

# Secure private directories (700)
echo "Securing private directories..."
find "$PROJECT_ROOT" -type d \( -name "private" -o -name "secrets" \) -exec chmod 700 {} \; 2>/dev/null
echo "✓ Private directories secured"

# Set collaborative directories (775)
echo "Setting collaborative directories..."
for dir in notebooks shared experiments; do
    if [[ -d "$PROJECT_ROOT/$dir" ]]; then
        chmod 775 "$PROJECT_ROOT/$dir"
    fi
done
echo "✓ Collaborative directories set"

echo ""
echo "Permissions fixed! Run audit to verify:"
echo "  ./audit_permissions_simple.sh $PROJECT_ROOT"
EOF

chmod +x fix_permissions_simple.sh

# Test on security_patterns
echo "Testing fix script..."
./fix_permissions_simple.sh security_patterns
```

**Validation:**
- [ ] Understand what the fix script does
- [ ] Know when to use automated vs. manual fixes
- [ ] Can verify fixes with audit script

---

# Phase 5: Production Implementation (15 minutes)

## Step 5.1: Use the Complete Setup Script

Now use the production-ready script from the solutions directory.

```bash
cd ~/ml-permissions-lab

# Reference the solutions directory path
SOLUTIONS_DIR="/home/s0v3r1gn/claude/ai-infrastructure-project/repositories/solutions/ai-infra-junior-engineer-solutions/modules/mod-002-linux-essentials/exercise-02/scripts"

# Run the complete setup script
echo "Creating production ML project with proper permissions..."
$SOLUTIONS_DIR/setup_ml_permissions.sh my-production-ml-project

# Explore the created structure
cd my-production-ml-project
echo ""
echo "Project structure created:"
find . -type d | head -20

# Review the documentation
echo ""
echo "=== README.md ==="
head -30 README.md

echo ""
echo "=== PERMISSIONS.md ==="
head -40 PERMISSIONS.md
```

**Expected Output:**
```
==========================================
 ML Project Permission Setup
==========================================

Creating project: my-production-ml-project
[... setup messages ...]

Project Created Successfully!
==========================================
```

**Validation:**
- [ ] Project structure created successfully
- [ ] All directories have appropriate permissions
- [ ] Documentation files present
- [ ] Sample files created with correct permissions

## Step 5.2: Run Production Audit

Use the full-featured audit script from solutions.

```bash
cd ~/ml-permissions-lab

# Run the production audit script
echo "Running production security audit..."
$SOLUTIONS_DIR/audit_permissions.sh my-production-ml-project
```

**Expected Output:**
```
==========================================
 Permission Security Audit
==========================================

Project: my-production-ml-project
Path: /home/username/ml-permissions-lab/my-production-ml-project

[1] Checking for world-writable files...
✓ No world-writable files

[2] Checking for world-writable directories...
✓ No world-writable directories

[3] Checking for 777 permissions...
✓ No files with 777 permissions

[4] Checking sensitive files...
✓ Sensitive files properly protected

[5] Checking secrets directory...
✓ Secrets directory has correct permissions (700)

[6] Checking SUID/SGID files...
✓ No SUID/SGID files

[7] Checking script permissions...
✓ All scripts are executable

==========================================
 Audit Summary
==========================================

✓ PASSED: All security checks passed
  No critical issues found
  Total checks passed: 7
```

**Validation:**
- [ ] Audit passes all checks
- [ ] No security warnings
- [ ] Understand what each check validates

## Step 5.3: Test Permission Scenarios

Verify permissions work correctly in practice.

```bash
cd ~/ml-permissions-lab/my-production-ml-project

echo "=== Testing Permission Scenarios ==="
echo ""

# Test 1: Can create files in collaborative directories
echo "Test 1: Creating file in notebooks/"
touch notebooks/test_notebook.ipynb
ls -l notebooks/test_notebook.ipynb
echo "Expected: Should have group-writable permissions if umask is set to 0002"
echo ""

# Test 2: Can create secure files
echo "Test 2: Creating secret in secrets directory"
(umask 0077 && echo "test_secret" > configs/secrets/test_key.yaml)
ls -l configs/secrets/test_key.yaml
echo "Expected: -rw------- (600)"
echo ""

# Test 3: Scripts are executable
echo "Test 3: Running training script"
./scripts/train.py
echo "Expected: Script should execute without 'Permission denied'"
echo ""

# Test 4: Can read but not write production models
echo "Test 4: Testing production model permissions"
touch models/production/test_model.h5
chmod 444 models/production/test_model.h5
cat models/production/test_model.h5 > /dev/null 2>&1 && echo "✓ Can read production model"
echo "test" >> models/production/test_model.h5 2>&1 || echo "✓ Cannot write to production model (correct!)"
echo ""

# Test 5: Private directories are inaccessible to others
echo "Test 5: Secrets directory permissions"
ls -ld configs/secrets
echo "Expected: drwx------ (700)"
echo ""

echo "=== All Tests Complete ==="
```

**Validation:**
- [ ] Can create files in appropriate directories
- [ ] Secrets are properly protected
- [ ] Scripts are executable
- [ ] Production files are read-only
- [ ] Private directories are secure

## Step 5.4: Create Your Own ML Project

Apply everything you've learned to create a custom ML project.

```bash
cd ~/ml-permissions-lab

# Use the setup script to create your own project
$SOLUTIONS_DIR/setup_ml_permissions.sh my-custom-ml-project --collaborative

cd my-custom-ml-project

# Customize for your needs
echo ""
echo "Now customize the project:"
echo "1. Add your own directories"
echo "2. Set appropriate permissions"
echo "3. Create security zones"
echo "4. Run audit to verify"

# Example customizations:
mkdir -p data/external_apis logs/experiments results/visualizations

# Set permissions
chmod 775 data/external_apis    # Collaborative
chmod 755 logs/experiments      # Standard
chmod 755 results/visualizations # Standard

# Create sample files with proper permissions
touch data/external_apis/api_response.json
chmod 664 data/external_apis/api_response.json

touch results/visualizations/accuracy_plot.png
chmod 644 results/visualizations/accuracy_plot.png

echo ""
echo "Custom directories created:"
find . -type d -name "external_apis" -o -name "experiments" -o -name "visualizations"

echo ""
echo "Running audit on customized project..."
cd ..
$SOLUTIONS_DIR/audit_permissions.sh my-custom-ml-project
```

**Validation:**
- [ ] Created custom ML project successfully
- [ ] Added custom directories with appropriate permissions
- [ ] Audit passes on customized project
- [ ] Understand how to apply permissions to new scenarios

---

# Common Issues and Solutions

## Issue 1: Cannot Access Directory

**Symptoms:**
```bash
cd my-directory
bash: cd: my-directory: Permission denied
```

**Diagnosis:**
```bash
ls -ld my-directory
# Output: drw-r--r-- (missing execute permission)
```

**Solution:**
```bash
chmod +x my-directory
# or
chmod 755 my-directory
```

**Prevention:** Always ensure directories have execute permission (at minimum 711 for owner-only traversal)

## Issue 2: Cannot Create Files in Directory

**Symptoms:**
```bash
touch my-directory/newfile.txt
touch: cannot touch 'my-directory/newfile.txt': Permission denied
```

**Diagnosis:**
```bash
ls -ld my-directory
# Output: dr-xr-xr-x (missing write permission)
```

**Solution:**
```bash
chmod u+w my-directory  # Add write for owner
# or
chmod 755 my-directory  # Standard directory permissions
```

**Prevention:** Writable directories need both write and execute: 755 (owner write) or 775 (group write)

## Issue 3: Script Won't Execute

**Symptoms:**
```bash
./train.sh
bash: ./train.sh: Permission denied
```

**Diagnosis:**
```bash
ls -l train.sh
# Output: -rw-r--r-- (missing execute permission)
```

**Solution:**
```bash
chmod +x train.sh
# or
chmod 755 train.sh
```

**Alternative:**
```bash
# Execute with interpreter directly
bash train.sh
python train.py
```

## Issue 4: Group Members Can't Modify Files

**Symptoms:**
Team members report "Permission denied" when trying to edit shared files.

**Diagnosis:**
```bash
ls -l shared/dataset.csv
# Output: -rw-r--r-- (group doesn't have write)

groups
# Check if user is in the correct group
```

**Solution:**
```bash
# Add group write permission
chmod g+w shared/dataset.csv

# Or set collaborative permissions
chmod 664 shared/dataset.csv

# Ensure user is in the correct group (may need sudo)
sudo usermod -a -G teamgroup username

# User must log out and back in for group changes
```

**Prevention:**
```bash
# Set collaborative umask
umask 0002

# Set default ACLs on shared directories
setfacl -d -m g:teamgroup:rw shared/
```

## Issue 5: Secrets Are Readable by Others

**Symptoms:**
Security audit fails with warning about sensitive files.

**Diagnosis:**
```bash
ls -l configs/secrets/api_keys.yaml
# Output: -rw-r--r-- (readable by others)

ls -ld configs/secrets
# Output: drwxr-xr-x (directory accessible by others)
```

**Solution:**
```bash
# Secure the file
chmod 600 configs/secrets/api_keys.yaml

# Secure the directory
chmod 700 configs/secrets

# Verify
ls -ld configs/secrets
ls -l configs/secrets/api_keys.yaml
```

**Prevention:**
```bash
# Always create secrets with restrictive umask
(umask 0077 && cat > api_keys.yaml << 'EOF'
key: secret
EOF
)

# Create secrets directory with correct permissions from start
mkdir -p configs/secrets
chmod 700 configs/secrets
```

## Issue 6: umask Changes Don't Persist

**Symptoms:**
```bash
umask 0002
# Close terminal and reopen
umask
# Output: 0022 (back to default)
```

**Solution:**
```bash
# Add to ~/.bashrc for persistence
echo "umask 0002" >> ~/.bashrc

# Reload configuration
source ~/.bashrc

# Verify
umask
```

**For Project-Specific:**
```bash
# Create project environment script
cat > setup_env.sh << 'EOF'
#!/bin/bash
umask 0002
export PROJECT_ROOT="$HOME/ml-projects"
echo "Project environment configured (umask: $(umask))"
EOF

# Source when working on project
source setup_env.sh
```

## Issue 7: Cannot Change Ownership

**Symptoms:**
```bash
chown alice:mlteam dataset.csv
chown: changing ownership of 'dataset.csv': Operation not permitted
```

**Diagnosis:**
- Changing ownership requires root privileges
- Regular users can only change group if they're members of both groups

**Solution:**
```bash
# Use sudo for chown
sudo chown alice:mlteam dataset.csv

# For group changes, if you're in both groups:
chgrp mlteam dataset.csv  # No sudo needed
```

**Alternative:**
```bash
# Work within your own directories
# Use ACLs instead of changing ownership
setfacl -m u:alice:rw dataset.csv
setfacl -m g:mlteam:rw dataset.csv
```

## Issue 8: Production Model Was Accidentally Modified

**Symptoms:**
Production model file was overwritten or corrupted.

**Diagnosis:**
```bash
ls -l models/production/v1.2.3.h5
# Output: -rw-rw-r-- (writable by owner and group)
```

**Solution:**
```bash
# Restore from backup if available
cp backups/v1.2.3.h5 models/production/

# Make read-only to prevent future accidents
chmod 444 models/production/v1.2.3.h5

# Verify
ls -l models/production/v1.2.3.h5
# Output: -r--r--r-- (read-only for everyone)
```

**Prevention:**
```bash
# Always make production models read-only
deploy_model() {
    local model=$1
    cp "$model" models/production/
    chmod 444 models/production/$(basename "$model")
    echo "Deployed read-only: $(basename "$model")"
}
```

---

# Best Practices Summary

## For Files

1. **Source Code and Documents: 644 (rw-r--r--)**
   - Owner can edit
   - Others can read
   - Not executable

2. **Scripts and Executables: 755 (rwxr-xr-x)**
   - Owner can edit and execute
   - Others can execute
   - Use for .sh, .py in scripts/

3. **Team-Editable Files: 664 (rw-rw-r--)**
   - Owner and group can edit
   - Others can read
   - Use for collaborative work

4. **Private Files (Credentials): 600 (rw-------)**
   - Owner only
   - Use for API keys, passwords, certificates

5. **Read-Only (Production): 444 (r--r--r--)**
   - Nobody can modify
   - Use for deployed production models

## For Directories

1. **Public Directories: 755 (rwxr-xr-x)**
   - Owner can modify contents
   - Others can access and list
   - Use for documentation, public data

2. **Collaborative Directories: 775 (rwxrwxr-x)**
   - Owner and group can modify
   - Others can access
   - Use for team projects, shared notebooks

3. **Private Directories: 700 (rwx------)**
   - Owner only
   - Use for secrets, credentials, private keys

4. **Restricted Directories: 750 (rwxr-x---)**
   - Owner full access
   - Group can access but not modify
   - Others have no access
   - Use for production, sensitive data

## For ML Projects

| Component | Directory Perm | File Perm | Rationale |
|-----------|---------------|-----------|-----------|
| Raw datasets | 755 | 644 | Immutable, everyone reads |
| Processed data | 775 | 664 | Team collaboration |
| Checkpoints | 775 | 664 | Team shares progress |
| Production models | 755 | 444 | Read-only deployment |
| Notebooks | 775 | 664 | Team collaboration |
| Scripts | 755 | 755 | Executable by all |
| Configs | 755 | 644 | Readable configuration |
| Secrets | 700 | 600 | Private credentials |
| Logs | 755 | 644 | System writes, team reads |

## General Principles

1. **Least Privilege**: Grant minimum necessary permissions
2. **Defense in Depth**: Secure both directory and files
3. **Immutable Production**: Make production artifacts read-only
4. **Audit Regularly**: Run permission audits weekly
5. **Document Policy**: Maintain PERMISSIONS.md in each project
6. **Use umask**: Set appropriate default permissions
7. **Leverage Groups**: Use groups for team collaboration
8. **ACLs for Exceptions**: Use ACLs when standard permissions insufficient

## Automation

```bash
# Set up project correctly from start
./setup_ml_permissions.sh project-name

# Regular audits
./audit_permissions.sh project-name

# Fix common issues
./fix_permissions.sh project-name

# Add to cron for regular audits
0 0 * * 0 /path/to/audit_permissions.sh /projects >> /var/log/permission_audit.log
```

---

# Completion Checklist

Use this checklist to verify you've completed all learning objectives:

## Understanding Permissions
- [ ] Can read and interpret permission strings (-rwxr-xr--)
- [ ] Understand numeric permission notation (644, 755, etc.)
- [ ] Know the difference between r, w, x for files and directories
- [ ] Can calculate permissions from symbolic to numeric and vice versa
- [ ] Understand file ownership (user, group, others)
- [ ] Know what umask is and how it affects new files

## Using chmod
- [ ] Can set permissions with numeric mode (chmod 644 file)
- [ ] Can modify permissions with symbolic mode (chmod u+x file)
- [ ] Can set permissions recursively (chmod -R)
- [ ] Know common permission patterns for different file types
- [ ] Can make files executable
- [ ] Can remove permissions

## Ownership Management
- [ ] Understand chown and chgrp commands
- [ ] Know when sudo is required
- [ ] Can check file ownership with ls -l
- [ ] Understand user and group IDs

## Advanced ACLs
- [ ] Know what ACLs are and when to use them
- [ ] Can view ACLs with getfacl
- [ ] Can set user-specific ACLs with setfacl
- [ ] Can set group ACLs
- [ ] Understand default ACLs for directories
- [ ] Can remove ACLs

## Security Best Practices
- [ ] Understand least privilege principle
- [ ] Can identify security zones (public, shared, restricted, private)
- [ ] Know how to secure sensitive files (600/700)
- [ ] Can make production files read-only (444)
- [ ] Understand the importance of auditing
- [ ] Can use security audit scripts

## ML Project Permissions
- [ ] Can set up an ML project with proper permissions
- [ ] Know appropriate permissions for datasets
- [ ] Understand model checkpoint permissions
- [ ] Can secure production models
- [ ] Know how to set up team collaboration
- [ ] Can secure credentials and secrets

## Practical Skills
- [ ] Created and configured multiple test directories
- [ ] Successfully used setup script for ML project
- [ ] Ran security audit and understood results
- [ ] Fixed permission issues
- [ ] Created custom permission structures
- [ ] Verified permissions with stat and ls

## Problem Solving
- [ ] Can diagnose permission errors
- [ ] Know how to fix "Permission denied" errors
- [ ] Can troubleshoot group access issues
- [ ] Understand how to prevent security issues
- [ ] Can recover from permission mistakes

---

# Next Steps

## Immediate Next Steps

1. **Practice More Scenarios**
   ```bash
   # Create different project types
   ./setup_ml_permissions.sh research-project
   ./setup_ml_permissions.sh production-deployment
   ./setup_ml_permissions.sh team-collaboration
   ```

2. **Customize Permission Policies**
   - Create custom security zones for your organization
   - Document your team's permission standards
   - Set up project templates

3. **Automate Permission Management**
   - Add permission audits to CI/CD pipelines
   - Create pre-commit hooks to check permissions
   - Schedule regular security audits

## Continue Learning

### Next Exercise
**Exercise 03: Process Management**
- Monitor ML training processes
- Control resource usage
- Manage background jobs
- Handle process signals

### Related Topics
- **Exercise 04: Shell Scripting** - Automate permission management
- **Module 003: Git Version Control** - Permissions in repositories
- **Module 009: Monitoring** - Security monitoring and alerting

## Advanced Topics to Explore

1. **SELinux and AppArmor**
   - Mandatory Access Control (MAC)
   - Security contexts
   - Policy management

2. **File Capabilities**
   - Fine-grained privilege management
   - Alternative to SUID/SGID

3. **Audit Logging**
   - Track permission changes
   - Monitor file access
   - Security compliance

4. **Automated Compliance**
   - OSSEC file integrity monitoring
   - Tripwire
   - Custom compliance scripts

## Real-World Application

Apply these skills to:
- Secure your actual ML projects
- Set up team collaboration environments
- Implement security best practices in production
- Contribute to open-source ML projects with proper permissions

---

# Additional Resources

## Official Documentation
- [Linux File Permissions](https://www.linux.com/training-tutorials/understanding-linux-file-permissions/)
- [chmod Manual](https://man7.org/linux/man-pages/man1/chmod.1.html)
- [ACL Tutorial](https://www.redhat.com/sysadmin/linux-access-control-lists)
- [umask Guide](https://www.cyberciti.biz/tips/understanding-linux-unix-umask-value-usage.html)

## Security Standards
- [CIS Benchmarks](https://www.cisecurity.org/)
- [NIST Security Guidelines](https://www.nist.gov/)
- [OWASP Security Practices](https://owasp.org/)

## Interactive Learning
- [Linux Journey - Permissions](https://linuxjourney.com/)
- [OverTheWire - Bandit](https://overthewire.org/wargames/bandit/)

## Books
- "UNIX and Linux System Administration Handbook" - Chapter on Permissions
- "Linux Security Cookbook" - Permission Management Recipes

## Video Tutorials
- Search for "Linux file permissions tutorial"
- "ACL tutorial Linux"
- "umask explained"

## Practice Environments
- Set up your own Linux VM for practice
- Use Docker containers for isolated testing
- Cloud platforms (AWS, GCP, Azure) free tiers

---

# Congratulations!

You've completed the File Permissions and Access Control implementation guide! You now have the skills to:

- Configure secure file permissions for ML infrastructure
- Implement team collaboration with appropriate access controls
- Use advanced ACLs for fine-grained permissions
- Apply security best practices to production ML systems
- Audit and fix permission issues
- Create and maintain secure ML project structures

**Key Achievement**: You can now set up and maintain secure, collaborative ML infrastructure with industry-standard permission management.

**Time Investment**: 90-120 minutes
**Skills Gained**: Production-ready Linux permission management for AI/ML teams

---

**Exercise 02: File Permissions and Access Control - ✅ COMPLETE**

Continue to Exercise 03: Process Management to learn how to monitor and control ML training processes.
