# Exercise 02: File Permissions and Access Control - Solution

## Overview

This solution demonstrates comprehensive Linux file permissions and access control management for multi-user ML infrastructure environments. It covers traditional Unix permissions, Access Control Lists (ACLs), umask, and security best practices for ML teams.

## Learning Objectives Covered

- ✅ Understand Linux permission model (user, group, other)
- ✅ Use chmod with numeric and symbolic notation
- ✅ Manage ownership with chown and chgrp
- ✅ Implement Access Control Lists (ACLs) for fine-grained access
- ✅ Configure default permissions with umask
- ✅ Apply security best practices for ML infrastructure
- ✅ Create secure shared directories for team collaboration

## Solution Structure

```
exercise-02/
├── README.md                          # This file - solution overview
├── IMPLEMENTATION_GUIDE.md            # Step-by-step implementation guide
├── scripts/                           # Permission management scripts
│   ├── setup_ml_permissions.sh       # Configure ML project with proper permissions
│   ├── audit_permissions.sh          # Security audit for permissions
│   ├── fix_permissions.sh            # Automated permission fixing
│   ├── setup_team_collaboration.sh   # Team collaboration environment
│   └── permission_calculator.sh      # Interactive permission calculator
├── examples/                          # Example permission scenarios
│   └── ml_project_structure/         # Sample ML project with permissions
└── docs/
    └── ANSWERS.md                    # Reflection question answers
```

## Key Concepts

### Permission Model

#### Permission String Format
```
-rwxr-xr--
│││││││││└─ Other execute
││││││││└── Other write
│││││││└─── Other read
││││││└──── Group execute
│││││└───── Group write
││││└────── Group read
│││└─────── User/Owner execute
││└──────── User/Owner write
│└───────── User/Owner read
└────────── File type (- = file, d = directory, l = link)
```

#### Numeric Permissions
```
r (read)    = 4
w (write)   = 2
x (execute) = 1

Common combinations:
0 = ---  (no permissions)
4 = r--  (read only)
5 = r-x  (read and execute)
6 = rw-  (read and write)
7 = rwx  (full permissions)

Complete permission examples:
644 = rw-r--r--  (standard file)
755 = rwxr-xr-x  (executable/directory)
600 = rw-------  (private file)
700 = rwx------  (private directory)
775 = rwxrwxr-x  (group collaborative)
664 = rw-rw-r--  (group editable file)
```

### ML Project Permission Strategy

#### Directory Permissions

| Directory | Permissions | Purpose |
|-----------|-------------|---------|
| `datasets/raw/` | 755 | Immutable data - owner modifies, all read |
| `datasets/processed/` | 775 | Team collaborative data processing |
| `models/checkpoints/` | 775 | Team shares training checkpoints |
| `models/production/` | 755 | Production models - controlled access |
| `notebooks/` | 775 | Team collaborative notebooks |
| `scripts/` | 755 | Executable scripts - team can run |
| `configs/` | 755 | Configuration files - controlled |
| `configs/secrets/` | 700 | API keys, credentials - owner only |
| `logs/` | 755 | Logs - system writes, team reads |

#### File Permissions

| File Type | Permissions | Purpose |
|-----------|-------------|---------|
| Source code | 644 | Read/write owner, read others |
| Scripts (*.sh, *.py) | 755 | Executable by all |
| Data files | 664 | Team editable |
| Production models | 644 | Read-only after deployment |
| Config files | 644 | Readable by team |
| Secrets | 600 | Owner only |
| Logs | 644 | Readable by team |

## Quick Start

### 1. Set Up ML Project with Proper Permissions

```bash
cd scripts
./setup_ml_permissions.sh my-ml-project
```

This creates a complete ML project structure with security-appropriate permissions:
```
my-ml-project/
├── datasets/
│   ├── raw/           (755 - immutable)
│   ├── processed/     (775 - collaborative)
│   └── external/      (755 - read-only)
├── models/
│   ├── checkpoints/   (775 - collaborative)
│   └── production/    (755 - controlled)
├── notebooks/         (775 - collaborative)
├── scripts/           (755 - executable)
├── configs/
│   ├── training/      (755)
│   └── secrets/       (700 - private)
├── logs/              (755 - system writable)
└── shared/            (775 - team collaboration)
```

### 2. Run Security Audit

```bash
./audit_permissions.sh my-ml-project
```

Checks for:
- World-writable files (security risk)
- Overly permissive directories
- Sensitive files with incorrect permissions
- SUID/SGID files
- Executable files review

### 3. Fix Common Permission Issues

```bash
./fix_permissions.sh my-ml-project
```

Automatically:
- Sets directory permissions to 755
- Sets file permissions to 644
- Makes scripts executable (755)
- Secures sensitive files (600)
- Protects secret directories (700)

### 4. Calculate Permissions Interactively

```bash
./permission_calculator.sh
```

Interactive tool to:
- Convert symbolic to numeric permissions
- Convert numeric to symbolic permissions
- Explain permission meanings
- Practice permission calculations

## Permission Commands Reference

### chmod - Change Permissions

#### Numeric Mode
```bash
# Set exact permissions
chmod 644 file.txt              # rw-r--r--
chmod 755 script.sh             # rwxr-xr-x
chmod 600 secret.key            # rw-------
chmod 700 private_dir/          # rwx------

# Recursive
chmod -R 755 directory/
```

#### Symbolic Mode
```bash
# Add permissions
chmod u+x script.sh             # Add execute for owner
chmod g+w dataset.csv           # Add write for group
chmod a+r readme.md             # Add read for all

# Remove permissions
chmod o-r secret.txt            # Remove read for others
chmod g-w file.txt              # Remove write for group

# Set exact permissions
chmod u=rwx,g=rx,o=r file      # rwxr-xr--

# Multiple changes
chmod u+x,g+x,o-r script.sh    # Add execute for user/group, remove read for others

# Recursive
chmod -R g+w shared/           # Add group write recursively
```

### chown - Change Ownership

```bash
# Change owner
sudo chown alice dataset.csv

# Change owner and group
sudo chown alice:mlteam model.h5

# Recursive
sudo chown -R alice:mlteam datasets/

# Change only group (alternative to chgrp)
sudo chown :mlteam shared_models/
```

### chgrp - Change Group

```bash
# Change group
sudo chgrp mlteam experiments/

# Recursive
sudo chgrp -R mlteam datasets/
```

### umask - Default Permissions

```bash
# View current umask
umask                  # Numeric (e.g., 0022)
umask -S              # Symbolic (e.g., u=rwx,g=rx,o=rx)

# Set umask
umask 0022            # Files: 644, Dirs: 755 (standard)
umask 0002            # Files: 664, Dirs: 775 (collaborative)
umask 0077            # Files: 600, Dirs: 700 (private)

# Make permanent (add to ~/.bashrc)
echo "umask 0002" >> ~/.bashrc
source ~/.bashrc
```

### ACLs - Advanced Access Control

```bash
# View ACLs
getfacl file.txt

# Set user ACL
setfacl -m u:alice:rw file.txt

# Set group ACL
setfacl -m g:mlteam:rwx directory/

# Set default ACL for new files in directory
setfacl -d -m g:mlteam:rw datasets/

# Remove specific ACL
setfacl -x u:alice file.txt

# Remove all ACLs
setfacl -b file.txt

# Copy ACL from one file to another
getfacl file1 | setfacl --set-file=- file2
```

## Real-World ML Scenarios

### Scenario 1: Shared Dataset Repository

**Requirement**: Multiple data scientists need read access to raw data. Only data engineers can modify it.

**Solution**:
```bash
mkdir -p datasets/raw
chmod 755 datasets/raw                    # Anyone can read, only owner can write

touch datasets/raw/imagenet_train.tar
chmod 644 datasets/raw/imagenet_train.tar  # Read-only for all

# Alternative with ACL for specific users
setfacl -m u:data_engineer:rw datasets/raw/imagenet_train.tar
setfacl -m g:ml_team:r datasets/raw/imagenet_train.tar
```

### Scenario 2: Collaborative Model Training

**Requirement**: Team members need to share training checkpoints and experiment results.

**Solution**:
```bash
mkdir -p models/checkpoints
chmod 775 models/checkpoints              # Team can read/write/access

# Set default ACL for new checkpoint files
setfacl -d -m g:ml_team:rw models/checkpoints/

# Create checkpoint
touch models/checkpoints/epoch_050.h5
# Automatically gets group write permission
```

### Scenario 3: Production Model Deployment

**Requirement**: Production models should be read-only to prevent accidental modification.

**Solution**:
```bash
mkdir -p models/production
chmod 755 models/production               # Controlled directory access

# Deploy model (make read-only)
cp models/checkpoints/best_model.h5 models/production/v1.2.3.h5
chmod 444 models/production/v1.2.3.h5    # Read-only for everyone (cannot be modified)
```

### Scenario 4: Secure Credentials Management

**Requirement**: API keys and database passwords should only be accessible by the service owner.

**Solution**:
```bash
mkdir -p configs/secrets
chmod 700 configs/secrets                 # Only owner can access directory

# Create secret file with restrictive umask
(umask 0077 && cat > configs/secrets/api_keys.yaml << 'EOF'
aws_access_key: AKIA...
aws_secret_key: secret...
EOF
)

chmod 600 configs/secrets/api_keys.yaml  # Owner read/write only
```

### Scenario 5: Team Notebooks Collaboration

**Requirement**: Team members should be able to edit shared Jupyter notebooks.

**Solution**:
```bash
mkdir -p notebooks
chmod 775 notebooks                       # Team collaborative

# Set umask for notebook creation
umask 0002

# Create notebook (will have 664 permissions)
touch notebooks/exploration.ipynb
# Result: rw-rw-r-- (owner and group can edit)
```

## Security Best Practices

### 1. Principle of Least Privilege

Give users only the minimum permissions needed:

```bash
# Bad: Everything writable
chmod 777 datasets/  # ✗ DANGEROUS

# Good: Appropriate permissions
chmod 755 datasets/  # ✓ Owner writes, others read
```

### 2. Protect Sensitive Files

```bash
# Credentials and secrets
chmod 600 *.pem *.key credentials.* secrets.*

# Private directories
chmod 700 ~/.ssh/ ~/credentials/
```

### 3. Review Permissions Regularly

```bash
# Find overly permissive files
find . -type f -perm -002  # World-writable files
find . -type d -perm -002  # World-writable directories

# Find files with 777 permissions
find . -perm 777
```

### 4. Use Groups for Team Collaboration

```bash
# Create ML team group
sudo groupadd mlteam

# Add users to group
sudo usermod -a -G mlteam alice
sudo usermod -a -G mlteam bob

# Set group ownership
sudo chgrp -R mlteam /shared/ml-projects/

# Set group permissions
chmod -R g+w /shared/ml-projects/
```

### 5. Set Appropriate Default Permissions

```bash
# In ~/.bashrc for team collaboration
umask 0002  # New files: 664, New dirs: 775

# Or for security-sensitive work
umask 0077  # New files: 600, New dirs: 700
```

## Permission Audit Checklist

- [ ] No world-writable files or directories
- [ ] Credentials have 600 permissions
- [ ] Secret directories have 700 permissions
- [ ] Scripts are executable (755)
- [ ] Production files are read-only
- [ ] Team directories have group write (775)
- [ ] Appropriate umask is set
- [ ] Groups are configured correctly
- [ ] Regular audit is scheduled

## Common Issues and Solutions

### Issue 1: Cannot Access Directory

**Symptom**: `Permission denied` when cd into directory

**Cause**: Directory doesn't have execute permission

**Solution**:
```bash
chmod +x directory/    # Add execute permission
# or
chmod 755 directory/   # Set standard directory permissions
```

### Issue 2: Cannot Create Files in Directory

**Symptom**: `Permission denied` when creating files

**Cause**: Directory doesn't have write permission

**Solution**:
```bash
chmod u+w directory/   # Add write for owner
# or
chmod 775 directory/   # Allow group write
```

### Issue 3: Script Not Executable

**Symptom**: `Permission denied` when running script

**Cause**: Script doesn't have execute permission

**Solution**:
```bash
chmod +x script.sh     # Add execute permission
# or
chmod 755 script.sh    # Standard script permissions
```

### Issue 4: Group Permissions Not Working

**Symptom**: Group members cannot access files

**Cause**: User not in group, or group doesn't own files

**Solution**:
```bash
# Check user's groups
groups username

# Add user to group
sudo usermod -a -G groupname username

# Change file group ownership
sudo chgrp -R groupname directory/

# User must log out and back in for group membership to take effect
```

### Issue 5: New Files Have Wrong Permissions

**Symptom**: New files created with incorrect permissions

**Cause**: umask not set correctly

**Solution**:
```bash
# Check current umask
umask

# Set appropriate umask
umask 0002  # For collaboration

# Make permanent
echo "umask 0002" >> ~/.bashrc
```

## Testing the Solution

### 1. Create Test ML Project
```bash
cd scripts
./setup_ml_permissions.sh test-ml-project
```

### 2. Verify Permissions
```bash
cd test-ml-project
ls -la
find . -ls | head -20
```

### 3. Run Security Audit
```bash
cd ../scripts
./audit_permissions.sh test-ml-project
```

### 4. Test Permission Changes
```bash
# Create file in collaborative directory
touch test-ml-project/notebooks/test.ipynb
ls -l test-ml-project/notebooks/test.ipynb
# Should have group write permission

# Try to create file in secrets directory
touch test-ml-project/configs/secrets/test.key
ls -l test-ml-project/configs/secrets/test.key
# Should be owner-only (600)
```

## Integration with Exercise 01

- Uses navigation skills from Exercise 01
- Applies file operations to permission management
- Extends ML project structures with proper security
- Builds on directory organization principles

## Skills Acquired

- ✅ Linux permission model mastery
- ✅ Numeric and symbolic chmod proficiency
- ✅ Ownership management with chown/chgrp
- ✅ ACL implementation for fine-grained control
- ✅ umask configuration and defaults
- ✅ Security best practices for ML infrastructure
- ✅ Team collaboration environment setup
- ✅ Permission auditing and fixing

## Time to Complete

- **Reading specifications**: 15 minutes
- **Understanding permission model**: 20 minutes
- **Practicing chmod commands**: 25 minutes
- **Implementing security scenarios**: 30 minutes
- **Testing and validation**: 20 minutes
- **Total**: 75-110 minutes

## Next Steps

- Complete Exercise 03: Process Management
- Learn about process monitoring for ML training jobs
- Understand resource management in Linux
- Apply permission skills to process control

## Resources

- [Linux File Permissions Tutorial](https://www.linux.com/training-tutorials/understanding-linux-file-permissions/)
- [ACL Tutorial](https://www.redhat.com/sysadmin/linux-access-control-lists)
- [umask Guide](https://www.cyberciti.biz/tips/understanding-linux-unix-umask-value-usage.html)
- [Security Best Practices](https://www.cisecurity.org/)

## Conclusion

This solution provides comprehensive tools and knowledge for managing file permissions in ML infrastructure environments. The skills learned here are essential for maintaining secure, collaborative ML projects with appropriate access control.

**Key Achievement**: Complete implementation of permission management for ML teams with automated setup, auditing, and fixing capabilities.

---

**Exercise 02: File Permissions and Access Control - ✅ COMPLETE**
