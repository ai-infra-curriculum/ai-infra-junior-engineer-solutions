# ML Version Control - Quick Reference

## Git LFS Commands

### Setup and Configuration

```bash
# Install Git LFS (one-time)
git lfs install

# Install for system (all repos)
git lfs install --system

# Check version
git lfs version

# Check installation
git lfs env
```

### Tracking Files

```bash
# Track file pattern
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "models/**/*.bin"

# Track specific file
git lfs track "model_large.pth"

# List tracked patterns
git lfs track

# Stop tracking pattern
git lfs untrack "*.pth"

# View .gitattributes
cat .gitattributes
```

### Working with LFS Files

```bash
# Add LFS file (automatic if pattern tracked)
git add model.pth
git commit -m "Add model"

# Check LFS status
git lfs status

# List LFS files in repository
git lfs ls-files

# List LFS files with sizes
git lfs ls-files -s

# Show file info
git lfs ls-files | grep model.pth
```

### Fetching and Pulling

```bash
# Clone with LFS files
git clone <repo-url>

# Clone without downloading LFS files
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>

# Download LFS files after clone
git lfs pull

# Fetch LFS files for current branch
git lfs fetch

# Fetch LFS files for specific remote
git lfs fetch origin

# Fetch recent LFS files only
git lfs fetch --recent

# Pull with LFS
git pull
# (automatically pulls LFS files)
```

### Pushing

```bash
# Push with LFS files
git push origin main
# (automatically uploads LFS files)

# Push only LFS files
git lfs push origin main

# Push all LFS files
git lfs push --all origin
```

### Managing LFS Cache

```bash
# Check LFS cache
du -sh .git/lfs

# Prune old LFS files
git lfs prune

# Dry run prune
git lfs prune --dry-run --verbose

# Prune with verification
git lfs prune --verify-remote

# Force prune
git lfs prune --force
```

### Checking File Status

```bash
# Check if file is LFS pointer
cat model.pth
# Shows:
# version https://git-lfs.github.com/spec/v1
# oid sha256:...
# size 98543210

# Get actual file size
git lfs ls-files -s | grep model.pth

# Verify LFS objects
git lfs fsck
```

### Migration

```bash
# Migrate existing files to LFS
git lfs migrate import --include="*.pth"

# Migrate with history rewrite
git lfs migrate import --include="*.pth" --everything

# Show what would be migrated
git lfs migrate info --include="*.pth"

# Migrate specific branch
git lfs migrate import --include="*.pth" --include-ref=refs/heads/main
```

### Troubleshooting

```bash
# Check LFS pointer vs actual file
file model.pth
# Should show: ASCII text (pointer) or data (actual file)

# Download specific file
git lfs pull --include="models/model.pth"

# Exclude files from pull
git lfs pull --exclude="old_models/*"

# Fix pointer files
git lfs checkout

# Check LFS logs
git lfs logs last
```

## DVC Commands

### Setup and Configuration

```bash
# Install DVC
pip install dvc
pip install dvc[s3]  # With S3 support
pip install dvc[gs]  # With Google Cloud support

# Initialize DVC in repository
dvc init

# Check version
dvc version

# Show DVC configuration
dvc config --list
```

### Remote Storage

```bash
# Add local remote
dvc remote add -d local /tmp/dvc-storage

# Add S3 remote
dvc remote add -d s3 s3://my-bucket/dvc-storage

# Add Google Cloud Storage remote
dvc remote add -d gs gs://my-bucket/dvc-storage

# Add Azure Blob Storage remote
dvc remote add -d azure azure://my-container/dvc-storage

# Add SSH remote
dvc remote add -d ssh ssh://user@server/path/to/storage

# List remotes
dvc remote list

# Set default remote
dvc remote default s3

# Remove remote
dvc remote remove local

# Modify remote URL
dvc remote modify s3 url s3://new-bucket/dvc-storage
```

### Tracking Data

```bash
# Track file with DVC
dvc add data/train.csv

# This creates:
# - data/train.csv.dvc  (tracked in Git)
# - Updates .gitignore   (ignores train.csv)

# Track directory
dvc add data/raw/

# Commit .dvc file to Git
git add data/train.csv.dvc .gitignore
git commit -m "Track training data"
```

### Pushing and Pulling Data

```bash
# Push data to remote
dvc push

# Push specific file
dvc push data/train.csv.dvc

# Push to specific remote
dvc push -r s3

# Pull data from remote
dvc pull

# Pull specific file
dvc pull data/train.csv.dvc

# Pull from specific remote
dvc pull -r s3
```

### Checking Status

```bash
# Check DVC status
dvc status

# Check status for specific remote
dvc status -r s3

# Check if data matches remote
dvc status --cloud

# List DVC-tracked files
dvc list . --dvc-only

# Show data statistics
dvc data ls
```

### Fetching Data

```bash
# Fetch data without checking out
dvc fetch

# Fetch from specific remote
dvc fetch -r s3

# Fetch all branches and tags
dvc fetch --all-branches --all-tags

# Checkout fetched data
dvc checkout
```

### Versioning Data

```bash
# Modify data
# ... update data/train.csv ...

# Update DVC tracking
dvc add data/train.csv

# Commit new version
git add data/train.csv.dvc
git commit -m "Update training data v2"

# Push new version
dvc push

# Checkout old version
git checkout HEAD~1 data/train.csv.dvc
dvc checkout

# Return to latest
git checkout main data/train.csv.dvc
dvc checkout
```

### Pipelines

```bash
# Create pipeline stage
dvc run -n preprocess \
  -d data/raw/train.csv \
  -o data/processed/train_clean.csv \
  python scripts/preprocess.py

# Run pipeline
dvc repro

# Show pipeline
dvc dag

# Show pipeline as ASCII
dvc dag --ascii

# Check pipeline status
dvc status
```

### Metrics and Params

```bash
# Track metrics file
dvc metrics show

# Show metrics for specific file
dvc metrics show metrics.json

# Compare metrics across commits
dvc metrics diff

# Track parameters
dvc params show

# Compare parameters
dvc params diff
```

### Managing Cache

```bash
# Show cache directory
dvc cache dir

# Check cache status
du -sh .dvc/cache

# Clean cache
dvc gc

# Clean with workspace check
dvc gc --workspace

# Clean keeping specific branches
dvc gc --all-branches --all-tags

# Dry run cleanup
dvc gc --dry --verbose
```

### Import and Get

```bash
# Import data from URL
dvc import https://example.com/data.csv data/external.csv

# Import from another DVC repository
dvc import git@github.com:user/repo data/train.csv

# Get file from DVC repository without tracking
dvc get git@github.com:user/repo data/train.csv

# Update imported data
dvc update data/external.csv.dvc
```

### Troubleshooting

```bash
# Check DVC installation
dvc doctor

# Verify data integrity
dvc checkout --relink

# Force checkout
dvc checkout --force

# Show DVC file contents
cat data/train.csv.dvc

# Debug mode
dvc --verbose pull
dvc --verbose push
```

## Git LFS + DVC Workflow

### Complete Setup

```bash
# Initialize repository
git init
git lfs install
dvc init

# Configure LFS for models
cat >> .gitattributes << EOF
*.pth filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
EOF

# Configure DVC for data
dvc remote add -d s3 s3://my-bucket/dvc-storage

# Commit configuration
git add .gitattributes .dvc/config
git commit -m "Configure LFS and DVC"
```

### Adding Data and Models

```bash
# Track dataset with DVC
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Track training data"
dvc push

# Track model with LFS
git add model.pth
git commit -m "Add trained model"
git push
```

### Cloning and Setup

```bash
# Clone repository
git clone <repo-url>
cd <repo>

# Pull DVC data
dvc pull

# LFS files pulled automatically with git clone
```

### Updating Versions

```bash
# Update data
# ... modify data/train.csv ...
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "Update training data v2"
dvc push

# Update model
# ... train new model.pth ...
git add model.pth
git commit -m "Update model v2"
git push
```

## Model Versioning Workflow

### Creating Model Release

```bash
# Train model
python train.py --config experiments/exp-001.yaml

# Save model and metadata
# Creates:
# - models/resnet50/model_v1.0.0.pth
# - models/resnet50/model_v1.0.0.json

# Add to Git (LFS automatic)
git add models/resnet50/model_v1.0.0.*
git commit -m "model: add ResNet-50 v1.0.0"

# Tag release
git tag -a model-v1.0.0 -m "Model v1.0.0 release

Test accuracy: 84.8%
Training: exp-001
Dataset: ImageNet v2023.1"

# Push
git push origin main
git push origin model-v1.0.0
```

### Checking Out Model Version

```bash
# List model tags
git tag -l "model-*"

# Checkout specific version
git checkout model-v1.0.0

# View model files
ls models/resnet50/

# View metadata
cat models/resnet50/model_v1.0.0.json

# Return to latest
git checkout main
```

### Comparing Models

```bash
# Compare two versions
git diff model-v1.0.0 model-v1.1.0 -- models/

# Show metadata differences
git diff model-v1.0.0:models/resnet50/model_v1.0.0.json \
         model-v1.1.0:models/resnet50/model_v1.1.0.json

# Compare experiment configs
git diff model-v1.0.0 model-v1.1.0 -- experiments/
```

## Experiment Tracking Workflow

### Create New Experiment

```bash
# Copy baseline config
cp experiments/exp-001-baseline.yaml experiments/exp-005-new-idea.yaml

# Edit configuration
vim experiments/exp-005-new-idea.yaml
# Update: id, description, hyperparameters

# Commit experiment config
git add experiments/exp-005-new-idea.yaml
git commit -m "experiment: new idea (exp-005)

Testing hypothesis: X will improve Y
Changes: A, B, C"
```

### Run and Track Experiment

```bash
# Run training
python train.py experiments/exp-005-new-idea.yaml

# Script updates config with results
# Commits results automatically or manually:

git add experiments/exp-005-new-idea.yaml
git commit -m "results: exp-005 completed

Test accuracy: 87.5%
Training time: 42.3 hours
Conclusion: Hypothesis confirmed!"
```

### Compare Experiments

```bash
# List all experiments
ls experiments/*.yaml

# Compare two experiments
diff experiments/exp-001-baseline.yaml \
     experiments/exp-005-new-idea.yaml

# Extract results
grep "test_accuracy" experiments/*.yaml

# View experiment history
git log --oneline experiments/
```

## Reproducibility Checklist

### Before Training

```bash
# Check Git status
git status
# Should be clean

# Check current commit
git rev-parse HEAD

# Verify data version
cat data/raw/train.csv.dvc | grep md5

# Check dependencies
pip freeze | grep -E "torch|numpy"

# Verify random seed in config
grep seed experiments/exp-001.yaml
```

### During Training

```bash
# Record provenance
python << EOF
import git
repo = git.Repo('.')
print(f"Commit: {repo.head.commit.hexsha}")
print(f"Branch: {repo.active_branch.name}")
print(f"Dirty: {repo.is_dirty()}")
EOF

# Save environment
pip freeze > requirements-frozen.txt
conda env export > environment-frozen.yaml
```

### After Training

```bash
# Validate model metadata
python scripts/check_model_metadata.py models/*/model_v*.json

# Validate experiment config
python scripts/validate_experiment.py experiments/exp-*.yaml

# Commit everything
git add models/ experiments/ requirements-frozen.txt
git commit -m "Complete training: exp-001

Model: v1.0.0
Accuracy: 84.8%
Reproducible: ✓"

# Tag if successful
git tag -a model-v1.0.0 -m "Production model"
```

## Common Patterns

### Pattern 1: New Dataset Version

```bash
# Update dataset
# ... download new data to data/raw/train.csv ...

# Update DVC tracking
dvc add data/raw/train.csv

# Commit and tag
git add data/raw/train.csv.dvc
git commit -m "data: update training dataset v2024.2"
git tag data-v2024.2

# Push
dvc push
git push origin main data-v2024.2
```

### Pattern 2: Experiment Iteration

```bash
# Baseline
python train.py experiments/exp-001-baseline.yaml
git commit -am "experiment: baseline (exp-001)"

# Iteration 1
cp experiments/exp-001-baseline.yaml experiments/exp-002-lr-tune.yaml
# ... modify config ...
python train.py experiments/exp-002-lr-tune.yaml
git commit -am "experiment: LR tuning (exp-002)"

# Iteration 2
cp experiments/exp-002-lr-tune.yaml experiments/exp-003-optimizer.yaml
# ... modify config ...
python train.py experiments/exp-003-optimizer.yaml
git commit -am "experiment: optimizer change (exp-003)"

# Best model
git tag model-v1.0.0 -m "Best model: exp-003"
```

### Pattern 3: Model Update

```bash
# Train improved model
python train.py experiments/exp-010-improvements.yaml

# Save as new version
# models/resnet50/model_v1.1.0.pth
# models/resnet50/model_v1.1.0.json

# Commit
git add models/resnet50/model_v1.1.0.*
git commit -m "model: add v1.1.0 (+2% accuracy)"

# Tag
git tag model-v1.1.0 -m "Improved model"

# Push
git push origin main model-v1.1.0
```

### Pattern 4: Rollback to Previous Version

```bash
# Checkout previous model
git checkout model-v1.0.0

# Pull corresponding data
dvc checkout

# Verify versions
cat models/resnet50/model_v1.0.0.json | grep version
cat data/raw/train.csv.dvc | grep md5

# Test old model
python test.py --model models/resnet50/model_v1.0.0.pth

# Return to latest
git checkout main
dvc checkout
```

## Configuration Files

### .gitattributes (Git LFS)

```bash
# PyTorch Models
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text

# TensorFlow Models
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text

# ONNX Models
*.onnx filter=lfs diff=lfs merge=lfs -text

# Pickle Files
*.pkl filter=lfs diff=lfs merge=lfs -text

# Model Weights
weights/*.bin filter=lfs diff=lfs merge=lfs -text
```

### .dvc/config (DVC)

```ini
[core]
    remote = s3
    autostage = true

['remote "s3"']
    url = s3://my-ml-project/dvc-storage
    region = us-west-2

['remote "local"']
    url = /tmp/dvc-storage
```

### .gitignore (Combined)

```bash
# Python
__pycache__/
*.py[cod]
.Python
venv/
*.egg-info/

# Data (tracked by DVC)
/data/raw/*.csv
/data/processed/
!/data/raw/*.csv.dvc

# Models (tracked by Git LFS)
# *.pth files in .gitattributes

# IDE
.vscode/
.idea/

# ML
mlruns/
wandb/

# DVC
/dvc.lock
.dvc/cache/
```

## Aliases

Add to `~/.gitconfig`:

```ini
[alias]
    # LFS shortcuts
    lfs-files = lfs ls-files
    lfs-size = lfs ls-files -s
    lfs-clean = lfs prune --verify-remote

    # Model management
    models = tag -l "model-*"
    model-show = "!f() { git show $1:models/; }; f"

    # Experiment tracking
    experiments = log --oneline --grep="experiment:"
    exp-list = "!ls experiments/*.yaml"
```

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# DVC shortcuts
alias dvcst='dvc status'
alias dvcpl='dvc pull'
alias dvcps='dvc push'
alias dvcls='dvc list . --dvc-only'

# Combined workflow
alias mlpull='git pull && dvc pull'
alias mlpush='git push && dvc push'
alias mlstatus='git status && dvc status'
```

## Resources

- [Git LFS Tutorial](https://github.com/git-lfs/git-lfs/wiki/Tutorial)
- [DVC Get Started](https://dvc.org/doc/start)
- [DVC With Git](https://dvc.org/doc/use-cases/versioning-data-and-model-files)
- [ML Model Versioning](https://neptune.ai/blog/version-control-for-ml-models)
- [DVC vs Git LFS](https://dvc.org/doc/user-guide/related-technologies)
