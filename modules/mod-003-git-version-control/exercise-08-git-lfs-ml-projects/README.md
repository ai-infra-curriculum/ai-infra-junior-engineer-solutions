# Exercise 08: Git LFS for ML Projects — Solution

## What the exercise asked for

Use Git Large File Storage (LFS) to version ML artifacts that
are too large for normal Git, and understand when LFS is the
right answer vs. when DVC / a model registry is better.

## When LFS is the right tool

LFS works for files that:
- Are too big for normal Git (>10MB).
- Change rarely.
- Have meaningful Git-style versioning needs (you want
  branches / merges to handle them).
- Stay reasonably sized in total (<100GB across the repo).

Typical ML use cases:
- Pretrained model checkpoints (<1GB).
- Small reference datasets.
- Tokenizer / preprocessor binaries.

## When LFS is NOT the right tool

Use DVC / a model registry instead when:
- Files are >1GB (LFS cost + bandwidth becomes painful).
- You version data through ML-specific lineage (DVC excels).
- You want artifact + experiment metadata coupled (MLflow,
  W&B).
- You're past 50GB total LFS storage (GitHub's free tier caps).

## Setup

```bash
# Install
brew install git-lfs        # macOS
sudo apt install git-lfs    # Debian/Ubuntu
git lfs install              # per-repo setup

# Track patterns
git lfs track "*.pt"
git lfs track "*.onnx"
git lfs track "*.bin"
git lfs track "models/*.ckpt"

# Verify
git lfs track   # lists current tracked patterns
cat .gitattributes
# models/*.ckpt filter=lfs diff=lfs merge=lfs -text

# Commit the .gitattributes
git add .gitattributes
git commit -m "chore: configure Git LFS for model artifacts"
```

## Working with LFS files

LFS is mostly transparent. `git add`, `git commit`, `git
push`, `git pull` all work normally. The difference:

- The Git object is a small pointer file (a few hundred bytes).
- The actual file content lives on the LFS server.
- `git clone` downloads pointer files first, then the LFS
  blobs.

```bash
# Inspect tracked LFS files
git lfs ls-files

# Inspect a specific file's LFS pointer
git show HEAD:models/model.pt
# version https://git-lfs.github.com/spec/v1
# oid sha256:...
# size 256781234

# Pull only pointers (no LFS content), then selectively fetch
git clone --filter=blob:none URL
git lfs fetch --include="models/specific-model.pt"
git lfs checkout
```

## Cost-management for LFS

GitHub charges for LFS bandwidth + storage:
- Free: 1GB storage, 1GB bandwidth/month.
- Paid: $5/month for 50GB storage + 50GB bandwidth.

Practical hygiene:
- Don't store experimental models in LFS — use ephemeral
  cloud storage.
- Periodically prune old LFS objects (they aren't auto-
  collected when you delete the file from Git):

```bash
# Remove LFS objects that are no longer referenced
git lfs prune
```

## Limits of LFS

- **Merge handling**: LFS pointer files merge fine, but the
  underlying binary content can't be 3-way merged. Conflict
  = pick one.
- **Diff**: `git diff` on an LFS file shows pointer-file
  diffs, not content diffs.
- **Cross-fork support**: external contributors to a public
  repo can't push LFS objects without push access. Limits
  open-source workflows.

## The DVC alternative

For ML-specific versioning beyond LFS:

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet

# DVC creates a .dvc file with metadata; commit that to Git
git add data/training_set.parquet.dvc .gitignore
git commit -m "data: add training set v1"

# Push the data to remote storage (S3, GCS, Azure, etc.)
dvc remote add -d myremote s3://my-bucket/dvc
dvc push
```

DVC adds:
- Pipeline definition (`dvc.yaml`).
- Experiment tracking.
- Reproducible runs (`dvc repro`).
- Lineage from data → model.

Use DVC when ML-workflow versioning matters more than
"just versioning some big files."

## Cross-references

- Exercise prompt: `ai-infra-junior-engineer-learning/lessons/mod-003-git-version-control/exercises/exercise-08-git-lfs-ml-projects.md`
- The MLOps track covers DVC at depth:
  `ai-infra-mlops-learning/lessons/01-mlops-foundations/`
- Module complete after this exercise.
