# Exercise 01: Creating Your First ML Project Repository - Answers

## Overview

This document provides detailed answers and explanations for Exercise 01 of the Git Version Control module. The exercise demonstrates how to initialize a Git repository for an ML project with proper structure, configuration, and commit history.

## Learning Objectives Covered

1. ✅ Initialize a Git repository
2. ✅ Configure Git user identity and repository settings
3. ✅ Understand the three-state Git workflow (working directory, staging area, repository)
4. ✅ Create proper .gitignore for ML projects
5. ✅ Make atomic commits with descriptive messages
6. ✅ Track project files and structure
7. ✅ Understand what to commit and what to ignore

## Questions and Answers

### Part 1: Repository Initialization

**Q1: How do you initialize a new Git repository?**

**Answer:**
```bash
git init
```

This command:
- Creates a `.git` directory in the current folder
- Initializes an empty Git repository
- Sets up Git's internal data structures
- Repository is now ready to track changes

**After initialization:**
```bash
$ ls -la .git/
drwxr-xr-x  9 user user  288 Oct 31 10:00 .
drwxr-xr-x  8 user user  256 Oct 31 10:00 ..
-rw-r--r--  1 user user   23 Oct 31 10:00 HEAD
drwxr-xr-x  2 user user   64 Oct 31 10:00 branches
-rw-r--r--  1 user user  137 Oct 31 10:00 config
-rw-r--r--  1 user user   73 Oct 31 10:00 description
drwxr-xr-x 12 user user  384 Oct 31 10:00 hooks
drwxr-xr-x  2 user user   64 Oct 31 10:00 info
drwxr-xr-x  4 user user  128 Oct 31 10:00 objects
drwxr-xr-x  4 user user  128 Oct 31 10:00 refs
```

**Q2: How do you configure Git user identity?**

**Answer:**
```bash
# Set name and email for this repository
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Or set globally for all repositories
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Importance:**
- Every commit records who made it
- Essential for collaboration and accountability
- Shows up in `git log` and GitHub/GitLab profiles
- Required by most Git hosting services

**Verification:**
```bash
git config user.name
git config user.email

# Or view all configuration
git config --list
```

**Q3: What other Git configurations are recommended for ML projects?**

**Answer:**

**Essential configurations:**
```bash
# Disable automatic line ending conversion on Unix/Linux/Mac
git config core.autocrlf input

# Case-sensitive file names (important for cross-platform)
git config core.ignorecase false

# Set default branch name
git config init.defaultBranch main

# Enable color output
git config color.ui auto

# Set default editor
git config core.editor "vim"  # or "code --wait" for VS Code
```

**ML-specific configurations:**
```bash
# Configure Git LFS (for large model files)
git lfs install

# Set larger HTTP buffer for large files
git config http.postBuffer 524288000  # 500 MB

# Configure diff tool for notebooks
git config diff.jupyternotebook.command 'git-nbdiffdriver diff'
```

### Part 2: .gitignore for ML Projects

**Q4: Why is .gitignore important for ML projects?**

**Answer:**

ML projects generate many files that **should NOT** be committed:

1. **Model Files** (large binary files):
   - `*.h5`, `*.pth`, `*.pkl` (100s of MB to GBs)
   - Should use Git LFS or external storage

2. **Datasets**:
   - Raw data can be GBs or TBs
   - Processed features can be very large
   - Use DVC (Data Version Control) instead

3. **Training Artifacts**:
   - Tensorboard logs, checkpoints
   - Can accumulate to 100s of GBs
   - Only commit final models

4. **Environment Files**:
   - Virtual environments (venv, conda)
   - Can be 100s of MB
   - Reproducible via requirements.txt

5. **Temporary Files**:
   - `__pycache__`, `.pytest_cache`
   - Cache files
   - Build artifacts

**Q5: What are the key sections in an ML project .gitignore?**

**Answer:**

**Our .gitignore includes:**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/

# ML Models (large files)
*.h5
*.pth
*.pkl
*.onnx

# Datasets
data/raw/
data/processed/
*.csv
*.parquet

# Training artifacts
runs/
logs/
tensorboard_logs/
checkpoints/

# Environments
venv/
.env
*.env

# IDEs
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db
```

**Best Practice:**
- Keep data directory structure (`data/raw/`, `data/processed/`)
- Use `.gitkeep` files to track empty directories
- Exclude actual data files

**Q6: What is .gitattributes and why do ML projects need it?**

**Answer:**

`.gitattributes` defines attributes for pathnames:

**Key purposes:**

1. **Line Ending Normalization**:
```gitattributes
* text=auto
*.py text eol=lf
*.sh text eol=lf
```

2. **Binary File Handling**:
```gitattributes
*.png binary
*.jpg binary
*.pkl binary
*.h5 binary
```

3. **Git LFS Configuration**:
```gitattributes
*.pth filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
```

4. **Diff Settings**:
```gitattributes
*.py diff=python
*.ipynb diff=jupyternotebook
```

5. **Merge Strategies**:
```gitattributes
*.pkl merge=binary
*.ipynb merge=jupyternotebook
```

**Benefits for ML:**
- Prevents corrupting binary model files
- Handles notebooks correctly
- Cross-platform compatibility
- Better diff output for code

### Part 3: Three-State Git Workflow

**Q7: What are the three states in Git?**

**Answer:**

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Working      │     │    Staging      │     │   Repository    │
│   Directory     │────>│     Area        │────>│    (Commits)    │
│   (Modified)    │     │   (Staged)      │     │  (Committed)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
    git add -->            git commit -->
```

**1. Working Directory:**
- Files in your project folder
- Can be modified, created, deleted
- Changes not yet tracked by Git
- `git status` shows as "modified" or "untracked"

**2. Staging Area (Index):**
- Prepared snapshot for next commit
- Files added with `git add`
- Allows selective commits
- `git status` shows as "changes to be committed"

**3. Repository:**
- Committed snapshots
- Permanent history
- Can be pushed to remote
- `git log` shows commit history

**Example workflow:**

```bash
# 1. Working Directory - modify files
echo "import torch" >> model.py

# 2. Staging Area - prepare for commit
git add model.py

# 3. Repository - save to history
git commit -m "Add PyTorch import"
```

**Q8: What is the difference between `git add .` and `git add -A`?**

**Answer:**

| Command | New Files | Modified Files | Deleted Files | Scope |
|---------|-----------|----------------|---------------|-------|
| `git add .` | ✅ | ✅ | ✅ | Current directory and subdirectories |
| `git add -A` | ✅ | ✅ | ✅ | Entire repository |
| `git add -u` | ❌ | ✅ | ✅ | Tracked files only |

**Recommendations:**
```bash
# Add specific files (best practice)
git add src/models/classifier.py
git add src/api/app.py

# Add directory
git add src/models/

# Add all Python files in current directory
git add *.py

# Add all (use cautiously)
git add -A
```

**Best Practice for ML:**
Always review before adding:
```bash
git status  # See what changed
git diff    # See actual changes
git add <specific-files>
git status  # Verify staged changes
git commit
```

### Part 4: Atomic Commits

**Q9: What is an atomic commit and why is it important?**

**Answer:**

**Atomic Commit** = One logical change per commit

**Characteristics:**
- Single, focused purpose
- Self-contained change
- Can be reverted independently
- Clear, descriptive message

**Good atomic commits:**
```bash
# ✅ Good: One purpose
git commit -m "Add image preprocessing module"

# ✅ Good: Single bug fix
git commit -m "Fix memory leak in batch processing"

# ✅ Good: Single feature
git commit -m "Add health check endpoint to API"
```

**Bad commits (not atomic):**
```bash
# ❌ Bad: Multiple unrelated changes
git commit -m "Add preprocessing, fix bug, update docs, refactor API"

# ❌ Bad: Too vague
git commit -m "Update code"

# ❌ Bad: Too large
# (Commit with 50 files changed across multiple modules)
```

**Benefits:**
1. **Easy to Review**: Clear what changed and why
2. **Easy to Revert**: Can undo specific changes
3. **Easy to Debug**: `git bisect` works better
4. **Clear History**: Understand project evolution
5. **Better Collaboration**: Team understands changes

**Q10: How should you structure commit messages?**

**Answer:**

**Standard format:**
```
<type>: <short summary> (50 chars or less)

<detailed explanation if needed>
- What changed
- Why it changed
- Any breaking changes or side effects
```

**Our commit message structure:**
```bash
git commit -m "Add image preprocessing module

Implements image preprocessing pipeline:
- Resizing and normalization
- ImageNet statistics (mean, std)
- Data augmentation (optional)
- Tensor conversion
- Batch processing support

Features:
- Configurable target size
- PIL and numpy array support
- Denormalization for visualization"
```

**Commit types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting, no code change
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance tasks

**Best practices:**
- Use imperative mood ("Add feature" not "Added feature")
- Capitalize first letter
- No period at end of subject
- Blank line between subject and body
- Wrap body at 72 characters
- Explain **what** and **why**, not **how**

### Part 5: Project Structure

**Q11: What is the recommended directory structure for ML projects?**

**Answer:**

**Our structure:**
```
ml-inference-api/
├── src/                    # Source code
│   ├── api/               # API endpoints
│   ├── models/            # ML models
│   ├── preprocessing/     # Data preprocessing
│   └── utils/             # Utilities
├── configs/               # Configuration files
│   ├── default.yaml
│   └── production.yaml
├── tests/                 # Test files
│   ├── unit/
│   └── integration/
├── scripts/               # Utility scripts
│   ├── train.py
│   └── deploy.sh
├── docs/                  # Documentation
│   ├── API.md
│   └── DEPLOYMENT.md
├── data/                  # Data (not committed)
│   ├── raw/
│   └── processed/
├── .gitignore            # Git ignore rules
├── .gitattributes        # Git attributes
├── requirements.txt      # Dependencies
└── README.md             # Project overview
```

**Why this structure:**

1. **src/**: All source code in one place
   - Importable as package
   - Clear module organization
   - Separates code from config/data

2. **configs/**: Environment-specific settings
   - YAML for readability
   - Easy to override
   - Version controlled

3. **tests/**: All tests together
   - Mirror src/ structure
   - Unit and integration separated
   - Easy to run: `pytest tests/`

4. **scripts/**: Standalone utilities
   - Training scripts
   - Deployment automation
   - Data processing

5. **data/**: Data files (excluded from Git)
   - Structure tracked with .gitkeep
   - Actual data in .gitignore
   - Clear separation (raw vs processed)

**Q12: How do you track empty directories in Git?**

**Answer:**

Git doesn't track empty directories. To track structure:

**Solution 1: .gitkeep files**
```bash
touch data/raw/.gitkeep
touch data/processed/.gitkeep
git add data/raw/.gitkeep
git add data/processed/.gitkeep
```

**Solution 2: README files**
```bash
echo "# Raw datasets" > data/raw/README.md
echo "# Processed features" > data/processed/README.md
git add data/raw/README.md
git add data/processed/README.md
```

**In .gitignore:**
```gitignore
# Ignore data but keep structure
data/raw/*
data/processed/*

# Don't ignore .gitkeep
!data/raw/.gitkeep
!data/processed/.gitkeep
```

### Part 6: Commit Order and History

**Q13: In what order should you make initial commits?**

**Answer:**

**Our commit sequence (10 atomic commits):**

1. **Initial commit**: `.gitignore` and `.gitattributes`
   - Sets up Git configuration first
   - Prevents accidentally committing large files

2. **Documentation**: `README.md`, `.env.example`
   - Project overview
   - Setup instructions

3. **Dependencies**: `requirements.txt`
   - Pin versions for reproducibility
   - Separate from code

4. **Configuration**: `configs/*.yaml`
   - Settings before code
   - Code will reference these

5. **Utilities**: `src/utils/logging.py`
   - Foundation utilities first
   - Used by other modules

6. **Preprocessing**: `src/preprocessing/image.py`
   - Data pipeline before models
   - Models depend on this

7. **Models**: `src/models/classifier.py`
   - Core ML functionality
   - Before API layer

8. **API**: `src/api/app.py`
   - Application layer last
   - Integrates all components

9. **Tests**: `tests/`
   - After implementation
   - Separate commit

10. **Structure**: Empty directories with .gitkeep
    - Final structure commit

**Why this order:**
- **Dependencies first**: Bottom-up approach
- **Configuration before code**: Code references config
- **Utilities before features**: Foundation first
- **Core before API**: Business logic before interface
- **Each commit is functional**: Repository works at each step

**Q14: How do you view commit history?**

**Answer:**

**Basic log:**
```bash
git log
```

**One-line format:**
```bash
git log --oneline
```

**With graph:**
```bash
git log --oneline --graph --all
```

**Specific file:**
```bash
git log -- src/models/classifier.py
```

**Show changes:**
```bash
git log -p
```

**Statistics:**
```bash
git log --stat
```

**Date range:**
```bash
git log --since="2024-10-01" --until="2024-10-31"
```

**Author:**
```bash
git log --author="John Doe"
```

**Search commit messages:**
```bash
git log --grep="bug fix"
```

**Our repository history:**
```bash
$ git log --oneline --graph
* a1b2c3d Add project directory structure
* d4e5f6g Add FastAPI REST API
* h7i8j9k Add image classification model wrapper
* l0m1n2o Add image preprocessing module
* p3q4r5s Add structured logging module
* t6u7v8w Add application configuration files
* x9y0z1a Add Python dependencies
* b2c3d4e Add project documentation and configuration template
* f5g6h7i Initial commit: Add .gitignore and .gitattributes
```

### Part 7: Checking Status and Diffs

**Q15: How do you check the status of your repository?**

**Answer:**

**Check status:**
```bash
git status
```

**Example output:**
```
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   src/models/classifier.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   src/api/app.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        src/models/new_model.py
```

**Short status:**
```bash
git status -s
```

Output:
```
M  src/models/classifier.py  # Modified and staged
 M src/api/app.py            # Modified, not staged
?? src/models/new_model.py   # Untracked
```

**Q16: How do you see what changed in files?**

**Answer:**

**Unstaged changes:**
```bash
git diff
```

**Staged changes:**
```bash
git diff --staged
# or
git diff --cached
```

**Specific file:**
```bash
git diff src/models/classifier.py
```

**Compare commits:**
```bash
git diff HEAD~1 HEAD
```

**Word-level diff:**
```bash
git diff --word-diff
```

**Statistics only:**
```bash
git diff --stat
```

**Example diff output:**
```diff
diff --git a/src/models/classifier.py b/src/models/classifier.py
index 1234567..abcdefg 100644
--- a/src/models/classifier.py
+++ b/src/models/classifier.py
@@ -45,6 +45,10 @@ class ImageClassifier:
     def predict(self, image_tensor):
+        # Add batch dimension
+        if image_tensor.dim() == 3:
+            image_tensor = image_tensor.unsqueeze(0)
+
         with torch.no_grad():
             outputs = self.model(image_tensor)
```

## Key Takeaways

1. **Always initialize repository before adding files**
2. **Configure .gitignore and .gitattributes first**
3. **Make atomic commits with clear messages**
4. **Follow dependency order (utilities → core → application)**
5. **Don't commit large files (models, datasets)**
6. **Track directory structure, not data**
7. **Use meaningful commit messages**
8. **Review changes before committing**
9. **Commit working code, not broken code**
10. **Keep commits focused and reviewable**

## Common Mistakes to Avoid

❌ **Don't:**
- Commit large model files directly
- Make vague commit messages ("fix stuff")
- Mix unrelated changes in one commit
- Commit broken/non-working code
- Commit sensitive data (API keys, passwords)
- Commit generated files (__pycache__, .pyc)
- Forget to add .gitignore

✅ **Do:**
- Use Git LFS for large files
- Write clear, descriptive messages
- Make focused, atomic commits
- Test before committing
- Use .env.example for secrets template
- Exclude build artifacts
- Set up .gitignore immediately

## Next Steps

After completing this exercise, you should be able to:
- Initialize Git repositories confidently
- Structure ML projects properly
- Create meaningful commit histories
- Understand Git's three-state workflow
- Use .gitignore and .gitattributes effectively

**Ready for Exercise 02:** Commits and History - dive deeper into commit management, viewing history, and understanding Git's internal structure.
