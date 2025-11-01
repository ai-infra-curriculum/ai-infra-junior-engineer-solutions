# Exercise 01: Creating Your First ML Project Repository - Solution

## Overview

This solution demonstrates how to properly initialize and structure a Git repository for a Machine Learning project. It implements a complete ML inference API with proper Git configuration, atomic commits, and best practices.

## Learning Objectives

This exercise covers:

- ✅ Initializing a Git repository
- ✅ Configuring Git user identity and settings
- ✅ Understanding Git's three-state workflow
- ✅ Creating comprehensive .gitignore for ML projects
- ✅ Using .gitattributes for file handling
- ✅ Making atomic commits with clear messages
- ✅ Proper project structure for ML applications
- ✅ Commit ordering and dependency management

## Solution Structure

```
exercise-01/
├── example-repo/           # Sample ML project repository
│   ├── src/               # Source code
│   │   ├── api/          # FastAPI application
│   │   ├── models/       # ML model wrapper
│   │   ├── preprocessing/  # Image preprocessing
│   │   └── utils/        # Logging utilities
│   ├── configs/          # YAML configuration
│   ├── tests/            # Test directory
│   ├── scripts/          # Utility scripts
│   ├── docs/             # Documentation
│   ├── data/             # Data directories (empty)
│   ├── .gitignore        # Comprehensive ignore rules
│   ├── .gitattributes    # File handling rules
│   ├── requirements.txt  # Python dependencies
│   ├── .env.example      # Environment template
│   └── README.md         # Project documentation
├── scripts/
│   └── init_repository.sh  # Automated Git initialization
├── docs/
│   └── ANSWERS.md        # Detailed exercise answers
└── README.md             # This file
```

## Quick Start

### Method 1: Run Automated Script

The easiest way to see the solution in action:

```bash
cd scripts/
./init_repository.sh
```

This script will:
1. Initialize a Git repository in `example-repo/`
2. Configure Git settings
3. Create 10 atomic commits demonstrating best practices
4. Display the commit history and statistics

### Method 2: Manual Initialization

Follow these steps to manually initialize the repository:

```bash
cd example-repo/

# 1. Initialize repository
git init

# 2. Configure Git user
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 3. Configure Git settings
git config core.autocrlf input
git config core.ignorecase false

# 4. Make first commit - Git configuration
git add .gitignore .gitattributes
git commit -m "Initial commit: Add .gitignore and .gitattributes"

# 5. Add documentation
git add README.md .env.example
git commit -m "Add project documentation and configuration template"

# 6. Add dependencies
git add requirements.txt
git commit -m "Add Python dependencies"

# 7. Add configuration
git add configs/
git commit -m "Add application configuration files"

# 8. Add logging utilities
git add src/utils/ src/__init__.py
git commit -m "Add structured logging module"

# 9. Add preprocessing
git add src/preprocessing/
git commit -m "Add image preprocessing module"

# 10. Add model
git add src/models/
git commit -m "Add image classification model wrapper"

# 11. Add API
git add src/api/
git commit -m "Add FastAPI REST API"

# 12. Add directory structure
touch data/raw/.gitkeep data/processed/.gitkeep tests/.gitkeep scripts/.gitkeep docs/.gitkeep
git add data/ tests/ scripts/ docs/
git commit -m "Add project directory structure"

# View history
git log --oneline --graph
```

## Exploring the Solution

### 1. Review Commit History

```bash
cd example-repo/
git log --oneline --graph --all
```

Expected output:
```
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

### 2. Examine a Specific Commit

```bash
# View commit details
git show <commit-hash>

# Example: View initial commit
git show $(git rev-list --max-parents=0 HEAD)
```

### 3. Check File Tracking

```bash
# List all tracked files
git ls-files

# View repository status
git status
```

### 4. Examine .gitignore

```bash
# View .gitignore
cat .gitignore

# Test if a file would be ignored
git check-ignore -v data/raw/dataset.csv
```

## Key Components

### .gitignore

Comprehensive ignore rules for ML projects:
- Python artifacts (`__pycache__`, `*.pyc`)
- Model files (`*.h5`, `*.pth`, `*.pkl`)
- Datasets (`data/raw/`, `*.csv`)
- Training artifacts (`runs/`, `logs/`)
- Virtual environments (`venv/`, `.env`)
- IDE files (`.vscode/`, `.idea/`)

**Total:** 200+ ignore patterns

### .gitattributes

File handling configuration:
- Line ending normalization (`* text=auto`)
- Binary file marking (images, models)
- Python-aware diffs (`*.py diff=python`)
- Notebook handling (`*.ipynb diff=jupyternotebook`)
- Git LFS configuration (commented out)

### Source Code

**src/api/app.py** (335 lines)
- FastAPI application
- Image classification endpoints
- Health checks and documentation
- Error handling

**src/models/classifier.py** (280 lines)
- PyTorch model wrapper
- Pre-trained model loading
- Batch inference
- Top-k predictions

**src/preprocessing/image.py** (260 lines)
- Image preprocessing pipeline
- Normalization and augmentation
- Batch processing
- PIL/numpy support

**src/utils/logging.py** (310 lines)
- Structured logging
- JSON formatting
- Context propagation
- Performance timing

### Configuration

**configs/default.yaml**
- Application settings
- Model configuration
- API settings
- Logging setup
- Security options

**configs/production.yaml**
- Production overrides
- Performance tuning
- Security hardening

## Commit Analysis

### Commit 1: Git Configuration
```bash
Files: .gitignore, .gitattributes
Purpose: Set up Git before adding code
Why first: Prevents accidentally committing large files
```

### Commit 2: Documentation
```bash
Files: README.md, .env.example
Purpose: Project overview and setup instructions
Benefit: New developers can understand the project
```

### Commit 3: Dependencies
```bash
Files: requirements.txt
Purpose: Pin all dependencies
Benefit: Reproducible environment
```

### Commits 4-8: Code (Bottom-up)
```bash
Order: utils → preprocessing → models → api
Reason: Dependencies first, application last
Benefit: Each commit builds on previous ones
```

### Commit 9: Structure
```bash
Files: Empty directories with .gitkeep
Purpose: Track directory structure
Benefit: Clear project organization
```

## Best Practices Demonstrated

### 1. Atomic Commits

Each commit:
- Has a single, clear purpose
- Can be reverted independently
- Includes descriptive message
- Contains related changes only

### 2. Commit Messages

Format:
```
<subject line: 50 chars or less>

<detailed explanation>
- What changed
- Why it changed
- Any side effects
```

### 3. Commit Order

Dependencies first:
1. Configuration (.gitignore)
2. Documentation
3. Dependencies
4. Utilities
5. Core functionality
6. Application layer

### 4. File Organization

- Source code in `src/`
- Config in `configs/`
- Tests in `tests/`
- Docs in `docs/`
- Data separate (not committed)

### 5. Ignore Patterns

Exclude:
- Generated files
- Large binary files
- Sensitive data
- Environment-specific files
- Build artifacts

## Testing the Solution

### 1. Verify Git State

```bash
cd example-repo/

# Check we're in a Git repository
git status

# Verify commit count
git rev-list --count HEAD  # Should be 10

# Verify all files tracked
git ls-files | wc -l       # Count tracked files
```

### 2. Test Ignore Rules

```bash
# Create test files
touch data/raw/dataset.csv
touch models/model.pth
touch .env

# Check if ignored
git status  # Should NOT show these files

# Verify ignore rules
git check-ignore -v data/raw/dataset.csv
git check-ignore -v models/model.pth
git check-ignore -v .env
```

### 3. Examine Commit Quality

```bash
# View commit messages
git log --pretty=format:"%h - %s%n%b%n"

# Check commit sizes (should be reasonable)
git log --stat

# View first commit
git show $(git rev-list --max-parents=0 HEAD)
```

## Common Questions

### Q: Why commit .gitignore first?

**A:** To prevent accidentally committing large files (models, datasets) in subsequent commits. Once a file is in Git history, it's difficult to remove.

### Q: Why separate commits for each module?

**A:** Atomic commits make it easier to:
- Review changes
- Revert specific features
- Understand project evolution
- Debug issues with `git bisect`

### Q: Should I commit model files?

**A:** No, use Git LFS or external storage:
- Models are large (100s of MB)
- Change frequently during training
- Should use versioning system (MLflow, DVC)

### Q: How do I handle secrets?

**A:** Never commit secrets:
- Use `.env` (in .gitignore)
- Provide `.env.example` template
- Use secret management tools (Vault, AWS Secrets Manager)

### Q: What about Jupyter notebooks?

**A:** Options:
1. Commit notebooks but clear outputs
2. Use `nbstripout` to auto-clear outputs
3. Convert to Python scripts for versioning
4. Use `nbdime` for better diffs

## Related Documentation

- **ANSWERS.md**: Detailed answers to exercise questions
- **example-repo/README.md**: Project documentation
- **Git official docs**: https://git-scm.com/doc

## Learning Resources

### Git Basics
- [Pro Git Book](https://git-scm.com/book)
- [Git Tutorial](https://www.atlassian.com/git/tutorials)
- [GitHub Guides](https://guides.github.com/)

### ML-Specific Git
- [DVC - Data Version Control](https://dvc.org/)
- [Git LFS](https://git-lfs.github.com/)
- [ML Ops Best Practices](https://ml-ops.org/)

## Next Steps

After completing this exercise:

1. ✅ Understand Git initialization
2. ✅ Know how to structure ML projects
3. ✅ Can create atomic commits
4. ✅ Understand Git's three states

**Ready for Exercise 02:** Commits and History
- Viewing and understanding commit history
- Undoing changes
- Amending commits
- Working with HEAD

## Verification Checklist

- [ ] Repository initialized with `git init`
- [ ] Git user configured
- [ ] .gitignore created and comprehensive
- [ ] .gitattributes configured for ML files
- [ ] 10 atomic commits created
- [ ] Each commit has clear message
- [ ] Commits in logical order
- [ ] Directory structure complete
- [ ] All source files tracked
- [ ] Large files ignored
- [ ] README.md comprehensive
- [ ] Configuration files present

## Support

If you have questions about this exercise:
1. Review `docs/ANSWERS.md` for detailed explanations
2. Check the example repository structure
3. Examine commit messages for context
4. Refer to Git documentation

## License

This solution is part of the AI Infrastructure Junior Engineer curriculum.
