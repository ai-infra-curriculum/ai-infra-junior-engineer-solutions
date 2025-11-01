# Git Branching - Quick Reference

## Creating Branches

### Basic Branch Creation

```bash
# Create branch (don't switch)
git branch feature/new-feature

# Create and switch (traditional)
git checkout -b feature/new-feature

# Create and switch (modern - Git 2.23+)
git switch -c feature/new-feature

# Create from specific commit
git branch feature/new-feature abc123
git switch -c feature/new-feature abc123

# Create from another branch
git branch feature/new-feature main
git switch -c feature/new-feature origin/main
```

### Branch Naming Conventions

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/batch-inference` |
| `fix/` | Bug fixes | `fix/memory-leak` |
| `hotfix/` | Critical production fixes | `hotfix/security-patch` |
| `experiment/` | Experimental features | `experiment/onnx-runtime` |
| `release/` | Release branches | `release/v1.0.0` |
| `docs/` | Documentation only | `docs/api-guide` |
| `refactor/` | Code refactoring | `refactor/model-loading` |
| `test/` | Test additions | `test/integration-tests` |

## Switching Branches

### Modern Approach (Git 2.23+)

```bash
# Switch to existing branch
git switch feature/my-feature

# Switch to previous branch
git switch -

# Create and switch
git switch -c feature/new-feature

# Force switch (discard local changes)
git switch --force feature/other-branch
git switch -f feature/other-branch
```

### Traditional Approach

```bash
# Switch to existing branch
git checkout feature/my-feature

# Switch to previous branch
git checkout -

# Create and switch
git checkout -b feature/new-feature

# Force switch
git checkout --force feature/other-branch
git checkout -f feature/other-branch
```

### Handling Uncommitted Changes

```bash
# Error: uncommitted changes
git switch feature/other-branch
# error: Your local changes would be overwritten

# Option 1: Stash changes
git stash
git switch feature/other-branch

# Option 2: Commit changes
git commit -am "WIP: work in progress"
git switch feature/other-branch

# Option 3: Discard changes (careful!)
git restore .
git switch feature/other-branch
```

## Listing Branches

### Local Branches

```bash
# List local branches
git branch

# List with last commit
git branch -v

# List with tracking info
git branch -vv

# List merged branches
git branch --merged

# List unmerged branches
git branch --no-merged

# List merged to specific branch
git branch --merged main
git branch --no-merged main
```

### Remote Branches

```bash
# List remote branches
git branch -r

# List all branches (local + remote)
git branch -a

# List remote branches with details
git branch -r -v
```

### Filtering Branches

```bash
# Branches matching pattern
git branch --list "feature/*"
git branch --list "fix/*"

# Branches containing commit
git branch --contains abc123

# Branches not containing commit
git branch --no-contains abc123

# Branches by author (with log)
git for-each-ref --format='%(refname:short)' refs/heads/ | \
  xargs -I {} sh -c 'echo {}; git log -1 --author="John" {}'
```

## Comparing Branches

### Commit Comparison

```bash
# Commits in feature not in main
git log main..feature/my-feature

# Commits in main not in feature
git log feature/my-feature..main

# Commits unique to each branch
git log --left-right --oneline main...feature/my-feature

# Oneline format
git log main..feature/my-feature --oneline

# With statistics
git log main..feature/my-feature --stat

# With patches
git log main..feature/my-feature -p

# Count commits
git rev-list --count main..feature/my-feature
```

### Code Comparison

```bash
# Diff between branches
git diff main..feature/my-feature
git diff main...feature/my-feature  # Common ancestor

# Show only file names
git diff main..feature/my-feature --name-only

# Show file names with status
git diff main..feature/my-feature --name-status

# Show statistics
git diff main..feature/my-feature --stat

# Diff specific file
git diff main..feature/my-feature -- path/to/file.py

# Diff excluding files
git diff main..feature/my-feature -- . ':(exclude)package-lock.json'

# Word-level diff
git diff main..feature/my-feature --word-diff

# Compact summary
git diff main..feature/my-feature --compact-summary
```

### Visualizing Branches

```bash
# Simple graph
git log --oneline --graph --all

# Detailed graph
git log --graph --oneline --decorate --all

# Custom format
git log --graph --pretty=format:'%C(yellow)%h%Creset -%C(cyan)%d%Creset %s %C(green)(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --all

# Show branch structure
git show-branch main feature/*

# Visual diff
git diff main..feature/my-feature --color-words
```

## Stashing Changes

### Basic Stashing

```bash
# Stash current changes
git stash

# Stash with message
git stash save "WIP: refactoring authentication"

# Stash including untracked files
git stash -u
git stash --include-untracked

# Stash including ignored files
git stash -a
git stash --all

# Stash specific files
git stash push -m "WIP: config changes" config.yaml settings.py
```

### Managing Stashes

```bash
# List stashes
git stash list

# Show stash contents
git stash show
git stash show stash@{0}

# Show stash diff
git stash show -p
git stash show -p stash@{1}

# Apply most recent stash (keep in stash list)
git stash apply

# Apply specific stash
git stash apply stash@{2}

# Apply and remove (pop)
git stash pop
git stash pop stash@{1}
```

### Stash Operations

```bash
# Create branch from stash
git stash branch feature/stashed-work stash@{0}

# Drop specific stash
git stash drop stash@{1}

# Clear all stashes
git stash clear

# Apply stash to different branch
git stash
git switch other-branch
git stash pop
```

### Advanced Stashing

```bash
# Stash only staged changes
git stash push -m "staged changes only" --staged

# Stash only unstaged changes
git stash push -m "unstaged changes" --keep-index

# Stash with pathspec
git stash push -m "only Python files" -- "*.py"

# Interactive stashing
git stash push -p
```

## Deleting Branches

### Local Branches

```bash
# Delete merged branch (safe)
git branch -d feature/completed-feature

# Force delete unmerged branch
git branch -D experiment/failed-feature

# Delete multiple branches
git branch -d feature/old-1 feature/old-2 feature/old-3

# Delete merged branches except main
git branch --merged | grep -v "main" | xargs git branch -d

# Delete branches matching pattern
git branch --list "feature/temp-*" | xargs git branch -D
```

### Remote Branches

```bash
# Delete remote branch
git push origin --delete feature/old-feature

# Alternative syntax
git push origin :feature/old-feature

# Delete multiple remote branches
git push origin --delete feature/old-1 feature/old-2

# Prune remote tracking branches
git fetch --prune
git fetch -p

# Remove all stale remote tracking branches
git remote prune origin
```

### Cleanup Scripts

```bash
# Delete all merged local branches
git branch --merged main | \
  grep -v "^\*" | \
  grep -v "main" | \
  xargs -r git branch -d

# Delete local branches with deleted remotes
git branch -vv | \
  grep ': gone]' | \
  awk '{print $1}' | \
  xargs -r git branch -D

# Prune and clean
git fetch --prune && \
git branch --merged main | \
  grep -v "main" | \
  xargs -r git branch -d
```

## Renaming Branches

### Local Branches

```bash
# Rename current branch
git branch -m new-name

# Rename specific branch
git branch -m old-name new-name

# Force rename (overwrite existing)
git branch -M old-name new-name
```

### Remote Branches

```bash
# Rename local, update remote
git branch -m old-name new-name
git push origin :old-name         # Delete old
git push origin new-name          # Push new
git push origin -u new-name       # Set upstream
```

## Branch Tracking

### Setting Upstream

```bash
# Set upstream when pushing
git push -u origin feature/my-feature

# Set upstream for existing branch
git branch --set-upstream-to=origin/feature/my-feature
git branch -u origin/feature/my-feature

# Push and set upstream (explicit)
git push --set-upstream origin feature/my-feature
```

### Viewing Tracking

```bash
# Show tracking branches
git branch -vv

# Show remote tracking branches
git remote show origin

# Show upstream for current branch
git rev-parse --abbrev-ref --symbolic-full-name @{u}
```

## Working with Remote Branches

### Fetching Branches

```bash
# Fetch all remote branches
git fetch origin

# Fetch specific branch
git fetch origin feature/remote-feature

# Fetch all remotes
git fetch --all

# Fetch and prune deleted branches
git fetch --prune
```

### Checking Out Remote Branches

```bash
# Checkout remote branch (creates local tracking)
git switch feature/remote-feature
git checkout feature/remote-feature

# Explicit tracking setup
git checkout -b feature/local-name origin/feature/remote-name

# Without tracking
git checkout --no-track origin/feature/remote-feature
```

### Pushing Branches

```bash
# Push current branch
git push

# Push to specific remote
git push origin feature/my-feature

# Push and set upstream
git push -u origin feature/my-feature

# Push all branches
git push --all origin

# Force push (dangerous!)
git push --force origin feature/my-feature
git push -f origin feature/my-feature

# Force push with safety
git push --force-with-lease origin feature/my-feature
```

## Branch Information

### Current Branch

```bash
# Show current branch
git branch --show-current

# Alternative method
git rev-parse --abbrev-ref HEAD

# Check if on specific branch
if [ "$(git branch --show-current)" = "main" ]; then
  echo "On main branch"
fi
```

### Branch Details

```bash
# Show branch commit
git rev-parse feature/my-feature

# Show branch short hash
git rev-parse --short feature/my-feature

# Show branch creation point
git reflog show feature/my-feature | tail -1

# Show branch age
git log -1 --format="%cr" feature/my-feature

# Show branch author
git log -1 --format="%an" feature/my-feature
```

### Branch Statistics

```bash
# Commits on branch since main
git rev-list --count main..feature/my-feature

# Files changed on branch
git diff --name-only main..feature/my-feature | wc -l

# Lines changed on branch
git diff --stat main..feature/my-feature

# Contributors to branch
git shortlog -sn main..feature/my-feature
```

## Advanced Operations

### Cherry-picking

```bash
# Apply commit to current branch
git cherry-pick abc123

# Apply multiple commits
git cherry-pick abc123 def456

# Apply commit range
git cherry-pick main~3..main

# Cherry-pick without committing
git cherry-pick -n abc123
git cherry-pick --no-commit abc123

# Continue after conflicts
git cherry-pick --continue

# Abort cherry-pick
git cherry-pick --abort
```

### Rebasing Branches

```bash
# Rebase onto main
git switch feature/my-feature
git rebase main

# Interactive rebase
git rebase -i main

# Continue after resolving conflicts
git rebase --continue

# Skip problematic commit
git rebase --skip

# Abort rebase
git rebase --abort

# Rebase preserving merges
git rebase --rebase-merges main
```

### Merging Branches

```bash
# Merge feature into current branch
git merge feature/my-feature

# Merge with commit message
git merge feature/my-feature -m "Merge feature X"

# Merge without fast-forward
git merge --no-ff feature/my-feature

# Merge with squash
git merge --squash feature/my-feature

# Abort merge
git merge --abort
```

## Branch Workflows

### Feature Branch Workflow

```bash
# 1. Create feature branch
git switch main
git pull origin main
git switch -c feature/new-feature

# 2. Develop feature
git add .
git commit -m "feat: implement new feature"

# 3. Keep updated
git fetch origin
git rebase origin/main

# 4. Push feature
git push -u origin feature/new-feature

# 5. After merge, delete
git switch main
git pull origin main
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

### Hotfix Workflow

```bash
# 1. Create hotfix from production
git switch main
git switch -c hotfix/critical-bug

# 2. Fix bug
git commit -am "fix: resolve critical bug"

# 3. Merge to main
git switch main
git merge hotfix/critical-bug

# 4. Tag release
git tag -a v1.0.1 -m "Hotfix: critical bug"

# 5. Merge to develop
git switch develop
git merge hotfix/critical-bug

# 6. Clean up
git branch -d hotfix/critical-bug
```

### Release Branch Workflow

```bash
# 1. Create release branch
git switch develop
git switch -c release/v1.0.0

# 2. Prepare release
git commit -am "chore: bump version to 1.0.0"

# 3. Merge to main
git switch main
git merge release/v1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"

# 4. Merge back to develop
git switch develop
git merge release/v1.0.0

# 5. Clean up
git branch -d release/v1.0.0
```

## Troubleshooting

### Common Issues

```bash
# Issue: Can't switch - uncommitted changes
git stash
git switch other-branch

# Issue: Branch already exists
git branch -D existing-branch  # Delete first
git switch -c existing-branch

# Issue: Lost branch after deletion
git reflog
git branch recovered-branch abc123

# Issue: Branch diverged from remote
git fetch origin
git rebase origin/feature/my-feature
# Or
git merge origin/feature/my-feature

# Issue: Accidentally committed to wrong branch
git log -1  # Get commit hash
git reset --hard HEAD~1  # Undo commit
git switch correct-branch
git cherry-pick abc123  # Apply to correct branch
```

### Recovery Commands

```bash
# Find deleted branch commit
git reflog
git fsck --lost-found

# Recover deleted branch
git branch recovered-feature abc123

# Restore accidentally deleted files
git restore --source=HEAD~1 file.py

# Undo branch reset
git reflog
git reset --hard HEAD@{1}
```

## Aliases

Add to `.gitconfig`:

```ini
[alias]
    # Branch shortcuts
    br = branch
    co = checkout
    sw = switch
    swc = switch -c

    # Branch listing
    branches = branch -a
    br-merged = branch --merged
    br-unmerged = branch --no-merged

    # Branch comparison
    br-diff = diff main...HEAD
    br-log = log main..HEAD --oneline

    # Branch visualization
    graph = log --graph --oneline --decorate --all
    tree = log --graph --pretty=format:'%C(yellow)%h%Creset -%C(cyan)%d%Creset %s %C(green)(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --all

    # Branch cleanup
    br-clean = !git branch --merged main | grep -v 'main' | xargs git branch -d
    br-prune = fetch --prune

    # Stash shortcuts
    st = stash
    stp = stash pop
    stl = stash list
    sts = stash show -p
```

Usage:
```bash
git sw -c feature/new-feature
git br-merged
git graph
git br-clean
```

## Best Practices

### Branch Naming

✅ **Good:**
- `feature/user-authentication`
- `fix/memory-leak-in-cache`
- `hotfix/security-vulnerability`
- `experiment/quantized-models`

❌ **Bad:**
- `johns-stuff`
- `temp`
- `new-branch`
- `fix-2`

### Branch Management

```bash
# Regularly update from main
git fetch origin
git rebase origin/main

# Clean up merged branches
git branch --merged | grep -v "main" | xargs git branch -d

# Keep branches small and focused
# - One feature per branch
# - Short-lived (days, not months)
# - Regular commits

# Use descriptive commit messages
git commit -m "feat: add batch processing for images"
# Not: "updates" or "WIP"
```

### Branch Protection

```bash
# Never force push to main/shared branches
git push --force origin feature/my-private-branch  # OK
git push --force origin main  # ❌ Never!

# Use --force-with-lease for safety
git push --force-with-lease origin feature/my-feature

# Always create branches for features
git switch -c feature/new-work  # ✅
# Not: git commit -am "new feature" on main  # ❌
```

## Configuration

```bash
# Default branch name
git config --global init.defaultBranch main

# Push current branch to same name
git config --global push.default current

# Automatically set up tracking
git config --global push.autoSetupRemote true

# Always show branch in prompt
git config --global oh-my-zsh.hide-status 0

# Color branch output
git config --global color.branch always

# Prune on fetch
git config --global fetch.prune true
```

## Resources

- [Git Branching Model](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [GitLab Flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html)
- [Pro Git: Branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
