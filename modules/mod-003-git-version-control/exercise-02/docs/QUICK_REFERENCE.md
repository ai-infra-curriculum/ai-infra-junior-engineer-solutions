# Git Commits and History - Quick Reference

## Conventional Commit Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

| Type | Purpose | Example |
|------|---------|---------|
| `feat` | New feature | `feat: add user authentication` |
| `fix` | Bug fix | `fix: handle null pointer in parser` |
| `docs` | Documentation | `docs: update API guide` |
| `style` | Formatting | `style: fix indentation` |
| `refactor` | Code restructuring | `refactor: extract validation logic` |
| `test` | Tests | `test: add unit tests for auth` |
| `chore` | Maintenance | `chore: update dependencies` |
| `perf` | Performance | `perf: optimize database queries` |

### Examples

```bash
# Simple feature
git commit -m "feat: add password reset endpoint"

# Feature with scope
git commit -m "feat(auth): add OAuth2 integration"

# Bug fix with explanation
git commit -m "fix: prevent memory leak in image processing

The preprocessing pipeline was not releasing PIL Image objects.
Added explicit cleanup after processing.

Fixes #123"

# Breaking change
git commit -m "feat!: change API response format

BREAKING CHANGE: Response now uses camelCase instead of snake_case"
```

## Viewing History

### Basic Log Commands

```bash
# Standard log
git log

# One line per commit
git log --oneline

# Last N commits
git log -5
git log --oneline -10

# With file statistics
git log --stat

# With actual code changes
git log -p
git log -p -2  # Last 2 commits with changes

# Graph view (useful with branches)
git log --oneline --graph --all

# Custom format
git log --pretty=format:"%h - %an, %ar : %s"
git log --pretty=format:"%h %s" --graph
```

### Filtering Commits

```bash
# By author
git log --author="John Doe"
git log --author="john@example.com"

# By date
git log --since="2024-01-01"
git log --since="2 weeks ago"
git log --after="2024-01-01" --before="2024-12-31"
git log --since="yesterday"

# By message content
git log --grep="fix"
git log --grep="bug" --grep="error" --all-match

# By file
git log -- path/to/file.py
git log --oneline -- src/models/

# By content (code search)
git log -S "function_name"        # When function was added/removed
git log -G "regex_pattern"        # Regex search in diffs
git log -S "class UserModel" -p   # Show actual changes
```

### Advanced Log Options

```bash
# Specific range
git log main..feature-branch
git log HEAD~5..HEAD

# Exclude merges
git log --no-merges

# Only merges
git log --merges

# First parent only (cleaner on merge-heavy histories)
git log --first-parent

# Specific line range in file
git log -L 10,20:src/app.py

# With diff stat
git log --stat --oneline

# Compact with file list
git log --oneline --name-only

# With commit graph
git log --all --decorate --oneline --graph
# Alias: git log --adog
```

## Inspecting Commits

### git show

```bash
# Show latest commit
git show
git show HEAD

# Show specific commit
git show abc123
git show HEAD~1        # Previous commit
git show HEAD~2        # Two commits ago
git show HEAD^^^       # Three carets = 3 commits back

# Show with stats only
git show --stat HEAD

# Show specific file from commit
git show HEAD:src/app.py
git show abc123:README.md

# Show file from previous commit
git show HEAD~1:configs/default.yaml
```

### Comparing Commits

```bash
# Diff between commits
git diff HEAD~1 HEAD
git diff abc123 def456

# Diff specific file
git diff HEAD~1 HEAD -- src/app.py

# Diff with stat
git diff --stat HEAD~2 HEAD

# Word-level diff
git diff --word-diff

# Only show changed file names
git diff --name-only HEAD~1 HEAD
git diff --name-status HEAD~1 HEAD
```

## Amending Commits

### Basic Amend

```bash
# Amend last commit message
git commit --amend -m "New message"

# Amend with editor
git commit --amend

# Add forgotten file to last commit
git add forgotten_file.py
git commit --amend --no-edit

# Amend both message and files
git add updated_file.py
git commit --amend -m "Better commit message"

# Amend author
git commit --amend --author="New Name <email@example.com>"

# Amend date
git commit --amend --date="2024-01-01 10:00:00"
```

### Important Notes

⚠️ **Never amend commits that have been pushed!**

```bash
# Safe: Local commits only
git commit --amend

# Dangerous: After push
git push
git commit --amend  # ❌ DON'T DO THIS

# If you must (forces rewrite on remote)
git push --force    # ⚠️ Very dangerous!
```

## Reverting Changes

### git revert

```bash
# Revert last commit
git revert HEAD

# Revert specific commit
git revert abc123

# Revert without opening editor
git revert HEAD --no-edit

# Revert multiple commits
git revert HEAD HEAD~1 HEAD~2 --no-edit

# Revert a merge commit
git revert -m 1 abc123

# Start revert but don't commit yet
git revert --no-commit HEAD
# Make additional changes
git revert --continue
# Or abort
git revert --abort
```

### git reset (Local Only!)

```bash
# Soft reset (keep changes staged)
git reset --soft HEAD~1

# Mixed reset (keep changes unstaged) - DEFAULT
git reset HEAD~1
git reset --mixed HEAD~1

# Hard reset (discard all changes)
git reset --hard HEAD~1

# Reset to specific commit
git reset --hard abc123

# Reset specific file
git reset HEAD file.py
```

### Revert vs Reset

| Aspect | `git revert` | `git reset` |
|--------|--------------|-------------|
| History | Preserved (adds new commit) | Rewritten (moves branch) |
| Safety | Safe for shared commits | Only for local commits |
| Result | Undo changes with new commit | Move branch pointer |
| Use case | After push | Before push |

## Searching History

### Find When Code Was Added/Removed

```bash
# When was this function added?
git log -S "def process_image" --oneline

# When was this regex pattern changed?
git log -G "import.*torch" --oneline

# Show the actual changes
git log -S "RateLimiter" -p

# Find all commits that touched this code
git log -S "authentication" --source --all
```

### Find Commits by File Content

```bash
# All commits that modified this file
git log -- path/to/file.py

# With diffs
git log -p -- path/to/file.py

# Specific line range history
git log -L 100,150:src/model.py
git log -L :function_name:src/utils.py
```

### Using git blame

```bash
# Who changed each line?
git blame src/app.py

# Specific line range
git blame -L 10,20 src/app.py

# Ignore whitespace changes
git blame -w src/app.py

# Show email instead of name
git blame -e src/app.py
```

## Tagging

### Creating Tags

```bash
# Lightweight tag
git tag v1.0.0

# Annotated tag (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0"

# Tag specific commit
git tag -a v0.9.0 abc123 -m "Beta release"

# Tag with detailed message
git tag -a v1.0.0 -m "Release 1.0.0

Features:
- User authentication
- API rate limiting
- Performance monitoring"
```

### Viewing Tags

```bash
# List all tags
git tag
git tag -l

# List tags matching pattern
git tag -l "v1.*"

# Show tag details
git show v1.0.0

# List tags with messages
git tag -n
git tag -n5  # Show first 5 lines of message
```

### Managing Tags

```bash
# Delete tag
git tag -d v1.0.0

# Push tags to remote
git push origin v1.0.0
git push origin --tags  # Push all tags

# Delete tag from remote
git push origin --delete v1.0.0

# Checkout tag
git checkout v1.0.0
```

## Reflog

```bash
# Show all HEAD movements
git reflog

# Show last N entries
git reflog -10

# Reflog for specific branch
git reflog show feature-branch

# Recover lost commit
git reflog
git reset --hard HEAD@{2}

# Find when branch was created
git reflog show feature-branch | tail -1
```

## Best Practices

### Commit Message Template

```
<type>(<scope>): <subject>
# ↑       ↑            ↑
# |       |            +-> Summary in present tense (50 chars max)
# |       +-------------> Optional: api, auth, db, etc.
# +---------------------> Type: feat, fix, docs, style, refactor, test, chore

<BLANK LINE>

[optional body: explain what and why, not how (72 chars per line)]

<BLANK LINE>

[optional footer: references, breaking changes]

# Examples:
# feat(auth): add JWT token validation
# fix: prevent race condition in cache
# docs: update API documentation
# style: format code with black
# refactor(db): extract query builder
# test: add integration tests for API
# chore: update dependencies
```

### Atomic Commits

✅ **Good** - Single logical change:
```bash
git commit -m "fix: handle null values in user input validation"
git commit -m "feat: add password strength meter"
git commit -m "docs: update installation instructions"
```

❌ **Bad** - Multiple unrelated changes:
```bash
git commit -m "fix bug, add feature, update docs, refactor code"
git commit -m "various changes"
git commit -m "WIP"
```

### When to Amend

✅ **Amend** when:
- Fixing typo in commit message
- Adding forgotten file to commit
- Improving last commit before push
- All changes are still local

❌ **Don't Amend** when:
- Commit has been pushed
- Others are working on the same branch
- Creating a new feature/fix

Use new commit instead!

### When to Revert vs Reset

**Use `git revert`** when:
- Commit has been pushed
- Working on shared branch
- Want to preserve history
- Need audit trail

**Use `git reset`** when:
- Changes are local only
- On your own branch
- Want to rewrite history
- Cleaning up before push

## Common Workflows

### Fix Typo in Last Commit Message

```bash
git commit -m "feat: add usr authentication"  # Oops, typo!
git commit --amend -m "feat: add user authentication"
```

### Add Forgotten File

```bash
git commit -m "feat: add user model"
# Oops, forgot the test file!
git add tests/test_user.py
git commit --amend --no-edit
```

### Undo Last Commit (Keep Changes)

```bash
git reset --soft HEAD~1
# Files still staged, ready to recommit
```

### Undo Last Commit (Discard Changes)

```bash
git reset --hard HEAD~1
# ⚠️ Changes are gone forever!
```

### Revert a Bad Commit After Push

```bash
git revert HEAD
# Creates new commit that undoes changes
git push
```

### Find When Bug Was Introduced

```bash
# Using bisect
git bisect start
git bisect bad                 # Current version has bug
git bisect good v1.0.0        # v1.0.0 was working
# Git checks out middle commit
# Test it, then:
git bisect good  # or bad
# Repeat until git finds the culprit
git bisect reset

# Or using log
git log -S "problematic_code" --oneline
git show abc123
```

### Review Changes Before Committing

```bash
git status          # See what's changed
git diff            # See unstaged changes
git diff --staged   # See staged changes
git add -p          # Interactively stage chunks
git commit          # Commit with detailed message
```

## Keyboard Shortcuts

When using `git log`:
- `Space` - Next page
- `b` - Previous page
- `q` - Quit
- `/` - Search forward
- `?` - Search backward
- `n` - Next search result
- `N` - Previous search result

## Configuration

```bash
# Set default editor
git config --global core.editor "vim"
git config --global core.editor "code --wait"  # VS Code

# Set commit message template
git config --global commit.template ~/.gitmessage

# Set log aliases
git config --global alias.lg "log --oneline --graph --all"
git config --global alias.last "log -1 HEAD --stat"
git config --global alias.hist "log --pretty=format:'%h %ad | %s%d [%an]' --graph --date=short"

# Auto-sign commits
git config --global commit.gpgsign true
```

## Troubleshooting

### Commit to Wrong Branch

```bash
# On wrong branch
git commit -m "feat: add feature"

# Fix it
git log -1  # Copy commit hash
git reset --hard HEAD~1  # Undo commit
git checkout correct-branch
git cherry-pick abc123  # Apply commit to correct branch
```

### Accidentally Committed Secret

```bash
# ⚠️ If not pushed yet:
git reset --hard HEAD~1
# Remove from .env file
# Add .env to .gitignore
git add .gitignore
git commit -m "chore: add .env to gitignore"

# If already pushed:
# 1. Change the secret immediately!
# 2. Use git filter-branch or BFG Repo-Cleaner
# 3. Force push (notify team first!)
```

### Lost Commit After Reset

```bash
git reflog
# Find the commit hash
git cherry-pick abc123
# Or
git reset --hard abc123
```

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Documentation](https://git-scm.com/doc)
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [Git Cheat Sheet](https://training.github.com/downloads/github-git-cheat-sheet/)
