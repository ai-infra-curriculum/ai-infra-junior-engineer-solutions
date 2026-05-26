# Exercise 03: Branching Strategies — Solution

## What the exercise asked for

Practice branching for parallel work on an ML project: feature
branches, branch naming conventions, switching contexts, and
cleaning up.

## Key commands + outputs

### Creating + switching branches

```bash
# Create a branch from current HEAD
git branch feat/new-model-version

# Create + switch in one step
git checkout -b feat/new-model-version

# Newer Git (>=2.23): the cleaner verb
git switch -c feat/new-model-version

# Switch to existing branch
git switch main
```

### Branch naming conventions

| Pattern | Example | Use |
|---|---|---|
| `feat/<short-desc>` | `feat/add-resnet-endpoint` | New feature |
| `fix/<ticket>-<desc>` | `fix/JIRA-123-batch-size-bug` | Bug fix tied to ticket |
| `chore/<desc>` | `chore/bump-torch-2.1` | Maintenance |
| `experiment/<desc>` | `experiment/try-distilbert` | Exploratory; may be thrown away |
| `release/<version>` | `release/v1.4.0` | Pre-release stabilization |
| `hotfix/<desc>` | `hotfix/prod-memory-leak` | Production fix that bypasses normal flow |

Use kebab-case in the description part. No spaces. Keep names
short (under 40 chars) so they're easy to type and read in CI
output.

### Listing branches

```bash
# All local branches
git branch

# Remote-tracking too
git branch -a

# With last-commit info
git branch -v

# Branches merged into main (candidates for deletion)
git branch --merged main
```

### Cleaning up

```bash
# Delete a local branch (refuses if not merged)
git branch -d feat/old-branch

# Force delete (use only on branches you don't care about)
git branch -D experiment/abandoned

# Delete a remote-tracking ref that's gone upstream
git fetch --prune

# Or all at once
git remote prune origin
```

## Branching workflow for ML projects

A common pattern:

```text
main (always deployable)
├── feat/add-fraud-model
├── feat/improve-preprocessing
├── experiment/try-lightgbm
└── fix/prod-503-on-large-images
```

Rules:
- `main` is always deployable.
- Feature branches are short-lived (≤ 1-2 weeks).
- Experiments are explicitly tagged as such; expected to be
  thrown away.
- Hotfixes go through CI like everything else (no `--no-verify`).

## When to *not* branch

Don't branch for:
- Trivial fixes that fit in one commit you'll push immediately
  (just push to main on small teams; via PR on larger teams).
- "Just in case I need to experiment" — make the branch when
  you actually start.

## Switching mid-work

When you need to switch to a different branch with unfinished
work:

```bash
# Save current work
git stash push -m "WIP on preprocessing refactor"

# Switch
git switch fix/JIRA-456

# ... fix and commit ...

# Switch back + restore
git switch feat/preprocessing-refactor
git stash pop
```

## Common mistakes

- Branching from a stale `main` — always `git fetch && git pull`
  before creating a branch.
- Letting a feature branch live for weeks (merge conflicts
  compound).
- Forgetting to delete merged branches (clutter).
- Force-deleting a branch with unmerged work you didn't realize
  was unmerged.

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-003-git-version-control/exercises/exercise-03-branching.md`
- Next: `exercise-04-merging-conflicts/`.
