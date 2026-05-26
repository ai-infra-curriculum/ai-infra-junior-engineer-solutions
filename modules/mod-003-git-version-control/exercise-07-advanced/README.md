# Exercise 07: Advanced Git — Solution

## What the exercise asked for

Practice the advanced Git operations every engineer eventually
needs: interactive rebase, cherry-pick, bisect, reflog,
worktrees, hooks.

## Interactive rebase: cleaning up history

```bash
# Clean up the last 5 commits before pushing
git rebase -i HEAD~5
```

This opens your editor with one line per commit. Replace
`pick` with:
- `r` (reword) — change the commit message.
- `e` (edit) — pause to amend the commit.
- `s` (squash) — combine into the previous commit.
- `f` (fixup) — combine, discard this message.
- `d` (drop) — delete the commit entirely.
- Reorder lines to reorder commits.

```text
pick abc1 feat: add training pipeline
squash def2 fix: typo in training pipeline
squash ghi3 fix: another typo
pick jkl4 feat: add evaluation pipeline
reword mno5 chore: bump torch
```

Result: 3 clean commits instead of 5 noisy ones.

> Rule: never interactive-rebase published commits.

## Cherry-pick: applying one commit elsewhere

```bash
# Find the commit you want
git log --oneline feat/some-branch

# Apply just that commit to current branch
git cherry-pick abc1234

# A range of commits
git cherry-pick abc1234..def5678
```

Use when:
- A bug fix from a release branch needs to go to main.
- One commit from an experiment turned out to be useful;
  the rest didn't.

## Bisect: finding the commit that broke things

```bash
# You know it broke. When?
git bisect start
git bisect bad             # current commit is bad
git bisect good v1.2.0     # v1.2.0 worked

# Git checks out a commit halfway between.
# Test it. Then:
git bisect bad      # or git bisect good

# Repeat. Git narrows down with binary search.
# When it finds the commit, it tells you the SHA.

git bisect reset   # restore your original HEAD
```

Bonus: automate the test:

```bash
git bisect start HEAD v1.2.0
git bisect run python test_the_thing.py
```

Git runs the script at each checkpoint; bad if exit code is
non-zero, good otherwise. Magical for finding regressions.

## Reflog: recovering lost work

`git reflog` records every HEAD move. If you've ever:
- Force-reset and lost commits.
- Deleted a branch with unmerged work.
- Done a rebase that went wrong.

…the lost commits are usually still recoverable via reflog.

```bash
git reflog
# 1a2b3c4 HEAD@{0}: reset: moving to HEAD~5
# 5d6e7f8 HEAD@{1}: commit: feat: thing I wanted to keep
# ...

# Recover
git switch -c recovered-work 5d6e7f8
```

Reflog entries expire after 90 days by default. Don't delay.

## Worktrees: multiple branches checked out simultaneously

```bash
# Add a worktree for a long-running experiment branch
git worktree add ../my-repo-experiment experiment/big-refactor

# Work in both directories simultaneously.
# Each worktree shares the same Git data but has its own HEAD.

# When done:
git worktree remove ../my-repo-experiment
```

Use when you need to bounce between branches without
disrupting one. Especially nice for ML where you might want
two training runs going on different experimental branches.

## Hooks: enforce conventions automatically

Pre-commit hooks run before each commit. Use them to:
- Run linters / formatters.
- Strip notebook outputs.
- Block commits containing secrets.

The `pre-commit` framework is the standard:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-added-large-files
      - id: detect-private-key
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout
```

Install: `pre-commit install`. Now every commit runs the hooks.

## Common mistakes

- Rebase-interactive on pushed commits (rewrites history;
  collaborators see chaos).
- Cherry-picking without testing (the commit's dependencies
  may not have come along).
- Bisecting without an automated test (you'll lose track of
  which commits are good/bad).
- Reflog after 90 days (gone).

## Cross-references

- Exercise prompt: `ai-infra-junior-engineer-learning/lessons/mod-003-git-version-control/exercises/exercise-07-advanced.md`
- Next: `exercise-08-git-lfs-ml-projects/`.
