# Exercise 02: Working with Commits and History — Solution

## What the exercise asked for

Practice atomic commits, conventional commit messages, history
navigation, amending, reverting, and `git show` inspection on a
realistic ML inference API.

## Key commands + outputs

### Atomic commits with conventional messages

```bash
# Stage only the files for one logical change
git add src/monitoring/metrics.py tests/test_metrics.py

# Commit with conventional format
git commit -m "feat(monitoring): add model latency metrics

- Track per-endpoint latency histogram
- Export via Prometheus client
- Add corresponding unit tests"
```

### Inspecting history

```bash
# Recent commits with one-line view
git log --oneline -10

# Commits touching a specific file
git log --oneline -- src/preprocessing.py

# Commits matching a message pattern (e.g., bug fixes)
git log --oneline --grep="^fix"

# Commits by a specific author in last 30 days
git log --oneline --author="$(git config user.email)" --since="30 days ago"

# Show what changed in a specific commit
git show abc123 --stat
git show abc123 -- src/api.py
```

### Searching for code changes (the pickaxe)

```bash
# Find when "rate_limit" was introduced
git log -S"rate_limit" --oneline

# Find commits where a specific function was modified
git log -L :preprocess_image:src/preprocessing.py
```

### Amending the last commit

```bash
# Forgot to stage a file
git add src/api/routes.py
git commit --amend --no-edit

# Or to fix the message
git commit --amend -m "feat(api): add rate-limiting middleware"
```

> Caveat: only amend commits you haven't pushed. Amending a
> pushed commit rewrites history; collaborators see it as a
> divergent branch.

### Reverting safely

```bash
# Create a new commit that undoes a specific commit
git revert abc123

# Revert a range (the order matters — newer first)
git revert HEAD~3..HEAD
```

`git revert` preserves history. Use it for any commit that's
been pushed. Use `git reset --hard` only on local-only commits.

## Conventional commit prefixes used in this exercise

| Prefix | Use |
|---|---|
| `feat:` | New user-facing feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code structure change with no behavior change |
| `test:` | Test code only |
| `chore:` | Build, dep updates, tooling |
| `perf:` | Performance improvement |

Add a scope in parens: `feat(api): ...`, `fix(monitoring): ...`.

## ML-infrastructure-specific commit patterns

```text
feat(serving): add ResNet50 endpoint
fix(preprocessing): clamp input dimensions to model spec
perf(inference): batch requests at the queue boundary
chore(deps): bump torch to 2.1.2 for CVE patch
```

## Quality checklist for your commit history

- [ ] Each commit changes one thing (atomic).
- [ ] Commit messages follow conventional format.
- [ ] No "WIP" or "fixes" without context in messages.
- [ ] No commits with `--no-verify` unless documented why.
- [ ] No force-pushes to shared branches.

## Common mistakes

- Bundling unrelated changes into one commit ("ugh just commit everything").
- Vague messages ("update files", "fix bug").
- Amending pushed commits.
- Using `reset --hard` on shared branches.

## Cross-references

- The matching exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-003-git-version-control/exercises/exercise-02-commits-history.md`
- Next exercise: `exercise-03-branching/`.
