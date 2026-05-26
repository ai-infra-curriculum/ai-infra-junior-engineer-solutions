# Exercise 04: Merging and Resolving Conflicts — Solution

## What the exercise asked for

Practice merge vs. rebase, conflict resolution, and choosing
the right strategy for ML team collaboration.

## Merge vs. rebase: when to use which

### Merge (preserves history as-is)

```bash
git switch main
git pull
git merge feat/add-resnet --no-ff
```

`--no-ff` forces a merge commit even when fast-forward is
possible. This preserves the fact that a branch existed. Useful
for teams that want history to reflect work organization.

### Rebase (rewrites your branch onto current main)

```bash
git switch feat/add-resnet
git fetch
git rebase origin/main
```

Rebase makes a linear history. Cleaner to read but rewrites
commits — never rebase commits others may have pulled.

### Rule of thumb

- **Local branch, not pushed**: rebase freely.
- **Pushed feature branch, no one else is working on it**:
  rebase + force-with-lease (`git push --force-with-lease`).
- **Shared branch with collaborators**: merge only.
- **`main`**: merge only.

## Resolving merge conflicts

When a merge or rebase produces conflicts:

```bash
# Git tells you which files have conflicts
$ git merge feat/add-resnet
CONFLICT (content): Merge conflict in src/api/routes.py
Automatic merge failed; fix conflicts and then commit the result.

# See the conflicted files
git status

# Inspect the conflict markers in the file
$ cat src/api/routes.py
...
<<<<<<< HEAD
def predict(image):
    return model.predict_one(image)
=======
def predict(image, batch=False):
    return model.predict(image, batch=batch)
>>>>>>> feat/add-resnet
...
```

### Three-way conflict resolution

The conflict has three views:
- `HEAD` — what's on main now.
- `feat/add-resnet` — what your branch tried to change.
- The merge base — the common ancestor (use `git show :1:src/api/routes.py` if needed).

Edit the file to combine both intentions, remove the markers,
then:

```bash
git add src/api/routes.py
git merge --continue   # or git rebase --continue
```

### Aborting

If the conflict is hopeless:

```bash
git merge --abort
# or
git rebase --abort
```

You're back to where you were before. No work lost.

## ML-specific conflict patterns

- **Pickled model files**: don't version-control them. Use DVC,
  MLflow, or a model registry. (See exercise-08-git-lfs-ml-
  projects for the LFS approach.)
- **Notebooks**: `.ipynb` files merge poorly because they're
  JSON with execution metadata. Use `nbstripout` to strip
  outputs before commit, or `jupytext` to keep `.py` alongside
  `.ipynb`.
- **Requirements files**: when two PRs both add deps, sort
  alphabetically + use lockfiles (`requirements-lock.txt`,
  `poetry.lock`, `uv.lock`).

## Tools that help

- `git mergetool` — opens a 3-way merge in vimdiff /
  meld / VS Code.
- `git rerere` — REuse REcorded REsolution. Caches your
  manual conflict resolutions; replays them on the next
  occurrence.

## Common mistakes

- Rebasing a shared branch (rewrites history others have
  pulled).
- Accepting "theirs" or "ours" blindly without reading the
  diff.
- Resolving a conflict by deleting both sides "to remove the
  markers."
- Committing without removing all `<<<<<<<` / `>>>>>>>`
  markers (use a pre-commit hook to catch this).

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-003-git-version-control/exercises/exercise-04-merging-conflicts.md`
- Next: `exercise-05-collaboration/`.
