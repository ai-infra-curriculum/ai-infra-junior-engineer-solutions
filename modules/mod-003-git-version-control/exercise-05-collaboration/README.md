# Exercise 05: Git Collaboration on ML Teams — Solution

## What the exercise asked for

Practice pull-request workflows, code review, branch protection,
and the team conventions that keep a Git repo healthy at small
team scale.

## The pull-request workflow

```bash
# 1. Start from a fresh main
git switch main && git pull

# 2. Create a feature branch
git switch -c feat/add-resnet-endpoint

# 3. Work, commit, repeat
# (Atomic commits with conventional messages — see exercise 02)

# 4. Push the branch
git push -u origin feat/add-resnet-endpoint

# 5. Open the PR from GitHub UI or:
gh pr create --title "feat(api): add ResNet endpoint" \
   --body "$(cat <<EOF
## Summary
Adds a new /v1/resnet endpoint serving ResNet50 image
classification.

## Test plan
- [ ] Unit tests pass (added in tests/test_resnet.py)
- [ ] Integration test against /v1/resnet returns expected class
- [ ] Latency p95 < 200ms on the dev cluster

## Risks
- New endpoint; no existing traffic affected.
- Memory footprint of ResNet50 weights: ~100MB.
EOF
)"
```

## What a good PR looks like

- **Title**: conventional-commit format. Under 70 chars.
- **Body**: Summary + test plan + risks. Reviewers should be
  able to assess from the body alone.
- **Size**: <500 lines diff if possible. Bigger PRs get worse
  review.
- **Commits**: clean history (squash if there's noise, but keep
  meaningful checkpoints).
- **CI green**: every check passes before requesting review.

## Code review etiquette

For the reviewer:
- Comment on *the code*, not the author.
- Distinguish "must change before merge" from "nice to have"
  (use `nit:` prefix for the latter).
- Approve when the code is good enough, even if you have
  preferences. Don't block on style if a linter would.

For the author:
- Respond to every comment, even if just "✓ done" or "agreed,
  pushing fix".
- Don't take it personally.
- Ask "why?" if you don't understand a review comment.

## Branch protection on `main`

Settings for production repos:
- Require pull requests before merging.
- Require at least 1 review (ideally 2 for sensitive code).
- Require status checks (CI must pass).
- Require branches to be up-to-date before merging.
- Restrict who can push directly to `main` (only admins, and
  only in emergencies).
- Require linear history (no merge commits, force rebases).

## Stale branches

```bash
# List branches not updated in 30 days
git for-each-ref --sort=-committerdate refs/heads/ \
  --format='%(refname:short) %(committerdate:short)' \
  | awk -v cutoff="$(date -v-30d +%Y-%m-%d)" '$2 < cutoff'

# Delete stale branches that have been merged
for b in $(git branch --merged main | grep -v '\* main'); do
  git branch -d "$b"
done
```

Most repos accumulate dead branches. A monthly cleanup is
healthy.

## Common mistakes

- Pushing to `main` directly (use PRs).
- Force-pushing to shared branches.
- "I'll squash later" — squashing is a per-PR decision, not a
  later cleanup.
- Approving without reading.
- Treating CI failures as "flaky tests" without investigating.

## Common ML-team additions

- **CODEOWNERS** file routes review requests automatically
  (e.g., changes to `serving/` → @ml-serving-team).
- **Templates** for PRs (`.github/PULL_REQUEST_TEMPLATE.md`).
- **Pre-merge checks** for model artifact size, license
  headers, dependency CVEs.

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-003-git-version-control/exercises/exercise-05-collaboration.md`
- Next: `exercise-06-ml-workflows/`.
