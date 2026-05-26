# Exercise 06: Git Workflows for ML Projects — Solution

## What the exercise asked for

Apply Git workflows to the realities of ML projects: large
data files, model artifacts, experiment branches, and the
inevitable Jupyter notebook conflicts.

## The 4 things that make ML Git different

1. **Big binary files**: model checkpoints, datasets, audio,
   images. Git is not optimized for binary content.
2. **Notebook files**: `.ipynb` is JSON with execution
   metadata; merges poorly.
3. **Experiment branches**: many short-lived branches for
   parameter sweeps, architecture tries.
4. **Reproducibility**: which code + data + hyperparameters
   produced which model.

## Pattern: don't commit data or models

```bash
# .gitignore
data/
models/*.bin
models/*.pt
models/*.onnx
*.pkl
*.h5
*.parquet
__pycache__/
.ipynb_checkpoints/
.venv/
```

Use:
- **DVC** or **LakeFS** for data versioning.
- **MLflow** or a model registry for model versioning.
- **Git** for code, configs, notebooks (stripped).

The git-lfs alternative is covered in exercise 08; it's
appropriate for medium-sized files in a single repo.

## Pattern: strip notebook outputs

Notebook outputs explode the file size and produce noisy diffs.
Install `nbstripout`:

```bash
pip install nbstripout
nbstripout --install  # sets up Git filter
```

Now commits automatically strip outputs. Pulls + checkouts
work as normal; outputs are local-only.

Alternative: use **jupytext** to keep a `.py` alongside the
`.ipynb`. Review the `.py` in PRs; the `.ipynb` is generated.

## Pattern: experiment-branch hygiene

For a hyperparameter sweep:

```bash
# Each experiment gets its own branch
git switch -c experiment/lr-1e-3
# ... train, log to MLflow ...
git commit -am "experiment: lr=1e-3, batch=64, epochs=10"
git push -u origin experiment/lr-1e-3

# After the sweep, identify the winner and merge ONLY that
git switch main
git merge experiment/lr-3e-4

# Delete the losers
git branch -D experiment/lr-1e-3 experiment/lr-1e-4 experiment/lr-1e-5
```

Don't merge all experiments to main — only the chosen one.
Treat the experiment branches as ephemeral.

## Pattern: link Git commits to model artifacts

For reproducibility, model artifacts should record their Git
SHA:

```python
import subprocess

def get_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()

# In your training script:
model_metadata = {
    "training_code_sha": get_git_sha(),
    "dataset_version": "v3.2",
    "hyperparameters": {...},
    "trained_at": "2026-01-15T10:00:00Z",
}
mlflow.log_dict(model_metadata, "metadata.json")
```

Months later, "which code trained this model?" is one MLflow
lookup away.

## Pattern: a Makefile for "the ML workflow"

```makefile
.PHONY: train evaluate deploy

train:
	python train.py --config configs/base.yaml

evaluate:
	python evaluate.py --model artifacts/model-$$(git rev-parse --short HEAD)

deploy: evaluate
	./deploy.sh
```

Anchoring artifacts on Git SHA gives you reproducibility +
prevents "which model is in production" ambiguity.

## Common mistakes

- Committing model files because "they're small enough" (they
  grow).
- Committing notebooks with outputs (huge diffs, secret leaks
  in cell output).
- Letting experiment branches pile up forever.
- Training-time code that doesn't record the Git SHA.

## Cross-references

- Exercise prompt: `ai-infra-junior-engineer-learning/lessons/mod-003-git-version-control/exercises/exercise-06-ml-workflows.md`
- Next: `exercise-07-advanced/`.
