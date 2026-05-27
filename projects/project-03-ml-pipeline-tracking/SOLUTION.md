# SOLUTION — ML Pipeline with Experiment Tracking

> Read this *after* you have wired the pipeline yourself. This file
> explains the architectural reasoning behind the Airflow + MLflow +
> DVC + Great Expectations stack, the trade-offs we accepted, and
> the failure modes each piece prevents. `SOLUTION_GUIDE.md` is the
> how-to-run-it; this is the *why-it's-shaped-this-way*.

## What this project is really teaching

A working ML pipeline is rarely an algorithm problem. It is an
**operational** problem:

- Which dataset produced which model?
- Why is yesterday's model better than today's, and what changed?
- When the input distribution shifts, how do we notice before our
  users do?
- If a regulator asks for the full provenance of a production
  prediction, can we produce it?

Five tools, four jobs:

| Tool | Job | What it gives you |
|---|---|---|
| Airflow | Orchestration | Repeatable execution + retries + scheduling |
| Great Expectations | Data validation | Caught-bad-input gate before training |
| DVC | Data versioning | Reproducibility: "which data made this model?" |
| MLflow | Experiment tracking | Discoverability: "which run is best?" |
| MinIO/S3 | Artifact storage | Long-term blob storage for datasets + models |

Pick any four of the five and you have a hole that bites you in
production. The pipeline is shaped to make every one of those
questions answerable from artifacts, not from human memory.

## Architectural decisions and *why*

### Decision 1: Validate-before-train, not validate-after-train

Great Expectations runs as a discrete pipeline step **between
ingestion and preprocessing**. A run that fails the data-quality
gate halts before consuming GPU hours. The validation result is a
typed artifact stored alongside the model in MLflow, so the
"why was this model rejected?" question is answerable months later.

**Anti-pattern to avoid**: validating inside the training task with
`assert` statements. When the assert fires, the GPU time is spent,
the diagnostic artifacts (sample bad rows, distribution stats) are
gone, and the failure mode is "training crashed" with no actionable
output.

### Decision 2: Every artifact is logged to MLflow, not just the model

A naive first attempt logs the trained model and the headline
accuracy number. The reference implementation logs:

- The training script itself (`mlflow.log_artifact("train.py")`).
- The Conda environment / requirements.txt.
- The Great Expectations validation report.
- The dataset hash (DVC pointer).
- The hyperparameters as MLflow params, not as a JSON blob.
- The per-class confusion matrix.

This is the difference between a model and a *reproducible model*.
Three months later, someone asking "what produced this model" can
answer it without you.

### Decision 3: DVC tracks data; Git tracks code; the pipeline links them

DVC stores the data blob in S3/MinIO and writes a small pointer
file (`.dvc`) into the git repo. Git tracks the pointer. The
combination gives you:

- A `git checkout` that brings the right code *and* points at the
  right data version.
- Diffs in code review that surface "the data changed" as an
  explicit, reviewable event.
- Per-run reproducibility without bloating the git history with
  multi-GB datasets.

**Anti-pattern to avoid**: dumping the dataset into a git LFS
repository. Works for small datasets, breaks at any meaningful
scale, and offers nothing DVC doesn't.

### Decision 4: MLflow Model Registry, not just experiment tracking

The pipeline ends with a `mlflow.register_model()` call into the
Model Registry, not just a `log_model()` into the experiment. The
distinction:

- **Experiment-level `log_model`**: "here is a model I trained".
- **Registry-level `register_model`**: "here is a model that's a
  candidate for promotion through dev → staging → prod".

The registry separates *experimentation* (running 200 things) from
*deployment candidacy* (the 5 things worth considering). Without
it, the deployment pipeline either consumes everything or relies
on naming conventions you have to enforce by hand.

### Decision 5: Airflow Task dependencies model the DAG, not the code

The DAG file builds tasks as `PythonOperator`s and wires them with
`>>` operators. Each task is a single Python function in the
`src/` package. Why not put the logic directly in the DAG file?

- Tasks become **importable from tests** that don't need Airflow
  running.
- Failures in a task can be **rerun independently** via
  `airflow tasks test`.
- The DAG file stays under 200 lines and reads as **what runs in
  what order**, not as a smear of business logic.

The senior-engineer track formalizes this as the "thin DAG, fat
package" pattern.

### Decision 6: Airflow retries set explicitly per task

The DAG sets `retries` and `retry_delay` per task, not on the DAG
default:

- **Ingestion**: 3 retries with exponential backoff (transient
  network failures are common).
- **Validation**: 0 retries (a validation failure should not be
  papered over by retry).
- **Training**: 0 retries (training is non-idempotent in subtle
  ways — random seeds, MLflow run IDs).
- **Register**: 2 retries (idempotent if the model artifact
  already exists in S3).

A blanket `default_args = {"retries": 3}` is comfortable but
hides decisions that should be deliberate.

### Decision 7: docker-compose for local dev, not Kubernetes

The local environment is `docker-compose up`. The same services in
production would run on Kubernetes (Airflow Helm chart, MLflow
deployed separately, MinIO replaced by S3). For a junior-tier
project, docker-compose:

- Starts in 30 seconds.
- Runs on a laptop without a cluster.
- Lets the student focus on the *pipeline shape*, not on
  Kubernetes plumbing.

The project-02 deployment pattern carries forward when the student
is ready.

## Trade-offs we deliberately accepted

### MLflow over Weights & Biases / Comet

MLflow is open-source, self-hostable, and integrates with the
exact tools (DVC, scikit-learn, PyTorch) the rest of the
curriculum uses. W&B is more polished UX-wise but its
hosted-by-default model conflicts with the data-stays-in-house
requirement junior-tier projects ladder up to.

### Airflow over Prefect / Dagster

Airflow's market share + ecosystem maturity outweighs Prefect/
Dagster's better DX. Junior engineers should learn the tool they'll
encounter in the field. The senior-engineer track explores the
alternatives.

### MinIO over real S3

MinIO is S3-compatible and runs in docker-compose. The training
code uses the boto3 S3 client; switching from MinIO to AWS S3 is
an environment variable change, not a code change. Real S3 costs
money on small budgets; MinIO is free.

### Great Expectations over custom asserts / Pandera / TFDV

GE's expectations are versioned, diffable in code review, and
produce a structured report artifact. Pandera is more pythonic
but lacks the artifact-store-output story. TFDV is more powerful
on huge datasets but its TF dependency adds runtime weight a
junior project shouldn't have to manage.

## What the tests cover

The `tests/` directory exercises:

1. **Each Airflow task in isolation** — `task.function(**context)`
   returns the expected output for known input.
2. **A docker-compose integration test** — bring up the full
   stack, trigger the DAG, assert it reaches success state.
3. **Reproducibility check** — run the pipeline twice on the same
   data and assert the registered model checksums match.

What the tests *don't* cover: model quality. That's the
`engineer/mod-106/ex-05-model-monitoring-drift` exercise.

## Common mistakes graders see

1. **Logging the model and nothing else to MLflow**: future-you
   can't reproduce anything.
2. **Storing the dataset in the git repo**: works at first, breaks
   at scale.
3. **Putting business logic directly in the DAG file**: untestable,
   unreusable, becomes a tangle.
4. **`retries=3` as a global default**: hides decisions that should
   be deliberate.
5. **Manually copying models from `mlruns/` into prod**: defeats
   the entire point of the registry.
6. **Validation step is a `try/except` that logs and continues**:
   data quality issues should *halt* the pipeline, not warn.

## When to go beyond this implementation

Beyond Project 03's scope:

- Feature-store integration (Feast, Tecton) so serving-time features
  match training-time features.
- Continuous training: schedule the pipeline daily/hourly, not
  manually.
- Champion-challenger A/B comparison instead of single-model
  promotion.
- Lineage UI (DataHub, OpenMetadata) wrapping the MLflow + DVC
  artifacts.

Each is its own later exercise. Master the deterministic, full-
provenance, single-model path first.

## Related curriculum touchpoints

- `engineer/mod-105/ex-04-workflow-orchestration-airflow` — same
  shape, larger scale, more rigor.
- `engineer/mod-106/ex-04-experiment-tracking-mlflow` — the
  registry's role in promoting models to production.
- `engineer/mod-106/ex-05-model-monitoring-drift` — what happens
  *after* a model is registered and serving.
- `mlops/project-1-ml-pipeline` — the next-tier version of this
  pipeline with full production rigor.
