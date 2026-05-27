# SOLUTION — Simple Model API

> Read this *after* you have built your own version. This file
> explains the reasoning behind the reference implementation: why we
> made the choices we did, what we deliberately ruled out, and what
> the common mistakes are. Use `SOLUTION_GUIDE.md` for the
> "how to run it" walkthrough; this file explains *why it's shaped
> the way it is*.

## What this project is really teaching

Loading a model and exposing it behind HTTP is the easy part —
twenty minutes of Flask. What this project is actually teaching is
the gap between **a model that runs in a notebook** and **a model that
serves traffic at 3 AM on a Sunday**:

- The model must load exactly once, not per request.
- The HTTP layer must reject malformed inputs without crashing.
- Logs must be machine-readable.
- The container must run as a non-root user with a minimal attack
  surface.
- Configuration must come from environment variables, not
  hardcoded constants.

Every architectural decision below traces back to one of those
production realities.

## Architectural decisions and *why*

### Decision 1: Model loaded once at process startup, never per request

In `model_loader.py` the model is loaded inside `ModelLoader.__init__`
and held in module-level state. A naive first attempt usually loads
the model inside the request handler, which adds 500ms-2s of latency
per request and lets a 50 RPS API consume 30+ GB of memory in
duplicated model state.

**Anti-pattern to avoid**: lazy-loading the model on first request
("warm-up on first call"). It still works, but the first real user
takes the warm-up hit and pod autoscaling thrashes because the
readiness probe passes before the model is loaded.

### Decision 2: PIL Image → tensor in one preprocessing function

`ModelLoader.preprocess_image()` is a single function that takes
either a file path or a `PIL.Image` and returns the model input
tensor. The Flask handler converts the uploaded multipart payload
into a PIL Image with `Image.open(io.BytesIO(...))` and then hands
it straight to the loader.

Why not pass raw bytes? Because the **same preprocessing function
must run in unit tests** that don't go through Flask. Centralizing
input handling at the PIL Image boundary is the cheapest way to
make the inference path independently testable.

### Decision 3: Configuration via environment variables (with a
config dataclass)

`config.py` declares a frozen dataclass populated from `os.environ`
at import time. This forces the question "what knobs does this
service have?" to be answered explicitly in one place. The CLI does
not accept hyperparameters as flags; Kubernetes ConfigMaps inject
them as env vars.

**Anti-pattern to avoid**: reading `os.environ.get("X", "default")`
scattered through the codebase. After three sprints you no longer
know what knobs exist.

### Decision 4: Three endpoints, three concerns

- `GET /health` — process is alive (liveness probe target).
- `GET /info` — model is loaded and ready (readiness probe target).
- `POST /predict` — the actual workload.

These are separated because Kubernetes liveness and readiness probes
have **different recovery semantics**: liveness failure kills the
pod, readiness failure removes it from the service mesh. Conflating
them means a slow model load can cause a kill loop instead of
graceful traffic draining.

### Decision 5: Structured JSON logging via Python's logging
module, not print()

Look at `app.py`'s logging config — every log line is structured
and emits JSON. `print()` to stdout works in development, but a
production log aggregator (Loki, Datadog, CloudWatch Logs Insights)
parses fields. The cost is a 15-line logging config; the value is
the entire observability layer becoming queryable.

### Decision 6: Top-K returned, not just argmax

`predict()` returns the top-K predictions sorted by confidence,
controlled by a `top_k` form field. ImageNet has 1,000 classes; the
single highest-scoring label is often wrong on out-of-distribution
inputs, and the confidence gap between the top-1 and top-2 is more
diagnostic than the top-1 alone.

Letting clients ask for top-5 also exposes a confidence-calibration
signal they can use downstream without retraining the model.

### Decision 7: Request size limit + safe filename

The Flask app sets `MAX_CONTENT_LENGTH` and runs `secure_filename()`
on every uploaded file. Without these, an attacker can either
exhaust memory by uploading multi-GB files or write outside the
intended directory via path traversal. The cost is two lines; the
risk it avoids is a 3 AM incident.

### Decision 8: Dockerfile uses multi-stage + non-root user

The Dockerfile compiles the heavy dependencies in a builder stage,
copies only the final site-packages into a slim runtime stage, and
creates a dedicated `appuser` to run the process. Image size drops
from ~3 GB to ~600 MB; the container can't escalate to root if a
dependency is compromised.

**Anti-pattern to avoid**: running as root with `apt-get install`
artifacts left in the runtime image. Easy to do, costly to fix.

## Trade-offs we deliberately accepted

### Single-process Flask, not Gunicorn

For a teaching project, single-process Flask is the right
simplification. For production, the same code runs under Gunicorn
with `gunicorn -w 4 -k gthread -b 0.0.0.0:5000 app:app` and that's
it — but the inference call is the bottleneck, and threading the
HTTP layer past the GIL won't help PyTorch CPU inference. Real
scale-up moves to vLLM/Triton/TorchServe, which is the subject of
the engineer-track `mod-110/ex-01` exercise.

### CPU inference by default

The Docker image targets CPU, not GPU. Most beginners can't afford
to learn CUDA + nvidia-docker simultaneously. Once the API works
on CPU, swapping to a GPU image is a runtime config change, not an
architecture change. Junior engineers should ship the CPU version,
prove the API works, then deal with GPU.

### Synchronous prediction, not batched

Each `/predict` call processes one image. Batching multiple
concurrent requests into a single forward pass would improve
throughput 3-5x at the cost of significant complexity (request
queue, batch-window scheduler, partial-batch timeout). For the
project's traffic levels, this is over-engineering. The
recommendation in the SOLUTION_GUIDE flags the batching pattern as
the next-iteration target.

## What the tests cover (and why)

`tests/test_app.py` is structured around the three failure
categories that matter most for an inference service:

1. **Happy-path correctness** — a known image returns expected
   labels.
2. **Input validation** — empty uploads, wrong file types,
   oversized payloads return 4xx, not 5xx.
3. **Probe semantics** — `/health` returns 200 even when the model
   is loading; `/info` returns 200 only when the model is ready.

What the tests *don't* cover: model accuracy on real-world inputs.
That's a model-evaluation concern, not an API concern, and lives in
the model-monitoring track.

## Common mistakes graders see

1. **Loading the model inside `/predict`**: easy to do, kills
   throughput, looks fine in single-request tests.
2. **`print()` instead of `logger`**: lines vanish into the void
   the moment you deploy.
3. **`USER root` in the Dockerfile**: works locally, blocks the
   pod from passing PodSecurityPolicy in any real cluster.
4. **No `secure_filename()`**: subtle, dangerous, ignored until
   it's a CVE.
5. **Returning the model's raw output tensor as JSON**: PyTorch
   tensors don't serialize. Always call `.tolist()` first.
6. **Reading config from `os.environ` inline**: works once,
   becomes unmanageable by project 3.

## When to go beyond this implementation

The implementation is a deliberate **floor**, not a target. Beyond
the project's scope, real production paths include:

- Move to ONNX or TensorRT for 2-5x inference speed-up.
- Add a Redis-backed prediction cache for repeated requests.
- Front the API with NGINX for TLS termination + rate limiting.
- Replace Flask with FastAPI for async + auto OpenAPI specs.
- Move to vLLM/Triton/TorchServe once you outgrow Flask.

Each of those is its own exercise later in the curriculum. Master
this one first.

## Related curriculum touchpoints

- `engineer/mod-103/ex-04-container-security` — supply-chain
  hardening of the Dockerfile you wrote here.
- `engineer/mod-104/ex-04-k8s-cluster-autoscaler` — what happens
  when the API needs to scale.
- `engineer/mod-108/ex-02-ml-model-monitoring` — what production
  observability looks like for the same API.
- `engineer/mod-110/ex-01-production-llm-serving` — the same shape
  at LLM scale with vLLM.
