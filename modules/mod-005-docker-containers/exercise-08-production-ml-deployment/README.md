# Exercise 08: Production ML Deployment with Docker — Solution

## What the exercise asked for

Build a production-grade Docker image for an ML serving
workload covering: multi-stage builds, security hardening,
caching, health checks, and the operational surface.

## Production Dockerfile pattern

See [`Dockerfile`](./Dockerfile) for the full reference.

Key principles applied:

1. **Multi-stage build** — separate the builder (with all the
   compilers + build deps) from the runtime (lean, no build
   chain).
2. **Pin the base image by digest** — not just version tag, so
   the image is reproducible.
3. **Non-root user** — required by modern admission controllers.
4. **Minimal runtime layers** — no curl, no shell utilities in
   the final image unless needed.
5. **Health check** — Docker / Kubernetes uses this to know
   when the container is ready.
6. **Cache-friendly layer ordering** — requirements before
   source so dep changes don't invalidate source caches.

## Operational surface

```bash
# Build
docker build -t my-org/recs:v1.2.3 .

# Run locally
docker run --rm -p 8080:8080 my-org/recs:v1.2.3

# Inspect what's in the image
docker history my-org/recs:v1.2.3
docker image inspect my-org/recs:v1.2.3

# Scan for vulnerabilities
trivy image my-org/recs:v1.2.3
# (Critical CVEs in production = block the deploy.)

# Sign the image (with Cosign keyless from CI)
cosign sign my-org/recs:v1.2.3
```

## What a good final image has

- **Small**: typically <500MB for Python ML, <2GB if GPU
  libraries are included.
- **Non-root**: `USER` directive, runs as a non-zero uid.
- **No shell utilities** if they're not needed at runtime (no
  `curl`, no `bash` debugging helpers).
- **Pinned base image** by digest, not tag.
- **No secrets baked in** — they come via env vars or mounts
  at runtime.
- **Single concern** per image (your serving service, not a
  monorepo of services).

## Health checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1
```

For Kubernetes, also define `readinessProbe` and
`livenessProbe` in the Pod spec. The Dockerfile HEALTHCHECK is
mainly for `docker run` / `docker-compose` scenarios.

## Common mistakes

- **Single-stage build** — ships the build chain (gcc, cmake,
  cuda dev libs) in the runtime image. 5GB image when 800MB
  would do.
- **Running as root** — admission policies reject.
- **`COPY . .` then `pip install`** — every code change
  invalidates the pip cache; rebuilds take forever.
- **Hardcoded secrets** — `ENV API_KEY=sk-...` ends up in
  the image. Forever.
- **`:latest` tag** — non-reproducible deploys.
- **No `.dockerignore`** — ships `.git`, `__pycache__`,
  `.venv`, test fixtures. Bloats image, leaks data.

## Cross-references

- [`Dockerfile`](./Dockerfile) — the worked example.
- [`docker-compose.yml`](./docker-compose.yml) — local dev
  + observability stack.
- Engineer-track for production-grade serving:
  `engineer-solutions/mod-103-containerization`.
- The security track for image signing + supply chain:
  `ai-infra-security-learning/lessons/mod-010-supply-chain-security/`.
