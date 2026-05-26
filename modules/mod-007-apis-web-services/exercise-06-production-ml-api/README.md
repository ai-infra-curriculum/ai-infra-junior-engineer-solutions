# Exercise 06: Production ML API — Solution

## What the exercise asked for

Build a production-grade ML serving API in FastAPI with:
input validation, model loading via lifespan, async patterns,
structured logging, metrics, rate limiting, health checks,
and error handling.

## The reference structure

See [`api.py`](./api.py) for a working FastAPI app showing the
patterns.

Key elements:

1. **Lifespan-managed model load** — load once at startup,
   not per request.
2. **Pydantic schemas** at the boundary for input validation.
3. **Async route handlers** — the framework is async-native.
4. **Structured JSON logging** — queryable in production.
5. **Prometheus metrics** at `/metrics`.
6. **Per-tenant rate limit** placeholder (Redis-backed in
   real deployments).
7. **Health endpoints**: `/healthz` (basic), `/readyz`
   (deep — model loaded? upstream reachable?).
8. **Error handlers** that return shape-consistent error
   responses, not stack traces.

## What "production" requires beyond this

This exercise's solution is a single-file app. A real
production deployment also has:

- **Tests** (unit + integration; see exercise-07 of mod-001).
- **Docker** (see exercise-08 of mod-005).
- **Kubernetes manifests** (see mod-006-kubernetes-intro).
- **CI/CD** that builds, signs, and deploys.
- **Observability**: distributed tracing, log aggregation.
- **Authentication**: API keys or OIDC.
- **Per-tenant authorization**: covered in the Engineer
  track's mod-104 multi-tenancy exercises and the Security
  track's Module 02.

## Operational checklist

Before this kind of API goes to production, verify:

- [ ] All endpoints have schema validation.
- [ ] No `*` CORS in production.
- [ ] Rate limits in place at gateway + app layer.
- [ ] Structured logs include request ID + tenant ID.
- [ ] `/metrics` is reachable from Prometheus.
- [ ] `/healthz` returns quickly (< 100ms); `/readyz`
      validates dependencies.
- [ ] Errors return JSON, not stack traces.
- [ ] No secrets in code; all from env / KMS.
- [ ] Model file integrity verified on load (signature check
      ideal; checksum minimum).
- [ ] Timeouts on every downstream call (no infinite waits).

## Common mistakes

- Loading the model on every request (cold start hell).
- Using `requests` (sync) in async code → blocks the event
  loop.
- No timeouts on downstream calls.
- Catching `Exception` and returning HTTP 500 with the stack
  trace.
- Hardcoding the model path; can't swap without code change.
- Not validating input shape → 500s on malformed requests.

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-007-apis-web-services/exercises/exercise-06-production-ml-api.md`
- Engineer-track production serving:
  `engineer-solutions/mod-101-foundations/exercise-08-production-model-serving`.
- LLM-specific serving:
  `engineer-solutions/mod-110-llm-infrastructure`.
