# Production ML serving Dockerfile reference.
#
# Multi-stage:
#   - builder: full toolchain, builds wheels.
#   - runtime: lean, runs as non-root.

# Pin by digest for reproducibility. Update the digest deliberately.
FROM python:3.11-slim-bookworm@sha256:8a02ad24a7fdc6c4c7b8c87b7d23e76aebaa2e5f3a99b4a93c4d4f5ee5e72a3b AS builder

WORKDIR /build

# Build deps (kept in the builder stage, not in runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Pip cache + requirements first → layer cache survives source changes
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy source last so source changes don't invalidate the pip layer
COPY src ./src


# ---- Runtime stage ----
FROM python:3.11-slim-bookworm@sha256:8a02ad24a7fdc6c4c7b8c87b7d23e76aebaa2e5f3a99b4a93c4d4f5ee5e72a3b

# Runtime-only deps (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd -g $APP_GID app && useradd -u $APP_UID -g app -m -s /usr/sbin/nologin app

WORKDIR /app

# Bring in deps + source from builder
COPY --from=builder --chown=app:app /root/.local /home/app/.local
COPY --from=builder --chown=app:app /build/src ./src

# PATH for user-installed packages
ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root
USER app:app

EXPOSE 8080

# Health check Kubernetes will not use directly; Pod spec defines
# readiness/liveness. This is for `docker run` scenarios.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1

ENTRYPOINT ["python", "-m", "src.server"]
