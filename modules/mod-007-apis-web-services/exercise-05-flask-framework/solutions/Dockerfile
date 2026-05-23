# Multi-stage Dockerfile for the Flask model API.
# Build stage installs deps into a wheelhouse, runtime stage copies only what's needed.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_LEVEL=INFO \
    MODEL_PATH=/models/current/model.joblib \
    FEATURE_COUNT=10

RUN groupadd -r app && useradd -r -g app -u 1000 app

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels

COPY app ./app
COPY wsgi.py .

USER 1000

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]
