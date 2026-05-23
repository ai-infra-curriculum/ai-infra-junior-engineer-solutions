# Exercise 05: Flask Framework — Solution

Reference solution for the [Flask Framework lecture](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning/blob/main/lessons/mod-007-apis-web-services/lecture-notes/05-flask-framework.md) in the learning repo.

A production-shaped Flask ML serving service implementing the same `/predict` contract as the FastAPI exercise (01), so a learner can compare the two side-by-side.

## What this demonstrates

- Application factory pattern + Blueprints
- Marshmallow-based input validation
- Flask-Smorest for automatic OpenAPI / Swagger UI (closest analog to FastAPI's auto-docs)
- Prometheus instrumentation via `prometheus-flask-exporter`
- Health (`/health`) and readiness (`/ready`) probes
- Configuration from environment variables
- Gunicorn production server in the Dockerfile
- Structured JSON logging
- pytest-based unit + integration tests using `app.test_client()`

## Files

```
solutions/
├── app/
│   ├── __init__.py         # create_app() factory
│   ├── config.py
│   ├── ml/
│   │   ├── __init__.py
│   │   └── loader.py       # thread-safe model loader
│   └── routes/
│       ├── __init__.py
│       ├── health.py
│       └── predict.py      # /predict endpoint, Flask-Smorest blueprint
├── requirements.txt
├── wsgi.py                 # Gunicorn entry point
├── Dockerfile
└── docker-compose.yml
tests/
└── test_app.py             # pytest, app.test_client() based
```

## Running

```bash
pip install -r solutions/requirements.txt

export MODEL_PATH=$(pwd)/tests/fixtures/model.joblib
export FEATURE_COUNT=4

# Development
flask --app solutions.wsgi run --debug --port 8000

# Production (in Docker)
docker compose -f solutions/docker-compose.yml up --build
```

Visit `http://localhost:8000/docs` for Swagger UI.

## Comparing with the FastAPI exercise (01)

Run both side-by-side at different ports and use the same Locust load test. Expected results:

- For a CPU-bound sklearn model behind 4 Gunicorn workers, throughput is comparable to FastAPI with 4 Uvicorn workers.
- p95 latency is within 10% between the two.
- The framework overhead is negligible vs. model inference time — pick the framework your team will operate best.
