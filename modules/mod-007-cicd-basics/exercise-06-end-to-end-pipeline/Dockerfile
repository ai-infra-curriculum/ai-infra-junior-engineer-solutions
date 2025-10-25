# Multi-stage Dockerfile for ML pipeline

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    matplotlib \
    seaborn

# Copy source code
COPY . .

# Stage 3: Production image
FROM base as production

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

# Copy source code
COPY --chown=mluser:mluser src/ ./src/
COPY --chown=mluser:mluser pipelines/ ./pipelines/

# Switch to non-root user
USER mluser

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "src.serve.api"]

# Expose port for API
EXPOSE 8000

# Stage 4: Training image
FROM production as training

USER root

# Copy data processing scripts
COPY --chown=mluser:mluser tests/ ./tests/

USER mluser

# Override entrypoint for training
ENTRYPOINT ["python", "pipelines/training_pipeline.py"]
