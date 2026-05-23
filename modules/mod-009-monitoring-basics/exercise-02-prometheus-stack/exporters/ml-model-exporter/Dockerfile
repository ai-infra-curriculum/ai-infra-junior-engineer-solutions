FROM python:3.11-slim

LABEL maintainer="ml-platform-team@company.com" \
      description="Custom Prometheus exporter for ML model metrics"

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy exporter code
COPY exporter.py .

# Create non-root user
RUN useradd -m -u 1000 exporter && \
    chown -R exporter:exporter /app

# Switch to non-root user
USER exporter

# Expose metrics port
EXPOSE 9101

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:9101/').raise_for_status()"

# Run exporter
CMD ["python", "-u", "exporter.py"]
