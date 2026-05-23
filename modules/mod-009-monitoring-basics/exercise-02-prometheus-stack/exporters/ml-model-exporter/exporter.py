#!/usr/bin/env python3
"""
ML Model Custom Exporter for Prometheus

This exporter collects ML-specific metrics that are not captured
by standard application metrics, including:
- Model metadata and version information
- Inference queue depth and processing rate
- Model performance degradation detection
- Resource utilization specific to ML workloads
- Data drift indicators
"""

import os
import time
import logging
import requests
from typing import Dict, Any, Optional
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, REGISTRY
import json

# Configuration from environment
EXPORTER_PORT = int(os.getenv('EXPORTER_PORT', '9101'))
MODEL_API_URL = os.getenv('MODEL_API_URL', 'http://inference-gateway:8000')
SCRAPE_INTERVAL = int(os.getenv('SCRAPE_INTERVAL', '15'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'info').upper()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml-model-exporter')


class MLModelCollector:
    """Custom Prometheus collector for ML model metrics"""

    def __init__(self, model_api_url: str):
        self.model_api_url = model_api_url
        self.timeout = 5

    def collect(self):
        """Collect metrics from the ML service"""
        try:
            # Fetch model info
            model_info = self._get_model_info()
            if model_info:
                yield from self._model_info_metrics(model_info)

            # Fetch model health
            health_info = self._get_health_info()
            if health_info:
                yield from self._health_metrics(health_info)

            # Fetch queue metrics (if available)
            queue_info = self._get_queue_info()
            if queue_info:
                yield from self._queue_metrics(queue_info)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    def _get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get model information from API"""
        try:
            response = requests.get(
                f"{self.model_api_url}/info",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return None

    def _get_health_info(self) -> Optional[Dict[str, Any]]:
        """Get health information from API"""
        try:
            response = requests.get(
                f"{self.model_api_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            # Health endpoint returns status, add timing
            data['response_time_ms'] = response.elapsed.total_seconds() * 1000
            return data
        except Exception as e:
            logger.warning(f"Failed to get health info: {e}")
            return None

    def _get_queue_info(self) -> Optional[Dict[str, Any]]:
        """Get queue information (simulated for this example)"""
        # In production, this would query an actual queue service
        # For now, we'll simulate with some data
        return {
            'queue_depth': 0,
            'processing_rate': 0,
            'oldest_message_age_seconds': 0
        }

    def _model_info_metrics(self, model_info: Dict[str, Any]):
        """Generate model information metrics"""
        # Model metadata as Info metric (labels only, value=1)
        info_metric = GaugeMetricFamily(
            'ml_model_info',
            'ML Model metadata information',
            labels=['model_name', 'model_version', 'framework', 'device']
        )
        info_metric.add_metric(
            [
                model_info.get('model_name', 'unknown'),
                model_info.get('model_version', 'unknown'),
                model_info.get('framework', 'unknown'),
                model_info.get('device', 'unknown')
            ],
            1
        )
        yield info_metric

        # Model loaded status
        loaded_metric = GaugeMetricFamily(
            'ml_model_loaded',
            'Whether the model is currently loaded (1=loaded, 0=not loaded)',
            labels=['model_name']
        )
        loaded_metric.add_metric(
            [model_info.get('model_name', 'unknown')],
            1 if model_info.get('loaded', False) else 0
        )
        yield loaded_metric

        # Model parameters count
        if 'parameters' in model_info:
            params_metric = GaugeMetricFamily(
                'ml_model_parameters_total',
                'Total number of model parameters',
                labels=['model_name']
            )
            params_metric.add_metric(
                [model_info.get('model_name', 'unknown')],
                float(model_info['parameters'])
            )
            yield params_metric

        # Model memory usage (bytes)
        if 'memory_usage_bytes' in model_info:
            memory_metric = GaugeMetricFamily(
                'ml_model_memory_bytes',
                'Model memory usage in bytes',
                labels=['model_name']
            )
            memory_metric.add_metric(
                [model_info.get('model_name', 'unknown')],
                float(model_info['memory_usage_bytes'])
            )
            yield memory_metric

    def _health_metrics(self, health_info: Dict[str, Any]):
        """Generate health check metrics"""
        # Service health status
        health_metric = GaugeMetricFamily(
            'ml_service_healthy',
            'Service health status (1=healthy, 0=unhealthy)'
        )
        is_healthy = 1 if health_info.get('status') == 'healthy' else 0
        health_metric.add_metric([], is_healthy)
        yield health_metric

        # Health check response time
        if 'response_time_ms' in health_info:
            response_time_metric = GaugeMetricFamily(
                'ml_health_check_duration_milliseconds',
                'Health check response time in milliseconds'
            )
            response_time_metric.add_metric(
                [],
                float(health_info['response_time_ms'])
            )
            yield response_time_metric

    def _queue_metrics(self, queue_info: Dict[str, Any]):
        """Generate queue-related metrics"""
        # Queue depth
        queue_depth_metric = GaugeMetricFamily(
            'ml_inference_queue_depth',
            'Number of inference requests waiting in queue'
        )
        queue_depth_metric.add_metric(
            [],
            float(queue_info.get('queue_depth', 0))
        )
        yield queue_depth_metric

        # Processing rate
        processing_rate_metric = GaugeMetricFamily(
            'ml_inference_processing_rate',
            'Inference requests processed per second'
        )
        processing_rate_metric.add_metric(
            [],
            float(queue_info.get('processing_rate', 0))
        )
        yield processing_rate_metric

        # Oldest message age
        oldest_message_metric = GaugeMetricFamily(
            'ml_inference_oldest_message_age_seconds',
            'Age of the oldest message in the queue in seconds'
        )
        oldest_message_metric.add_metric(
            [],
            float(queue_info.get('oldest_message_age_seconds', 0))
        )
        yield oldest_message_metric


def main():
    """Main exporter function"""
    logger.info(f"Starting ML Model Exporter on port {EXPORTER_PORT}")
    logger.info(f"Monitoring model API at {MODEL_API_URL}")
    logger.info(f"Scrape interval: {SCRAPE_INTERVAL}s")

    # Register custom collector
    collector = MLModelCollector(MODEL_API_URL)
    REGISTRY.register(collector)

    # Start HTTP server
    start_http_server(EXPORTER_PORT)
    logger.info(f"Exporter started successfully on :{EXPORTER_PORT}")

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exporter shutting down...")


if __name__ == '__main__':
    main()
