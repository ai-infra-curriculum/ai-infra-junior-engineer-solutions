# Step-by-Step Implementation Guide: Logging Pipeline

## Overview

Build centralized logging infrastructure! Learn Fluentd, Elasticsearch, Kibana, log aggregation, parsing, and log-based alerting.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Deploy EFK/ELK stack
✅ Aggregate logs with Fluentd
✅ Parse and enrich logs
✅ Search logs in Elasticsearch
✅ Visualize logs in Kibana
✅ Implement log retention
✅ Create log-based alerts

---

## Deploy EFK Stack

```bash
# Install Elasticsearch
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch \
  --namespace logging \
  --create-namespace \
  --set replicas=3 \
  --set volumeClaimTemplate.resources.requests.storage=30Gi

# Install Kibana
helm install kibana elastic/kibana \
  --namespace logging \
  --set elasticsearchHosts="http://elasticsearch-master:9200"

# Install Fluentd
helm install fluentd fluent/fluentd \
  --namespace logging \
  --set elasticsearch.host=elasticsearch-master
```

---

## Fluentd Configuration

```yaml
# fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: logging
data:
  fluent.conf: |
    # Input: Kubernetes logs
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>

    # Parse and enrich
    <filter kubernetes.**>
      @type kubernetes_metadata
      @id filter_kube_metadata
    </filter>

    # Parse JSON logs
    <filter kubernetes.**>
      @type parser
      key_name log
      reserve_data true
      <parse>
        @type json
      </parse>
    </filter>

    # Add ML-specific fields
    <filter kubernetes.var.log.containers.ml-api**>
      @type record_transformer
      <record>
        service "ml-inference"
        environment "production"
      </record>
    </filter>

    # Output to Elasticsearch
    <match **>
      @type elasticsearch
      host elasticsearch-master
      port 9200
      logstash_format true
      logstash_prefix fluentd
      include_tag_key true
      type_name _doc
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes.system.buffer
        flush_mode interval
        retry_type exponential_backoff
        flush_thread_count 2
        flush_interval 5s
        retry_forever
        retry_max_interval 30
        chunk_limit_size 2M
        queue_limit_length 8
        overflow_action block
      </buffer>
    </match>
```

---

## Structured Logging in App

```python
import structlog
import sys

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

# Log with context
log.info(
    "prediction_complete",
    model_version="v2.0",
    latency_ms=123,
    input_shape=[32, 224, 224],
    prediction_class="cat",
    confidence=0.95
)

log.error(
    "model_load_failed",
    model_path="/models/resnet50.pth",
    error_type="FileNotFoundError",
    exc_info=True
)
```

---

## Kibana Queries

```
# Search for errors
level:ERROR

# ML predictions
service:ml-inference AND event:prediction_complete

# High latency
latency_ms:>1000

# Time range
@timestamp:[now-1h TO now]

# Aggregation query
{
  "query": {"match": {"service": "ml-inference"}},
  "aggs": {
    "avg_latency": {"avg": {"field": "latency_ms"}},
    "error_rate": {
      "filters": {
        "filters": {
          "errors": {"match": {"level": "ERROR"}}
        }
      }
    }
  }
}
```

---

## Log Retention Policy

```bash
# Elasticsearch ILM policy
curl -X PUT "localhost:9200/_ilm/policy/logs-policy" -H 'Content-Type: application/json' -d'
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": {
            "max_size": "50GB",
            "max_age": "1d"
          }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "shrink": {"number_of_shards": 1}
        }
      },
      "delete": {
        "min_age": "30d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}'
```

---

## Best Practices

✅ Use structured logging (JSON)
✅ Include context in logs
✅ Implement log sampling for high-volume
✅ Set retention policies
✅ Use log levels appropriately
✅ Add correlation IDs
✅ Redact sensitive data
✅ Monitor logging pipeline health

---

**Logging Pipeline mastered!** 📝
