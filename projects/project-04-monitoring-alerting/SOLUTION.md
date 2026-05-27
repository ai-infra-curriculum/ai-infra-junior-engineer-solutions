# SOLUTION — Monitoring & Alerting System

> Read this *after* you have wired the stack yourself. This file
> explains *why* the components are shaped the way they are, what
> we deliberately rejected, and which production failures each
> piece prevents. `SOLUTION_GUIDE.md` is the how-to-run-it.

## What this project is really teaching

A monitoring stack is judged by exactly two things:

1. **How fast can you tell that something is wrong?** (MTTD —
   mean time to detect.)
2. **When something is wrong, can you figure out what's broken
   without paging four people?** (MTTR — mean time to resolve.)

Tools are not what determines those numbers. What determines them
is:

- The **golden signals** are instrumented at the right layers.
- Alerts fire on **symptoms users feel** (latency, errors), not on
  **causes you suspect** (CPU > 80%).
- Each alert has a runbook.
- Logs are **queryable** (structured + indexed), not just
  collected.
- Dashboards are **read in 5 seconds**, not 5 minutes.

Every decision below ties back to one of those.

## Architectural decisions and *why*

### Decision 1: Prometheus pull-based scrape, not OTLP push

Prometheus comes to your service every 15 seconds and scrapes
`/metrics`. The OTLP push model is the more modern protocol; we
picked Prometheus pull because:

- **Service discovery is free** in Kubernetes via the
  `kubernetes_sd_config`. New pods are scraped automatically; you
  don't manage push endpoints.
- **Targets that aren't responding are immediately visible** in
  the `up{}` metric — push-based systems just silently stop
  producing data.
- It's the dominant tool by deployment count; the operational
  muscle memory transfers across jobs.

The trade-off is firewall-traversal complexity in heterogeneous
environments. For an in-cluster ML platform that's not a real
problem.

### Decision 2: Four-signal dashboard, not 200-metric dashboard

The Grafana dashboards visualize Google's four golden signals plus
the saturation-equivalent for each ML-specific resource:

| Signal | Metric |
|---|---|
| Latency | `http_request_duration_seconds` p50/p95/p99 |
| Traffic | `http_requests_total` rate |
| Errors | `http_requests_total{status=~"5.."}` rate |
| Saturation | CPU + memory + GPU utilization |

A 200-metric dashboard looks impressive in screenshots and is
useless during incidents. A four-signal dashboard fits on one
screen and answers "is anything wrong?" in five seconds.

The **detailed** per-component dashboards exist (one per service)
but live one level down — opened only after the four-signal view
points you at a service.

### Decision 3: Alerts on symptoms, not causes

The alert rules fire on:

- p95 latency > target.
- Error rate > target.
- Request rate dropped to zero (something silently broke).
- SLO error budget burn rate above sustainable.

Not on:

- CPU > 80%.
- Memory > 80%.
- Pod restart counter increased.

Why? Because CPU at 80% might mean "everything's fine, we're
busy". Pod restart might mean "Kubernetes recovered, no human
intervention needed". An alert that fires for a cause without a
user-visible symptom is **noise that trains operators to ignore
the page**.

The cause-level metrics are still scraped — they appear on the
diagnostic dashboards once an alert fires. They just don't *cause*
alerts.

### Decision 4: ELK for logs (Filebeat → Logstash → Elasticsearch
→ Kibana), structured JSON in, queryable out

The application emits structured JSON logs (one event per line,
all fields machine-parseable). Filebeat ships them to Logstash
which enriches + parses them; Elasticsearch indexes them; Kibana
queries them.

Two key design choices:

1. **Structured at the source**, not at the parser. Logstash grok
   patterns are a tax you pay forever; JSON-from-the-app is a
   one-time investment.
2. **Index lifecycle policy**: hot → warm → cold tiers with
   age-based transitions and a 30/90/365 day retention budget.
   Without ILM, Elasticsearch grows linearly forever and the
   cluster eventually OOMs.

**Anti-pattern to avoid**: tailing pod stdout with `kubectl logs`
and calling that monitoring. Works for development; vanishes the
moment a pod restarts.

### Decision 5: Alertmanager with **inhibitions** and grouping,
not just routes

Alertmanager isn't just a router. The reference config uses:

- **`group_by: [alertname, service]`** — collapse identical alerts
  from 12 replicas into one notification.
- **`group_wait: 30s`** — give the storm 30s to coalesce before
  paging.
- **`repeat_interval: 4h`** — re-fire unresolved alerts every 4
  hours, not every minute.
- **`inhibit_rules`** — if `ServiceDown` is firing for an upstream
  service, suppress the dependent service's downstream alerts.

A storm of 200 pages at 2 AM is worse than one page at 2 AM. The
inhibition/grouping config is the single highest-leverage piece of
Alertmanager.

### Decision 6: Runbook URL on every alert

Every alert rule includes an `annotations.runbook_url` field
pointing at a markdown runbook in this repo's `runbooks/`
directory. When the page arrives at 3 AM:

- The on-call clicks the link.
- The runbook has three sections: "what this alert means", "what
  to check first", "how to mitigate now".

An alert without a runbook is a wake-up call that delivers no
information. Runbooks let on-call rotate without rebuilding
institutional memory.

### Decision 7: PagerDuty for paging, Slack for low-urgency,
Email never

The Alertmanager routes:

- **Critical** → PagerDuty (paging + escalation).
- **Warning** → Slack `#ml-ops-alerts` channel (visible, not paging).
- **Info** → Slack `#ml-ops-noise` (forwarded only for context).

Email is never a destination. Email-driven alerting is the
canonical anti-pattern: it routes to a shared inbox no one
monitors, nothing escalates, and the alert is invisible until
someone scrolls past it.

## Trade-offs we deliberately accepted

### Prometheus + Grafana + ELK, not Datadog / New Relic

Hosted commercial APM gives 80% of this in a single signup. We
ship the self-hosted stack because:

- It teaches the operational shape.
- The audit + data-residency story is simpler.
- Cost at meaningful scale flips toward self-hosted hard.

That said: a startup with three engineers and no on-call rotation
should use Datadog. The trade-off is real.

### One Prometheus, not Thanos / Cortex

A single Prometheus instance is sufficient for a junior-tier
project. Real production at multi-region scale needs Thanos or
Cortex for HA + long-term storage. That's the engineer-track
`mod-108` exercise's territory.

### Static alert thresholds, not dynamic / ML-based

The alerts use static thresholds (p95 > 500ms, error rate > 1%).
Dynamic thresholds based on baselines (e.g. PagerDuty's adaptive
alerts) work better in some scenarios but require enough data to
build the baseline and are harder to explain to a new on-call.
Static thresholds are the better starting point.

## What the tests cover

`tests/` exercises:

1. **Each alert rule** with a synthetic Prometheus value that
   should trigger / not trigger it (using `promtool test rules`).
2. **The full stack via docker-compose** — bring everything up,
   trigger a synthetic alert, assert the notification fires.
3. **Runbook links resolve** — every alert's runbook_url maps to
   an existing file in `runbooks/`.

These tests catch the most common configuration-drift bugs (a
rule edit that breaks the threshold, a runbook URL pointing at a
file that no longer exists).

## Common mistakes graders see

1. **Alerting on CPU%** — guarantees noise, teaches on-call to
   ignore pages.
2. **No alert grouping** — one outage becomes 200 pages.
3. **No runbook URL** — pages are useless without context.
4. **Logging via `print()` instead of structured JSON** — the
   logs collect, but Kibana can't filter them.
5. **No index lifecycle policy on Elasticsearch** — the cluster
   eventually OOMs from data growth.
6. **Email as a paging channel** — the page disappears.
7. **Dashboard with 50+ panels** — useless during an incident.
8. **No `up{}` metric on the dashboard** — silent target failure
   is invisible.

## When to go beyond this implementation

Beyond Project 04's scope:

- SLO-based alerting via the multi-window multi-burn-rate pattern.
- OpenTelemetry as a unified metrics+logs+traces source.
- Distributed tracing (Tempo/Jaeger) for cross-service request
  flow.
- Synthetic monitoring (`blackbox_exporter`) to assert from
  outside the cluster.
- Anomaly detection on the metric stream for unknown unknowns.

Each is its own later exercise. Master the four-signal +
symptom-alerting + runbook-linked path first.

## Related curriculum touchpoints

- `junior/project-01,02,03` — the services this stack monitors.
- `engineer/mod-108/ex-01-observability-stack` — same stack, full
  production rigor with SLO tracking.
- `engineer/mod-108/ex-02-ml-model-monitoring` — model-specific
  signals (drift, fairness) layered on top.
- `engineer/mod-104/ex-05-service-mesh-observability` — what
  service-mesh-driven observability looks like at the next tier.
