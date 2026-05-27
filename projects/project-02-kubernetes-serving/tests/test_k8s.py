"""Integration tests for the Kubernetes deployment of the model API.

These tests target a *live* cluster — they read Deployments, Services,
HPA, ConfigMaps, and the inference Service over HTTP. When no cluster is
reachable (or the deployment hasn't been applied) every test in this
module is auto-skipped, so the file can run in CI without producing
false failures.

Run from a workstation with ``kubectl`` configured:

    pytest tests/test_k8s.py          # quick checks only
    pytest tests/test_k8s.py -m slow  # load + rolling-update tests too
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import requests

try:
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    KUBERNETES_IMPORT_OK = True
except ImportError:  # pragma: no cover - module is optional
    client = None  # type: ignore[assignment]
    config = None  # type: ignore[assignment]
    ApiException = Exception  # type: ignore[assignment]
    KUBERNETES_IMPORT_OK = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NAMESPACE = "ml-serving"
DEPLOYMENT_NAME = "model-api"
SERVICE_NAME = "model-api-service"
HPA_NAME = "model-api-hpa"
CONFIGMAP_NAME = "model-api-config"
APP_LABEL_SELECTOR = "app=model-api"
SERVICE_PORT = 80
MIN_REPLICAS_EXPECTED = 3


# ---------------------------------------------------------------------------
# Kubernetes client setup
# ---------------------------------------------------------------------------


def setup_k8s_client() -> Tuple[Any, Any, Any]:
    """Configure the Kubernetes client for in-cluster or local use."""
    if not KUBERNETES_IMPORT_OK:
        raise RuntimeError("kubernetes Python client is not installed")
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    return (
        client.AppsV1Api(),
        client.CoreV1Api(),
        client.AutoscalingV1Api(),
    )


def _k8s_available() -> bool:
    if not KUBERNETES_IMPORT_OK:
        return False
    try:
        apps_v1, _, _ = setup_k8s_client()
        apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        return True
    except Exception as exc:
        logger.debug("Kubernetes deployment unreachable: %s", exc)
        return False


# Skip *every* test in the module when the cluster / deployment isn't
# reachable. Pytest evaluates this once per session.
pytestmark = pytest.mark.skipif(
    not _k8s_available(),
    reason=(
        "Skipping k8s integration tests: cluster unreachable or "
        f"deployment '{DEPLOYMENT_NAME}' missing in namespace '{NAMESPACE}'."
    ),
)


@pytest.fixture(scope="module")
def k8s_clients() -> Tuple[Any, Any, Any]:
    return setup_k8s_client()


@pytest.fixture(scope="module")
def apps_v1(k8s_clients: Tuple[Any, Any, Any]) -> Any:
    return k8s_clients[0]


@pytest.fixture(scope="module")
def core_v1(k8s_clients: Tuple[Any, Any, Any]) -> Any:
    return k8s_clients[1]


@pytest.fixture(scope="module")
def autoscaling_v1(k8s_clients: Tuple[Any, Any, Any]) -> Any:
    return k8s_clients[2]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def run_kubectl(command: List[str]) -> Dict[str, Any]:
    """Run ``kubectl <command> -o json`` and return the parsed payload."""
    full_command = ["kubectl", *command, "-o", "json"]
    result = subprocess.run(
        full_command, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kubectl failed: {' '.join(full_command)}\nstderr: {result.stderr}"
        )
    return json.loads(result.stdout)


def wait_for_condition(
    check_func: Callable[[], bool],
    timeout: int = 300,
    interval: int = 5,
    condition_name: str = "condition",
) -> bool:
    """Poll ``check_func`` until it returns True or ``timeout`` elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if check_func():
                return True
        except Exception as exc:
            logger.debug("Check '%s' raised %s; retrying", condition_name, exc)
        logger.debug("Waiting for %s ...", condition_name)
        time.sleep(interval)
    return False


def get_service_url(
    core_v1: Any, service_name: str, namespace: str
) -> Optional[str]:
    """Resolve a service to a callable HTTP URL when possible."""
    service = core_v1.read_namespaced_service(service_name, namespace)
    port = service.spec.ports[0].port if service.spec.ports else SERVICE_PORT
    service_type = service.spec.type
    if service_type == "LoadBalancer":
        ingress = service.status.load_balancer.ingress
        if not ingress:
            return None
        host = ingress[0].ip or ingress[0].hostname
        return f"http://{host}:{port}"
    if service_type == "NodePort":
        nodes = core_v1.list_node()
        if not nodes.items:
            return None
        node_addr = next(
            (a.address for a in nodes.items[0].status.addresses if a.type == "InternalIP"),
            None,
        )
        if node_addr is None:
            return None
        node_port = service.spec.ports[0].node_port
        return f"http://{node_addr}:{node_port}"
    # ClusterIP — use the in-cluster DNS name. Tests in the cluster can hit
    # this directly; tests outside it will need a port-forward.
    return f"http://{service_name}.{namespace}.svc.cluster.local:{port}"


def _list_pods(core_v1: Any) -> List[Any]:
    return core_v1.list_namespaced_pod(
        namespace=NAMESPACE, label_selector=APP_LABEL_SELECTOR
    ).items


# ---------------------------------------------------------------------------
# Deployment tests
# ---------------------------------------------------------------------------


class TestDeployment:
    """Static configuration checks for the Deployment object."""

    def test_deployment_exists(self, apps_v1: Any) -> None:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        assert deployment is not None
        assert deployment.metadata.name == DEPLOYMENT_NAME

    def test_deployment_replicas(self, apps_v1: Any) -> None:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        desired = deployment.spec.replicas
        current = deployment.status.replicas or 0
        ready = deployment.status.ready_replicas or 0
        assert desired >= MIN_REPLICAS_EXPECTED
        assert current == desired, f"current={current} desired={desired}"
        assert ready == desired, f"ready={ready} desired={desired}"

    def test_deployment_image(self, apps_v1: Any) -> None:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        container = deployment.spec.template.spec.containers[0]
        assert container.image, "Container image must be set"
        assert ":latest" not in container.image, (
            "Avoid ':latest' tags in production deployments"
        )
        assert "model-api" in container.image

    def test_deployment_resource_limits(self, apps_v1: Any) -> None:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        resources = deployment.spec.template.spec.containers[0].resources
        assert resources.requests, "resource requests must be set"
        assert resources.limits, "resource limits must be set"
        for key in ("cpu", "memory"):
            assert key in resources.requests
            assert key in resources.limits

    def test_deployment_health_probes(self, apps_v1: Any) -> None:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        container = deployment.spec.template.spec.containers[0]
        assert container.liveness_probe is not None, "liveness probe missing"
        assert container.readiness_probe is not None, "readiness probe missing"
        assert container.liveness_probe.http_get.path.startswith("/health")
        assert container.readiness_probe.http_get.path.startswith("/health")
        assert container.readiness_probe.initial_delay_seconds >= 5

    def test_deployment_update_strategy(self, apps_v1: Any) -> None:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        strategy = deployment.spec.strategy
        assert strategy.type == "RollingUpdate"
        rolling = strategy.rolling_update
        assert rolling is not None
        # maxSurge=1, maxUnavailable=0 is the recommended zero-downtime
        # configuration for stateless services.
        assert str(rolling.max_surge) in {"1", "25%"}
        assert str(rolling.max_unavailable) in {"0", "0%"}


# ---------------------------------------------------------------------------
# Pod tests
# ---------------------------------------------------------------------------


class TestPods:
    """Runtime checks against the Pod objects backing the deployment."""

    def test_all_pods_running(self, core_v1: Any, apps_v1: Any) -> None:
        pods = _list_pods(core_v1)
        assert pods, "No pods returned by label selector"
        for pod in pods:
            assert pod.status.phase == "Running", (
                f"pod {pod.metadata.name} is in phase {pod.status.phase}"
            )
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        assert len(pods) >= deployment.spec.replicas

    def test_all_pods_ready(self, core_v1: Any) -> None:
        for pod in _list_pods(core_v1):
            ready_condition = next(
                (c for c in (pod.status.conditions or []) if c.type == "Ready"),
                None,
            )
            assert ready_condition is not None
            assert ready_condition.status == "True"
            for container_status in pod.status.container_statuses or []:
                assert container_status.ready

    def test_no_pod_restarts(self, core_v1: Any) -> None:
        for pod in _list_pods(core_v1):
            for container_status in pod.status.container_statuses or []:
                assert container_status.restart_count < 3, (
                    f"{pod.metadata.name}/{container_status.name} restarted "
                    f"{container_status.restart_count} times"
                )

    def test_pod_resource_usage(self, core_v1: Any) -> None:
        # ``kubectl top pods`` requires the metrics-server. Treat its absence
        # as a soft skip rather than a failure.
        try:
            output = subprocess.run(
                [
                    "kubectl",
                    "top",
                    "pods",
                    "-n",
                    NAMESPACE,
                    "-l",
                    APP_LABEL_SELECTOR,
                    "--no-headers",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("metrics-server unavailable")
        assert output.stdout.strip(), "kubectl top returned no rows"
        for line in output.stdout.strip().splitlines():
            parts = line.split()
            # name cpu memory — sanity-check the row is well-formed.
            assert len(parts) >= 3, f"Unexpected row: {line!r}"


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------


class TestService:
    """Checks for the Service in front of the model API."""

    def test_service_exists(self, core_v1: Any) -> None:
        service = core_v1.read_namespaced_service(SERVICE_NAME, NAMESPACE)
        assert service.metadata.name == SERVICE_NAME

    def test_service_endpoints(self, core_v1: Any) -> None:
        endpoints = core_v1.read_namespaced_endpoints(SERVICE_NAME, NAMESPACE)
        assert endpoints.subsets, "Service has no endpoint subsets"
        addresses = [
            addr for subset in endpoints.subsets for addr in (subset.addresses or [])
        ]
        assert len(addresses) >= MIN_REPLICAS_EXPECTED

    def test_service_health_endpoint(self, core_v1: Any) -> None:
        url = get_service_url(core_v1, SERVICE_NAME, NAMESPACE)
        if url is None or url.endswith(".svc.cluster.local:" + str(SERVICE_PORT)):
            pytest.skip("Service URL is in-cluster only; run with port-forward")
        response = requests.get(f"{url}/health", timeout=10)
        assert response.status_code == 200
        assert response.json().get("status") == "healthy"

    def test_service_metrics_endpoint(self, core_v1: Any) -> None:
        url = get_service_url(core_v1, SERVICE_NAME, NAMESPACE)
        if url is None or url.endswith(".svc.cluster.local:" + str(SERVICE_PORT)):
            pytest.skip("Service URL is in-cluster only; run with port-forward")
        response = requests.get(f"{url}/metrics", timeout=10)
        assert response.status_code == 200
        assert "model_api_requests_total" in response.text

    @pytest.mark.slow
    def test_service_load_balancing(self, core_v1: Any) -> None:
        url = get_service_url(core_v1, SERVICE_NAME, NAMESPACE)
        if url is None or url.endswith(".svc.cluster.local:" + str(SERVICE_PORT)):
            pytest.skip("Service URL is in-cluster only; run with port-forward")
        hostnames: Dict[str, int] = {}
        for _ in range(120):
            try:
                response = requests.get(f"{url}/", timeout=5)
            except requests.RequestException:
                continue
            host_header = response.headers.get("X-Pod-Name")
            if host_header:
                hostnames[host_header] = hostnames.get(host_header, 0) + 1
        if not hostnames:
            pytest.skip("Service did not return pod identifiers; cannot verify spread")
        # Expect every pod to receive at least one request.
        pods = _list_pods(core_v1)
        assert len(hostnames) >= min(len(pods), MIN_REPLICAS_EXPECTED)


# ---------------------------------------------------------------------------
# Auto-scaling tests
# ---------------------------------------------------------------------------


class TestAutoScaling:
    """Horizontal Pod Autoscaler checks."""

    def test_hpa_exists(self, autoscaling_v1: Any) -> None:
        hpa = autoscaling_v1.read_namespaced_horizontal_pod_autoscaler(
            HPA_NAME, NAMESPACE
        )
        assert hpa.spec.scale_target_ref.name == DEPLOYMENT_NAME

    def test_hpa_configuration(self, autoscaling_v1: Any) -> None:
        hpa = autoscaling_v1.read_namespaced_horizontal_pod_autoscaler(
            HPA_NAME, NAMESPACE
        )
        assert hpa.spec.min_replicas == 3
        assert hpa.spec.max_replicas == 10
        # autoscaling/v1 exposes the CPU target on a top-level field.
        assert hpa.spec.target_cpu_utilization_percentage == 70

    def test_hpa_current_metrics(self, autoscaling_v1: Any) -> None:
        hpa = autoscaling_v1.read_namespaced_horizontal_pod_autoscaler(
            HPA_NAME, NAMESPACE
        )
        assert hpa.status.current_replicas is not None
        # current_cpu_utilization_percentage may be None during the first
        # ~30s after HPA creation; treat that as a soft skip.
        cpu = hpa.status.current_cpu_utilization_percentage
        if cpu is None:
            pytest.skip("HPA metrics not yet available; rerun in 30s.")
        assert 0 <= cpu <= 100

    @pytest.mark.slow
    def test_hpa_scale_up(self, autoscaling_v1: Any) -> None:
        initial = autoscaling_v1.read_namespaced_horizontal_pod_autoscaler(
            HPA_NAME, NAMESPACE
        ).status.current_replicas
        # Generate load with a busybox pod.
        subprocess.run(
            [
                "kubectl",
                "run",
                "load-gen",
                "-n",
                NAMESPACE,
                "--image=busybox",
                "--restart=Never",
                "--rm",
                "--",
                "/bin/sh",
                "-c",
                f"while true; do wget -qO- http://{SERVICE_NAME}/predict "
                "--post-data='{\"instances\":[[1,2,3]]}' "
                "--header='Content-Type: application/json'; done",
            ],
            check=False,
        )

        def _scaled_up() -> bool:
            current = autoscaling_v1.read_namespaced_horizontal_pod_autoscaler(
                HPA_NAME, NAMESPACE
            ).status.current_replicas or 0
            return current > initial

        try:
            assert wait_for_condition(
                _scaled_up, timeout=300, interval=10, condition_name="HPA scale-up"
            ), "HPA did not scale up under load"
        finally:
            subprocess.run(
                ["kubectl", "delete", "pod", "load-gen", "-n", NAMESPACE],
                check=False,
            )

    @pytest.mark.slow
    def test_hpa_scale_down(self, autoscaling_v1: Any) -> None:
        # Ensure load generator is gone so the cluster can scale back down.
        subprocess.run(
            ["kubectl", "delete", "pod", "load-gen", "-n", NAMESPACE],
            check=False,
        )

        def _scaled_down() -> bool:
            hpa = autoscaling_v1.read_namespaced_horizontal_pod_autoscaler(
                HPA_NAME, NAMESPACE
            )
            return (hpa.status.current_replicas or 0) <= hpa.spec.min_replicas

        assert wait_for_condition(
            _scaled_down, timeout=900, interval=30, condition_name="HPA scale-down"
        ), "HPA did not scale back down after load stopped"


# ---------------------------------------------------------------------------
# Rolling update tests
# ---------------------------------------------------------------------------


class TestRollingUpdate:
    @pytest.mark.slow
    def test_rolling_update_zero_downtime(self, apps_v1: Any, core_v1: Any) -> None:
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        original_image = deployment.spec.template.spec.containers[0].image
        url = get_service_url(core_v1, SERVICE_NAME, NAMESPACE)
        if url is None or url.endswith(".svc.cluster.local:" + str(SERVICE_PORT)):
            pytest.skip("Service URL is in-cluster only; run with port-forward")

        # Patch the image to a no-op tag change (re-tag) — in a real test
        # this would point at a different SHA. Here we add an annotation
        # bump that forces a rollout without changing the image.
        body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": str(time.time())
                        }
                    }
                }
            }
        }
        apps_v1.patch_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE, body)

        failures = 0
        attempts = 0
        deadline = time.time() + 180
        while time.time() < deadline:
            attempts += 1
            try:
                resp = requests.get(f"{url}/health", timeout=5)
                if resp.status_code >= 500:
                    failures += 1
            except requests.RequestException:
                failures += 1
            time.sleep(1)
        assert failures / max(attempts, 1) < 0.02, (
            f"Detected too many failures during rollout: {failures}/{attempts}"
        )
        # Rollout finishes asynchronously; wait for the deployment to settle.
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "status",
                "deployment",
                DEPLOYMENT_NAME,
                "-n",
                NAMESPACE,
                "--timeout=300s",
            ],
            check=True,
        )
        # Sanity check: image is unchanged.
        deployment = apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
        assert deployment.spec.template.spec.containers[0].image == original_image

    @pytest.mark.slow
    def test_rolling_update_rollback(self) -> None:
        # Trigger a no-op update, then ``kubectl rollout undo``. Both should
        # leave the deployment in a Ready state.
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "restart",
                "deployment",
                DEPLOYMENT_NAME,
                "-n",
                NAMESPACE,
            ],
            check=True,
        )
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "status",
                "deployment",
                DEPLOYMENT_NAME,
                "-n",
                NAMESPACE,
                "--timeout=300s",
            ],
            check=True,
        )
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "undo",
                "deployment",
                DEPLOYMENT_NAME,
                "-n",
                NAMESPACE,
            ],
            check=True,
        )
        subprocess.run(
            [
                "kubectl",
                "rollout",
                "status",
                "deployment",
                DEPLOYMENT_NAME,
                "-n",
                NAMESPACE,
                "--timeout=300s",
            ],
            check=True,
        )


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_configmap_exists(self, core_v1: Any) -> None:
        cm = core_v1.read_namespaced_config_map(CONFIGMAP_NAME, NAMESPACE)
        for key in ("model_name", "log_level", "max_batch_size"):
            assert key in cm.data, f"ConfigMap missing key '{key}'"
            assert cm.data[key], f"ConfigMap key '{key}' is empty"

    def test_pods_use_configmap(self, core_v1: Any) -> None:
        pods = _list_pods(core_v1)
        assert pods, "No pods found to exec against"
        pod_name = pods[0].metadata.name
        result = subprocess.run(
            ["kubectl", "exec", "-n", NAMESPACE, pod_name, "--", "env"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip(f"kubectl exec failed: {result.stderr}")
        env_lines = result.stdout.splitlines()
        env_keys = {line.split("=", 1)[0] for line in env_lines if "=" in line}
        for key in ("MODEL_NAME", "LOG_LEVEL", "MAX_BATCH_SIZE"):
            assert key in env_keys, f"Env var {key} missing from pod"


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


class TestPerformance:
    @pytest.mark.slow
    def test_latency_under_load(self, core_v1: Any) -> None:
        url = get_service_url(core_v1, SERVICE_NAME, NAMESPACE)
        if url is None or url.endswith(".svc.cluster.local:" + str(SERVICE_PORT)):
            pytest.skip("Service URL is in-cluster only; run with port-forward")
        latencies_ms: List[float] = []
        for _ in range(100):
            start = time.time()
            try:
                response = requests.post(
                    f"{url}/predict",
                    json={"instances": [[1.0, 2.0, 3.0]]},
                    timeout=5,
                )
                response.raise_for_status()
                latencies_ms.append((time.time() - start) * 1000)
            except requests.RequestException as exc:
                logger.warning("Request failed: %s", exc)
        assert latencies_ms, "All requests failed"
        latencies_ms.sort()
        p95 = latencies_ms[int(len(latencies_ms) * 0.95) - 1]
        p50 = latencies_ms[len(latencies_ms) // 2]
        assert p95 < 500, f"p95 latency too high: {p95:.1f}ms"
        if p50 > 200:
            logger.warning("p50 latency is %.1fms (above 200ms target)", p50)

    @pytest.mark.slow
    def test_throughput(self, core_v1: Any) -> None:
        # Use ``kubectl run`` + a busybox client to drive sustained load;
        # parsing the result is out-of-scope for unit-test framework. The
        # test simply verifies that a 1000-request burst completes with
        # <1% error rate.
        url = get_service_url(core_v1, SERVICE_NAME, NAMESPACE)
        if url is None or url.endswith(".svc.cluster.local:" + str(SERVICE_PORT)):
            pytest.skip("Service URL is in-cluster only; run with port-forward")
        successes = failures = 0
        for _ in range(1000):
            try:
                response = requests.post(
                    f"{url}/predict",
                    json={"instances": [[1.0, 2.0, 3.0]]},
                    timeout=5,
                )
                if response.status_code == 200:
                    successes += 1
                else:
                    failures += 1
            except requests.RequestException:
                failures += 1
        error_rate = failures / max(successes + failures, 1)
        assert error_rate < 0.01, f"Error rate {error_rate:.2%} exceeds 1%"


# ---------------------------------------------------------------------------
# Monitoring tests
# ---------------------------------------------------------------------------


class TestMonitoring:
    PROMETHEUS_SERVICE = "prometheus-server"
    PROMETHEUS_NS = "monitoring"

    def _prom_query(self, query: str) -> Optional[Dict[str, Any]]:
        # Expect a kubectl port-forward on localhost:9090 in the calling
        # environment. Skip silently if not reachable.
        try:
            response = requests.get(
                "http://localhost:9090/api/v1/query",
                params={"query": query},
                timeout=5,
            )
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        return response.json()

    def test_prometheus_scraping(self) -> None:
        body = self._prom_query('up{job="model-api"}')
        if body is None:
            pytest.skip("Prometheus not reachable on localhost:9090")
        results = body.get("data", {}).get("result", [])
        assert results, "Prometheus reports no model-api targets"
        for series in results:
            assert series["value"][1] == "1", "Target is not up"

    def test_metrics_available(self) -> None:
        for metric in (
            "model_api_requests_total",
            "model_api_request_duration_seconds_count",
            "model_api_predictions_total",
        ):
            body = self._prom_query(metric)
            if body is None:
                pytest.skip("Prometheus not reachable on localhost:9090")
            assert body.get("data", {}).get("result"), (
                f"Metric {metric} is missing from Prometheus"
            )


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    pytest.main([__file__, "-v"])
