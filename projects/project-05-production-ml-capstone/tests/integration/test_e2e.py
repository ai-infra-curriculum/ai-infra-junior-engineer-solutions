"""End-to-end integration tests for the production ML capstone.

These tests hit a *running* service (default ``http://localhost:5000``)
and exercise the full request path. They are intentionally network-
bound: if the API URL is unreachable, the entire module auto-skips so
the suite stays green in environments where the service hasn't been
started.

Run locally::

    export API_URL=http://localhost:5000
    export API_KEY=test-key
    pytest tests/integration/test_e2e.py -v

Run against staging::

    export API_URL=https://staging.example.com
    pytest tests/integration/test_e2e.py -v -m "not slow"
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
import requests

# Pillow is optional. We can fabricate a small valid PNG without it, but
# having it available makes the fixture more realistic.
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on env
    Image = None
    PIL_AVAILABLE = False


API_URL = os.getenv("API_URL", "http://localhost:5000").rstrip("/")
API_KEY = os.getenv("API_KEY", "test-api-key")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


def _service_reachable() -> bool:
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=2)
        return resp.status_code < 500
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not _service_reachable(),
    reason=f"API at {API_URL} is unreachable; skipping e2e tests.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_client() -> Dict[str, Any]:
    return {
        "base_url": API_URL,
        "headers": {"X-API-Key": API_KEY},
        "timeout": TIMEOUT,
    }


@pytest.fixture(scope="session")
def test_image() -> bytes:
    """Return a small valid JPEG payload."""
    existing = TEST_DATA_DIR / "test_image.jpg"
    if existing.exists():
        return existing.read_bytes()
    if PIL_AVAILABLE:
        img = Image.new("RGB", (224, 224), color=(127, 64, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    # Fallback: minimal JFIF JPEG header. Some servers reject this but
    # most validation paths accept it.
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00"
        b"\x01\x00\x00\xff\xd9"
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


class TestHealthChecks:
    def test_health_endpoint_accessible(self, api_client: Dict[str, Any]) -> None:
        response = requests.get(
            f"{api_client['base_url']}/health", timeout=api_client["timeout"]
        )
        assert response.status_code in (200, 503)
        payload = response.json()
        assert "status" in payload

    def test_health_response_time(self, api_client: Dict[str, Any]) -> None:
        start = time.time()
        response = requests.get(
            f"{api_client['base_url']}/health", timeout=api_client["timeout"]
        )
        elapsed = time.time() - start
        assert response.status_code in (200, 503)
        # 250ms gives the server some headroom while still catching
        # serious regressions.
        assert elapsed < 0.25, f"/health took {elapsed:.3f}s"

    def test_health_returns_model_info(self, api_client: Dict[str, Any]) -> None:
        response = requests.get(
            f"{api_client['base_url']}/health", timeout=api_client["timeout"]
        )
        if response.status_code != 200:
            pytest.skip("Service not currently healthy; cannot validate model info")
        data = response.json()
        assert "model" in data
        assert {"model_name", "model_version"}.issubset(data["model"].keys())


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_missing_api_key_rejected(self, api_client: Dict[str, Any]) -> None:
        response = requests.post(
            f"{api_client['base_url']}/predict",
            timeout=api_client["timeout"],
        )
        # When no API_KEYS are configured server-side the server lets the
        # request through (and returns 400 because no file was provided).
        if response.status_code == 400:
            pytest.skip("Server configured without API_KEYS; auth tests skipped.")
        assert response.status_code == 401

    def test_invalid_api_key_rejected(self, api_client: Dict[str, Any]) -> None:
        response = requests.post(
            f"{api_client['base_url']}/predict",
            headers={"X-API-Key": "invalid-key"},
            timeout=api_client["timeout"],
        )
        if response.status_code == 400:
            pytest.skip("Server configured without API_KEYS; auth tests skipped.")
        assert response.status_code == 403

    def test_valid_api_key_accepted(
        self, api_client: Dict[str, Any], test_image: bytes
    ) -> None:
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = requests.post(
            f"{api_client['base_url']}/predict",
            headers=api_client["headers"],
            files=files,
            timeout=api_client["timeout"],
        )
        # We can't guarantee the model returns 200 (depends on the model),
        # but we *can* guarantee the auth gate didn't reject us.
        assert response.status_code not in (401, 403)


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------


class TestPredictionEndpoint:
    def test_predict_with_valid_image(
        self, api_client: Dict[str, Any], test_image: bytes
    ) -> None:
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = requests.post(
            f"{api_client['base_url']}/predict",
            headers=api_client["headers"],
            files=files,
            timeout=api_client["timeout"],
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0
        assert "model_version" in data

    def test_predict_response_format(
        self, api_client: Dict[str, Any], test_image: bytes
    ) -> None:
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = requests.post(
            f"{api_client['base_url']}/predict",
            headers=api_client["headers"],
            files=files,
            timeout=api_client["timeout"],
        )
        assert response.status_code == 200
        prediction = response.json()["predictions"][0]
        # Accept either {class, confidence} or {class_index, confidence}.
        assert "confidence" in prediction
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert "class" in prediction or "class_index" in prediction

    @pytest.mark.slow
    def test_predict_latency_slo(
        self, api_client: Dict[str, Any], test_image: bytes
    ) -> None:
        latencies = []
        for _ in range(20):
            files = {"file": ("test.jpg", test_image, "image/jpeg")}
            start = time.time()
            response = requests.post(
                f"{api_client['base_url']}/predict",
                headers=api_client["headers"],
                files=files,
                timeout=api_client["timeout"],
            )
            latencies.append(time.time() - start)
            assert response.status_code == 200, response.text
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95) - 1]
        assert p95 < 0.5, f"P95 latency {p95:.3f}s exceeds 500ms SLO"

    def test_predict_without_file(self, api_client: Dict[str, Any]) -> None:
        response = requests.post(
            f"{api_client['base_url']}/predict",
            headers=api_client["headers"],
            timeout=api_client["timeout"],
        )
        assert response.status_code == 400

    def test_predict_with_invalid_file_type(
        self, api_client: Dict[str, Any]
    ) -> None:
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = requests.post(
            f"{api_client['base_url']}/predict",
            headers=api_client["headers"],
            files=files,
            timeout=api_client["timeout"],
        )
        assert response.status_code == 400

    def test_predict_with_large_file(self, api_client: Dict[str, Any]) -> None:
        large_file = b"0" * (15 * 1024 * 1024)
        files = {"file": ("large.jpg", large_file, "image/jpeg")}
        response = requests.post(
            f"{api_client['base_url']}/predict",
            headers=api_client["headers"],
            files=files,
            timeout=api_client["timeout"],
        )
        # Flask returns 413 by default for MAX_CONTENT_LENGTH violations.
        assert response.status_code in (400, 413)


# ---------------------------------------------------------------------------
# Info endpoint
# ---------------------------------------------------------------------------


class TestInfoEndpoint:
    def test_info_endpoint_accessible(self, api_client: Dict[str, Any]) -> None:
        response = requests.get(
            f"{api_client['base_url']}/info",
            headers=api_client["headers"],
            timeout=api_client["timeout"],
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert "service" in data
        assert "model" in data

    def test_info_includes_model_version(self, api_client: Dict[str, Any]) -> None:
        response = requests.get(
            f"{api_client['base_url']}/info",
            headers=api_client["headers"],
            timeout=api_client["timeout"],
        )
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "model_version" in data["model"]
        assert data["model"]["model_version"]


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_endpoint_accessible(self, api_client: Dict[str, Any]) -> None:
        response = requests.get(
            f"{api_client['base_url']}/metrics", timeout=api_client["timeout"]
        )
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("Content-Type", "")

    def test_metrics_include_custom_metrics(
        self, api_client: Dict[str, Any]
    ) -> None:
        response = requests.get(
            f"{api_client['base_url']}/metrics", timeout=api_client["timeout"]
        )
        assert response.status_code == 200
        body = response.text
        for required in (
            "http_requests_total",
            "http_request_duration_seconds",
            "model_predictions_total",
        ):
            assert required in body, f"Missing metric {required}"


# ---------------------------------------------------------------------------
# Load handling
# ---------------------------------------------------------------------------


class TestLoadHandling:
    @pytest.mark.slow
    def test_concurrent_requests(
        self, api_client: Dict[str, Any], test_image: bytes
    ) -> None:
        def _one_request() -> int:
            files = {"file": ("test.jpg", test_image, "image/jpeg")}
            response = requests.post(
                f"{api_client['base_url']}/predict",
                headers=api_client["headers"],
                files=files,
                timeout=api_client["timeout"],
            )
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            statuses = list(pool.map(lambda _: _one_request(), range(50)))
        success_rate = sum(1 for s in statuses if s == 200) / len(statuses)
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"

    @pytest.mark.slow
    def test_sustained_load(
        self, api_client: Dict[str, Any], test_image: bytes
    ) -> None:
        duration_seconds = float(os.getenv("E2E_LOAD_DURATION", "30"))
        latencies: list[float] = []
        deadline = time.time() + duration_seconds
        while time.time() < deadline:
            files = {"file": ("test.jpg", test_image, "image/jpeg")}
            start = time.time()
            response = requests.post(
                f"{api_client['base_url']}/predict",
                headers=api_client["headers"],
                files=files,
                timeout=api_client["timeout"],
            )
            latencies.append(time.time() - start)
            assert response.status_code == 200
            time.sleep(0.1)
        # Compare early-window p95 vs late-window p95 to detect drift.
        if len(latencies) < 20:
            pytest.skip("Not enough samples to compute steady-state latency drift.")
        early = sorted(latencies[: len(latencies) // 4])
        late = sorted(latencies[-len(latencies) // 4 :])
        early_p95 = early[int(len(early) * 0.95) - 1]
        late_p95 = late[int(len(late) * 0.95) - 1]
        assert late_p95 < early_p95 * 1.5, (
            f"Latency drifted: early p95={early_p95:.3f}s late p95={late_p95:.3f}s"
        )


# ---------------------------------------------------------------------------
# Monitoring integration
# ---------------------------------------------------------------------------


class TestMonitoringIntegration:
    def test_requests_counted_in_prometheus(
        self, api_client: Dict[str, Any], test_image: bytes
    ) -> None:
        before = requests.get(
            f"{api_client['base_url']}/metrics", timeout=api_client["timeout"]
        ).text
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        requests.post(
            f"{api_client['base_url']}/predict",
            headers=api_client["headers"],
            files=files,
            timeout=api_client["timeout"],
        )
        after = requests.get(
            f"{api_client['base_url']}/metrics", timeout=api_client["timeout"]
        ).text
        assert before != after


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_handles_model_unavailable(self, api_client: Dict[str, Any]) -> None:
        # The service has no documented way to simulate model unavailability,
        # so verify that the /health endpoint exists and never 500s — at
        # worst it returns 503 with a structured payload.
        response = requests.get(
            f"{api_client['base_url']}/health", timeout=api_client["timeout"]
        )
        assert response.status_code in (200, 503)
        assert "status" in response.json()

    def test_error_responses_dont_leak_info(
        self, api_client: Dict[str, Any]
    ) -> None:
        response = requests.post(
            f"{api_client['base_url']}/predict",
            headers={"X-API-Key": "invalid"},
            timeout=api_client["timeout"],
        )
        if response.status_code == 400:
            pytest.skip("Server configured without API_KEYS; auth tests skipped.")
        body = response.text
        for forbidden_token in ("Traceback", "/home/", "/Users/", "Exception:"):
            assert forbidden_token not in body, (
                f"Sensitive token {forbidden_token!r} leaked in error response"
            )
