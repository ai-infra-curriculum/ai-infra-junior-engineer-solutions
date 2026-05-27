"""Light-weight ``ModelLoader`` used by the project-02 Flask app.

The real classifier lives in project-01 (``model_loader.py``); for the
Kubernetes-serving exercise we only need an object that:

- takes ~1-3 seconds to "load" (so readiness-probe behaviour is observable),
- exposes ``predict(instances)`` returning predictions per instance,
- can be swapped for the project-01 implementation in production.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelLoader:
    """Tiny stand-in for a real torchvision/transformers model."""

    model_name: str = "resnet50"
    version: str = "1.0"
    classes: List[str] = field(
        default_factory=lambda: ["cat", "dog", "bird", "car", "plane"]
    )
    load_seconds: float = 1.5
    _loaded: bool = False

    def load(self) -> "ModelLoader":
        """Simulate the wall-clock cost of loading a real model."""
        logger.info("Loading model '%s' version=%s ...", self.model_name, self.version)
        start = time.time()
        time.sleep(self.load_seconds)
        self._loaded = True
        logger.info(
            "Model '%s' loaded in %.2fs", self.model_name, time.time() - start
        )
        return self

    def is_ready(self) -> bool:
        return self._loaded

    def predict(self, instances: List[Any]) -> List[Dict[str, Any]]:
        """Return a fake-but-stable prediction per input instance."""
        if not self._loaded:
            raise RuntimeError("Model is not loaded; call load() first.")
        rng = random.Random(0)
        predictions: List[Dict[str, Any]] = []
        for instance in instances:
            seed = hash(repr(instance)) & 0xFFFF
            rng.seed(seed)
            cls = rng.choice(self.classes)
            confidence = round(rng.uniform(0.6, 0.99), 4)
            predictions.append({"class": cls, "confidence": confidence})
        return predictions
