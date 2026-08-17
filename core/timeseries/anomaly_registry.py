"""Registry for time-series anomaly detectors."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol
from urllib.request import Request, urlopen

from schemas.timeseries import TimeSeriesSeries


@dataclass
class AnomalyDetectorOutput:
    anomaly_points: list[dict]
    scores: list[dict]
    anomaly_spans: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class AnomalyDetector(Protocol):
    name: str

    def detect(self, series: TimeSeriesSeries, *, params: dict) -> AnomalyDetectorOutput:
        ...


_ANOMALY_DETECTORS: dict[str, AnomalyDetector] = {}
_DEFAULT_ANOMALY_DETECTOR = "zscore"


def register_anomaly_detector(detector: AnomalyDetector) -> None:
    name = str(detector.name).strip().lower()
    if not name:
        raise ValueError("Anomaly detector name must not be empty.")
    _ANOMALY_DETECTORS[name] = detector


def register_api_anomaly_detector(
    name: str,
    *,
    endpoint: str,
    timeout_seconds: float = 30.0,
    headers: dict[str, str] | None = None,
) -> None:
    register_anomaly_detector(
        ApiAnomalyDetector(
            name=name,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            headers=headers or {},
        )
    )


def get_anomaly_detector(name: str | None = None) -> AnomalyDetector:
    detector_name = str(name or _DEFAULT_ANOMALY_DETECTOR).strip().lower()
    detector = _ANOMALY_DETECTORS.get(detector_name)
    if detector is None:
        available = ", ".join(available_anomaly_detectors()) or "none"
        raise ValueError(f"Unknown anomaly detector '{detector_name}'. Available anomaly detectors: {available}.")
    return detector


def default_anomaly_detector_name() -> str:
    return _DEFAULT_ANOMALY_DETECTOR


def set_default_anomaly_detector(name: str) -> None:
    """Select the default from the currently registered anomaly detectors."""
    normalized = str(name).strip().lower()
    if normalized not in _ANOMALY_DETECTORS:
        available = ", ".join(available_anomaly_detectors()) or "none"
        raise ValueError(f"Unknown anomaly detector '{normalized}'. Available anomaly detectors: {available}.")
    global _DEFAULT_ANOMALY_DETECTOR
    _DEFAULT_ANOMALY_DETECTOR = normalized


def available_anomaly_detectors() -> list[str]:
    return sorted(_ANOMALY_DETECTORS)


def unregister_anomaly_detector(name: str) -> None:
    normalized = str(name).strip().lower()
    if normalized == "zscore":
        raise ValueError("The built-in zscore detector cannot be unregistered.")
    _ANOMALY_DETECTORS.pop(normalized, None)


class ZScoreAnomalyDetector:
    name = "zscore"

    def detect(self, series: TimeSeriesSeries, *, params: dict) -> AnomalyDetectorOutput:
        from core.timeseries.anomaly_adapter import detect_zscore_anomalies

        threshold = float(params.get("zscore_threshold", params.get("threshold", 2.5)))
        anomaly_points, scores = detect_zscore_anomalies(series, threshold=threshold)
        return AnomalyDetectorOutput(
            anomaly_points=anomaly_points,
            anomaly_spans=[],
            scores=scores,
            diagnostics={"threshold": threshold, "model_family": "zscore"},
        )


@dataclass
class ApiAnomalyDetector:
    name: str
    endpoint: str
    timeout_seconds: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)

    def detect(self, series: TimeSeriesSeries, *, params: dict) -> AnomalyDetectorOutput:
        payload = {
            "task": "anomaly",
            "detector_name": self.name,
            "series": series.model_dump(mode="json"),
            "params": params,
        }
        response = _post_json(self.endpoint, payload, timeout_seconds=self.timeout_seconds, headers=self.headers)
        diagnostics = dict(response.get("diagnostics") or {})
        diagnostics["model_family"] = "api"
        diagnostics["endpoint"] = self.endpoint
        return AnomalyDetectorOutput(
            anomaly_points=list(response.get("anomaly_points") or []),
            anomaly_spans=list(response.get("anomaly_spans") or []),
            scores=list(response.get("scores") or []),
            diagnostics=diagnostics,
        )


def _post_json(endpoint: str, payload: dict[str, Any], *, timeout_seconds: float, headers: dict[str, str]) -> dict:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError(f"Anomaly API detector endpoint '{endpoint}' must return a JSON object.")
    return decoded


register_anomaly_detector(ZScoreAnomalyDetector())
