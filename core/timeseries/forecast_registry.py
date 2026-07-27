"""Registry for time-series forecast models."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol
from urllib.request import Request, urlopen

from schemas.timeseries import TimeSeriesPoint, TimeSeriesSeries


@dataclass
class ForecastModelOutput:
    forecast_points: list[TimeSeriesPoint]
    confidence_interval: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class ForecastModel(Protocol):
    name: str

    def forecast(self, series: TimeSeriesSeries, *, horizon: int, params: dict) -> ForecastModelOutput:
        ...


_FORECAST_MODELS: dict[str, ForecastModel] = {}
_DEFAULT_FORECAST_MODEL = "linear_regression"


def register_forecast_model(model: ForecastModel) -> None:
    name = str(model.name).strip().lower()
    if not name:
        raise ValueError("Forecast model name must not be empty.")
    _FORECAST_MODELS[name] = model


def register_api_forecast_model(
    name: str,
    *,
    endpoint: str,
    timeout_seconds: float = 30.0,
    headers: dict[str, str] | None = None,
) -> None:
    register_forecast_model(
        ApiForecastModel(
            name=name,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            headers=headers or {},
        )
    )


def get_forecast_model(name: str | None = None) -> ForecastModel:
    model_name = str(name or _DEFAULT_FORECAST_MODEL).strip().lower()
    model = _FORECAST_MODELS.get(model_name)
    if model is None:
        available = ", ".join(available_forecast_models()) or "none"
        raise ValueError(f"Unknown forecast model '{model_name}'. Available forecast models: {available}.")
    return model


def default_forecast_model_name() -> str:
    return _DEFAULT_FORECAST_MODEL


def available_forecast_models() -> list[str]:
    return sorted(_FORECAST_MODELS)


class LinearRegressionForecastModel:
    name = "linear_regression"

    def forecast(self, series: TimeSeriesSeries, *, horizon: int, params: dict) -> ForecastModelOutput:
        from core.timeseries.forecast_adapter import linear_forecast

        return ForecastModelOutput(
            forecast_points=linear_forecast(series, horizon),
            diagnostics={"model_family": "linear_regression"},
        )


@dataclass
class ApiForecastModel:
    name: str
    endpoint: str
    timeout_seconds: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)

    def forecast(self, series: TimeSeriesSeries, *, horizon: int, params: dict) -> ForecastModelOutput:
        payload = {
            "task": "forecast",
            "model_name": self.name,
            "series": series.model_dump(mode="json"),
            "horizon": horizon,
            "params": params,
        }
        response = _post_json(self.endpoint, payload, timeout_seconds=self.timeout_seconds, headers=self.headers)
        points = [
            TimeSeriesPoint(timestamp=str(point["timestamp"]), value=float(point["value"]))
            for point in response.get("forecast_points", [])
            if isinstance(point, dict) and "timestamp" in point and "value" in point
        ]
        if len(points) != horizon:
            raise ValueError(
                f"Forecast API model '{self.name}' returned {len(points)} points; expected {horizon}."
            )
        diagnostics = dict(response.get("diagnostics") or {})
        diagnostics["model_family"] = "api"
        diagnostics["endpoint"] = self.endpoint
        return ForecastModelOutput(
            forecast_points=points,
            confidence_interval=list(response.get("confidence_interval") or []),
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
        raise ValueError(f"Forecast API model endpoint '{endpoint}' must return a JSON object.")
    return decoded


register_forecast_model(LinearRegressionForecastModel())
