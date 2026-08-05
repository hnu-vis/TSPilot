"""VictoriaMetrics connector implementation.

VictoriaMetrics exposes a Prometheus-compatible HTTP API (``/api/v1/query``,
``/api/v1/query_range``, ``/api/v1/label/__name__/values`` …) and answers
PromQL/MetricsQL, so the connector is a thin subclass of the Prometheus
connector that only changes the default port and the reported type.
"""

from __future__ import annotations

from ..connector import DatabaseType
from .prometheus import PrometheusConfig, PrometheusConnector


class VictoriaMetricsConfig(PrometheusConfig):
    """VictoriaMetrics HTTP configuration (single-node default port 8428)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8428,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 8428,
            database=database,
            username=username,
            password=password,
            timeout=timeout,
            **kwargs,
        )


class VictoriaMetricsConnector(PrometheusConnector):
    """Connector for VictoriaMetrics over its Prometheus-compatible HTTP API."""

    @property
    def dialect(self) -> str:
        return "victoriametrics"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.VICTORIAMETRICS
