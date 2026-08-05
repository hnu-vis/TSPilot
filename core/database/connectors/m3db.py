"""M3DB connector implementation.

M3 (via the m3query / m3coordinator component) exposes a Prometheus-compatible
HTTP API and answers PromQL, so the connector is a thin subclass of the
Prometheus connector that only changes the default port (m3query defaults to
7201) and the reported type.
"""

from __future__ import annotations

from ..connector import DatabaseType
from .prometheus import PrometheusConfig, PrometheusConnector


class M3DBConfig(PrometheusConfig):
    """M3 query HTTP configuration (m3query default port 7201)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7201,
        database: str = "",
        username: str = "",
        password: str = "",
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(
            host=host,
            port=port or 7201,
            database=database,
            username=username,
            password=password,
            timeout=timeout,
            **kwargs,
        )


class M3DBConnector(PrometheusConnector):
    """Connector for M3DB over the m3query Prometheus-compatible HTTP API."""

    @property
    def dialect(self) -> str:
        return "m3db"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.M3DB
