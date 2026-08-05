"""ITTIA DB connector (SQL over ODBC).

ITTIA DB SQL (server edition) is reachable over ODBC. Thin subclass of the
generic ODBC connector; override the ODBC driver name via the ``driver`` config
key if needed.
"""

from __future__ import annotations

from ..connector import DatabaseType
from .odbc_base import ODBCConfig, ODBCConnector


class ITTIADBConfig(ODBCConfig):
    def __init__(self, host: str = "localhost", port: int = 8710, **kwargs):
        super().__init__(host=host, port=port or 8710, **kwargs)


class ITTIADBConnector(ODBCConnector):
    ODBC_DRIVER = "ITTIA DB"

    @property
    def dialect(self) -> str:
        return "ittiadb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.ITTIADB
