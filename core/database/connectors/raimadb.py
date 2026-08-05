"""RaimaDB connector (SQL over ODBC).

RaimaDB (RDM) in its server/SQL edition is reachable over ODBC. Thin subclass of
the generic ODBC connector; set the RaimaDB ODBC driver name via the ``driver``
config key if it differs from the default.
"""

from __future__ import annotations

from ..connector import DatabaseType
from .odbc_base import ODBCConfig, ODBCConnector


class RaimaDBConfig(ODBCConfig):
    def __init__(self, host: str = "localhost", port: int = 21553, **kwargs):
        super().__init__(host=host, port=port or 21553, **kwargs)


class RaimaDBConnector(ODBCConnector):
    ODBC_DRIVER = "RaimaDB"

    @property
    def dialect(self) -> str:
        return "raimadb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.RAIMADB
