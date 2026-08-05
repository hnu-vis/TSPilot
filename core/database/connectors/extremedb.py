"""eXtremeDB connector (SQL over ODBC).

McObject eXtremeDB in its client/server SQL edition is reachable over ODBC. Thin
subclass of the generic ODBC connector; override the ODBC driver name via the
``driver`` config key if needed.
"""

from __future__ import annotations

from ..connector import DatabaseType
from .odbc_base import ODBCConfig, ODBCConnector


class ExtremeDBConfig(ODBCConfig):
    def __init__(self, host: str = "localhost", port: int = 5001, **kwargs):
        super().__init__(host=host, port=port or 5001, **kwargs)


class ExtremeDBConnector(ODBCConnector):
    ODBC_DRIVER = "eXtremeDB SQL"

    @property
    def dialect(self) -> str:
        return "extremedb"

    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.EXTREMEDB
