from __future__ import annotations

import pytest

from app.settings import get_settings
from core.database.connector import QueryResult
from core.report.composer import missing_requirements
from runtime.request_state import apply_observation
from schemas.database_context import DatabaseContext
from schemas.database import DatabaseEvidence
from schemas.state import RequestStateModel
from schemas.tool import ToolObservation
from tools.sql_query import SqlQueryInput, SqlQueryTool
from tools.registry import ToolSpec


class _FakeConnector:
    last_query: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def execute(self, query: str, params=None, timeout=None):
        self.last_query = query
        return QueryResult(
            columns=["grp", "avg_value"],
            rows=[{"grp": "hourly", "avg_value": 12.5}],
            row_count=1,
            execution_time_ms=3,
        )


@pytest.mark.asyncio
async def test_sql_query_executes_read_only_explicit_query(monkeypatch):
    from tools import sql_query as module

    async def fake_load_databases():
        return None

    async def fake_get_database(database_id: str):
        return {"type": "timescaledb", "database": "demo"}

    async def fake_create_connector(**config):
        return _FakeConnector()

    monkeypatch.setattr(module.DatabaseFactory, "load_databases", fake_load_databases)
    monkeypatch.setattr(module.DatabaseFactory, "get_database", fake_get_database)
    monkeypatch.setattr(module.DatabaseFactory, "create_connector", fake_create_connector)

    result = await SqlQueryTool(get_settings()).execute(
        SqlQueryInput(
            database_context=DatabaseContext(database_id="demo", database_type="timescaledb"),
            query="SELECT 'hourly' AS grp, 12.5 AS avg_value",
            query_language="sql",
            purpose="compute hourly aggregate",
        )
    )

    assert result["result_type"] == "table"
    assert result["data"]["rows"] == [{"grp": "hourly", "avg_value": 12.5}]
    assert result["metadata"]["sql_query_mode"] == "explicit"
    assert result["diagnostics"]["sql_query"]["row_count"] == 1


@pytest.mark.asyncio
async def test_sql_query_adds_flux_date_import(monkeypatch):
    from tools import sql_query as module

    connector = _FakeConnector()

    async def fake_load_databases():
        return None

    async def fake_get_database(database_id: str):
        return {"type": "influxdb", "database": "demo"}

    async def fake_create_connector(**config):
        return connector

    monkeypatch.setattr(module.DatabaseFactory, "load_databases", fake_load_databases)
    monkeypatch.setattr(module.DatabaseFactory, "get_database", fake_get_database)
    monkeypatch.setattr(module.DatabaseFactory, "create_connector", fake_create_connector)

    await SqlQueryTool(get_settings()).execute(
        SqlQueryInput(
            database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
            query='from(bucket: "bitcoin") |> map(fn: (r) => ({ r with hour: date.hour(t: r._time) }))',
            query_language="flux",
        )
    )

    assert connector.last_query is not None
    assert connector.last_query.startswith('import "date"\n')


@pytest.mark.asyncio
async def test_sql_query_unified_tool_executes_explicit_query(monkeypatch):
    from tools import sql_query as module

    async def fake_load_databases():
        return None

    async def fake_get_database(database_id: str):
        return {"type": "timescaledb", "database": "demo"}

    async def fake_create_connector(**config):
        return _FakeConnector()

    monkeypatch.setattr(module.DatabaseFactory, "load_databases", fake_load_databases)
    monkeypatch.setattr(module.DatabaseFactory, "get_database", fake_get_database)
    monkeypatch.setattr(module.DatabaseFactory, "create_connector", fake_create_connector)

    result = await SqlQueryTool(get_settings()).execute(
        SqlQueryInput(
            database_context=DatabaseContext(database_id="demo", database_type="timescaledb"),
            query="SELECT 'hourly' AS grp, 12.5 AS avg_value",
            query_language="sql",
            purpose="compute aggregate",
        )
    )

    assert result["metadata"]["sql_query_mode"] == "explicit"
    assert result["data"]["rows"][0]["grp"] == "hourly"


@pytest.mark.asyncio
async def test_sql_query_rejects_write_sql():
    tool = SqlQueryTool(get_settings())

    with pytest.raises(ValueError, match="Only read-only|Write or DDL"):
        await tool.execute(
            SqlQueryInput(
                database_context=DatabaseContext(database_id="demo", database_type="timescaledb"),
                query="DELETE FROM prices",
                query_language="sql",
            )
        )


def test_explicit_sql_query_evidence_satisfies_answer_requirements():
    request_state = RequestStateModel(
        request_id="req_sql_query_requirement",
        message="按小时聚合并判断周期性。",
        status="running",
        answer_requirements=["seasonality"],
    )
    payload = {
        "evidence_id": "evi_analysis",
        "result_type": "table",
        "database": "demo",
        "query_language": "sql",
        "query": "SELECT hour, AVG(value) FROM prices GROUP BY hour",
        "summary": "Loaded 24 rows.",
        "data": {"rows": [{"hour": 0, "avg": 1.0}]},
        "columns": ["hour", "avg"],
        "metadata": {"sql_query_mode": "explicit"},
        "diagnostics": {},
    }

    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="Loaded 24 rows.", payload=payload),
        payload,
        ToolSpec(
            tool_name="sql_query",
            description="test",
            input_model=SqlQueryInput,
            output_model=DatabaseEvidence,
            tool=SqlQueryTool(get_settings()),
            prompt_visible=True,
            runtime_access="request_state_read",
            result_target="evidence",
            produces_terminal_payload=False,
            supports_streaming=False,
        ),
    )

    assert missing_requirements(request_state) == []
