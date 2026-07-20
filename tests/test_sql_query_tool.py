from __future__ import annotations

import pytest

from app.settings import get_settings
from core.database.connector import ColumnSchema, DatabaseSchema, QueryResult, TableSchema
from core.report.composer import missing_requirements
from runtime.request_state import apply_observation
from schemas.database_context import DatabaseContext
from schemas.database import DatabaseEvidence
from schemas.state import RequestStateModel
from schemas.tool import ToolObservation
from tools.sql_query import SqlQueryInput, SqlQueryTool
from tools.query_database import QueryDatabaseTool
from tools.registry import ToolSpec


class _FakeConnector:
    def __init__(self):
        self.last_query: str | None = None
        self.executed_queries: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def execute(self, query: str, params=None, timeout=None):
        self.last_query = query
        self.executed_queries.append(query)
        return QueryResult(
            columns=["grp", "avg_value"],
            rows=[{"grp": "hourly", "avg_value": 12.5}],
            row_count=1,
            execution_time_ms=3,
        )

    async def get_schema(self):
        return DatabaseSchema(
            database="demo",
            tables=[
                TableSchema(
                    name="prices",
                    columns=[
                        ColumnSchema(name="timestamp", data_type="datetime"),
                        ColumnSchema(name="value", data_type="float"),
                    ],
                )
            ],
        )


class _FailOnceConnector(_FakeConnector):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    async def execute(self, query: str, params=None, timeout=None):
        self.executed_queries.append(query)
        self.last_query = query
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("syntax error near bad_query")
        return QueryResult(
            columns=["timestamp", "value"],
            rows=[{"timestamp": "2026-01-01T00:00:00Z", "value": 1.0}],
            row_count=1,
            execution_time_ms=3,
        )


class _QueryLLM:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        payload = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return type("_Response", (), {"content": __import__("json").dumps(payload)})()


def test_reference_dataset_filter_handles_naive_rows_and_utc_request_range():
    tool = QueryDatabaseTool(get_settings())
    rows = [
        {"timestamp": "2016-01-11 16:50:00", "value": "1"},
        {"timestamp": "2016-01-11 17:00:00", "value": "2"},
        {"timestamp": "2016-01-11 17:10:00", "value": "3"},
    ]

    filtered = tool._filter_rows(
        rows,
        "timestamp",
        {"start": "2016-01-11T17:00:00Z", "end": "2016-01-11T17:10:00Z"},
    )

    assert [row["value"] for row in filtered] == ["2", "3"]


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
async def test_sql_query_automatic_mode_executes_llm_generated_query(monkeypatch):
    from tools import sql_query as module

    connector = _FakeConnector()

    async def fake_load_databases():
        return None

    async def fake_get_database(database_id: str):
        return {"type": "timescaledb", "database": "demo"}

    async def fake_create_connector(**config):
        return connector

    monkeypatch.setattr(module.DatabaseFactory, "load_databases", fake_load_databases)
    monkeypatch.setattr(module.DatabaseFactory, "get_database", fake_get_database)
    monkeypatch.setattr(module.DatabaseFactory, "create_connector", fake_create_connector)

    llm = _QueryLLM([
        {
            "query": "SELECT timestamp, value FROM prices",
            "query_language": "sql",
            "purpose": "load price series",
            "expected_result_type": "timeseries",
            "selected_fields": ["value"],
            "assumptions": [],
            "confidence": 0.91,
        }
    ])

    result = await SqlQueryTool(get_settings(), llm=llm).execute(
        SqlQueryInput(
            message="分析价格趋势",
            database_context=DatabaseContext(database_id="demo", database_type="timescaledb"),
        )
    )

    assert connector.executed_queries == ["SELECT timestamp, value FROM prices"]
    assert result["metadata"]["sql_query_mode"] == "llm"
    assert result["metadata"]["generation_mode"] == "llm"
    assert result["diagnostics"]["llm_query_generation"]["selected_fields"] == ["value"]


@pytest.mark.asyncio
async def test_sql_query_automatic_mode_rejects_llm_write_query_before_execution(monkeypatch):
    from tools import sql_query as module

    connector = _FakeConnector()

    async def fake_load_databases():
        return None

    async def fake_get_database(database_id: str):
        return {"type": "timescaledb", "database": "demo"}

    async def fake_create_connector(**config):
        return connector

    monkeypatch.setattr(module.DatabaseFactory, "load_databases", fake_load_databases)
    monkeypatch.setattr(module.DatabaseFactory, "get_database", fake_get_database)
    monkeypatch.setattr(module.DatabaseFactory, "create_connector", fake_create_connector)

    llm = _QueryLLM([
        {
            "query": "DELETE FROM prices",
            "query_language": "sql",
            "purpose": "malicious write",
            "expected_result_type": "table",
            "selected_fields": [],
            "assumptions": [],
            "confidence": 0.9,
        }
    ])

    with pytest.raises(ValueError, match="Only read-only|Write or DDL"):
        await SqlQueryTool(get_settings(), llm=llm).execute(
            SqlQueryInput(
                message="删除数据",
                database_context=DatabaseContext(database_id="demo", database_type="timescaledb"),
            )
        )

    assert connector.executed_queries == []


@pytest.mark.asyncio
async def test_sql_query_automatic_mode_repairs_failed_llm_query(monkeypatch):
    from tools import sql_query as module

    connector = _FailOnceConnector()

    async def fake_load_databases():
        return None

    async def fake_get_database(database_id: str):
        return {"type": "timescaledb", "database": "demo"}

    async def fake_create_connector(**config):
        return connector

    monkeypatch.setattr(module.DatabaseFactory, "load_databases", fake_load_databases)
    monkeypatch.setattr(module.DatabaseFactory, "get_database", fake_get_database)
    monkeypatch.setattr(module.DatabaseFactory, "create_connector", fake_create_connector)

    llm = _QueryLLM([
        {
            "query": "SELECT bad_query FROM prices",
            "query_language": "sql",
            "purpose": "load series",
            "expected_result_type": "timeseries",
            "selected_fields": ["value"],
            "assumptions": [],
            "confidence": 0.6,
        },
        {
            "query": "SELECT timestamp, value FROM prices",
            "query_language": "sql",
            "purpose": "repair series query",
            "expected_result_type": "timeseries",
            "selected_fields": ["value"],
            "assumptions": [],
            "confidence": 0.9,
        },
    ])

    result = await SqlQueryTool(get_settings(), llm=llm).execute(
        SqlQueryInput(
            message="分析价格趋势",
            database_context=DatabaseContext(database_id="demo", database_type="timescaledb"),
        )
    )

    assert connector.executed_queries == ["SELECT bad_query FROM prices", "SELECT timestamp, value FROM prices"]
    assert llm.calls == 2
    assert result["diagnostics"]["llm_query_generation"]["repaired_from_query"] == "SELECT bad_query FROM prices"


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
