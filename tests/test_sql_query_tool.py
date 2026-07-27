from __future__ import annotations

import pytest

from app.settings import get_settings
from core.database.connector import ColumnSchema, DatabaseSchema, QueryResult, TableSchema
from core.database.engine import normalize_query_result
from runtime.request_state import apply_observation, build_request_state
from schemas.api import ChatRequest
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


class _TimestampOnlyConnector(_FakeConnector):
    async def execute(self, query: str, params=None, timeout=None):
        self.last_query = query
        self.executed_queries.append(query)
        return QueryResult(
            columns=["timestamp"],
            rows=[{"timestamp": "2023-01-01T00:00:00Z"}],
            row_count=1,
            execution_time_ms=3,
        )


class _QueryLLM:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.calls = 0
        self.messages = []

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.messages.append(messages)
        payload = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return type("_Response", (), {"content": __import__("json").dumps(payload)})()


class _BitcoinConnector(_FakeConnector):
    async def get_schema(self):
        return DatabaseSchema(
            database="bitcoin",
            tables=[
                TableSchema(
                    name="coindesk",
                    columns=[
                        ColumnSchema(name="_time", data_type="datetime"),
                        ColumnSchema(name="_value", data_type="float"),
                        ColumnSchema(name="code", data_type="string"),
                        ColumnSchema(name="crypto", data_type="string"),
                    ],
                )
            ],
            metadata={"value_domains": {"coindesk": {"code": ["EUR", "GBP", "USD"], "crypto": ["bitcoin"]}}},
        )


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


def test_reference_dataset_filter_does_not_fallback_to_full_dataset_when_range_is_empty():
    tool = QueryDatabaseTool(get_settings())
    rows = [
        {"timestamp": "2016-01-11 17:00:00", "value": "2"},
        {"timestamp": "2016-01-11 17:10:00", "value": "3"},
    ]

    filtered = tool._filter_rows(
        rows,
        "timestamp",
        {"start": "2023-01-01T00:00:00Z", "end": "2023-01-02T00:00:00Z"},
    )

    assert filtered == []


def test_sql_query_runtime_field_check_treats_time_aliases_as_present():
    executor = __import__("tools.sql_query", fromlist=["_ExplicitQueryExecutor"])._ExplicitQueryExecutor(get_settings())

    missing = executor._runtime_missing_items(
        selected_fields=["_time", "price"],
        columns=["timestamp", "price"],
        row_count=1,
        query=None,
        query_language=None,
    )

    assert missing == []


def test_sql_query_runtime_field_check_uses_flux_keep_projection():
    executor = __import__("tools.sql_query", fromlist=["_ExplicitQueryExecutor"])._ExplicitQueryExecutor(get_settings())

    missing = executor._runtime_missing_items(
        selected_fields=[],
        columns=["timestamp", "code"],
        row_count=1,
        query='from(bucket: "demo") |> keep(columns: ["_time", "price", "code"])',
        query_language="flux",
    )

    assert missing == [
        "selected result fields are not present in returned columns: price"
    ]


def test_sql_query_runtime_field_check_applies_flux_rename_aliases():
    executor = __import__("tools.sql_query", fromlist=["_ExplicitQueryExecutor"])._ExplicitQueryExecutor(get_settings())

    missing = executor._runtime_missing_items(
        selected_fields=[],
        columns=["timestamp", "price", "code"],
        row_count=1,
        query=(
            'from(bucket: "demo") '
            '|> keep(columns: ["_time", "_value", "code"]) '
            '|> rename(columns: {_time: "timestamp", _value: "price"})'
        ),
        query_language="flux",
    )

    assert missing == []


def test_sql_query_runtime_field_check_prefers_projected_output_over_selected_fields():
    executor = __import__("tools.sql_query", fromlist=["_ExplicitQueryExecutor"])._ExplicitQueryExecutor(get_settings())

    missing = executor._runtime_missing_items(
        selected_fields=["_time", "price"],
        columns=["earliest_time", "latest_time"],
        row_count=1,
        query='from(bucket: "demo") |> keep(columns: ["_time"]) |> min(column: "_time")',
        query_language="flux",
    )

    assert missing == []


def test_table_evidence_ids_are_unique_per_query():
    first = normalize_query_result(
        database_id="demo",
        database_type="influxdb",
        query_language="flux",
        query='from(bucket: "demo") |> count()',
        result=QueryResult(columns=["count"], rows=[{"count": 2}], row_count=1, execution_time_ms=1),
    )
    second = normalize_query_result(
        database_id="demo",
        database_type="influxdb",
        query_language="flux",
        query='from(bucket: "demo") |> min(column: "_time")',
        result=QueryResult(
            columns=["earliest_time"],
            rows=[{"earliest_time": "2023-01-01T00:00:00Z"}],
            row_count=1,
            execution_time_ms=1,
        ),
    )

    assert first.result_type == "table"
    assert second.result_type == "table"
    assert first.evidence_id != second.evidence_id


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
            "task_coverage": {
                "satisfied": ["selected price timestamps and values"],
                "missing": ["trend conclusion still needs analysis"],
                "next_action_hint": "run code_interpreter over the returned time series",
            },
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
    assert llm.calls == 1
    assert result["metadata"]["sql_query_mode"] == "llm"
    assert result["metadata"]["generation_mode"] == "llm"
    assert result["diagnostics"]["llm_query_generation"]["selected_fields"] == ["value"]
    coverage = result["diagnostics"]["task_coverage"]
    assert coverage["satisfied"] == ["selected price timestamps and values"]
    assert coverage["missing"] == ["trend conclusion still needs analysis"]
    assert coverage["next_action_hint"] == "run code_interpreter over the returned time series"
    assert coverage["requires_followup"] is True
    assert coverage["result_summary"]["row_count"] == 1
    assert result["diagnostics"]["llm_query_generation"]["task_coverage"]["missing"] == [
        "trend conclusion still needs analysis"
    ]
    prompt_payload = __import__("json").loads(llm.messages[0][1][1].split("LLM SQL Query Generation JSON:\n", 1)[1])
    schema_linking = prompt_payload["request"]["schema_preview"]["schema_linking"]
    assert schema_linking["schema_linking"]["sources"][0]["name"] == "prices"
    assert result["diagnostics"]["schema_linking_generation"]["schema_linking"]["sources"][0]["name"] == "prices"


@pytest.mark.asyncio
async def test_sql_query_marks_missing_selected_fields_from_runtime_result(monkeypatch):
    from tools import sql_query as module

    connector = _TimestampOnlyConnector()

    async def fake_load_databases():
        return None

    async def fake_get_database(database_id: str):
        return {"type": "influxdb", "database": "demo"}

    async def fake_create_connector(**config):
        return connector

    monkeypatch.setattr(module.DatabaseFactory, "load_databases", fake_load_databases)
    monkeypatch.setattr(module.DatabaseFactory, "get_database", fake_get_database)
    monkeypatch.setattr(module.DatabaseFactory, "create_connector", fake_create_connector)

    llm = _QueryLLM([
        {
            "query": "from(bucket: \"demo\") |> last() |> keep(columns: [\"_time\", \"price\"])",
            "query_language": "flux",
            "purpose": "load latest price",
            "expected_result_type": "table",
            "selected_fields": ["price"],
            "assumptions": [],
            "task_coverage": {
                "satisfied": ["latest timestamp selected"],
                "missing": [],
                "next_action_hint": None,
            },
            "confidence": 0.9,
        }
    ])

    result = await SqlQueryTool(get_settings(), llm=llm).execute(
        SqlQueryInput(
            message="返回最新价格",
            database_context=DatabaseContext(database_id="demo", database_type="influxdb"),
        )
    )

    coverage = result["diagnostics"]["task_coverage"]

    assert coverage["runtime_requires_followup"] is True
    assert coverage["runtime_missing"] == [
        "selected result fields are not present in returned columns: price"
    ]


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


@pytest.mark.asyncio
async def test_explicit_flux_query_rejects_missing_user_value_domain_filters(monkeypatch):
    from tools import sql_query as module

    connector = _BitcoinConnector()

    async def fake_load_databases():
        return None

    async def fake_get_database(database_id: str):
        return {"type": "influxdb", "database": "bitcoin", "bucket": "bitcoin"}

    async def fake_create_connector(**config):
        return connector

    monkeypatch.setattr(module.DatabaseFactory, "load_databases", fake_load_databases)
    monkeypatch.setattr(module.DatabaseFactory, "get_database", fake_get_database)
    monkeypatch.setattr(module.DatabaseFactory, "create_connector", fake_create_connector)
    request_state = build_request_state(
        ChatRequest(
            message="查询当前数据源中比特币 USD 价格的最晚一条原始记录",
            database_context={"database_id": "bitcoin", "database_type": "influxdb"},
        ),
        get_settings(),
    )

    with pytest.raises(ValueError, match="code='USD'|code=\\'USD\\'|code=.*USD"):
        await SqlQueryTool(get_settings()).execute(
            SqlQueryInput(
                database_context=DatabaseContext(database_id="bitcoin", database_type="influxdb"),
                query=(
                    'from(bucket: "bitcoin")\n'
                    "  |> range(start: 1970-01-01T00:00:00Z)\n"
                    '  |> filter(fn: (r) => r._measurement == "coindesk")\n'
                    '  |> filter(fn: (r) => r._field == "price")\n'
                    '  |> sort(columns: ["_time"], desc: true)\n'
                    "  |> limit(n: 1)"
                ),
                query_language="flux",
                purpose="inspect latest raw rows",
            ),
            request_state=request_state,
        )

    assert connector.executed_queries == []


def test_explicit_sql_query_evidence_is_available_for_gap_assessment():
    request_state = RequestStateModel(
        request_id="req_sql_query_requirement",
        message="按小时聚合并判断周期性。",
        status="running",
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

    assert request_state.latest_database_evidence is not None
    assert request_state.latest_database_evidence.evidence_id == "evi_analysis"
    assert "evi_analysis" in request_state.database_evidence_artifacts
