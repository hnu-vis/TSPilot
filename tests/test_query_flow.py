from __future__ import annotations
import asyncio
from pathlib import Path
import tempfile

from core.database.connector import ColumnSchema, DatabaseSchema, QueryResult, TableSchema
from core.database.contracts import QueryRequestContext
from core.database.engine import infer_evidence_family
from runtime.request_state import enrich_observation_payload, build_request_state, apply_observation
from schemas.api import ChatRequest
from schemas.database_context import DatabaseContext
from schemas.tool import ToolObservation
from core.database.query_flow import (
    CompositeDialectRenderer,
    DatabaseQueryFlow,
    DefaultFieldMapper,
    DefaultIntentInterpreter,
    DefaultLogicalQueryPlanner,
    DefaultQueryValidator,
)
from core.database.schema_linking import SchemaLinkingPipeline
from app.settings import get_settings
from types import SimpleNamespace
from tools.query_database import QueryDatabaseInput, QueryDatabaseTool


def _build_influx_schema() -> DatabaseSchema:
    return DatabaseSchema(
        database="bitcoin",
        metadata={
            "value_domains": {
                "coindesk": {
                    "_field": ["price"],
                    "code": ["EUR", "GBP", "USD"],
                    "crypto": ["bitcoin"],
                }
            }
        },
        tables=[
            TableSchema(
                name="coindesk",
                columns=[
                    ColumnSchema(name="_time", data_type="datetime"),
                    ColumnSchema(name="price", data_type="float"),
                    ColumnSchema(name="code", data_type="string"),
                    ColumnSchema(name="crypto", data_type="string"),
                ],
            )
        ],
    )


def _build_influx_schema_without_domains() -> DatabaseSchema:
    return DatabaseSchema(
        database="bitcoin",
        tables=[
            TableSchema(
                name="coindesk",
                columns=[
                    ColumnSchema(name="_time", data_type="datetime"),
                    ColumnSchema(name="price", data_type="float"),
                    ColumnSchema(name="code", data_type="string"),
                    ColumnSchema(name="crypto", data_type="string"),
                ],
            )
        ],
    )


def test_flux_renderer_preserves_absolute_time_range_for_seasonality_request():
    context = QueryRequestContext(
        database_id="influxdb2-bitcoin-sample",
        database_type="influxdb",
        message="请判断 Bitcoin USD 在这个时间范围内有没有明显每天或每周重复的周期性波动。",
        time_range={
            "start": "2023-01-04T23:04:00Z",
            "end": "2023-02-03T22:47:00Z",
        },
        constraints={"max_points": 240},
    )
    schema = _build_influx_schema()
    intent = DefaultIntentInterpreter().interpret(context=context)
    mappings = DefaultFieldMapper().map_fields(context=context, schema=schema, intent=intent)
    plan = DefaultLogicalQueryPlanner().build_plan(
        context=context,
        schema=schema,
        intent=intent,
        field_mappings=mappings,
    )

    rendered = CompositeDialectRenderer({"type": "influxdb", "bucket": "bitcoin"}).render(
        context=context,
        plan=plan,
    )

    assert rendered.query_language == "flux"
    assert "range(start: 2023-01-04T23:04:00Z, stop: 2023-02-03T22:47:00Z)" in rendered.query_text
    assert "mean()" not in rendered.query_text
    assert 'r._field == "price"' in rendered.query_text
    assert 'r.code == "USD"' in rendered.query_text
    assert 'r.crypto == "bitcoin"' in rendered.query_text


def test_flux_renderer_uses_all_history_range_when_time_range_is_missing():
    context = QueryRequestContext(
        database_id="influxdb2-bitcoin-sample",
        database_type="influxdb",
        message="总共有多少条数据",
        time_range=None,
        constraints={},
    )
    schema = _build_influx_schema()
    intent = DefaultIntentInterpreter().interpret(context=context)
    mappings = DefaultFieldMapper().map_fields(context=context, schema=schema, intent=intent)
    plan = DefaultLogicalQueryPlanner().build_plan(
        context=context,
        schema=schema,
        intent=intent,
        field_mappings=mappings,
    )

    rendered = CompositeDialectRenderer({"type": "influxdb", "bucket": "bitcoin"}).render(
        context=context,
        plan=plan,
    )

    assert "range(start: 1970-01-01T00:00:00Z)" in rendered.query_text
    assert "range(start: -7d)" not in rendered.query_text
    assert "count()" in rendered.query_text
    assert "mean()" not in rendered.query_text


def test_seasonality_request_with_max_points_stays_timeseries():
    message = (
        "请重新查询 Bitcoin USD 的原始时间序列用于周期性分析，不要使用 max 这类单值聚合，"
        "请尽量返回不超过 240 个点，保留时间戳与价格字段，max_points=240。"
    )
    context = QueryRequestContext(
        database_id="influxdb2-bitcoin-sample",
        database_type="influxdb",
        message=message,
        time_range={
            "start": "2023-01-04T23:04:00Z",
            "end": "2023-02-03T22:47:00Z",
        },
        constraints={"max_points": 240, "avoid_aggregations": ["max"]},
    )
    schema = _build_influx_schema()
    intent = DefaultIntentInterpreter().interpret(context=context)
    mappings = DefaultFieldMapper().map_fields(context=context, schema=schema, intent=intent)
    plan = DefaultLogicalQueryPlanner().build_plan(
        context=context,
        schema=schema,
        intent=intent,
        field_mappings=mappings,
    )

    assert infer_evidence_family(message) == "timeseries"
    assert intent.query_shape == "raw_timeseries"
    assert not intent.filters.get("aggregation")
    assert not any(projection.aggregation for projection in plan.projections)


def test_max_request_with_negated_outlier_filter_uses_scalar_price_max():
    message = (
        "请查询 influxdb2-bitcoin-sample 中 Bitcoin USD 在指定时间范围的原始数据最大值是多少，"
        "并给出最大值对应时间。不要过滤异常值。"
    )
    context = QueryRequestContext(
        database_id="influxdb2-bitcoin-sample",
        database_type="influxdb",
        message=message,
        time_range={
            "start": "2023-01-04T23:04:00Z",
            "end": "2023-02-03T22:47:00Z",
        },
        constraints={"max_points": 240},
    )
    schema = _build_influx_schema()
    intent = DefaultIntentInterpreter().interpret(context=context)
    mappings = DefaultFieldMapper().map_fields(context=context, schema=schema, intent=intent)
    plan = DefaultLogicalQueryPlanner().build_plan(
        context=context,
        schema=schema,
        intent=intent,
        field_mappings=mappings,
    )
    rendered = CompositeDialectRenderer({"type": "influxdb", "bucket": "bitcoin"}).render(
        context=context,
        plan=plan,
    )

    assert infer_evidence_family(message) == "statistics"
    assert intent.query_shape == "scalar_aggregate"
    assert intent.filters["aggregation"] == "max"
    assert plan.projections == [
        plan.projections[0].__class__(
            source="c1",
            column="price",
            alias="max_price",
            aggregation="max",
        )
    ]
    assert 'r._field == "price"' in rendered.query_text
    assert 'r._field == "_time"' not in rendered.query_text
    assert "|> max()" in rendered.query_text


def test_validator_flags_missing_required_value_filters():
    context = QueryRequestContext(
        database_id="influxdb2-bitcoin-sample",
        database_type="influxdb",
        message="请判断 Bitcoin USD 在这个时间范围内有没有明显每天或每周重复的周期性波动。",
        time_range={
            "start": "2023-01-04T23:04:00Z",
            "end": "2023-02-03T22:47:00Z",
        },
    )
    schema = _build_influx_schema()
    intent = DefaultIntentInterpreter().interpret(context=context)
    mappings = DefaultFieldMapper().map_fields(context=context, schema=schema, intent=intent)
    plan = DefaultLogicalQueryPlanner().build_plan(
        context=context,
        schema=schema,
        intent=intent,
        field_mappings=mappings,
    )
    rendered = CompositeDialectRenderer({"type": "influxdb", "bucket": "bitcoin"}).render(
        context=context,
        plan=plan,
    )
    broken = rendered.__class__(
        query_text=rendered.query_text
        .replace('\n  |> filter(fn: (r) => r.code == "USD")', "")
        .replace('\n  |> filter(fn: (r) => r.crypto == "bitcoin")', ""),
        query_language=rendered.query_language,
        structured_request=rendered.structured_request,
        warnings=list(rendered.warnings),
    )

    validation = DefaultQueryValidator().validate(
        context=context,
        plan=plan,
        rendered_query=broken,
    )

    assert not validation.valid
    assert any(issue.code == "required_filter_missing" for issue in validation.issues)


def test_schema_linking_pipeline_outputs_plan_and_required_filters_once():
    context = QueryRequestContext(
        database_id="influxdb2-bitcoin-sample",
        database_type="influxdb",
        message="查询 Bitcoin USD 价格的最晚一条原始记录",
    )
    schema = _build_influx_schema()
    intent = DefaultIntentInterpreter().interpret(context=context)

    result = SchemaLinkingPipeline().ground(context=context, schema=schema, intent=intent)

    assert result.linking.sources[0].name == "coindesk"
    assert result.field_mappings[0].field_name == "price"
    assert result.plan.schema_linking == result.linking.to_dict()
    assert {(item.column, item.value) for item in result.required_filters} == {
        ("code", "USD"),
        ("crypto", "bitcoin"),
    }
    assert {
        (item["column"], tuple(item["values"]))
        for item in result.diagnostics()["candidate_filters"]
    } == {
        ("code", ("EUR", "GBP", "USD")),
        ("crypto", ("bitcoin",)),
    }


def test_sql_renderer_uses_absolute_time_range_and_requested_aggregation():
    schema = DatabaseSchema(
        database="demo",
        tables=[
            TableSchema(
                name="prices",
                schema="public",
                columns=[
                    ColumnSchema(name="timestamp", data_type="timestamp"),
                    ColumnSchema(name="price", data_type="float"),
                ],
            )
        ],
    )
    context = QueryRequestContext(
        database_id="demo",
        database_type="timescaledb",
        message="查询这个时间范围内 price 的平均值",
        time_range={"start": "2023-01-01T00:00:00Z", "end": "2023-01-02T00:00:00Z"},
    )
    intent = DefaultIntentInterpreter().interpret(context=context)
    mappings = DefaultFieldMapper().map_fields(context=context, schema=schema, intent=intent)
    plan = DefaultLogicalQueryPlanner().build_plan(
        context=context,
        schema=schema,
        intent=intent,
        field_mappings=mappings,
    )

    rendered = CompositeDialectRenderer({"type": "timescaledb"}).render(context=context, plan=plan)

    assert "AVG" in rendered.query_text
    assert "2023-01-01T00:00:00Z" in rendered.query_text
    assert "2023-01-02T00:00:00Z" in rendered.query_text


class _FakeConnector:
    dialect = "timescaledb"

    async def get_schema(self) -> DatabaseSchema:
        return DatabaseSchema(
            database="demo",
            tables=[
                TableSchema(
                    name="prices",
                    schema="public",
                    columns=[
                        ColumnSchema(name="timestamp", data_type="timestamp"),
                        ColumnSchema(name="price", data_type="float"),
                    ],
                )
            ],
        )

    async def execute(self, query: str):
        return QueryResult(
            columns=["timestamp", "value"],
            rows=[
                {"timestamp": "2023-01-01T00:00:00Z", "value": 1.0},
                {"timestamp": "2023-01-01T01:00:00Z", "value": 2.0},
            ],
            row_count=2,
            execution_time_ms=5,
        )


class _FakeInfluxProbeConnector:
    dialect = "flux"

    async def get_schema(self) -> DatabaseSchema:
        return _build_influx_schema_without_domains()

    async def probe_value_domains(self, *, source_name: str, columns: list[str], limit: int = 100):
        return {"code": ["USD", "EUR"], "crypto": ["bitcoin"]}

    async def execute(self, query: str):
        return QueryResult(
            columns=["time", "value", "code", "crypto"],
            rows=[
                {"time": "2023-01-04T23:33:00Z", "value": 16422.3833, "code": "USD", "crypto": "bitcoin"},
            ],
            row_count=1,
            execution_time_ms=8,
        )


class _CountingSchemaLinkingPipeline(SchemaLinkingPipeline):
    def __init__(self):
        super().__init__()
        self.ground_calls = 0

    def ground(self, *, context, schema, intent):
        self.ground_calls += 1
        return super().ground(context=context, schema=schema, intent=intent)


def test_database_query_flow_attaches_query_trace():
    with tempfile.TemporaryDirectory() as tmpdir:
        flow = DatabaseQueryFlow(
            connector=_FakeConnector(),
            config={"type": "timescaledb", "snapshot_dir": tmpdir},
        )
        evidence = asyncio.run(
            flow.run(
                context=QueryRequestContext(
                    database_id="demo",
                    database_type="timescaledb",
                    message="分析 price 的趋势",
                    time_range={"start": "2023-01-01T00:00:00Z", "end": "2023-01-02T00:00:00Z"},
                )
            )
        )

        assert evidence.result_type == "timeseries"
        assert evidence.diagnostics["query_trace"]["logical_plan"]["time_range"]["start"] == "2023-01-01T00:00:00Z"
        assert evidence.diagnostics["query_trace"]["rendered_query"]["query_text"]
        snapshot_ref = evidence.diagnostics["query_trace"]["snapshot_ref"]
        assert snapshot_ref["artifact_kind"] == "query_result_snapshot"
        assert evidence.diagnostics["query_snapshot_ref"]["artifact_id"] == snapshot_ref["artifact_id"]
        assert Path(snapshot_ref["uri"]).exists()


def test_database_query_flow_uses_schema_linking_pipeline_once_in_default_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = _CountingSchemaLinkingPipeline()
        flow = DatabaseQueryFlow(
            connector=_FakeConnector(),
            config={"type": "timescaledb", "snapshot_dir": tmpdir},
            schema_linking_pipeline=pipeline,
        )
        asyncio.run(
            flow.run(
                context=QueryRequestContext(
                    database_id="demo",
                    database_type="timescaledb",
                    message="分析 price 的趋势",
                    time_range={"start": "2023-01-01T00:00:00Z", "end": "2023-01-02T00:00:00Z"},
                )
            )
        )

        assert pipeline.ground_calls == 1


def test_database_query_flow_can_probe_value_domains_before_rendering():
    with tempfile.TemporaryDirectory() as tmpdir:
        flow = DatabaseQueryFlow(
            connector=_FakeInfluxProbeConnector(),
            config={"type": "influxdb", "bucket": "bitcoin", "snapshot_dir": tmpdir},
        )
        evidence = asyncio.run(
            flow.run(
                context=QueryRequestContext(
                    database_id="influxdb2-bitcoin-sample",
                    database_type="influxdb",
                    message="请判断 Bitcoin USD 在这个时间范围内有没有明显每天或每周重复的周期性波动。",
                    time_range={"start": "2023-01-04T23:04:00Z", "end": "2023-02-03T22:47:00Z"},
                )
            )
        )

        query_text = evidence.query or ""
        assert 'r.code == "USD"' in query_text
        assert 'r.crypto == "bitcoin"' in query_text
        assert evidence.diagnostics["query_trace"]["logical_plan"]["notes"][-1] == "value_domains_probed=true"


def test_query_observation_can_expose_snapshot_ref_after_state_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        flow = DatabaseQueryFlow(
            connector=_FakeConnector(),
            config={"type": "timescaledb", "snapshot_dir": tmpdir},
        )
        evidence = asyncio.run(
            flow.run(
                context=QueryRequestContext(
                    database_id="demo",
                    database_type="timescaledb",
                    message="分析 price 的趋势",
                    time_range={"start": "2023-01-01T00:00:00Z", "end": "2023-01-02T00:00:00Z"},
                )
            )
        )
        request_state = build_request_state(
            ChatRequest(
                message="分析 price 的趋势",
                database_context=DatabaseContext(database_id="demo", database_type="timescaledb"),
            ),
            get_settings(),
        )
        tool_spec = SimpleNamespace(result_target="evidence")
        full_payload = evidence.model_dump(mode="json")
        apply_observation(
            request_state,
            ToolObservation(tool_name="query_database", success=True, summary="ok", payload={}, error=None),
            full_payload,
            tool_spec,
        )
        enriched = enrich_observation_payload(
            request_state,
            ToolObservation(tool_name="query_database", success=True, summary="ok", payload={}, error=None),
            full_payload,
            tool_spec,
        )
        assert enriched.payload["diagnostics"]["query_snapshot_ref"]["artifact_kind"] == "query_result_snapshot"


def test_reference_dataset_max_points_does_not_sample_analysis_evidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_path = tmp_path / "series.csv"
        dataset_path.write_text(
            "timestamp,value\n"
            + "\n".join(
                f"2023-01-01T00:{index:02d}:00Z,{index}"
                for index in range(10)
            ),
            encoding="utf-8",
        )
        tool = QueryDatabaseTool(get_settings())
        evidence = tool._reference_dataset_timeseries(
            QueryDatabaseInput(
                message="分析 value 的趋势",
                database_context=DatabaseContext(database_id="demo", database_type="csv"),
                constraints={"max_points": 3},
            ),
            tmp_path / "database.yaml",
            {"type": "csv"},
            {
                "dataset_path": str(dataset_path),
                "timestamp_column": "timestamp",
                "field_columns": ["value"],
                "source": "test",
            },
        )

        assert len(evidence["data"]["points"]) == 10
        assert len(evidence["data"]["rows"]) == 10
        assert evidence["diagnostics"]["is_full_fidelity"] is True
        assert evidence["diagnostics"]["sampling_policy"]["requested_max_points"] == 3


def test_reference_dataset_timeseries_returns_empty_evidence_for_unmatched_time_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_path = tmp_path / "series.csv"
        dataset_path.write_text(
            "timestamp,value\n"
            "2016-01-11T17:00:00Z,2\n"
            "2016-01-11T17:10:00Z,3\n",
            encoding="utf-8",
        )
        tool = QueryDatabaseTool(get_settings())
        evidence = tool._reference_dataset_timeseries(
            QueryDatabaseInput(
                message="查询 value",
                database_context=DatabaseContext(database_id="demo", database_type="csv"),
                time_range={"start": "2023-01-01T00:00:00Z", "end": "2023-01-02T00:00:00Z"},
            ),
            tmp_path / "database.yaml",
            {"type": "csv"},
            {
                "dataset_path": str(dataset_path),
                "timestamp_column": "timestamp",
                "field_columns": ["value"],
                "source": "test",
            },
        )

        assert evidence["summary"] == "No rows matched the requested time range for value."
        assert evidence["data"]["points"] == []
        assert evidence["data"]["rows"] == []
        assert evidence["diagnostics"]["row_count_total"] == 0
        assert evidence["diagnostics"]["no_data_reason"] == "time_range_filter_matched_no_rows"


def test_prompt_safe_evidence_still_samples_full_artifact_for_react_context():
    points = [
        {"timestamp": f"2023-01-01T00:{index:02d}:00Z", "value": float(index)}
        for index in range(40)
    ]
    full_payload = {
        "evidence_id": "evi_demo_prompt_safe",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "reference_dataset",
        "query": "reference_dataset:value",
        "summary": "Loaded 40 points.",
        "data": {
            "points": points,
            "rows": [{"timestamp": item["timestamp"], "value": item["value"]} for item in points],
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        "columns": ["timestamp", "value"],
        "metadata": {"database_type": "csv"},
        "diagnostics": {"is_full_fidelity": True},
    }
    request_state = build_request_state(ChatRequest(message="分析趋势"), get_settings())
    apply_observation(
        request_state,
        ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={}),
        full_payload,
        SimpleNamespace(result_target="evidence"),
    )

    assert len(request_state.database_evidence_artifacts["evi_demo_prompt_safe"].data["points"]) == 40
    assert len(request_state.latest_database_evidence.data["points"]) == 24
    assert request_state.latest_database_evidence.diagnostics["summary_stats"]["points_count"] == 40
