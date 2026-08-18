from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.settings import get_settings
from core.visualization import PresentationCatalog, VisualizationMaterializer, VisualizationSemanticValidator
from runtime.request_state import build_request_state
from schemas.analysis import DerivedEvidence
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsight, InsightEvidenceRef, InsightItem
from schemas.output import VisualGoal, VisualLayerPlan
from schemas.timeseries import AnomalyResult


def _state(rows: list[dict]):
    state = build_request_state(
        ChatRequest(message="analyze the series", database_context={"database_id": "demo", "database_type": "unit"}),
        get_settings(),
    )
    evidence = DatabaseEvidence(
        evidence_id="evi_series", result_type="timeseries", database="demo",
        summary=f"Loaded {len(rows)} rows.",
        data={"rows": rows, "time_field": "timestamp", "value_field": "value"},
        columns=list(rows[0]) if rows else [],
    )
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    state.latest_database_evidence = evidence
    state.context_budget["max_visible_points"] = 32
    return state


def _goal(*layers: VisualLayerPlan, required_roles: list[str] | None = None):
    return VisualGoal(
        purpose="show the decision in temporal context", title="Decision chart",
        required_roles=required_roles or [layer.role for layer in layers], layers=list(layers),
    )


def test_catalog_exposes_typed_reference_contracts_without_data_rows():
    rows = [{"timestamp": f"2026-01-{day:02d}T00:00:00Z", "value": float(day)} for day in range(1, 20)]
    inventory = PresentationCatalog(_state(rows)).planner_inventory()

    view = next(item for item in inventory["sources"] if item["source_ref"] == "view:evidence:evi_series:default")
    assert inventory["schema_version"] == "semantic-source-v1"
    assert view["shape"] == "timeseries"
    assert view["row_count"] == len(rows)
    assert {field["name"] for field in view["schema_fields"]} == {"timestamp", "value"}
    assert "preview" not in view
    assert "rows" not in view
    assert "examples" not in str(view["data_structure"])
    assert view["projection_root"]["record_path_candidates"] == ["$.records"]


def test_semantic_projection_supports_sparse_fields_in_heterogeneous_records():
    state = _state([{"timestamp": "2026-01-01", "value": 10.0}])
    state.derived_evidence_artifacts["dev_mixed"] = DerivedEvidence(
        evidence_id="dev_mixed",
        name="History and forecast",
        shape="timeseries",
        rows=[
            {"timestamp": "2026-01-01", "value": 10.0, "type": "history"},
            {
                "timestamp": "2026-01-02", "value": 11.0, "type": "forecast",
                "lower": 9.0, "upper": 13.0,
            },
        ],
        lineage=["evidence:evi_series"],
        transform_summary="Combined existing history and forecast rows.",
    )
    catalog = PresentationCatalog(state)
    plan = SimpleNamespace(
        view_id="mixed_interval",
        name="Mixed interval",
        grain="time_point",
        source_ref="view:derived_evidence:dev_mixed",
        record_path="$.records",
        fields=[
            SimpleNamespace(name="time", semantic_role="time", source_path="$.timestamp"),
            SimpleNamespace(name="lower", semantic_role="prediction_lower", source_path="$.lower"),
            SimpleNamespace(name="upper", semantic_role="prediction_upper", source_path="$.upper"),
        ],
    )

    refs = catalog.materialize_semantic_views([plan])
    rows = catalog.resolve(refs[0]).value.rows

    assert rows == [
        {"time": "2026-01-01", "lower": None, "upper": None},
        {"time": "2026-01-02", "lower": 9.0, "upper": 13.0},
    ]


def test_semantic_projection_flattens_an_llm_selected_nested_record_grain():
    state = _state([{"timestamp": "2026-01-01T00:00:00Z", "value": 10.0}])
    state.insight_set.insights = [KeyInsight(
        insight_id="ins_turns",
        insight_key="turning_intervals",
        name="Turning intervals",
        insight_type="pattern",
        statement="Two decline-to-rise intervals were identified.",
        value={"count": 2},
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_series")],
        items=[
            InsightItem(
                item_id="summary",
                value={"summary": "collection metadata without nested event records"},
            ),
            InsightItem(
                item_id="interval_1",
                value={"summary": "first interval"},
                dimensions={"items": [
                    {"role": "decline_end", "timestamp": "2026-01-01T01:00:00Z", "value": 9.0},
                    {"role": "rise_start", "timestamp": "2026-01-01T02:00:00Z", "value": 9.5},
                ]},
            ),
            InsightItem(
                item_id="interval_2",
                value={"summary": "second interval"},
                dimensions={"items": [
                    {"role": "decline_end", "timestamp": "2026-01-01T03:00:00Z", "value": 8.0},
                    {"role": "rise_start", "timestamp": "2026-01-01T04:00:00Z", "value": 8.5},
                ]},
            ),
        ],
    )]
    catalog = PresentationCatalog(state)

    insight_inventory = next(
        source
        for source in catalog.planner_inventory()["sources"]
        if source["source_ref"] == "insight:ins_turns"
    )
    assert "$.items[*].items" in insight_inventory["projection_root"]["record_path_candidates"]

    refs = catalog.materialize_semantic_views([SimpleNamespace(
        view_id="turning_boundaries",
        name="Turning boundaries",
        grain="event",
        source_ref="insight:ins_turns",
        record_path="$.items[*].items",
        fields=[
            SimpleNamespace(name="role", semantic_role="event_role", source_path="$.role"),
            SimpleNamespace(name="timestamp", semantic_role="time", source_path="$.timestamp"),
            SimpleNamespace(name="value", semantic_role="observed_value", source_path="$.value"),
        ],
    )])

    projected = [
        {key: value for key, value in row.items() if key != "__binding_id"}
        for row in catalog.resolve(refs[0]).value.rows
    ]
    assert projected == [
        {"role": "decline_end", "timestamp": "2026-01-01T01:00:00Z", "value": 9.0},
        {"role": "rise_start", "timestamp": "2026-01-01T02:00:00Z", "value": 9.5},
        {"role": "decline_end", "timestamp": "2026-01-01T03:00:00Z", "value": 8.0},
        {"role": "rise_start", "timestamp": "2026-01-01T04:00:00Z", "value": 8.5},
    ]


def test_semantic_view_inventory_retains_nested_query_completeness_context():
    state = _state([{"timestamp": "2026-01-01T00:00:00Z", "value": 10.0}])
    evidence = state.latest_database_evidence
    evidence.diagnostics = {"is_full_fidelity": True, "truncated": False}
    evidence.metadata = {
        "query_execution": {
            "mode": "range",
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-01-02T00:00:00Z",
        }
    }
    catalog = PresentationCatalog(state)
    refs = catalog.materialize_semantic_views([SimpleNamespace(
        view_id="complete_series",
        name="Complete series",
        grain="time_point",
        source_ref="view:evidence:evi_series:default",
        record_path=None,
        fields=[
            SimpleNamespace(name="timestamp", semantic_role="time", source_path="$.timestamp"),
            SimpleNamespace(name="value", semantic_role="observed_value", source_path="$.value"),
        ],
    )])

    semantic = catalog.semantic_inventory(refs)["views"][0]

    assert semantic["materialization_complete"] is True
    assert semantic["query_context"][0]["query_execution"]["end"] == "2026-01-02T00:00:00Z"


def test_semantic_insight_view_inherits_evidence_completeness_and_colocated_items():
    state = _state([{"timestamp": "2026-01-01T00:00:00Z", "value": 10.0}])
    state.latest_database_evidence.diagnostics = {"is_full_fidelity": True, "truncated": False}
    insight = KeyInsight(
        insight_id="ins_located", insight_key="located_peak", name="Located peak", insight_type="extreme",
        statement="The peak is 20 at 2026-01-01T01:00:00Z.", method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_series")],
        items=[InsightItem(
            item_id="peak", label="Peak", timestamp="2026-01-01T01:00:00Z", value=20.0,
        )],
    )
    state.insight_set.insights = [insight]
    catalog = PresentationCatalog(state)
    refs = catalog.materialize_semantic_views([SimpleNamespace(
        view_id="located_peak", name="Located peak", grain="event",
        source_ref="insight:ins_located", record_path="$.items",
        fields=[
            SimpleNamespace(name="timestamp", semantic_role="time", source_path="$.timestamp"),
            SimpleNamespace(name="value", semantic_role="observed_value", source_path="$.value"),
        ],
    )])

    semantic = catalog.semantic_inventory(refs)["views"][0]

    assert "preview" not in semantic
    assert "records" not in semantic
    assert semantic["materialization_complete"] is True
    assert {item["name"]: item["data_type"] for item in semantic["schema_fields"]} == {
        "timestamp": "time", "value": "number",
    }


def test_insight_preference_resolves_its_related_complete_evidence_view():
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
        {"timestamp": "2026-01-02T00:00:00Z", "value": 12.0},
    ]
    state = _state(rows)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_trend",
            insight_key="trend",
            name="Overall trend",
            insight_type="analysis",
            statement="The full-interval fitted trend rises.",
            value="up",
            method="code_interpreter",
            status="verified",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_series")],
        ),
        KeyInsight(
            insight_id="insight_count",
            insight_key="row_count",
            name="Input count",
            insight_type="count",
            statement="There are two rows.",
            value=2,
            method="sql_query",
            status="verified",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_series")],
        ),
    ]
    catalog = PresentationCatalog(state)

    preferred, unknown = catalog.expand_preferences(["insight:trend"])

    assert unknown == set()
    assert "insight:insight_trend" in preferred
    assert "view:evidence:evi_series:default" in preferred
    targeted_refs = {
        item["source_ref"]
        for item in catalog.targeted_planner_inventory(preferred)["sources"]
    }
    assert targeted_refs == {
        "insight:insight_trend",
        "view:evidence:evi_series:default",
    }


def test_layers_keep_base_series_and_semantic_insight_items_separate():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [{"timestamp": (start + timedelta(hours=i)).isoformat(), "value": float(i + 1)} for i in range(100)]
    state = _state(rows)
    insight = KeyInsight(
        insight_id="insight_trade", insight_key="trade.best", name="Best trade", insight_type="extreme",
        semantic_class="decision_points", derivation="optimization", value_shape="record_set",
        statement="Buy then sell.", value={}, method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_series")],
        items=[
            InsightItem(item_id="buy", label="Buy", timestamp=rows[10]["timestamp"], value=11.0),
            InsightItem(item_id="sell", label="Sell", timestamp=rows[90]["timestamp"], value=91.0),
        ],
    )
    state.insight_set.insights = [insight]
    result = VisualizationMaterializer(state).materialize(_goal(
        VisualLayerPlan(role="clean_series", source_ref="view:evidence:evi_series:default", mark="line", encoding={"x": "timestamp", "y": "value"}),
        VisualLayerPlan(role="buy", source_ref="insight:insight_trade#buy", mark="point", encoding={"x": "timestamp", "y": "value"}),
        VisualLayerPlan(role="sell", source_ref="insight:insight_trade#sell", mark="point", encoding={"x": "timestamp", "y": "value"}),
    ))

    assert result.schema_version == "3"
    assert len(result.datasets[0].series[0].points) == 100
    assert [layer.role for layer in result.layers] == ["clean_series", "buy", "sell"]
    assert {binding.item_id for binding in result.bindings} == {"buy", "sell"}
    assert all(layer.source_ref for layer in result.layers)


def test_derived_evidence_preserves_authoritative_anomaly_lineage():
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
        {"timestamp": "2026-01-02T00:00:00Z", "value": 999.0},
        {"timestamp": "2026-01-03T00:00:00Z", "value": 12.0},
    ]
    state = _state(rows)
    anomaly = AnomalyResult(
        anomaly_id="anomaly_evi_series", detector_name="unit",
        anomaly_points=[rows[1]], diagnostics={"resolved_evidence_id": "evi_series"},
    )
    state.anomaly_artifacts[anomaly.anomaly_id] = anomaly
    derived = DerivedEvidence(
        evidence_id="dev_clean", name="Clean series", shape="timeseries",
        rows=[rows[0], rows[2]],
        lineage=["evidence:evi_series", "anomaly:anomaly_evi_series"],
        transform_summary="Excluded authoritative anomaly points.",
    )
    state.derived_evidence_artifacts[derived.evidence_id] = derived

    result = VisualizationMaterializer(state).materialize(_goal(
        VisualLayerPlan(role="clean_series", source_ref="view:derived_evidence:dev_clean", mark="line", encoding={"x": "timestamp", "y": "value"}),
        VisualLayerPlan(role="excluded_anomalies", source_ref="view:anomaly:anomaly_evi_series:points", mark="point", encoding={"x": "timestamp", "y": "value"}),
    ))

    assert [len(dataset.series[0].points) for dataset in result.datasets] == [2, 1]
    assert result.source_refs == ["view:derived_evidence:dev_clean", "view:anomaly:anomaly_evi_series:points"]


def test_materializer_preserves_llm_authored_roles_without_rematching_them():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    goal = _goal(
        VisualLayerPlan(role="series", source_ref="view:evidence:evi_series:default", mark="line", encoding={"x": "timestamp", "y": "value"}),
        required_roles=["series"],
    )
    result = VisualizationMaterializer(state).materialize(goal)
    assert result.required_roles == ["series"]
    assert [layer.role for layer in result.layers] == ["series"]


def test_semantic_validator_rejects_a_missing_required_visual_role():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    with pytest.raises(ValueError, match="same-role layer.*decision_point"):
        _goal(
            VisualLayerPlan(
                role="series",
                source_ref="view:evidence:evi_series:default",
                mark="line",
                encoding={"x": "timestamp", "y": "value"},
            ),
            required_roles=["series", "decision_point"],
        )


def test_semantic_validator_rejects_a_one_point_line_as_non_verifying():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    goal = _goal(VisualLayerPlan(
        role="trend_basis",
        source_ref="view:evidence:evi_series:default",
        mark="line",
        encoding={"x": "timestamp", "y": "value"},
    ))
    payload = VisualizationMaterializer(state).materialize(goal)

    with pytest.raises(ValueError, match="requires at least two grounded points"):
        VisualizationSemanticValidator(PresentationCatalog(state)).validate(goal, payload)


def test_materializer_rejects_ungrounded_encoding_instead_of_guessing_a_field():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    goal = _goal(VisualLayerPlan(
        role="series", source_ref="view:evidence:evi_series:default", mark="line",
        encoding={"x": "invented_time", "y": "value"},
    ))
    with pytest.raises(ValueError, match="unavailable fields.*invented_time"):
        VisualizationMaterializer(state).materialize(goal)


@pytest.mark.parametrize("mark", ["text", "table"])
def test_non_visual_marks_are_rejected_by_the_contract(mark):
    with pytest.raises(ValueError, match="not a graphical visualization mark"):
        VisualLayerPlan(role="details", source_ref="view:evidence:evi_series:default", mark=mark, encoding={})


def test_structured_field_encodings_are_normalized_to_public_field_names():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    result = VisualizationMaterializer(state).materialize(_goal(VisualLayerPlan.model_validate({
        "role": "series", "source_ref": "view:evidence:evi_series:default", "mark": "line",
        "encoding": {"x": {"field": "timestamp", "data_type": "time"}, "y": {"field": "value", "data_type": "number"}},
    })))
    assert result.layers[0].encoding == {"x": "timestamp", "y": "value"}


def test_one_grounded_record_can_drive_multiple_semantic_point_layers():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    trade = KeyInsight(
        insight_id="insight_trade_record", name="Optimal trade", insight_type="analysis",
        statement="Optimal buy and sell.", method="code_interpreter",
        value={
            "buy_time": "2026-01-01T00:00:00Z", "buy_price": 10.0,
            "sell_time": "2026-01-02T00:00:00Z", "sell_price": 15.0, "profit": 5.0,
        },
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_series")],
    )
    state.insight_set.insights = [trade]
    result = VisualizationMaterializer(state).materialize(_goal(
        VisualLayerPlan(role="buy", source_ref="insight:insight_trade_record", mark="point", encoding={"x": "buy_time", "y": "buy_price"}),
        VisualLayerPlan(role="sell", source_ref="insight:insight_trade_record", mark="point", encoding={"x": "sell_time", "y": "sell_price"}),
    ))

    assert result.datasets[0].series[0].points[0].y == 10.0
    assert result.datasets[1].series[0].points[0].y == 15.0


def test_materializer_allows_llm_to_reuse_a_projection_for_distinct_visual_roles():
    state = _state([
        {"timestamp": "2026-01-01", "value": 10.0, "role": "buy"},
        {"timestamp": "2026-01-02", "value": 15.0, "role": "sell"},
    ])
    goal = _goal(
        VisualLayerPlan(
            role="buy", source_ref="view:evidence:evi_series:default", mark="point",
            encoding={"x": "timestamp", "y": "value"},
        ),
        VisualLayerPlan(
            role="sell", source_ref="view:evidence:evi_series:default", mark="point",
            encoding={"x": "timestamp", "y": "value"},
        ),
    )

    result = VisualizationMaterializer(state).materialize(goal)
    assert [layer.role for layer in result.layers] == ["buy", "sell"]


def test_read_only_filters_create_distinct_layers_without_mutating_upstream_rows():
    rows = [
        {"timestamp": "2026-01-01", "value": 10.0, "role": "buy"},
        {"timestamp": "2026-01-02", "value": 15.0, "role": "sell"},
    ]
    state = _state(rows)
    original = [dict(row) for row in state.latest_database_evidence.data["rows"]]
    result = VisualizationMaterializer(state).materialize(_goal(
        VisualLayerPlan.model_validate({
            "role": "buy", "source_ref": "view:evidence:evi_series:default", "mark": "point",
            "encoding": {"x": "timestamp", "y": "value", "color": "role", "shape": "role"},
            "transform": [{"type": "filter", "field": "role", "operator": "eq", "value": "buy"}],
        }),
        VisualLayerPlan.model_validate({
            "role": "sell", "source_ref": "view:evidence:evi_series:default", "mark": "point",
            "encoding": {"x": "timestamp", "y": "value", "color": "role", "shape": "role"},
            "transform": [{"type": "filter", "field": "role", "operator": "eq", "value": "sell"}],
        }),
    ))

    assert [dataset.series[0].points[0].y for dataset in result.datasets] == [10.0, 15.0]
    assert [layer.transform[0]["value"] for layer in result.layers] == ["buy", "sell"]
    assert state.latest_database_evidence.data["rows"] == original


def test_visual_transform_contract_rejects_calculation_operations():
    with pytest.raises(ValueError, match="Input should be 'filter'"):
        VisualLayerPlan.model_validate({
            "role": "profit", "source_ref": "view:evidence:evi_series:default", "mark": "line",
            "encoding": {"x": "timestamp", "y": "value"},
            "transform": [{"type": "calculate", "field": "profit", "value": "sell-buy"}],
        })


def test_category_encoding_splits_one_grounded_source_into_visual_series():
    state = _state([
        {"timestamp": "2026-01-01", "value": 10.0, "role": "buy"},
        {"timestamp": "2026-01-02", "value": 15.0, "role": "sell"},
    ])
    result = VisualizationMaterializer(state).materialize(_goal(VisualLayerPlan(
        role="trade", source_ref="view:evidence:evi_series:default", mark="point",
        encoding={"x": "timestamp", "y": "value", "series": "role", "color": "role", "shape": "role"},
    )))

    assert [series.name for series in result.datasets[0].series] == ["trade: buy", "trade: sell"]
    assert [series.points[0].metadata["role"] for series in result.datasets[0].series] == ["buy", "sell"]
    assert state.latest_database_evidence.data["rows"][0]["role"] == "buy"


def test_verified_return_rate_scalar_is_losslessly_projected_as_a_bar():
    state = _state([{"timestamp": "2026-01-01", "value": 10.0}])
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_return_rate", insight_key="return_rate", name="Return rate",
        insight_type="point_value", statement="Return rate is 44.3196 percent.", value=44.3196,
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_series")],
        calculation_trace={"formula": "upstream_verified_return_rate", "unit": "percent"},
    )]

    result = VisualizationMaterializer(state).materialize(_goal(VisualLayerPlan(
        role="return_rate", source_ref="insight:insight_return_rate", mark="bar",
        encoding={"x": "label", "y": "value"}, label="Return rate (%)",
    )))

    point = result.datasets[0].series[0].points[0]
    assert point.x == "Return rate"
    assert point.y == 44.3196
    assert state.insight_set.insights[0].value == 44.3196


def test_line_segment_highlight_preserves_full_context_and_upstream_values():
    rows = [
        {"timestamp": f"2026-01-0{day}T00:00:00Z", "value": float(day)}
        for day in range(1, 6)
    ]
    state = _state(rows)
    original = [dict(row) for row in state.latest_database_evidence.data["rows"]]
    result = VisualizationMaterializer(state).materialize(_goal(
        VisualLayerPlan(
            role="full_series", source_ref="view:evidence:evi_series:default", mark="line",
            encoding={"x": "timestamp", "y": "value"},
        ),
        VisualLayerPlan.model_validate({
            "role": "highlighted_interval", "source_ref": "view:evidence:evi_series:default", "mark": "line",
            "encoding": {"x": "timestamp", "y": "value"},
            "transform": [{
                "type": "filter", "field": "timestamp", "operator": "between",
                "value": ["2026-01-02T00:00:00Z", "2026-01-04T00:00:00Z"],
            }],
        }),
    ))

    assert len(result.datasets[0].series[0].points) == 5
    assert [point.y for point in result.datasets[1].series[0].points] == [2.0, 3.0, 4.0]
    assert state.latest_database_evidence.data["rows"] == original


def test_renderer_native_mark_multi_field_encoding_and_presentation_are_open_ended():
    rows = [{
        "timestamp": "2026-01-01T00:00:00Z",
        "open": 10.0, "close": 12.0, "low": 9.0, "high": 13.0,
        "volume": 1200.0,
    }]
    state = _state(rows)
    original = [dict(row) for row in state.latest_database_evidence.data["rows"]]
    result = VisualizationMaterializer(state).materialize(_goal(VisualLayerPlan(
        role="ohlc", source_ref="view:evidence:evi_series:default", mark="candlestick",
        encoding={
            "x": "timestamp", "y": ["open", "close", "low", "high"],
            "tooltip": ["timestamp", "volume"],
        },
        presentation={
            "itemStyle": {"color": "#087f5b", "color0": "#b42318"},
            "emphasis": {"focus": "series"},
        },
    )))

    assert result.layers[0].mark == "candlestick"
    assert result.layers[0].encoding["y"] == ["open", "close", "low", "high"]
    assert result.layers[0].encoding["tooltip"] == ["timestamp", "volume"]
    assert result.layers[0].presentation["emphasis"] == {"focus": "series"}
    assert result.datasets[0].series[0].points[0].metadata == rows[0]
    assert state.latest_database_evidence.data["rows"] == original


@pytest.mark.parametrize("data_key", ["data", "source", "dataset", "series", "encode", "dimensions"])
def test_renderer_presentation_cannot_override_grounded_data_binding(data_key):
    state = _state([{"timestamp": "2026-01-01T00:00:00Z", "value": 10.0}])
    layer = VisualLayerPlan(
        role="series", source_ref="view:evidence:evi_series:default", mark="effectScatter",
        encoding={"x": "timestamp", "y": "value"},
        presentation={"emphasis": {data_key: [{"value": 999.0}]}},
    )

    with pytest.raises(ValueError, match="may carry data"):
        VisualizationMaterializer(state).materialize(_goal(layer))


def test_renderer_native_highlight_type_is_not_rejected_by_semantic_role_shape_rules():
    state = _state([{"timestamp": "2026-01-01T00:00:00Z", "value": 10.0}])
    result = VisualizationMaterializer(
        state, visual_constraints={"required_highlights": ["decision"]},
    ).materialize(_goal(VisualLayerPlan(
        role="decision", source_ref="view:evidence:evi_series:default", mark="effectScatter",
        encoding={"x": "timestamp", "y": "value"},
        presentation={"rippleEffect": {"scale": 3}},
    )))

    assert result.layers[0].mark == "effectScatter"


def test_chart_level_presentation_is_kept_separate_from_series_and_data():
    state = _state([{"timestamp": "2026-01-01T00:00:00Z", "value": 10.0}])
    goal = VisualGoal(
        purpose="show intensity", title="Heatmap", required_roles=["intensity"],
        presentation={
            "visualMap": {"type": "continuous", "calculable": True},
            "dataZoom": [{"type": "inside"}],
            "tooltip": {"trigger": "item"},
        },
        layers=[VisualLayerPlan(
            role="intensity", source_ref="view:evidence:evi_series:default", mark="heatmap",
            encoding={"x": "timestamp", "value": "value"},
            presentation={"emphasis": {"focus": "series"}},
        )],
    )

    result = VisualizationMaterializer(state).materialize(goal)

    assert result.presentation["visualMap"]["type"] == "continuous"
    assert result.presentation["dataZoom"] == [{"type": "inside"}]
    assert "visualMap" not in result.layers[0].presentation


def test_related_time_layers_remain_on_one_canvas_even_when_value_scales_differ():
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
        {"timestamp": "2026-01-02T00:00:00Z", "value": 12.0},
    ]
    state = _state(rows)
    state.derived_evidence_artifacts["dev_large_scale"] = DerivedEvidence(
        evidence_id="dev_large_scale", name="Large scale time series", shape="timeseries",
        rows=[
            {"timestamp": "2026-01-01T00:00:00Z", "value": 100_000.0},
            {"timestamp": "2026-01-02T00:00:00Z", "value": 120_000.0},
        ],
        lineage=["evidence:evi_series"], transform_summary="Existing large-scale comparison series.",
    )
    goal = VisualGoal(
        purpose="compare time-aligned measures", title="Shared time chart",
        presentation={"grid": [{"top": "8%"}, {"top": "60%"}]},
        layers=[
            VisualLayerPlan(
                role="actual", source_ref="view:evidence:evi_series:default", mark="line",
                encoding={"x": "timestamp", "y": "value"}, presentation={"xAxisIndex": 0, "yAxisIndex": 0},
            ),
            VisualLayerPlan(
                role="comparison", source_ref="view:derived_evidence:dev_large_scale", mark="line",
                encoding={"x": "timestamp", "y": "value"}, presentation={"xAxisIndex": 0, "yAxisIndex": 1},
            ),
        ],
    )

    result = VisualizationMaterializer(state).materialize(goal)

    assert result.layout == "overlay"


def test_one_canvas_rejects_mixed_time_and_scalar_x_domains_for_llm_replanning():
    state = _state([{"timestamp": "2026-01-01T00:00:00Z", "value": 10.0}])
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_rate", name="Change rate", insight_type="point_value",
        statement="Change rate is 12 percent.", value=12.0, method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_series")],
    )]
    goal = _goal(
        VisualLayerPlan(
            role="actual", source_ref="view:evidence:evi_series:default", mark="line",
            encoding={"x": "timestamp", "y": "value"},
        ),
        VisualLayerPlan(
            role="change_rate", source_ref="insight:insight_rate", mark="bar",
            encoding={"x": "label", "y": "value"},
        ),
    )

    with pytest.raises(ValueError, match="compatible x-domain semantics"):
        VisualizationMaterializer(state).materialize(goal)


def test_between_filter_requires_exactly_two_boundaries():
    with pytest.raises(ValueError, match="exactly two boundary values"):
        VisualLayerPlan.model_validate({
            "role": "interval", "source_ref": "view:evidence:evi_series:default", "mark": "line",
            "encoding": {"x": "timestamp", "y": "value"},
            "transform": [{"type": "filter", "field": "timestamp", "operator": "between", "value": ["t0"]}],
        })
