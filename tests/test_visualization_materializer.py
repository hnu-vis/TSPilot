from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.settings import get_settings
from core.visualization import PresentationCatalog, VisualizationMaterializer
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


def test_catalog_exposes_typed_bounded_data_views():
    rows = [{"timestamp": f"2026-01-{day:02d}T00:00:00Z", "value": float(day)} for day in range(1, 20)]
    inventory = PresentationCatalog(_state(rows)).planner_inventory()

    view = next(item for item in inventory["sources"] if item["source_ref"] == "view:evidence:evi_series:default")
    assert inventory["schema_version"] == "3"
    assert view["shape"] == "timeseries"
    assert view["row_count"] == len(rows)
    assert {field["name"] for field in view["schema_fields"]} == {"timestamp", "value"}
    assert len(view["preview"]) == 4
    assert "rows" not in view


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


def test_semantic_validator_rejects_missing_required_role():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    goal = _goal(
        VisualLayerPlan(role="series", source_ref="view:evidence:evi_series:default", mark="line", encoding={"x": "timestamp", "y": "value"}),
        required_roles=["series", "decision_point"],
    )
    with pytest.raises(ValueError, match="missing required roles"):
        VisualizationMaterializer(state).materialize(goal)


def test_semantic_validator_rejects_unavailable_encoding_field():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    goal = _goal(VisualLayerPlan(
        role="series", source_ref="view:evidence:evi_series:default", mark="line",
        encoding={"x": "invented_time", "y": "value"},
    ))
    with pytest.raises(ValueError, match="unavailable fields"):
        VisualizationMaterializer(state).materialize(goal)


def test_table_is_a_first_class_mark():
    rows = [{"category": "a", "value": 1.0}, {"category": "b", "value": 2.0}]
    result = VisualizationMaterializer(_state(rows)).materialize(_goal(
        VisualLayerPlan(role="details", source_ref="view:evidence:evi_series:default", mark="table", encoding={"columns": ["category", "value"]}),
    ))
    assert result.layers[0].mark == "table"
    assert result.datasets[0].rows == rows
    assert result.datasets[0].columns == ["category", "value"]


def test_structured_field_encodings_are_normalized_to_public_field_names():
    state = _state([{"timestamp": "2026-01-01", "value": 1.0}])
    result = VisualizationMaterializer(state).materialize(_goal(VisualLayerPlan.model_validate({
        "role": "series", "source_ref": "view:evidence:evi_series:default", "mark": "line",
        "encoding": {"x": {"field": "timestamp", "data_type": "time"}, "y": {"field": "value", "data_type": "number"}},
    })))
    assert result.layers[0].encoding == {"x": "timestamp", "y": "value"}


def test_one_grounded_scalar_record_can_drive_multiple_semantic_point_layers_and_table():
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
        VisualLayerPlan(role="summary", source_ref="insight:insight_trade_record", mark="table", encoding={}),
    ))

    assert result.datasets[0].series[0].points[0].y == 10.0
    assert result.datasets[1].series[0].points[0].y == 15.0
    assert result.datasets[2].rows[0]["profit"] == 5.0
