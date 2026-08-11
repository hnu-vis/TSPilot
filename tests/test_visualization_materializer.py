from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.settings import get_settings
from core.visualization import PresentationCatalog, VisualizationMaterializer
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.data_fact import DataFact, FactEvidenceRef, FactItem
from schemas.output import VisualIntent
from schemas.timeseries import AnomalyResult, ForecastResult, TimeSeriesPoint


def _state(rows: list[dict]):
    state = build_request_state(
        ChatRequest(message="show the data", database_context={"database_id": "demo", "database_type": "unit"}),
        get_settings(),
    )
    evidence = DatabaseEvidence(
        evidence_id="evi_series",
        result_type="timeseries",
        database="demo",
        summary=f"Loaded {len(rows)} rows.",
        data={"rows": rows, "time_field": "timestamp", "value_field": "value"},
        columns=list(rows[0]) if rows else [],
    )
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    state.latest_database_evidence = evidence
    state.context_budget["max_visible_points"] = 32
    return state


def _top3_fact(rows: list[dict]) -> DataFact:
    selected = sorted(rows, key=lambda row: row["value"], reverse=True)[:3]
    return DataFact(
        fact_id="fact_top3",
        fact_key="price.top3",
        name="Price Top 3",
        fact_type="extreme",
        semantic_class="ranking_set",
        derivation="top_k",
        value_shape="ranked_set",
        statement="The three highest prices.",
        value=selected,
        items=[
            FactItem(
                item_id=f"top-{rank}", rank=rank, timestamp=row["timestamp"], value=row["value"],
                evidence_refs=[FactEvidenceRef(source_type="query", source_id="evi_series")],
            )
            for rank, row in enumerate(selected, start=1)
        ],
        method="code_interpreter",
        evidence_refs=[FactEvidenceRef(source_type="query", source_id="evi_series")],
    )


def test_catalog_exposes_bounded_semantic_inventory_not_full_arrays():
    rows = [{"timestamp": f"2026-01-{day:02d}T00:00:00Z", "value": float(day)} for day in range(1, 20)]
    state = _state(rows)
    inventory = PresentationCatalog(state).planner_inventory()

    evidence = next(item for item in inventory["sources"] if item["source_ref"] == "evidence:evi_series")
    assert evidence["row_count"] == len(rows)
    assert evidence["candidate_time_fields"] == ["timestamp"]
    assert evidence["candidate_numeric_fields"] == ["value"]
    assert len(evidence["preview"]) == 4
    assert "rows" not in evidence


def test_timeseries_highlight_samples_context_and_preserves_fact_marks():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"timestamp": (start + timedelta(hours=index)).isoformat(), "value": float(index + 1)}
        for index in range(100)
    ]
    state = _state(rows)
    fact = _top3_fact(rows)
    state.fact_set.facts = [fact]
    intent = VisualIntent(
        purpose="show top three in their existing temporal context",
        template_id="timeseries.highlight",
        title="Top 3 in context",
        source_refs=["evidence:evi_series"],
        fact_refs=["fact:fact_top3"],
    )

    result = VisualizationMaterializer(state).materialize(intent)

    assert result.schema_version == "2"
    assert len(result.dataset.series[0].points) <= 32
    assert len(result.layers[-1].points) == 3
    assert len(result.bindings) == 3
    assert all(point.binding_id for point in result.layers[-1].points)


def test_ranked_rows_render_as_ranking_without_inventing_context():
    rows = [
        {"timestamp": "2026-01-01", "value": 30.0},
        {"timestamp": "2026-01-02", "value": 20.0},
        {"timestamp": "2026-01-03", "value": 10.0},
    ]
    state = _state(rows)
    fact = _top3_fact(rows)
    state.fact_set.facts = [fact]
    intent = VisualIntent(
        purpose="rank the available top values",
        template_id="ranking.topk",
        title="Price Top 3",
        source_refs=["fact:fact_top3"],
        fact_refs=["fact:fact_top3"],
        encodings={"x": "timestamp", "y": "value"},
    )

    result = VisualizationMaterializer(state).materialize(intent)

    assert [point.y for point in result.dataset.series[0].points] == [30.0, 20.0, 10.0]
    assert result.template_id == "ranking.topk"
    assert len(result.bindings) == 3


def test_forecast_is_one_grounded_view_with_history_boundary_and_all_future_points():
    rows = [{"timestamp": f"2026-01-{day:02d}T00:00:00Z", "value": 100.0 + day} for day in range(1, 29)]
    state = _state(rows)
    forecast = ForecastResult(
        forecast_id="forecast_evi_series",
        model_name="unit",
        horizon=7,
        forecast_points=[TimeSeriesPoint(timestamp=f"2026-02-{day:02d}T00:00:00Z", value=130.0 + day) for day in range(1, 8)],
        diagnostics={"coverage": {"input_evidence_refs": ["evi_series"]}},
    )
    state.forecast_artifacts[forecast.forecast_id] = forecast
    intent = VisualIntent(
        purpose="show recent history and the next seven predictions",
        template_id="timeseries.forecast",
        title="Price forecast",
        source_refs=["evidence:evi_series", "forecast:forecast_evi_series"],
    )

    result = VisualizationMaterializer(state).materialize(intent)

    assert len(result.dataset.series) == 2
    assert result.dataset.series[0].role == "historical"
    assert result.dataset.series[1].role == "forecast"
    assert len(result.dataset.series[1].points) == 7
    assert sum(layer.kind == "rule" for layer in result.layers) == 1
    assert len(result.bindings) == 7


def test_unreadable_shared_scale_uses_explicit_facets():
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "small": 1.0, "large": 1_000_000.0},
        {"timestamp": "2026-01-02T00:00:00Z", "small": 1.1, "large": 2_000_000.0},
        {"timestamp": "2026-01-03T00:00:00Z", "small": 1.2, "large": 3_000_000.0},
    ]
    state = _state(rows)
    intent = VisualIntent(
        purpose="compare both time series",
        template_id="timeseries.comparison",
        title="Comparison",
        source_refs=["evidence:evi_series"],
        encodings={"x": "timestamp"},
    )

    result = VisualizationMaterializer(state).materialize(intent)
    assert result.layout == "facets"


def test_distribution_and_relationship_templates_use_general_data_shapes():
    rows = [{"group": "a" if index < 5 else "b", "x": float(index), "y": float(index * index)} for index in range(10)]
    state = _state(rows)
    materializer = VisualizationMaterializer(state)

    histogram = materializer.materialize(VisualIntent(
        purpose="show distribution", template_id="distribution.histogram", title="Distribution",
        source_refs=["evidence:evi_series"], encodings={"value": "y"},
    ))
    scatter = materializer.materialize(VisualIntent(
        purpose="show relationship", template_id="relationship.scatter", title="Relationship",
        source_refs=["evidence:evi_series"], encodings={"x": "x", "y": "y"},
    ))

    assert histogram.dataset.series[0].points
    assert len(scatter.dataset.series[0].points) == 10


def test_detail_interval_anomaly_category_and_boxplot_templates_materialize():
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "group": "a", "value": 10.0},
        {"timestamp": "2026-01-02T00:00:00Z", "group": "a", "value": 12.0},
        {"timestamp": "2026-01-03T00:00:00Z", "group": "b", "value": 30.0},
        {"timestamp": "2026-01-04T00:00:00Z", "group": "b", "value": 32.0},
    ]
    state = _state(rows)
    interval = DataFact(
        fact_id="fact_interval", name="Selected interval", fact_type="custom",
        statement="January 2 through January 3.",
        value={"start": rows[1]["timestamp"], "end": rows[2]["timestamp"]},
        method="code_interpreter", evidence_refs=[FactEvidenceRef(source_type="query", source_id="evi_series")],
    )
    state.fact_set.facts = [interval]
    anomaly = AnomalyResult(
        anomaly_id="anomaly_evi_series", detector_name="unit",
        anomaly_points=[{"timestamp": rows[2]["timestamp"], "value": 30.0}],
        diagnostics={"resolved_evidence_id": "evi_series"},
    )
    state.anomaly_artifacts[anomaly.anomaly_id] = anomaly
    materializer = VisualizationMaterializer(state)

    table = materializer.materialize(VisualIntent(
        purpose="show details", template_id="table.detail", title="Details", source_refs=["evidence:evi_series"],
    ))
    interval_view = materializer.materialize(VisualIntent(
        purpose="show selected interval", template_id="interval.highlight", title="Interval",
        source_refs=["evidence:evi_series"], fact_refs=["fact:fact_interval"],
    ))
    anomaly_view = materializer.materialize(VisualIntent(
        purpose="show anomaly", template_id="timeseries.anomaly", title="Anomaly",
        source_refs=["anomaly:anomaly_evi_series"],
    ))
    category = materializer.materialize(VisualIntent(
        purpose="compare categories", template_id="category.comparison", title="Categories",
        source_refs=["evidence:evi_series"], encodings={"x": "group", "y": "value"},
    ))
    boxplot = materializer.materialize(VisualIntent(
        purpose="compare distributions", template_id="distribution.boxplot", title="Boxplot",
        source_refs=["evidence:evi_series"], encodings={"x": "group", "y": "value"},
    ))

    assert len(table.dataset.rows) == 4
    assert any(layer.kind == "area" for layer in interval_view.layers)
    assert anomaly_view.layers[-1].role == "anomaly"
    assert category.dataset.series[0].points
    assert len(boxplot.dataset.series[0].points) == 2
