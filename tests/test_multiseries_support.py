from __future__ import annotations
from core.database.engine import normalize_query_result
from core.timeseries.normalization import normalize_timeseries_evidence


def test_normalize_timeseries_evidence_can_select_named_series():
    evidence = normalize_query_result(
        database_id="demo",
        database_type="influxdb",
        query_language="unit",
        query="demo_query",
        result=type(
            "Result",
            (),
            {
                "rows": [
                    {"timestamp": "2016-01-11T00:00:00", "series": "a", "value": 1.0},
                    {"timestamp": "2016-01-11T01:00:00", "series": "a", "value": 2.0},
                    {"timestamp": "2016-01-11T00:00:00", "series": "b", "value": 10.0},
                    {"timestamp": "2016-01-11T01:00:00", "series": "b", "value": 20.0},
                ],
                "columns": ["timestamp", "series", "value"],
            },
        )(),
    )

    selected = normalize_timeseries_evidence(evidence, series_name="b")

    assert selected.series_name == "b"
    assert selected.value_field == "value"
    assert [point.value for point in selected.points] == [10.0, 20.0]


def test_normalize_timeseries_evidence_preserves_non_numeric_result_columns():
    evidence = normalize_query_result(
        database_id="demo",
        database_type="influxdb",
        query_language="flux",
        query="bounds_query",
        result=type(
            "Result",
            (),
            {
                "rows": [
                    {"bound": "earliest", "timestamp": "2023-01-01T00:00:00Z", "price": 10.0},
                    {"bound": "latest", "timestamp": "2023-01-02T00:00:00Z", "price": 12.0},
                ],
                "columns": ["bound", "timestamp", "price"],
                "row_count": 2,
                "truncated": False,
                "execution_time_ms": 1,
            },
        )(),
    )

    assert evidence.result_type == "timeseries"
    assert evidence.columns == ["bound", "timestamp", "price"]
    assert evidence.data["rows"] == [
        {"bound": "earliest", "timestamp": "2023-01-01T00:00:00Z", "price": 10.0},
        {"bound": "latest", "timestamp": "2023-01-02T00:00:00Z", "price": 12.0},
    ]
    assert evidence.data["points"] == [
        {"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
        {"timestamp": "2023-01-02T00:00:00Z", "value": 12.0},
    ]
