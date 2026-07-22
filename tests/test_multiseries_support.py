from __future__ import annotations
from pathlib import Path

import yaml

from app.settings import get_settings
from core.database.engine import normalize_query_result
from core.timeseries.normalization import normalize_timeseries_evidence
from schemas.database_context import DatabaseContext
from tools.query_database import QueryDatabaseInput, QueryDatabaseTool


def test_reference_dataset_query_can_return_multiple_series():
    settings = get_settings()
    tool = QueryDatabaseTool(settings)
    config_path = Path("/home/feilvvl/TSPilot-v0.2/configs/databases/influxdb/influxdb2_energydata.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reference_dataset = config["reference_dataset"]
    validated_input = QueryDatabaseInput(
        message="对比 appliances_energy_wh 和 lights_energy_wh 在 2016-01-11 到 2016-01-12 的走势。",
        database_context=DatabaseContext(database_id="influxdb2-energydata", database_type="influxdb"),
        time_range={"start": "2016-01-11T17:00:00", "end": "2016-01-12T23:00:00"},
        constraints={"max_points": 24},
    )

    evidence = tool._reference_dataset_timeseries(validated_input, config_path, config, reference_dataset)

    assert evidence["result_type"] == "timeseries"
    assert evidence["data"]["value_field"] == "appliances_energy_wh"
    assert len(evidence["data"]["series"]) == 2
    assert [item["series_name"] for item in evidence["data"]["series"]] == [
        "appliances_energy_wh",
        "lights_energy_wh",
    ]
    assert evidence["diagnostics"]["series_count"] == 2
    assert evidence["columns"] == ["timestamp", "appliances_energy_wh", "lights_energy_wh"]


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
