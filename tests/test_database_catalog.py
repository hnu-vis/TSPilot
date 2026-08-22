from __future__ import annotations

from fastapi import HTTPException
import pytest

from app.routes.resources import _editable_payload_to_config, _public_database_config, _validate_complete_database_config
from core.database import ConnectorFactory, DatabaseType, database_catalog, supported_database_types


EXPECTED_TYPES = (
    "influxdb", "influxdb3", "kdb", "prometheus", "timescaledb", "dolphindb",
    "druid", "questdb", "tdengine", "iotdb", "victoriametrics", "griddb",
    "arc", "m3db", "cratedb", "cnosdb", "arcadedb", "greptimedb", "db2",
    "riak_ts", "bangdb", "machbase", "openmldb", "opengemini",
)


def test_product_catalog_enum_and_connector_factory_have_one_support_boundary():
    assert supported_database_types() == EXPECTED_TYPES
    assert {item.value for item in DatabaseType} == set(EXPECTED_TYPES)
    assert {item.value for item in ConnectorFactory._CONNECTORS} == set(EXPECTED_TYPES)


def test_catalog_entries_define_connection_defaults_and_unique_extra_fields():
    assert len(database_catalog()) == 24
    for entry in database_catalog():
        assert entry["defaults"]["host"]
        assert entry["defaults"]["port"] > 0
        keys = [field["key"] for field in entry["extraFields"]]
        assert len(keys) == len(set(keys))


def test_every_catalog_entry_constructs_its_registered_connector_from_defaults():
    for entry in database_catalog():
        defaults = dict(entry["defaults"])
        extra = defaults.pop("extra", {})
        connector = ConnectorFactory.create(entry["type"], **defaults, **extra)
        assert connector.database_type.value == entry["type"]


def test_frontend_extra_payload_is_flattened_but_public_config_hides_secrets():
    config = _editable_payload_to_config({
        "name": "metrics",
        "type": "influxdb",
        "host": "localhost",
        "port": 8086,
        "extra": {"version": "2", "org": "acme", "bucket": "telemetry", "token": "secret"},
    })
    assert config["org"] == "acme"
    assert config["bucket"] == "telemetry"
    assert config["token"] == "secret"

    public = _public_database_config("metrics", config)
    assert public["extra"] == {"org": "acme", "bucket": "telemetry"}
    assert "token" not in public["extra"]


def test_complete_config_validation_requires_common_and_type_specific_fields():
    with pytest.raises(HTTPException, match="host, port, org, bucket, token"):
        _validate_complete_database_config({"type": "influxdb"})

    _validate_complete_database_config({
        "type": "influxdb",
        "host": "localhost",
        "port": 8086,
        "org": "acme",
        "bucket": "telemetry",
        "token": "secret",
    })
