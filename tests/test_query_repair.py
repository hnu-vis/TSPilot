from __future__ import annotations

from core.database.repair import classify_query_error, repair_read_only_query


def test_repair_read_only_query_adds_flux_date_import_from_query_shape():
    result = repair_read_only_query(
        query='from(bucket: "b") |> map(fn: (r) => ({ r with hour: date.hour(t: r._time) }))',
        query_language="flux",
    )

    assert result.changed is True
    assert result.reason == "add_flux_date_import"
    assert result.query.startswith('import "date"\n')


def test_repair_read_only_query_adds_flux_date_import_from_error():
    result = repair_read_only_query(
        query='from(bucket: "b") |> map(fn: (r) => ({ r with hour: date.hour(t: r._time) }))',
        query_language="flux",
        error='error @1: undefined identifier date',
    )

    assert result.changed is True
    assert result.reason == "add_flux_date_import"


def test_repair_read_only_query_names_multiple_flux_results():
    query = 'from(bucket: "b") |> range(start: -1d)\n\nfrom(bucket: "b") |> range(start: -7d)'

    result = repair_read_only_query(
        query=query,
        query_language="flux",
        error='tried to produce more than one result with the name "_result"',
    )

    assert result.changed is True
    assert result.reason == "name_flux_results"
    assert 'yield(name: "result_1")' in result.query
    assert 'yield(name: "result_2")' in result.query


def test_classify_query_error_suggests_flux_date_import():
    classified = classify_query_error("undefined identifier date")

    assert classified["suggestion"] == "add_flux_date_import"
