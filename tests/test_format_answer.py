from __future__ import annotations

import asyncio

from app.settings import get_settings
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.analysis import AnalysisResult
from schemas.database import DatabaseEvidence
from schemas.insight import VerifiedFact
from tools.format_answer import FormatAnswerInput, FormatAnswerTool


def test_format_answer_allows_explicit_included_fact_without_unrelated_missing_requirement():
    request_state = build_request_state(
        ChatRequest(message="请把价格分成高位、低位和中间区间"),
        get_settings(),
    )
    fact = VerifiedFact(
        fact_id="fact_bucket",
        fact_type="categorization",
        statement="低位 <= 10，中间区间 10 到 20，高位 >= 20。",
        confidence=0.9,
        evidence={"low_max": 10, "high_min": 20},
        verification_rule="deterministic_quartile_bucket_from_points",
    )
    request_state.verified_facts = [fact]

    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(
                summary_goal="输出分类结果",
                include_fact_ids=["fact_bucket"],
                section_plan=["facts"],
            ),
            request_state=request_state,
        )
    )

    assert result["summary"] == fact.statement
    assert result["sections"][0]["section_type"] == "facts"


def test_format_answer_allows_statistics_evidence_for_count_direct_answer():
    request_state = build_request_state(
        ChatRequest(
            message="总共有多少条数据？",
            database_context={"database_id": "demo", "database_type": "influxdb"},
        ),
        get_settings(),
    )
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_count_stats",
        result_type="statistics",
        database="demo",
        query_language="reference_dataset",
        query="reference_dataset:value:statistics",
        summary="Computed statistics over 19735 rows.",
        data={"statistics": {"count": 19735}},
        columns=["metric", "value"],
        metadata={},
        diagnostics={},
    )

    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(
                summary_goal="回答总条数",
                direct_answer="共有 19,735 条数据。",
                section_plan=["conclusion"],
            ),
            request_state=request_state,
        )
    )

    assert result["summary"] == "共有 19,735 条数据。"
    assert result["references"][0]["source_id"] == "evi_count_stats"


def test_format_answer_renders_real_query_as_fenced_code_section():
    request_state = build_request_state(
        ChatRequest(
            message="查询最近的数据",
            database_context={"database_id": "demo", "database_type": "timescaledb"},
        ),
        get_settings(),
    )
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_recent_rows",
        result_type="table",
        database="demo",
        query_language="timescaledb",
        query="SELECT time, value FROM metrics ORDER BY time DESC LIMIT 10",
        summary="Loaded 10 recent rows.",
        data={"rows": [{"time": "2026-01-01T00:00:00Z", "value": 1.0}]},
        columns=["time", "value"],
        metadata={"sql_query_mode": "llm"},
        diagnostics={},
    )

    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(summary_goal="展示查询结果"),
            request_state=request_state,
        )
    )

    assert result["sections"][1]["section_type"] == "query"
    assert result["sections"][1]["content"].startswith("```sql\nSELECT")
    assert result["sections"][2]["section_type"] == "table"


def test_format_answer_preserves_direct_answer_for_timeseries_query_evidence():
    request_state = build_request_state(
        ChatRequest(
            message="查询当前数据源中比特币 USD 价格的最晚一条原始记录",
            database_context={"database_id": "bitcoin", "database_type": "influxdb"},
        ),
        get_settings(),
    )
    query = (
        'from(bucket: "bitcoin")\n'
        "  |> range(start: 0)\n"
        '  |> filter(fn: (r) => r._measurement == "coindesk" and r.code == "USD")\n'
        '  |> filter(fn: (r) => r._field == "price")\n'
        '  |> sort(columns: ["_time"], desc: true)\n'
        "  |> limit(n: 1)"
    )
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_latest_bitcoin_price",
        result_type="timeseries",
        database="bitcoin",
        query_language="flux",
        query=query,
        summary="Loaded 1 rows across 1 series.",
        data={
            "points": [{"timestamp": "2023-02-03T22:47:00+00:00", "value": 23428.6802}],
            "rows": [{"timestamp": "2023-02-03T22:47:00+00:00", "value": 23428.6802}],
            "time_field": "timestamp",
            "value_field": "value",
        },
        columns=["timestamp", "value"],
        metadata={"sql_query_mode": "llm"},
        diagnostics={},
    )

    direct_answer = "最晚一条原始记录时间为 2023-02-03T22:47:00+00:00，价格为 23428.6802 USD。"
    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(
                summary_goal="回答最新价格",
                direct_answer=direct_answer,
                section_plan=["summary", "query"],
            ),
            request_state=request_state,
        )
    )

    assert result["summary"] == direct_answer
    assert result["sections"][0]["content"] == direct_answer
    assert result["sections"][1]["section_type"] == "query"
    assert result["sections"][1]["content"].startswith("```flux\n")


def test_format_answer_does_not_render_internal_reference_dataset_query_section():
    request_state = build_request_state(
        ChatRequest(
            message="总共有多少条数据？",
            database_context={"database_id": "demo", "database_type": "influxdb"},
        ),
        get_settings(),
    )
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_count_stats",
        result_type="statistics",
        database="demo",
        query_language="reference_dataset",
        query="reference_dataset:value:statistics",
        summary="Computed statistics over 19735 rows.",
        data={"statistics": {"count": 19735}},
        columns=["metric", "value"],
        metadata={},
        diagnostics={},
    )

    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(summary_goal="回答总条数"),
            request_state=request_state,
        )
    )

    assert "query" not in [section["section_type"] for section in result["sections"]]


def test_format_answer_assembles_selected_analysis_results():
    request_state = build_request_state(
        ChatRequest(message="汇总两个分析结果"),
        get_settings(),
    )
    first = AnalysisResult(
        analysis_id="ana_ratio",
        analysis_goal="ratio",
        code_hash="sha256:ratio",
        input_evidence_id="evi",
        input_row_count=10,
        status="succeeded",
        summary="比例为 60%。",
        result={"summary": "比例为 60%。", "metrics": {"ratio": 0.6}, "details": {}},
        diagnostics={},
    )
    second = AnalysisResult(
        analysis_id="ana_bucket",
        analysis_goal="bucket",
        code_hash="sha256:bucket",
        input_evidence_id="evi",
        input_row_count=10,
        status="succeeded",
        summary="低位/中位/高位已划分。",
        result={"summary": "低位/中位/高位已划分。", "metrics": {}, "details": {}},
        diagnostics={},
    )
    request_state.analysis_artifacts = {
        first.analysis_id: first,
        second.analysis_id: second,
    }

    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(
                summary_goal="汇总",
                include_analysis_ids=["ana_ratio", "ana_bucket"],
                section_plan=["analysis"],
            ),
            request_state=request_state,
        )
    )

    assert "比例为 60%" in result["summary"]
    assert "低位/中位/高位已划分" in result["summary"]
    assert result["sections"][0]["section_type"] == "analysis"
    assert "指标: ratio: 0.6" in result["sections"][0]["content"]
    assert result["sections"][0]["structured_payload"]["metrics"] == [{"ratio": 0.6}, {}]
    assert [ref["source_type"] for ref in result["references"]] == ["analysis", "analysis"]


def test_format_answer_preserves_direct_answer_when_analysis_exists():
    request_state = build_request_state(
        ChatRequest(message="分析趋势", database_context={"database_id": "demo", "database_type": "influxdb"}),
        get_settings(),
    )
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_trend",
        result_type="timeseries",
        database="demo",
        query_language="flux",
        query='from(bucket: "bitcoin") |> range(start: -30d)',
        summary="Loaded 50 points.",
        data={"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
        columns=["timestamp", "value"],
        metadata={},
        diagnostics={},
    )
    analysis = AnalysisResult(
        analysis_id="ana_trend",
        analysis_goal="trend",
        code_hash="sha256:trend",
        input_evidence_id="evi_trend",
        input_row_count=50,
        status="succeeded",
        summary="趋势分析已完成。",
        result={"summary": "趋势分析已完成。", "metrics": {"change": -0.2}, "details": {}},
        diagnostics={},
    )
    request_state.analysis_artifacts = {analysis.analysis_id: analysis}

    direct_answer = "整体窄幅震荡，略有回落。"
    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(
                summary_goal="给出趋势结论",
                direct_answer=direct_answer,
                include_analysis_ids=["ana_trend"],
                section_plan=["summary", "analysis", "query"],
            ),
            request_state=request_state,
        )
    )

    assert result["summary"] == direct_answer
    assert result["sections"][0]["content"] == direct_answer
    assert any(section["section_type"] == "analysis" for section in result["sections"])


def test_format_answer_summarizes_all_database_evidence_artifacts():
    request_state = build_request_state(
        ChatRequest(
            message="请返回总数、最早记录和采样说明，并展示每项查询语句和实际返回行数。",
            database_context={"database_id": "bitcoin", "database_type": "influxdb"},
        ),
        get_settings(),
    )
    count = DatabaseEvidence(
        evidence_id="evi_count",
        result_type="table",
        database="bitcoin",
        query_language="flux",
        query='from(bucket: "bitcoin") |> range(start: 0) |> count()',
        summary="Loaded 1 rows.",
        data={"rows": [{"count": 2680}]},
        columns=["count"],
        metadata={"purpose": "返回总记录数"},
        diagnostics={"row_count_total": 1, "artifact_ref": "evidence:evi_count"},
    )
    earliest = DatabaseEvidence(
        evidence_id="evi_earliest",
        result_type="table",
        database="bitcoin",
        query_language="flux",
        query='from(bucket: "bitcoin") |> range(start: 0) |> sort(columns: ["_time"]) |> limit(n: 5)',
        summary="Loaded 5 rows.",
        data={
            "rows": [
                {"timestamp": "2023-01-01T00:00:00Z", "value": 1.0},
                {"timestamp": "2023-01-01T00:01:00Z", "value": 2.0},
            ]
        },
        columns=["timestamp", "value"],
        metadata={"purpose": "返回最早 5 条原始记录"},
        diagnostics={"row_count_total": 5, "artifact_ref": "evidence:evi_earliest"},
    )
    sampled = DatabaseEvidence(
        evidence_id="evi_sampled",
        result_type="timeseries",
        database="bitcoin",
        query_language="flux",
        query='from(bucket: "bitcoin") |> range(start: 0)',
        summary="Loaded 2680 points.",
        data={"points": [{"timestamp": "2023-01-01T00:00:00Z", "value": 1.0}]},
        columns=["timestamp", "value"],
        metadata={"purpose": "返回原始价格序列"},
        diagnostics={
            "summary_stats": {"points_count": 2680},
            "prompt_sampling": {
                "sampled_for_prompt": True,
                "full_counts": {"points_count": 2680},
                "visible_counts": {"points_count": 24},
                "full_artifact_ref": "evidence:evi_sampled",
            },
            "artifact_ref": "evidence:evi_sampled",
        },
    )
    request_state.database_evidence_artifacts = {
        count.evidence_id: count,
        earliest.evidence_id: earliest,
        sampled.evidence_id: sampled,
    }
    request_state.latest_database_evidence = sampled

    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(
                summary_goal="汇总查询结果",
                direct_answer="已完成查询。",
                section_plan=["summary", "query_results", "conclusion"],
            ),
            request_state=request_state,
        )
    )

    query_results = next(section for section in result["sections"] if section["section_type"] == "query_results")
    assert "返回总记录数" in query_results["content"]
    assert "返回最早 5 条原始记录" in query_results["content"]
    assert "结果值：count = 2680" in query_results["content"]
    assert "实际返回行数：5" in query_results["content"]
    assert "当前展示的是采样预览" in query_results["content"]
    assert "| timestamp | value |" in query_results["content"]
    assert '```flux\nfrom(bucket: "bitcoin") |> range(start: 0) |> count()\n```' in query_results["content"]
    assert [ref["source_id"] for ref in result["references"][:3]] == ["evi_count", "evi_earliest", "evi_sampled"]
