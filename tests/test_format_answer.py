from __future__ import annotations

import asyncio

from app.settings import get_settings
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.analysis import AnalysisResult
from schemas.insight import VerifiedFact
from tools.format_answer import FormatAnswerInput, FormatAnswerTool


def test_format_answer_allows_explicit_included_fact_without_unrelated_missing_requirement():
    request_state = build_request_state(
        ChatRequest(message="请把价格分成高位、低位和中间区间"),
        get_settings(),
    )
    request_state.answer_requirements = ["trend"]
    request_state.answer_coverage = {"trend": False}
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
    assert [ref["source_type"] for ref in result["references"]] == ["analysis", "analysis"]
