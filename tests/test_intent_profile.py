from __future__ import annotations

from app.settings import Settings
from core.intent import build_intent_profile_fallback
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.database_context import DatabaseContext


def test_fallback_intent_profile_does_not_infer_forecast_from_keywords():
    profile = build_intent_profile_fallback("请预测 appliances_energy_wh 接下来几个点的走势")

    assert profile["requested_capabilities"] == ["query"]
    assert profile["required_outputs"] == ["conclusion"]


def test_fallback_intent_profile_does_not_infer_anomaly_from_keywords():
    profile = build_intent_profile_fallback("检查 appliances_energy_wh 有没有异常点")

    assert profile["requested_capabilities"] == ["query"]
    assert profile["required_outputs"] == ["conclusion"]


def test_request_state_uses_minimal_profile_until_llm_intent_is_available():
    request_state = build_request_state(
        ChatRequest(
            message="给出起始值、结束值、涨跌幅、最高最低，然后预测接下来 6 个点",
            database_context=DatabaseContext(database_id="influxdb2-bitcoin-sample", database_type="influxdb"),
        ),
        Settings(),
    )

    assert request_state.requested_capabilities == ["query"]


def test_fallback_intent_profile_does_not_encode_analysis_heuristics():
    profile = build_intent_profile_fallback("计算起始值、结束值、涨跌幅、最高最低")

    assert profile["requested_capabilities"] == ["query"]
    assert profile["required_outputs"] == ["conclusion"]
