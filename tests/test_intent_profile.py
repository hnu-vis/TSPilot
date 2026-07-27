from __future__ import annotations

from app.settings import Settings
from core.intent import build_intent_profile_fallback
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.database_context import DatabaseContext


def test_fallback_intent_profile_tracks_forecast_requirement():
    profile = build_intent_profile_fallback("请预测 appliances_energy_wh 接下来几个点的走势")

    assert "forecast" in profile["requested_capabilities"]
    assert "forecast" in profile["required_outputs"]


def test_fallback_intent_profile_tracks_anomaly_requirement():
    profile = build_intent_profile_fallback("检查 appliances_energy_wh 有没有异常点")

    assert "anomaly" in profile["requested_capabilities"]
    assert "anomaly" in profile["required_outputs"]


def test_request_state_initializes_fallback_requirements_for_mixed_forecast_request():
    request_state = build_request_state(
        ChatRequest(
            message="给出起始值、结束值、涨跌幅、最高最低，然后预测接下来 6 个点",
            database_context=DatabaseContext(database_id="influxdb2-bitcoin-sample", database_type="influxdb"),
        ),
        Settings(),
    )

    assert "analysis" in request_state.requested_capabilities
    assert "forecast" in request_state.requested_capabilities


def test_fallback_intent_profile_tracks_boundary_change_and_extrema_outputs():
    profile = build_intent_profile_fallback("计算起始值、结束值、涨跌幅、最高最低")

    assert "analysis" in profile["requested_capabilities"]
    assert "analysis" in profile["required_outputs"]
    assert "code_interpreter" not in profile["required_outputs"]
