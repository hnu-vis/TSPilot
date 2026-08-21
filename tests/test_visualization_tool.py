from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.server import create_app
from app.settings import get_settings
from core.completion import evaluate_goal_completion
from core.harness.observation_view import model_observation_view
from core.visualization import PresentationCatalog, VisualizationArtifactStore
from core.harness import build_action_space, build_observation_frame
from runtime.request_state import apply_observation, build_request_state
from schemas.analysis import AnalysisResult, ComputedInsight, DerivedEvidence
from runtime.action_policy import validate_action
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsight, InsightEvidenceRef, InsightItem
from schemas.task_contract import TaskContract, TaskContractOutput
from schemas.timeseries import AnomalyResult
from schemas.visualization import VisualizationPayload
from schemas.tool import ToolObservation
from tools.base import StructuredToolError
from tools.visualization import (
    VisualCompositionGoal,
    VisualCompositionLayer,
    VisualizationInput,
    VisualizationTool,
    _expand_source_preferences,
    _semantic_error,
)
from tools.registry import build_tool_registry


class _PlannerLlm:
    def __init__(
        self,
        payload: str,
        audit_payload: str | list[str] | None = None,
        projection_payload: str | None = None,
    ):
        self.payload = payload
        self.projection_payload = projection_payload
        self.audit_payloads = (
            list(audit_payload)
            if isinstance(audit_payload, list)
            else [audit_payload or '{"decision":"approve","issues":[],"required_data_request":null}']
        )
        self.calls = 0
        self.projection_prompts = []
        self.audit_prompts = []

    async def ainvoke(self, _messages):
        if _is_verification_prompt(_messages):
            return SimpleNamespace(content=_verification_payload(_messages), response_metadata={})
        if _is_evidence_consumption_prompt(_messages):
            return SimpleNamespace(content=_evidence_consumption_payload(_messages), response_metadata={})
        if _is_projection_prompt(_messages):
            self.projection_prompts.append(_messages)
            return SimpleNamespace(
                content=_projection_ir_payload(
                    self.projection_payload or _projection_for_chart_payload(self.payload)
                ),
                response_metadata={},
            )
        if _is_composition_prompt(_messages):
            return SimpleNamespace(content=_composition_payload(self.payload), response_metadata={})
        self.calls += 1
        if "independently audit" in str(_messages[0][1]):
            self.audit_prompts.append(_messages)
            return SimpleNamespace(
                content=self.audit_payloads[min(len(self.audit_prompts) - 1, len(self.audit_payloads) - 1)],
                response_metadata={},
            )
        return SimpleNamespace(content=_encoding_payload(self.payload), response_metadata={})


class _SequencePlannerLlm:
    def __init__(self, payloads: list[str]):
        self.payloads = list(payloads)
        self.calls = 0
        self.composition_calls = 0
        self.chart_prompts = []

    async def ainvoke(self, _messages):
        if _is_verification_prompt(_messages):
            return SimpleNamespace(content=_verification_payload(_messages), response_metadata={})
        if _is_evidence_consumption_prompt(_messages):
            return SimpleNamespace(content=_evidence_consumption_payload(_messages), response_metadata={})
        if _is_projection_prompt(_messages):
            return SimpleNamespace(
                content=_projection_ir_payload(_projection_for_chart_payload(self.payloads[-1])),
                response_metadata={},
            )
        if _is_composition_prompt(_messages):
            payload = self.payloads[min(self.composition_calls, len(self.payloads) - 1)]
            self.composition_calls += 1
            return SimpleNamespace(content=_composition_payload(payload), response_metadata={})
        if "independently audit" in str(_messages[0][1]):
            self.calls += 1
            return SimpleNamespace(
                content='{"decision":"approve","issues":[],"required_data_request":null}',
                response_metadata={},
            )
        self.chart_prompts.append(_messages)
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return SimpleNamespace(content=_encoding_payload(payload), response_metadata={})


def _is_projection_prompt(messages) -> bool:
    return "You are the semantic projection stage" in str(messages[0][1])


def _is_composition_prompt(messages) -> bool:
    return "You design the semantic composition" in str(messages[0][1])


def _is_encoding_prompt(messages) -> bool:
    return "You bind exact semantic fields" in str(messages[0][1])


def _is_verification_prompt(messages) -> bool:
    return "You define the presentation goal" in str(messages[0][1])


def _is_evidence_consumption_prompt(messages) -> bool:
    prompt = str(messages[0][1])
    return (
        "You bind a fixed visualization goal" in prompt
        or "You resolve only the primary axis-bearing data relationship" in prompt
    )


def _is_chart_prompt(messages) -> bool:
    return _is_encoding_prompt(messages)


def _evidence_consumption_payload(messages) -> str:
    prompt = str(messages[0][1])
    match = re.search(r"Allowed lineage source refs: (\[[^\n]*\])", prompt)
    refs = json.loads(match.group(1)) if match else []
    return json.dumps({
        "decision": "ready",
        "rationale": "The selected test lineage supplies the fixed visual goal.",
        "source_uses": [
            {"source_ref": ref, "purpose": "Supply grounded test evidence."}
            for ref in refs
        ],
        "required_data_request": None,
    })


def _strict_dependency_payload(value: dict | None) -> dict | None:
    if value is None:
        return None
    result = dict(value)
    result.setdefault("message", None)
    result.setdefault("required_fields", [])
    result.setdefault("required_properties", [])
    result.setdefault("input_evidence", None)
    result.setdefault("input_source_refs", [])
    result.setdefault("insight_requests", [])
    result["insight_requests"] = [
        {
            "name": item["name"],
            "insight_type": item["insight_type"],
            "insight_key": item.get("insight_key"),
        }
        for item in result["insight_requests"]
    ]
    return result


def _projection_ir_payload(payload: str | dict) -> str:
    decoded = json.loads(payload) if isinstance(payload, str) else dict(payload)
    views = []
    for raw_view in decoded.get("semantic_views", []):
        view = dict(raw_view)
        view.setdefault("record_path", None)
        view.setdefault("mode", "records")
        if view["mode"] == "events":
            view["mode"] = "wide_events"
        views.append(view)
    return json.dumps({
        "semantic_views": views,
        "required_data_request": _strict_dependency_payload(decoded.get("required_data_request")),
    })


def _chart_ir_payload(payload: str | dict) -> str:
    decoded = json.loads(payload) if isinstance(payload, str) else json.loads(json.dumps(payload))
    requirement = _strict_dependency_payload(decoded.get("required_data_request"))
    if requirement is not None:
        return json.dumps({"visual_goals": [], "required_data_request": requirement})
    if any(
        layer.get("layer_type")
        for goal in decoded.get("visual_goals", [])
        for layer in goal.get("layers", [])
    ):
        for goal in decoded.get("visual_goals", []):
            for layer in goal.get("layers", []):
                layer.pop("mark", None)
                layer["source_ref"] = _test_semantic_source_ref(layer["source_ref"])
                if layer.get("interval_source_ref"):
                    layer["interval_source_ref"] = _test_semantic_source_ref(
                        layer["interval_source_ref"]
                    )
                if layer.get("layer_type") != "interval_overlay":
                    for key in (
                        "interval_source_ref", "interval_start_field", "interval_end_field",
                        "interval_start_value", "interval_end_value",
                    ):
                        layer.pop(key, None)
                elif layer.get("interval_source_ref") is not None:
                    layer.pop("interval_start_value", None)
                    layer.pop("interval_end_value", None)
                else:
                    layer.pop("interval_source_ref", None)
                    layer.pop("interval_start_field", None)
                    layer.pop("interval_end_field", None)
        return json.dumps(decoded)
    goals = []
    for raw_goal in decoded.get("visual_goals", []):
        chart_presentation = raw_goal.get("presentation") or {}
        data_zoom = chart_presentation.get("dataZoom") or []
        zoom = data_zoom[0] if data_zoom else {}
        layers = []
        for raw_layer in raw_goal.get("layers", []):
            encoding_items = []
            for channel, raw_value in (raw_layer.get("encoding") or {}).items():
                normalized_channel = {"timestamp": "x", "time": "x", "value": "y"}.get(channel, channel)
                if normalized_channel not in {"x", "y", "value", "lower", "upper", "series", "label"}:
                    continue
                field = raw_value if isinstance(raw_value, str) else raw_value.get("field") if isinstance(raw_value, dict) else None
                if field:
                    encoding_items.append({"channel": normalized_channel, "field": field})
            mark = str(raw_layer.get("mark") or "line")
            transforms = raw_layer.get("transform") or []
            between = next(
                (
                    item
                    for item in transforms
                    if item.get("operator") == "between" and isinstance(item.get("value"), list)
                ),
                None,
            )
            if between:
                layer_type = "interval_overlay"
            elif mark == "band":
                layer_type = "band"
            elif mark in {"point", "scatter", "rule", "rect"}:
                layer_type = "event_points"
            elif mark in {"bar", "boxplot"}:
                layer_type = "comparison"
            else:
                layer_type = "series"
            layer_presentation = raw_layer.get("presentation") or {}
            line_style = layer_presentation.get("lineStyle") or {}
            width = float(line_style.get("width") or 2)
            symbol = str(layer_presentation.get("symbol") or raw_layer.get("symbol") or "none")
            if symbol not in {"none", "circle", "diamond", "triangle", "pin"}:
                symbol = "circle"
            layer = {
                "layer_type": layer_type,
                "role": raw_layer["role"],
                "source_ref": _test_semantic_source_ref(raw_layer["source_ref"]),
                "encodings": encoding_items,
                "emphasis": "strong" if width >= 3 else "subtle" if width <= 1.5 else "normal",
                "line_style": line_style.get("type", "solid"),
                "symbol": symbol,
                "axis": "secondary" if layer_presentation.get("yAxisIndex") == 1 else "primary",
                "label": raw_layer.get("label"),
            }
            if between:
                layer["interval_start_value"] = between["value"][0]
                layer["interval_end_value"] = between["value"][1]
            layers.append(layer)
        legend = chart_presentation.get("legend") or {}
        tooltip = chart_presentation.get("tooltip") or {}
        goals.append({
            "purpose": raw_goal["purpose"],
            "title": raw_goal["title"],
            "priority": raw_goal.get("priority", "primary"),
            "summary": raw_goal.get("summary"),
            "required_roles": raw_goal.get("required_roles", []),
            "show_legend": legend.get("show", True),
            "tooltip": tooltip.get("trigger", "axis") if tooltip.get("show", True) else "none",
            "enable_zoom": bool(data_zoom),
            "viewport_start": zoom.get("startValue"),
            "viewport_end": zoom.get("endValue"),
            "y_scale": "log" if any(item.get("type") == "log" for item in chart_presentation.get("yAxis", [])) else "linear",
            "layers": layers,
        })
    return json.dumps({"visual_goals": goals, "required_data_request": None})


def _composition_payload(payload: str | dict) -> str:
    chart = json.loads(_chart_ir_payload(payload))
    if chart.get("required_data_request") is not None:
        return json.dumps({"visual_goals": [], "required_data_request": chart["required_data_request"]})
    goals = []
    for goal_index, raw_goal in enumerate(chart.get("visual_goals", [])):
        layers = []
        for layer_index, raw_layer in enumerate(raw_goal.get("layers", [])):
            layer_type = raw_layer["layer_type"]
            layers.append({
                "layer_id": f"layer_{goal_index}_{layer_index}",
                "family": (
                    "primary"
                    if layer_index == 0
                    else "highlight" if layer_type == "interval_overlay" else "support"
                ),
                "layer_type": layer_type,
                "role": raw_layer["role"],
                "purpose": f"Render the fixed {raw_layer['role']} evidence role.",
                "source_ref": raw_layer["source_ref"],
                "interval_source_ref": raw_layer.get("interval_source_ref"),
                "label": raw_layer.get("label"),
            })
        goals.append({
            key: raw_goal.get(key)
            for key in (
                "purpose", "title", "priority", "summary", "show_legend", "tooltip",
                "enable_zoom", "viewport_start", "viewport_end", "y_scale",
            )
        } | {"layers": layers})
    return json.dumps({"visual_goals": goals, "required_data_request": None})


def _encoding_payload(payload: str | dict) -> str:
    chart = json.loads(_chart_ir_payload(payload))
    if chart.get("required_data_request") is not None:
        return json.dumps({"layers": [], "required_data_request": chart["required_data_request"]})
    layers = []
    for goal_index, goal in enumerate(chart.get("visual_goals", [])):
        for layer_index, raw_layer in enumerate(goal.get("layers", [])):
            channels = {
                item["channel"]: item["field"]
                for item in raw_layer["encodings"]
            }
            layer = {
                "layer_id": f"layer_{goal_index}_{layer_index}",
                "emphasis": raw_layer["emphasis"],
                "line_style": raw_layer["line_style"],
                "symbol": raw_layer["symbol"],
                "axis": raw_layer["axis"],
            }
            if raw_layer["layer_type"] == "reference_line":
                layer["value_field"] = channels["value"]
                layer["label_field"] = channels.get("label")
            elif raw_layer["layer_type"] == "annotation":
                layer["content_field"] = channels["label"]
                layer["value_field"] = channels.get("value")
                layer["x_field"] = channels.get("x")
            else:
                layer["x_field"] = channels["x"]
                layer["series_field"] = channels.get("series")
                layer["label_field"] = channels.get("label")
            if raw_layer["layer_type"] == "band":
                layer["lower_field"] = channels["lower"]
                layer["upper_field"] = channels["upper"]
            elif raw_layer["layer_type"] not in {"reference_line", "annotation"}:
                layer["y_field"] = channels["y"]
            if raw_layer.get("interval_source_ref") is not None:
                layer["interval_start_field"] = raw_layer["interval_start_field"]
                layer["interval_end_field"] = raw_layer["interval_end_field"]
            elif raw_layer["layer_type"] == "interval_overlay":
                layer["interval_start_value"] = raw_layer["interval_start_value"]
                layer["interval_end_value"] = raw_layer["interval_end_value"]
            layers.append(layer)
    return json.dumps({"layers": layers, "required_data_request": None})


def _strict_test_payload(payload: str | dict, messages) -> str:
    if _is_verification_prompt(messages):
        decoded = json.loads(payload) if isinstance(payload, str) else dict(payload)
        if "outcome" not in decoded:
            decoded = {"outcome": decoded}
        outcome = decoded["outcome"]
        outcome.setdefault("proof_obligations", [])
        outcome["required_data_request"] = _strict_dependency_payload(
            outcome.get("required_data_request")
        )
        return json.dumps(decoded)
    if "independently audit" in str(messages[0][1]) and "assessments" in str(messages[0][1]):
        decoded = json.loads(payload) if isinstance(payload, str) else dict(payload)
        if "outcome" not in decoded:
            decoded = {"outcome": decoded}
        outcome = decoded["outcome"]
        outcome["required_data_request"] = _strict_dependency_payload(
            outcome.get("required_data_request")
        )
        return json.dumps(decoded)
    if _is_projection_prompt(messages):
        return _projection_ir_payload(payload)
    if _is_composition_prompt(messages):
        return _composition_payload(payload)
    if _is_encoding_prompt(messages):
        return _encoding_payload(payload)
    decoded = json.loads(payload) if isinstance(payload, str) else dict(payload)
    if "required_data_request" in decoded:
        decoded["required_data_request"] = _strict_dependency_payload(decoded.get("required_data_request"))
    return json.dumps(decoded)


def _verification_payload(messages) -> str:
    prompt = str(messages[0][1])
    insight_ids = list(dict.fromkeys(re.findall(r'"insight_id":\s*"([^"]+)"', prompt)))
    return json.dumps({"outcome": {
            "decision": "visualize",
            "target_insight_ids": insight_ids,
            "verification_question": "Does the grounded visual evidence support the requested relationship?",
            "interpretation": "Inspect the complete contextual data and the highlighted analytical relationship.",
            "visual_relation": "grounded_comparison",
            "proof_obligations": [],
            "required_context": ["complete contextual evidence"],
            "non_visual_insight_ids": [],
            "required_data_request": None,
        }})


def _projection_for_chart_payload(payload: str) -> str:
    decoded = json.loads(payload)
    requirement = decoded.get("required_data_request")
    if requirement:
        return json.dumps({"semantic_views": [], "required_data_request": requirement})
    layers = [
        layer
        for goal in decoded.get("visual_goals", [])
        for layer in goal.get("layers", [])
    ]
    source_refs = list(dict.fromkeys(
        str(layer.get("source_ref"))
        for layer in layers
        if layer.get("source_ref")
    )) or ["view:evidence:evi_full:default"]
    views = []
    for source_ref in source_refs:
        fields = []
        seen = set()
        for layer in layers:
            if layer.get("source_ref") != source_ref:
                continue
            for value in (layer.get("encoding") or {}).values():
                values = value if isinstance(value, list) else [value]
                for item in values:
                    field_name = item if isinstance(item, str) else item.get("field") if isinstance(item, dict) else None
                    if field_name and field_name not in seen:
                        seen.add(field_name)
                        fields.append({"name": field_name, "semantic_role": field_name, "source_path": f"$.{field_name}"})
        if not fields:
            fields = [{"name": "value", "semantic_role": "measure", "source_path": "$.value"}]
        view_id = _test_semantic_view_id(source_ref)
        views.append({
            "view_id": view_id,
            "name": "Test semantic view",
            "purpose": "Support the chart plan under test",
            "grain": "records",
            "source_ref": source_ref,
            "fields": fields,
        })
    return json.dumps({
        "semantic_views": views,
        "required_data_request": None,
    })


def _test_semantic_view_id(source_ref: str) -> str:
    return f"test_{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(source_ref))}"


def _test_semantic_source_ref(source_ref: str) -> str:
    if source_ref.startswith("semantic:"):
        return source_ref
    return f"semantic:{_test_semantic_view_id(source_ref)}"


class _WorkflowPlannerLlm:
    def __init__(self, *, plans: list[str], audits: list[str]):
        self.plans = list(plans)
        self.audits = list(audits)
        self.plan_calls = 0
        self.audit_calls = 0

    async def ainvoke(self, messages):
        if _is_verification_prompt(messages):
            payload = _verification_payload(messages)
        elif _is_evidence_consumption_prompt(messages):
            payload = _evidence_consumption_payload(messages)
        elif _is_composition_prompt(messages):
            payload = self.plans[min(self.plan_calls, len(self.plans) - 1)]
        elif "independently audit" in str(messages[0][1]):
            payload = self.audits[min(self.audit_calls, len(self.audits) - 1)]
            self.audit_calls += 1
        else:
            payload = self.plans[min(self.plan_calls, len(self.plans) - 1)]
            self.plan_calls += 1
        return SimpleNamespace(content=_strict_test_payload(payload, messages), response_metadata={})


class _TwoStagePlannerLlm:
    def __init__(self, *, projection: dict, chart: dict):
        self.projection = projection
        self.chart = chart
        self.projection_calls = 0
        self.chart_calls = 0
        self.verification_prompts = []
        self.projection_prompts = []
        self.chart_prompts = []
        self.audit_prompts = []

    async def ainvoke(self, messages):
        if _is_verification_prompt(messages):
            self.verification_prompts.append(messages)
            return SimpleNamespace(content=_verification_payload(messages), response_metadata={})
        if _is_evidence_consumption_prompt(messages):
            return SimpleNamespace(content=_evidence_consumption_payload(messages), response_metadata={})
        if "independently audit" in str(messages[0][1]):
            self.audit_prompts.append(messages)
            return SimpleNamespace(
                content='{"decision":"approve","issues":[],"required_data_request":null}',
                response_metadata={},
            )
        if _is_projection_prompt(messages):
            self.projection_calls += 1
            self.projection_prompts.append(messages)
            payload = self.projection
        elif _is_composition_prompt(messages):
            payload = self.chart
        else:
            self.chart_calls += 1
            self.chart_prompts.append(messages)
            payload = self.chart
        return SimpleNamespace(content=_strict_test_payload(payload, messages), response_metadata={})


class _RepairingTwoStagePlannerLlm:
    def __init__(self, *, projections: list[dict], chart: dict | list[dict]):
        self.projections = projections
        self.charts = chart if isinstance(chart, list) else [chart]
        self.projection_prompts = []
        self.chart_calls = 0
        self.composition_calls = 0

    async def ainvoke(self, messages):
        if _is_verification_prompt(messages):
            return SimpleNamespace(content=_verification_payload(messages), response_metadata={})
        if _is_evidence_consumption_prompt(messages):
            return SimpleNamespace(content=_evidence_consumption_payload(messages), response_metadata={})
        if "independently audit" in str(messages[0][1]):
            return SimpleNamespace(
                content='{"decision":"approve","issues":[],"required_data_request":null}',
                response_metadata={},
            )
        if _is_projection_prompt(messages):
            index = min(len(self.projection_prompts), len(self.projections) - 1)
            self.projection_prompts.append(messages)
            payload = self.projections[index]
        elif _is_composition_prompt(messages):
            dependency_sequence = all(
                item.get("required_data_request") is not None
                for item in self.charts
            )
            payload = self.charts[
                min(self.composition_calls, len(self.charts) - 1)
                if dependency_sequence else -1
            ]
            self.composition_calls += 1
        else:
            payload = self.charts[min(self.chart_calls, len(self.charts) - 1)]
            self.chart_calls += 1
        return SimpleNamespace(content=_strict_test_payload(payload, messages), response_metadata={})


class _VerificationOnlyLlm:
    def __init__(self, payload: dict):
        self.payload = payload

    async def ainvoke(self, messages):
        if not _is_verification_prompt(messages):
            raise AssertionError("visualization must stop after the verification decision")
        return SimpleNamespace(content=_strict_test_payload(self.payload, messages), response_metadata={})


class _StrictPlannerLlm:
    def __init__(self, *, projection: dict, chart: dict):
        self.projection = projection
        self.chart = chart
        self.structured_calls = []

    def with_structured_output(self, schema, **kwargs):
        self.structured_calls.append({"schema": schema, **kwargs})
        owner = self

        class _Runnable:
            async def ainvoke(self, messages):
                if _is_verification_prompt(messages):
                    content = _verification_payload(messages)
                elif _is_evidence_consumption_prompt(messages):
                    content = _evidence_consumption_payload(messages)
                elif _is_projection_prompt(messages):
                    content = _projection_ir_payload(owner.projection)
                elif _is_composition_prompt(messages):
                    content = _composition_payload(owner.chart)
                elif _is_encoding_prompt(messages):
                    content = _encoding_payload(owner.chart)
                elif "independently audit" in str(messages[0][1]):
                    content = json.dumps({
                        "decision": "approve",
                        "issues": [],
                        "required_data_request": None,
                    })
                else:
                    raise AssertionError("unexpected visualization LLM stage")
                payload = json.loads(content)
                return {
                    "raw": SimpleNamespace(content=content, response_metadata={}),
                    "parsed": schema.model_validate(payload),
                    "parsing_error": None,
                }

        return _Runnable()


class _RenderAuditStub:
    def __init__(self, decisions: list[dict]):
        self.decisions = list(decisions)
        self.calls: list[list[VisualizationPayload]] = []

    async def audit(self, *, visualizations, **_kwargs):
        self.calls.append(list(visualizations))
        return self.decisions[min(len(self.calls) - 1, len(self.decisions) - 1)]


def _state(point_count: int = 500):
    state = build_request_state(
        ChatRequest(
            message="Show the complete series and its maximum point.",
            database_context={"database_id": "demo", "database_type": "unit"},
        ),
        get_settings(),
    )
    rows = [
        {"timestamp": f"2026-01-{index // 24 + 1:02d}T{index % 24:02d}:00:00Z", "value": float(index)}
        for index in range(point_count)
    ]
    evidence = DatabaseEvidence(
        evidence_id="evi_full",
        result_type="timeseries",
        database="demo",
        query_language="unit",
        query="unit:full",
        summary=f"Loaded {point_count} points.",
        data={"rows": rows, "time_field": "timestamp", "value_field": "value"},
        columns=["timestamp", "value"],
        diagnostics={"is_full_fidelity": True},
    )
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    state.latest_database_evidence = evidence
    return state


def test_planner_inventory_distinguishes_materialization_from_interval_coverage():
    inventory = PresentationCatalog(_state(25)).planner_inventory()
    source = next(item for item in inventory["sources"] if item["source_ref"] == "view:evidence:evi_full:default")

    assert source["materialization_complete"] is True
    assert source["time_range"] == {
        "field": "timestamp",
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-02T00:00:00Z",
    }
    assert source["query_context"][0]["query"] == "unit:full"
    assert "full_fidelity" not in source


def test_planner_inventory_hides_partial_insights_but_keeps_verified_ones():
    state = _state(25)
    partial = KeyInsight(
        insight_id="ins_partial",
        insight_key="partial_point",
        name="Partial point",
        insight_type="point_value",
        statement="Unverified point.",
        method="code_interpreter",
        status="partial",
        items=[InsightItem(item_id="p1", timestamp="2026-01-01T01:00:00Z", value=1.0)],
    )
    verified = partial.model_copy(update={
        "insight_id": "ins_verified",
        "insight_key": "verified_point",
        "name": "Verified point",
        "status": "verified",
    })
    state.insight_set.insights = [partial, verified]

    refs = {
        item["source_ref"]
        for item in PresentationCatalog(state).planner_inventory()["sources"]
    }

    assert "insight:ins_partial" not in refs
    assert "insight:ins_partial#p1" not in refs
    assert "insight:ins_verified" in refs
    assert "insight:ins_verified#p1" not in refs
    verified_source = next(
        item
        for item in PresentationCatalog(state).planner_inventory()["sources"]
        if item["source_ref"] == "insight:ins_verified"
    )
    assert verified_source["insight_key"] == "verified_point"
    assert verified_source["name"] == "Verified point"
    assert verified_source["items"][0]["value"] == 1.0


def _state_with_analysis_views():
    state = _state(25)
    derived = DerivedEvidence(
        evidence_id="dev_endpoints",
        name="interval endpoints",
        shape="timeseries",
        rows=[
            {"role": "start", "timestamp": "2026-01-01T00:00:00Z", "value": 0.0},
            {"role": "end", "timestamp": "2026-01-02T00:00:00Z", "value": 24.0},
        ],
        lineage=["evidence:evi_full"],
        transform_summary="First and last observations.",
    )
    analysis = AnalysisResult(
        analysis_id="ana_demo",
        analysis_goal="calculate interval change",
        code_hash="abc",
        input_evidence_id="evi_full",
        input_row_count=25,
        status="succeeded",
        summary="Computed interval change.",
        computed_insights=[
            ComputedInsight(
                insight_key="absolute_change",
                value=24.0,
                calculation_trace="last minus first",
                derived_evidence_ids=[derived.evidence_id],
            )
        ],
        derived_evidence=[derived],
    )
    state.analysis_artifacts[analysis.analysis_id] = analysis
    state.derived_evidence_artifacts[derived.evidence_id] = derived
    return state


def test_planner_inventory_exposes_renderable_sources_not_storage_artifacts():
    inventory = PresentationCatalog(_state_with_analysis_views()).planner_inventory()
    refs = {item["source_ref"] for item in inventory["sources"]}

    assert "analysis:ana_demo" not in refs
    assert "derived_evidence:dev_endpoints" not in refs
    assert "evidence:evi_full" not in refs
    assert "view:evidence:evi_full:default" in refs
    assert "view:derived_evidence:dev_endpoints" in refs


def test_verified_numeric_scalar_inventory_supports_only_lossless_graphical_marks():
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_change",
            insight_key="absolute_change",
            name="absolute change",
            insight_type="change",
            statement="Absolute change is 24.",
            value=24.0,
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "last-first"},
        )
    ]
    inventory = PresentationCatalog(state).planner_inventory()
    source = next(item for item in inventory["sources"] if item["source_ref"] == "insight:insight_change")
    assert source["render_capabilities"]["scalar_only"] is True
    assert source["render_capabilities"]["timestamped_numeric"] is False
    assert source["render_capabilities"]["renderable"] is True
    assert source["render_capabilities"]["renderer_series_type"] == "open"


def test_visual_composition_allows_one_semantic_role_to_use_distinct_guides():
    goal = VisualCompositionGoal(
        purpose="Inspect observations against their summary",
        title="Observed measure",
        priority="primary",
        summary=None,
        show_legend=True,
        tooltip="axis",
        enable_zoom=False,
        viewport_start=None,
        viewport_end=None,
        y_scale="linear",
        layers=[
            VisualCompositionLayer(
                layer_id="host",
                family="primary",
                layer_type="series",
                role="summary_evidence",
                purpose="Show the complete host series",
                source_ref="semantic:series",
                interval_source_ref=None,
                label=None,
            ),
            VisualCompositionLayer(
                layer_id="mean",
                family="support",
                layer_type="reference_line",
                role="summary_evidence",
                purpose="Show the grounded mean guide",
                source_ref="semantic:summary",
                interval_source_ref=None,
                label="Mean",
            ),
            VisualCompositionLayer(
                layer_id="judgment",
                family="support",
                layer_type="annotation",
                role="summary_evidence",
                purpose="Show the grounded judgment",
                source_ref="semantic:summary",
                interval_source_ref=None,
                label="Judgment",
            ),
        ],
    )

    assert [layer.role for layer in goal.layers] == ["summary_evidence"] * 3


def test_visual_composition_rejects_mixed_host_coordinate_domains():
    common = {
        "purpose": "Compare incompatible hosts",
        "title": "Mixed hosts",
        "priority": "primary",
        "summary": None,
        "show_legend": True,
        "tooltip": "axis",
        "enable_zoom": False,
        "viewport_start": None,
        "viewport_end": None,
        "y_scale": "linear",
    }
    with pytest.raises(ValueError, match="cannot mix temporal and categorical axis domains"):
        VisualCompositionGoal(**common, layers=[
            VisualCompositionLayer(
                layer_id="series",
                family="primary",
                layer_type="series",
                role="history",
                purpose="Show history",
                source_ref="semantic:history",
                interval_source_ref=None,
                label=None,
            ),
            VisualCompositionLayer(
                layer_id="bars",
                family="support",
                layer_type="comparison",
                role="summary",
                purpose="Show categorical summary",
                source_ref="semantic:summary",
                interval_source_ref=None,
                label=None,
            ),
        ])


@pytest.mark.asyncio
async def test_visualization_uses_provider_strict_schema_for_every_active_planning_stage(tmp_path):
    projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose complete observations",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "record_path": "$.records",
            "mode": "records",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Show complete observations",
            "title": "Observed series",
            "priority": "primary",
            "summary": None,
            "required_roles": ["observations"],
            "show_legend": True,
            "tooltip": "axis",
            "enable_zoom": False,
            "viewport_start": None,
            "viewport_end": None,
            "y_scale": "linear",
            "layers": [{
                "layer_type": "series",
                "role": "observations",
                "source_ref": "semantic:observations",
                "mark": "line",
                "encodings": [
                    {"channel": "x", "field": "time"},
                    {"channel": "y", "field": "measure"},
                ],
                "interval_source_ref": None,
                "interval_start_field": None,
                "interval_end_field": None,
                "interval_start_value": None,
                "interval_end_value": None,
                "emphasis": "normal",
                "line_style": "solid",
                "symbol": "none",
                "axis": "primary",
                "label": "Observed value",
            }],
        }],
        "required_data_request": None,
    }
    llm = _StrictPlannerLlm(projection=projection, chart=chart)

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the observed series."), request_state=_state(5))

    assert result["status"] == "created"
    assert len(llm.structured_calls) == 5
    assert all(call["method"] == "json_schema" for call in llm.structured_calls)
    assert all(call["strict"] is True for call in llm.structured_calls)
    assert all(call["include_raw"] is True for call in llm.structured_calls)

    def assert_closed_required_objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for value in node.values():
                assert_closed_required_objects(value)
        elif isinstance(node, list):
            for value in node:
                assert_closed_required_objects(value)

    for call in llm.structured_calls:
        provider_schema = convert_to_openai_tool(
            call["schema"], strict=True,
        )["function"]["parameters"]
        assert "oneOf" not in json.dumps(provider_schema)
        assert_closed_required_objects(provider_schema)


@pytest.mark.asyncio
async def test_temporal_context_is_preferred_over_isolated_scalar_layer(tmp_path):
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_change_rate",
            insight_key="change_rate",
            name="change rate",
            insight_type="change_rate",
            statement="The interval change rate is 24%.",
            value=0.24,
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "(end-start)/start"},
        )
    ]
    plan = (
        '{"visual_goals":[{"purpose":"verify interval change","title":"Change","priority":"primary",'
        '"summary":"The scalar rate remains in the answer while the chart preserves temporal context.",'
        '"required_roles":["series"],"layers":['
        '{"role":"series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    llm = _PlannerLlm(plan)
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(
        llm=llm, artifact_store=store,
    ).execute(
        VisualizationInput(
            message="Analyze and visually verify the interval change rate.",
            source_refs=["evidence:evi_full", "insight:insight_change_rate"],
        ),
        request_state=state,
    )

    visualization = result["visualizations"][0]
    assert llm.calls == 2  # chart planning plus independent semantic audit
    assert [layer["mark"] for layer in visualization["layers"]] == ["line"]
    complete = store.get(visualization["visualization_id"])
    assert complete is not None
    assert len(complete.datasets) == 1
    assert state.insight_set.insights[0].value == 0.24


@pytest.mark.asyncio
async def test_structured_insight_supports_multiple_located_projections(tmp_path):
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_trade",
            insight_key="max_trade_return",
            name="maximum single-trade return",
            insight_type="optimization",
            statement="Buy at 10 and sell at 25 for a profit of 15.",
            value={
                "max_profit_amount": 15.0,
                "max_profit_ratio": 1.5,
                "buy_time": "2026-01-01T00:00:00Z",
                "sell_time": "2026-01-02T00:00:00Z",
                "buy_price": 10.0,
                "sell_price": 25.0,
            },
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "sell_price-buy_price"},
        )
    ]
    plan = (
        '{"visual_goals":[{"purpose":"verify the optimal trade","title":"Optimal trade",'
        '"priority":"primary","summary":null,"required_roles":["series","buy","sell"],"layers":['
        '{"role":"series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"buy","source_ref":"insight:insight_trade","mark":"point",'
        '"encoding":{"x":"buy_time","y":"buy_price"},"label":"Buy"},'
        '{"role":"sell","source_ref":"insight:insight_trade","mark":"point",'
        '"encoding":{"x":"sell_time","y":"sell_price"},"label":"Sell"}]}],"required_data_request":null}'
    )
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(
        llm=_PlannerLlm(plan), artifact_store=store,
    ).execute(
        VisualizationInput(
            message="Exclude anomalies and verify the maximum single-trade return.",
            source_refs=["evidence:evi_full", "insight:insight_trade"],
        ),
        request_state=state,
    )

    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert complete.datasets[1].series[0].points[0].x == "2026-01-01T00:00:00Z"
    assert complete.datasets[2].series[0].points[0].x == "2026-01-02T00:00:00Z"


@pytest.mark.asyncio
async def test_structured_insight_accepts_semantic_encoding_channel_aliases(tmp_path):
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_trade_aliases",
            insight_key="max_trade_return_aliases",
            name="maximum single-trade return",
            insight_type="optimization",
            statement="Buy at 10 and sell at 25 for a profit of 15.",
            value={
                "max_profit": 15.0,
                "buy_time": "2026-01-01T00:00:00Z",
                "sell_time": "2026-01-02T00:00:00Z",
                "buy_price": 10.0,
                "sell_price": 25.0,
            },
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "sell_price-buy_price"},
        )
    ]
    plan = (
        '{"visual_goals":[{"purpose":"verify trade aliases","title":"Optimal trade",'
        '"priority":"primary","summary":null,"required_roles":["buy","sell"],"layers":['
        '{"role":"buy","source_ref":"insight:insight_trade_aliases","mark":"point",'
        '"encoding":{"timestamp":"buy_time","value":"buy_price"},"label":"Buy"},'
        '{"role":"sell","source_ref":"insight:insight_trade_aliases","mark":"point",'
        '"encoding":{"timestamp":"sell_time","value":"sell_price"},"label":"Sell"}]}],'
        '"required_data_request":null}'
    )
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(
        llm=_PlannerLlm(plan), artifact_store=store,
    ).execute(
        VisualizationInput(message="Verify the trade."),
        request_state=state,
    )

    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    buy_points = complete.datasets[0].series[0].points
    sell_points = complete.datasets[1].series[0].points
    assert [(point.x, point.y) for point in buy_points] == [("2026-01-01T00:00:00Z", 10.0)]
    assert [(point.x, point.y) for point in sell_points] == [("2026-01-02T00:00:00Z", 25.0)]
    assert complete.layers[0].encoding == {"x": "buy_time", "y": "buy_price"}
    assert complete.layers[1].encoding == {"x": "sell_time", "y": "sell_price"}


@pytest.mark.asyncio
async def test_two_stage_llm_projects_nested_forecast_insight_and_composes_interval(tmp_path):
    state = _state(25)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_forecast_nested",
        insight_key="week_ahead_forecast",
        name="week ahead forecast",
        insight_type="series",
        statement="Daily forecast points with central price and lower/upper uncertainty bounds.",
        value={"direction": "上涨", "change_pct": 8.0},
        value_shape="collection",
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
        items=[
            InsightItem(
                item_id="f1", label="forecast_point", timestamp="2026-01-03T00:00:00Z",
                value={"predicted_price": 25.0, "lower_price": 22.0, "upper_price": 28.0},
            ),
            InsightItem(
                item_id="f2", label="forecast_point", timestamp="2026-01-04T00:00:00Z",
                value={"predicted_price": 27.0, "lower_price": 23.0, "upper_price": 31.0},
            ),
        ],
    )]
    projection = {
        "semantic_views": [
            {
                "view_id": "history",
                "name": "Observed price history",
                "purpose": "Provide temporal context for the forecast",
                "grain": "observation",
                "source_ref": "view:evidence:evi_full:default",
                "record_path": "$.records",
                "fields": [
                    {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                    {"name": "observed_price", "semantic_role": "observed_price", "source_path": "$.value"},
                ],
            },
            {
                "view_id": "forecast",
                "name": "Forecast path and interval",
                "purpose": "Expose the central forecast and uncertainty interval",
                "grain": "forecast_point",
                "source_ref": "insight:insight_forecast_nested",
                "record_path": "$.items",
                "fields": [
                    {"name": "time", "semantic_role": "forecast_time", "source_path": "$.timestamp"},
                    {"name": "central", "semantic_role": "forecast_central", "source_path": "$.value.predicted_price"},
                    {"name": "lower", "semantic_role": "forecast_lower", "source_path": "$.value.lower_price"},
                    {"name": "upper", "semantic_role": "forecast_upper", "source_path": "$.value.upper_price"},
                ],
            },
            {
                "view_id": "forecast_summary",
                "name": "Forecast direction summary",
                "purpose": "Expose the overall direction and magnitude at summary grain",
                "grain": "forecast_summary",
                "source_ref": "insight:insight_forecast_nested",
                "record_path": "$.value",
                "fields": [
                    {"name": "direction", "semantic_role": "forecast_direction", "source_path": "$.direction"},
                    {"name": "change_pct", "semantic_role": "forecast_change_percent", "source_path": "$.change_pct"},
                ],
            },
        ],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Verify the forecast in historical context",
            "title": "Observed and forecast price",
            "priority": "primary",
            "summary": "History, central forecast, and uncertainty interval.",
            "required_roles": ["history", "forecast", "uncertainty"],
            "layers": [
                {"role": "history", "source_ref": "semantic:history", "mark": "line", "encoding": {"x": "time", "y": "observed_price"}},
                {"role": "forecast", "source_ref": "semantic:forecast", "mark": "line", "encoding": {"x": "time", "y": "central"}},
                {"role": "uncertainty", "source_ref": "semantic:forecast", "mark": "band", "encoding": {"x": "time", "lower": "lower", "upper": "upper"}},
            ],
        }],
        "required_data_request": None,
    }
    llm = _TwoStagePlannerLlm(projection=projection, chart=chart)
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(
            message="Show the history, week-ahead forecast, and its interval.",
            source_refs=["evidence:evi_full", "insight:insight_forecast_nested"],
        ),
        request_state=state,
    )

    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert llm.projection_calls == 1
    assert llm.chart_calls == 1
    assert [layer.source_ref for layer in complete.layers] == [
        "semantic:history", "semantic:forecast", "semantic:forecast",
    ]
    forecast_points = complete.datasets[1].series[0].points
    interval_points = complete.datasets[2].series[0].points
    assert [point.y for point in forecast_points] == [25.0, 27.0]
    assert [(point.lower, point.upper) for point in interval_points] == [(22.0, 28.0), (23.0, 31.0)]
    assert {binding.item_id for binding in complete.bindings if binding.item_id} == {"f1", "f2"}
    assert complete.layout == "overlay"
    assert complete.source_refs == [
        "view:evidence:evi_full:default",
        "insight:insight_forecast_nested",
    ]
    fresh_catalog = PresentationCatalog(state)
    assert all(fresh_catalog.resolve(ref) for ref in complete.source_refs)


@pytest.mark.asyncio
async def test_visualization_tool_projects_nested_insight_records_with_wildcard_path(tmp_path):
    state = _state(6)
    state.latest_database_evidence.data["rows"][-1]["value"] = 987654.321
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_nested_reversal",
        insight_key="nested_reversal",
        name="Nested reversal interval",
        insight_type="interval",
        statement="The selected interval declines into a turning point and then rises.",
        value={"count": 1},
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
        items=[InsightItem(
            item_id="interval_1",
            value={"summary": "decline then rise"},
            dimensions={"items": [
                {"role": "decline_end", "timestamp": "2026-01-01T01:00:00Z", "value": 1.0},
                {"role": "turn", "timestamp": "2026-01-01T02:00:00Z", "value": 2.0},
                {"role": "rise_start", "timestamp": "2026-01-01T03:00:00Z", "value": 3.0},
            ]},
        )],
    )]
    projection = {
        "semantic_views": [
            {
                "view_id": "full_series",
                "name": "Full observed series",
                "purpose": "Provide the complete temporal context",
                "grain": "observation",
                "source_ref": "view:evidence:evi_full:default",
                "record_path": "$.records",
                "fields": [
                    {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                    {"name": "price", "semantic_role": "observed_price", "source_path": "$.value"},
                ],
            },
            {
                "view_id": "reversal_events",
                "name": "Reversal events",
                "purpose": "Expose each located boundary in the nested analytical result",
                "grain": "event",
                "source_ref": "insight:insight_nested_reversal",
                "record_path": "$.items[*].items",
                "fields": [
                    {"name": "event_role", "semantic_role": "event_role", "source_path": "$.role"},
                    {"name": "time", "semantic_role": "event_time", "source_path": "$.timestamp"},
                    {"name": "price", "semantic_role": "event_price", "source_path": "$.value"},
                ],
            },
        ],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Verify the decline-to-rise interval in context",
            "title": "Reversal in context",
            "priority": "primary",
            "summary": "The full series provides context and the nested interval records mark its boundaries.",
            "required_roles": ["complete_series", "reversal_events"],
            "layers": [
                {
                    "role": "complete_series",
                    "source_ref": "semantic:full_series",
                    "mark": "line",
                    "encoding": {"x": "time", "y": "price"},
                },
                {
                    "role": "reversal_events",
                    "source_ref": "semantic:reversal_events",
                    "mark": "point",
                    "encoding": {"x": "time", "y": "price", "series": "event_role"},
                },
            ],
        }],
        "required_data_request": None,
    }
    llm = _TwoStagePlannerLlm(projection=projection, chart=chart)
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(
            message="Show the full series and the exact decline-to-rise interval.",
            source_refs=["insight:nested_reversal"],
        ),
        request_state=state,
    )

    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert result["status"] == "created"
    assert len(complete.datasets[0].series[0].points) == 6
    assert sum(len(series.points) for series in complete.datasets[1].series) == 3
    assert {series.name.rsplit(": ", 1)[-1] for series in complete.datasets[1].series} == {
        "decline_end", "turn", "rise_start",
    }
    prompts = [
        *llm.verification_prompts,
        *llm.projection_prompts,
        *llm.chart_prompts,
        *llm.audit_prompts,
    ]
    assert prompts
    assert all("987654.321" not in str(messages) for messages in prompts)
    assert "The selected interval declines into a turning point and then rises." in str(
        llm.projection_prompts[0]
    )
    assert '"source_ref": "view:evidence:evi_full:default"' in str(llm.projection_prompts[0])
    assert '"source_ref": "insight:insight_nested_reversal"' in str(llm.chart_prompts[0])


@pytest.mark.asyncio
async def test_semantic_projection_repairs_from_path_execution_feedback_without_chart_fallback(tmp_path):
    invalid_projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose the requested temporal measure",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.missing.measure"},
            ],
        }],
        "required_data_request": None,
    }
    repaired_projection = {
        **invalid_projection,
        "semantic_views": [{
            **invalid_projection["semantic_views"][0],
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }],
    }
    chart = {
        "visual_goals": [{
            "purpose": "Show the observed series",
            "title": "Observed series",
            "priority": "primary",
            "summary": None,
            "required_roles": ["observed_series"],
            "layers": [{
                "role": "observed_series",
                "source_ref": "semantic:observations",
                "mark": "line",
                "encoding": {"x": "time", "y": "measure"},
            }],
        }],
        "required_data_request": None,
    }
    llm = _RepairingTwoStagePlannerLlm(
        projections=[invalid_projection, repaired_projection], chart=chart,
    )

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the observed series."), request_state=_state(5))

    assert len(llm.projection_prompts) == 2
    assert "semantic source path '$.missing.measure' is unavailable in every record" in llm.projection_prompts[1][0][1]
    assert "Valid source paths for this record_path: ['$.timestamp', '$.value']" in llm.projection_prompts[1][0][1]
    assert "executable_path_contracts" in llm.projection_prompts[0][0][1]
    assert llm.chart_calls == 1
    assert result["visualizations"][0]["datasets"][0]["row_count"] == 5


@pytest.mark.asyncio
async def test_semantic_projection_keeps_scalar_evidence_as_an_explicit_guide(tmp_path):
    state = _state(5)
    state.derived_evidence_artifacts["dev_status"] = DerivedEvidence(
        evidence_id="dev_status",
        name="Observed summary",
        shape="scalar",
        scalar={"mean_measure": 2.0, "judgment": "low"},
        lineage=["evidence:evi_full"],
        transform_summary="Counted status records.",
    )
    projection = {
        "semantic_views": [{
            "view_id": "status_only",
            "name": "Observed summary",
            "purpose": "Expose grounded mean and judgment guides",
            "grain": "status",
            "source_ref": "view:derived_evidence:dev_status",
            "record_path": None,
            "mode": "records",
            "fields": [
                {
                    "name": "mean_measure",
                    "semantic_role": "observed_measure_average",
                    "source_path": "$.mean_measure",
                },
                {
                    "name": "judgment",
                    "semantic_role": "observed_measure_judgment",
                    "source_path": "$.judgment",
                },
            ],
        }, {
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose the grounded time relationship",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "record_path": "$.records",
            "mode": "records",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Show the observed series",
            "title": "Observed series",
            "priority": "primary",
            "summary": None,
            "required_roles": ["observed_series", "summary_guide"],
            "show_legend": True,
            "tooltip": "axis",
            "enable_zoom": False,
            "viewport_start": None,
            "viewport_end": None,
            "y_scale": "linear",
            "layers": [{
                "layer_type": "series",
                "role": "observed_series",
                "source_ref": "semantic:observations",
                "encodings": [
                    {"channel": "x", "field": "time"},
                    {"channel": "y", "field": "measure"},
                ],
                "emphasis": "normal",
                "line_style": "solid",
                "symbol": "none",
                "axis": "primary",
                "label": None,
            }, {
                "layer_type": "reference_line",
                "role": "summary_guide",
                "source_ref": "semantic:status_only",
                "encodings": [{"channel": "value", "field": "mean_measure"}],
                "emphasis": "strong",
                "line_style": "dashed",
                "symbol": "none",
                "axis": "primary",
                "label": "Mean measure",
            }, {
                "layer_type": "annotation",
                "role": "summary_guide",
                "source_ref": "semantic:status_only",
                "encodings": [{"channel": "label", "field": "judgment"}],
                "emphasis": "normal",
                "line_style": "solid",
                "symbol": "none",
                "axis": "primary",
                "label": "Judgment",
            }],
        }],
        "required_data_request": None,
    }
    llm = _RepairingTwoStagePlannerLlm(
        projections=[projection], chart=chart,
    )

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the observed relationship."), request_state=state)

    assert result["status"] == "created"
    assert len(llm.projection_prompts) == 1
    assert llm.chart_calls == 1
    assert [layer["mark"] for layer in result["visualizations"][0]["layers"]] == [
        "line", "rule", "annotation",
    ]


@pytest.mark.asyncio
async def test_visualization_tool_preserves_code_insight_x_in_annotation(tmp_path):
    state = _state(5)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_code_decision",
        insight_key="code_decision",
        name="Code-derived decision",
        insight_type="decision_point",
        statement="The selected decision occurs at the third observation.",
        value={"count": 1},
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
        calculation_trace={"method": "optimized grounded objective"},
        items=[InsightItem(
            item_id="decision_1",
            timestamp="2026-01-01T02:00:00Z",
            value=2.0,
            label="Selected decision",
            dimensions={"role": "decision"},
        )],
    )]
    projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose the complete host series",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "record_path": None,
            "mode": "records",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }, {
            "view_id": "decision",
            "name": "Code-derived decision",
            "purpose": "Preserve the located code result as an annotation",
            "grain": "decision",
            "source_ref": "insight:insight_code_decision",
            "record_path": None,
            "mode": "records",
            "fields": [
                {"name": "time", "semantic_role": "decision_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "decision_value", "source_path": "$.value"},
                {"name": "role", "semantic_role": "decision_role", "source_path": "$.role"},
            ],
        }],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Locate the code-derived decision in context",
            "title": "Observed series and decision",
            "priority": "primary",
            "summary": None,
            "required_roles": ["observations", "decision"],
            "show_legend": True,
            "tooltip": "axis",
            "enable_zoom": False,
            "viewport_start": None,
            "viewport_end": None,
            "y_scale": "linear",
            "layers": [{
                "layer_type": "series",
                "role": "observations",
                "source_ref": "semantic:observations",
                "encodings": [
                    {"channel": "x", "field": "time"},
                    {"channel": "y", "field": "measure"},
                ],
                "emphasis": "normal",
                "line_style": "solid",
                "symbol": "none",
                "axis": "primary",
                "label": "Observed",
            }, {
                "layer_type": "annotation",
                "role": "decision",
                "source_ref": "semantic:decision",
                "encodings": [
                    {"channel": "x", "field": "time"},
                    {"channel": "value", "field": "measure"},
                    {"channel": "label", "field": "role"},
                ],
                "emphasis": "strong",
                "line_style": "solid",
                "symbol": "pin",
                "axis": "primary",
                "label": "Code result",
            }],
        }],
        "required_data_request": None,
    }
    llm = _RepairingTwoStagePlannerLlm(projections=[projection], chart=chart)
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(message="Show the code-derived decision on the series."),
        request_state=state,
    )

    assert result["status"] == "created"
    visualization = store.get(result["visualization_ids"][0])
    assert visualization is not None
    annotation = visualization.layers[1]
    assert annotation.mark == "annotation"
    assert annotation.encoding == {"label": "role", "y": "measure", "x": "time"}
    assert annotation.points[0].x == "2026-01-01T02:00:00Z"
    assert annotation.points[0].y == 2.0
    assert annotation.points[0].label == "decision"


@pytest.mark.asyncio
async def test_semantic_projection_repairs_mutually_exclusive_records_and_events(tmp_path):
    invalid_projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose observations",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "mode": "records",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
            "events": [{
                "event_role": "incorrectly_mixed_event",
                "timestamp_path": "$.timestamp",
                "value_path": "$.value",
            }],
        }],
        "required_data_request": None,
    }
    repaired_projection = {
        **invalid_projection,
        "semantic_views": [{
            key: value
            for key, value in invalid_projection["semantic_views"][0].items()
            if key != "events"
        }],
    }
    chart = {
        "visual_goals": [{
            "purpose": "Show the observed series",
            "title": "Observed series",
            "priority": "primary",
            "summary": None,
            "required_roles": ["observed_series"],
            "layers": [{
                "role": "observed_series",
                "source_ref": "semantic:observations",
                "mark": "line",
                "encoding": {"x": "time", "y": "measure"},
            }],
        }],
        "required_data_request": None,
    }
    llm = _RepairingTwoStagePlannerLlm(
        projections=[invalid_projection, repaired_projection], chart=chart,
    )

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the observed series."), request_state=_state(5))

    assert len(llm.projection_prompts) == 2
    assert "Extra inputs are not permitted" in str(llm.projection_prompts[1][0][1])
    assert result["status"] == "created"


@pytest.mark.asyncio
async def test_visualization_repairs_degenerate_proof_id_without_reinjecting_outer_error(tmp_path):
    long_id = "complete_context_" + ("bitcoin_timeseries_context_" * 20)
    projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose the complete observed context",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "record_path": "$.records",
            "mode": "records",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Show the observed context",
            "title": "Observed series",
            "priority": "primary",
            "summary": None,
            "required_roles": ["complete_context"],
            "layers": [{
                "role": "complete_context",
                "source_ref": "semantic:observations",
                "mark": "line",
                "encoding": {"x": "time", "y": "measure"},
            }],
        }],
        "required_data_request": None,
    }

    class DegenerateProofPlannerLlm:
        def __init__(self):
            self.verification_prompts = []
            self.all_prompts = []

        async def ainvoke(self, messages):
            self.all_prompts.append(messages)
            if _is_verification_prompt(messages):
                index = len(self.verification_prompts)
                self.verification_prompts.append(messages)
                obligation_id = long_id if index == 0 else "complete_context"
                payload = {
                    "decision": "visualize",
                    "target_insight_ids": [],
                    "verification_question": "Does the complete series show the requested relationship?",
                    "interpretation": "Inspect the complete observed context.",
                    "visual_relation": "observed time-series relationship",
                    "proof_obligations": [{
                        "obligation_id": obligation_id,
                        "description": "Complete observed context.",
                        "required": True,
                    }],
                    "required_context": ["complete observed context"],
                    "non_visual_insight_ids": [],
                    "required_data_request": None,
                }
            elif _is_projection_prompt(messages):
                payload = projection
            elif "independently audit" in str(messages[0][1]):
                payload = {
                    "assessments": [{
                        "obligation_id": "complete_context",
                        "status": "directly_materialized",
                        "missing_evidence_kind": None,
                        "view_ids": ["observations"],
                        "rationale": "The complete observed view directly covers the obligation.",
                    }],
                    "decision": "approve",
                    "issues": [],
                    "required_data_request": None,
                }
            else:
                payload = chart
            return SimpleNamespace(
                content=_strict_test_payload(payload, messages),
                response_metadata={},
            )

    llm = DegenerateProofPlannerLlm()
    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(
        VisualizationInput(
            message="Show the observed series.",
            constraints={
                "repair_contract": {"execution_error": "OUTER_ERROR_SHOULD_NOT_REENTER"},
                "mode": "repair",
            },
        ),
        request_state=_state(5),
    )

    assert result["status"] == "created"
    assert len(llm.verification_prompts) == 2
    assert "at most 32 characters" in str(llm.verification_prompts[1])
    assert all("OUTER_ERROR_SHOULD_NOT_REENTER" not in str(prompt) for prompt in llm.all_prompts)
    assert result["visualizations"][0]["verification"]["proof_obligations"][0]["obligation_id"] == "complete_context"


@pytest.mark.asyncio
async def test_chart_requirement_repairs_ungrounded_input_evidence_with_llm(tmp_path):
    projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Observed series",
            "purpose": "Expose observations",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "record_path": "$.records",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }],
        "required_data_request": None,
    }
    requirement = {
        "required_action": "code_interpreter",
        "purpose": "Calculate a missing interval",
        "message": None,
        "required_shape": "intervals",
        "required_fields": ["time", "lower", "upper"],
        "required_properties": ["aligned with observations"],
        "insight_requests": [{
            "name": "prediction interval",
            "insight_type": "prediction_interval",
            "insight_key": "prediction_interval",
        }],
    }
    invalid = {"visual_goals": [], "required_data_request": {**requirement, "input_evidence": "the observed data"}}
    repaired = {"visual_goals": [], "required_data_request": {**requirement, "input_evidence": "semantic:observations"}}
    llm = _RepairingTwoStagePlannerLlm(
        projections=[projection], chart=[invalid, repaired],
    )

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show a real interval."), request_state=_state(5))

    assert llm.composition_calls == 2
    assert llm.chart_calls == 0
    assert result["status"] == "needs_sources"
    assert result["required_data_request"]["required_action"] == "code_interpreter"
    assert result["required_data_request"]["input_source_refs"] == ["evidence:evi_full"]


def test_direct_timestamp_value_insight_is_a_renderable_locator():
    state = _state(25)
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_start",
            insight_key="start_value",
            name="start value",
            insight_type="point_value",
            statement="Start value is 0.",
            value={"timestamp": "2026-01-01T00:00:00Z", "value": 0.0},
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "first observation"},
        )
    ]
    inventory = PresentationCatalog(state).planner_inventory()
    source = next(item for item in inventory["sources"] if item["source_ref"] == "insight:insight_start")

    assert source["locator_fields"] == ["timestamp", "value"]
    assert source["render_capabilities"]["timestamped_numeric"] is True
    assert source["render_capabilities"]["scalar_only"] is False


@pytest.mark.asyncio
async def test_visualization_tool_persists_every_timeseries_point_and_returns_descriptor(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"show complete pattern","title":"Complete series",'
        '"priority":"primary","summary":null,"required_roles":["base_series"],"layers":['
        '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":{"field":"timestamp","type":"temporal"},"y":{"field":"value","type":"quantitative"}},"label":"Value"}]}],"required_data_request":null}'
    )
    store = VisualizationArtifactStore(tmp_path)
    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(message="Show the complete series."),
        request_state=_state(),
    )

    descriptor = result["visualizations"][0]
    assert descriptor["data_ref"].endswith("/data")
    assert descriptor["datasets"][0]["row_count"] == 500
    assert descriptor["datasets"][0]["series"] == []
    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert complete.datasets[0].row_count == 500
    assert len(complete.datasets[0].series[0].points) == 500


@pytest.mark.asyncio
async def test_visualization_tool_repairs_invalid_llm_plan_inside_tool_boundary(tmp_path):
    llm = _SequencePlannerLlm(
        [
            '{"visual_goals":[{"purpose":"show complete pattern","title":"Complete series",'
            '"priority":"primary","summary":null,"required_roles":["base_series"],"layers":['
            '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
            '"encoding":{"x":"missing_timestamp","y":"value"},"label":"Value"}]}],"required_data_request":null}',
            '{"visual_goals":[{"purpose":"show complete pattern","title":"Complete series",'
            '"priority":"primary","summary":null,"required_roles":["base_series"],"layers":['
            '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
            '"encoding":{"x":"timestamp","y":"value"},"label":"Value"}]}],"required_data_request":null}',
        ]
    )

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the complete series."), request_state=_state(25))

    assert llm.calls == 3  # invalid materialization, repaired plan, and semantic audit
    assert result["visualizations"][0]["datasets"][0]["row_count"] == 25
    repaired_prompt = str(llm.chart_prompts[1][0][1])
    assert "Closed encoding field contract" in repaired_prompt
    assert '"name": "timestamp"' in repaired_prompt
    assert '"name": "value"' in repaired_prompt
    assert '"allowed_consumers": ["event_points", "series", "reference_line", "annotation"]' in repaired_prompt
    assert "missing_timestamp" not in repaired_prompt
    assert "validation error(s)" in repaired_prompt
    assert '"semantic:test_view_evidence_evi_full_default"' in repaired_prompt
    assert "Delete or completely rebuild every rejected layer" in repaired_prompt


@pytest.mark.asyncio
async def test_incompatible_chart_domains_are_replanned_inside_the_tool_boundary(tmp_path):
    state = _state_with_analysis_views()
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_change",
            insight_key="absolute_change",
            name="absolute change",
            insight_type="change",
            statement="Absolute change is 24.",
            value=24.0,
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
            calculation_trace={"formula": "last-first"},
        )
    ]
    invalid = (
        '{"visual_goals":[{"purpose":"verify change","title":"Change","priority":"primary",'
        '"summary":null,"required_roles":["base_series","change"],"layers":['
        '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"change","source_ref":"insight:insight_change","mark":"point",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    repaired = (
        '{"visual_goals":[{"purpose":"verify change","title":"Change","priority":"primary",'
        '"summary":null,"required_roles":["base_series","endpoints"],"layers":['
        '{"role":"base_series","source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null},'
        '{"role":"endpoints","source_ref":"view:derived_evidence:dev_endpoints","mark":"point",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    llm = _SequencePlannerLlm([invalid, repaired])

    result = await VisualizationTool(
        llm=llm, artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(
        VisualizationInput(
            message="Verify interval change.",
            source_refs=["evidence:evi_full", "analysis:ana_demo"],
        ),
        request_state=state,
    )

    assert llm.calls == 2  # fixed-composition field encoding and semantic audit
    assert result["visualizations"][0]["required_roles"] == ["base_series", "endpoints"]


def test_visualization_failure_receipt_does_not_expose_internal_inventory():
    error = _semantic_error(
        ValueError("layer uses unavailable field"),
        {"sources": [{"source_ref": "view:evidence:evi_full:default"}]},
    )

    payload = error.validation_failure
    assert "presentation_inventory" not in payload["repair_contract"]
    assert "view:evidence:" not in str(payload)


@pytest.mark.asyncio
async def test_visualization_tool_requests_full_sql_evidence_instead_of_falling_back(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[],"required_data_request":{"required_action":"sql_query","purpose":"show complete series with max point",'
        '"required_shape":"full_timeseries","required_fields":["timestamp","value"],'
        '"required_properties":["complete requested time range","maximum row with timestamp"],"insight_requests":[]}}'
    )
    tool = VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path))

    result = await tool.execute(VisualizationInput(message="Show all points and max."), request_state=_state(1))

    assert result["status"] == "needs_sources"
    assert result["required_data_request"]["required_action"] == "sql_query"


@pytest.mark.asyncio
async def test_visualization_routes_missing_calculated_layer_to_code_interpreter(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[],"required_data_request":{"required_action":"code_interpreter",'
        '"purpose":"calculate the optimal single-trade decision points",'
        '"required_shape":"decision_points","required_fields":["timestamp","value"],'
        '"required_properties":["buy precedes sell","exclude authoritative anomalies"],'
        '"insight_requests":[{"insight_key":"optimal_trade","name":"Optimal single trade",'
        '"insight_type":"optimization"}]}}'
    )
    tool = VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path))

    result = await tool.execute(VisualizationInput(message="Show the full series and optimal trade."), request_state=_state(25))

    request = result["required_data_request"]
    assert result["status"] == "needs_sources"
    assert request["required_action"] == "code_interpreter"
    assert request["insight_requests"][0]["insight_key"] == "optimal_trade"


@pytest.mark.asyncio
async def test_visualization_normalizes_view_ref_for_code_interpreter_repair_contract(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[],"required_data_request":{"required_action":"code_interpreter",'
        '"purpose":"derive renderer-ready dimensions","required_shape":"records",'
        '"required_fields":["timestamp","value"],"required_properties":["preserve values"],'
        '"input_evidence":"view:evidence:evi_full:default","insight_requests":[{'
        '"insight_key":"renderer_dimensions","name":"Renderer dimensions","insight_type":"series"}]}}'
    )
    tool = VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path))

    result = await tool.execute(VisualizationInput(message="Show a specialized chart."), request_state=_state(25))

    assert result["required_data_request"]["input_source_refs"] == ["evidence:evi_full"]


@pytest.mark.asyncio
async def test_visualization_planner_requests_missing_decision_points(tmp_path):
    plan = (
        '{"visual_goals":[],"required_data_request":{'
        '"required_action":"code_interpreter","purpose":"calculate requested buy and sell points",'
        '"required_shape":"decision_points","required_fields":["timestamp","value"],'
        '"required_properties":["buy precedes sell"],"insight_requests":[{'
        '"insight_key":"optimal_trade","name":"Optimal single trade","insight_type":"optimization"}]}}'
    )
    tool = VisualizationTool(
        llm=_PlannerLlm(plan), artifact_store=VisualizationArtifactStore(tmp_path),
    )

    result = await tool.execute(
        VisualizationInput(message="Show the complete series with buy and sell points."),
        request_state=_state(25),
    )

    assert result["status"] == "needs_sources"
    assert result["required_data_request"]["required_action"] == "code_interpreter"


def test_visualization_data_endpoint_returns_complete_artifact(tmp_path, monkeypatch):
    store = VisualizationArtifactStore(tmp_path)
    state = _state(25)
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )

    import asyncio
    result = asyncio.run(VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(message="Show trend."), request_state=state,
    ))
    monkeypatch.setattr("app.routes.visualizations.get_visualization_artifact_store", lambda: store)
    response = TestClient(create_app()).get(result["visualizations"][0]["data_ref"])

    assert response.status_code == 200
    assert len(response.json()["datasets"][0]["series"][0]["points"]) == 25


def test_visualization_is_a_nonterminal_required_action_when_requested(tmp_path):
    state = _state(10)
    state.requested_capabilities = ["query", "visualization"]
    action_space = build_action_space(build_observation_frame(state)).model_view()
    spec = build_tool_registry(
        get_settings(),
        llm=_PlannerLlm("{}"),
        visualization_artifact_store=VisualizationArtifactStore(tmp_path),
    ).resolve("visualization")

    assert action_space["required_actions"][0]["action"] == "visualization"
    assert spec.produces_terminal_payload is False
    assert spec.result_target == "visualization"


@pytest.mark.asyncio
async def test_completion_distinguishes_full_timeseries_evidence_from_visual_delivery(tmp_path):
    state = _state(25)
    state.task_contract = TaskContract(
        goal="Show the complete series.",
        required_outputs=[
            TaskContractOutput(
                id="complete_timeseries",
                description="complete time-series data",
                output_type="evidence",
                evidence_kind="time_series",
            ),
            TaskContractOutput(
                id="visualization",
                description="chart of all time-series points",
                output_type="visualization",
                evidence_kind="analysis",
            ),
        ],
    )

    before = evaluate_goal_completion(state)
    assert before.can_answer is False
    assert before.missing_evidence == ["visualization"]

    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}',
        audit_payload=(
            '{"decision":"approve","issues":[],"required_data_request":null}'
        ),
    )
    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show trend."), request_state=state)
    state.visualizations = [VisualizationPayload.model_validate(item) for item in result["visualizations"]]
    state.completion_state["latest_gap_assessment"] = {
        "can_answer": False,
        "covered": ["complete time-series data"],
        "missing": ["可视化验证通过"],
    }

    after = evaluate_goal_completion(state)
    assert after.can_answer is True
    assert after.missing_evidence == []


def test_terminate_accepts_sql_verified_derived_insight_without_analysis_artifact():
    state = _state(25)
    state.task_contract = TaskContract(
        goal="Show the complete series and maximum.",
        required_outputs=[
            TaskContractOutput(
                id="max_value",
                description="maximum value",
                output_type="insight",
                evidence_kind="derived_insight",
                measures=["max"],
            ),
            TaskContractOutput(
                id="chart",
                description="complete series with maximum point",
                output_type="visualization",
                evidence_kind="visualization",
            ),
        ],
    )
    state.insight_set.insights = [
        KeyInsight(
            insight_id="insight_max",
            insight_key="max_value",
            name="max_value",
            insight_type="extreme",
            statement="max_value is 24",
            value=24,
            method="sql_query",
            evidence_refs=[
                InsightEvidenceRef(
                    source_type="query",
                    source_id="evi_full",
                    label="complete series",
                )
            ],
            calculation_trace={"operator": "max", "value_key": "value"},
        )
    ]
    state.visualizations = [
        VisualizationPayload(
            visualization_id="viz_complete",
            data_ref="/api/v1/visualizations/viz_complete/data",
            purpose="show complete pattern and maximum",
            title="Complete series",
            source_refs=["evidence:evi_full", "insight:insight_max"],
            required_roles=["base_series", "max_value_highlight"],
            accessibility={"description": "Complete series with maximum point."},
        )
    ]

    allowed, reason = validate_action(
        state,
        "terminate",
        {"direct_answer": "The complete series and maximum are shown."},
    )

    assert allowed is True
    assert reason is None


@pytest.mark.asyncio
async def test_visualization_observation_is_a_react_receipt_not_a_render_payload(tmp_path):
    state = _state(25)
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["base_series"],"layers":[{"role":"base_series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the complete trend."), request_state=state)
    spec = SimpleNamespace(result_target="visualization")
    observation = apply_observation(
        state,
        ToolObservation(
            tool_name="visualization",
            success=True,
            summary=result["summary"],
            payload=result,
        ),
        result,
        spec,
    )

    payload = model_observation_view(observation)["payload"]
    assert payload["visualization_ids"] == result["visualization_ids"]
    assert payload["grounded_by"] == ["evidence:evi_full"]
    assert payload["verification"][0]["full_fidelity"] is True
    assert payload["verification"][0]["datasets"][0]["row_count"] == 25
    assert payload["verification"][0]["materialized_roles"] == ["base_series"]
    assert "visualizations" not in payload
    assert "data_ref" not in str(payload)
    assert "visualization_count" not in payload
    assert "view:evidence:" not in str(payload)


def test_visualization_tool_expands_outer_artifact_refs_to_internal_views():
    catalog = PresentationCatalog(_state_with_analysis_views())

    expanded, unknown = _expand_source_preferences(
        ["analysis:ana_demo", "derived_evidence:dev_endpoints"],
        catalog,
    )

    assert unknown == set()
    assert expanded == {
        "view:evidence:evi_full:default",
        "view:derived_evidence:dev_endpoints",
    }
    assert "analysis:ana_demo" not in expanded
    assert "derived_evidence:dev_endpoints" not in expanded


@pytest.mark.asyncio
async def test_react_policy_reuses_current_visualization_instead_of_regenerating(tmp_path):
    state = _state(25)
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Trend","priority":"primary","summary":null,'
        '"required_roles":["base_series"],"layers":[{"role":"base_series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":null}]}],"required_data_request":null}'
    )
    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show trend."), request_state=state)
    state.visualizations = [VisualizationPayload.model_validate(item) for item in result["visualizations"]]

    allowed, reason = validate_action(
        state,
        "visualization",
        {"message": "Show trend.", "source_refs": ["evidence:evi_full"]},
    )

    assert allowed is False
    assert "Reuse its visualization_id" in reason


@pytest.mark.asyncio
async def test_visualization_carries_key_insight_verification_through_projection_and_artifact(tmp_path):
    state = _state(25)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_max",
        insight_key="maximum_value",
        name="Maximum value",
        insight_type="extreme",
        statement="The maximum value is 24 at the end of the interval.",
        value=24.0,
        method="sql_query",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
    )]
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"verify the maximum in context","title":"Maximum in full context",'
        '"priority":"primary","summary":"The full interval makes the maximum inspectable.",'
        '"required_roles":["complete_series"],"layers":[{"role":"complete_series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":"Observed value"}]}],'
        '"required_data_request":null}'
    )

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(
        VisualizationInput(
            message="Show the maximum in its full interval.",
            source_refs=["insight:maximum_value"],
        ),
        request_state=state,
    )

    verification = result["visualizations"][0]["verification"]
    assert verification["target_insight_ids"] == ["insight_max"]
    assert verification["verification_question"]
    assert verification["interpretation"]
    assert '"target_insight_ids": ["insight_max"]' in str(llm.projection_prompts[0][0][1])
    assert '"source_ref": "view:evidence:evi_full:default"' in str(llm.projection_prompts[0][0][1])
    assert '"preferred_by_caller": true' in str(llm.projection_prompts[0][0][1])


@pytest.mark.asyncio
async def test_line_visualization_closes_proof_obligation_through_semantic_evidence_bundle(tmp_path):
    projection = {
        "semantic_views": [{
            "view_id": "observations",
            "name": "Complete observed series",
            "purpose": "Supply complete context for the claimed line relationship",
            "grain": "observation",
            "source_ref": "view:evidence:evi_full:default",
            "record_path": "$.records",
            "mode": "records",
            "fields": [
                {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                {"name": "measure", "semantic_role": "observed_measure", "source_path": "$.value"},
            ],
        }],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Inspect the observed relationship in complete context",
            "title": "Observed series",
            "priority": "primary",
            "summary": "The complete series is the visual proof context.",
            "required_roles": ["complete_context"],
            "show_legend": True,
            "tooltip": "axis",
            "enable_zoom": False,
            "viewport_start": None,
            "viewport_end": None,
            "y_scale": "linear",
            "layers": [{
                "layer_type": "series",
                "role": "complete_context",
                "source_ref": "semantic:observations",
                "encodings": [
                    {"channel": "x", "field": "time"},
                    {"channel": "y", "field": "measure"},
                ],
                "interval_source_ref": None,
                "interval_start_field": None,
                "interval_end_field": None,
                "interval_start_value": None,
                "interval_end_value": None,
                "emphasis": "normal",
                "line_style": "solid",
                "symbol": "none",
                "axis": "primary",
                "label": "Observed value",
            }],
        }],
        "required_data_request": None,
    }

    class ProofBundlePlannerLlm:
        def __init__(self):
            self.chart_prompts = []

        async def ainvoke(self, messages):
            if _is_verification_prompt(messages):
                return SimpleNamespace(content=_strict_test_payload({
                    "decision": "visualize",
                    "target_insight_ids": [],
                    "verification_question": "Does the complete series show the requested relationship?",
                    "interpretation": "Inspect the complete observed line.",
                    "visual_relation": "observed time-series relationship",
                    "proof_obligations": [{
                        "obligation_id": "complete_context",
                        "description": "The complete observed time series that makes the relationship inspectable.",
                        "required": True,
                    }],
                    "required_context": ["complete observed series"],
                    "non_visual_insight_ids": [],
                    "required_data_request": None,
                }, messages), response_metadata={})
            if _is_projection_prompt(messages):
                return SimpleNamespace(content=_strict_test_payload(projection, messages), response_metadata={})
            if "independently audit" in str(messages[0][1]):
                payload = {
                    "assessments": [{
                        "obligation_id": "complete_context",
                        "status": "directly_materialized",
                        "missing_evidence_kind": None,
                        "view_ids": ["observations"],
                        "rationale": "The complete observed view directly covers the obligation.",
                    }],
                    "decision": "approve",
                    "issues": [],
                    "required_data_request": None,
                }
                return SimpleNamespace(
                    content=json.dumps({"outcome": payload}),
                    response_metadata={},
                )
            self.chart_prompts.append(messages)
            return SimpleNamespace(content=_strict_test_payload(chart, messages), response_metadata={})

    llm = ProofBundlePlannerLlm()
    result = await VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path)).execute(
        VisualizationInput(message="Show the observed series."), request_state=_state(5),
    )

    verification = result["visualizations"][0]["verification"]
    assert verification["proof_obligations"] == [{
        "obligation_id": "complete_context",
        "description": "The complete observed time series that makes the relationship inspectable.",
        "required": True,
    }]
    assert '"source_ref": "semantic:observations"' in str(llm.chart_prompts[0][0][1])
    assert "Proof evidence bundle" in str(llm.chart_prompts[0][0][1])


@pytest.mark.asyncio
async def test_visualization_repairs_invalid_insight_reference_with_llm_replanning(tmp_path):
    state = _state(25)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_max",
        insight_key="maximum_value",
        name="Maximum value",
        insight_type="extreme",
        statement="The maximum value is 24 at the end of the interval.",
        value=24.0,
        method="sql_query",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
    )]
    chart = {
        "visual_goals": [{
            "purpose": "verify the maximum in context",
            "title": "Maximum in full context",
            "priority": "primary",
            "summary": "The full interval makes the maximum inspectable.",
            "required_roles": ["complete_series"],
            "layers": [{
                "role": "complete_series",
                "source_ref": "view:evidence:evi_full:default",
                "mark": "line",
                "encoding": {"x": "timestamp", "y": "value"},
                "label": "Observed value",
            }],
        }],
        "required_data_request": None,
    }

    class RepairingVerificationLlm:
        def __init__(self):
            self.verification_prompts = []

        async def ainvoke(self, messages):
            if _is_verification_prompt(messages):
                self.verification_prompts.append(messages)
                insight_id = "insight:insight_max" if len(self.verification_prompts) == 1 else "insight_max"
                return SimpleNamespace(content=json.dumps({"outcome": {
                    "decision": "visualize",
                    "target_insight_ids": [insight_id],
                        "verification_question": "Does the full series confirm the maximum?",
                        "interpretation": "Inspect the maximum against all observed values.",
                        "visual_relation": "maximum in complete series",
                        "proof_obligations": [],
                        "required_context": ["complete series"],
                    "non_visual_insight_ids": [],
                    "required_data_request": None,
                }}), response_metadata={})
            if _is_evidence_consumption_prompt(messages):
                return SimpleNamespace(
                    content=_evidence_consumption_payload(messages),
                    response_metadata={},
                )
            if _is_projection_prompt(messages):
                return SimpleNamespace(content=_projection_ir_payload(_projection_for_chart_payload(json.dumps(chart))), response_metadata={})
            if _is_composition_prompt(messages):
                return SimpleNamespace(content=_composition_payload(chart), response_metadata={})
            if _is_encoding_prompt(messages):
                return SimpleNamespace(content=_encoding_payload(chart), response_metadata={})
            if "independently audit" in str(messages[0][1]):
                return SimpleNamespace(content=json.dumps({
                    "decision": "approve",
                    "issues": [],
                    "required_data_request": None,
                }), response_metadata={})
            raise AssertionError("unexpected visualization LLM stage")

    llm = RepairingVerificationLlm()
    result = await VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path)).execute(
        VisualizationInput(message="Show the maximum in context."),
        request_state=state,
    )

    assert result["status"] == "created"
    assert result["visualizations"][0]["verification"]["target_insight_ids"] == ["insight_max"]
    assert len(llm.verification_prompts) == 2
    assert "preceding candidate was rejected" in str(llm.verification_prompts[1][0][1])
    assert "'insight:insight_max'" in str(llm.verification_prompts[1][0][1])


@pytest.mark.asyncio
async def test_explicit_cleaned_trajectory_is_requested_after_goal_and_before_projection(tmp_path):
    state = _state(25)
    anomaly_id = "anomaly_evi_full"
    timestamp = "2026-01-01T05:00:00Z"
    state.anomaly_artifacts[anomaly_id] = AnomalyResult(
        anomaly_id=anomaly_id,
        detector_name="test_detector",
        anomaly_points=[{"timestamp": timestamp, "value": 999.0, "score": 12.0}],
        diagnostics={"detected_count": 1},
    )
    state.insight_set.insights = [KeyInsight(
        insight_id="ins_filtered_target",
        insight_key="filtered_target",
        name="Filtered target",
        insight_type="interval",
        statement="After excluding anomalies, the selected target starts at the excluded point.",
        value={"start_time": timestamp, "start_value": 999.0},
        method="code_interpreter",
        evidence_refs=[
            InsightEvidenceRef(source_type="query", source_id="evi_full"),
            InsightEvidenceRef(source_type="anomaly", source_id=anomaly_id),
        ],
        calculation_trace="Excluded anomaly points, then selected the same remaining point.",
    )]

    class ConsumptionPlannerLlm:
        def __init__(self):
            self.goal_prompts = []
            self.consumption_prompts = []

        async def ainvoke(self, messages):
            if _is_verification_prompt(messages):
                self.goal_prompts.append(messages)
                return SimpleNamespace(content=_strict_test_payload({
                    "decision": "visualize",
                    "target_insight_ids": ["ins_filtered_target"],
                    "verification_question": "Can the target be inspected on a cleaned-only trajectory?",
                    "interpretation": "Show only the exclusion-applied trajectory around the target.",
                    "visual_relation": "target in cleaned-only temporal context",
                    "proof_obligations": [{
                        "obligation_id": "filtered_context",
                        "description": "The transformed context after exclusions.",
                        "required": True,
                    }],
                    "required_context": ["transformed context", "excluded points"],
                    "non_visual_insight_ids": [],
                    "required_data_request": None,
                }, messages), response_metadata={})
            if _is_evidence_consumption_prompt(messages):
                self.consumption_prompts.append(messages)
                return SimpleNamespace(content=json.dumps({
                    "decision": "needs_sources",
                    "rationale": "The explicitly requested cleaned-only trajectory is not materialized.",
                    "source_uses": [],
                    "required_data_request": {
                        "required_action": "code_interpreter",
                        "purpose": "Materialize the exclusion-applied series requested by the presentation goal.",
                        "message": "Produce the cleaned-only transformed context.",
                        "required_shape": "timeseries",
                        "required_fields": ["timestamp", "value", "exclusion_status"],
                        "required_properties": ["anomaly exclusions applied"],
                        "input_evidence": "evi_full",
                        "input_source_refs": ["evidence:evi_full", f"anomaly:{anomaly_id}"],
                        "insight_requests": [{
                            "name": "Cleaned trajectory context",
                            "insight_type": "timeseries_context",
                            "insight_key": "cleaned_context",
                        }],
                    },
                }), response_metadata={})
            raise AssertionError("semantic projection must not run before missing upstream evidence is resolved")

    llm = ConsumptionPlannerLlm()
    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(
        VisualizationInput(
            message="Show the filtered target on the cleaned-only exclusion-applied trajectory.",
            source_refs=["insight:filtered_target"],
        ),
        request_state=state,
    )

    assert result["status"] == "needs_sources"
    assert result["required_data_request"]["required_action"] == "code_interpreter"
    assert result["required_data_request"]["insight_requests"]
    assert len(llm.goal_prompts) == 1
    assert len(llm.consumption_prompts) == 1
    assert "projection_root" not in str(llm.goal_prompts[0])
    consumption_prompt = str(llm.consumption_prompts[0])
    assert "Excluded anomaly points, then selected the same remaining point." in consumption_prompt
    assert timestamp in consumption_prompt
    assert '"value": 999.0' in consumption_prompt


@pytest.mark.asyncio
async def test_visualization_preserves_full_series_and_highlights_interval_inside_broader_viewport(tmp_path):
    state = _state(100)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_reversal",
        insight_key="reversal_interval",
        name="Reversal interval",
        insight_type="interval",
        statement="The series falls and then rises inside the located interval.",
        value={
            "start_time": "2026-01-02T12:00:00Z",
            "end_time": "2026-01-03T12:00:00Z",
            "start_value": 36.0,
            "end_value": 60.0,
        },
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
    )]
    projection = {
        "semantic_views": [
            {
                "view_id": "full_series",
                "name": "Complete observed series",
                "purpose": "Preserve the complete temporal context",
                "grain": "observation",
                "source_ref": "view:evidence:evi_full:default",
                "record_path": None,
                "mode": "records",
                "fields": [
                    {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                    {"name": "measure", "semantic_role": "observed_value", "source_path": "$.value"},
                ],
            },
            {
                "view_id": "target_bounds",
                "name": "Authoritative target boundaries",
                "purpose": "Keep the located Key Insight as the interval boundary source",
                "grain": "target interval",
                "source_ref": "insight:insight_reversal",
                "record_path": None,
                "mode": "records",
                "fields": [
                    {"name": "start", "semantic_role": "interval_start", "source_path": "$.start_time"},
                    {"name": "end", "semantic_role": "interval_end", "source_path": "$.end_time"},
                ],
            },
        ],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "verify the located reversal interval",
            "title": "Reversal in monthly context",
            "priority": "primary",
            "summary": "The full series remains scrollable while the located interval is emphasized.",
            "required_roles": ["complete_series", "highlighted_interval"],
            "show_legend": True,
            "tooltip": "axis",
            "enable_zoom": True,
            "viewport_start": "2026-01-02T00:00:00Z",
            "viewport_end": "2026-01-04T00:00:00Z",
            "y_scale": "linear",
            "layers": [
                {
                    "layer_type": "series", "role": "complete_series", "source_ref": "semantic:full_series",
                    "encodings": [{"channel": "x", "field": "time"}, {"channel": "y", "field": "measure"}],
                    "emphasis": "subtle", "line_style": "solid", "symbol": "none", "axis": "primary",
                    "label": "Observed value",
                },
                {
                    "layer_type": "interval_overlay", "role": "highlighted_interval", "source_ref": "semantic:full_series",
                    "encodings": [{"channel": "x", "field": "time"}, {"channel": "y", "field": "measure"}],
                    "interval_source_ref": "semantic:target_bounds",
                    "interval_start_field": "start",
                    "interval_end_field": "end",
                    "emphasis": "strong", "line_style": "solid", "symbol": "none", "axis": "primary",
                    "label": "Located reversal interval",
                },
            ],
        }],
        "required_data_request": None,
    }
    llm = _TwoStagePlannerLlm(projection=projection, chart=chart)

    store = VisualizationArtifactStore(tmp_path)
    result = await VisualizationTool(
        llm=llm,
        artifact_store=store,
    ).execute(
        VisualizationInput(
            message="Show the full series and highlight the located reversal interval.",
            source_refs=["insight:reversal_interval"],
        ),
        request_state=state,
    )

    visualization = store.get(result["visualization_ids"][0])
    assert visualization is not None
    assert len(visualization.datasets[0].series[0].points) == 100
    assert len(visualization.datasets[1].series[0].points) == 25
    assert "insight:insight_reversal" in visualization.source_refs
    assert visualization.presentation["dataZoom"][0] == {
        "type": "inside",
        "filterMode": "filter",
        "startValue": "2026-01-02T00:00:00Z",
        "endValue": "2026-01-04T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_visualization_expands_wide_turning_record_and_compiles_interval_overlay(tmp_path):
    state = _state(100)
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_turning_window",
        insight_key="turning_window",
        name="Most significant rise-then-fall window",
        insight_type="analysis",
        statement="The located window rises to a peak and then falls.",
        value={
            "start_time": "2026-01-02T12:00:00Z",
            "peak_time": "2026-01-03T00:00:00Z",
            "end_time": "2026-01-03T12:00:00Z",
            "start_price": 36.0,
            "peak_price": 48.0,
            "end_price": 60.0,
        },
        method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_full")],
    )]
    projection = {
        "semantic_views": [
            {
                "view_id": "full_series",
                "name": "Complete observed series",
                "purpose": "Provide full temporal context",
                "grain": "observation",
                "source_ref": "view:evidence:evi_full:default",
                "record_path": "$.records",
                "mode": "records",
                "fields": [
                    {"name": "time", "semantic_role": "observation_time", "source_path": "$.timestamp"},
                    {"name": "price", "semantic_role": "observed_price", "source_path": "$.value"},
                ],
            },
            {
                "view_id": "turning_bounds",
                "name": "Turning window bounds",
                "purpose": "Expose authoritative interval boundaries",
                "grain": "interval",
                "source_ref": "insight:insight_turning_window",
                "record_path": "$.value",
                "mode": "records",
                "fields": [
                    {"name": "start", "semantic_role": "interval_start", "source_path": "$.start_time"},
                    {"name": "end", "semantic_role": "interval_end", "source_path": "$.end_time"},
                ],
            },
            {
                "view_id": "turning_events",
                "name": "Turning window events",
                "purpose": "Expand start, peak, and end into located event rows",
                "grain": "event",
                "source_ref": "insight:insight_turning_window",
                "record_path": "$.value",
                "mode": "events",
                "events": [
                    {"event_role": "start", "timestamp_path": "$.start_time", "value_path": "$.start_price"},
                    {"event_role": "peak", "timestamp_path": "$.peak_time", "value_path": "$.peak_price"},
                    {"event_role": "end", "timestamp_path": "$.end_time", "value_path": "$.end_price"},
                ],
            },
        ],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Verify the turning window in complete context",
            "title": "Rise-then-fall turning window",
            "priority": "primary",
            "summary": "The complete series, selected interval, and located events share one time axis.",
            "required_roles": ["complete_series", "turning_interval", "turning_events"],
            "show_legend": True,
            "tooltip": "axis",
            "enable_zoom": True,
            "viewport_start": "2026-01-02T00:00:00Z",
            "viewport_end": "2026-01-04T00:00:00Z",
            "y_scale": "linear",
            "layers": [
                {
                    "layer_type": "series", "role": "complete_series", "source_ref": "semantic:full_series",
                    "mark": "line", "encodings": [{"channel": "x", "field": "time"}, {"channel": "y", "field": "price"}],
                    "interval_source_ref": None, "interval_start_field": None, "interval_end_field": None,
                    "interval_start_value": None, "interval_end_value": None, "emphasis": "subtle",
                    "line_style": "solid", "symbol": "none", "axis": "primary", "label": "Observed price",
                },
                {
                    "layer_type": "interval_overlay", "role": "turning_interval", "source_ref": "semantic:full_series",
                    "mark": "line", "encodings": [{"channel": "x", "field": "time"}, {"channel": "y", "field": "price"}],
                    "interval_source_ref": "semantic:turning_bounds", "interval_start_field": "start", "interval_end_field": "end",
                    "interval_start_value": None, "interval_end_value": None, "emphasis": "strong",
                    "line_style": "dashed", "symbol": "none", "axis": "primary", "label": "Turning interval",
                },
                {
                    "layer_type": "event_points", "role": "turning_events", "source_ref": "semantic:turning_events",
                    "mark": "point", "encodings": [{"channel": "x", "field": "timestamp"}, {"channel": "y", "field": "value"}, {"channel": "series", "field": "event_role"}],
                    "interval_source_ref": None, "interval_start_field": None, "interval_end_field": None,
                    "interval_start_value": None, "interval_end_value": None, "emphasis": "strong",
                    "line_style": "solid", "symbol": "circle", "axis": "primary", "label": "Turning event",
                },
            ],
        }],
        "required_data_request": None,
    }
    llm = _TwoStagePlannerLlm(projection=projection, chart=chart)
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(
            message="Show the complete series and verify the rise-then-fall turning window.",
            source_refs=["insight:turning_window"],
        ),
        request_state=state,
    )

    visualization = store.get(result["visualization_ids"][0])
    assert visualization is not None
    assert llm.projection_calls == 1
    assert llm.chart_calls == 1
    assert [dataset.row_count for dataset in visualization.datasets] == [100, 25, 3]
    assert visualization.layers[1].transform == [{
        "type": "filter",
        "field": "time",
        "operator": "between",
        "value": ["2026-01-02T12:00:00Z", "2026-01-03T12:00:00Z"],
    }]
    assert {series.role.rsplit(":", 1)[-1] for series in visualization.datasets[2].series} == {
        "start", "peak", "end",
    }
    assert [layer.mark for layer in visualization.layers] == ["line", "line", "point"]


@pytest.mark.asyncio
async def test_long_event_table_stays_row_preserving_and_compiles_to_point_mark(tmp_path):
    bad_projection = {
        "semantic_views": [{
            "view_id": "turning_events",
            "name": "Located turning events",
            "purpose": "Expose already-long event records",
            "grain": "event",
            "source_ref": "view:derived_evidence:dev_endpoints",
            "record_path": "$.records",
            "mode": "wide_events",
            "events": [{
                "event_role": "located_event",
                "timestamp_path": "$.timestamp",
                "value_path": "$.value",
            }],
        }],
        "required_data_request": None,
    }
    repaired_projection = {
        "semantic_views": [{
            "view_id": "turning_events",
            "name": "Located turning events",
            "purpose": "Expose already-long event records",
            "grain": "event",
            "source_ref": "view:derived_evidence:dev_endpoints",
            "record_path": "$.records",
            "mode": "records",
            "fields": [
                {"name": "role", "semantic_role": "event_role", "source_path": "$.role"},
                {"name": "time", "semantic_role": "event_time", "source_path": "$.timestamp"},
                {"name": "value", "semantic_role": "event_value", "source_path": "$.value"},
            ],
        }],
        "required_data_request": None,
    }
    chart = {
        "visual_goals": [{
            "purpose": "Verify located events",
            "title": "Located turning events",
            "priority": "primary",
            "summary": None,
            "required_roles": ["turning_events"],
            "show_legend": True,
            "tooltip": "item",
            "enable_zoom": False,
            "viewport_start": None,
            "viewport_end": None,
            "y_scale": "linear",
            "layers": [{
                "layer_type": "event_points",
                "role": "turning_events",
                "source_ref": "semantic:turning_events",
                "encodings": [
                    {"channel": "x", "field": "time"},
                    {"channel": "y", "field": "value"},
                    {"channel": "label", "field": "role"},
                ],
                "interval_source_ref": None,
                "interval_start_field": None,
                "interval_end_field": None,
                "interval_start_value": None,
                "interval_end_value": None,
                "emphasis": "strong",
                "line_style": "solid",
                "symbol": "diamond",
                "axis": "primary",
                "label": "Turning events",
            }],
        }],
        "required_data_request": None,
    }
    llm = _RepairingTwoStagePlannerLlm(
        projections=[bad_projection, repaired_projection],
        chart=chart,
    )
    store = VisualizationArtifactStore(tmp_path)

    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(
            message="Show the located turning events.",
            source_refs=["view:derived_evidence:dev_endpoints"],
        ),
        request_state=_state_with_analysis_views(),
    )

    visualization = store.get(result["visualization_ids"][0])
    assert visualization is not None
    assert len(llm.projection_prompts) == 2
    assert visualization.datasets[0].row_count == 2
    assert visualization.layers[0].mark == "point"
    assert [point.label for point in visualization.datasets[0].series[0].points] == ["start", "end"]


@pytest.mark.asyncio
async def test_not_visualizable_conclusion_is_closed_without_publishing_fallback(tmp_path):
    llm = _VerificationOnlyLlm({
        "decision": "not_visualizable",
        "target_insight_ids": [],
        "verification_question": None,
        "interpretation": "The requested causal explanation cannot be verified by an observational chart.",
        "visual_relation": None,
        "required_context": [],
        "non_visual_insight_ids": [],
        "required_data_request": None,
    })

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Prove what caused the change."), request_state=_state(25))

    assert result["status"] == "unavailable"
    assert "causal explanation" in result["unavailable_reason"]
    assert result["visualization_ids"] == []
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_completed_visual_dependency_becomes_unavailable_instead_of_looping(tmp_path):
    requirement = {
        "required_action": "sql_query",
        "purpose": "load the missing complete comparison set",
        "message": "Load every comparison candidate.",
        "required_shape": "complete_records",
        "required_fields": ["category", "value"],
        "required_properties": ["complete comparison set"],
        "input_evidence": None,
        "input_source_refs": [],
        "insight_requests": [],
    }
    state = _state(25)
    state.observations = [
        ToolObservation(
            tool_name="visualization",
            success=True,
            summary="Additional evidence is required.",
            payload={"status": "needs_sources", "required_data_request": requirement},
        ),
        ToolObservation(
            tool_name="sql_query",
            success=True,
            summary="The source owner completed.",
            payload={"evidence_id": "evi_full"},
        ),
    ]
    llm = _VerificationOnlyLlm({
        "decision": "needs_sources",
        "target_insight_ids": [],
        "verification_question": None,
        "interpretation": None,
        "visual_relation": "complete comparison",
        "required_context": ["every comparison candidate"],
        "non_visual_insight_ids": [],
        "required_data_request": requirement,
    })

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Compare every candidate."), request_state=state)

    assert result["status"] == "unavailable"
    assert "remained unavailable" in result["unavailable_reason"]
    assert result["required_data_request"] is None


@pytest.mark.asyncio
async def test_disabled_render_audit_does_not_block_candidate_publication(tmp_path):
    initial = (
        '{"visual_goals":[{"purpose":"trend","title":"Ambiguous trend","priority":"primary",'
        '"summary":null,"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":"Value"}]}],"required_data_request":null}'
    )
    repaired = (
        '{"visual_goals":[{"purpose":"trend","title":"Observed value over time","priority":"primary",'
        '"summary":"Read the observed series from the first timestamp to the last.",'
        '"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":"Observed value"}]}],"required_data_request":null}'
    )
    render_auditor = _RenderAuditStub([
        {"decision": "revise", "issues": ["The title does not state what is observed."]},
        {"decision": "approve", "issues": []},
    ])

    result = await VisualizationTool(
        llm=_SequencePlannerLlm([initial, repaired]),
        artifact_store=VisualizationArtifactStore(tmp_path),
        render_auditor=render_auditor,
    ).execute(VisualizationInput(message="Show the observed trend."), request_state=_state(25))

    assert result["status"] == "created"
    assert result["visualizations"][0]["title"] == "Ambiguous trend"
    assert render_auditor.calls == []
    assert [item.name for item in tmp_path.glob("*.json")] == [
        f'{result["visualization_ids"][0]}.json'
    ]


@pytest.mark.asyncio
async def test_candidate_semantic_audit_can_block_publication(tmp_path):
    llm = _PlannerLlm(
        '{"visual_goals":[{"purpose":"trend","title":"Observed trend","priority":"primary",'
        '"summary":null,"required_roles":["series"],"layers":[{"role":"series",'
        '"source_ref":"view:evidence:evi_full:default","mark":"line",'
        '"encoding":{"x":"timestamp","y":"value"},"label":"Observed value"}]}],"required_data_request":null}',
        audit_payload=(
            '{"decision":"unavailable","issues":["Do not publish this candidate"],'
            '"required_data_request":null}'
        ),
    )

    result = await VisualizationTool(
        llm=llm,
        artifact_store=VisualizationArtifactStore(tmp_path),
    ).execute(VisualizationInput(message="Show the observed trend."), request_state=_state(25))

    assert result["status"] == "unavailable"
    assert len(llm.audit_prompts) == 1
    assert result["visualizations"] == []
    assert list(tmp_path.glob("*.json")) == []
