"""Tool execution runtime."""
from __future__ import annotations

from dataclasses import dataclass
import re

from schemas.state import ConversationStateModel, RequestStateModel
from schemas.tool import ToolCall, ToolObservation
from runtime.action_policy import runtime_action_constraints
from tools.registry import ToolRegistry, ToolSpec


@dataclass
class ExecutionResult:
    """Full execution result."""

    tool_spec: ToolSpec
    observation: ToolObservation
    full_payload: dict


class ToolExecutor:
    """Resolve, validate, and invoke one tool."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def execute(
        self,
        action_name: str,
        action_input: dict,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
        action_reason: str | None = None,
    ) -> ExecutionResult:
        tool_spec = self._registry.resolve(action_name)
        normalized_input = self._normalize_action_input(action_name, action_input, request_state)
        validated = tool_spec.input_model.model_validate(normalized_input)
        request_state.tool_history.append(
            ToolCall(
                tool_name=action_name,
                tool_input=validated.model_dump(mode="json"),
                iteration=request_state.iteration,
                reason=action_reason,
            )
        )

        request_state_view = request_state.model_copy(deep=True)
        conversation_state_view = conversation_state.model_copy(deep=True)
        if tool_spec.runtime_access == "none":
            raw_result = await tool_spec.tool.execute(validated)
        elif tool_spec.runtime_access == "request_state_read":
            raw_result = await tool_spec.tool.execute(validated, request_state=request_state_view)
        else:
            raw_result = await tool_spec.tool.execute(
                validated,
                request_state=request_state_view,
                conversation_state=conversation_state_view,
            )

        full_payload = tool_spec.output_model.model_validate(raw_result).model_dump(mode="json")
        visible_payload, truncated = self._truncate_payload(full_payload, request_state)
        observation = ToolObservation(
            tool_name=action_name,
            success=True,
            summary=tool_spec.tool.summarize(full_payload),
            payload=visible_payload,
            error=None,
            payload_truncated=truncated,
            payload_ref=(
                f"obs:{request_state.request_id}:{request_state.iteration}:{action_name}"
                if truncated
                else None
            ),
        )
        return ExecutionResult(tool_spec=tool_spec, observation=observation, full_payload=full_payload)

    def _normalize_action_input(self, action_name: str, action_input: dict, request_state: RequestStateModel) -> dict:
        normalized = dict(action_input or {})
        if action_name == "terminate":
            for key in (
                "include_analysis_ids",
                "include_fact_ids",
                "include_visualization_ids",
                "section_plan",
                "unavailable_outputs",
            ):
                if normalized.get(key) in (None, False, ""):
                    normalized[key] = []
            for key in ("result", "direct_answer", "summary_goal", "unavailable_reason"):
                if isinstance(normalized.get(key), (dict, list)):
                    import json

                    normalized[key] = json.dumps(normalized[key], ensure_ascii=False)
        if "time_range" in normalized:
            normalized["time_range"] = self._normalize_time_range_hint(normalized.get("time_range"), normalized)
        normalized["fact_requests"] = self._normalize_fact_requests(normalized.get("fact_requests"), normalized)
        normalized = self._apply_runtime_input_guidance(action_name, normalized, request_state)
        if action_name == "anomaly":
            if str(normalized.get("detector_name") or "").strip().lower() in {"default", "auto", "none", "null"}:
                normalized.pop("detector_name", None)
        if action_name == "forecast":
            if str(normalized.get("model_name") or "").strip().lower() in {"default", "auto", "none", "null"}:
                normalized.pop("model_name", None)
        if action_name == "code_interpreter" and not str(normalized.get("code") or "").strip():
            normalized.setdefault("database_evidence", "latest")
            normalized.setdefault("analysis_goal", request_state.message)
            normalized.setdefault(
                "analysis_request",
                {
                    "goal": request_state.message,
                    "required_outputs": self._contract_required_outputs(request_state),
                    "mode": "canonical_timeseries_metrics",
                },
            )
            if not normalized.get("required_outputs"):
                normalized["required_outputs"] = self._contract_required_outputs(request_state)
        state_database_context = request_state.database_context
        if state_database_context is not None:
            state_payload = state_database_context.model_dump(mode="json")
            raw_context = normalized.get("database_context")
            if isinstance(raw_context, dict):
                normalized["database_context"] = {
                    **state_payload,
                    **{key: value for key, value in raw_context.items() if value not in (None, "", [], {})},
                }
            elif "database_context" not in normalized or raw_context in (None, "", {}, []):
                normalized["database_context"] = state_payload

        return normalized

    def _apply_runtime_input_guidance(self, action_name: str, normalized: dict, request_state: RequestStateModel) -> dict:
        constraints = runtime_action_constraints(request_state)
        for item in constraints.get("required_actions", []) or []:
            if not isinstance(item, dict) or item.get("action") != action_name:
                continue
            guidance = item.get("input_guidance") if isinstance(item.get("input_guidance"), dict) else {}
            guidance_constraints = guidance.get("constraints") if isinstance(guidance.get("constraints"), dict) else {}
            if guidance_constraints:
                merged_constraints = dict(normalized.get("constraints") or {})
                merged_constraints.update(guidance_constraints)
                normalized["constraints"] = merged_constraints
            if action_name in {"forecast", "anomaly", "code_interpreter"} and not normalized.get("database_evidence"):
                normalized["database_evidence"] = guidance.get("database_evidence") or "latest"
            if action_name == "code_interpreter" and isinstance(guidance.get("analysis_request"), dict):
                normalized.setdefault("analysis_request", guidance["analysis_request"])
                required_outputs = guidance["analysis_request"].get("required_outputs")
                if isinstance(required_outputs, list) and not normalized.get("required_outputs"):
                    normalized["required_outputs"] = required_outputs
                if not self._user_explicitly_requested_custom_code(request_state):
                    normalized.pop("code", None)
            break
        if action_name == "sql_query":
            merged_constraints = dict(normalized.get("constraints") or {})
            if self._latest_sql_shape_requested_raw(request_state):
                merged_constraints["evidence_shape"] = "raw_timeseries"
                merged_constraints["dialect_complexity_policy"] = "simple_raw_evidence"
            if merged_constraints:
                normalized["constraints"] = merged_constraints
        return normalized

    def _user_explicitly_requested_custom_code(self, request_state: RequestStateModel) -> bool:
        text = str(request_state.message or "").lower()
        return any(token in text for token in ("python", "代码", "code", "script", "脚本"))

    def _latest_sql_shape_requested_raw(self, request_state: RequestStateModel) -> bool:
        latest = request_state.observations[-1] if request_state.observations else None
        if latest is None or latest.tool_name != "sql_query" or latest.success:
            return False
        payload = latest.payload if isinstance(latest.payload, dict) else {}
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else payload
        issues = diagnostics.get("query_shape_issues") if isinstance(diagnostics.get("query_shape_issues"), list) else []
        return any(
            isinstance(issue, dict) and issue.get("recommended_shape") == "raw_series"
            for issue in issues
        )

    def _contract_required_outputs(self, request_state: RequestStateModel) -> list[str]:
        contract = request_state.task_contract
        if contract is not None:
            outputs = getattr(contract, "required_outputs", []) or []
            names = []
            for output in outputs:
                value = getattr(output, "id", None) or getattr(output, "description", None) or output
                if str(value or "").strip():
                    names.append(str(value).strip())
            if names:
                return names
        profile = request_state.intent_profile if isinstance(request_state.intent_profile, dict) else {}
        raw_outputs = profile.get("required_outputs") if isinstance(profile.get("required_outputs"), list) else []
        return [str(item).strip() for item in raw_outputs if str(item).strip()]

    def _normalize_time_range_hint(self, value, normalized_input: dict):
        if value in (None, "", [], {}):
            return None
        if isinstance(value, dict):
            normalized = dict(value)
            if "stop" in normalized and "end" not in normalized:
                normalized["end"] = normalized.pop("stop")
            return normalized
        if isinstance(value, str):
            text = value.strip()
            iso_range = re.match(r"^([^/]+)\s*/\s*([^/]+)$", text)
            if iso_range:
                return {"start": iso_range.group(1).strip(), "end": iso_range.group(2).strip()}
            constraints = normalized_input.setdefault("constraints", {})
            if isinstance(constraints, dict):
                constraints.setdefault("time_range_text", text)
            return None
        return None

    def _normalize_fact_requests(self, value, normalized_input: dict) -> list:
        if value in (None, "", False):
            return []
        if not isinstance(value, list):
            constraints = normalized_input.setdefault("constraints", {})
            if isinstance(constraints, dict):
                constraints.setdefault("fact_request_hints", []).append(value)
            return []
        normalized: list = []
        hints: list = []
        for item in value:
            if isinstance(item, dict):
                if item.get("name") and item.get("fact_type"):
                    normalized.append(item)
                else:
                    hints.append(item)
                continue
            if isinstance(item, str) and item.strip():
                hints.append(item.strip())
        if hints:
            constraints = normalized_input.setdefault("constraints", {})
            if isinstance(constraints, dict):
                existing = constraints.setdefault("fact_request_hints", [])
                if isinstance(existing, list):
                    existing.extend(hints)
        return normalized

    def _truncate_payload(self, payload: dict, request_state: RequestStateModel) -> tuple[dict, bool]:
        max_chars = int(request_state.context_budget.get("max_observation_chars", 1600))
        rendered = str(payload)
        if len(rendered) <= max_chars:
            return payload, False

        visible = dict(payload)
        if isinstance(visible.get("fact_candidates"), list):
            visible["fact_candidates"] = visible["fact_candidates"][:8]
        if isinstance(visible.get("completed_facts"), list):
            visible["completed_facts"] = visible["completed_facts"][:6]
        if isinstance(visible.get("verified_facts"), list):
            visible["verified_facts"] = visible["verified_facts"][:6]
        if isinstance(visible.get("rejected_facts"), list):
            visible["rejected_facts"] = visible["rejected_facts"][:6]
        if isinstance(visible.get("forecast_points"), list):
            visible["forecast_points"] = visible["forecast_points"][:12]
        if isinstance(visible.get("anomaly_points"), list):
            visible["anomaly_points"] = visible["anomaly_points"][:12]
        if isinstance(visible.get("scores"), list):
            visible["scores"] = visible["scores"][:12]
        if isinstance(visible.get("result"), dict):
            result = dict(visible["result"])
            if isinstance(result.get("details"), list):
                result["details"] = result["details"][:12]
            if isinstance(result.get("rows"), list):
                result["rows"] = result["rows"][:12]
            visible["result"] = result
        if isinstance(visible.get("data"), dict):
            data = dict(visible["data"])
            if isinstance(data.get("points"), list):
                limit = int(request_state.context_budget.get("max_visible_points", 240))
                data["points"] = data["points"][:limit]
            if isinstance(data.get("rows"), list):
                limit = int(request_state.context_budget.get("max_visible_rows", 60))
                data["rows"] = data["rows"][:limit]
            if isinstance(data.get("series"), list):
                data["series"] = [
                    self._summarize_series_for_observation(series)
                    for series in data["series"][:6]
                    if isinstance(series, dict)
                ]
            visible["data"] = data
        if isinstance(visible.get("visualizations"), list):
            summarized_visualizations = []
            for item in visible["visualizations"][:4]:
                if not isinstance(item, dict):
                    continue
                copy_item = dict(item)
                chart = copy_item.get("chart")
                if isinstance(chart, dict):
                    x_axis = list(chart.get("x_axis_data") or [])
                    series = list(chart.get("series_data") or [])
                    copy_item["chart"] = {
                        "x_axis_count": len(x_axis),
                        "series": [
                            {
                                "name": series_item.get("name"),
                                "points_count": len(series_item.get("data") or []),
                            }
                            for series_item in series[:4]
                            if isinstance(series_item, dict)
                        ],
                    }
                if isinstance(copy_item.get("annotations"), list):
                    copy_item["annotations"] = copy_item["annotations"][:12]
                summarized_visualizations.append(copy_item)
            visible["visualizations"] = summarized_visualizations
        if isinstance(visible.get("diagnostics"), dict):
            diagnostics = dict(visible["diagnostics"])
            visible["diagnostics"] = {
                key: value
                for key, value in diagnostics.items()
                if key in {"artifact_kind", "artifact_ref", "snapshot_ref", "threshold", "series_name", "query_trace", "runtime_ms", "sandbox"}
            }
        return visible, True

    def _summarize_series_for_observation(self, series: dict) -> dict:
        item = {
            key: value
            for key, value in series.items()
            if key not in {"points", "rows"}
        }
        points = series.get("points")
        if isinstance(points, list):
            item["points_count"] = len(points)
            item["points"] = _sample_edges(points, limit=12)
        rows = series.get("rows")
        if isinstance(rows, list):
            item["rows_count"] = len(rows)
            item["rows"] = _sample_edges(rows, limit=12)
        return item


def _sample_edges(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return [*items[:head], *items[-tail:]]
