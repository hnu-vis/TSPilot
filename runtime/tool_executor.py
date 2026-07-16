"""Tool execution runtime."""
from __future__ import annotations

from dataclasses import dataclass

from schemas.state import ConversationStateModel, RequestStateModel
from schemas.tool import ToolCall, ToolObservation
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
        validated = tool_spec.input_model.model_validate(action_input)
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
