"""Action-scoped output selection for runtime tool routing."""
from __future__ import annotations

import re
from typing import Any

from core.completion import current_todo
from core.harness import default_capability_registry
from schemas.state import RequestStateModel


def select_outputs_for_action(
    request_state: RequestStateModel,
    action_name: str,
    *,
    fallback_outputs: list[Any] | None = None,
    query_task_contract: dict | None = None,
) -> dict:
    """Return the minimal output contract that the requested action may satisfy."""

    registry = default_capability_registry()
    action = str(action_name or "").strip().lower()
    action_capability = registry.capability_for_action(action)
    if not action_capability:
        return {"goal": request_state.message, "required_outputs": [], "missing": []}

    active = current_todo(request_state)
    if isinstance(active, dict):
        task_type = str(active.get("task_type") or "").strip().lower()
        if task_type and task_type != "answer" and not registry.action_matches_task_type(action, task_type):
            return {"goal": request_state.message, "required_outputs": [], "missing": []}
        content = str(active.get("content") or "").strip()
        if content and registry.action_matches_task_type(action, task_type):
            return {
                "goal": content,
                "required_outputs": [content],
                "missing": [content],
                **({"query_task_contract": query_task_contract} if isinstance(query_task_contract, dict) else {}),
            }

    contract_outputs = _contract_outputs_for_action(request_state, action_capability)
    if contract_outputs:
        labels = [_output_label(output) for output in contract_outputs]
        labels = [label for label in labels if label]
        return {
            "goal": request_state.message,
            "required_outputs": labels,
            "missing": labels,
            **({"query_task_contract": query_task_contract} if isinstance(query_task_contract, dict) else {}),
        }

    fallback = [
        _output_label(item)
        for item in (fallback_outputs or [])
        if _output_matches_action(item, action_capability)
    ]
    fallback = _dedupe([label for label in fallback if label])
    return {
        "goal": request_state.message,
        "required_outputs": fallback,
        "missing": fallback,
        **({"query_task_contract": query_task_contract} if isinstance(query_task_contract, dict) else {}),
    }


def _contract_outputs_for_action(request_state: RequestStateModel, action_capability: str) -> list[Any]:
    contract = request_state.task_contract
    outputs = getattr(contract, "required_outputs", []) if contract is not None else []
    selected = []
    for output in outputs or []:
        if not getattr(output, "required", True):
            continue
        if _contract_output_is_covered(request_state, output):
            continue
        if _output_matches_action(output, action_capability):
            selected.append(output)
    return selected


def _contract_output_is_covered(request_state: RequestStateModel, output: Any) -> bool:
    capabilities = _output_capabilities(output)
    if "query" in capabilities and request_state.latest_database_evidence is not None:
        return True
    if "analysis" in capabilities and request_state.latest_analysis_id is not None:
        return True
    if "anomaly" in capabilities and request_state.latest_anomaly is not None:
        return True
    if "forecast" in capabilities:
        points = getattr(request_state.latest_forecast, "forecast_points", None)
        return isinstance(points, list) and bool(points)
    if "external_knowledge" in capabilities and request_state.latest_rag is not None:
        return True
    if "skill" in capabilities and request_state.latest_skill is not None:
        return True
    if "visualization" in capabilities and request_state.visualizations:
        return True
    if "answer" in capabilities and request_state.final_answer_draft is not None:
        return True
    return False


def _output_matches_action(output: Any, action_capability: str) -> bool:
    capabilities = _output_capabilities(output)
    if not capabilities:
        return action_capability == "analysis" and _looks_like_analysis_output(output)
    return action_capability in capabilities


def _output_capabilities(output: Any) -> set[str]:
    registry = default_capability_registry()
    raw_values = []
    if isinstance(output, dict):
        raw_values.extend(
            output.get(key)
            for key in ("evidence_kind", "output_type", "kind", "task_type")
            if output.get(key) not in (None, "", [], {})
        )
    elif not isinstance(output, str):
        raw_values.extend(
            getattr(output, key, None)
            for key in ("evidence_kind", "output_type")
            if getattr(output, key, None) not in (None, "", [], {})
        )
    capabilities: set[str] = set()
    for value in raw_values:
        for part in _split_kind_value(value):
            normalized = registry.normalize_id(part)
            if normalized == "conclusion":
                normalized = "answer"
            if normalized in {"database_or_analysis", "raw_or_analysis"}:
                capabilities.update({"query", "analysis"})
            elif normalized in {"computed", "calculated"}:
                capabilities.add("analysis")
            elif normalized in {"database_evidence", "database", "sql", "raw"}:
                capabilities.add("query")
            elif normalized in {"derived", "statistical"}:
                capabilities.add("analysis")
            elif normalized:
                capabilities.add(normalized)
    if not capabilities:
        capabilities.update(_infer_capabilities_from_text(_output_text(output)))
    return {item for item in capabilities if item}


def _split_kind_value(value: Any) -> list[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    return [part for part in re.split(r"(?:_or_|/|\||,|\s+or\s+)", text) if part]


def _looks_like_analysis_output(output: Any) -> bool:
    return "analysis" in _infer_capabilities_from_text(_output_text(output))


def _infer_capabilities_from_text(text: str) -> set[str]:
    value = str(text or "").strip().lower()
    if not value:
        return set()
    capability_terms = {
        "forecast": ("forecast", "prediction", "predict", "projection", "未来", "预测", "预估"),
        "anomaly": ("anomaly", "outlier", "spike", "异常", "离群", "突增", "突降"),
        "analysis": (
            "analysis",
            "analyze",
            "metric",
            "statistics",
            "statistical",
            "compute",
            "calculate",
            "derived",
            "trend",
            "ratio",
            "volatility",
            "drawdown",
            "计算",
            "指标",
            "统计",
            "分析",
            "趋势",
            "波动",
            "回撤",
            "差值",
        ),
        "query": ("query", "evidence", "data", "rows", "points", "fetch", "load", "查询", "数据", "记录", "序列"),
        "visualization": ("visualization", "visual", "chart", "plot", "graph", "可视化", "图表", "曲线"),
        "answer": ("conclusion", "answer", "summary", "final", "结论", "回答", "总结", "汇总"),
    }
    inferred = set()
    for capability, terms in capability_terms.items():
        if any(term in value for term in terms):
            inferred.add(capability)
    return inferred


def _output_label(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        for key in ("description", "id", "name", "output_type", "evidence_kind"):
            value = output.get(key)
            if str(value or "").strip():
                return str(value).strip()
        return ""
    for key in ("description", "id", "output_type", "evidence_kind"):
        value = getattr(output, key, None)
        if str(value or "").strip():
            return str(value).strip()
    return str(output or "").strip()


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        values = []
        for key in ("id", "description", "name", "output_type", "evidence_kind", "success_criteria"):
            if output.get(key) not in (None, "", [], {}):
                values.append(str(output.get(key)))
        for key in ("measures", "dimensions"):
            raw = output.get(key)
            if isinstance(raw, list):
                values.extend(str(item) for item in raw)
        return " ".join(values)
    values = []
    for key in ("id", "description", "output_type", "evidence_kind", "success_criteria"):
        value = getattr(output, key, None)
        if value not in (None, "", [], {}):
            values.append(str(value))
    for key in ("measures", "dimensions"):
        raw = getattr(output, key, None)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
    return " ".join(values)


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
