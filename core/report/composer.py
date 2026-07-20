"""Report-style answer composition helpers."""
from __future__ import annotations

from schemas.output import AnswerSection
from schemas.state import RequestStateModel


def missing_requirements(request_state: RequestStateModel) -> list[str]:
    missing: list[str] = []
    latest_evidence = request_state.latest_database_evidence
    has_explicit_sql_query_evidence = bool(
        latest_evidence is not None
        and isinstance(latest_evidence.metadata, dict)
        and latest_evidence.metadata.get("sql_query_mode") == "explicit"
    )
    has_database_answer_evidence = bool(
        latest_evidence is not None
        and latest_evidence.result_type in {"statistics", "table", "schema", "metric_list"}
    )
    for requirement in request_state.answer_requirements:
        if requirement in {"plan", "conclusion"}:
            continue
        if has_explicit_sql_query_evidence or (requirement == "analysis" and has_database_answer_evidence):
            continue
        if not request_state.answer_coverage.get(requirement, False):
            missing.append(requirement)
    return missing


def build_summary(request_state: RequestStateModel, facts: list, fallback: str) -> str:
    subject = subject_label(request_state)
    parts: list[str] = []

    trend_fact = next((fact for fact in facts if fact.fact_type == "trend"), None)
    extrema_fact = next((fact for fact in facts if fact.fact_type in {"extreme", "extrema"}), None)

    if trend_fact is not None:
        parts.append(trend_fact.statement)
    elif facts:
        parts.append(facts[0].statement)
    elif request_state.latest_database_evidence is not None:
        parts.append(request_state.latest_database_evidence.summary)

    if extrema_fact is not None:
        parts.append(extrema_fact.statement)

    if request_state.latest_anomaly is not None:
        anomaly_points = request_state.latest_anomaly.anomaly_points
        if anomaly_points:
            preview = ", ".join(
                f"{point.get('timestamp')}={point.get('value')}"
                for point in anomaly_points[:3]
            )
            parts.append(
                f"异常检测发现 {len(anomaly_points)} 个异常点，典型异常包括 {preview}。"
            )
        else:
            parts.append("异常检测未发现显著异常点。")

    if request_state.latest_forecast is not None:
        forecast_points = request_state.latest_forecast.forecast_points
        if forecast_points:
            first_point = forecast_points[0]
            last_point = forecast_points[-1]
            direction = trend_direction(first_point.value, last_point.value)
            parts.append(
                f"{subject} 的短期预测共 {len(forecast_points)} 个点，预测区间内整体{direction}，"
                f"从 {first_point.value:.2f} 变化到 {last_point.value:.2f}。"
            )

    compact = " ".join(part.strip() for part in parts if part and part.strip())
    return compact or fallback


def build_forecast_section(forecast) -> AnswerSection:
    forecast_points = forecast.forecast_points
    if not forecast_points:
        content = f"已执行预测模型 {forecast.model_name}，但未返回预测点。"
    else:
        first_point = forecast_points[0]
        last_point = forecast_points[-1]
        direction = trend_direction(first_point.value, last_point.value)
        preview = "\n".join(
            f"- {point.timestamp}: {point.value:.2f}" for point in forecast_points[:5]
        )
        content = (
            f"使用 {forecast.model_name} 生成了 {len(forecast_points)} 个短期预测点。"
            f" 预测走势整体{direction}，首个预测值为 {first_point.value:.2f}，"
            f"最后一个预测值为 {last_point.value:.2f}。\n"
            f"预测点预览：\n{preview}"
        )
    return AnswerSection(
        section_type="forecast",
        heading="Forecast",
        content=content,
        structured_payload={"forecast_id": forecast.forecast_id, "horizon": forecast.horizon},
    )


def build_anomaly_section(anomaly) -> AnswerSection:
    anomaly_points = anomaly.anomaly_points
    if not anomaly_points:
        content = f"使用 {anomaly.detector_name} 未检测到显著异常点。"
    else:
        preview = "\n".join(
            f"- {point.get('timestamp')}: value={point.get('value')}, score={point.get('score')}"
            for point in anomaly_points[:5]
        )
        content = (
            f"使用 {anomaly.detector_name} 检测到 {len(anomaly_points)} 个异常点。"
            f" 典型异常如下：\n{preview}"
        )
    return AnswerSection(
        section_type="anomaly",
        heading="Anomaly Detection",
        content=content,
        structured_payload={"anomaly_id": anomaly.anomaly_id, "detector": anomaly.detector_name},
    )


def ordered_sections(sections_by_type: dict[str, AnswerSection], section_plan: list[str]) -> list[AnswerSection]:
    ordered: list[AnswerSection] = []
    seen: set[str] = set()
    for section_type in section_plan:
        section = sections_by_type.get(section_type)
        if section is not None and section.section_type not in seen:
            ordered.append(section)
            seen.add(section.section_type)
    default_order = [
        "summary",
        "query",
        "plan",
        "analysis",
        "facts",
        "statistics",
        "metric_list",
        "schema",
        "table",
        "anomaly",
        "forecast",
        "rag",
        "skill",
        "conclusion",
    ]
    for section_type in default_order:
        section = sections_by_type.get(section_type)
        if section is not None and section.section_type not in seen:
            ordered.append(section)
            seen.add(section.section_type)
    for section_type, section in sections_by_type.items():
        if section_type not in seen:
            ordered.append(section)
            seen.add(section_type)
    return ordered


def subject_label(request_state: RequestStateModel) -> str:
    evidence = request_state.latest_database_evidence
    if evidence is not None:
        data = evidence.data or {}
        if isinstance(data, dict):
            if data.get("value_field"):
                return str(data["value_field"])
            if data.get("series_name"):
                return str(data["series_name"])
        if evidence.columns and len(evidence.columns) > 1:
            return str(evidence.columns[1])
    return "该指标"


def trend_direction(start_value: float, end_value: float) -> str:
    if end_value > start_value:
        return "上升"
    if end_value < start_value:
        return "下降"
    return "基本持平"
