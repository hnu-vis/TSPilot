"""Forecast tool placeholder."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
from math import ceil
import re
from statistics import median
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import field_validator

from core.database.dialects import dialect_for_database
from core.timeseries.evidence_resolution import resolve_database_evidence
from core.timeseries.forecast_registry import default_forecast_model_name, get_forecast_model
from core.timeseries.normalization import normalize_timeseries_evidence
from runtime.llm_trace import llm_trace_span
from runtime.timeout_policy import load_timeout_policy
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsightRequest
from schemas.timeseries import ForecastPlan, ForecastResult, TimeSeriesSeries
from tools.base import BaseTool, StructuredToolError


DEFAULT_MAX_DIRECT_STEPS = 48
_DURATION_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "seconds": 1,
    "秒": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "minutes": 60,
    "分钟": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "hours": 3600,
    "小时": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
    "天": 86400,
    "日": 86400,
    "w": 604800,
    "week": 604800,
    "weeks": 604800,
    "周": 604800,
    "mo": 2592000,
    "month": 2592000,
    "months": 2592000,
    "月": 2592000,
    "y": 31536000,
    "yr": 31536000,
    "year": 31536000,
    "years": 31536000,
    "年": 31536000,
}
_DURATION_UNIT_PATTERN = "|".join(
    sorted((re.escape(unit) for unit in _DURATION_UNIT_SECONDS), key=len, reverse=True)
)
_DURATION_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_DURATION_UNIT_PATTERN})", re.IGNORECASE)


class ForecastInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    horizon: int | str | dict | None = None
    model_name: str | None = None
    series_name: str | None = None
    constraints: dict | None = Field(default_factory=dict)
    insight_requests: list[KeyInsightRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_insight_contracts(self):
        if self.insight_requests:
            raise ValueError("forecast returns an analysis artifact; Key Insight requests must target sql_query or code_interpreter")
        return self

    @field_validator("horizon", mode="before")
    @classmethod
    def normalize_horizon(cls, value):
        if isinstance(value, dict):
            for key in ("steps", "horizon", "points", "count"):
                if key in value:
                    return cls.normalize_horizon(value[key])
            return value
        if isinstance(value, str):
            if _parse_duration_seconds(value) is not None:
                return value
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return value


class _ForecastInputAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe_to_forecast: bool
    reason: str = Field(min_length=1)
    quality_issues: list[str] = Field(default_factory=list)


class ForecastTool(BaseTool):
    def __init__(
        self,
        llm=None,
        *,
        llm_timeout_seconds: float | None = None,
        external_request_timeout_seconds: float | None = None,
    ):
        policy = load_timeout_policy().tool("forecast")
        self._llm = llm
        self._llm_timeout_seconds = float(
            llm_timeout_seconds
            if llm_timeout_seconds is not None
            else policy.stage_seconds("llm_call_seconds")
        )
        self._external_request_timeout_seconds = float(
            external_request_timeout_seconds
            if external_request_timeout_seconds is not None
            else policy.stage_seconds("external_request_seconds")
        )

    async def execute(self, validated_input: ForecastInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = resolve_database_evidence(database_evidence, request_state, tool_label="Forecast")
        if database_evidence is None:
            raise ValueError("Forecast requires database_evidence or a latest_database_evidence in request state.")
        constraints = validated_input.constraints or {}
        preferred_series = validated_input.series_name or constraints.get("series_name")
        try:
            series = normalize_timeseries_evidence(
                database_evidence,
                series_name=preferred_series,
                value_field=preferred_series,
            )
        except ValueError as exc:
            raise _insufficient_timeseries_evidence_error(
                message=str(exc),
                tool_name="forecast",
                evidence=database_evidence,
            ) from exc
        series, input_policy_diagnostics = _apply_forecast_input_policy(
            series=series,
            database_evidence=database_evidence,
            request_state=request_state,
            constraints=constraints,
        )
        # The semantic gate owns dependency routing for raw forecast inputs. Once a
        # matching anomaly artifact has been applied, the specialized anomaly tool
        # owns point selection and the forecaster consumes that filtered contract;
        # asking the same gate to veto it again creates an unrepairable
        # anomaly/forecast loop rather than a new source dependency.
        if self._llm is not None and not input_policy_diagnostics.get("source_anomaly_id"):
            assessment = await self._assess_input_quality(
                series=series,
                evidence=database_evidence,
                policy_diagnostics=input_policy_diagnostics,
                response_language=getattr(request_state, "response_language", "en"),
            )
            if not assessment.safe_to_forecast:
                raise _semantic_input_quality_error(
                    assessment=assessment,
                    evidence=database_evidence,
                    anomaly_already_applied=bool(input_policy_diagnostics.get("source_anomaly_id")),
                )
        quality = _validate_forecast_evidence_quality(database_evidence, series, request_state, constraints)
        forecast_plan = _resolve_forecast_plan(validated_input, constraints, series)
        horizon = forecast_plan.requested_steps
        model_name = validated_input.model_name or constraints.get("model_name") or default_forecast_model_name()
        model = get_forecast_model(model_name)
        if forecast_plan.mode == "rolling":
            model_output = await asyncio.wait_for(
                asyncio.to_thread(
                    _rolling_forecast,
                    model=model,
                    series=series,
                    horizon=horizon,
                    chunk_steps=forecast_plan.recommended_chunk_steps or forecast_plan.max_direct_steps,
                    params=constraints,
                ),
                timeout=self._external_request_timeout_seconds,
            )
        else:
            model_output = await asyncio.wait_for(
                asyncio.to_thread(model.forecast, series, horizon=horizon, params=constraints),
                timeout=self._external_request_timeout_seconds,
            )
        forecast_points = model_output.forecast_points
        return ForecastResult(
            forecast_id=f"forecast_{database_evidence.evidence_id}",
            model_name=model.name,
            horizon=horizon,
            status="succeeded",
            forecast_plan=forecast_plan,
            forecast_points=forecast_points,
            confidence_interval=model_output.confidence_interval,
            diagnostics={
                **model_output.diagnostics,
                "series_name": series.series_name,
                "model_registry_name": model.name,
                "horizon": horizon,
                "forecast_plan": forecast_plan.model_dump(mode="json"),
                "input_quality": quality,
                "coverage": {
                    "input_evidence_refs": [database_evidence.evidence_id],
                    "covered_outputs": ["forecast_points", "forecast_horizon", "input_database_evidence"],
                    "missing_outputs": [],
                    "row_count_semantics": "forecast_points",
                    "training_point_count": len(series.points),
                    "forecast_point_count": len(forecast_points),
                    "horizon": horizon,
                    "can_answer": bool(forecast_points and len(forecast_points) >= horizon),
                },
                **input_policy_diagnostics,
            },
        ).model_dump(mode="json")

    async def _assess_input_quality(
        self,
        *,
        series: TimeSeriesSeries,
        evidence: DatabaseEvidence,
        policy_diagnostics: dict,
        response_language: str,
    ) -> _ForecastInputAssessment:
        points = [point.model_dump(mode="json") for point in series.points]
        if len(points) > 16:
            preview = {
                "start_window": points[:8],
                "end_window": points[-8:],
            }
            sample_layout = (
                "Two separate chronological edge windows. Adjacency exists only within each window; "
                "the last start_window point and first end_window point are not adjacent."
            )
        else:
            preview = {"complete_series": points}
            sample_layout = "The complete chronological series; consecutive records are adjacent."
        payload = {
            "evidence_id": evidence.evidence_id,
            "series_name": series.series_name,
            "point_count": len(points),
            "ordered_samples": preview,
            "sample_layout": sample_layout,
            "input_policy": policy_diagnostics,
        }
        system = (
            "Assess whether a time-series input is semantically credible enough for a specialized forecast model. "
            "Return one schema-valid object. Judge discontinuities, impossible scale changes, corruption-like values, and whether "
            "a few points visibly dominate the series. Do not forecast, clean, clip, replace, or calculate user conclusions. "
            "The payload can contain two distant edge windows: never treat the gap between those windows or the difference between "
            "their endpoint levels as an adjacent discontinuity. A gradual level change over a long time span is not corruption. "
            "Judge local transitions only where the sample_layout says points are adjacent. If a matching anomaly artifact has "
            "already been applied, assess the filtered series and do not demand the same anomaly step again. Reject that filtered "
            "series only when the visible adjacent points still contain a concrete residual corruption pattern, and identify those "
            "points in the reason. safe_to_forecast must be false when visible evidence is clearly contaminated enough to make the "
            "model output misleading; explain the semantic quality issue concisely."
        )
        messages = [("system", system), ("human", json.dumps(payload, ensure_ascii=False, default=str))]
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                async with llm_trace_span(
                    "Forecast Assessment Repair" if attempt else "Forecast Input Assessment",
                    summary=(
                        "修正预测输入质量评估"
                        if response_language == "zh" and attempt
                        else "评估时序输入是否适合预测"
                        if response_language == "zh"
                        else "Repair the forecast input assessment"
                        if attempt
                        else "Assess whether the time series is suitable for forecasting"
                    ),
                    messages=messages,
                ) as trace_span:
                    if hasattr(self._llm, "with_structured_output"):
                        runnable = self._llm.with_structured_output(
                            _ForecastInputAssessment, method="json_schema", include_raw=True,
                        )
                        bundle = await asyncio.wait_for(
                            runnable.ainvoke(messages), timeout=self._llm_timeout_seconds
                        )
                        if isinstance(bundle, dict):
                            trace_response = bundle.get("raw")
                            if trace_span is not None:
                                trace_span.attach_response(
                                    trace_response,
                                    messages=messages,
                                    output_text=str(getattr(trace_response, "content", trace_response) or ""),
                                )
                            parsed = bundle.get("parsed")
                            if parsed is None:
                                raise ValueError(
                                    bundle.get("parsing_error")
                                    or "forecast input assessment was not parsed"
                                )
                        else:
                            parsed = bundle
                            if trace_span is not None:
                                trace_span.attach_response(
                                    bundle,
                                    messages=messages,
                                    output_text=str(getattr(bundle, "content", bundle) or ""),
                                )
                        return (
                            parsed
                            if isinstance(parsed, _ForecastInputAssessment)
                            else _ForecastInputAssessment.model_validate(parsed)
                        )
                    response = await asyncio.wait_for(
                        self._llm.ainvoke(messages), timeout=self._llm_timeout_seconds
                    )
                    if trace_span is not None:
                        trace_span.attach_response(
                            response,
                            messages=messages,
                            output_text=str(getattr(response, "content", response) or ""),
                        )
                    return _ForecastInputAssessment.model_validate_json(
                        str(getattr(response, "content", response))
                    )
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    messages.append(("human", f"Correct the forecast input assessment schema error: {exc}"))
        raise ValueError(f"Forecast input semantic assessment failed: {last_error}") from last_error


def _semantic_input_quality_error(
    *,
    assessment: _ForecastInputAssessment,
    evidence: DatabaseEvidence,
    anomaly_already_applied: bool,
) -> StructuredToolError:
    required_action = "sql_query" if anomaly_already_applied else "anomaly"
    repair_contract = {
        "mode": "forecast_input_quality_repair",
        "failed_tool": "forecast",
        "input_evidence": evidence.evidence_id,
        "quality_issues": assessment.quality_issues,
        "reason": assessment.reason,
    }
    return StructuredToolError(
        f"Forecast input failed semantic quality assessment: {assessment.reason}",
        error_type="forecast_input_semantic_quality",
        retryable=True,
        diagnostics=repair_contract,
        recommended_next_action=required_action,
        validation_failure={
            "tool": required_action,
            "scope": "forecast_input_quality",
            "capability": "forecast",
            "error_code": "forecast_input_semantic_quality",
            "message": assessment.reason,
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": required_action,
                "max_equivalent_retries": 2,
                "allow_same_action": False,
                "terminal_after_exhausted": True,
            },
        },
    )


def _insufficient_timeseries_evidence_error(
    *,
    message: str,
    tool_name: str,
    evidence: DatabaseEvidence,
) -> StructuredToolError:
    repair_contract = {
        "mode": "timeseries_input_repair",
        "failed_tool": tool_name,
        "input_evidence": evidence.evidence_id,
        "required_evidence_shape": "raw_timeseries",
        "required_min_points": 2,
        "constraints": {
            "evidence_shape": "raw_timeseries",
            "dialect_complexity_policy": "simple_raw_evidence",
        },
    }
    return StructuredToolError(
        message,
        error_type="insufficient_timeseries_evidence",
        retryable=True,
        diagnostics={
            "input_evidence_id": evidence.evidence_id,
            "result_type": evidence.result_type,
            "required_min_points": 2,
            "required_evidence_shape": "raw_timeseries",
        },
        recommended_next_action="sql_query",
        validation_failure={
            "tool": "sql_query",
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": "sql_query",
                "max_equivalent_retries": 2,
                "terminal_after_exhausted": False,
            },
        },
    )


def _resolve_forecast_plan(
    validated_input: ForecastInput,
    constraints: dict,
    series: TimeSeriesSeries,
) -> ForecastPlan:
    sampling_interval = _infer_sampling_interval_seconds(series)
    duration_seconds = _duration_seconds_from_inputs(validated_input.horizon, constraints)
    explicit_steps = _horizon_steps_from_value(validated_input.horizon)
    if explicit_steps is None:
        explicit_steps = _horizon_steps_from_value(
            constraints.get("horizon") if constraints.get("horizon") is not None else constraints.get("forecast_horizon")
        )

    if explicit_steps is not None:
        requested_steps = explicit_steps
        horizon_source = "explicit_steps"
        duration_seconds = None
    elif duration_seconds is not None:
        if sampling_interval is None:
            raise ValueError("Forecast duration requires at least two parseable timestamps to infer the sampling interval.")
        requested_steps = max(1, ceil(duration_seconds / sampling_interval))
        horizon_source = "duration_from_user"
    else:
        requested_steps = max(3, min(24, ceil(len(series.points) * 0.1)))
        horizon_source = "inferred_short_term_default"

    max_direct_steps = _positive_int(constraints.get("max_direct_steps"), DEFAULT_MAX_DIRECT_STEPS)
    forecast_start, forecast_end = _forecast_bounds(series, sampling_interval, requested_steps)
    if requested_steps > max_direct_steps:
        return ForecastPlan(
            mode="rolling",
            horizon_source=horizon_source,
            requested_steps=requested_steps,
            resolved_steps=requested_steps,
            sampling_interval_seconds=sampling_interval,
            forecast_duration_seconds=duration_seconds,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            max_direct_steps=max_direct_steps,
            recommended_chunk_steps=max_direct_steps,
            reason=(
                f"Requested horizon {requested_steps} exceeds max direct forecast "
                f"window {max_direct_steps}; forecast will be generated in rolling chunks."
            ),
        )

    return ForecastPlan(
        mode="direct",
        horizon_source=horizon_source,
        requested_steps=requested_steps,
        resolved_steps=requested_steps,
        sampling_interval_seconds=sampling_interval,
        forecast_duration_seconds=duration_seconds,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        max_direct_steps=max_direct_steps,
        recommended_chunk_steps=None,
        reason=None,
    )


def _horizon_steps_from_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float):
        return max(1, int(value))
    if isinstance(value, dict):
        for key in ("steps", "horizon", "points", "count"):
            steps = _horizon_steps_from_value(value.get(key))
            if steps is not None:
                return steps
        return None
    if isinstance(value, str):
        if _parse_duration_seconds(value) is not None:
            return None
        match = re.search(r"\d+", value)
        if match:
            return max(1, int(match.group(0)))
    return None


def _rolling_forecast(
    *,
    model,
    series: TimeSeriesSeries,
    horizon: int,
    chunk_steps: int,
    params: dict,
):
    from core.timeseries.forecast_registry import ForecastModelOutput

    remaining = max(1, horizon)
    chunk_steps = max(1, chunk_steps)
    working_series = series.model_copy(deep=True)
    forecast_points = []
    confidence_interval = []
    chunk_diagnostics = []
    chunk_index = 0
    while remaining > 0:
        chunk_index += 1
        current_horizon = min(chunk_steps, remaining)
        chunk_output = model.forecast(working_series, horizon=current_horizon, params=params)
        if len(chunk_output.forecast_points) != current_horizon:
            raise ValueError(
                f"Rolling forecast chunk {chunk_index} returned "
                f"{len(chunk_output.forecast_points)} points; expected {current_horizon}."
            )
        forecast_points.extend(chunk_output.forecast_points)
        confidence_interval.extend(chunk_output.confidence_interval)
        chunk_diagnostics.append(
            {
                "chunk_index": chunk_index,
                "requested_steps": current_horizon,
                "returned_steps": len(chunk_output.forecast_points),
                "diagnostics": chunk_output.diagnostics,
            }
        )
        working_series = working_series.model_copy(
            update={"points": [*working_series.points, *chunk_output.forecast_points]}
        )
        remaining -= current_horizon

    return ForecastModelOutput(
        forecast_points=forecast_points,
        confidence_interval=confidence_interval,
        diagnostics={
            "model_family": "rolling",
            "rolling_chunks": chunk_diagnostics,
            "rolling_chunk_count": chunk_index,
            "rolling_chunk_steps": chunk_steps,
        },
    )


def _apply_forecast_input_policy(
    *,
    series: TimeSeriesSeries,
    database_evidence: DatabaseEvidence,
    request_state,
    constraints: dict,
) -> tuple[TimeSeriesSeries, dict]:
    policy = str(constraints.get("input_policy") or "exclude_detected_anomalies").strip().lower()
    diagnostics = {
        "input_policy": policy,
        "selected_evidence_id": database_evidence.evidence_id,
        "training_point_count_before_policy": len(series.points),
        "training_point_count_after_policy": len(series.points),
        "excluded_anomaly_count": 0,
    }
    if policy in {"selected", "current", "as_provided", "raw"} or request_state is None:
        return series, diagnostics

    exclusion_keys, source = _forecast_exclusion_keys(database_evidence, request_state)
    if not exclusion_keys:
        if source:
            diagnostics["input_policy_note"] = source
        return series, diagnostics

    filtered_points = [
        point
        for point in series.points
        if _point_key(point.timestamp, point.value) not in exclusion_keys
    ]
    excluded_count = len(series.points) - len(filtered_points)
    if excluded_count <= 0:
        return series, diagnostics
    if len(filtered_points) < 2:
        raise ValueError("Forecast cannot exclude detected anomalies because fewer than two training points would remain.")

    diagnostics.update(
        {
            "training_point_count_after_policy": len(filtered_points),
            "excluded_anomaly_count": excluded_count,
            **source,
        }
    )
    return series.model_copy(update={"points": filtered_points}), diagnostics


def _forecast_exclusion_keys(
    database_evidence: DatabaseEvidence,
    request_state,
) -> tuple[set[tuple[str, float]], dict | str | None]:
    anomaly = getattr(request_state, "latest_anomaly", None)
    if anomaly is not None:
        anomaly_diagnostics = anomaly.diagnostics if isinstance(anomaly.diagnostics, dict) else {}
        anomaly_evidence_id = anomaly_diagnostics.get("resolved_evidence_id") or _evidence_id_from_anomaly_id(anomaly.anomaly_id)
        if anomaly_evidence_id == database_evidence.evidence_id:
            keys = _keys_from_rows(anomaly.anomaly_points)
            if keys:
                return keys, {
                    "source_anomaly_id": anomaly.anomaly_id,
                    "input_policy_note": "forecast training excluded points detected by the anomaly tool",
                }
        elif anomaly_evidence_id:
            anomaly_note = "latest anomaly result belongs to a different evidence artifact"
        else:
            anomaly_note = None
    else:
        anomaly_note = None

    return set(), anomaly_note


def _keys_from_rows(rows: Any) -> set[tuple[str, float]]:
    if not isinstance(rows, list):
        return set()
    keys = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _point_key(
            row.get("timestamp") or row.get("_time") or row.get("time"),
            row.get("value") if row.get("value") is not None else row.get("price") if row.get("price") is not None else row.get("_value"),
        )
        if key is not None:
            keys.add(key)
    return keys


def _point_key(timestamp: Any, value: Any) -> tuple[str, float] | None:
    if timestamp is None or value is None:
        return None
    parsed = _parse_timestamp(str(timestamp))
    if parsed is None:
        normalized_timestamp = str(timestamp)
    else:
        normalized_timestamp = parsed.isoformat()
    try:
        normalized_value = float(value)
    except (TypeError, ValueError):
        return None
    return normalized_timestamp, normalized_value


def _evidence_id_from_anomaly_id(anomaly_id: str | None) -> str | None:
    text = str(anomaly_id or "")
    prefix = "anomaly_"
    if text.startswith(prefix):
        return text[len(prefix):]
    return None


def _duration_seconds_from_inputs(horizon: Any, constraints: dict) -> int | None:
    for value in (
        horizon,
        constraints.get("forecast_duration"),
        constraints.get("horizon_duration"),
        constraints.get("duration"),
        constraints.get("forecast_window"),
        constraints.get("period"),
    ):
        seconds = _parse_duration_seconds(value)
        if seconds is not None:
            return seconds
    return None


def _parse_duration_seconds(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("duration", "window", "period", "forecast_duration", "horizon_duration"):
            seconds = _parse_duration_seconds(value.get(key))
            if seconds is not None:
                return seconds
        return None
    if not isinstance(value, str):
        return None
    total = 0.0
    for amount, unit in _DURATION_RE.findall(value):
        total += float(amount) * _DURATION_UNIT_SECONDS[unit.lower()]
    if total <= 0:
        return None
    return max(1, int(total))


def _infer_sampling_interval_seconds(series: TimeSeriesSeries) -> int | None:
    parsed = [_parse_timestamp(point.timestamp) for point in series.points]
    parsed = [item for item in parsed if item is not None]
    if len(parsed) < 2:
        return None
    deltas = [
        int((current - previous).total_seconds())
        for previous, current in zip(parsed, parsed[1:])
        if current > previous
    ]
    if not deltas:
        return None
    return max(1, int(median(deltas)))


def _forecast_bounds(
    series: TimeSeriesSeries,
    sampling_interval_seconds: int | None,
    requested_steps: int,
) -> tuple[str | None, str | None]:
    if sampling_interval_seconds is None:
        return None, None
    last_timestamp = _parse_timestamp(series.points[-1].timestamp)
    if last_timestamp is None:
        return None, None
    forecast_start = last_timestamp + timedelta(seconds=sampling_interval_seconds)
    forecast_end = last_timestamp + timedelta(seconds=sampling_interval_seconds * requested_steps)
    return forecast_start.isoformat(), forecast_end.isoformat()


def _validate_forecast_evidence_quality(
    evidence: DatabaseEvidence,
    series: TimeSeriesSeries,
    request_state,
    constraints: dict,
) -> dict:
    raw_limit = dialect_for_database(evidence.query_language).raw_limit_without_downsampling(
        evidence.query,
        evidence.query_language,
    )
    coverage_start = series.points[0].timestamp
    coverage_end = series.points[-1].timestamp
    sampling_interval = _infer_sampling_interval_seconds(series)
    requested_end = _requested_time_range_end(request_state, constraints)
    covers_requested_range = True
    if requested_end is not None:
        observed_end = _parse_timestamp(coverage_end)
        tolerance_seconds = sampling_interval or 0
        covers_requested_range = bool(
            observed_end is not None
            and observed_end + timedelta(seconds=tolerance_seconds) >= requested_end
        )
    quality = {
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "sampling_interval_seconds": sampling_interval,
        "raw_limit_query": raw_limit,
        "covers_requested_range": covers_requested_range,
        "requested_range_end": requested_end.isoformat() if requested_end is not None else None,
    }
    if raw_limit and requested_end is not None and not covers_requested_range:
        raise ValueError(
            "Forecast evidence is incomplete for the requested time range because the query uses a raw limit. "
            "Use full-range aggregation or representative downsampling before forecasting."
        )
    return quality


def _requested_time_range_end(request_state, constraints: dict) -> datetime | None:
    for candidate in (
        constraints.get("time_range"),
        getattr(request_state, "time_range", None) if request_state is not None else None,
    ):
        if not isinstance(candidate, dict):
            continue
        parsed = _parse_timestamp(candidate.get("end") or candidate.get("to") or candidate.get("stop"))
        if parsed is not None:
            return parsed
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("Z", "+00:00")
    if "T" not in normalized and " " in normalized:
        normalized = normalized.replace(" ", "T")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)
