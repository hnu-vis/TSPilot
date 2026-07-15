"""Build cross-database metric semantic context from schema and query results."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from common.schemas.metric_context import (
    MetricContext,
    MetricSemanticContext,
    MetricSourceContext,
    TimeSemanticContext,
)
from common.units import infer_measure_unit


class MetricContextBuilder:
    """Infer a unified metric context for Prometheus, InfluxDB, TimescaleDB, and peers."""

    def build_pre_query(
        self,
        *,
        database_id: str,
        db_type: str,
        dialect: str,
        schema: Any | None,
        user_message: str,
        semantic_context: dict[str, Any] | None = None,
        schema_linking: dict[str, Any] | None = None,
        query_plan: dict[str, Any] | None = None,
    ) -> MetricContext:
        """Build metric context before query generation from schema and user intent."""
        if isinstance(semantic_context, dict) and semantic_context:
            return self._build_pre_query_from_model_context(
                database_id=database_id,
                db_type=db_type,
                dialect=dialect,
                schema=schema,
                semantic_context=semantic_context,
                schema_linking=schema_linking,
                query_plan=query_plan,
            )

        message = user_message or ""
        source = MetricSourceContext(database_id=database_id, db_type=db_type, dialect=dialect)
        metric_name = self._infer_metric_name_from_message(
            message=message,
            db_type=db_type,
            dialect=dialect,
            schema=schema,
        )
        requested_name = self._infer_requested_metric_label(message)
        table = self._find_table(schema, metric_name)
        labels = self._infer_requested_labels(
            message=message,
            table=table,
            db_type=db_type,
            dialect=dialect,
        )
        time_column = self._infer_time_column(table)
        value_column = self._infer_value_column(table, db_type=db_type, dialect=dialect, message=message)
        query_lookback = self._extract_requested_lookback(message)
        query_step = self._choose_query_step(query_lookback)
        metric = MetricSemanticContext(
            name=metric_name,
            requested_name=requested_name,
            display_name=requested_name or self._display_name(metric_name),
            description=self._infer_description(schema, metric_name),
            value_type=self._infer_value_type(
                query=message,
                metric_name=metric_name,
                db_type=db_type,
            ),
            value_unit=self._infer_value_unit(schema=schema, metric_name=metric_name, value_column=value_column),
            aggregation=self._infer_requested_aggregation(message),
            labels=labels,
            time_column=time_column,
            value_column=value_column,
        )
        time = TimeSemanticContext(
            query_lookback=query_lookback,
            query_step=query_step,
            forecast_step=query_step,
        )
        evidence = []
        if metric_name:
            evidence.append("metric selected from user request and schema")
        if query_lookback:
            evidence.append("lookback inferred from user request")
        if query_step:
            evidence.append("query step selected from lookback")
        if labels:
            evidence.append("labels inferred from user request and schema preview")
        if requested_name:
            evidence.append("target label inferred from user request")
        confidence = "high" if metric_name and query_lookback else "medium" if metric_name or query_lookback else "low"
        return MetricContext(source=source, metric=metric, time=time, confidence=confidence, evidence=evidence)

    def _build_pre_query_from_model_context(
        self,
        *,
        database_id: str,
        db_type: str,
        dialect: str,
        schema: Any | None,
        semantic_context: dict[str, Any],
        schema_linking: dict[str, Any] | None,
        query_plan: dict[str, Any] | None,
    ) -> MetricContext:
        """Build pre-query context from a validated model-owned semantic contract."""
        source = MetricSourceContext(database_id=database_id, db_type=db_type, dialect=dialect)
        metric_in = semantic_context.get("metric") if isinstance(semantic_context.get("metric"), dict) else {}
        time_in = semantic_context.get("time") if isinstance(semantic_context.get("time"), dict) else {}
        plan_metric = self._metric_hints_from_query_plan(query_plan)
        linking_metric = self._metric_hints_from_schema_linking(schema_linking)

        model_metric_name = self._validated_source_name(schema, metric_in.get("name"))
        metric_name = self._first_text(
            model_metric_name,
            self._validated_source_name(schema, plan_metric.get("name")),
            self._validated_source_name(schema, linking_metric.get("name")),
        )
        requested_name = self._first_text(metric_in.get("requested_name"))
        display_name = self._first_text(
            metric_in.get("display_name"),
            requested_name,
            metric_name,
        )
        time_column = self._first_text(
            self._validated_column_name(schema, metric_name, metric_in.get("time_column")),
            plan_metric.get("time_column"),
            linking_metric.get("time_column"),
        )
        value_column = self._first_text(
            self._validated_column_name(schema, metric_name, metric_in.get("value_column")),
            plan_metric.get("value_column"),
            linking_metric.get("value_column"),
        )
        labels = metric_in.get("labels") if isinstance(metric_in.get("labels"), dict) else {}
        aggregation = self._normalize_aggregation(
            self._first_text(metric_in.get("aggregation"), plan_metric.get("aggregation"))
        )
        query_lookback = self._normalize_duration(
            self._first_text(
                time_in.get("query_lookback"),
                (query_plan or {}).get("time_range", {}).get("lookback") if isinstance((query_plan or {}).get("time_range"), dict) else None,
            )
        )
        query_step = self._normalize_duration(self._first_text(time_in.get("query_step")))
        if not query_step:
            query_step = self._choose_query_step(query_lookback)
        forecast_step = self._normalize_duration(self._first_text(time_in.get("forecast_step"))) or query_step

        metric = MetricSemanticContext(
            name=metric_name,
            requested_name=requested_name,
            display_name=display_name or self._display_name(metric_name),
            description=self._first_text(metric_in.get("description"), self._infer_description(schema, metric_name)),
            value_type=self._first_text(metric_in.get("value_type")) or self._infer_value_type(
                query="",
                metric_name=metric_name,
                db_type=db_type,
            ),
            value_unit=self._first_text(metric_in.get("value_unit")) or self._infer_value_unit(
                schema=schema,
                metric_name=metric_name,
                value_column=value_column,
            ),
            aggregation=aggregation,
            labels=labels,
            time_column=time_column,
            value_column=value_column,
        )
        time = TimeSemanticContext(
            query_lookback=query_lookback,
            query_step=query_step,
            forecast_step=forecast_step,
        )
        confidence = str(semantic_context.get("confidence") or "").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "high" if metric_name or value_column else "medium"
        evidence = [
            str(item).strip()
            for item in list(semantic_context.get("evidence") or [])
            if str(item).strip()
        ]
        if "semantic context supplied by model contract" not in evidence:
            evidence.append("semantic context supplied by model contract")
        return MetricContext(source=source, metric=metric, time=time, confidence=confidence, evidence=evidence)

    def build(
        self,
        *,
        database_id: str,
        db_type: str,
        dialect: str,
        schema: Any | None,
        query: str,
        query_language: str,
        query_mode: str | None,
        execution: dict[str, Any] | None,
        data: dict[str, Any],
    ) -> MetricContext:
        """Build a metric context from the available database and result metadata."""
        execution = execution if isinstance(execution, dict) else {}
        time_series = data.get("time_series") if isinstance(data, dict) else None
        source = MetricSourceContext(database_id=database_id, db_type=db_type, dialect=dialect)
        metric_name = self._infer_metric_name(
            query=query,
            db_type=db_type,
            dialect=dialect,
            schema=schema,
            time_series=time_series,
        )
        labels = self._infer_labels(query=query, db_type=db_type, dialect=dialect)
        time_column = time_series.get("time_column") if isinstance(time_series, dict) else None
        value_column = time_series.get("value_column") if isinstance(time_series, dict) else None
        observed = self._infer_observed_time_context(time_series)
        query_step = self._normalize_duration(execution.get("step"))
        query_lookback = self._normalize_duration(execution.get("lookback"))
        metric = MetricSemanticContext(
            name=metric_name,
            display_name=self._display_name(metric_name),
            description=self._infer_description(schema, metric_name),
            value_type=self._infer_value_type(query=query, metric_name=metric_name, db_type=db_type),
            value_unit=self._infer_value_unit(schema=schema, metric_name=metric_name, value_column=value_column),
            aggregation=self._infer_aggregation(query=query, query_mode=query_mode),
            labels=labels,
            time_column=time_column,
            value_column=value_column,
        )
        time = TimeSemanticContext(
            query_lookback=query_lookback,
            query_step=query_step,
            observed_step=observed.get("observed_step") or query_step,
            observed_points=observed.get("observed_points"),
            first_timestamp=observed.get("first_timestamp"),
            last_timestamp=observed.get("last_timestamp"),
            forecast_step=observed.get("observed_step") or query_step,
        )
        confidence = "high" if metric_name and time.observed_step else "medium" if metric_name or time.observed_step else "low"
        evidence = []
        if metric_name:
            evidence.append("metric name inferred from query/schema")
        if time.observed_step:
            evidence.append("sampling interval inferred from returned timestamps")
        if query_step:
            evidence.append("query step provided by execution plan")
        return MetricContext(source=source, metric=metric, time=time, confidence=confidence, evidence=evidence)

    def _infer_metric_name(
        self,
        *,
        query: str,
        db_type: str,
        dialect: str,
        schema: Any | None,
        time_series: Any,
    ) -> str | None:
        query_text = query or ""
        if db_type == "prometheus" or dialect == "prometheus":
            prometheus_name = self._infer_prometheus_metric_name(query_text)
            if prometheus_name:
                return prometheus_name

        table_names = [
            getattr(table, "name", "")
            for table in list(getattr(schema, "tables", []) or [])
        ]
        query_lower = query_text.lower()
        for name in table_names:
            if name and name.lower() in query_lower:
                return name

        if isinstance(time_series, dict):
            value_column = time_series.get("value_column")
            if value_column and value_column != "value":
                return str(value_column)
        return None

    def _metric_hints_from_query_plan(self, query_plan: dict[str, Any] | None) -> dict[str, Any]:
        """Read structural metric hints from a model-owned query plan."""
        if not isinstance(query_plan, dict):
            return {}
        hints: dict[str, Any] = {}
        sources = query_plan.get("sources") if isinstance(query_plan.get("sources"), list) else []
        if sources and isinstance(sources[0], dict):
            source = sources[0]
            hints["name"] = source.get("name")
            hints["time_column"] = source.get("time_column")
            value_columns = source.get("value_columns") if isinstance(source.get("value_columns"), list) else []
            if value_columns:
                hints["value_column"] = value_columns[0]
        projections = query_plan.get("projections") if isinstance(query_plan.get("projections"), list) else []
        for projection in projections:
            if not isinstance(projection, dict):
                continue
            if projection.get("column") and not hints.get("value_column"):
                hints["value_column"] = projection.get("column")
            if projection.get("aggregation"):
                hints["aggregation"] = projection.get("aggregation")
                break
        return hints

    def _metric_hints_from_schema_linking(self, schema_linking: dict[str, Any] | None) -> dict[str, Any]:
        """Read structural metric hints from validated schema linking."""
        if not isinstance(schema_linking, dict):
            return {}
        hints: dict[str, Any] = {}
        sources = schema_linking.get("sources") if isinstance(schema_linking.get("sources"), list) else []
        if sources and isinstance(sources[0], dict):
            source = sources[0]
            hints["name"] = source.get("name")
            hints["time_column"] = source.get("time_column")
            value_columns = source.get("value_columns") if isinstance(source.get("value_columns"), list) else []
            if value_columns:
                hints["value_column"] = value_columns[0]
        time_columns = schema_linking.get("time_columns") if isinstance(schema_linking.get("time_columns"), list) else []
        value_columns = schema_linking.get("value_columns") if isinstance(schema_linking.get("value_columns"), list) else []
        if time_columns and not hints.get("time_column"):
            hints["time_column"] = time_columns[0]
        if value_columns and not hints.get("value_column"):
            hints["value_column"] = value_columns[0]
        return hints

    def _first_text(self, *values: Any) -> str | None:
        for value in values:
            text = str(value or "").strip()
            if text and text.lower() != "null":
                return text
        return None

    def _validated_source_name(self, schema: Any | None, value: Any) -> str | None:
        text = self._first_text(value)
        if not text:
            return None
        if not schema:
            return text
        table_names = {
            str(getattr(table, "name", "") or "")
            for table in list(getattr(schema, "tables", []) or [])
        }
        return text if text in table_names else None

    def _validated_column_name(self, schema: Any | None, source_name: str | None, value: Any) -> str | None:
        text = self._first_text(value)
        if not text:
            return None
        if not schema or text in {"value", "_value"}:
            return text
        candidate_tables = [
            table
            for table in list(getattr(schema, "tables", []) or [])
            if not source_name or str(getattr(table, "name", "") or "") == source_name
        ] or list(getattr(schema, "tables", []) or [])
        for table in candidate_tables:
            columns = {
                str(getattr(column, "name", "") or "")
                for column in list(getattr(table, "columns", []) or [])
            }
            if text in columns:
                return text
        return None

    def _normalize_aggregation(self, value: Any) -> str | None:
        text = str(value or "").strip().lower()
        aliases = {
            "average": "avg",
            "mean": "avg",
            "total": "sum",
            "minimum": "min",
            "maximum": "max",
        }
        text = aliases.get(text, text)
        return text if text in {"avg", "sum", "count", "min", "max", "first", "last", "rate", "increase"} else None

    def _infer_prometheus_metric_name(self, query: str) -> str | None:
        candidates = re.findall(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?:\{|\[|$)", query)
        functions = {
            "sum",
            "avg",
            "min",
            "max",
            "count",
            "rate",
            "irate",
            "increase",
            "max_over_time",
            "min_over_time",
            "avg_over_time",
            "sum_over_time",
        }
        for candidate in candidates:
            if candidate.lower() not in functions:
                return candidate
        return None

    def _infer_labels(self, *, query: str, db_type: str, dialect: str) -> dict[str, list[Any] | str]:
        if db_type != "prometheus" and dialect != "prometheus":
            return {}
        selector = re.search(r"\{([^}]*)\}", query or "")
        if not selector:
            return {}
        labels: dict[str, list[Any] | str] = {}
        for key, op, value in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(=~|!~|=|!=)\s*"([^"]*)"', selector.group(1)):
            labels[key] = value if op == "=" else f"{op}{value}"
        return labels

    def _infer_description(self, schema: Any | None, metric_name: str | None) -> str | None:
        if not schema or not metric_name:
            return None
        for table in list(getattr(schema, "tables", []) or []):
            if getattr(table, "name", None) == metric_name:
                return getattr(table, "description", None)
        return None

    def _infer_value_type(self, *, query: str, metric_name: str | None, db_type: str) -> str:
        query_lower = (query or "").lower()
        if "histogram_quantile" in query_lower or (metric_name or "").endswith("_bucket"):
            return "histogram"
        if db_type == "prometheus" and ((metric_name or "").endswith("_total") or "increase(" in query_lower or "rate(" in query_lower):
            return "counter"
        return "unknown"

    def _infer_value_unit(self, *, schema: Any | None = None, metric_name: str | None, value_column: str | None) -> str | None:
        schema_unit = self._schema_value_unit(schema, metric_name=metric_name, value_column=value_column)
        return schema_unit or infer_measure_unit(metric_name, value_column)

    def _schema_value_unit(self, schema: Any | None, *, metric_name: str | None, value_column: str | None) -> str | None:
        if not schema:
            return None
        tables = list(getattr(schema, "tables", []) or [])
        candidate_tables = [
            table for table in tables
            if not metric_name or str(getattr(table, "name", "") or "") == str(metric_name)
        ] or tables
        for table in candidate_tables:
            table_unit = str(getattr(table, "unit", "") or "").strip()
            if table_unit:
                return table_unit
            for column in list(getattr(table, "columns", []) or []):
                column_name = str(getattr(column, "name", "") or "")
                if value_column and column_name != str(value_column):
                    continue
                unit = str(getattr(column, "unit", "") or "").strip()
                if unit:
                    return unit
        return None

    def _infer_aggregation(self, *, query: str, query_mode: str | None) -> str | None:
        query_lower = (query or "").lower()
        for name in ("increase", "rate", "sum", "avg", "min", "max", "count"):
            if f"{name}(" in query_lower:
                return name
        return query_mode

    def _infer_observed_time_context(self, time_series: Any) -> dict[str, Any]:
        if not isinstance(time_series, dict):
            return {}
        points = [
            point for point in time_series.get("points", [])
            if isinstance(point, dict) and point.get("timestamp") not in (None, "")
        ]
        timestamps = [self._parse_datetime(point.get("timestamp")) for point in points]
        timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
        result: dict[str, Any] = {"observed_points": len(points)}
        if timestamps:
            result["first_timestamp"] = timestamps[0].isoformat()
            result["last_timestamp"] = timestamps[-1].isoformat()
        if len(timestamps) >= 2:
            deltas = [
                timestamps[index + 1] - timestamps[index]
                for index in range(len(timestamps) - 1)
                if timestamps[index + 1] >= timestamps[index]
            ]
            if deltas:
                seconds = sorted(int(delta.total_seconds()) for delta in deltas)
                median_seconds = seconds[len(seconds) // 2]
                result["observed_step"] = self._format_duration(timedelta(seconds=median_seconds))
        return result

    def _parse_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value / 1000 if value > 10_000_000_000 else value)
            except Exception:
                return None
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _normalize_duration(self, value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        return None if text.lower() in {"null", "none", "duration_or_null"} else text

    def _format_duration(self, delta: timedelta) -> str:
        seconds = int(delta.total_seconds())
        if seconds <= 0:
            return "0s"
        units = (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60), ("s", 1))
        for suffix, size in units:
            if seconds % size == 0:
                return f"{seconds // size}{suffix}"
        return f"{seconds}s"

    def _display_name(self, metric_name: str | None) -> str | None:
        if not metric_name:
            return None
        return metric_name.replace("_", " ").strip().title()

    def _infer_requested_metric_label(self, message: str) -> str | None:
        """Extract a business target label from the user's natural-language request."""
        text = (message or "").strip()
        if not text:
            return None
        text = re.sub(
            r"\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[?？!！。,.，；;:：]", " ", text)
        cleaned = re.sub(
            r"(下一?周|下周|未来[一二三四五六七八九十\d]+周|未来[一二三四五六七八九十\d]+天|"
            r"最近[一二三四五六七八九十\d]+[天周月小时分钟]*|"
            r"\d+\s*(秒|分钟|分|小时|天|周|月|second|seconds|minute|minutes|hour|hours|day|days|week|weeks))",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(请问|我想知道|帮我|看一下|查询|预测|预估|估计|大概|大约|约|应该|会|是|有|到|"
            r"多少|如何|怎样|怎么样|what|how|forecast|predict|estimate|please|show|query|tell me)",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(总共的?|总的?|总量|累计|平均|均值|最高|最低|最大|最小|峰值|极值|"
            r"latest|highest|lowest|maximum|minimum|avg|average|max|min|sum|total)",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" 的")
        if not cleaned:
            return None
        chinese_candidates = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]*(?:量|率|数|值|费用|成本|功率|能耗|耗电|用电)[\u4e00-\u9fffA-Za-z0-9_]*", cleaned)
        if chinese_candidates:
            return min(chinese_candidates, key=len)
        english_tokens = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", cleaned)
            if token.lower() not in {"the", "a", "an", "of", "for", "in", "next", "last", "this", "week", "day"}
        ]
        if english_tokens:
            return " ".join(english_tokens[:4])
        return cleaned if len(cleaned) <= 32 else None

    def _infer_metric_name_from_message(
        self,
        *,
        message: str,
        db_type: str,
        dialect: str,
        schema: Any | None,
    ) -> str | None:
        message_lower = (message or "").lower()
        table_names = [
            getattr(table, "name", "")
            for table in list(getattr(schema, "tables", []) or [])
        ]
        for name in table_names:
            if name and name.lower() in message_lower:
                return name

        if db_type == "prometheus" or dialect == "prometheus":
            scored: list[tuple[int, str]] = []
            for name in table_names:
                lowered = name.lower()
                score = 0
                if "http" in message_lower and "http" in lowered:
                    score += 4
                if any(term in message_lower for term in ("request", "请求")) and "request" in lowered:
                    score += 4
                if any(term in message_lower for term in ("成功", "success", "successful", "2xx", "200")):
                    if "request" in lowered or "http" in lowered:
                        score += 2
                if lowered.endswith("_total") and any(term in message_lower for term in ("总数", "total", "counter")):
                    score += 2
                if name.startswith(("prometheus_", "scrape_")) or name == "up":
                    score -= 2
                if score > 0:
                    scored.append((score, name))
            if scored:
                return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]

        return table_names[0] if len(table_names) == 1 else None

    def _find_table(self, schema: Any | None, metric_name: str | None) -> Any | None:
        if not schema or not metric_name:
            return None
        for table in list(getattr(schema, "tables", []) or []):
            if getattr(table, "name", None) == metric_name:
                return table
        return None

    def _infer_requested_labels(
        self,
        *,
        message: str,
        table: Any | None,
        db_type: str,
        dialect: str,
    ) -> dict[str, list[Any] | str]:
        if db_type != "prometheus" and dialect != "prometheus":
            return {}
        available = {
            str(getattr(column, "name", "")).removeprefix("label_")
            for column in list(getattr(table, "columns", []) or [])
            if str(getattr(column, "name", "")).startswith("label_")
        }
        message_lower = (message or "").lower()
        labels: dict[str, list[Any] | str] = {}
        if any(term in message_lower for term in ("成功", "success", "successful", "2xx", "ok")):
            if "code" in available:
                labels["code"] = "200"
            elif "status" in available:
                labels["status"] = "200"
            elif "status_code" in available:
                labels["status_code"] = "200"
        return labels

    def _infer_time_column(self, table: Any | None) -> str | None:
        for column in list(getattr(table, "columns", []) or []):
            name = str(getattr(column, "name", ""))
            data_type = str(getattr(column, "data_type", "")).lower()
            if name.lower() in {"time", "timestamp", "ts", "datetime"} or "time" in data_type:
                return name
        return None

    def _infer_value_column(self, table: Any | None, *, db_type: str, dialect: str, message: str = "") -> str | None:
        if db_type == "prometheus" or dialect == "prometheus":
            return "value"
        columns = list(getattr(table, "columns", []) or [])
        message_lower = (message or "").lower()
        for column in columns:
            name = str(getattr(column, "name", ""))
            if name and name.lower() in message_lower:
                return name
        preferred = ("value", "count", "requests", "total", "duration", "usage")
        for candidate in preferred:
            for column in columns:
                name = str(getattr(column, "name", "")).lower()
                if name == candidate or name.endswith(f"_{candidate}"):
                    return str(getattr(column, "name", ""))
        for column in columns:
            data_type = str(getattr(column, "data_type", "")).lower()
            if any(kind in data_type for kind in ("int", "float", "double", "decimal", "number")):
                name = str(getattr(column, "name", ""))
                if name.lower() not in {"time", "timestamp", "ts"}:
                    return name
        return None

    def _extract_requested_lookback(self, message: str) -> str | None:
        text = (message or "").lower()
        relative_aliases = [
            (("上周", "上一周", "过去一周", "过去 1 周", "最近一周", "最近 1 周", "last week", "previous week", "past week"), "1w"),
            (("昨天", "昨日", "yesterday"), "1d"),
            (("今天", "今日", "today"), "1d"),
            (("上个月", "上一月", "过去一个月", "过去 1 个月", "last month", "previous month", "past month"), "30d"),
        ]
        for terms, duration in relative_aliases:
            if any(term in text for term in terms):
                return duration

        match = re.search(
            r"(\d+)\s*(秒|分钟|分|小时|天|周|月|second|seconds|sec|minute|minutes|min|hour|hours|day|days|week|weeks)",
            text,
        )
        if not match:
            return None
        amount = int(match.group(1))
        unit = match.group(2)
        unit_map = {
            "秒": "s",
            "second": "s",
            "seconds": "s",
            "sec": "s",
            "分钟": "m",
            "分": "m",
            "minute": "m",
            "minutes": "m",
            "min": "m",
            "小时": "h",
            "hour": "h",
            "hours": "h",
            "天": "d",
            "day": "d",
            "days": "d",
            "周": "w",
            "week": "w",
            "weeks": "w",
            "月": "d",
        }
        if unit == "月":
            return f"{amount * 30}d"
        return f"{amount}{unit_map.get(unit, 'm')}"

    def _choose_query_step(self, lookback: str | None) -> str | None:
        if not lookback:
            return None
        delta = self._duration_to_timedelta(lookback)
        if not delta:
            return None
        seconds = delta.total_seconds()
        if seconds <= 15 * 60:
            return "15s"
        if seconds <= 6 * 60 * 60:
            return "1m"
        if seconds <= 2 * 24 * 60 * 60:
            return "5m"
        if seconds <= 14 * 24 * 60 * 60:
            return "1h"
        return "6h"

    def _duration_to_timedelta(self, duration: str) -> timedelta | None:
        match = re.fullmatch(r"(\d+)\s*([smhdw])", duration.strip().lower())
        if not match:
            return None
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "s":
            return timedelta(seconds=amount)
        if unit == "m":
            return timedelta(minutes=amount)
        if unit == "h":
            return timedelta(hours=amount)
        if unit == "d":
            return timedelta(days=amount)
        if unit == "w":
            return timedelta(weeks=amount)
        return None

    def _infer_requested_aggregation(self, message: str) -> str | None:
        message_lower = (message or "").lower()
        if any(term in message_lower for term in ("平均", "均值", "avg", "average", "mean")):
            return "avg"
        if any(term in message_lower for term in ("最大", "最高", "max", "maximum")):
            return "max"
        if any(term in message_lower for term in ("最小", "最低", "min", "minimum")):
            return "min"
        if any(term in message_lower for term in ("总数", "合计", "求和", "total", "sum")):
            return "sum"
        if any(term in message_lower for term in ("数量", "count", "how many")):
            return "count"
        return None
