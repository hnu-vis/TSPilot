from __future__ import annotations

import json


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        intent_response = _maybe_intent_response(user_prompt)
        if intent_response:
            return intent_response
        context_json = user_prompt.split("Context JSON:\n", 1)[1]
        context = json.loads(context_json)

        if context.get("database_context") is None:
            return _turn(
                "No datasource is available, so I can only attempt final assembly and let the runtime fail closed.",
                "format_answer",
                {"summary_goal": context["message"], "include_fact_ids": [], "include_visualization_ids": [], "section_plan": []},
            )

        if context.get("latest_database_evidence") is None:
            action_input = {
                "message": context["message"],
                "database_context": context["database_context"],
                "time_range": context.get("time_range"),
                "constraints": context.get("constraints", {}),
                "selected_database": context.get("selected_database"),
                "selected_database_type": context.get("selected_database_type"),
                "history": context.get("history", []),
            }
            return _turn("I need evidence before producing facts.", "sql_query", action_input)

        latest_evidence = context.get("latest_database_evidence") or {}
        result_type = latest_evidence.get("result_type")
        if result_type in {"schema", "metric_list", "statistics", "table"}:
            action_input = {
                "summary_goal": context["message"],
                "include_fact_ids": [],
                "include_visualization_ids": [
                    visualization["visualization_id"]
                    for visualization in context.get("visualizations", [])
                ],
                "section_plan": ["summary", result_type],
            }
            return _turn(
                "I already have a non-timeseries evidence family and should answer directly.",
                "format_answer",
                action_input,
            )

        if _analysis_count(context) == 0:
            action_input = _analysis_action_input(latest_evidence, context.get("message", "Analyze evidence."))
            return _turn(
                "I have evidence and should run generated analysis code.",
                "insight",
                action_input,
            )

        analysis_ids = _analysis_ids(context)
        action_input = {
            "summary_goal": context["message"],
            "include_analysis_ids": analysis_ids,
            "include_fact_ids": [],
            "include_visualization_ids": [
                visualization["visualization_id"]
                for visualization in context.get("visualizations", [])
            ],
            "section_plan": ["summary", "analysis", "visualization"],
        }
        return _turn(
            "I have enough verified output to assemble the final answer.",
            "format_answer",
            action_input,
        )


class CasualLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        intent_response = _maybe_intent_response(user_prompt)
        if intent_response:
            return intent_response
        context_json = user_prompt.split("Context JSON:\n", 1)[1]
        context = json.loads(context_json)
        return _turn(
            "This is a conversational request without a datasource, so I should answer directly.",
            "format_answer",
            {
                "summary_goal": "Answer the user's conversational request.",
                "direct_answer": "你好！我是 TSPilot，可以帮你查询和分析时序数据。你可以选择数据库后直接问趋势、异常、预测或指标解释。",
                "include_fact_ids": [],
                "include_visualization_ids": [],
                "section_plan": ["summary"],
            },
        )


class ComplexReActLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        intent_response = _maybe_intent_response(user_prompt)
        if intent_response:
            return intent_response
        context_json = user_prompt.split("Context JSON:\n", 1)[1]
        context = json.loads(context_json)

        if context.get("database_context") is None:
            return _turn(
                "No datasource is available, so I can only attempt final assembly and let the runtime fail closed.",
                "format_answer",
                {"summary_goal": context["message"], "include_fact_ids": [], "include_visualization_ids": [], "section_plan": []},
            )

        if not context.get("todo_list"):
            action_input = {
                "message": context["message"],
                "current_intent": "chat_analysis",
                "requested_fact_types": ["trend", "anomaly", "forecast"],
                "focus": "先规划，再查库，随后做趋势、异常和预测分析，最后整合回答。",
                "todos": [
                    {"content": "查询目标时间范围内的时序证据", "task_type": "query", "status": "in_progress", "priority": 1},
                    {"content": "提炼趋势事实", "task_type": "insight", "status": "pending", "priority": 2},
                    {"content": "检查异常点", "task_type": "anomaly", "status": "pending", "priority": 3},
                    {"content": "生成短期预测", "task_type": "forecast", "status": "pending", "priority": 4},
                    {"content": "汇总最终答案", "task_type": "answer", "status": "pending", "priority": 5},
                ],
            }
            return _turn(
                "This request is multi-step and should start with an explicit todo plan.",
                "todowrite",
                action_input,
            )

        if context.get("latest_database_evidence") is None:
            action_input = {
                "message": context["message"],
                "database_context": context["database_context"],
                "time_range": context.get("time_range"),
                "constraints": context.get("constraints", {}),
                "selected_database": context.get("selected_database"),
                "selected_database_type": context.get("selected_database_type"),
                "history": context.get("history", []),
            }
            return _turn("I have a plan and need time-series evidence next.", "sql_query", action_input)

        latest_evidence = context.get("latest_database_evidence") or {}
        if _analysis_count(context) == 0:
            action_input = _analysis_action_input(latest_evidence, "整体趋势、变化幅度和关键峰值")
            return _turn(
                "I need generated analysis before any downstream analytics summary.",
                "insight",
                action_input,
            )

        if context.get("latest_anomaly") is None:
            action_input = {
                "database_evidence": latest_evidence,
                "constraints": {"zscore_threshold": 2.0},
            }
            return _turn(
                "I should detect anomalies on the retrieved time series.",
                "anomaly",
                action_input,
            )

        if context.get("latest_forecast") is None:
            action_input = {
                "database_evidence": latest_evidence,
                "horizon": 6,
                "constraints": {"horizon": 6},
            }
            return _turn(
                "I should produce a short-term forecast before finalizing the answer.",
                "forecast",
                action_input,
            )

        action_input = {
            "summary_goal": context["message"],
            "include_analysis_ids": _analysis_ids(context),
            "include_fact_ids": [],
            "include_visualization_ids": [
                visualization["visualization_id"]
                for visualization in context.get("visualizations", [])
            ],
            "section_plan": ["summary", "analysis", "anomaly", "forecast", "visualization"],
        }
        return _turn(
            "I have enough verified evidence, anomaly findings, and forecast output to assemble the answer.",
            "format_answer",
            action_input,
        )


class RepeatingTodoLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        intent_response = _maybe_intent_response(user_prompt)
        if intent_response:
            return intent_response
        context_json = user_prompt.split("Context JSON:\n", 1)[1]
        context = json.loads(context_json)

        latest_observation_summaries = context.get("latest_observation_summaries", [])
        latest_observation = latest_observation_summaries[-1] if latest_observation_summaries else None

        if context.get("latest_database_evidence") is None and not context.get("todo_list"):
            return _turn(
                "I should plan first.",
                "todowrite",
                {
                    "message": "先规划，再查库。",
                    "current_intent": "chat_analysis",
                    "requested_fact_types": ["trend", "anomaly"],
                    "focus": "趋势和异常",
                    "todos": [
                        {"content": "查询时序数据", "task_type": "query", "status": "in_progress", "priority": 1},
                        {"content": "分析趋势", "task_type": "insight", "status": "pending", "priority": 2},
                        {"content": "检查异常", "task_type": "anomaly", "status": "pending", "priority": 3},
                        {"content": "汇总结论", "task_type": "answer", "status": "pending", "priority": 4},
                    ],
                },
            )

        if context.get("latest_database_evidence") is None:
            return _turn(
                "I now need evidence.",
                "sql_query",
                {
                    "message": context["message"],
                    "database_context": context["database_context"],
                    "time_range": context.get("time_range"),
                    "constraints": context.get("constraints", {}),
                },
            )

        if _analysis_count(context) == 0:
            return _turn(
                "I should run generated insight analysis.",
                "insight",
                _analysis_action_input(context["latest_database_evidence"], "趋势和极值"),
            )

        if latest_observation and latest_observation["tool_name"] == "todowrite" and latest_observation["success"] is False:
            return _turn(
                "The runtime rejected another plan update, so I should do anomaly detection next.",
                "anomaly",
                {"constraints": {"zscore_threshold": 2.5}},
            )

        if context.get("latest_anomaly") is None:
            return _turn(
                "Runtime already advanced the plan, so I should do anomaly detection next.",
                "anomaly",
                {"constraints": {"zscore_threshold": 2.5}},
            )

        return _turn(
            "I now have enough outputs.",
            "format_answer",
            {
                "summary_goal": context["message"],
                "include_analysis_ids": _analysis_ids(context),
                "include_fact_ids": [],
                "include_visualization_ids": [
                    viz["visualization_id"] for viz in context.get("visualizations", [])
                ],
                "section_plan": ["summary", "analysis", "anomaly"],
            },
        )


class TodoScopeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        intent_response = _maybe_intent_response(user_prompt)
        if intent_response:
            return intent_response
        context_json = user_prompt.split("Context JSON:\n", 1)[1]
        context = json.loads(context_json)

        if not context.get("todo_list"):
            return _turn(
                "I should create a plan first.",
                "todowrite",
                {
                    "message": "围绕趋势与异常进行分析。",
                    "current_intent": "chat_analysis",
                    "requested_fact_types": ["trend", "anomaly"],
                    "focus": "趋势和异常，不需要预测。",
                    "todos": [
                        {"content": "查询时序数据", "task_type": "query", "status": "completed", "priority": 1},
                        {"content": "提炼趋势事实", "task_type": "insight", "status": "in_progress", "priority": 2},
                        {"content": "检查异常", "task_type": "anomaly", "status": "pending", "priority": 3},
                        {"content": "汇总结论", "task_type": "answer", "status": "pending", "priority": 4},
                    ],
                },
            )

        latest_observation_summaries = context.get("latest_observation_summaries", [])
        latest_observation = latest_observation_summaries[-1] if latest_observation_summaries else None
        if latest_observation and latest_observation["tool_name"] == "forecast" and latest_observation["success"] is False:
            return _turn(
                "Forecast failed because evidence is missing, so I should query the database next.",
                "sql_query",
                {
                    "message": context["message"],
                    "database_context": context["database_context"],
                    "time_range": context.get("time_range"),
                    "constraints": context.get("constraints", {}),
                },
            )

        if latest_observation and latest_observation["tool_name"] == "format_answer" and latest_observation["success"] is False:
            return _turn(
                "The answer assembly reported missing anomaly output, so I should run anomaly detection.",
                "anomaly",
                {
                    "database_evidence": context.get("latest_database_evidence"),
                    "constraints": {"zscore_threshold": 2.5},
                },
            )

        if context.get("latest_database_evidence") is not None and _analysis_count(context) == 0:
            return _turn(
                "I have evidence now and should run generated analysis.",
                "insight",
                _analysis_action_input(context.get("latest_database_evidence"), "趋势"),
            )

        if _analysis_count(context) > 0 and context.get("latest_anomaly") is None:
            return _turn(
                "The user asked for anomalies, so I should run anomaly detection before answering.",
                "anomaly",
                {
                    "database_evidence": context.get("latest_database_evidence"),
                    "constraints": {"zscore_threshold": 2.5},
                },
            )

        if _analysis_count(context) > 0 and context.get("latest_anomaly") is not None:
            return _turn(
                "I have enough facts to answer.",
                "format_answer",
                {
                    "summary_goal": context["message"],
                    "include_analysis_ids": _analysis_ids(context),
                    "include_fact_ids": [],
                    "include_visualization_ids": [
                        viz["visualization_id"] for viz in context.get("visualizations", [])
                    ],
                    "section_plan": ["summary", "analysis", "anomaly"],
                },
            )

        return _turn(
            "I will jump to forecast even though the current step is insight.",
            "forecast",
            {"horizon": 6},
        )


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _turn(thought: str, action: str, action_input: dict) -> _FakeResponse:
    return _FakeResponse(
        json.dumps(
            {
                "thought": thought,
                "action": action,
                "action_input": action_input,
            },
            ensure_ascii=False,
        )
    )


def _maybe_intent_response(user_prompt: str) -> _FakeResponse | None:
    if "Parse the user's data-analysis intent" not in user_prompt:
        return None
    context_json = user_prompt.split("Context JSON:\n", 1)[1]
    context = json.loads(context_json)
    fallback = context.get("fallback_intent_profile") or {}
    message = str(context.get("message") or "")
    normalized = message.lower()
    fact_types = list(fallback.get("requested_fact_types") or [])
    if "最大" in normalized or "max" in normalized:
        fact_types = [item for item in fact_types if item != "outlier"]
        if "extreme" not in fact_types:
            fact_types.append("extreme")
    payload = {
        "primary_goal": message,
        "analysis_kind": "statistical_summary" if "extreme" in fact_types else fallback.get("analysis_kind", "timeseries_analysis"),
        "requested_fact_types": fact_types,
        "requested_metrics": ["max_or_min"] if "extreme" in fact_types else fallback.get("requested_metrics", []),
        "data_policy": {
            "preserve_raw_values": "extreme" in fact_types,
            "filter_outliers": False if "extreme" in fact_types else None,
        },
        "required_outputs": ["conclusion", "analysis"] if "extreme" in fact_types else fallback.get("required_outputs", ["conclusion"]),
        "needs_plan": fallback.get("needs_plan", False),
    }
    return _FakeResponse(json.dumps(payload, ensure_ascii=False))


def _analysis_count(context: dict) -> int:
    workspace = context.get("analysis_workspace") or {}
    return int(workspace.get("analysis_count") or 0)


def _analysis_ids(context: dict) -> list[str]:
    workspace = context.get("analysis_workspace") or {}
    return [
        analysis.get("analysis_id")
        for analysis in workspace.get("analyses", [])
        if analysis.get("analysis_id")
    ]


def _analysis_action_input(evidence, goal: str) -> dict:
    return {
        "database_evidence": evidence,
        "analysis_goal": goal,
        "code_type": "python_rows_v1",
        "analysis_code": (
            "values = [float(row.get('value')) for row in rows if row.get('value') is not None]\n"
            "summary = f'Analyzed {len(rows)} rows.'\n"
            "if values:\n"
            "    summary = f'Analyzed {len(rows)} rows; first={values[0]:.2f}, last={values[-1]:.2f}.'\n"
            "result = {'summary': summary, 'metrics': {'row_count': len(rows)}, 'details': {}}\n"
        ),
        "expected_result_schema": {"summary": "str", "metrics": "dict", "details": "dict"},
    }
