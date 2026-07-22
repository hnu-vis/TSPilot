from __future__ import annotations

import json


def _context_from_prompt(user_prompt: str) -> dict:
    if "Runtime State JSON:\n" in user_prompt:
        context_json = user_prompt.split("Runtime State JSON:\n", 1)[1]
    else:
        context_json = user_prompt.split("Context JSON:\n", 1)[1]
    return _compat_context(json.loads(context_json))


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        runtime_response = _runtime_evaluation_response(user_prompt)
        if runtime_response is not None:
            return runtime_response
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)

        if context.get("database_context") is None:
            return _turn(
                "No datasource is available, so I can only attempt final assembly and let the runtime fail closed.",
                "terminate",
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
                "terminate",
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
            "terminate",
            action_input,
        )


class CasualLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        runtime_response = _runtime_evaluation_response(user_prompt)
        if runtime_response is not None:
            return runtime_response
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)
        return _turn(
            "This is a conversational request without a datasource, so I should answer directly.",
            "terminate",
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
        runtime_response = _runtime_evaluation_response(user_prompt)
        if runtime_response is not None:
            return runtime_response
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)

        if context.get("database_context") is None:
            return _turn(
                "No datasource is available, so I can only attempt final assembly and let the runtime fail closed.",
                "terminate",
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
            "terminate",
            action_input,
        )


class RepeatingTodoLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        runtime_response = _runtime_evaluation_response(user_prompt)
        if runtime_response is not None:
            return runtime_response
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)

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
            "terminate",
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
        runtime_response = _runtime_evaluation_response(user_prompt)
        if runtime_response is not None:
            return runtime_response
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)

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

        if (
            latest_observation
            and latest_observation["tool_name"] in {"format_answer", "terminate"}
            and latest_observation["success"] is False
        ):
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
                "terminate",
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


def _query_generation_response(user_prompt: str) -> _FakeResponse | None:
    if "LLM SQL Query Generation JSON:" not in user_prompt:
        return None
    payload = json.loads(user_prompt.split("LLM SQL Query Generation JSON:\n", 1)[1])
    request = payload.get("request") or {}
    schema_preview = request.get("schema_preview") or {}
    database_type = str(request.get("database_type") or "").lower()
    tables = schema_preview.get("tables_or_measurements") or []
    table = tables[0] if tables and isinstance(tables[0], dict) else {}
    table_name = table.get("name") or "measurement"
    fields = table.get("field_columns") or []
    field = fields[0] if fields else "value"
    time_range = request.get("time_range") or {}
    start = time_range.get("start") or "1970-01-01T00:00:00Z"
    end = time_range.get("end")
    if database_type == "influxdb":
        query = (
            f'from(bucket: "energydata")\n'
            f'  |> range(start: {start}{", stop: " + end if end else ""})\n'
            f'  |> filter(fn: (r) => r._measurement == "{table_name}")\n'
            f'  |> filter(fn: (r) => r._field == "{field}")'
        )
        language = "flux"
    elif database_type == "prometheus":
        query = str(table_name)
        language = "promql"
    else:
        query = f'SELECT "timestamp", "{field}" AS value FROM "{table_name}"'
        language = "sql"
    return _FakeResponse(
        json.dumps(
            {
                "query": query,
                "query_language": language,
                "purpose": "load grounded evidence",
                "expected_result_type": "timeseries",
                "selected_fields": [field],
                "assumptions": [],
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )
    )


def _runtime_evaluation_response(user_prompt: str) -> _FakeResponse | None:
    if "TSPilot Plan Requirement JSON:" in user_prompt:
        payload = json.loads(user_prompt.split("TSPilot Plan Requirement JSON:\n", 1)[1])
        context = payload.get("context") or {}
        message = str(context.get("message") or "")
        requires_plan = "完成以下任务" in message or message.count(";") >= 3
        return _FakeResponse(
            json.dumps(
                {
                    "requires_plan": requires_plan,
                    "reason": "multiple independently verifiable deliverables" if requires_plan else "single-step request",
                    "deliverables": ["count", "head", "tail", "bounds"] if requires_plan else [],
                    "confidence": 0.9,
                    "next_action_hint": "call todowrite" if requires_plan else None,
                },
                ensure_ascii=False,
            )
        )
    if "TSPilot Step Completion JSON:" in user_prompt:
        payload = json.loads(user_prompt.split("TSPilot Step Completion JSON:\n", 1)[1])
        context = payload.get("context") or {}
        tool_payload = context.get("tool_payload") or {}
        evidence_id = tool_payload.get("evidence_id")
        return _FakeResponse(
            json.dumps(
                {
                    "completed": bool(evidence_id or context.get("tool_name") in {"insight", "anomaly", "forecast", "format_answer", "terminate"}),
                    "reason": "latest tool output satisfies the active todo",
                    "missing_items": [],
                    "satisfied_items": ["active_todo"],
                    "evidence_refs": [f"evidence:{evidence_id}"] if evidence_id else [],
                    "next_action_hint": None,
                    "confidence": 0.9,
                },
                ensure_ascii=False,
            )
        )
    if "TSPilot Answerability JSON:" in user_prompt:
        return _FakeResponse(
            json.dumps(
                {
                    "can_answer": True,
                    "reason": "available outputs are sufficient for this test scenario",
                    "missing_items": [],
                    "answerable_from": ["evidence:latest"],
                    "next_action_hint": None,
                    "confidence": 0.9,
                },
                ensure_ascii=False,
            )
        )
    return None


def _compat_context(context: dict) -> dict:
    if "task" not in context:
        return context
    task = context.get("task") or {}
    state = context.get("state") or {}
    evidence = context.get("evidence") or {}
    outputs = context.get("outputs") or {}
    execution = state.get("execution") or {}
    return {
        **context,
        "message": task.get("message"),
        "database_context": task.get("database_context"),
        "selected_database": task.get("selected_database"),
        "selected_database_type": task.get("selected_database_type"),
        "time_range": task.get("time_range"),
        "constraints": task.get("constraints") or {},
        "history": task.get("history") or [],
        "execution_state": execution,
        "todo_list": state.get("todo_list"),
        "plan_current_step": state.get("plan_current_step"),
        "planning_complete": state.get("planning_complete"),
        "requested_fact_types": state.get("requested_fact_types"),
        "focus": state.get("focus"),
        "latest_database_evidence": evidence.get("latest"),
        "query_history": evidence.get("prior_queries") or [],
        "latest_insight": outputs.get("latest_insight"),
        "analysis_workspace": outputs.get("analysis_workspace") or {},
        "latest_forecast": outputs.get("latest_forecast"),
        "latest_anomaly": outputs.get("latest_anomaly"),
        "latest_rag": outputs.get("latest_rag"),
        "latest_skill": outputs.get("latest_skill"),
        "verified_facts": outputs.get("verified_facts") or [],
        "visualizations": outputs.get("visualizations") or [],
        "latest_observation_summaries": context.get("recent_observations") or [],
        "available_actions": context.get("available_actions") or [],
    }


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
