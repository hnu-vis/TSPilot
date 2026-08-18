from __future__ import annotations

import json

_LAST_CONTEXT: dict | None = None


def _context_from_prompt(user_prompt: str) -> dict:
    global _LAST_CONTEXT
    if "Runtime State JSON:\n" in user_prompt:
        context_json = user_prompt.split("Runtime State JSON:\n", 1)[1]
        _LAST_CONTEXT = _compat_context(json.loads(context_json))
        return _LAST_CONTEXT
    if "Context JSON:\n" in user_prompt:
        context_json = user_prompt.split("Context JSON:\n", 1)[1]
        _LAST_CONTEXT = _compat_context(json.loads(context_json))
        return _LAST_CONTEXT
    task = _section_json(user_prompt, "User Task:", "Available Tools:")
    outer_state = _section_json(user_prompt, "Outer ReAct State:", None)
    _LAST_CONTEXT = _compat_context({"task": task, **outer_state})
    return _LAST_CONTEXT


def _section_json(text: str, start_marker: str, end_marker: str | None) -> dict:
    if start_marker not in text:
        return {}
    chunk = text.split(start_marker, 1)[1]
    if end_marker and end_marker in chunk:
        chunk = chunk.split(end_marker, 1)[0]
    chunk = chunk.strip()
    if not chunk:
        return {}
    return json.loads(chunk)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)

        if context.get("database_context") is None:
            return _turn(
                "No datasource is available, so I can only attempt final assembly and let the runtime fail closed.",
                "terminate",
                {"summary_goal": context["message"], "include_insight_ids": [], "include_visualization_ids": [], "section_plan": []},
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
            return _turn("I need evidence before producing insights.", "sql_query", action_input)

        latest_evidence = context.get("latest_database_evidence") or {}
        result_type = latest_evidence.get("result_type")
        if result_type in {"schema", "metric_list", "statistics", "table"}:
            action_input = {
                "summary_goal": context["message"],
                "include_insight_ids": [],
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
                "code_interpreter",
                action_input,
            )

        analysis_ids = _analysis_ids(context)
        action_input = {
            "summary_goal": context["message"],
            "include_analysis_ids": analysis_ids,
            "include_insight_ids": [],
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


class SandboxAnalysisLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)

        if _analysis_count(context) > 0:
            return _turn(
                "The code interpreter result is available, so I can assemble the final answer.",
                "terminate",
                {
                    "summary_goal": context["message"],
                    "include_analysis_ids": _analysis_ids(context),
                    "include_insight_ids": [],
                    "include_visualization_ids": [],
                    "section_plan": ["summary", "analysis"],
                },
            )

        if context.get("latest_database_evidence") is None:
            return _turn(
                "I need grounded database rows before running the code interpreter.",
                "sql_query",
                {
                    "message": context["message"],
                    "database_context": context["database_context"],
                    "time_range": context.get("time_range"),
                    "constraints": context.get("constraints", {}),
                },
            )

        latest_evidence = context.get("latest_database_evidence") or {}
        if _analysis_count(context) == 0:
            return _turn(
                "The SQL evidence is available, and this example needs the subprocess code interpreter.",
                "code_interpreter",
                {
                    "database_evidence": latest_evidence,
                    "analysis_goal": "compute pairwise deltas with the code interpreter",
                    "insight_requests": [{"insight_key": "pairwise_deltas", "name": "Pairwise deltas", "insight_type": "distribution"}],
                    "code": (
                        "values = []\n"
                        "for row in rows:\n"
                        "    for key, raw_value in row.items():\n"
                        "        if key in {'timestamp', 'time', '_time'}:\n"
                        "            continue\n"
                        "        if isinstance(raw_value, (int, float)):\n"
                        "            values.append(float(raw_value))\n"
                        "            break\n"
                        "deltas = [right - left for left, right in zip(values, values[1:])]\n"
                        "result = {\n"
                        "    'computed_insights': [{'insight_key': 'pairwise_deltas', 'value': {'value_count': len(values), 'delta_count': len(deltas), 'max_delta': max(deltas) if deltas else None, 'min_delta': min(deltas) if deltas else None, 'first_three_deltas': deltas[:3]}, 'calculation_trace': {'operation': 'adjacent difference'}}],\n"
                        "    'derived_evidence': [],\n"
                        "}\n"
                    ),
                    "constraints": {"timeout_seconds": 5},
                },
            )

        return _turn(
            "The code interpreter result is available, so I can assemble the final answer.",
            "terminate",
            {
                "summary_goal": context["message"],
                "include_analysis_ids": _analysis_ids(context),
                "include_insight_ids": [],
                "include_visualization_ids": [],
                "section_plan": ["summary", "analysis"],
            },
        )


class CasualLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
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
                "include_insight_ids": [],
                "include_visualization_ids": [],
                "section_plan": ["summary"],
            },
        )


class CodeRequiredRepairLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)

        if _analysis_count(context) > 0:
            return _turn(
                "The financial metrics are available.",
                "terminate",
                {
                    "summary_goal": context["message"],
                    "include_analysis_ids": _analysis_ids(context),
                    "include_insight_ids": [],
                    "section_plan": ["summary", "analysis"],
                },
            )

        latest_code_failure = _latest_failed_observation(context, "code_interpreter")
        if latest_code_failure is not None or _requires_code_repair(context):
            return _turn(
                "The template cannot compute these metrics, so I need generated code.",
                "code_interpreter",
                {
                    "database_evidence": "latest",
                    "analysis_goal": "compute total return volatility and max drawdown",
                    "insight_requests": _financial_insight_requests(),
                    "code": _financial_metrics_code(),
                    "constraints": {"timeout_seconds": 5},
                },
            )

        if context.get("latest_database_evidence") is None:
            return _turn(
                "I need rows before computing financial metrics.",
                "sql_query",
                {
                    "message": context["message"],
                    "database_context": context["database_context"],
                    "time_range": context.get("time_range"),
                    "constraints": context.get("constraints", {}),
                },
            )

        return _turn(
            "I will first ask for the requested metrics through the analysis request.",
            "code_interpreter",
            {
                "database_evidence": "latest",
                "analysis_goal": "compute total return volatility and max drawdown",
                "insight_requests": _financial_insight_requests(),
                "code": _financial_metrics_code(),
            },
        )


class ComplexReActLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        query_response = _query_generation_response(user_prompt)
        if query_response is not None:
            return query_response
        context = _context_from_prompt(user_prompt)

        if context.get("database_context") is None:
            return _turn(
                "No datasource is available, so I can only attempt final assembly and let the runtime fail closed.",
                "terminate",
                {"summary_goal": context["message"], "include_insight_ids": [], "include_visualization_ids": [], "section_plan": []},
            )

        if not context.get("todo_list"):
            action_input = {
                "message": context["message"],
                "current_intent": "chat_analysis",
                "requested_capabilities": ["query", "analysis", "anomaly", "forecast"],
                "focus": "先规划，再查库，随后做趋势、异常和预测分析，最后整合回答。",
                "todos": [
                    {"content": "查询目标时间范围内的时序证据", "task_type": "query", "status": "in_progress", "priority": 1},
                    {"content": "提炼趋势事实", "task_type": "code_interpreter", "status": "pending", "priority": 2},
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
                "code_interpreter",
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
            "include_insight_ids": [],
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
                    "requested_capabilities": ["query", "analysis", "anomaly"],
                    "focus": "趋势和异常",
                    "todos": [
                        {"content": "查询时序数据", "task_type": "query", "status": "in_progress", "priority": 1},
                        {"content": "分析趋势", "task_type": "code_interpreter", "status": "pending", "priority": 2},
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
                "I should run generated code_interpreter analysis.",
                "code_interpreter",
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
                "include_insight_ids": [],
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
                    "requested_capabilities": ["query", "analysis", "anomaly"],
                    "focus": "趋势和异常，不需要预测。",
                    "todos": [
                        {"content": "查询时序数据", "task_type": "query", "status": "completed", "priority": 1},
                        {"content": "提炼趋势事实", "task_type": "code_interpreter", "status": "in_progress", "priority": 2},
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
            and latest_observation["tool_name"] == "terminate"
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
                "code_interpreter",
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
                "I have enough insights to answer.",
                "terminate",
                {
                    "summary_goal": context["message"],
                    "include_analysis_ids": _analysis_ids(context),
                    "include_insight_ids": [],
                    "include_visualization_ids": [
                        viz["visualization_id"] for viz in context.get("visualizations", [])
                    ],
                    "section_plan": ["summary", "analysis", "anomaly"],
                },
            )

        return _turn(
            "I will jump to forecast even though the current step is code_interpreter.",
            "forecast",
            {"horizon": 6},
        )


class BitcoinMultiQueryLLM:
    def __init__(self):
        self.calls = 0
        self.agent_turn = 0
        self._pending_query: tuple[str, str] | None = None

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
        if any(
            label in user_prompt
            for label in ("LLM SQL Query Generation JSON:\n", "LLM 查询生成输入 JSON：\n")
        ):
            if self._pending_query is None:
                raise AssertionError("SQL generation was invoked without a pending natural-language query goal.")
            purpose, query = self._pending_query
            self._pending_query = None
            return _FakeResponse(json.dumps({
                "query": query,
                "query_language": "flux",
                "purpose": purpose,
                "expected_result_type": "table",
                "selected_fields": [],
                "assumptions": [],
                "confidence": 0.99,
            }, ensure_ascii=False))
        internal_response = _query_generation_response(user_prompt)
        if internal_response is not None:
            return internal_response
        _context_from_prompt(user_prompt)

        self.agent_turn += 1
        database_context = {
            "database_id": "influxdb2-bitcoin-sample",
            "database_type": "influxdb",
        }
        base_flux = (
            'from(bucket: "bitcoin")\n'
            "  |> range(start: 0)\n"
            '  |> filter(fn: (r) => r._measurement == "coindesk")\n'
            '  |> filter(fn: (r) => r.code == "USD")\n'
            '  |> filter(fn: (r) => r.crypto == "bitcoin")\n'
            '  |> filter(fn: (r) => r._field == "price")'
        )
        queries = {
            2: (
                "返回USD价格数据的总记录数",
                base_flux
                + '\n  |> count()\n  |> keep(columns: ["_value"])\n  |> rename(columns: {_value: "count"})',
            ),
            3: (
                "返回按时间升序排列的最早5条原始记录",
                base_flux
                + '\n  |> sort(columns: ["_time"], desc: false)\n  |> limit(n: 5)\n'
                + '  |> keep(columns: ["_time", "_value", "code", "crypto", "description", "symbol"])\n'
                + '  |> rename(columns: {_value: "price"})',
            ),
            4: (
                "返回按时间降序排列的最晚5条原始记录",
                base_flux
                + '\n  |> sort(columns: ["_time"], desc: true)\n  |> limit(n: 5)\n'
                + '  |> keep(columns: ["_time", "_value", "code", "crypto", "description", "symbol"])\n'
                + '  |> rename(columns: {_value: "price"})',
            ),
            5: (
                "返回整个数据集的最早时间和最晚时间，精确到秒",
                "earliest = "
                + base_flux
                + '\n  |> first()\n  |> map(fn: (r) => ({ r with bound: "earliest" }))\n'
                + "latest = "
                + base_flux
                + '\n  |> last()\n  |> map(fn: (r) => ({ r with bound: "latest" }))\n'
                + 'union(tables: [earliest, latest])\n  |> keep(columns: ["bound", "_time", "_value"])\n'
                + '  |> rename(columns: {_value: "price"})',
            ),
        }

        if self.agent_turn == 1:
            return _turn(
                "I should plan the independent query deliverables first.",
                "todowrite",
                {
                    "message": "查询比特币 USD 价格多分项结果",
                    "current_intent": "database_query",
                    "requested_capabilities": ["query", "analysis"],
                    "focus": "每项都必须有查询语句和实际返回行数",
                    "todos": [
                        {"content": "查询总记录数", "task_type": "query", "status": "in_progress", "priority": 1},
                        {"content": "查询最早5条", "task_type": "query", "status": "pending", "priority": 2},
                        {"content": "查询最晚5条", "task_type": "query", "status": "pending", "priority": 3},
                        {"content": "查询时间边界", "task_type": "query", "status": "pending", "priority": 4},
                        {"content": "汇总最终答案", "task_type": "answer", "status": "pending", "priority": 5},
                    ],
                },
            )
        if self.agent_turn in queries:
            purpose, query = queries[self.agent_turn]
            self._pending_query = (purpose, query)
            return _turn(
                purpose,
                "sql_query",
                {
                    "message": purpose,
                    "purpose": purpose,
                },
            )
        return _turn(
            "I have enough database evidence artifacts to assemble the answer.",
            "terminate",
            {
                "summary_goal": "汇总比特币 USD 价格多分项查询结果",
                "direct_answer": "已完成全部分项查询，以下结果均来自实际数据库查询 evidence。",
                "include_insight_ids": [],
                "include_visualization_ids": [],
                "section_plan": ["summary", "query_results", "conclusion"],
            },
        )


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


def _required_action_names(context: dict) -> list[str]:
    state = context.get("state") if isinstance(context.get("state"), dict) else {}
    constraints = context.get("next_action_constraints")
    if not isinstance(constraints, dict):
        constraints = state.get("next_action_constraints") if isinstance(state, dict) else {}
    if not isinstance(constraints, dict):
        return []
    return [
        str(item.get("action"))
        for item in constraints.get("required_actions") or []
        if isinstance(item, dict) and item.get("action")
    ]


def _required_action_guidance(context: dict, action: str) -> dict:
    state = context.get("state") if isinstance(context.get("state"), dict) else {}
    constraints = context.get("next_action_constraints")
    if not isinstance(constraints, dict):
        constraints = state.get("next_action_constraints") if isinstance(state, dict) else {}
    if not isinstance(constraints, dict):
        return {}
    for item in constraints.get("required_actions") or []:
        if isinstance(item, dict) and item.get("action") == action:
            guidance = item.get("input_guidance")
            return dict(guidance) if isinstance(guidance, dict) else {}
    return {}


def _artifact_ref_values(context: dict, kind: str) -> list[str]:
    artifacts = context.get("artifacts") if isinstance(context.get("artifacts"), dict) else {}
    refs = artifacts.get("refs") if isinstance(artifacts.get("refs"), dict) else {}
    raw = refs.get(kind)
    values = raw if isinstance(raw, list) else [raw] if raw else []
    return [str(item) for item in values if isinstance(item, str) and item]


def _artifact_facts(context: dict, kind: str | None = None) -> list[dict]:
    artifacts = context.get("artifacts") if isinstance(context.get("artifacts"), dict) else {}
    facts = artifacts.get("facts") if isinstance(artifacts.get("facts"), list) else []
    return [
        item for item in facts
        if isinstance(item, dict) and (kind is None or item.get("kind") == kind)
    ]


def _verified_insight_summary(context: dict) -> str | None:
    lines = []
    for insight in _verified_insights(context):
        statement = str(insight.get("statement") or insight.get("name") or insight.get("insight_key") or "Insight")
        if "value" in insight:
            statement += f" Value: {json.dumps(insight['value'], ensure_ascii=False, default=str)}"
        if insight.get("items"):
            statement += f" Items: {json.dumps(insight['items'], ensure_ascii=False, default=str)}"
        lines.append(statement)
    return "\n".join(lines) or None


def _query_results_text(context: dict) -> str | None:
    blocks = []
    for fact in _artifact_facts(context, "database_evidence"):
        purpose = str(fact.get("purpose") or fact.get("summary") or fact.get("source_ref") or "Database query")
        records = [item for item in fact.get("records") or [] if isinstance(item, dict)]
        columns = [str(item) for item in fact.get("columns") or []]
        if not columns and records:
            columns = list(records[0])
        block = [f"查询目的：{purpose}", f"实际返回行数：{int(fact.get('row_count') or 0)}"]
        query = str(fact.get("query") or "")
        if query:
            block.extend([f"```{fact.get('query_language') or 'text'}", query, "```"])
        if len(records) == 1 and len(records[0]) == 1:
            key, value = next(iter(records[0].items()))
            block.append(f"结果值：{key} = {value}")
        elif records and columns:
            block.append("| " + " | ".join(columns) + " |")
            block.append("| " + " | ".join("---" for _ in columns) + " |")
            for record in records:
                block.append("| " + " | ".join(str(record.get(column, "")) for column in columns) + " |")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks) or None


def _latest_artifact_ref(refs: dict, kind: str) -> str | None:
    raw = refs.get(kind)
    values = raw if isinstance(raw, list) else [raw] if raw else []
    return next((str(item) for item in reversed(values) if isinstance(item, str) and item), None)


def _verified_insights(context: dict) -> list[dict]:
    state = context.get("state") if isinstance(context.get("state"), dict) else {}
    insight_state = state.get("insight_state") if isinstance(state.get("insight_state"), dict) else {}
    return [
        item
        for item in insight_state.get("recent_insights") or []
        if isinstance(item, dict) and item.get("status") == "verified"
    ]


def _verified_insight_keys(context: dict) -> list[str]:
    return [str(item["insight_key"]) for item in _verified_insights(context) if item.get("insight_key")]


def _visualization_source_refs(context: dict) -> list[str]:
    refs = [
        f"insight:{item['insight_id']}"
        for item in _verified_insights(context)
        if item.get("insight_id")
    ]
    for kind in ("database_evidence", "analysis", "derived_evidence", "forecast", "anomaly"):
        refs.extend(_artifact_ref_values(context, kind))
    return list(dict.fromkeys(refs))


def _turn(thought: str, action: str, action_input: dict) -> _FakeResponse:
    context = _LAST_CONTEXT or {}
    required_actions = _required_action_names(context)
    if "terminate" in required_actions and action != "terminate":
        guidance = _required_action_guidance(context, "terminate")
        thought = "The grounded outputs are answerable and the latest receipt requires final assembly."
        action = "terminate"
        action_input = {
            "summary_goal": str(context.get("message") or "Answer from grounded artifacts."),
            **guidance,
        }
    elif "visualization" in required_actions and action != "visualization":
        source_refs = _visualization_source_refs(context)
        insight_keys = _verified_insight_keys(context)
        thought = (
            f"Verified Insights: {', '.join(insight_keys) or 'none'}; "
            "Verification question: does the complete contextual series support the calculated relationship; "
            f"Context: {', '.join(source_refs) or 'no grounded refs'} with complete interval coverage."
        )
        action = "visualization"
        action_input = {
            "message": str(context.get("message") or "Verify the grounded analytical relationship."),
            "source_refs": source_refs,
            "constraints": dict(context.get("constraints") or {}),
        }
    if action == "terminate" and "response_plan" not in action_input:
        presentation = ((context.get("artifacts") or {}).get("presentation") or {})
        presentation_sources = [
            item for item in presentation.get("sources") or [] if isinstance(item, dict) and item.get("source_ref")
        ]
        source_refs: list[str] = [str(item["source_ref"]) for item in presentation_sources]
        source_refs.extend(_artifact_ref_values(context, "analysis"))
        source_refs.extend(_artifact_ref_values(context, "database_evidence"))
        latest_evidence = context.get("latest_database_evidence") or {}
        if not source_refs and isinstance(latest_evidence, dict) and latest_evidence.get("evidence_id"):
            source_refs.append(f"evidence:{latest_evidence['evidence_id']}")
        if not source_refs:
            source_refs.extend(f"analysis:{analysis_id}" for analysis_id in _analysis_ids(context))
        latest_forecast = context.get("latest_forecast") or {}
        if not presentation_sources and isinstance(latest_forecast, dict) and latest_forecast.get("forecast_id"):
            source_refs.append(f"forecast:{latest_forecast['forecast_id']}")
        latest_anomaly = context.get("latest_anomaly") or {}
        if not presentation_sources and isinstance(latest_anomaly, dict) and latest_anomaly.get("anomaly_id"):
            source_refs.append(f"anomaly:{latest_anomaly['anomaly_id']}")
        grounded_summary = next(
            (
                str(item.get("summary"))
                for item in reversed(presentation_sources)
                if item.get("kind") in {"analysis", "forecast", "anomaly"} and item.get("summary")
            ),
            None,
        )
        summary = str(
            action_input.get("direct_answer")
            or grounded_summary
            or _verified_insight_summary(context)
            or action_input.get("summary_goal")
            or context.get("message")
            or "Answer assembled from the available evidence."
        )
        source_kinds = {str(item.get("kind")) for item in presentation_sources}
        requested_sections = [str(item) for item in action_input.get("section_plan") or []]
        section_type = (
            "query_results" if "query_results" in requested_sections
            else "forecast" if "forecast" in source_kinds or (isinstance(latest_forecast, dict) and latest_forecast.get("forecast_id"))
            else "anomaly" if "anomaly" in source_kinds or (isinstance(latest_anomaly, dict) and latest_anomaly.get("anomaly_id"))
            else "analysis" if "analysis" in source_kinds or _analysis_ids(context)
            else "answer"
        )
        if section_type == "query_results":
            sections = [{
                "section_type": "query_results",
                "heading": None,
                "content": _query_results_text(context) or summary,
                "source_refs": _artifact_ref_values(context, "database_evidence"),
            }]
        else:
            sections = []
            for artifact_kind, result_section in (
                ("analysis", "analysis"),
                ("anomaly", "anomaly"),
                ("forecast", "forecast"),
            ):
                artifact_refs = _artifact_ref_values(context, artifact_kind)
                if artifact_refs:
                    sections.append({
                        "section_type": result_section,
                        "heading": None,
                        "content": summary,
                        "source_refs": artifact_refs,
                    })
            if not sections:
                sections = [{
                    "section_type": section_type,
                    "heading": None,
                    "content": summary,
                    "source_refs": list(dict.fromkeys(source_refs)),
                }]
        unavailable_outputs = list(action_input.get("unavailable_outputs") or [])
        unavailable_reason = action_input.get("unavailable_reason")
        latest_observation = (context.get("latest_observation_summaries") or [{}])[-1]
        latest_payload = latest_observation.get("payload") if isinstance(latest_observation, dict) else None
        latest_payload = latest_payload if isinstance(latest_payload, dict) else latest_observation
        if (
            isinstance(latest_observation, dict)
            and latest_observation.get("tool_name") == "visualization"
            and isinstance(latest_payload, dict)
            and latest_payload.get("status") == "unavailable"
        ):
            if "visualization" not in unavailable_outputs:
                unavailable_outputs.append("visualization")
            unavailable_reason = str(
                latest_payload.get("unavailable_reason")
                or latest_observation.get("summary")
                or "Visual verification was unavailable."
            )
        action_input = {
            "response_plan": {
                "title": None,
                "summary": summary,
                "sections": sections,
                "visualization_ids": [
                    item.get("visualization_id")
                    for item in context.get("visualizations") or []
                    if isinstance(item, dict) and item.get("visualization_id")
                ],
            },
            "unavailable_outputs": unavailable_outputs,
            "unavailable_reason": unavailable_reason,
        }
    payload = {
        "thought": thought,
        "previous_observation_assessment": _auto_previous_observation_assessment(_LAST_CONTEXT),
        "action": action,
        "action_input": action_input,
    }
    return _FakeResponse(json.dumps(payload, ensure_ascii=False))


def _auto_previous_observation_assessment(context: dict | None) -> dict | None:
    if not context:
        return None
    observations = context.get("latest_observation_summaries") or []
    if not observations:
        return None
    latest = observations[-1]
    if not latest.get("success") or latest.get("tool_name") in {"todowrite", "todo_assessment"}:
        return None
    active_todo = next(
        (todo for todo in context.get("todo_list") or [] if todo.get("status") == "in_progress"),
        None,
    )
    if not active_todo:
        return None
    task_type = str(active_todo.get("task_type") or "").lower()
    tool_name = str(latest.get("tool_name") or "")
    if task_type == "query" and tool_name != "sql_query":
        return None
    if task_type == "answer" and tool_name != "terminate":
        return None
    if task_type not in {"", "generic", "query", "answer"} and tool_name != task_type:
        return None
    payload = latest.get("payload") if isinstance(latest.get("payload"), dict) else {}
    return {
        "completed_active_todo": True,
        "reason": f"Previous {tool_name} observation satisfies active todo: {active_todo.get('content')}.",
        "evidence_refs": _evidence_refs_from_payload(payload),
    }


def _evidence_refs_from_payload(payload: dict) -> list[str]:
    refs = []
    for key, prefix in (
        ("evidence_id", "evidence"),
        ("analysis_id", "analysis"),
        ("forecast_id", "forecast"),
        ("anomaly_id", "anomaly"),
    ):
        value = payload.get(key)
        if value:
            refs.append(f"{prefix}:{value}")
    return refs


def _query_generation_response(user_prompt: str) -> _FakeResponse | None:
    if (
        "User Task:" not in user_prompt
        and "Outer ReAct State:" not in user_prompt
        and "visualization" in _required_action_names(_LAST_CONTEXT or {})
    ):
        return _FakeResponse(json.dumps({
            "decision": "not_visualizable",
            "target_insight_ids": [],
            "verification_question": None,
            "interpretation": "The fake unit-test model cannot inspect a rendered visual relationship.",
            "visual_relation": None,
            "required_context": [],
            "non_visual_insight_ids": [],
            "required_data_request": None,
        }, ensure_ascii=False))
    try:
        internal_payload = json.loads(user_prompt)
    except (TypeError, json.JSONDecodeError):
        internal_payload = None
    if (
        isinstance(internal_payload, dict)
        and isinstance(internal_payload.get("todos"), list)
        and "user_request" in internal_payload
    ):
        bindings = []
        todos = internal_payload["todos"]
        for position, item in enumerate(todos):
            content = str(item.get("content") or "").lower() if isinstance(item, dict) else ""
            if any(token in content for token in ("查询", "查库", "query", "retrieve")):
                task_type = "query"
            elif any(token in content for token in ("异常", "anomal")):
                task_type = "anomaly"
            elif any(token in content for token in ("预测", "forecast", "predict")):
                task_type = "forecast"
            elif any(token in content for token in ("可视", "图表", "visual", "chart")):
                task_type = "visualization"
            elif position == len(todos) - 1 or any(token in content for token in ("总结", "汇总", "answer")):
                task_type = "answer"
            else:
                task_type = "code_interpreter"
            bindings.append({
                "index": int(item.get("index", position)) if isinstance(item, dict) else position,
                "task_type": task_type,
                "reason": f"The Todo's acceptance result belongs to {task_type}.",
            })
        return _FakeResponse(json.dumps({"bindings": bindings}, ensure_ascii=False))
    if (
        isinstance(internal_payload, dict)
        and isinstance(internal_payload.get("insight_requests"), list)
        and "canonical_inputs" in internal_payload
        and "artifact_sources" in internal_payload
    ):
        requested_keys = [
            str(item.get("insight_key"))
            for item in internal_payload["insight_requests"]
            if isinstance(item, dict) and item.get("insight_key")
        ]
        computed_entries = ",\n".join(
            "        " + repr({
                "insight_key": insight_key,
                "value": {
                    "row_count": "__ROW_COUNT__",
                    "first_value": "__FIRST_VALUE__",
                    "last_value": "__LAST_VALUE__",
                },
                "calculation_trace": {
                    "operation": "summarize canonical numeric observations",
                    "input": "rows",
                },
            })
            for insight_key in requested_keys
        )
        computed_entries = (
            computed_entries
            .replace("'__ROW_COUNT__'", "len(rows)")
            .replace("'__FIRST_VALUE__'", "values[0] if values else None")
            .replace("'__LAST_VALUE__'", "values[-1] if values else None")
        )
        code = (
            "values = []\n"
            "for row in rows:\n"
            "    for field_name, raw_value in row.items():\n"
            "        if field_name in {'timestamp', 'time', '_time'}:\n"
            "            continue\n"
            "        if isinstance(raw_value, (int, float)):\n"
            "            values.append(float(raw_value))\n"
            "            break\n"
            "result = {\n"
            "    'computed_insights': [\n"
            f"{computed_entries}\n"
            "    ],\n"
            "    'derived_evidence': [],\n"
            "}\n"
        )
        return _FakeResponse(json.dumps({"code": code}, ensure_ascii=False))
    if isinstance(internal_payload, dict) and isinstance(internal_payload.get("computed_insights"), list):
        requests_by_key = {
            item.get("insight_key"): item
            for item in internal_payload.get("requests", [])
            if isinstance(item, dict) and item.get("insight_key")
        }
        return _FakeResponse(json.dumps({
            "bindings": [
                {
                    "insight_key": item.get("insight_key"),
                    "supported": True,
                    "unsupported_reason": None,
                    "statement": f"Computed {item.get('insight_key')} from grounded evidence.",
                    "derived_from": requests_by_key.get(item.get("insight_key"), {}).get("derived_from", []),
                    "item_annotations": [],
                }
                for item in internal_payload["computed_insights"]
                if isinstance(item, dict) and item.get("insight_key")
            ]
        }, ensure_ascii=False))
    schema_label = next(
        (label for label in ("LLM Schema Linking JSON:\n", "LLM 模式映射输入 JSON：\n") if label in user_prompt),
        None,
    )
    if schema_label:
        payload = json.loads(user_prompt.split(schema_label, 1)[1])
        schema_preview = payload.get("schema_preview") or {}
        tables = schema_preview.get("tables_or_measurements") or []
        table = tables[0] if tables and isinstance(tables[0], dict) else {}
        table_name = table.get("name") or "measurement"
        fields = _schema_value_fields(table)
        message = str(payload.get("message") or "")
        field = next((name for name in fields if name in message), fields[0] if fields else "value")
        return _FakeResponse(
            json.dumps(
                {
                    "sources": [{"name": table_name, "source_type": "measurement"}],
                    "value_columns": [{"name": field, "source": table_name}],
                    "dimension_columns": [],
                    "required_filters": [],
                    "candidate_filters": [],
                    "unresolved_terms": [],
                    "confidence": "high",
                    "evidence": ["fake schema linking selected the first schema source and value column"],
                },
                ensure_ascii=False,
            )
        )
    query_label = next(
        (label for label in ("LLM SQL Query Generation JSON:\n", "LLM 查询生成输入 JSON：\n") if label in user_prompt),
        None,
    )
    if query_label is None:
        return None
    payload = json.loads(user_prompt.split(query_label, 1)[1])
    request = payload.get("request") or {}
    schema_preview = request.get("schema_preview") or {}
    database_type = str(request.get("database_type") or "").lower()
    tables = schema_preview.get("tables_or_measurements") or []
    table = tables[0] if tables and isinstance(tables[0], dict) else {}
    table_name = table.get("name") or "measurement"
    linking = schema_preview.get("schema_linking") if isinstance(schema_preview.get("schema_linking"), dict) else {}
    linked_values = linking.get("value_columns") if isinstance(linking.get("value_columns"), list) else []
    linked_names = [str(item.get("name")) for item in linked_values if isinstance(item, dict) and item.get("name")]
    fields = linked_names or _schema_value_fields(table)
    message = str(request.get("message") or "")
    field = next((name for name in fields if name in message), fields[0] if fields else "value")
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


def _schema_value_fields(table: dict) -> list[str]:
    values = table.get("field_values") or table.get("field_columns") or []
    fields = [str(item) for item in values if str(item).strip()]
    if fields:
        return fields
    columns = table.get("columns") if isinstance(table.get("columns"), list) else []
    return [
        str(item.get("name"))
        for item in columns
        if isinstance(item, dict)
        and item.get("name")
        and str(item.get("name")).lower() not in {"time", "timestamp", "_time"}
    ]


def _compat_context(context: dict) -> dict:
    if "task" not in context:
        return context
    task = context.get("task") or {}
    state = context.get("state") or {}
    evidence = context.get("evidence") or {}
    outputs = context.get("outputs") or {}
    artifacts = context.get("artifacts") or {}
    refs = artifacts.get("refs") if isinstance(artifacts.get("refs"), dict) else {}
    presentation = artifacts.get("presentation") if isinstance(artifacts.get("presentation"), dict) else {}
    presentation_sources = [item for item in presentation.get("sources") or [] if isinstance(item, dict)]
    execution = state.get("execution") or {}
    observations = _observation_summaries(context)
    latest_observation = observations[-1] if observations else {}
    latest_observation_payload = latest_observation.get("payload") if isinstance(latest_observation.get("payload"), dict) else latest_observation
    latest_sql_observation = next(
        (
            item
            for item in reversed(observations)
            if item.get("tool_name") == "sql_query" and item.get("success") is not False
        ),
        None,
    )
    latest_sql_payload = (
        latest_sql_observation.get("payload")
        if isinstance(latest_sql_observation, dict) and isinstance(latest_sql_observation.get("payload"), dict)
        else latest_sql_observation
    )
    latest_evidence = (
        evidence.get("latest")
        or _latest_ref_payload(refs, "database_evidence")
        or _latest_ref_payload(refs, "evidence")
        or latest_sql_payload
    )
    if not latest_evidence:
        presentation_evidence = next((item for item in reversed(presentation_sources) if item.get("kind") == "evidence"), None)
        if presentation_evidence:
            source_ref = str(presentation_evidence.get("source_ref") or "")
            latest_evidence = {
                **presentation_evidence,
                "evidence_id": source_ref.split(":", 1)[1] if source_ref.startswith("evidence:") else source_ref,
            }
    if not latest_evidence and latest_observation.get("tool_name") == "sql_query" and latest_observation.get("success") is not False:
        latest_evidence = latest_observation_payload
    if not latest_evidence:
        evidence_ref = _latest_artifact_ref(refs, "database_evidence") or _latest_artifact_ref(refs, "evidence")
        if evidence_ref:
            latest_evidence = {
                "evidence_id": evidence_ref.split(":", 1)[-1],
                "resource_ref": evidence_ref,
                "result_type": "timeseries",
            }
    analyses = _ref_payloads(refs, "analysis")
    if not analyses:
        analyses = [
            {
                **item,
                "analysis_id": str(item.get("source_ref") or "").split(":", 1)[-1],
            }
            for item in presentation_sources
            if item.get("kind") == "analysis"
        ]
    if latest_observation.get("tool_name") == "code_interpreter" and latest_observation.get("success") is not False:
        analyses.append(latest_observation_payload)
    if not analyses:
        analyses = _analysis_refs(refs)
    analysis_count = len(analyses)
    latest_anomaly = outputs.get("latest_anomaly") or _latest_ref_payload(refs, "anomaly")
    if not latest_anomaly and latest_observation.get("tool_name") == "anomaly" and latest_observation.get("success") is not False:
        latest_anomaly = latest_observation_payload
    if not latest_anomaly:
        anomaly_ref = _latest_artifact_ref(refs, "anomaly")
        if anomaly_ref:
            latest_anomaly = {
                "anomaly_id": anomaly_ref.split(":", 1)[-1],
                "resource_ref": anomaly_ref,
            }
    latest_forecast = outputs.get("latest_forecast") or _latest_ref_payload(refs, "forecast")
    if not latest_forecast and latest_observation.get("tool_name") == "forecast" and latest_observation.get("success") is not False:
        latest_forecast = latest_observation_payload
    if not latest_forecast:
        forecast_ref = _latest_artifact_ref(refs, "forecast")
        if forecast_ref:
            latest_forecast = {
                "forecast_id": forecast_ref.split(":", 1)[-1],
                "resource_ref": forecast_ref,
            }
    todo_list = state.get("todo_list")
    if not todo_list:
        todo_list = _todo_list_from_progress(state.get("todo_progress"))
    visualizations = outputs.get("visualizations") or [
        {"visualization_id": ref.split(":", 1)[1]}
        for ref in _artifact_ref_values(context, "visualization")
        if ref.startswith("visualization:")
    ]
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
        "todo_list": todo_list,
        "plan_current_step": state.get("plan_current_step"),
        "planning_complete": state.get("planning_complete"),
        "requested_capabilities": state.get("requested_capabilities"),
        "task_contract": state.get("task_contract"),
        "next_action_constraints": state.get("next_action_constraints") or {},
        "focus": state.get("focus"),
        "latest_database_evidence": latest_evidence,
        "query_history": evidence.get("prior_queries") or [],
        "analysis_workspace": outputs.get("analysis_workspace") or {
            "analysis_count": analysis_count,
            "analyses": analyses,
        },
        "latest_forecast": latest_forecast,
        "latest_anomaly": latest_anomaly,
        "latest_rag": outputs.get("latest_rag"),
        "latest_skill": outputs.get("latest_skill"),
        "visualizations": visualizations,
        "latest_observation_summaries": observations,
        "available_actions": context.get("available_actions") or [],
    }


def _todo_list_from_progress(progress) -> list[dict]:
    if not isinstance(progress, dict) or int(progress.get("total") or 0) <= 0:
        return []
    result = []
    current = progress.get("current")
    if isinstance(current, dict):
        result.append(dict(current))
    pending = progress.get("pending_preview")
    if isinstance(pending, list):
        result.extend(dict(item) for item in pending if isinstance(item, dict))
    return result or [{}]


def _observation_summaries(context: dict) -> list[dict]:
    observations = []
    for item in context.get("recent_trajectory") or []:
        if not isinstance(item, dict):
            continue
        observation = item.get("observation")
        if isinstance(observation, dict):
            normalized = dict(observation)
            normalized.setdefault("tool_name", item.get("action"))
            if "tool_name" not in normalized and normalized.get("tool"):
                normalized["tool_name"] = normalized.get("tool")
            normalized.setdefault("success", item.get("status") != "failed")
            observations.append(normalized)
    latest = context.get("last_observation")
    if isinstance(latest, dict):
        normalized = dict(latest)
        if "tool_name" not in normalized and normalized.get("tool"):
            normalized["tool_name"] = normalized.get("tool")
        observations.append(normalized)
    return observations


def _latest_failed_observation(context: dict, tool_name: str) -> dict | None:
    for observation in reversed(context.get("latest_observation_summaries") or []):
        if not isinstance(observation, dict):
            continue
        if observation.get("success") is False and observation.get("tool_name") == tool_name:
            return observation
    return None


def _requires_code_repair(context: dict) -> bool:
    constraints = context.get("next_action_constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    for item in constraints.get("required_actions") or []:
        if not isinstance(item, dict) or item.get("action") != "code_interpreter":
            continue
        guidance = item.get("input_guidance") if isinstance(item.get("input_guidance"), dict) else {}
        repair_contract = guidance.get("repair_contract") if isinstance(guidance.get("repair_contract"), dict) else {}
        if guidance.get("requires_code") is True or repair_contract.get("mode") in {
            "generated_code_required",
            "code_execution_repair",
            "analysis_artifact_repair",
        }:
            return True
    return False


def _ref_payloads(refs: dict, prefix: str) -> list[dict]:
    result = []
    raw = refs.get(prefix)
    if isinstance(raw, list):
        result.extend(item for item in raw if isinstance(item, dict))
    elif isinstance(raw, dict):
        result.append(raw)
    marker = f"{prefix}:"
    for ref, payload in refs.items():
        if str(ref).startswith(marker) and isinstance(payload, dict):
            result.append(payload)
    return result


def _latest_ref_payload(refs: dict, prefix: str) -> dict | None:
    matches = _ref_payloads(refs, prefix)
    return matches[-1] if matches else None


def _analysis_refs(refs: dict) -> list[dict]:
    raw = refs.get("analysis")
    values = raw if isinstance(raw, list) else [raw] if raw else []
    result = []
    for value in values:
        text = str(value or "")
        if text.startswith("analysis:"):
            result.append({"analysis_id": text.split("analysis:", 1)[1]})
    latest = str(refs.get("latest_analysis") or "")
    if latest.startswith("analysis:"):
        analysis_id = latest.split("analysis:", 1)[1]
        if not any(item.get("analysis_id") == analysis_id for item in result):
            result.append({"analysis_id": analysis_id})
    return result


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
        "insight_requests": [{"insight_key": "series_summary", "name": "Series summary", "insight_type": "custom"}],
        "analysis_code": (
            "value_keys = ('value', '_value', 'price', 'appliances_energy_wh')\n"
            "values = []\n"
            "for row in rows:\n"
            "    for key in value_keys:\n"
            "        if row.get(key) is None:\n"
            "            continue\n"
            "        if isinstance(row.get(key), (int, float)):\n"
            "            values.append(float(row.get(key)))\n"
            "            break\n"
            "result = {'computed_insights': [{'insight_key': 'series_summary', 'value': {'row_count': len(rows), 'first_value': values[0] if values else None, 'last_value': values[-1] if values else None}, 'calculation_trace': {'operation': 'summarize canonical values', 'input_row_count': len(rows)}}], 'derived_evidence': []}\n"
        ),
    }


def _financial_metrics_code() -> str:
    return (
        "values = []\n"
        "value_columns = [column for column in columns if column != 'timestamp']\n"
        "for row in rows:\n"
        "    raw = None\n"
        "    for key in ['value', '_value', 'price'] + value_columns:\n"
        "        if row.get(key) is not None:\n"
        "            raw = row.get(key)\n"
        "            break\n"
        "    if isinstance(raw, (int, float)):\n"
        "        values.append(float(raw))\n"
        "returns = [(right / left) - 1 for left, right in zip(values, values[1:]) if left != 0]\n"
        "total_return = (values[-1] / values[0] - 1) if len(values) >= 2 and values[0] != 0 else None\n"
        "volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0\n"
        "peak = values[0] if values else 0.0\n"
        "max_drawdown = 0.0\n"
        "for value in values:\n"
        "    peak = max(peak, value)\n"
        "    if peak:\n"
        "        max_drawdown = min(max_drawdown, value / peak - 1)\n"
        "result = {\n"
        "    'computed_insights': [\n"
        "        {'insight_key': 'total_return', 'value': float(total_return) if total_return is not None else None, 'unavailable_reason': None if total_return is not None else 'Insufficient nonzero values.', 'calculation_trace': {'formula': 'last / first - 1'}},\n"
        "        {'insight_key': 'volatility', 'value': float(volatility), 'calculation_trace': {'formula': 'sample standard deviation of period returns', 'return_count': len(returns)}},\n"
        "        {'insight_key': 'max_drawdown', 'value': float(max_drawdown), 'calculation_trace': {'formula': 'min(value / running_peak - 1)'}},\n"
        "    ],\n"
        "    'derived_evidence': [],\n"
        "}\n"
    )


def _financial_insight_requests() -> list[dict]:
    return [
        {"insight_key": "total_return", "name": "Total return", "insight_type": "return"},
        {"insight_key": "volatility", "name": "Volatility", "insight_type": "volatility"},
        {"insight_key": "max_drawdown", "name": "Maximum drawdown", "insight_type": "drawdown"},
    ]
