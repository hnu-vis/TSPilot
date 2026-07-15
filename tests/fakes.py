from __future__ import annotations

import json


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
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
            return _turn("I need evidence before producing facts.", "query_database", action_input)

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

        if context.get("latest_insight") is None:
            action_input = {
                "database_evidence": latest_evidence,
                "requested_fact_types": context.get("requested_fact_types", []),
                "focus": context.get("focus"),
                "constraints": context.get("constraints", {}),
            }
            return _turn(
                "I have evidence and should convert it into verified facts.",
                "insight",
                action_input,
            )

        action_input = {
            "summary_goal": context["message"],
            "include_fact_ids": [
                fact["fact_id"]
                for fact in context["latest_insight"].get("verified_facts", [])
            ],
            "include_visualization_ids": [
                visualization["visualization_id"]
                for visualization in context.get("visualizations", [])
            ],
            "section_plan": ["summary", "facts", "visualization"],
        }
        return _turn(
            "I have enough verified output to assemble the final answer.",
            "format_answer",
            action_input,
        )


class ComplexReActLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
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
            return _turn("I have a plan and need time-series evidence next.", "query_database", action_input)

        latest_evidence = context.get("latest_database_evidence") or {}
        if context.get("latest_insight") is None:
            action_input = {
                "database_evidence": latest_evidence,
                "requested_fact_types": ["trend", "change_percent", "extrema"],
                "focus": "整体趋势、变化幅度和关键峰值",
                "constraints": context.get("constraints", {}),
            }
            return _turn(
                "I need verified facts before any downstream analytics summary.",
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
            "include_fact_ids": [
                fact["fact_id"]
                for fact in context.get("latest_insight", {}).get("verified_facts", [])
            ],
            "include_visualization_ids": [
                visualization["visualization_id"]
                for visualization in context.get("visualizations", [])
            ],
            "section_plan": ["summary", "facts", "anomaly", "forecast", "visualization"],
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
                "query_database",
                {
                    "message": context["message"],
                    "database_context": context["database_context"],
                    "time_range": context.get("time_range"),
                    "constraints": context.get("constraints", {}),
                },
            )

        if context.get("latest_insight") is None:
            return _turn(
                "I should convert evidence into facts.",
                "insight",
                {
                    "database_evidence": context["latest_database_evidence"],
                    "requested_fact_types": ["trend", "extrema"],
                    "focus": "趋势和极值",
                    "constraints": context.get("constraints", {}),
                },
            )

        if latest_observation and latest_observation["tool_name"] == "todowrite" and latest_observation["success"] is False:
            return _turn(
                "Planning is already available, so I should do anomaly detection next.",
                "anomaly",
                {"constraints": {"zscore_threshold": 2.5}},
            )

        if context.get("latest_anomaly") is None:
            return _turn(
                "I want to rewrite the plan again.",
                "todowrite",
                {"message": "重复规划。", "todos": [{"content": "重复规划", "task_type": "plan", "status": "in_progress", "priority": 1}]},
            )

        return _turn(
            "I now have enough outputs.",
            "format_answer",
            {
                "summary_goal": context["message"],
                "include_fact_ids": [
                    fact["fact_id"]
                    for fact in context.get("latest_insight", {}).get("verified_facts", [])
                ],
                "include_visualization_ids": [
                    viz["visualization_id"] for viz in context.get("visualizations", [])
                ],
                "section_plan": ["summary", "facts", "anomaly"],
            },
        )


class TodoScopeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, config=None, stop=None, **kwargs):
        self.calls += 1
        user_prompt = messages[-1][1]
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
                "The runtime rejected forecast because the current step is insight, so I should follow the plan.",
                "insight",
                {
                    "database_evidence": context.get("latest_database_evidence"),
                    "requested_fact_types": ["trend"],
                    "focus": "趋势",
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
