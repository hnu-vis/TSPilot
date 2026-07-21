from __future__ import annotations

import asyncio
import json

import pytest

from agents.data_agent import DataAgent


def test_parse_turn_rejects_repeated_json_turn_output():
    first_turn = {
        "thought": "Plan first.",
        "action": "todowrite",
        "action_input": {
            "message": "Analyze seasonality from database data.",
            "todos": [
                {"content": "Query data", "task_type": "query", "status": "in_progress", "priority": 1}
            ],
        },
    }
    repeated = f"{json.dumps(first_turn)} {json.dumps(first_turn)}"

    with pytest.raises(ValueError, match="exactly one complete JSON object"):
        DataAgent(prompt_builder=None, llm=None)._parse_turn(repeated)


def test_parse_turn_rejects_bare_format_answer_input():
    bare_input = {
        "summary_goal": "Answer the seasonality question.",
        "direct_answer": "No clear daily or weekly seasonality was found.",
        "include_fact_ids": ["fact_1"],
        "include_visualization_ids": ["viz_1"],
        "section_plan": ["summary"],
    }

    with pytest.raises(ValueError, match="missing 'action'"):
        DataAgent(prompt_builder=None, llm=None)._parse_turn(json.dumps(bare_input))


def test_parse_turn_rejects_bare_todowrite_input():
    bare_input = {
        "message": "Plan database analysis.",
        "current_intent": "seasonality analysis",
        "requested_fact_types": ["seasonality"],
        "focus": "Bitcoin USD",
        "todos": [
            {"content": "Query data", "task_type": "query", "status": "in_progress", "priority": 1}
        ],
    }

    with pytest.raises(ValueError, match="missing 'action'"):
        DataAgent(prompt_builder=None, llm=None)._parse_turn(json.dumps(bare_input))


def test_parse_turn_accepts_dbgpt_style_step_fields():
    payload = {
        "thought": "需要先看原始样本。",
        "action_intention": "拉取原始数据",
        "action_reason": "确认过滤条件",
        "action": "sql_query",
        "action_input": {
            "database_context": {"database_id": "demo", "database_type": "sqlite"},
            "query": "SELECT * FROM metrics LIMIT 5",
            "query_language": "sql",
        },
    }

    turn = DataAgent(prompt_builder=None, llm=None)._parse_turn(json.dumps(payload, ensure_ascii=False))

    assert turn.action == "sql_query"
    assert turn.action_intention == "拉取原始数据"
    assert turn.action_reason == "确认过滤条件"


def test_next_turn_repairs_invalid_first_model_output_once():
    valid_turn = {
        "thought": "Use the current state to assemble the answer.",
        "action": "terminate",
        "action_input": {
            "summary_goal": "Answer",
            "direct_answer": "Done.",
            "include_fact_ids": [],
            "include_visualization_ids": [],
            "section_plan": [],
        },
    }
    repeated = f"{json.dumps(valid_turn)} {json.dumps(valid_turn)}"
    agent = DataAgent(prompt_builder=_PromptBuilder(), llm=_BadThenGoodLLM(repeated, json.dumps(valid_turn)))

    turn = asyncio.run(agent.next_turn(request_state=None, conversation_state=None))

    assert turn.action == "terminate"
    assert turn.action_input["direct_answer"] == "Done."


class _PromptBuilder:
    def build_system_prompt(self):
        return "system"

    def build_user_prompt(self, request_state, conversation_state):
        return "Context JSON:\n{}"


class _BadThenGoodLLM:
    def __init__(self, first: str, second: str):
        self._responses = [first, second]
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        response = self._responses[self.calls]
        self.calls += 1
        return _Response(response)


class _Response:
    def __init__(self, content: str):
        self.content = content
