from __future__ import annotations

import asyncio
import json

import pytest

from agents.data_agent import DataAgent


class _RepairCapturingLLM:
    def __init__(self):
        self.messages = []
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            return '{"thought":"需要查询","action_intention":"查询数据"}'
        return (
            '{"thought":"补齐 action","action":"sql_query",'
            '"action_input":{"message":"查询数据","database_context":{"database_id":"demo","database_type":"sqlite"}}}'
        )


class _TwoBadThenGoodLLM:
    def __init__(self):
        self.messages = []
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages.append(messages)
        if self.calls <= 2:
            return '{"thought":"需要查询"} {"action":"sql_query","action_input":{}}'
        return (
            '{"thought":"补齐 action","action":"sql_query",'
            '"action_input":{"message":"查询数据","database_context":{"database_id":"demo","database_type":"sqlite"}}}'
        )


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


def test_parse_turn_rejects_bare_terminal_input():
    bare_input = {
        "summary_goal": "Answer the seasonality question.",
        "direct_answer": "No clear daily or weekly seasonality was found.",
        "include_insight_ids": ["insight_1"],
        "include_visualization_ids": ["viz_1"],
        "section_plan": ["summary"],
    }

    with pytest.raises(ValueError, match="missing 'action'"):
        DataAgent(prompt_builder=None, llm=None)._parse_turn(json.dumps(bare_input))


def test_parse_turn_does_not_treat_top_level_answer_as_react_action_input():
    payload = {
        "thought": "已经可以回答。",
        "answer": "在2023年1月5日到2月3日之间，比特币兑美元价格第一次涨到24000美元的时间是2023-02-02T00:48:00+00:00，当时价格为24099.5781美元。",
    }

    with pytest.raises(ValueError, match="missing 'action'"):
        DataAgent(prompt_builder=None, llm=None)._parse_turn(json.dumps(payload, ensure_ascii=False))


def test_parse_turn_rejects_bare_todowrite_input():
    bare_input = {
        "message": "Plan database analysis.",
        "current_intent": "seasonality analysis",
        "requested_capabilities": ["query", "analysis"],
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


def test_repair_prompt_requires_explicit_action_and_action_input():
    llm = _RepairCapturingLLM()
    request_state = _RequestState()

    turn = asyncio.run(DataAgent(prompt_builder=_PromptBuilder(), llm=llm).next_turn(request_state, None))

    assert turn.action == "sql_query"
    repair_prompt = llm.messages[1][-1][1]
    assert "action field is mandatory" in repair_prompt
    assert "action_input field is mandatory" in repair_prompt
    assert "todowrite, sql_query, code_interpreter, forecast, anomaly, visualization, rag, skill, terminate" in repair_prompt
    assert "Do not return only thought, task_contract, action_intention, or action_reason" in repair_prompt
    repairs = request_state.completion_state["llm_diagnostics"]["react_turn_repairs"]
    assert repairs[0]["source"] == "data_agent.next_turn"
    assert repairs[0]["repair_index"] == 0
    assert "missing 'action'" in repairs[0]["parser_error"]
    assert repairs[0]["failed_output"]["char_count"] > 0
    assert repairs[0]["failed_output"]["starts_with"].startswith('{"thought"')


def test_next_turn_allows_second_llm_repair_attempt_without_guessing_action_locally():
    llm = _TwoBadThenGoodLLM()

    turn = asyncio.run(DataAgent(prompt_builder=_PromptBuilder(), llm=llm).next_turn(_RequestState(), None))

    assert turn.action == "sql_query"
    assert llm.calls == 3
    assert "final repair attempt" in llm.messages[2][-1][1]


def test_parse_turn_accepts_llm_task_contract():
    payload = {
        "thought": "先明确用户可见输出合同。",
        "task_contract": {
            "source": "llm",
            "goal": "返回总数和最早记录",
            "required_outputs": [
                {
                    "id": "total_count",
                    "description": "返回总记录数",
                    "output_type": "count",
                    "evidence_kind": "database",
                    "required": True,
                    "measures": ["count"],
                    "dimensions": [],
                    "time_scope": None,
                    "success_criteria": "结果包含总记录数",
                }
            ],
            "constraints": {},
            "assumptions": [],
            "evidence_quality_notes": [],
        },
        "action": "sql_query",
        "action_input": {"message": "查询总数", "database_context": {"database_id": "demo", "database_type": "sqlite"}},
    }

    turn = DataAgent(prompt_builder=None, llm=None)._parse_turn(json.dumps(payload, ensure_ascii=False))

    assert turn.task_contract is not None
    assert turn.task_contract.required_outputs[0].id == "total_count"


def test_next_turn_repairs_invalid_first_model_output_once():
    valid_turn = {
        "thought": "Use the current state to assemble the answer.",
        "action": "terminate",
        "action_input": {
            "summary_goal": "Answer",
            "direct_answer": "Done.",
            "include_insight_ids": [],
            "include_visualization_ids": [],
            "section_plan": [],
        },
    }
    repeated = f"{json.dumps(valid_turn)} {json.dumps(valid_turn)}"
    agent = DataAgent(prompt_builder=_PromptBuilder(), llm=_BadThenGoodLLM(repeated, json.dumps(valid_turn)))

    turn = asyncio.run(agent.next_turn(request_state=_RequestState(), conversation_state=None))

    assert turn.action == "terminate"
    assert turn.action_input["direct_answer"] == "Done."


def test_next_turn_uses_structured_output_when_available():
    llm = _StructuredLLM()

    turn = asyncio.run(DataAgent(prompt_builder=_PromptBuilder(), llm=llm).next_turn(_RequestState(), None))

    assert turn.action == "sql_query"
    assert turn.action_input["message"] == "查询数据"
    assert llm.calls == 1
    assert llm.structured_calls == 1


class _PromptBuilder:
    def build_system_prompt(self, response_language="en"):
        return "system"

    def build_user_prompt(self, request_state, conversation_state):
        return "Context JSON:\n{}"


class _RequestState:
    def __init__(self):
        self.completion_state = {}
        self.response_language = "en"


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


class _StructuredLLM:
    def __init__(self):
        self.calls = 0
        self.structured_calls = 0

    def with_structured_output(self, schema, **kwargs):
        self.structured_schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.structured_calls += 1
        return {
            "raw": _Response(""),
            "parsed": self.structured_schema(
                thought="需要查询。",
                action="sql_query",
                action_input={"message": "查询数据", "database_context": {"database_id": "demo", "database_type": "sqlite"}},
            ),
            "parsing_error": None,
        }
