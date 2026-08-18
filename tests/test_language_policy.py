from __future__ import annotations

import asyncio
import json

from app.settings import get_settings
from core.database.llm_query import LLMQueryGenerator
from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import build_conversation_state, build_request_state
from runtime.plain_chat import PlainChatService
from runtime.prompt_locale import prompt_locale_instruction
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from tools.format_answer import FormatAnswerInput, FormatAnswerTool


class _QueryLLM:
    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return json.dumps(
            {
                "query": "SELECT 1",
                "query_language": "sql",
                "purpose": "测试",
                "expected_result_type": "table",
                "selected_fields": [],
                "assumptions": [],
                "task_coverage": {
                    "satisfied": ["已覆盖"],
                    "missing": [],
                    "next_action_hint": None,
                },
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )


def test_request_state_detects_response_language():
    settings = get_settings()

    zh_state = build_request_state(ChatRequest(message="分析比特币价格走势"), settings)
    en_state = build_request_state(ChatRequest(message="Analyze Bitcoin price trends"), settings)

    assert zh_state.response_language == "zh"
    assert en_state.response_language == "en"


def test_data_agent_prompt_exposes_language_policy():
    settings = get_settings()
    request = ChatRequest(message="分析比特币价格走势")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")

    builder = DataAgentPromptBuilder()
    system_prompt = builder.build_system_prompt(request_state.response_language)
    context = builder.build_context(request_state, conversation_state)

    assert "task.response_language controls all natural-language fields" in system_prompt
    assert "当前查询语言：简体中文" in system_prompt
    assert context["task"]["response_language"] == "zh"


def test_prompt_locale_keeps_machine_timestamps_but_humanizes_user_facing_time():
    zh_instruction = prompt_locale_instruction("zh")
    en_instruction = prompt_locale_instruction("en")

    assert "2023年1月4日 23:48（UTC）" in zh_instruction
    assert "即使这些自然语言位于 JSON 字符串字段内" in zh_instruction
    assert "结构化数据、JSON、引用、定位器与代码中的时间值保持原样" in zh_instruction
    assert "Jan 4, 2023 at 11:48 PM UTC" in en_instruction
    assert "prose is carried in a JSON string field" in en_instruction
    assert "keep timestamp values unchanged in structured data" in en_instruction


def test_llm_query_prompt_carries_response_language():
    llm = _QueryLLM()
    request_state = build_request_state(ChatRequest(message="分析比特币价格走势"), get_settings())

    asyncio.run(
        LLMQueryGenerator(llm).generate(
            database_id="demo",
            database_type="timescaledb",
            message=request_state.message,
            schema_preview={},
            time_range=None,
            constraints={},
            history=[],
            request_state=request_state,
        )
    )

    user_prompt = llm.messages[1][1]
    payload = json.loads(user_prompt.split("LLM 查询生成输入 JSON：\n", 1)[1])
    assert payload["request"]["response_language"] == "zh"
    assert "所有面向用户或描述性的自然语言内容都必须使用简体中文" in llm.messages[0][1]


def test_english_query_uses_english_prompt_locale():
    llm = _QueryLLM()
    request_state = build_request_state(ChatRequest(message="Analyze Bitcoin price trends"), get_settings())

    asyncio.run(
        LLMQueryGenerator(llm).generate(
            database_id="demo",
            database_type="timescaledb",
            message=request_state.message,
            schema_preview={},
            time_range=None,
            constraints={},
            history=[],
            request_state=request_state,
        )
    )

    assert "Current query language: English" in llm.messages[0][1]
    assert llm.messages[1][1].startswith("LLM SQL Query Generation JSON:\n")


def test_plain_chat_prompt_uses_query_locale():
    settings = get_settings()
    request = ChatRequest(message="解释什么是时间序列")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")

    messages = PlainChatService(llm=None)._messages(request_state, conversation_state)

    assert "当前查询语言：简体中文" in messages[0][1]
    assert messages[1][1].startswith("对话上下文 JSON：\n")


def test_format_answer_uses_chinese_templates_for_chinese_request():
    request_state = build_request_state(
        ChatRequest(
            message="请返回总数、最早记录，并展示每项查询语句和实际返回行数。",
            database_context={"database_id": "demo", "database_type": "timescaledb"},
        ),
        get_settings(),
    )
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_rows",
        result_type="table",
        database="demo",
        query_language="sql",
        query="SELECT time, value FROM metrics LIMIT 1",
        summary="Loaded 1 row.",
        data={"rows": [{"time": "2026-01-01T00:00:00Z", "value": 1.0}]},
        columns=["time", "value"],
        metadata={"sql_query_mode": "llm", "purpose": "返回最早记录"},
        diagnostics={},
    )

    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(response_plan={
                "title": "结果",
                "summary": "已返回最早记录。",
                "sections": [
                    {"section_type": "query", "heading": "查询", "content": "已执行查询。", "source_refs": []},
                    {"section_type": "table", "heading": "表格结果", "content": "已返回一行。", "source_refs": []},
                ],
                "visualization_ids": [],
            }),
            request_state=request_state,
        )
    )

    assert result["sections"][0]["heading"] == "查询"
    assert result["sections"][1]["heading"] == "表格结果"


def test_format_answer_uses_english_templates_for_english_request():
    request_state = build_request_state(
        ChatRequest(
            message="Return the earliest row and show the query.",
            database_context={"database_id": "demo", "database_type": "timescaledb"},
        ),
        get_settings(),
    )
    request_state.latest_database_evidence = DatabaseEvidence(
        evidence_id="evi_rows",
        result_type="table",
        database="demo",
        query_language="sql",
        query="SELECT time, value FROM metrics LIMIT 1",
        summary="Loaded 1 row.",
        data={"rows": [{"time": "2026-01-01T00:00:00Z", "value": 1.0}]},
        columns=["time", "value"],
        metadata={"sql_query_mode": "llm", "purpose": "Return the earliest row"},
        diagnostics={},
    )

    result = asyncio.run(
        FormatAnswerTool().execute(
            FormatAnswerInput(response_plan={
                "title": "Result",
                "summary": "Returned the earliest row.",
                "sections": [
                    {"section_type": "query", "heading": "Query", "content": "Executed the query.", "source_refs": []},
                    {"section_type": "table", "heading": "Table Result", "content": "Returned one row.", "source_refs": []},
                ],
                "visualization_ids": [],
            }),
            request_state=request_state,
        )
    )

    assert result["sections"][0]["heading"] == "Query"
    assert result["sections"][1]["heading"] == "Table Result"
