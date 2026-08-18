from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agents.data_agent import DataAgent
from runtime.llm_trace import llm_trace_scope, llm_trace_span, tool_trace_scope
from runtime.trace import TraceEventModel
from tools.code_interpreter import CodeInterpreterTool
from tools.visualization import _invoke_structured


class _VisualizationProbe(BaseModel):
    value: str


class _QueueLLM:
    def __init__(self, *responses: str):
        self.responses = list(responses)

    async def ainvoke(self, _messages):
        return SimpleNamespace(content=self.responses.pop(0))


class _PromptBuilder:
    def build_system_prompt(self, _response_language):
        return "system"

    def build_user_prompt(self, _request_state, _conversation_state):
        return "context"


@pytest.mark.asyncio
async def test_llm_trace_span_emits_progressive_parented_events_without_model_metadata(monkeypatch):
    events: asyncio.Queue[TraceEventModel] = asyncio.Queue()
    monkeypatch.setattr(
        "runtime.token_usage._encoding_for_model",
        lambda _model: SimpleNamespace(encode=lambda text: str(text).split()),
    )

    messages = [
        ("system", "system rules"),
        ("user", [
            {"type": "text", "text": "generate private query"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,private"}},
        ]),
    ]
    with tool_trace_scope(parent_id="iteration-3", events=events):
        async with llm_trace_span(
            "Schema Linking",
            summary="Identify the query fields",
            messages=messages,
        ) as span:
            assert span is not None
            span.attach_response(
                SimpleNamespace(content='{"query":"ready"}'),
                messages=messages,
                model="must-not-leak",
            )

    start = events.get_nowait()
    end = events.get_nowait()

    assert start.event_type == "trace_span_start"
    assert end.event_type == "trace_span_end"
    assert start.payload["span_id"] == end.payload["span_id"]
    assert start.payload["parent_id"] == end.payload["parent_id"] == "iteration-3"
    assert start.payload["status"] == "running"
    assert start.payload["input_summary"] == {
        "message_count": 2,
        "roles": ["system", "user"],
        "character_count": len("system rules") + len("generate private query"),
        "multimodal_part_count": 1,
    }
    assert start.payload["input_preview"] == [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "generate private query\n[image_url omitted]"},
    ]
    assert end.payload["status"] == "complete"
    assert end.payload["output_summary"] == {
        "character_count": len('{"query":"ready"}'),
        "format": "json",
        "multimodal_part_count": 0,
    }
    assert end.payload["token_usage"]["total_tokens"] > 0
    assert end.payload["output_preview"] == '{"query":"ready"}'
    assert "model" not in str(start.payload)
    assert "model" not in str(end.payload)
    assert "generate private query" in str(start.payload)
    assert "base64,private" not in str(start.payload)


@pytest.mark.asyncio
async def test_llm_trace_span_records_failed_invocation_as_terminal_leaf():
    events: asyncio.Queue[TraceEventModel] = asyncio.Queue()

    with pytest.raises(RuntimeError, match="provider failed"):
        with tool_trace_scope(parent_id="iteration-4", events=events):
            async with llm_trace_span("Chart Planning", messages=[("user", "private chart prompt")]):
                raise RuntimeError("provider failed")

    start = events.get_nowait()
    end = events.get_nowait()
    assert start.event_type == "trace_span_start"
    assert end.event_type == "trace_span_end"
    assert end.payload["status"] == "error"
    assert end.payload["error"] == "provider failed"
    assert start.payload["input_summary"]["message_count"] == 1
    assert end.payload["input_preview"][0]["content"] == "private chart prompt"


@pytest.mark.asyncio
async def test_llm_trace_span_is_not_emitted_outside_a_request_scope():
    async with llm_trace_span("Outer Agent Decision") as span:
        assert span is None


@pytest.mark.asyncio
async def test_every_react_decision_invocation_including_repair_has_its_own_span():
    llm = _QueueLLM(
        '{"thought":"missing action"}',
        '{"thought":"repaired","action":"sql_query","action_input":{"message":"query"}}',
    )
    request_state = SimpleNamespace(response_language="en", completion_state={})
    events: asyncio.Queue[TraceEventModel] = asyncio.Queue()

    with llm_trace_scope(parent_id="iteration-2", events=events):
        turn = await DataAgent(_PromptBuilder(), llm).next_turn(request_state, None)

    assert turn.action == "sql_query"
    trace_events = [events.get_nowait() for _ in range(events.qsize())]
    starts = [event for event in trace_events if event.event_type == "trace_span_start"]
    ends = [event for event in trace_events if event.event_type == "trace_span_end"]
    assert [event.payload["title"] for event in starts] == [
        "ReAct Decision",
        "ReAct Decision Repair",
    ]
    assert [event.payload["span_id"] for event in starts] == [
        event.payload["span_id"] for event in ends
    ]
    assert all(event.payload["parent_id"] == "iteration-2" for event in trace_events)


@pytest.mark.asyncio
async def test_code_and_visualization_invocations_attach_token_counts_to_leaf_events(monkeypatch):
    monkeypatch.setattr(
        "runtime.token_usage._encoding_for_model",
        lambda _model: SimpleNamespace(encode=lambda text: str(text).split()),
    )
    llm = _QueueLLM(
        '{"code":"result = {\\"computed_insights\\": [], \\"derived_evidence\\": []}"}',
        '{"value":"chart"}',
    )
    events: asyncio.Queue[TraceEventModel] = asyncio.Queue()

    with tool_trace_scope(parent_id="iteration-8", events=events):
        code = await CodeInterpreterTool(llm=llm)._generate_code(
            goal="analyze",
            requests=[],
            context={},
            response_language="en",
        )
        response, _, parsed, error = await _invoke_structured(
            llm,
            _VisualizationProbe,
            [("user", "plan a chart")],
            timeout_seconds=5,
            trace_title="Chart Planning",
        )

    assert code.startswith("result =")
    assert response is not None
    assert parsed == _VisualizationProbe(value="chart")
    assert error is None
    trace_events = [events.get_nowait() for _ in range(events.qsize())]
    ends = [event for event in trace_events if event.event_type == "trace_span_end"]
    assert [event.payload["title"] for event in ends] == ["Analysis Planning", "Chart Planning"]
    assert all(event.payload["token_usage"]["total_tokens"] > 0 for event in ends)
