"""LLM-driven outer data agent."""
from __future__ import annotations

import json
import re
import time

from agents.base import BaseAgent
from prompts.data_agent import DataAgentPromptBuilder
from runtime.llm_trace import llm_trace_span
from runtime.token_usage import record_llm_token_usage
from schemas.agent_turn import ReActTurn
from schemas.state import ConversationStateModel, RequestStateModel


class DataAgent(BaseAgent):
    """Choose one outer action per turn via an LLM."""

    def __init__(self, prompt_builder: DataAgentPromptBuilder, llm):
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._structured_llm = self._build_structured_llm(llm)

    async def next_turn(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> ReActTurn:
        system_prompt = self._prompt_builder.build_system_prompt(request_state.response_language)
        user_prompt = self._prompt_builder.build_user_prompt(request_state, conversation_state)
        content, structured_turn = await self._invoke_turn(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ],
            request_state=request_state,
            source="data_agent.next_turn",
            trace_title="ReAct Decision",
            trace_summary="Choose the next action from the current task state",
        )
        if structured_turn is not None:
            return structured_turn
        try:
            return self._parse_turn(content)
        except ValueError as first_error:
            self._record_repair_attempt(
                request_state,
                source="data_agent.next_turn",
                repair_index=0,
                parser_error=first_error,
                failed_content=content,
                final_attempt=False,
            )
            repaired_content = content
            last_error = first_error
            for repair_index in range(2):
                repaired_content, structured_turn = await self._invoke_turn(
                    [
                        ("system", system_prompt),
                        ("user", user_prompt),
                        ("assistant", repaired_content),
                        ("user", self._repair_prompt(last_error, final_attempt=repair_index == 1)),
                    ],
                    request_state=request_state,
                    source="data_agent.contract_repair",
                    trace_title="ReAct Decision Repair",
                    trace_summary="Repair an invalid ReAct decision contract",
                )
                if structured_turn is not None:
                    return structured_turn
                try:
                    return self._parse_turn(repaired_content)
                except ValueError as repair_error:
                    self._record_repair_attempt(
                        request_state,
                        source="data_agent.contract_repair",
                        repair_index=repair_index + 1,
                        parser_error=repair_error,
                        failed_content=repaired_content,
                        final_attempt=repair_index == 1,
                    )
                    last_error = repair_error
                    continue
            try:
                return self._parse_turn(repaired_content)
            except ValueError as second_error:
                raise ValueError(
                    f"Model failed to produce exactly one valid ReAct JSON object after repair. "
                    f"first_error={first_error}; second_error={second_error}"
                ) from second_error

    def _repair_prompt(self, parser_error: ValueError, *, final_attempt: bool) -> str:
        skeleton = (
            '{"thought":"","previous_observation_assessment":null,"task_contract":null,'
            '"action":"sql_query","action_input":{}}'
        )
        extra = (
            "This is the final repair attempt. Start from this skeleton and replace action/action_input with the correct next tool call: "
            + skeleton
            if final_attempt
            else ""
        )
        return (
            "Your previous response violated the ReAct output contract. "
            "Return exactly one JSON object and nothing else. "
            "The JSON object must use this schema: "
            "{\"thought\": str, \"previous_observation_assessment\": object|null, "
            "\"task_contract\": object|null, "
            "\"action\": str, \"action_input\": object}. "
            "The top-level action field is mandatory and must be exactly one of: "
            "todowrite, sql_query, code_interpreter, forecast, anomaly, visualization, rag, skill, terminate. "
            "The top-level action_input field is mandatory and must be an object matching that action. "
            "Do not return only thought or task_contract. "
            "If you already know the next step from thought, put the corresponding tool name in action and its input in action_input. "
            f"{extra} "
            "Do not mention this repair instruction, parser errors, JSON format, or contract violations in thought/action_input. "
            "Continue solving the original user task using the current context. "
            f"Parser error: {parser_error}"
        )

    async def _invoke_model(
        self,
        messages,
        *,
        request_state=None,
        source: str = "data_agent",
        trace_title: str,
        trace_summary: str | None = None,
    ) -> str:
        async with llm_trace_span(trace_title, summary=trace_summary, messages=messages) as trace_span:
            started_at = time.perf_counter()
            response = await self._llm.ainvoke(messages)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            content = str(content)
            usage = record_llm_token_usage(
                request_state,
                source=source,
                response=response,
                messages=messages,
                output_text=content,
                duration_ms=duration_ms,
            )
            if trace_span is not None:
                trace_span.attach_output(response, output_text=content)
                trace_span.attach_token_usage(usage)
            return content

    async def _invoke_turn(
        self,
        messages,
        *,
        request_state=None,
        source: str = "data_agent",
        trace_title: str,
        trace_summary: str | None = None,
    ) -> tuple[str, ReActTurn | None]:
        if self._structured_llm is None:
            return await self._invoke_model(
                messages,
                request_state=request_state,
                source=source,
                trace_title=trace_title,
                trace_summary=trace_summary,
            ), None

        async with llm_trace_span(trace_title, summary=trace_summary, messages=messages) as trace_span:
            started_at = time.perf_counter()
            result = await self._structured_llm.ainvoke(messages)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            raw_response = result.get("raw") if isinstance(result, dict) else result
            parsed = result.get("parsed") if isinstance(result, dict) else None
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
            if isinstance(parsed, ReActTurn):
                content = json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False)
            else:
                content = self._structured_failure_text(result, parsing_error)
            usage = record_llm_token_usage(
                request_state,
                source=source,
                response=raw_response,
                messages=messages,
                output_text=content,
                duration_ms=duration_ms,
            )
            if trace_span is not None:
                trace_span.attach_output(raw_response, output_text=content)
                trace_span.attach_token_usage(usage)
            return content, parsed if isinstance(parsed, ReActTurn) else None

    def _build_structured_llm(self, llm):
        builder = getattr(llm, "with_structured_output", None)
        if not callable(builder):
            return None
        try:
            return builder(ReActTurn, method="function_calling", include_raw=True)
        except Exception:
            return None

    def _structured_failure_text(self, result, parsing_error) -> str:
        if parsing_error is not None:
            return str(parsing_error)
        if isinstance(result, dict):
            raw = result.get("raw")
            content = getattr(raw, "content", None)
            if content:
                return str(content)
            tool_calls = getattr(raw, "tool_calls", None)
            if tool_calls:
                return json.dumps(tool_calls, ensure_ascii=False)
        return str(result)

    def _record_repair_attempt(
        self,
        request_state,
        *,
        source: str,
        repair_index: int,
        parser_error: ValueError,
        failed_content: str,
        final_attempt: bool,
    ) -> None:
        if request_state is None:
            return
        diagnostics = request_state.completion_state.setdefault("llm_diagnostics", {})
        repairs = diagnostics.setdefault("react_turn_repairs", [])
        repairs.append(
            {
                "source": source,
                "repair_index": repair_index,
                "final_attempt": final_attempt,
                "parser_error_type": parser_error.__class__.__name__,
                "parser_error": _truncate_middle(str(parser_error), max_chars=2000),
                "failed_output": _failed_output_summary(failed_content),
            }
        )

    def _parse_turn(self, content: str) -> ReActTurn:
        stripped = content.strip()
        decoded = self._decode_json_turn(stripped)
        if isinstance(decoded, dict):
            return self._turn_from_dict(decoded, content)

        patterns = [
            re.compile(
                r"Thought:\s*(?P<thought>.*?)\s*(?:Action Intention:\s*(?P<intention>.*?)\s*)?(?:Action Reason:\s*(?P<reason>.*?)\s*)?Action:\s*(?P<action>[^\n]+)\s*Action Input:\s*(?P<input>\{.*\})\s*$",
                re.DOTALL,
            ),
            re.compile(
                r"Action:\s*(?P<action>[^\n]+)\s*Action Input:\s*(?P<input>\{.*\})\s*$",
                re.DOTALL,
            ),
        ]
        match = None
        thought = ""
        for pattern in patterns:
            match = pattern.search(content.strip())
            if match:
                thought = match.groupdict().get("thought", "").strip()
                break
        if not match:
            raise ValueError(f"Model output did not match the ReAct contract: {content}")
        action = match.group("action").strip()
        raw_input = match.group("input").strip()
        try:
            action_input = json.loads(raw_input)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Action Input was not valid JSON: {exc}. raw_action_input={raw_input} full_turn={content}"
            ) from exc
        if not isinstance(action_input, dict):
            raise ValueError("Action Input must decode to a JSON object.")
        return ReActTurn(
            thought=thought,
            previous_observation_assessment=None,
            action=action,
            action_input=action_input,
        )

    def _decode_json_turn(self, stripped: str) -> dict | None:
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if stripped.startswith("{"):
                raise ValueError(
                    "Model output must be exactly one complete JSON object; "
                    "multiple JSON objects, trailing text, or truncated JSON are not allowed."
                ) from exc
            decoded = None
        if isinstance(decoded, dict):
            return decoded
        return None

    def _turn_from_dict(self, decoded: dict, original_content: str) -> ReActTurn:
        action = str(decoded.get("action", "")).strip()
        action_input = decoded.get("action_input")
        if not action:
            raise ValueError(f"Structured model output was missing 'action': {original_content}")
        if not isinstance(action_input, dict):
            raise ValueError(f"Structured model output must include object 'action_input': {original_content}")
        thought = str(decoded.get("thought", "")).strip()
        return ReActTurn(
            thought=thought,
            task_contract=decoded.get("task_contract") or action_input.get("task_contract"),
            previous_observation_assessment=decoded.get("previous_observation_assessment"),
            action=action,
            action_input=action_input,
        )

    def _optional_string(self, value) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _regex_group(self, match, name: str) -> str | None:
        try:
            value = match.groupdict().get(name)
        except IndexError:
            value = None
        return self._optional_string(value)


def _failed_output_summary(content: str) -> dict:
    text = str(content or "")
    return {
        "char_count": len(text),
        "starts_with": text[:500],
        "ends_with": text[-500:] if len(text) > 500 else "",
    }


def _truncate_middle(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    edge = max((max_chars - 20) // 2, 1)
    return f"{text[:edge]} ... <truncated> ... {text[-edge:]}"
