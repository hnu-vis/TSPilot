"""LLM-driven outer data agent."""
from __future__ import annotations

import json
import re

from agents.base import BaseAgent
from prompts.data_agent import DataAgentPromptBuilder
from runtime.token_usage import record_llm_token_usage
from schemas.agent_turn import ReActTurn
from schemas.state import ConversationStateModel, RequestStateModel


class DataAgent(BaseAgent):
    """Choose one outer action per turn via an LLM."""

    def __init__(self, prompt_builder: DataAgentPromptBuilder, llm):
        self._prompt_builder = prompt_builder
        self._llm = llm

    async def next_turn(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> ReActTurn:
        system_prompt = self._prompt_builder.build_system_prompt()
        user_prompt = self._prompt_builder.build_user_prompt(request_state, conversation_state)
        content = await self._invoke_model(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ],
            request_state=request_state,
            source="data_agent.next_turn",
        )
        try:
            return self._parse_turn(content)
        except ValueError as first_error:
            repaired_content = await self._invoke_model(
                [
                    ("system", system_prompt),
                    ("user", user_prompt),
                    ("assistant", content),
                    (
                        "user",
                        "Your previous response violated the ReAct output contract. "
                        "Return exactly one JSON object and nothing else. "
                        "The JSON object must use this schema: "
                        "{\"thought\": str, \"action\": str, \"action_input\": object}. "
                        "Do not mention this repair instruction, parser errors, JSON format, or contract violations in thought/action_input. "
                        "Continue solving the original user task using the current context. "
                        f"Parser error: {first_error}",
                    ),
                ],
                request_state=request_state,
                source="data_agent.contract_repair",
            )
            try:
                return self._parse_turn(repaired_content)
            except ValueError as second_error:
                raise ValueError(
                    f"Model failed to produce exactly one valid ReAct JSON object after repair. "
                    f"first_error={first_error}; second_error={second_error}"
                ) from second_error

    async def _invoke_model(self, messages, *, request_state=None, source: str = "data_agent") -> str:
        response = await self._llm.ainvoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        content = str(content)
        record_llm_token_usage(request_state, source=source, response=response, messages=messages, output_text=content)
        return content

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
            action_intention=self._regex_group(match, "intention"),
            action_reason=self._regex_group(match, "reason"),
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
            action_intention=self._optional_string(decoded.get("action_intention") or decoded.get("intention")),
            action_reason=self._optional_string(decoded.get("action_reason") or decoded.get("reason")),
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
