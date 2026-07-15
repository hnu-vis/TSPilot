"""LLM-driven outer data agent."""
from __future__ import annotations

import json
import re

from agents.base import BaseAgent
from prompts.data_agent import DataAgentPromptBuilder
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
        response = await self._llm.ainvoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return self._parse_turn(str(content))

    def _parse_turn(self, content: str) -> ReActTurn:
        stripped = content.strip()
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = None

        if isinstance(decoded, dict):
            action = str(decoded.get("action", "")).strip()
            action_input = decoded.get("action_input")
            if not action:
                raise ValueError(f"Structured model output was missing 'action': {content}")
            if not isinstance(action_input, dict):
                raise ValueError(f"Structured model output must include object 'action_input': {content}")
            thought = str(decoded.get("thought", "")).strip()
            return ReActTurn(thought=thought, action=action, action_input=action_input)

        patterns = [
            re.compile(
                r"Thought:\s*(?P<thought>.*?)\s*Action:\s*(?P<action>[^\n]+)\s*Action Input:\s*(?P<input>\{.*\})\s*$",
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
        return ReActTurn(thought=thought, action=action, action_input=action_input)
