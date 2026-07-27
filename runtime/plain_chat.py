"""Plain LLM chat path for requests without a selected database."""
from __future__ import annotations

import json

from runtime.token_usage import record_llm_token_usage, token_usage_summary
from schemas.api import ChatResponse
from schemas.output import FinalAnswer
from schemas.state import ConversationStateModel, RequestStateModel


class PlainChatService:
    """Answer conversational requests without invoking the data ReAct loop."""

    def __init__(self, llm):
        self._llm = llm

    async def run(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> ChatResponse:
        answer = await self._generate_answer(request_state, conversation_state)
        return ChatResponse(
            conversation_id=request_state.conversation_id or "",
            request_id=request_state.request_id,
            status="completed",
            response_kind="final_answer",
            used_tools=[],
            answer=answer,
            trace=[],
            token_usage=token_usage_summary(request_state),
        )

    async def _generate_answer(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> FinalAnswer:
        messages = self._messages(request_state, conversation_state)
        content = await self._invoke(messages, request_state=request_state, source="plain_chat")
        try:
            return self._parse_answer(content)
        except ValueError as first_error:
            repair_messages = [
                *messages,
                ("assistant", content),
                (
                    "user",
                    "Your previous response did not match the required JSON final-answer schema. "
                    "Return exactly one JSON object with this shape and no extra text: "
                    "{\"title\": string|null, \"summary\": string, \"sections\": [], \"references\": [], \"visualizations\": []}. "
                    f"Parser error: {first_error}",
                ),
            ]
            repaired = await self._invoke(
                repair_messages,
                request_state=request_state,
                source="plain_chat.repair",
            )
            return self._parse_answer(repaired)

    async def _invoke(self, messages, *, request_state: RequestStateModel, source: str) -> str:
        response = await self._llm.ainvoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        content = str(content)
        record_llm_token_usage(
            request_state,
            source=source,
            response=response,
            messages=messages,
            output_text=content,
        )
        return content

    def _messages(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> list[tuple[str, str]]:
        context = {
            "message": request_state.message,
            "history": [
                message.model_dump(mode="json") if hasattr(message, "model_dump") else message
                for message in conversation_state.recent_messages[-10:]
            ],
            "time_range": request_state.time_range,
            "constraints": request_state.constraints,
            "database_context": None,
        }
        return [
            (
                "system",
                "You are TSPilot in plain chatbot mode. "
                "No database is selected, so do not claim that you queried data, used tools, or inspected local datasets. "
                "Answer conversationally and directly. If the user asks for data analysis that requires a database, "
                "explain that they should select a database before asking that analysis question. "
                "Respond in the user's language. "
                "Return exactly one JSON object and nothing else. "
                "Schema: {\"title\": string|null, \"summary\": string, \"sections\": [], \"references\": [], \"visualizations\": []}.",
            ),
            (
                "user",
                "Context JSON:\n"
                + json.dumps(context, ensure_ascii=False),
            ),
        ]

    def _parse_answer(self, content: str) -> FinalAnswer:
        try:
            decoded = json.loads(content.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Plain chat response was not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Plain chat response must be a JSON object.")

        candidate = decoded.get("answer") if isinstance(decoded.get("answer"), dict) else decoded
        if isinstance(decoded.get("action_input"), dict):
            action_input = decoded["action_input"]
            if isinstance(action_input.get("direct_answer"), str):
                candidate = {
                    "title": None,
                    "summary": action_input["direct_answer"],
                    "sections": [],
                    "references": [],
                    "visualizations": [],
                }

        try:
            return FinalAnswer.model_validate(candidate)
        except Exception as exc:
            raise ValueError(f"Plain chat response did not match FinalAnswer schema: {exc}") from exc
