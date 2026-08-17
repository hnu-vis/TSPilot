"""Chat API route."""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.deps import get_plain_chat_service_for_model, get_react_loop_for_model
from app.settings import get_settings
from app.model_config import get_model_config_store
from runtime.request_state import (
    build_conversation_state,
    build_request_state,
    normalize_chat_request,
)
from schemas.api import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def chat(request: ChatRequest):
    """Handle one chat request as JSON or SSE."""

    normalized = normalize_chat_request(request)
    if normalized.model_id and get_model_config_store().connection("llm", normalized.model_id) is None:
        raise HTTPException(status_code=422, detail=f"Unknown language model connection '{normalized.model_id}'.")
    settings = get_settings()
    request_state = build_request_state(normalized, settings)
    conversation_state = build_conversation_state(normalized, request_state.conversation_id or "")

    if request_state.database_context is None:
        plain_chat = get_plain_chat_service_for_model(normalized.model_id)
        response = await plain_chat.run(request_state, conversation_state)
        if not normalized.stream:
            return JSONResponse(content=response.model_dump(mode="json"))

        async def plain_event_stream() -> AsyncIterator[str]:
            if response.answer is not None:
                yield _sse_frame(
                    "final_answer",
                    {
                        "answer": response.answer.model_dump(mode="json"),
                        "token_usage": response.token_usage,
                    },
                )

        return StreamingResponse(plain_event_stream(), media_type="text/event-stream")

    react_loop = get_react_loop_for_model(normalized.model_id)

    if not normalized.stream:
        response = await react_loop.run(request_state, conversation_state)
        return JSONResponse(content=response.model_dump(mode="json"))

    async def event_stream() -> AsyncIterator[str]:
        async for event in react_loop.run_sse(request_state, conversation_state):
            yield _sse_frame(event.event_type, event.payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
