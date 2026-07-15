"""Chat API route."""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.deps import get_react_loop
from app.settings import get_settings
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
    settings = get_settings()
    request_state = build_request_state(normalized, settings)
    conversation_state = build_conversation_state(normalized, request_state.conversation_id or "")
    react_loop = get_react_loop()

    if not normalized.stream:
        response = await react_loop.run(request_state, conversation_state)
        return JSONResponse(content=response.model_dump(mode="json"))

    async def event_stream() -> AsyncIterator[str]:
        async for event in react_loop.run_sse(request_state, conversation_state):
            yield _sse_frame(event.event_type, event.payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
