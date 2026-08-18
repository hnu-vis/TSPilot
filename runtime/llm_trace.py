"""Request-local tracing for every LLM call made during a ReAct round."""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, Sequence

from runtime.token_usage import measure_llm_token_usage
from runtime.trace import TraceEventModel

LLM_INPUT_PREVIEW_MAX_CHARS = 20_000
LLM_OUTPUT_PREVIEW_MAX_CHARS = 12_000
_DATA_URL_PATTERN = re.compile(r"data:[^\s\"']*?;base64,[A-Za-z0-9+/=_-]+", re.IGNORECASE)


@dataclass(frozen=True)
class LLMTraceContext:
    """The exact public ReAct node that owns one or more model calls."""

    parent_id: str
    events: asyncio.Queue[TraceEventModel]


@dataclass
class LLMTraceSpan:
    """Mutable completion metadata for one concrete model invocation."""

    span_id: str
    parent_id: str
    title: str
    summary: str | None = None
    token_usage: dict | None = None
    input_summary: dict | None = None
    output_summary: dict | None = None
    input_preview: list[dict[str, str]] | None = None
    output_preview: str | None = None

    def attach_input(self, messages: Sequence[Any] | None) -> None:
        summary = _input_summary(messages)
        if summary is not None:
            self.input_summary = summary
            self.input_preview = _input_preview(messages)

    def attach_output(self, response=None, *, output_text: str | None = None) -> None:
        self.output_summary = _output_summary(response, output_text=output_text)
        self.output_preview = _output_preview(response, output_text=output_text)

    def attach_token_usage(self, usage: dict | None) -> None:
        self.token_usage = _public_token_usage(usage)

    def attach_response(
        self,
        response,
        *,
        messages: list | None = None,
        output_text: str | None = None,
        model: str | None = None,
    ) -> None:
        """Attach safe I/O shape metadata and request-accounting token counts."""

        self.attach_input(messages)
        self.attach_output(response, output_text=output_text)
        self.attach_token_usage(measure_llm_token_usage(
            response=response,
            messages=messages,
            output_text=output_text,
            model=model,
        ))


_LLM_TRACE_CONTEXT: ContextVar[LLMTraceContext | None] = ContextVar(
    "tspilot_llm_trace_context",
    default=None,
)


@contextmanager
def llm_trace_scope(
    *,
    parent_id: str,
    events: asyncio.Queue[TraceEventModel],
) -> Iterator[None]:
    """Attach model calls to one exact public ReAct node for this task context."""

    normalized_parent = str(parent_id or "").strip()
    if not normalized_parent:
        raise ValueError("LLM trace parent_id must be non-empty.")
    token = _LLM_TRACE_CONTEXT.set(LLMTraceContext(parent_id=normalized_parent, events=events))
    try:
        yield
    finally:
        _LLM_TRACE_CONTEXT.reset(token)


@contextmanager
def tool_trace_scope(
    *,
    parent_id: str,
    events: asyncio.Queue[TraceEventModel],
) -> Iterator[None]:
    """Backward-compatible alias for tool-owned LLM trace scopes."""

    with llm_trace_scope(parent_id=parent_id, events=events):
        yield


@asynccontextmanager
async def llm_trace_span(
    title: str,
    *,
    summary: str | None = None,
    messages: Sequence[Any] | None = None,
) -> AsyncIterator[LLMTraceSpan | None]:
    """Emit progressive start/end events for one actual LLM invocation.

    Outside an explicit request trace scope this is intentionally a no-op, so
    detached background work cannot be misrepresented as part of a live run.
    """

    context = _LLM_TRACE_CONTEXT.get()
    normalized_title = str(title or "").strip()
    if context is None:
        yield None
        return
    if not normalized_title:
        raise ValueError("LLM trace title must be non-empty inside an LLM trace scope.")

    span = LLMTraceSpan(
        span_id=f"{context.parent_id}:llm:{uuid.uuid4().hex}",
        parent_id=context.parent_id,
        title=normalized_title,
        summary=str(summary).strip() if summary else None,
        input_summary=_input_summary(messages),
        input_preview=_input_preview(messages),
    )
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    context.events.put_nowait(
        TraceEventModel(
            event_type="trace_span_start",
            payload={
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "kind": "llm",
                "title": span.title,
                "summary": span.summary,
                "input_summary": span.input_summary,
                "input_preview": span.input_preview,
                "status": "running",
                "started_at": started_at,
            },
        )
    )

    error: BaseException | None = None
    try:
        yield span
    except BaseException as exc:
        error = exc
        raise
    finally:
        elapsed = max(0.0, time.monotonic() - started_monotonic)
        completed_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "kind": "llm",
            "title": span.title,
            "summary": span.summary,
            "status": "error" if error is not None else "complete",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": int(round(elapsed * 1000)),
            "elapsed_seconds": round(elapsed, 3),
        }
        if span.token_usage:
            payload["token_usage"] = span.token_usage
        if span.input_summary:
            payload["input_summary"] = span.input_summary
        if span.output_summary:
            payload["output_summary"] = span.output_summary
        if span.input_preview:
            payload["input_preview"] = span.input_preview
        if span.output_preview is not None:
            payload["output_preview"] = span.output_preview
        if error is not None:
            payload["error"] = _error_message(error)
        context.events.put_nowait(TraceEventModel(event_type="trace_span_end", payload=payload))


def _public_token_usage(usage: dict | None) -> dict | None:
    """Expose counts only; provider/model details are not presentation data."""

    if not isinstance(usage, dict):
        return None
    counted = usage.get("provider") if isinstance(usage.get("provider"), dict) else None
    if counted is None and isinstance(usage.get("estimated"), dict):
        counted = usage["estimated"]
    if counted is None:
        counted = usage
    result = {
        key: counted[key]
        for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(counted.get(key), (int, float)) and not isinstance(counted.get(key), bool)
    }
    return result or None


def _input_summary(messages: Sequence[Any] | None) -> dict | None:
    """Describe an invocation input without retaining prompt content."""

    if messages is None or isinstance(messages, (str, bytes, bytearray)):
        return None
    try:
        items = list(messages)
    except TypeError:
        return None

    roles: list[str] = []
    character_count = 0
    multimodal_part_count = 0
    for message in items:
        role, content = _message_shape(message)
        if role and role not in roles:
            roles.append(role)
        text_count, media_count = _content_shape(content)
        character_count += text_count
        multimodal_part_count += media_count

    return {
        "message_count": len(items),
        "roles": roles,
        "character_count": character_count,
        "multimodal_part_count": multimodal_part_count,
    }


def _output_summary(response, *, output_text: str | None = None) -> dict:
    """Describe an invocation output without retaining generated content."""

    content = output_text if output_text is not None else getattr(response, "content", response)
    character_count, multimodal_part_count = _content_shape(content)
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            output_format = "empty"
        else:
            try:
                json.loads(stripped)
            except ValueError:
                output_format = "text"
            else:
                output_format = "json"
    elif content is None:
        output_format = "empty"
    else:
        output_format = "structured"

    return {
        "character_count": character_count,
        "format": output_format,
        "multimodal_part_count": multimodal_part_count,
    }


def _input_preview(messages: Sequence[Any] | None) -> list[dict[str, str]] | None:
    """Return bounded textual messages while replacing binary/multimodal payloads."""

    if messages is None or isinstance(messages, (str, bytes, bytearray)):
        return None
    try:
        items = list(messages)
    except TypeError:
        return None

    remaining = LLM_INPUT_PREVIEW_MAX_CHARS
    preview: list[dict[str, str]] = []
    for message in items:
        role, content = _message_shape(message)
        text = _preview_content(content)
        bounded = _bounded_preview(text, remaining)
        preview.append({"role": role or "message", "content": bounded})
        remaining -= len(bounded)
        if remaining <= 0:
            break
    return preview


def _output_preview(response, *, output_text: str | None = None) -> str:
    content = output_text if output_text is not None else getattr(response, "content", response)
    return _bounded_preview(_preview_content(content), LLM_OUTPUT_PREVIEW_MAX_CHARS)


def _preview_content(content: Any) -> str:
    if isinstance(content, str):
        return _safe_preview_string(content)
    if content is None:
        return ""
    if isinstance(content, (bytes, bytearray)):
        return f"[binary payload omitted: {len(content)} bytes]"
    if isinstance(content, (list, tuple)):
        parts = [_preview_content(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        part_type = str(content.get("type") or "").lower()
        if part_type in {"image", "image_url", "input_image", "audio", "input_audio", "video"}:
            return f"[{part_type or 'media'} omitted]"
        if "text" in content:
            return _preview_content(content.get("text"))
        if "content" in content:
            return _preview_content(content.get("content"))
        return json.dumps(_safe_preview_value(content), ensure_ascii=False, default=str, indent=2)
    if hasattr(content, "model_dump"):
        try:
            return json.dumps(
                _safe_preview_value(content.model_dump(mode="json")),
                ensure_ascii=False,
                default=str,
                indent=2,
            )
        except Exception:
            pass
    return _safe_preview_string(str(content))


def _safe_preview_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return "[nested value omitted]"
    if isinstance(value, str):
        return _safe_preview_string(value)
    if isinstance(value, (bytes, bytearray)):
        return f"[binary payload omitted: {len(value)} bytes]"
    if isinstance(value, dict):
        safe = {}
        for key, nested in value.items():
            normalized_key = str(key)
            if normalized_key.lower() in {"image", "image_url", "input_image", "audio", "input_audio", "video"}:
                safe[normalized_key] = "[media omitted]"
            else:
                safe[normalized_key] = _safe_preview_value(nested, depth=depth + 1)
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_preview_value(item, depth=depth + 1) for item in value]
    return value


def _safe_preview_string(value: str) -> str:
    return _DATA_URL_PATTERN.sub("[data URL omitted]", value)


def _bounded_preview(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}\n… [{omitted} chars omitted]"


def _message_shape(message: Any) -> tuple[str | None, Any]:
    if isinstance(message, (tuple, list)) and len(message) >= 2:
        return _normalized_role(message[0]), message[1]
    if isinstance(message, dict):
        return _normalized_role(message.get("role") or message.get("type")), message.get("content")
    role = getattr(message, "role", None) or getattr(message, "type", None)
    if role is None:
        role = message.__class__.__name__.removesuffix("Message")
    return _normalized_role(role), getattr(message, "content", None)


def _normalized_role(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    aliases = {"human": "user", "ai": "assistant"}
    return aliases.get(normalized, normalized) or None


def _content_shape(content: Any) -> tuple[int, int]:
    if isinstance(content, str):
        return len(content), 0
    if content is None or isinstance(content, (bool, int, float)):
        return 0, 0
    if isinstance(content, (list, tuple)):
        shapes = [_content_shape(item) for item in content]
        return sum(shape[0] for shape in shapes), sum(shape[1] for shape in shapes)
    if isinstance(content, dict):
        part_type = str(content.get("type") or "").lower()
        if part_type in {"image", "image_url", "input_image", "audio", "input_audio", "video"}:
            return 0, 1
        if "text" in content:
            return _content_shape(content.get("text"))
        if "content" in content:
            return _content_shape(content.get("content"))
        return 0, 0
    return 0, 0


def _error_message(error: BaseException) -> str:
    message = str(error).strip()
    return message[:1000] if message else error.__class__.__name__
