"""Persistent conversation trace logging."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from app.settings import Settings
from runtime.trace import TraceEventModel
from schemas.api import ChatResponse
from schemas.state import RequestStateModel


class ConversationTraceLogger:
    """Write one complete request trace and a compact index entry."""

    schema_version = "conversation_trace_v1"

    def __init__(self, settings: Settings):
        self._settings = settings

    def persist(
        self,
        *,
        request_state: RequestStateModel,
        response: ChatResponse,
        internal_trace: list[TraceEventModel],
        mode: str,
        public_trace: list[TraceEventModel] | None = None,
        interrupted: bool = False,
    ) -> Path | None:
        if not self._settings.conversation_log_enabled:
            return None

        try:
            root = self._settings.resolved_conversation_log_dir
            request_id = _safe_name(request_state.request_id)
            if request_state.request_log_dir:
                request_dir = Path(request_state.request_log_dir).resolve()
            else:
                conversation_id = _safe_name(request_state.conversation_id or "unknown_conversation")
                request_dir = root / conversation_id / request_id
            request_dir.mkdir(parents=True, exist_ok=True)
            path = request_dir / "conversation_trace.json"
            envelope = self._build_envelope(
                request_state=request_state,
                response=response,
                internal_trace=internal_trace,
                public_trace=public_trace or [],
                mode=mode,
                interrupted=interrupted,
                path=path,
            )
            self._write_request_files(request_dir, envelope)
            self._append_index(root / "index.jsonl", envelope, path)
            return path
        except Exception as exc:
            print(f"Failed to persist conversation trace log: {exc}", file=sys.stderr)
            return None

    def _build_envelope(
        self,
        *,
        request_state: RequestStateModel,
        response: ChatResponse,
        internal_trace: list[TraceEventModel],
        public_trace: list[TraceEventModel],
        mode: str,
        interrupted: bool,
        path: Path,
    ) -> dict:
        captured_at = datetime.now(timezone.utc).isoformat()
        state_payload = request_state.model_dump(mode="json")
        response_payload = response.model_dump(mode="json")
        return {
            "schema_version": self.schema_version,
            "captured_at": captured_at,
            "mode": mode,
            "interrupted": interrupted,
            "conversation_id": request_state.conversation_id,
            "request_id": request_state.request_id,
            "status": response.status,
            "response_kind": response.response_kind,
            "request": {
                "message": request_state.message,
                "database_context": (
                    request_state.database_context.model_dump(mode="json")
                    if request_state.database_context
                    else None
                ),
                "selected_database": request_state.selected_database,
                "selected_database_type": request_state.selected_database_type,
                "time_range": request_state.time_range,
                "constraints": request_state.constraints,
                "history": [message.model_dump(mode="json") for message in request_state.history],
            },
            "response": response_payload,
            "summary": {
                "used_tools": response.used_tools,
                "iteration": request_state.iteration,
                "max_iterations": request_state.max_iterations,
                "error": response.error,
                "errors": request_state.errors,
                "llm_diagnostics": (request_state.completion_state or {}).get("llm_diagnostics"),
                "todo_count": len(request_state.todo_list),
                "observation_count": len(request_state.observations),
                "internal_trace_event_count": len(internal_trace),
                "public_trace_event_count": len(public_trace),
                "final_answer_summary": response.answer.summary if response.answer else None,
            },
            "trace": {
                "internal": [event.model_dump(mode="json") for event in internal_trace],
                "public": [event.model_dump(mode="json") for event in public_trace],
            },
            "state": state_payload,
            "log_path": str(path),
            "request_log_dir": str(path.parent),
        }

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _write_jsonl_atomic(self, path: Path, items: list[dict]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp_path.replace(path)

    def _write_request_files(self, request_dir: Path, envelope: dict) -> None:
        request_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(request_dir / "conversation_trace.json", envelope)
        self._write_json_atomic(request_dir / "request.json", envelope["request"])
        self._write_json_atomic(request_dir / "response.json", envelope["response"])
        self._write_json_atomic(request_dir / "summary.json", envelope["summary"])
        self._write_json_atomic(request_dir / "state.json", envelope["state"])
        self._write_jsonl_atomic(request_dir / "trace_internal.jsonl", envelope["trace"]["internal"])
        self._write_jsonl_atomic(request_dir / "trace_public.jsonl", envelope["trace"]["public"])
        self._write_jsonl_atomic(request_dir / "tool_calls.jsonl", envelope["state"].get("tool_history", []))
        self._write_jsonl_atomic(request_dir / "observations.jsonl", envelope["state"].get("observations", []))

    def _append_index(self, index_path: Path, envelope: dict, path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        summary = envelope["summary"]
        entry = {
            "captured_at": envelope["captured_at"],
            "conversation_id": envelope["conversation_id"],
            "request_id": envelope["request_id"],
            "mode": envelope["mode"],
            "interrupted": envelope["interrupted"],
            "status": envelope["status"],
            "response_kind": envelope["response_kind"],
            "used_tools": summary["used_tools"],
            "iteration": summary["iteration"],
            "error": summary["error"],
            "message_preview": _preview(envelope["request"]["message"]),
            "log_path": str(path),
            "request_log_dir": envelope.get("request_log_dir"),
        }
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return normalized or "unknown"


def _preview(value: str, limit: int = 160) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
