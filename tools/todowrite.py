"""Todo writer tool."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from core.completion import normalize_todo_for_completion
from tools.base import BaseTool


class TodoItem(BaseModel):
    content: str
    task_type: str = "generic"
    status: str = "pending"
    priority: int = 2
    notes: str | None = None
    acceptance_criteria: str | None = None
    result_ref: str | None = None
    completion_reason: str | None = None


class TodoWriteInput(BaseModel):
    message: str | None = None
    current_intent: str | None = None
    requested_capabilities: list[str] = Field(default_factory=list)
    focus: str | None = None
    task_contract: dict | None = None
    todos: list[dict] = Field(default_factory=list)
    evidence_summary: dict | str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("todo_list") and not normalized.get("todos"):
            normalized["todos"] = normalized["todo_list"]
        if not normalized.get("message"):
            normalized["message"] = (
                normalized.get("focus")
                or normalized.get("current_intent")
                or normalized.get("rationale")
                or "Create a todo plan."
            )
        return normalized


class TodoWriteResult(BaseModel):
    summary: str
    task_contract: dict | None = None
    todos: list[dict] = Field(default_factory=list)
    in_progress: dict | None = None
    current_step: int = 0
    planning_complete: bool = False
    completed_count: int = 0
    pending_count: int = 0


class TodoWriteTool(BaseTool):
    async def execute(self, validated_input: TodoWriteInput, **kwargs) -> dict:
        todos = [
            normalized
            for index, todo in enumerate(validated_input.todos, start=1)
            if (normalized := self._normalize_todo(index, todo)) is not None
        ]
        if self._looks_like_placeholder_plan(todos):
            todos = self._todos_from_message(validated_input.message or validated_input.focus or "")
        if not todos:
            todos = [
                TodoItem(
                    content=validated_input.focus or validated_input.message,
                    task_type="query",
                    status="in_progress",
                    priority=1,
                    notes=validated_input.evidence_summary,
                )
            ]
        todos = self._enforce_single_in_progress(todos)
        in_progress = next((todo.model_dump(mode="json") for todo in todos if todo.status == "in_progress"), None)
        completed_count = sum(1 for todo in todos if todo.status == "completed")
        pending_count = sum(1 for todo in todos if todo.status != "completed")
        current_step = next((index for index, todo in enumerate(todos, start=1) if todo.status == "in_progress"), 0)
        summary = self._build_summary(todos, completed_count, pending_count)
        return TodoWriteResult(
            summary=summary,
            task_contract=validated_input.task_contract,
            todos=[todo.model_dump(mode="json") for todo in todos],
            in_progress=in_progress,
            current_step=current_step,
            planning_complete=all(todo.status == "completed" for todo in todos),
            completed_count=completed_count,
            pending_count=pending_count,
        ).model_dump(mode="json")

    def _looks_like_placeholder_plan(self, todos: list[TodoItem]) -> bool:
        if not todos:
            return False
        placeholder_count = sum(1 for todo in todos if re.fullmatch(r"step_\d+", str(todo.content or "")))
        return placeholder_count == len(todos)

    def _todos_from_message(self, message: str) -> list[TodoItem]:
        items = []
        pattern = re.compile(
            r"(?:^|[；;。\\n])\\s*(?:\\d+|[一二三四五六七八九十]+)(?:[\\.、\\)]|\\s+)\\s*([^；;。\\n]+)",
            flags=re.MULTILINE,
        )
        for index, match in enumerate(pattern.finditer(message or ""), start=1):
            content = match.group(1).strip()
            if not content:
                continue
            items.append(
                TodoItem(
                    content=content,
                    task_type=self._infer_task_type(content),
                    status="pending",
                    priority=index,
                )
            )
        if items and all(item.task_type != "answer" for item in items):
            items[-1] = items[-1].model_copy(update={"task_type": "answer"})
        return self._enforce_single_in_progress(items)

    def _normalize_todo(self, index: int, raw_todo: dict) -> TodoItem | None:
        if isinstance(raw_todo, str):
            raw_todo = {"content": raw_todo}
        if not isinstance(raw_todo, dict):
            return None
        content = str(
            raw_todo.get("content")
            or raw_todo.get("task")
            or raw_todo.get("title")
            or raw_todo.get("id")
            or f"step_{index}"
        ).strip()
        task_type = str(
            raw_todo.get("task_type")
            or raw_todo.get("todo_type")
            or raw_todo.get("kind")
            or self._infer_task_type(content)
        ).strip().lower()
        if task_type == "plan" or self._is_internal_query_preparation(content):
            return None
        status = str(raw_todo.get("status") or "pending").strip().lower()
        if status not in {"pending", "in_progress", "completed"}:
            status = "pending"
        priority = raw_todo.get("priority", 2)
        try:
            priority = int(priority)
        except (TypeError, ValueError):
            priority = 2
        notes = raw_todo.get("notes")
        normalized = TodoItem(
            content=content,
            task_type=task_type,
            status=status,
            priority=priority,
            notes=notes,
            acceptance_criteria=raw_todo.get("acceptance_criteria") or raw_todo.get("criteria"),
            result_ref=raw_todo.get("result_ref"),
            completion_reason=raw_todo.get("completion_reason"),
        )
        return TodoItem.model_validate(normalize_todo_for_completion(normalized.model_dump(mode="json")))

    def _enforce_single_in_progress(self, todos: list[TodoItem]) -> list[TodoItem]:
        seen_in_progress = False
        normalized = []
        for todo in sorted(todos, key=lambda item: (item.priority, item.content)):
            current = todo.model_copy(deep=True)
            if current.status == "in_progress":
                if seen_in_progress:
                    current.status = "pending"
                seen_in_progress = True
            normalized.append(current)
        if not seen_in_progress:
            for current in normalized:
                if current.status == "pending":
                    current.status = "in_progress"
                    break
        return normalized

    def _build_summary(self, todos: list[TodoItem], completed_count: int, pending_count: int) -> str:
        in_progress = next((todo.content for todo in todos if todo.status == "in_progress"), None)
        if in_progress:
            return (
                f"Todo plan updated with {len(todos)} steps. "
                f"Current step: {in_progress}. Completed {completed_count}, remaining {pending_count}."
            )
        return f"Todo plan updated with {len(todos)} steps. Completed {completed_count}, remaining {pending_count}."

    def _infer_task_type(self, content: str) -> str:
        normalized = content.lower()
        if self._is_internal_query_preparation(content):
            return "query"
        if any(token in normalized for token in ["查询", "查库", "取数", "query", "retrieve evidence"]):
            return "query"
        if any(token in normalized for token in ["洞察", "事实", "趋势", "周期", "seasonality", "trend", "code interpreter"]):
            return "code_interpreter"
        if any(token in normalized for token in ["异常", "anomaly", "outlier"]):
            return "anomaly"
        if any(token in normalized for token in ["预测", "forecast", "predict"]):
            return "forecast"
        if any(token in normalized for token in ["总结", "回答", "汇总", "answer", "format"]):
            return "answer"
        if any(token in normalized for token in ["规划", "计划", "todo", "plan"]):
            return "plan"
        return "generic"

    def _is_internal_query_preparation(self, content: str) -> bool:
        normalized = content.lower()
        internal_tokens = (
            "schema linking",
            "schema-linked",
            "grounding",
            "确认数据源",
            "确认字段",
            "字段确认",
            "生成查询",
            "生成flux",
            "生成 flux",
            "查询计划",
            "可执行flux",
            "可执行 flux",
        )
        return any(token in normalized for token in internal_tokens)
