"""Todo writer tool."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from core.completion import normalize_todo_for_completion
from core.harness import default_capability_registry
from runtime.llm_trace import llm_trace_span
from runtime.timeout_policy import load_timeout_policy
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
    todos: list[dict | str] = Field(default_factory=list)
    evidence_summary: dict | str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("todo_list") and not normalized.get("todos"):
            normalized["todos"] = normalized["todo_list"]
        if isinstance(normalized.get("todos"), list):
            normalized["todos"] = [
                cls._normalize_raw_todo_item(index, todo)
                for index, todo in enumerate(normalized["todos"], start=1)
            ]
        if not normalized.get("message"):
            normalized["message"] = (
                normalized.get("focus")
                or normalized.get("current_intent")
                or normalized.get("rationale")
                or "Create a todo plan."
            )
        return normalized

    @staticmethod
    def _normalize_raw_todo_item(index: int, todo: Any) -> dict | str:
        if isinstance(todo, str):
            return {
                "content": todo,
                "status": "pending",
                "priority": index,
            }
        if isinstance(todo, dict):
            normalized = dict(todo)
            normalized.setdefault("priority", index)
            return normalized
        return todo


class TodoWriteResult(BaseModel):
    summary: str
    task_contract: dict | None = None
    todos: list[dict] = Field(default_factory=list)
    in_progress: dict | None = None
    current_step: int = 0
    planning_complete: bool = False
    completed_count: int = 0
    pending_count: int = 0


class _TodoCapabilityBinding(BaseModel):
    index: int = Field(ge=0)
    task_type: Literal[
        "query", "anomaly", "forecast", "code_interpreter",
        "visualization", "answer", "rag", "skill",
    ]
    reason: str = Field(min_length=1)


class _TodoCapabilityBindings(BaseModel):
    bindings: list[_TodoCapabilityBinding] = Field(min_length=1)


class TodoWriteTool(BaseTool):
    def __init__(self, llm=None, *, llm_timeout_seconds: float | None = None):
        self._llm = llm
        self._llm_timeout_seconds = float(
            llm_timeout_seconds
            if llm_timeout_seconds is not None
            else load_timeout_policy().tool("todowrite").stage_seconds("llm_call_seconds")
        )

    async def execute(self, validated_input: TodoWriteInput, **kwargs) -> dict:
        total_todos = len(validated_input.todos)
        todos = [
            normalized
            for index, todo in enumerate(validated_input.todos, start=1)
            if (normalized := self._normalize_todo(index, todo, validated_input, total_todos=total_todos)) is not None
        ]
        source_text = validated_input.message or validated_input.focus or ""
        if not todos:
            todos = self._todos_from_text(source_text, validated_input)
        elif self._looks_like_placeholder_plan(todos):
            todos = self._todos_from_text(source_text, validated_input)
        elif self._looks_like_collapsed_multistep_plan(todos):
            source_text = todos[0].content
            if validated_input.message and validated_input.message != source_text:
                source_text = f"{validated_input.message}\n{source_text}"
            expanded = self._todos_from_text(source_text, validated_input)
            if expanded:
                todos = expanded
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
        if self._llm is not None:
            todos = await self._bind_task_types(todos, validated_input)
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

    async def _bind_task_types(
        self,
        todos: list[TodoItem],
        validated_input: TodoWriteInput,
    ) -> list[TodoItem]:
        """Let the LLM bind each user-visible step to its owning capability."""

        payload = {
            "user_request": validated_input.message,
            "task_contract": validated_input.task_contract,
            "todos": [
                {"index": index, "content": todo.content}
                for index, todo in enumerate(todos)
            ],
        }
        system = (
            "Map every Todo to the single tool capability that owns its acceptance result. "
            "Interpret each Todo by meaning and dependencies; never align Todos to task-contract outputs by list position. "
            "Use query for database retrieval, anomaly for detection results, forecast for prediction generation, "
            "code_interpreter for calculations over existing artifacts, visualization for charts, answer for final synthesis, "
            "rag for external retrieval, and skill for packaged workflows. A prerequisite mentioned in a Todo does not change "
            "the owner of its requested result. Return exactly one binding for every supplied zero-based index."
        )
        messages = [("system", system), ("human", json.dumps(payload, ensure_ascii=False, default=str))]
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                async with llm_trace_span(
                    "Task Classification Repair" if attempt else "Task Classification",
                    summary="Bind each task to its owning capability",
                    messages=messages,
                ) as trace_span:
                    if hasattr(self._llm, "with_structured_output"):
                        runnable = self._llm.with_structured_output(
                            _TodoCapabilityBindings, method="json_schema", include_raw=True,
                        )
                        bundle = await asyncio.wait_for(
                            runnable.ainvoke(messages), timeout=self._llm_timeout_seconds
                        )
                        if isinstance(bundle, dict):
                            trace_response = bundle.get("raw")
                            if trace_span is not None:
                                trace_span.attach_response(
                                    trace_response,
                                    messages=messages,
                                    output_text=str(getattr(trace_response, "content", trace_response) or ""),
                                )
                            parsed = bundle.get("parsed")
                            if parsed is None:
                                raise ValueError(
                                    bundle.get("parsing_error")
                                    or "Todo capability binding was not parsed"
                                )
                        else:
                            parsed = bundle
                            if trace_span is not None:
                                trace_span.attach_response(
                                    bundle,
                                    messages=messages,
                                    output_text=str(getattr(bundle, "content", bundle) or ""),
                                )
                        result = (
                            parsed
                            if isinstance(parsed, _TodoCapabilityBindings)
                            else _TodoCapabilityBindings.model_validate(parsed)
                        )
                    else:
                        response = await asyncio.wait_for(
                            self._llm.ainvoke(messages), timeout=self._llm_timeout_seconds
                        )
                        if trace_span is not None:
                            trace_span.attach_response(
                                response,
                                messages=messages,
                                output_text=str(getattr(response, "content", response) or ""),
                            )
                        result = _TodoCapabilityBindings.model_validate_json(
                            str(getattr(response, "content", response))
                        )
                by_index = {item.index: item for item in result.bindings}
                expected = set(range(len(todos)))
                if set(by_index) != expected:
                    raise ValueError(f"Todo capability bindings must cover indexes {sorted(expected)} exactly")
                return [
                    todo.model_copy(update={"task_type": by_index[index].task_type})
                    for index, todo in enumerate(todos)
                ]
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    messages.append(("human", f"Correct the capability binding contract error: {exc}"))
        raise ValueError(f"Todo capability binding failed: {last_error}") from last_error

    def _looks_like_placeholder_plan(self, todos: list[TodoItem]) -> bool:
        if not todos:
            return False
        placeholder_count = sum(1 for todo in todos if re.fullmatch(r"step_\d+", str(todo.content or "")))
        return placeholder_count == len(todos)

    def _looks_like_collapsed_multistep_plan(self, todos: list[TodoItem]) -> bool:
        if len(todos) != 1:
            return False
        return len(self._extract_numbered_items(todos[0].content)) >= 2

    def _todos_from_text(self, text: str, validated_input: TodoWriteInput | None = None) -> list[TodoItem]:
        items = []
        for index, content in enumerate(self._extract_numbered_items(text), start=1):
            if not content:
                continue
            items.append(
                TodoItem(
                    content=content,
                    task_type=self._todo_task_type(index, content, len(self._extract_numbered_items(text)), validated_input),
                    status="pending",
                    priority=index,
                )
            )
        if items and all(item.task_type != "answer" for item in items):
            items[-1] = items[-1].model_copy(update={"task_type": "answer"})
        return self._enforce_single_in_progress(items)

    def _extract_numbered_items(self, text: str) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        marker = r"(?:\d+|[一二三四五六七八九十]+)\s*(?:[\.、\)]|）)"
        pattern = re.compile(
            rf"(?:^|[\n\r；;。:：])\s*{marker}\s*(.*?)"
            rf"(?=(?:[\n\r；;。]\s*{marker})|\Z)",
            flags=re.DOTALL,
        )
        items = []
        for match in pattern.finditer(normalized_text):
            content = re.sub(r"\s+", " ", match.group(1)).strip(" \t\r\n；;。")
            if content:
                items.append(content)
        return items

    def _normalize_todo(
        self,
        index: int,
        raw_todo: dict | str,
        validated_input: TodoWriteInput | None = None,
        *,
        total_todos: int = 0,
    ) -> TodoItem | None:
        if isinstance(raw_todo, str):
            raw_todo = {"content": raw_todo}
        if not isinstance(raw_todo, dict):
            return None
        content = str(
            raw_todo.get("content")
            or raw_todo.get("description")
            or raw_todo.get("task")
            or raw_todo.get("title")
            or raw_todo.get("id")
            or f"step_{index}"
        ).strip()
        explicit_task_type = str(
            raw_todo.get("task_type")
            or raw_todo.get("todo_type")
            or raw_todo.get("kind")
            or ""
        ).strip().lower()
        task_type = self._normalize_task_type(
            explicit_task_type or self._todo_task_type(index, content, total_todos, validated_input)
        )
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

    def _normalize_task_type(self, task_type: str) -> str:
        normalized = str(task_type or "").strip().lower()
        if normalized in {"list", "todo_list", "todos", "planning"}:
            return "plan"
        if normalized in {"data", "dataset", "timeseries", "time_series", "series", "records"}:
            return "query"
        return normalized or "generic"

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

    def _contract_task_type(self, index: int, validated_input: TodoWriteInput | None) -> str:
        if validated_input is None:
            return "generic"
        task_contract = validated_input.task_contract if isinstance(validated_input.task_contract, dict) else {}
        outputs = task_contract.get("required_outputs") if isinstance(task_contract.get("required_outputs"), list) else []
        output = outputs[index - 1] if 0 <= index - 1 < len(outputs) and isinstance(outputs[index - 1], dict) else {}
        raw_kind = output.get("evidence_kind") or output.get("output_type")
        kind = default_capability_registry().normalize_id(str(raw_kind or ""))
        if kind:
            task_type = default_capability_registry().task_type_for_capability(kind)
            if task_type in {"sql_query", "terminate"}:
                return "query" if task_type == "sql_query" else "answer"
            return task_type
        return "generic"

    def _todo_task_type(
        self,
        index: int,
        content: str,
        total_todos: int,
        validated_input: TodoWriteInput | None,
    ) -> str:
        inferred = self._infer_task_type_from_content(content)
        if inferred != "generic":
            return inferred
        contract_type = self._contract_task_type(index, validated_input)
        if contract_type and contract_type != "generic":
            return contract_type
        if total_todos > 1 and index == total_todos:
            return "answer"
        return "generic"

    def _infer_task_type_from_content(self, content: str) -> str:
        text = str(content or "").strip().lower()
        if not text:
            return "generic"
        if re.search(r"异常|异常检测|离群|突增|突降|anomal|outlier|spike", text, flags=re.IGNORECASE):
            return "anomaly"
        if re.search(r"可视化|综合图|图表|绘图|visuali[sz]|chart|plot|graph", text, flags=re.IGNORECASE):
            return "visualization"
        if re.search(r"预测|预估|未来|forecast|predict|projection", text, flags=re.IGNORECASE):
            return "forecast"
        if re.search(
            r"计算|统计|平均|均值|最大|最小|最新|标准差|波动|回撤|指标|变化率|差值|分析|"
            r"calculate|compute|statistic|average|mean|max|min|std|volatility|drawdown|analysis",
            text,
            flags=re.IGNORECASE,
        ):
            return "code_interpreter"
        if re.search(r"结论|解释|回答|总结|汇总|中文|conclusion|answer|explain|summari[sz]e", text, flags=re.IGNORECASE):
            return "answer"
        if re.search(r"查询|获取|读取|加载|数据|序列|记录|query|fetch|load|retrieve|data|series|record", text, flags=re.IGNORECASE):
            return "query"
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
