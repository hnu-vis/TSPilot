"""DB-GPT-style tool action output boundary."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ActionOutput(BaseModel):
    """Separated tool output views for ReAct, UI, and internal resources.

    `observations` is the only tool result view intended for the outer model.
    `view` is the public/UI view. `resource_value` is a compact artifact
    receipt; canonical full data lives in request state or durable artifact
    storage and is addressed by `resource_ref`.
    """

    tool_name: str
    success: bool
    content: str
    observations: dict | str
    view: dict | str | None = None
    resource_type: str | None = None
    resource_value: dict | None = None
    resource_ref: str | None = None
    memory_fragment: dict | str | None = None
    error: str | None = None
    have_retry: bool = True
    terminate: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)
