"""One parsed ReAct turn."""
from __future__ import annotations

from pydantic import BaseModel


class ReActTurn(BaseModel):
    thought: str
    action: str
    action_input: dict


class ReActTurnParseError(BaseModel):
    error_code: str
    message: str
    raw_turn: str

