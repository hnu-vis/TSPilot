"""Base agent interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.agent_turn import ReActTurn
from schemas.state import ConversationStateModel, RequestStateModel


class BaseAgent(ABC):
    """Stable outer-agent interface."""

    @abstractmethod
    async def next_turn(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> ReActTurn:
        raise NotImplementedError

