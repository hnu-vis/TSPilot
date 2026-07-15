"""Base tool contract."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Shared tool interface."""

    @abstractmethod
    async def execute(self, validated_input, **kwargs) -> dict:
        raise NotImplementedError

    def summarize(self, payload: dict) -> str:
        return payload.get("summary", self.__class__.__name__)

