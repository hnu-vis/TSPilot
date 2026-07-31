"""Base tool contract."""
from __future__ import annotations

from abc import ABC, abstractmethod


class StructuredToolError(Exception):
    """Tool failure with machine-readable observation payload."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        retryable: bool = True,
        diagnostics: dict | None = None,
        recommended_next_action: str | None = None,
        validation_failure: dict | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.diagnostics = diagnostics or {}
        self.recommended_next_action = recommended_next_action
        self.validation_failure = validation_failure or {}

    def to_observation_payload(self) -> dict:
        payload = {
            "error": str(self),
            "error_type": self.error_type,
            "retryable": self.retryable,
            "diagnostics": self.diagnostics,
        }
        if self.recommended_next_action:
            payload["recommended_next_action"] = self.recommended_next_action
        if self.validation_failure:
            payload["validation_failure"] = self.validation_failure
        return payload


class BaseTool(ABC):
    """Shared tool interface."""

    @abstractmethod
    async def execute(self, validated_input, **kwargs) -> dict:
        raise NotImplementedError

    def summarize(self, payload: dict) -> str:
        return payload.get("summary", self.__class__.__name__)
