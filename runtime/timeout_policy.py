"""Validated, single-source timeout policy for runtime and Tools."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


class RuntimeTimeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_deadline_seconds: float = Field(gt=0)
    agent_turn_seconds: float = Field(gt=0)


class ServiceTimeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_transport_seconds: float = Field(gt=0)
    embedding_request_seconds: float = Field(gt=0)


class ToolTimeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_seconds: float = Field(gt=0)
    stages: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stage_budgets(self):
        invalid = {
            name: seconds
            for name, seconds in self.stages.items()
            if not str(name).strip() or isinstance(seconds, bool) or float(seconds) <= 0
        }
        if invalid:
            raise ValueError(f"Tool stage timeouts must be positive seconds: {invalid}")
        oversized = {
            name: seconds
            for name, seconds in self.stages.items()
            if float(seconds) > self.execution_seconds
        }
        if oversized:
            raise ValueError(
                "Tool stage timeouts cannot exceed the Tool execution budget: "
                f"{oversized} > {self.execution_seconds}"
            )
        return self

    def stage_seconds(self, name: str) -> float:
        try:
            return float(self.stages[name])
        except KeyError as exc:
            raise KeyError(f"Timeout policy has no stage '{name}'") from exc


class TimeoutPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    runtime: RuntimeTimeouts
    services: ServiceTimeouts
    tools: dict[str, ToolTimeouts]

    @model_validator(mode="after")
    def validate_request_budget(self):
        longest = max(item.execution_seconds for item in self.tools.values())
        if self.runtime.request_deadline_seconds < longest:
            raise ValueError(
                "request_deadline_seconds must be at least the longest Tool execution budget"
            )
        return self

    def validate_tool_inventory(self, registered_names) -> None:
        registered = {str(name) for name in registered_names}
        configured = set(self.tools)
        missing = sorted(registered - configured)
        unknown = sorted(configured - registered)
        if missing or unknown:
            raise ValueError(
                f"Timeout policy Tool inventory mismatch: missing={missing}, unknown={unknown}"
            )

    def tool(self, name: str) -> ToolTimeouts:
        try:
            return self.tools[name]
        except KeyError as exc:
            raise KeyError(f"Timeout policy has no Tool '{name}'") from exc


def default_timeout_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "timeouts.yaml"


@lru_cache(maxsize=8)
def load_timeout_policy(path: str | Path | None = None) -> TimeoutPolicy:
    resolved = Path(path).expanduser().resolve() if path else default_timeout_config_path().resolve()
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Timeout policy file was not found: {resolved}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Unable to read timeout policy '{resolved}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Timeout policy '{resolved}' must contain a YAML object")
    return TimeoutPolicy.model_validate(payload)
