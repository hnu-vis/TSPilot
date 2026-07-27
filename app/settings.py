"""Runtime settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_tspilot_root() -> str:
    project_root = _default_project_root()
    return str(project_root.resolve())


class Settings(BaseSettings):
    """Backend settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_title: str = "TSPilot v0.2 API"
    app_version: str = "0.1.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"

    backend_host: str = Field(default="0.0.0.0", alias="TSPILOT_BACKEND_HOST")
    backend_port: int = Field(default=5680, alias="TSPILOT_BACKEND_PORT")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://api.openai.com/v1", alias="OPENAI_API_BASE")
    openai_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_MODEL")
    openai_temperature: float = 0.0

    tspilot_root: str = Field(default_factory=_default_tspilot_root, alias="TSPILOT_ROOT")
    database_config_dir: str | None = Field(default=None, alias="TSPILOT_DATABASE_CONFIG_DIR")
    knowledge_base_dir: str | None = Field(default=None, alias="TSPILOT_KNOWLEDGE_BASE_DIR")
    conversation_log_enabled: bool = Field(default=True, alias="TSPILOT_CONVERSATION_LOG_ENABLED")
    conversation_log_dir: str | None = Field(default=None, alias="TSPILOT_CONVERSATION_LOG_DIR")

    max_iterations: int = 30
    max_prompt_tokens: int = 12000
    max_history_messages: int = 12
    max_tool_history_items: int = 8
    max_observation_chars: int = 1600
    max_visible_rows: int = 60
    max_visible_points: int = 240

    @property
    def resolved_database_config_dir(self) -> Path:
        if self.database_config_dir:
            return Path(self.database_config_dir).resolve()
        return (Path(self.tspilot_root) / "configs" / "databases").resolve()

    @property
    def resolved_knowledge_base_dir(self) -> Path:
        if self.knowledge_base_dir:
            return Path(self.knowledge_base_dir).resolve()
        return Path(self.tspilot_root).resolve()

    @property
    def resolved_conversation_log_dir(self) -> Path:
        if self.conversation_log_dir:
            return Path(self.conversation_log_dir).resolve()
        return (Path(self.tspilot_root) / "cache_data" / "conversation_logs").resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
