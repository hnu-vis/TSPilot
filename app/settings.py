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
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_api_base: str | None = Field(default=None, alias="EMBEDDING_API_BASE")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    memory_embedding_top_k: int = Field(default=6, alias="MEMORY_EMBEDDING_TOP_K")
    memory_embedding_score_threshold: float = Field(default=0.25, alias="MEMORY_EMBEDDING_SCORE_THRESHOLD")
    memory_embedding_cache_dir: str | None = Field(default=None, alias="MEMORY_EMBEDDING_CACHE_DIR")
    insight_memory_learning_enabled: bool = Field(default=True, alias="INSIGHT_MEMORY_LEARNING_ENABLED")
    insight_memory_learning_dir: str | None = Field(default=None, alias="INSIGHT_MEMORY_LEARNING_DIR")
    insight_memory_learning_batch_size: int = Field(default=20, alias="INSIGHT_MEMORY_LEARNING_BATCH_SIZE")
    insight_memory_learning_max_wait_seconds: float = Field(default=600.0, alias="INSIGHT_MEMORY_LEARNING_MAX_WAIT_SECONDS")
    insight_memory_learning_poll_seconds: float = Field(default=5.0, alias="INSIGHT_MEMORY_LEARNING_POLL_SECONDS")
    insight_memory_learning_lease_seconds: float = Field(default=180.0, alias="INSIGHT_MEMORY_LEARNING_LEASE_SECONDS")
    insight_memory_learning_max_attempts: int = Field(default=3, alias="INSIGHT_MEMORY_LEARNING_MAX_ATTEMPTS")
    insight_memory_learning_llm_chunk_size: int = Field(default=5, alias="INSIGHT_MEMORY_LEARNING_LLM_CHUNK_SIZE")

    tspilot_root: str = Field(default_factory=_default_tspilot_root, alias="TSPILOT_ROOT")
    database_config_dir: str | None = Field(default=None, alias="TSPILOT_DATABASE_CONFIG_DIR")
    knowledge_base_dir: str | None = Field(default=None, alias="TSPILOT_KNOWLEDGE_BASE_DIR")
    model_config_path: str | None = Field(default=None, alias="TSPILOT_MODEL_CONFIG_PATH")
    model_config_dir: str | None = Field(default=None, alias="TSPILOT_MODEL_CONFIG_DIR")
    conversation_log_enabled: bool = Field(default=True, alias="TSPILOT_CONVERSATION_LOG_ENABLED")
    conversation_log_dir: str | None = Field(default=None, alias="TSPILOT_CONVERSATION_LOG_DIR")
    visualization_artifact_dir: str | None = Field(default=None, alias="TSPILOT_VISUALIZATION_ARTIINSIGHT_DIR")

    max_iterations: int = 30
    max_prompt_tokens: int = 12000
    max_history_messages: int = 12
    max_tool_history_items: int = 8
    max_observation_chars: int = 1600
    max_visible_rows: int = 60
    max_visible_points: int = 240
    agent_turn_timeout_seconds: float = Field(default=45.0, alias="TSPILOT_AGENT_TURN_TIMEOUT_SECONDS")
    request_deadline_seconds: float = Field(default=180.0, alias="TSPILOT_REQUEST_DEADLINE_SECONDS")

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
    def resolved_model_config_path(self) -> Path:
        if self.model_config_path:
            return Path(self.model_config_path).resolve()
        return (Path(self.tspilot_root) / "configs" / "models.json").resolve()

    @property
    def resolved_model_config_dir(self) -> Path:
        if self.model_config_dir:
            return Path(self.model_config_dir).resolve()
        if self.model_config_path:
            return self.resolved_model_config_path.with_suffix("")
        return (Path(self.tspilot_root) / "configs" / "models").resolve()

    @property
    def resolved_conversation_log_dir(self) -> Path:
        if self.conversation_log_dir:
            return Path(self.conversation_log_dir).resolve()
        return (Path(self.tspilot_root) / "cache_data" / "conversation_logs").resolve()

    @property
    def resolved_visualization_artifact_dir(self) -> Path:
        if self.visualization_artifact_dir:
            return Path(self.visualization_artifact_dir).resolve()
        return (Path(self.tspilot_root) / "cache_data" / "visualizations").resolve()

    @property
    def resolved_embedding_api_key(self) -> str | None:
        return self.embedding_api_key or self.openai_api_key

    @property
    def resolved_embedding_api_base(self) -> str:
        return self.embedding_api_base or self.openai_api_base

    @property
    def resolved_memory_embedding_cache_dir(self) -> Path:
        if self.memory_embedding_cache_dir:
            return Path(self.memory_embedding_cache_dir).resolve()
        return (Path(self.tspilot_root) / "cache_data" / "database" / "insight_memory_embeddings").resolve()

    @property
    def resolved_insight_memory_learning_dir(self) -> Path:
        if self.insight_memory_learning_dir:
            return Path(self.insight_memory_learning_dir).resolve()
        return (Path(self.tspilot_root) / "cache_data" / "database" / "insight_memory_learning").resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
