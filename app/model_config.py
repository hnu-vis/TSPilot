"""Persistent, runtime-editable model configuration.

Environment settings remain the source of defaults. This store only records explicit
workspace overrides, so deployments can keep using environment-based configuration.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from app.settings import Settings, get_settings


@dataclass(frozen=True)
class AIModelSettings:
    api_base: str
    api_key: str | None
    model: str
    temperature: float
    embedding_api_base: str
    embedding_api_key: str | None
    embedding_model: str


class ModelConfigStore:
    """Read and atomically update model overrides without exposing stored secrets."""

    def __init__(self, path: Path, settings: Settings):
        self.path = path
        self.settings = settings

    def read_overrides(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read model configuration: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Model configuration must contain a JSON object.")
        return payload

    def effective_ai(self, llm_connection_id: str | None = None) -> AIModelSettings:
        llm = self.active_connection("llm")
        if llm_connection_id:
            selected = self.connection("llm", llm_connection_id)
            if selected is None:
                raise ValueError(f"Unknown language model connection '{llm_connection_id}'.")
            llm = selected
        embedding = self.active_connection("embedding")
        llm_key = _optional_string(llm.get("api_key")) or self.settings.openai_api_key
        embedding_key = _optional_string(embedding.get("api_key")) or self.settings.embedding_api_key or llm_key
        llm_base = _string(llm.get("api_base"), self.settings.openai_api_base)
        return AIModelSettings(
            api_base=llm_base,
            api_key=llm_key,
            model=_string(llm.get("model"), self.settings.openai_model),
            temperature=self.settings.openai_temperature,
            embedding_api_base=_string(
                embedding.get("api_base"),
                self.settings.embedding_api_base or llm_base,
            ),
            embedding_api_key=embedding_key,
            embedding_model=_string(embedding.get("model"), self.settings.embedding_model),
        )

    def public_config(self) -> dict[str, Any]:
        overrides = self.read_overrides()
        machine_learning = _mapping(overrides.get("machine_learning"))
        return {
            "ai": {
                "llm": self.public_connections("llm"),
                "embedding": self.public_connections("embedding"),
            },
            "machine_learning": machine_learning,
        }

    def connections(self, section: str) -> list[dict[str, Any]]:
        """Return environment defaults merged with workspace-defined connections."""
        self._validate_section(section)
        default = self._default_connection(section)
        section_config = self._section_config(section)
        stored_models = section_config.get("models")
        if isinstance(stored_models, list):
            stored = [_mapping(item) for item in stored_models if isinstance(item, dict)]
        elif section_config.get("model") or section_config.get("api_base"):
            stored = [{"id": f"{section}-default", **section_config}]
        else:
            stored = []
        file_models = self._file_ai_connections(section)
        by_id = {default["id"]: default}
        for item in [*stored, *file_models]:
            connection_id = _string(item.get("id"), str(uuid4()))
            resolved = self._resolve_ai_connection(item)
            by_id[connection_id] = {**by_id.get(connection_id, {}), **resolved, "id": connection_id, "source": "workspace"}
        return list(by_id.values())

    def active_connection(self, section: str) -> dict[str, Any]:
        connections = self.connections(section)
        active_id = _optional_string(self._section_config(section).get("active_id"))
        return next((item for item in connections if item["id"] == active_id), connections[0])

    def public_connections(self, section: str) -> dict[str, Any]:
        active = self.active_connection(section)
        fallback_key = self.settings.openai_api_key if section == "llm" else (
            self.settings.embedding_api_key or self.settings.openai_api_key
        )
        models = []
        for item in self.connections(section):
            models.append({
                "id": item["id"],
                "provider": "OpenAI compatible",
                "api_base": item["api_base"],
                "model": item["model"],
                "api_key_configured": bool(_optional_string(item.get("api_key")) or fallback_key),
                "is_active": item["id"] == active["id"],
                "source": item.get("source", "workspace"),
                "config_path": item.get("config_path"),
            })
        return {"active_id": active["id"], "models": models}

    def upsert_ai(self, section: str, values: dict[str, Any]) -> str:
        """Create or update one connection while preserving all sibling models."""
        self._validate_section(section)
        active_before_update = self.active_connection(section)["id"]
        connection_id = _string(values.pop("id", None), f"{section}-{uuid4().hex}")
        if section not in {"llm", "embedding"}:
            raise ValueError(f"Unsupported AI model section '{section}'.")
        payload = self.read_overrides()
        ai = payload.setdefault("ai", {})
        current = _mapping(ai.get(section))
        if isinstance(current.get("models"), list):
            stored = current["models"]
        elif current.get("model") or current.get("api_base"):
            stored = [{"id": f"{section}-default", **current}]
        else:
            stored = []
        normalized = [_mapping(item) for item in stored if isinstance(item, dict)]
        file_existing = self._raw_ai_file_connection(section, connection_id)
        existing = file_existing or next((item for item in normalized if item.get("id") == connection_id), None) or {"id": connection_id}
        existing = {key: value for key, value in existing.items() if key != "config_path"}
        file_payload = {**existing, "schema_version": "1", "kind": section, "id": connection_id}
        for key, value in values.items():
            if key != "api_key" or value is not None:
                file_payload[key] = value
        if "api_base" in values:
            file_payload.pop("api_base_env", None)
        if values.get("api_key") is not None:
            file_payload.pop("api_key_env", None)
        target_path = self._ai_model_directory(section) / f"{self._safe_config_filename(file_payload['model'])}.json"
        previous_path = self._ai_file_path_by_id(section, connection_id)
        self._write_path(target_path, file_payload)
        if previous_path is not None and previous_path != target_path:
            previous_path.unlink()
        normalized = [item for item in normalized if item.get("id") != connection_id]
        current = {"models": normalized, "active_id": current.get("active_id") or active_before_update}
        ai[section] = current
        self._write(payload)
        return connection_id

    def activate_ai(self, section: str, connection_id: str) -> None:
        self._validate_section(section)
        if connection_id not in {item["id"] for item in self.connections(section)}:
            raise ValueError(f"Unknown {section} model connection '{connection_id}'.")
        payload = self.read_overrides()
        section_config = payload.setdefault("ai", {}).setdefault(section, {})
        section_config["active_id"] = connection_id
        self._write(payload)

    def connection(self, section: str, connection_id: str) -> dict[str, Any] | None:
        return next((item for item in self.connections(section) if item["id"] == connection_id), None)

    def connection_api_key(self, section: str, connection_id: str) -> str | None:
        connection = self.connection(section, connection_id)
        if connection is None:
            return None
        configured = _optional_string(connection.get("api_key"))
        if configured:
            return configured
        if section == "llm":
            return self.settings.openai_api_key
        return self.settings.embedding_api_key or self.settings.openai_api_key

    def delete_ai(self, section: str, connection_id: str) -> None:
        self._validate_section(section)
        payload = self.read_overrides()
        section_config = _mapping(payload.setdefault("ai", {}).get(section))
        stored = section_config.get("models") if isinstance(section_config.get("models"), list) else []
        remaining = [item for item in stored if _mapping(item).get("id") != connection_id]
        file_path = self._ai_file_path_by_id(section, connection_id)
        if file_path is not None:
            file_path.unlink()
        elif len(remaining) == len(stored):
            raise ValueError(f"Model connection '{connection_id}' is provided by the environment and cannot be removed.")
        section_config["models"] = remaining
        if section_config.get("active_id") == connection_id:
            section_config.pop("active_id", None)
        payload["ai"][section] = section_config
        self._write(payload)

    def _section_config(self, section: str) -> dict[str, Any]:
        self._validate_section(section)
        return _mapping(_mapping(self.read_overrides().get("ai")).get(section))

    def _default_connection(self, section: str) -> dict[str, Any]:
        if section == "llm":
            return {
                "id": "llm-default",
                "api_base": self.settings.openai_api_base,
                "api_key": self.settings.openai_api_key,
                "model": self.settings.openai_model,
                "source": "environment",
            }
        return {
            "id": "embedding-default",
            "api_base": self.settings.embedding_api_base or self.settings.openai_api_base,
            "api_key": self.settings.embedding_api_key or self.settings.openai_api_key,
            "model": self.settings.embedding_model,
            "source": "environment",
        }

    def _file_ai_connections(self, section: str) -> list[dict[str, Any]]:
        directory = self._ai_model_directory(section)
        if not directory.exists():
            return []
        items = []
        for path in sorted(directory.glob("*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Unable to read AI model config '{path}': {exc}") from exc
            if not isinstance(item, dict) or item.get("kind") != section:
                raise RuntimeError(f"AI model config '{path}' has an invalid kind.")
            items.append({**item, "config_path": str(path)})
        return items

    def _raw_ai_file_connection(self, section: str, connection_id: str) -> dict[str, Any] | None:
        return next((item for item in self._file_ai_connections(section) if item.get("id") == connection_id), None)

    def _ai_file_path_by_id(self, section: str, connection_id: str) -> Path | None:
        item = self._raw_ai_file_connection(section, connection_id)
        return Path(item["config_path"]) if item and item.get("config_path") else None

    def _resolve_ai_connection(self, item: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(item)
        if item.get("api_base_env"):
            resolved["api_base"] = self._setting_for_env_ref(str(item["api_base_env"]))
        if item.get("api_key_env"):
            resolved["api_key"] = self._setting_for_env_ref(str(item["api_key_env"]))
        return resolved

    def _setting_for_env_ref(self, name: str) -> str | None:
        values = {
            "OPENAI_API_BASE": self.settings.openai_api_base,
            "OPENAI_API_KEY": self.settings.openai_api_key,
            "EMBEDDING_API_BASE": self.settings.embedding_api_base or self.settings.openai_api_base,
            "EMBEDDING_API_KEY": self.settings.embedding_api_key or self.settings.openai_api_key,
        }
        if name not in values:
            raise RuntimeError(f"Unsupported model config environment reference '{name}'.")
        return values[name]

    def _ai_model_directory(self, section: str) -> Path:
        self._validate_section(section)
        return self.settings.resolved_model_config_dir / "ai" / section

    @staticmethod
    def _safe_config_filename(value: Any) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip(".-")
        return normalized or uuid4().hex

    @staticmethod
    def _validate_section(section: str) -> None:
        if section not in {"llm", "embedding"}:
            raise ValueError(f"Unsupported AI model section '{section}'.")

    def update_machine_learning(self, values: dict[str, str]) -> None:
        payload = self.read_overrides()
        current = payload.setdefault("machine_learning", {})
        current.update(values)
        self._write(payload)

    def external_machine_models(self, task: str) -> list[dict[str, Any]]:
        directory = self._machine_model_directory(task)
        if not directory.exists():
            return []
        models = []
        for path in sorted(directory.glob("*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Unable to read external model config '{path}': {exc}") from exc
            if not isinstance(item, dict):
                raise RuntimeError(f"External model config '{path}' must contain a JSON object.")
            if item.get("task") != task:
                raise RuntimeError(f"External model config '{path}' has an invalid task.")
            models.append(item)
        return models

    def external_machine_model(self, task: str, name: str) -> dict[str, Any] | None:
        normalized = self._machine_model_name(name)
        return next((item for item in self.external_machine_models(task) if item.get("name") == normalized), None)

    def upsert_external_machine_model(self, task: str, values: dict[str, Any]) -> dict[str, Any]:
        name = self._machine_model_name(values.get("name"))
        existing = self.external_machine_model(task, name) or {}
        config = {
            **existing,
            "schema_version": "1",
            "task": task,
            "name": name,
            "endpoint": _string(values.get("endpoint"), ""),
            "timeout_seconds": float(values.get("timeout_seconds", existing.get("timeout_seconds", 30.0))),
        }
        if values.get("api_key") is not None:
            config["api_key"] = _optional_string(values.get("api_key"))
        self._write_path(self._machine_model_directory(task) / f"{name}.json", config)
        return config

    def delete_external_machine_model(self, task: str, name: str) -> None:
        path = self._machine_model_directory(task) / f"{self._machine_model_name(name)}.json"
        if not path.is_file():
            raise ValueError(f"External {task} model '{name}' was not found.")
        path.unlink()

    def public_machine_models(self, task: str, registered_names: list[str], active_name: str) -> list[dict[str, Any]]:
        external = {item["name"]: item for item in self.external_machine_models(task)}
        return [
            {
                "id": name,
                "name": name,
                "source": "api" if name in external else "built_in",
                "endpoint": external.get(name, {}).get("endpoint"),
                "timeout_seconds": external.get(name, {}).get("timeout_seconds"),
                "api_key_configured": bool(external.get(name, {}).get("api_key")),
                "is_active": name == active_name,
                "config_path": str(self._machine_model_directory(task) / f"{name}.json") if name in external else None,
            }
            for name in registered_names
        ]

    def _machine_model_directory(self, task: str) -> Path:
        if task not in {"forecast", "anomaly"}:
            raise ValueError(f"Unsupported machine learning task '{task}'.")
        return self.settings.resolved_model_config_dir / "machine_learning" / task

    @staticmethod
    def _machine_model_name(value: Any) -> str:
        name = str(value or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name):
            raise ValueError("Model name may contain lowercase letters, numbers, dots, dashes, and underscores.")
        return name

    def _write(self, payload: dict[str, Any]) -> None:
        self._write_path(self.path, payload)

    @staticmethod
    def _write_path(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)


@lru_cache(maxsize=1)
def get_model_config_store() -> ModelConfigStore:
    settings = get_settings()
    return ModelConfigStore(settings.resolved_model_config_path, settings)


def apply_persisted_machine_learning_defaults() -> None:
    """Apply valid persisted selections after built-in model registration."""
    from core.timeseries.anomaly_registry import register_api_anomaly_detector, set_default_anomaly_detector
    from core.timeseries.forecast_registry import register_api_forecast_model, set_default_forecast_model

    store = get_model_config_store()
    for item in store.external_machine_models("forecast"):
        register_api_forecast_model(
            item["name"], endpoint=item["endpoint"], timeout_seconds=item["timeout_seconds"],
            headers=_authorization_headers(item.get("api_key")),
        )
    for item in store.external_machine_models("anomaly"):
        register_api_anomaly_detector(
            item["name"], endpoint=item["endpoint"], timeout_seconds=item["timeout_seconds"],
            headers=_authorization_headers(item.get("api_key")),
        )
    selected = _mapping(store.read_overrides().get("machine_learning"))
    if selected.get("forecast_model"):
        set_default_forecast_model(str(selected["forecast_model"]))
    if selected.get("anomaly_detector"):
        set_default_anomaly_detector(str(selected["anomaly_detector"]))


def _authorization_headers(api_key: Any) -> dict[str, str]:
    normalized = _optional_string(api_key)
    return {"Authorization": f"Bearer {normalized}"} if normalized else {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string(value: Any, default: str) -> str:
    normalized = str(value or "").strip()
    return normalized or default


def _optional_string(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None
