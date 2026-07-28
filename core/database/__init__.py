"""Database module initialization."""
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

from .connector import DBConnector, QueryResult, DBConfig, DatabaseSchema, TableSchema, ColumnSchema, DatabaseType
from .connection_pool import ConnectionPool, PoolHealth
from .result_processor import ResultProcessor, ProcessedResult, Aggregation, PaginatedResult
from .engine import (
    build_reference_dataset_statistics_evidence,
    build_reference_dataset_timeseries_evidence,
    execute_query,
    execute_range_query,
    infer_evidence_family,
    infer_prometheus_metric,
    normalize_query_result,
)
from .schema import schema_preview, metric_list_preview
from .repair import classify_query_error, should_retry_query
from .profile_cache import DEFAULT_PROFILE_TTL_SECONDS, profile_is_fresh, profile_path, read_profile, utc_now_iso, write_profile
try:
    from .semantic_context import MetricContextBuilder
except Exception:
    MetricContextBuilder = None
from .schema_linker import SchemaLinker
from .schema_linking import SchemaLinkingPipeline, SchemaLinkingPipelineResult
from .query_compiler import QueryCompiler, CompiledQuery
from .query_plan import (
    DatabaseQueryPlan,
    LinkedColumn,
    LinkedSource,
    QueryAlignment,
    QueryFilter,
    QueryJoin,
    QueryProjection,
    QuerySource,
    SchemaLinkingResult,
    TimeRangePlan,
    query_plan_from_dict,
    schema_linking_from_dict,
)
from .connectors import (
    InfluxDBConnector,
    InfluxDBConfig,
    TimescaleDBConnector,
    TimescaleDBConfig,
    PrometheusConnector,
    PrometheusConfig,
    DevMockPrometheusConnector,
    IoTDBConnector,
    IoTDBConfig,
    QuestDBConnector,
    QuestDBConfig,
    ClickHouseConnector,
    ClickHouseConfig,
)

try:
    from .query_translator import QueryTranslator, TranslationResult, ValidationResult
except Exception:
    QueryTranslator = None
    TranslationResult = None
    ValidationResult = None

try:
    from .metadata_fetcher import MetadataFetcher, TableMetadata, ColumnMetadata, MetricMetadata, TableSizeInfo
except Exception:
    MetadataFetcher = None
    TableMetadata = None
    ColumnMetadata = None
    MetricMetadata = None
    TableSizeInfo = None


class DatabaseFactory:
    """Factory for managing database connections and configurations."""

    _databases: dict[str, dict] = {}
    _pools: dict[str, ConnectionPool] = {}
    _connectors: dict[str, DBConnector] = {}
    _loaded = False
    _project_root = Path(__file__).resolve().parents[2]
    _config_dir = _project_root / "configs" / "databases"
    _database_dir = _project_root / "cache_data" / "database"
    _database_file = _database_dir / "databases.json"
    _profile_dir = _database_dir / "profiles"
    _TYPE_ALIASES = {
        "timescale": DatabaseType.TIMESCALEDB.value,
        "timescaledb": DatabaseType.TIMESCALEDB.value,
        "clickhouse": DatabaseType.CLICKHOUSE.value,
        "ch": DatabaseType.CLICKHOUSE.value,
    }

    @classmethod
    def normalize_database_type(cls, db_type: object) -> str:
        """Normalize user/config database type aliases to supported enum values."""
        if isinstance(db_type, DatabaseType):
            return db_type.value
        normalized = str(db_type or "").strip().lower()
        normalized = cls._TYPE_ALIASES.get(normalized, normalized)
        return DatabaseType(normalized).value

    @classmethod
    async def load_databases(cls) -> None:
        """Load database configurations from runtime storage."""
        if cls._loaded:
            return

        cls._database_dir.mkdir(parents=True, exist_ok=True)
        if not cls._database_file.exists():
            cls._databases = {}
        else:
            try:
                payload = json.loads(cls._database_file.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    raw_databases = payload.get("databases", payload)
                    if isinstance(raw_databases, list):
                        cls._databases = {
                            str(item.get("id") or item.get("name")): item
                            for item in raw_databases
                            if isinstance(item, dict) and (item.get("id") or item.get("name"))
                        }
                    elif isinstance(raw_databases, dict):
                        cls._databases = {
                            str(db_id): config
                            for db_id, config in raw_databases.items()
                            if isinstance(config, dict)
                        }
                    else:
                        cls._databases = {}
                else:
                    cls._databases = {}
            except Exception as e:
                logger.warning(f"Failed to load database configs: {e}")
                cls._databases = {}

        bootstrapped = cls._bootstrap_project_configs()
        if bootstrapped:
            await cls.save_databases()

        cls._loaded = True

    @classmethod
    def _bootstrap_project_configs(cls) -> bool:
        """Load bundled database configs into runtime storage when missing."""
        if not cls._config_dir.exists():
            return False

        changed = False
        project_config_paths = cls._iter_project_config_paths()
        project_config_sources = {
            str(config_path.relative_to(cls._project_root))
            for config_path in project_config_paths
        }

        for db_id, config in list(cls._databases.items()):
            config_source = config.get("config_source")
            if not isinstance(config_source, str):
                continue
            source_path = cls._project_root / config_source
            try:
                source_path.relative_to(cls._config_dir)
            except ValueError:
                continue
            if config_source not in project_config_sources:
                cls._databases.pop(db_id, None)
                cls._connectors.pop(db_id, None)
                changed = True

        for config_path in project_config_paths:
            config = cls._load_project_config(config_path)
            if not config:
                continue

            db_id = str(config.get("id") or config.get("name") or config_path.stem)
            if db_id in cls._databases:
                merged = cls._merge_project_config(cls._databases[db_id], config)
                if merged != cls._databases[db_id]:
                    cls._databases[db_id] = merged
                    changed = True
                continue

            cls._databases[db_id] = config
            changed = True

        return changed

    @classmethod
    def _iter_project_config_paths(cls) -> list[Path]:
        """Return project database connection configs from nested config folders."""
        config_paths = []
        for config_path in cls._config_dir.rglob("*.y*ml"):
            if any(part.startswith(".") or part == "SPEC" for part in config_path.parts):
                continue
            config_paths.append(config_path)
        return sorted(config_paths, key=lambda path: path.relative_to(cls._config_dir).as_posix())

    @classmethod
    def _merge_project_config(cls, current: dict, project_config: dict) -> dict:
        """Merge project-managed config metadata into an existing runtime config."""
        merged = dict(current)
        project_keys = cls._get_project_config_keys(project_config)
        previous_project_keys = cls._get_previous_project_config_keys(current)
        for key in previous_project_keys:
            if key not in project_keys:
                merged.pop(key, None)

        merged["config_source"] = project_config.get("config_source")
        merged["project_config_keys"] = project_config.get("project_config_keys", [])
        if "reference_dataset" in project_config:
            merged["reference_dataset"] = project_config["reference_dataset"]
        else:
            merged.pop("reference_dataset", None)
        if "influxdb_tasks" in project_config:
            merged["influxdb_tasks"] = project_config["influxdb_tasks"]
        else:
            merged.pop("influxdb_tasks", None)
        for key, value in project_config.items():
            if key in {"config_source", "project_config_keys", "reference_dataset", "influxdb_tasks"}:
                continue
            if merged.get(key) in (None, "", [], {}):
                merged[key] = value
        return merged

    @classmethod
    def _get_project_config_keys(cls, project_config: dict) -> set[str]:
        """Return keys managed by the project YAML config."""
        keys = project_config.get("project_config_keys")
        if isinstance(keys, list):
            return {str(key) for key in keys}
        return {
            str(key)
            for key in project_config
            if key not in {"config_source", "project_config_keys"}
        }

    @classmethod
    def _get_previous_project_config_keys(cls, current: dict) -> set[str]:
        """Return previously managed project keys for stale-key cleanup."""
        keys = current.get("project_config_keys")
        if isinstance(keys, list):
            return {str(key) for key in keys}
        if not isinstance(current.get("config_source"), str):
            return set()
        return {
            str(key)
            for key in current
            if key not in {"config_source", "project_config_keys", "status"}
        }

    @classmethod
    def _load_reference_dataset_config(cls, config_path: Path) -> dict | None:
        """Load an optional reference dataset config adjacent to a connection config."""
        reference_dir = config_path.parent / config_path.stem
        if not reference_dir.exists():
            return None

        for candidate in ("reference_dataset.yaml", "reference_dataset.yml"):
            reference_path = reference_dir / candidate
            if not reference_path.exists():
                continue
            try:
                payload = yaml.safe_load(reference_path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                logger.warning(f"Failed to read reference dataset config {reference_path}: {e}")
                return None

            if not isinstance(payload, dict):
                return None

            dataset_path = payload.get("dataset_path")
            if dataset_path:
                resolved_path = Path(str(dataset_path))
                if not resolved_path.is_absolute():
                    resolved_path = (cls._project_root / resolved_path).resolve()
                payload["resolved_dataset_path"] = str(resolved_path)

            payload["config_source"] = str(reference_path.relative_to(cls._project_root))
            return payload

        return None

    @classmethod
    def _load_project_config(cls, config_path: Path) -> dict | None:
        """Convert a project YAML config into the runtime database shape."""
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning(f"Failed to read database config {config_path}: {e}")
            return None

        if not isinstance(payload, dict):
            return None

        db_type = payload.get("db_type") or payload.get("type")
        host = payload.get("host")
        port = payload.get("port")
        if not db_type or not host or port in (None, ""):
            return None

        try:
            normalized_type = cls.normalize_database_type(db_type)
        except ValueError:
            logger.warning(f"Skipping unsupported database config {config_path}: {db_type}")
            return None

        name = str(payload.get("name") or config_path.stem)
        auth = payload.get("auth") if isinstance(payload.get("auth"), dict) else {}
        username = payload.get("username", auth.get("username"))
        password = payload.get("password", auth.get("password"))
        ssl_enabled = payload.get("ssl_enabled")
        if ssl_enabled is None:
            ssl_enabled = payload.get("ssl", False)

        config = {
            **payload,
            "id": name,
            "name": name,
            "db_type": normalized_type,
            "type": normalized_type,
            "host": str(host),
            "port": int(port),
            "username": username,
            "password": password,
            "database": payload.get("database"),
            "ssl_enabled": bool(ssl_enabled),
            "status": payload.get("status", "disconnected"),
            "config_source": str(config_path.relative_to(cls._project_root)),
        }
        reference_dataset = cls._load_reference_dataset_config(config_path)
        if reference_dataset:
            config["reference_dataset"] = reference_dataset
        config["project_config_keys"] = sorted(
            key for key in config if key not in {"config_source", "project_config_keys"}
        )
        return config

    @classmethod
    async def save_databases(cls) -> None:
        """Persist database configurations to runtime storage."""
        cls._database_dir.mkdir(parents=True, exist_ok=True)
        payload = {"databases": cls._databases}
        temp_file = cls._database_file.with_suffix(".json.tmp")
        temp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_file.chmod(0o600)
        temp_file.replace(cls._database_file)

    @classmethod
    async def _ensure_loaded(cls) -> None:
        await cls.load_databases()

    @classmethod
    async def _test_database(cls, db_id: str, db_config: dict) -> bool:
        """Test one saved database config and update cached runtime state."""
        try:
            connector = await cls.create_connector(**db_config)
            result = await connector.test_connection()
            if result.get("success", False):
                await cls.mark_connected(db_id, connector, persist=False)
                return True
        except Exception as e:
            logger.debug(f"Database health check failed for {db_id}: {e}")

        await cls.mark_disconnected(db_id, persist=False)
        return False

    @classmethod
    async def init_pools(cls) -> None:
        """Initialize connection pools for all configured databases."""
        await cls._ensure_loaded()
        for db_id, db_config in cls._databases.items():
            try:
                await cls._test_database(db_id, db_config)
            except Exception as e:
                logger.warning(f"Failed to init pool for {db_id}: {e}")

    @classmethod
    async def close_pools(cls) -> None:
        """Close all connection pools."""
        for pool in cls._pools.values():
            try:
                await pool.close()
            except Exception:
                pass
        cls._pools.clear()
        cls._connectors.clear()

    @classmethod
    async def list_databases(cls) -> list[dict]:
        """List all configured databases."""
        await cls._ensure_loaded()
        if cls._bootstrap_project_configs():
            await cls.save_databases()
        for db_id, db_config in cls._databases.items():
            await cls._test_database(db_id, db_config)
        await cls.save_databases()

        databases = [
            {
                "id": db_id,
                "name": db_config.get("name", db_id),
                "type": db_config.get("type", db_config.get("db_type", "unknown")),
                "status": db_config.get("status", "disconnected"),
                "host": db_config.get("host"),
                "port": db_config.get("port"),
                "username": db_config.get("username"),
                "database": db_config.get("database"),
                "ssl_enabled": db_config.get("ssl_enabled", False),
            }
            for db_id, db_config in cls._databases.items()
        ]
        return databases

    @classmethod
    async def get_database(cls, db_id: str) -> dict | None:
        """Get a configured database by id."""
        await cls._ensure_loaded()
        return cls._databases.get(db_id)

    @classmethod
    async def mark_connected(cls, db_id: str, connector: DBConnector, persist: bool = True) -> None:
        """Mark a configured database as successfully tested."""
        cls._connectors[db_id] = connector
        if db_id in cls._databases:
            cls._databases[db_id]["status"] = "connected"
            if persist:
                await cls.save_databases()

    @classmethod
    async def mark_disconnected(cls, db_id: str, persist: bool = True) -> None:
        """Mark a configured database as disconnected."""
        cls._connectors.pop(db_id, None)
        if db_id in cls._databases:
            cls._databases[db_id]["status"] = "disconnected"
            if persist:
                await cls.save_databases()

    @classmethod
    async def create_connector(cls, **kwargs) -> DBConnector:
        """Create a new database connector."""
        db_type = kwargs.pop("db_type", kwargs.get("type"))
        if "ssl_enabled" in kwargs and "ssl" not in kwargs:
            kwargs["ssl"] = kwargs["ssl_enabled"]
        if (
            str(db_type).lower() == DatabaseType.PROMETHEUS.value
            and kwargs.get("dev_mock_react")
        ):
            return DevMockPrometheusConnector(PrometheusConfig(**kwargs))
        return ConnectorFactory.create(db_type=db_type, **kwargs)

    @classmethod
    def profile_cache_path(cls, db_id: str) -> Path:
        """Return the persistent profile cache path for one database."""
        return profile_path(cls._profile_dir, db_id)

    @classmethod
    def read_profile_cache(cls, db_id: str) -> dict[str, Any] | None:
        """Read one stored database profile cache payload."""
        return read_profile(cls._profile_dir, db_id)

    @classmethod
    def profile_ttl_seconds(cls, config: dict) -> int:
        """Return configured profile cache TTL in seconds."""
        raw_value = config.get("schema_profile_ttl_seconds", config.get("profile_ttl_seconds", DEFAULT_PROFILE_TTL_SECONDS))
        try:
            return max(1, int(raw_value))
        except Exception:
            return DEFAULT_PROFILE_TTL_SECONDS

    @classmethod
    async def load_schema_with_profile_cache(
        cls,
        db_id: str,
        config: dict,
        *,
        force_refresh: bool = False,
    ) -> tuple[DatabaseSchema, dict[str, Any]]:
        """Load schema and merge a persistent data profile when available."""
        ttl_seconds = cls.profile_ttl_seconds(config)
        cached = cls.read_profile_cache(db_id)
        cached_profile = cached.get("data_profile") if isinstance(cached, dict) else None
        use_cache = (
            isinstance(cached_profile, dict)
            and not force_refresh
            and profile_is_fresh(cached or {}, ttl_seconds=ttl_seconds)
        )
        connector_config = dict(config)
        if use_cache:
            connector_config["schema_data_profile_enabled"] = False
        connector = await cls.create_connector(**connector_config)
        async with connector:
            schema = await connector.get_schema()
        if use_cache:
            schema.metadata["data_profile"] = cached_profile
            schema.metadata["profile_cache"] = {
                "source": "persistent_cache",
                "path": str(cls.profile_cache_path(db_id)),
                "generated_at": cached.get("generated_at"),
                "ttl_seconds": ttl_seconds,
                "fresh": True,
            }
            return schema, schema.metadata["profile_cache"]

        data_profile = schema.metadata.get("data_profile")
        cache_meta = {
            "source": "live_refresh" if data_profile else "unavailable",
            "path": str(cls.profile_cache_path(db_id)),
            "generated_at": utc_now_iso(),
            "ttl_seconds": ttl_seconds,
            "fresh": bool(data_profile),
        }
        if isinstance(data_profile, dict):
            payload = {
                "database_id": db_id,
                "database_type": config.get("type") or config.get("db_type"),
                "generated_at": cache_meta["generated_at"],
                "ttl_seconds": ttl_seconds,
                "data_profile": data_profile,
            }
            write_profile(cls._profile_dir, db_id, payload)
            cache_meta["source"] = "live_refresh_persisted"
        schema.metadata["profile_cache"] = cache_meta
        return schema, cache_meta

    @classmethod
    async def add_database(cls, **config) -> str:
        """Add a new database configuration."""
        await cls._ensure_loaded()
        db_id = config.get("name", str(len(cls._databases)))
        config["id"] = db_id
        config["status"] = config.get("status", "disconnected")
        cls._databases[db_id] = config
        await cls.save_databases()
        await cls.refresh_database_profile(db_id)
        return db_id

    @classmethod
    async def update_database(cls, db_id: str, **updates) -> dict | None:
        """Update an existing database configuration."""
        await cls._ensure_loaded()
        if db_id not in cls._databases:
            return None

        current_config = cls._databases[db_id]
        next_config = {
            **current_config,
            **{key: value for key, value in updates.items() if value is not None},
        }
        next_config["id"] = db_id
        next_config["status"] = "disconnected"

        cls._databases[db_id] = next_config
        cls._connectors.pop(db_id, None)
        await cls.save_databases()
        await cls.refresh_database_profile(db_id)
        return next_config

    @classmethod
    async def delete_database(cls, db_id: str) -> bool:
        """Delete a database configuration."""
        await cls._ensure_loaded()
        if db_id in cls._databases:
            del cls._databases[db_id]
            if db_id in cls._connectors:
                del cls._connectors[db_id]
            try:
                cls.profile_cache_path(db_id).unlink(missing_ok=True)
            except Exception:
                pass
            await cls.save_databases()
            return True
        return False

    @classmethod
    async def refresh_database_profile(cls, db_id: str) -> dict[str, Any]:
        """Refresh and persist one database profile without failing config writes."""
        await cls._ensure_loaded()
        config = cls._databases.get(db_id)
        if not isinstance(config, dict):
            return {"success": False, "error": f"Database '{db_id}' was not found."}
        if isinstance(config.get("reference_dataset"), dict):
            return {"success": False, "skipped": True, "reason": "reference_dataset_profile_is_derived_from_config"}
        try:
            _schema, cache_meta = await cls.load_schema_with_profile_cache(db_id, config, force_refresh=True)
            return {"success": cache_meta.get("source") != "unavailable", "profile_cache": cache_meta}
        except Exception as exc:
            logger.debug(f"Failed to refresh profile for {db_id}: {exc}")
            return {"success": False, "error": str(exc), "profile_cache": {"path": str(cls.profile_cache_path(db_id))}}

    @classmethod
    async def health_check(cls) -> bool:
        """Check database health."""
        return True


class ConnectorFactory:
    """Factory for creating database connectors."""

    _CONNECTORS = {
        DatabaseType.INFLUXDB: (InfluxDBConnector, InfluxDBConfig),
        DatabaseType.TIMESCALEDB: (TimescaleDBConnector, TimescaleDBConfig),
        DatabaseType.PROMETHEUS: (PrometheusConnector, PrometheusConfig),
        DatabaseType.IOTDB: (IoTDBConnector, IoTDBConfig),
        DatabaseType.QUESTDB: (QuestDBConnector, QuestDBConfig),
        DatabaseType.CLICKHOUSE: (ClickHouseConnector, ClickHouseConfig),
    }

    @classmethod
    def create(
        cls,
        db_type: DatabaseType | str,
        **kwargs,
    ) -> DBConnector:
        """Create a database connector.

        Args:
            db_type: Database type (enum or string)
            **kwargs: Configuration parameters

        Returns:
            DBConnector instance

        Raises:
            ValueError: If database type is not supported
        """
        if isinstance(db_type, str):
            db_type = DatabaseType(DatabaseFactory.normalize_database_type(db_type))

        if db_type not in cls._CONNECTORS:
            raise ValueError(f"Unsupported database type: {db_type}")

        connector_class, config_class = cls._CONNECTORS[db_type]
        config = config_class(**kwargs)
        return connector_class(config)

    @classmethod
    def register(
        cls,
        db_type: DatabaseType,
        connector_class: type[DBConnector],
        config_class: type[DBConfig],
    ) -> None:
        """Register a new connector type.

        Args:
            db_type: Database type enum
            connector_class: Connector class
            config_class: Config class
        """
        cls._CONNECTORS[db_type] = (connector_class, config_class)


__all__ = [
    "DBConnector",
    "QueryResult",
    "DBConfig",
    "DatabaseSchema",
    "TableSchema",
    "ColumnSchema",
    "DatabaseType",
    "DatabaseFactory",
    "ConnectionPool",
    "PoolHealth",
    "QueryTranslator",
    "TranslationResult",
    "ValidationResult",
    "ResultProcessor",
    "ProcessedResult",
    "Aggregation",
    "PaginatedResult",
    "MetricContextBuilder",
    "SchemaLinker",
    "SchemaLinkingPipeline",
    "SchemaLinkingPipelineResult",
    "QueryCompiler",
    "CompiledQuery",
    "DatabaseQueryPlan",
    "LinkedColumn",
    "LinkedSource",
    "QueryAlignment",
    "QueryFilter",
    "QueryJoin",
    "QueryProjection",
    "QuerySource",
    "SchemaLinkingResult",
    "TimeRangePlan",
    "query_plan_from_dict",
    "schema_linking_from_dict",
    "MetadataFetcher",
    "TableMetadata",
    "ColumnMetadata",
    "MetricMetadata",
    "TableSizeInfo",
    "ConnectorFactory",
    "InfluxDBConnector",
    "InfluxDBConfig",
    "TimescaleDBConnector",
    "TimescaleDBConfig",
    "PrometheusConnector",
    "PrometheusConfig",
    "IoTDBConnector",
    "IoTDBConfig",
    "QuestDBConnector",
    "QuestDBConfig",
]
