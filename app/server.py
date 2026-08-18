"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import get_settings
from app.model_config import apply_persisted_machine_learning_defaults
from app.routes.chat import router as chat_router
from app.routes.resources import router as resources_router
from app.routes.visualizations import router as visualizations_router
from app.deps import get_insight_memory_learning_worker
import app.deps as runtime_dependencies
from core.key_insight.learning import reset_legacy_insight_memory_once, separate_insight_memory_scopes_once


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    apply_persisted_machine_learning_defaults()
    worker = None
    if settings.insight_memory_learning_enabled:
        reset_legacy_insight_memory_once(
            root=settings.resolved_insight_memory_learning_dir,
            embedding_root=settings.resolved_memory_embedding_cache_dir,
        )
        separate_insight_memory_scopes_once(
            root=settings.resolved_insight_memory_learning_dir,
            embedding_root=settings.resolved_memory_embedding_cache_dir,
        )
        if settings.openai_api_key:
            worker = get_insight_memory_learning_worker()
            worker.start()
    try:
        yield
    finally:
        if worker is not None:
            await worker.stop()
        registry_factory = runtime_dependencies.get_tool_registry
        cache_info = getattr(registry_factory, "cache_info", None)
        if callable(cache_info) and cache_info().currsize:
            await registry_factory().close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)
    app.include_router(resources_router)
    app.include_router(visualizations_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
