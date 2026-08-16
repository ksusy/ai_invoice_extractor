"""ASGI entry point – FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.upload import router as upload_router
from src.api.routes.results import router as results_router
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup / shutdown hooks."""
    from src.infrastructure.db.database import close_db, init_db

    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    await init_db()
    logger.info("Application started – DB initialized")
    yield
    await close_db()
    logger.info("Application shutdown – DB closed")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # -- Health check --
    @app.get("/health", tags=["monitoring"])
    async def health_check():
        return {"status": "ok"}

    # -- Register routers --
    app.include_router(upload_router, prefix="/api/v1")
    app.include_router(results_router, prefix="/api/v1")

    return app


app = create_app()
