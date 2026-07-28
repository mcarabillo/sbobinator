"""Sbobinator — FastAPI audio transcription service entry point."""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from backend.api import router
from utils.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle startup and shutdown events."""
    logger.info(settings().summary())
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    cfg = settings()

    app = FastAPI(
        title="Sbobinator",
        description="AI-powered audio transcription service",
        version="0.1.0",
        debug=cfg.debug,
        lifespan=lifespan,
    )

    # --- Middleware ---
    if cfg.cors.origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # --- Logging ---
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # --- Routers ---
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    cfg = settings()
    uvicorn.run(
        "main:app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.debug,
        log_level=cfg.logging.level.lower(),
    )
