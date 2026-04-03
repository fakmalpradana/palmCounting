# Copyright © 2026 Geo AI Twinverse.
# Contributors: Fikri Kurniawan, Fairuz Akmal Pradana

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers import admin as admin_router
from app.routers import auth as auth_router
from app.routers import inference as inference_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    # Log the contents of the baked-in default models directory so we can
    # verify the ONNX file made it into the Docker image on Cloud Run.
    default_models_dir = (
        Path(__file__).resolve().parent / "routers" / ".." / "models" / "default"
    ).resolve()
    log.info("DEFAULT_MODELS_DIR resolved to: %s", default_models_dir)
    if default_models_dir.exists():
        contents = [f.name for f in sorted(default_models_dir.iterdir())]
        log.info("DEFAULT_MODELS_DIR contents: %s", contents)
    else:
        log.warning("DEFAULT_MODELS_DIR does not exist in the container!")
    yield
    # ── Shutdown (nothing to clean up) ───────────────────────────────────


app = FastAPI(
    title="Palm Counter API",
    description="YOLO ONNX palm-tree detection — GeoTIFF in, GeoJSON out.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(admin_router.router)
app.include_router(inference_router.router)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Serve the frontend from ./static at root
app.mount("/", StaticFiles(directory="static", html=True), name="static")
