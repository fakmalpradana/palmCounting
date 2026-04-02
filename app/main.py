# Copyright © 2026 Geo AI Twinverse.
# Contributors: Fikri Kurniawan, Fairuz Akmal Pradana

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers import admin as admin_router
from app.routers import auth as auth_router
from app.routers import inference as inference_router

app = FastAPI(
    title="Palm Counter API",
    description="YOLO ONNX palm-tree detection — GeoTIFF in, GeoJSON out.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(admin_router.router)
app.include_router(inference_router.router)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Serve the frontend from ./static at root
app.mount("/", StaticFiles(directory="static", html=True), name="static")
