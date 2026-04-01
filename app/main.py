from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import inference as inference_router

app = FastAPI(
    title="Palm Counter API",
    description="YOLO ONNX palm-tree detection — GeoTIFF in, GeoJSON out.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_router.router)

# Serve the frontend from ./static at root
app.mount("/", StaticFiles(directory="static", html=True), name="static")
