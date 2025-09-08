"""
Highlight Detector FastAPI Server

Main application entry point with API routes and WebSocket support.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .core.config import get_settings
from .core.database import init_db
from .core.websocket import ConnectionManager
from .api.routes import sessions, render, upload, health
from .services.task_manager import TaskManager
from .models.schemas import (
    SessionCreate, SessionResponse, MediaFileResponse,
    HighlightEvent, RenderJobCreate, RenderJobResponse,
    OutputPreset, DetectionProgress
)

# Global instances
task_manager = TaskManager()
websocket_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    await init_db()
    await task_manager.start()
    
    yield
    
    # Shutdown
    await task_manager.stop()

# Create FastAPI app
app = FastAPI(
    title="Highlight Detector API",
    description="Sports and Podcast Highlight Detection Service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(render.router, prefix="/api", tags=["render"])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Handle client messages if needed
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

@app.get("/api/presets")
async def get_output_presets() -> List[OutputPreset]:
    """Get available output presets."""
    return [
        OutputPreset(
            id="vertical",
            name="Vertical (9:16)",
            width=1080,
            height=1920,
            aspectRatio="9:16",
            cropStrategy="motion_centroid"
        ),
        OutputPreset(
            id="square",
            name="Square (1:1)",
            width=1080,
            height=1080,
            aspectRatio="1:1",
            cropStrategy="center"
        ),
        OutputPreset(
            id="wide",
            name="Wide (16:9)",
            width=1920,
            height=1080,
            aspectRatio="16:9",
            cropStrategy="center"
        )
    ]

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
