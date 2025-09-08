"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class DetectionMode(str, Enum):
    """Detection mode enumeration."""
    SPORTS = "sports"
    PODCAST = "podcast"


class SessionStatus(str, Enum):
    """Session status enumeration."""
    IDLE = "idle"
    DETECTING = "detecting"
    COMPLETED = "completed"
    ERROR = "error"


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RENDERING = "rendering"
    COMPLETED = "completed"
    ERROR = "error"


class CropStrategy(str, Enum):
    """Crop strategy enumeration."""
    CENTER = "center"
    MOTION_CENTROID = "motion_centroid"
    FACE_TRACKING = "face_tracking"


# Media File Models
class MediaFileResponse(BaseModel):
    """Media file response model."""
    id: str
    name: str
    path: str
    size: int
    duration: float
    resolution: Dict[str, int] = Field(..., description="Width and height")
    fps: float
    audio_format: str
    video_format: str
    created_at: datetime

    class Config:
        from_attributes = True


# Session Models
class SessionCreate(BaseModel):
    """Session creation request model."""
    media_file: MediaFileResponse
    mode: DetectionMode


class SessionResponse(BaseModel):
    """Session response model."""
    id: str
    media_file: MediaFileResponse
    mode: DetectionMode
    status: SessionStatus
    progress: float
    events: List["HighlightEvent"] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Highlight Event Models
class EventFeatures(BaseModel):
    """Event features model."""
    audio_peak: float
    motion_magnitude: float
    voice_activity: float
    prosody: float
    shot_boundary: float
    scoreboard_change: Optional[float] = None
    replay_cue: Optional[float] = None
    laughter: Optional[float] = None
    applause: Optional[float] = None
    excitement: Optional[float] = None
    topic_shift: Optional[float] = None


class EventEvidence(BaseModel):
    """Event evidence model."""
    audio_chart: Optional[str] = None
    motion_chart: Optional[str] = None
    scoreboard_before: Optional[str] = None
    scoreboard_after: Optional[str] = None
    classifier_logits: List[float]
    top_features: List[str]


class HighlightEvent(BaseModel):
    """Highlight event model."""
    id: str
    start_time: float
    end_time: float
    confidence: float
    label: str
    category: str
    features: EventFeatures
    evidence: EventEvidence
    created_at: datetime

    class Config:
        from_attributes = True


# Output Preset Models
class OutputPreset(BaseModel):
    """Output preset model."""
    id: str
    name: str
    width: int
    height: int
    aspect_ratio: str
    crop_strategy: CropStrategy


# Render Job Models
class RenderJobCreate(BaseModel):
    """Render job creation request model."""
    session_id: str
    event_ids: List[str]
    preset: OutputPreset


class RenderJobResponse(BaseModel):
    """Render job response model."""
    id: str
    session_id: str
    event_ids: List[str]
    preset: OutputPreset
    status: JobStatus
    progress: float
    output_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Detection Progress Models
class DetectionProgress(BaseModel):
    """Detection progress model."""
    stage: str
    progress: float
    message: str
    estimated_time_remaining: Optional[int] = None


# Upload Models
class UploadResponse(BaseModel):
    """File upload response model."""
    success: bool
    media_file: Optional[MediaFileResponse] = None
    error: Optional[str] = None


# Health Check Models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: datetime
    database_connected: bool
    active_connections: int


# Error Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime


# WebSocket Message Models
class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime


# Update forward references
SessionResponse.model_rebuild()
HighlightEvent.model_rebuild()
