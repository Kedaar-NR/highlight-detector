"""
Database configuration and models for the Highlight Detector server.
"""

import asyncio
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import json

from .config import get_settings

Base = declarative_base()

# Database models
class SessionModel(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True)
    media_file_id = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    status = Column(String, default="idle")
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(Text)  # Store additional metadata as JSON

class MediaFileModel(Base):
    __tablename__ = "media_files"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    resolution_width = Column(Integer, nullable=False)
    resolution_height = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    audio_format = Column(String, nullable=False)
    video_format = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class HighlightEventModel(Base):
    __tablename__ = "highlight_events"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    category = Column(String, nullable=False)
    features_json = Column(Text)  # Store features as JSON
    evidence_json = Column(Text)  # Store evidence as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

class RenderJobModel(Base):
    __tablename__ = "render_jobs"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False)
    event_ids_json = Column(Text, nullable=False)  # Store event IDs as JSON array
    preset_json = Column(Text, nullable=False)  # Store preset as JSON
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    output_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Database engine and session
engine = None
async_session = None

async def init_db():
    """Initialize database connection."""
    global engine, async_session
    
    settings = get_settings()
    
    # Create SQLite database file path
    db_path = "highlight_detector.db"
    
    # Create async engine for SQLite
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=settings.DEBUG,
        future=True
    )
    
    # Create async session factory
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncSession:
    """Get database session."""
    if async_session is None:
        await init_db()
    
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Database operations
class DatabaseManager:
    """Database operations manager."""
    
    def __init__(self):
        self.session: Optional[AsyncSession] = None
    
    async def __aenter__(self):
        if async_session is None:
            await init_db()
        self.session = async_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def create_session(self, session_data: dict) -> SessionModel:
        """Create a new session."""
        session = SessionModel(**session_data)
        self.session.add(session)
        await self.session.commit()
        await self.session.refresh(session)
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionModel]:
        """Get session by ID."""
        return await self.session.get(SessionModel, session_id)
    
    async def update_session(self, session_id: str, updates: dict) -> Optional[SessionModel]:
        """Update session."""
        session = await self.get_session(session_id)
        if session:
            for key, value in updates.items():
                setattr(session, key, value)
            await self.session.commit()
            await self.session.refresh(session)
        return session
    
    async def create_media_file(self, media_data: dict) -> MediaFileModel:
        """Create a new media file record."""
        media_file = MediaFileModel(**media_data)
        self.session.add(media_file)
        await self.session.commit()
        await self.session.refresh(media_file)
        return media_file
    
    async def get_media_file(self, file_id: str) -> Optional[MediaFileModel]:
        """Get media file by ID."""
        return await self.session.get(MediaFileModel, file_id)
    
    async def create_highlight_event(self, event_data: dict) -> HighlightEventModel:
        """Create a new highlight event."""
        # Convert features and evidence to JSON strings
        if 'features' in event_data:
            event_data['features_json'] = json.dumps(event_data.pop('features'))
        if 'evidence' in event_data:
            event_data['evidence_json'] = json.dumps(event_data.pop('evidence'))
        
        event = HighlightEventModel(**event_data)
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event
    
    async def get_events_for_session(self, session_id: str) -> List[HighlightEventModel]:
        """Get all events for a session."""
        result = await self.session.execute(
            "SELECT * FROM highlight_events WHERE session_id = :session_id ORDER BY start_time",
            {"session_id": session_id}
        )
        return result.fetchall()
    
    async def create_render_job(self, job_data: dict) -> RenderJobModel:
        """Create a new render job."""
        # Convert event_ids and preset to JSON strings
        if 'event_ids' in job_data:
            job_data['event_ids_json'] = json.dumps(job_data.pop('event_ids'))
        if 'preset' in job_data:
            job_data['preset_json'] = json.dumps(job_data.pop('preset'))
        
        job = RenderJobModel(**job_data)
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job
    
    async def get_render_job(self, job_id: str) -> Optional[RenderJobModel]:
        """Get render job by ID."""
        return await self.session.get(RenderJobModel, job_id)
    
    async def update_render_job(self, job_id: str, updates: dict) -> Optional[RenderJobModel]:
        """Update render job."""
        job = await self.get_render_job(job_id)
        if job:
            for key, value in updates.items():
                setattr(job, key, value)
            await self.session.commit()
            await self.session.refresh(job)
        return job
