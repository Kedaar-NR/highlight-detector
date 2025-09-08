"""
Session management API routes.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db, DatabaseManager
from ...core.websocket import ConnectionManager
from ...services.task_manager import TaskManager
from ...models.schemas import (
    SessionCreate, SessionResponse, HighlightEvent, 
    DetectionProgress, MediaFileResponse
)

router = APIRouter()

# Global instances
websocket_manager = ConnectionManager()
task_manager = TaskManager()


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a new detection session."""
    session_id = str(uuid.uuid4())
    
    async with DatabaseManager() as db_manager:
        # Create session record
        session_data_dict = {
            'id': session_id,
            'media_file_id': session_data.media_file.id,
            'mode': session_data.mode.value,
            'status': 'idle',
            'progress': 0.0,
            'metadata_json': None
        }
        
        session = await db_manager.create_session(session_data_dict)
        
        # Convert to response model
        session_response = SessionResponse(
            id=session.id,
            media_file=session_data.media_file,
            mode=session_data.mode,
            status=session.status,
            progress=session.progress,
            events=[],
            created_at=session.created_at,
            updated_at=session.updated_at
        )
    
    return session_response


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get session by ID."""
    async with DatabaseManager() as db_manager:
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get media file
        media_file = await db_manager.get_media_file(session.media_file_id)
        if not media_file:
            raise HTTPException(status_code=404, detail="Media file not found")
        
        # Get events
        events = await db_manager.get_events_for_session(session_id)
        
        # Convert to response models
        media_file_response = MediaFileResponse(
            id=media_file.id,
            name=media_file.name,
            path=media_file.path,
            size=media_file.size,
            duration=media_file.duration,
            resolution={
                'width': media_file.resolution_width,
                'height': media_file.resolution_height
            },
            fps=media_file.fps,
            audio_format=media_file.audio_format,
            video_format=media_file.video_format,
            created_at=media_file.created_at
        )
        
        highlight_events = []
        for event in events:
            # Parse JSON fields
            import json
            features = json.loads(event.features_json) if event.features_json else {}
            evidence = json.loads(event.evidence_json) if event.evidence_json else {}
            
            highlight_event = HighlightEvent(
                id=event.id,
                start_time=event.start_time,
                end_time=event.end_time,
                confidence=event.confidence,
                label=event.label,
                category=event.category,
                features=features,
                evidence=evidence,
                created_at=event.created_at
            )
            highlight_events.append(highlight_event)
        
        session_response = SessionResponse(
            id=session.id,
            media_file=media_file_response,
            mode=session.mode,
            status=session.status,
            progress=session.progress,
            events=highlight_events,
            created_at=session.created_at,
            updated_at=session.updated_at
        )
    
    return session_response


@router.post("/sessions/{session_id}/detect")
async def start_detection(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Start highlight detection for a session."""
    async with DatabaseManager() as db_manager:
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status == 'detecting':
            raise HTTPException(
                status_code=400, 
                detail="Detection already in progress"
            )
        
        # Update session status
        await db_manager.update_session(session_id, {
            'status': 'detecting',
            'progress': 0.0
        })
    
    # Start detection task in background
    background_tasks.add_task(
        run_detection_task,
        session_id,
        websocket_manager
    )
    
    return {"message": "Detection started"}


async def run_detection_task(session_id: str, ws_manager: ConnectionManager):
    """Background task to run highlight detection."""
    try:
        # This would integrate with the actual detection pipeline
        # For now, we'll simulate the process
        
        stages = [
            ("ingest", "Ingesting media file..."),
            ("audio_features", "Extracting audio features..."),
            ("vision_features", "Analyzing visual content..."),
            ("fusion", "Combining features..."),
            ("classification", "Identifying highlights..."),
            ("complete", "Detection complete!")
        ]
        
        for i, (stage, message) in enumerate(stages):
            progress = (i + 1) / len(stages)
            
            # Send progress update
            await ws_manager.send_detection_progress(session_id, {
                "stage": stage,
                "progress": progress,
                "message": message,
                "estimated_time_remaining": max(0, (len(stages) - i - 1) * 10)
            })
            
            # Simulate processing time
            import asyncio
            await asyncio.sleep(2)
        
        # Send completion
        await ws_manager.send_detection_complete(session_id, 0)
        
    except Exception as e:
        # Send error
        await ws_manager.send_error(session_id, str(e))


@router.get("/sessions/{session_id}/events", response_model=List[HighlightEvent])
async def get_session_events(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all events for a session."""
    async with DatabaseManager() as db_manager:
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        events = await db_manager.get_events_for_session(session_id)
        
        highlight_events = []
        for event in events:
            # Parse JSON fields
            import json
            features = json.loads(event.features_json) if event.features_json else {}
            evidence = json.loads(event.evidence_json) if event.evidence_json else {}
            
            highlight_event = HighlightEvent(
                id=event.id,
                start_time=event.start_time,
                end_time=event.end_time,
                confidence=event.confidence,
                label=event.label,
                category=event.category,
                features=features,
                evidence=evidence,
                created_at=event.created_at
            )
            highlight_events.append(highlight_event)
    
    return highlight_events


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a session and all its data."""
    async with DatabaseManager() as db_manager:
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete all events for this session
        events = await db_manager.get_events_for_session(session_id)
        for event in events:
            await db_manager.session.delete(event)
        
        # Delete session
        await db_manager.session.delete(session)
        await db_manager.session.commit()
    
    return {"message": "Session deleted successfully"}
