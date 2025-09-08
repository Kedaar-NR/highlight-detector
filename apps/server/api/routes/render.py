"""
Render job API routes.
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
    RenderJobCreate, RenderJobResponse, OutputPreset
)

router = APIRouter()

# Global instances
websocket_manager = ConnectionManager()
task_manager = TaskManager()


@router.post("/render", response_model=RenderJobResponse)
async def create_render_job(
    job_data: RenderJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a new render job."""
    job_id = str(uuid.uuid4())
    
    async with DatabaseManager() as db_manager:
        # Verify session exists
        session = await db_manager.get_session(job_data.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create render job record
        job_data_dict = {
            'id': job_id,
            'session_id': job_data.session_id,
            'event_ids': job_data.event_ids,
            'preset': job_data.preset.dict(),
            'status': 'pending',
            'progress': 0.0,
            'output_path': None
        }
        
        job = await db_manager.create_render_job(job_data_dict)
        
        # Convert to response model
        job_response = RenderJobResponse(
            id=job.id,
            session_id=job.session_id,
            event_ids=job_data.event_ids,
            preset=job_data.preset,
            status=job.status,
            progress=job.progress,
            output_path=job.output_path,
            created_at=job.created_at,
            updated_at=job.updated_at
        )
    
    # Start render task in background
    background_tasks.add_task(
        run_render_task,
        job_id,
        job_data,
        websocket_manager
    )
    
    return job_response


@router.get("/render/{job_id}", response_model=RenderJobResponse)
async def get_render_job(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get render job by ID."""
    async with DatabaseManager() as db_manager:
        job = await db_manager.get_render_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Render job not found")
        
        # Parse JSON fields
        import json
        event_ids = json.loads(job.event_ids_json) if job.event_ids_json else []
        preset_data = json.loads(job.preset_json) if job.preset_json else {}
        
        preset = OutputPreset(**preset_data)
        
        job_response = RenderJobResponse(
            id=job.id,
            session_id=job.session_id,
            event_ids=event_ids,
            preset=preset,
            status=job.status,
            progress=job.progress,
            output_path=job.output_path,
            created_at=job.created_at,
            updated_at=job.updated_at
        )
    
    return job_response


async def run_render_task(
    job_id: str,
    job_data: RenderJobCreate,
    ws_manager: ConnectionManager
):
    """Background task to run video rendering."""
    try:
        async with DatabaseManager() as db_manager:
            # Update job status to rendering
            await db_manager.update_render_job(job_id, {
                'status': 'rendering',
                'progress': 0.0
            })
            
            # Send progress update
            await ws_manager.send_render_progress(job_id, {
                "stage": "initializing",
                "progress": 0.0,
                "message": "Preparing render job..."
            })
            
            # Simulate rendering process
            stages = [
                ("preparing", "Preparing video segments..."),
                ("processing", "Processing video frames..."),
                ("encoding", "Encoding final video..."),
                ("finalizing", "Finalizing output..."),
                ("complete", "Render complete!")
            ]
            
            for i, (stage, message) in enumerate(stages):
                progress = (i + 1) / len(stages)
                
                # Update database
                await db_manager.update_render_job(job_id, {
                    'progress': progress
                })
                
                # Send progress update
                await ws_manager.send_render_progress(job_id, {
                    "stage": stage,
                    "progress": progress,
                    "message": message
                })
                
                # Simulate processing time
                import asyncio
                await asyncio.sleep(3)
            
            # Generate output path
            from ...core.config import get_settings
            settings = get_settings()
            output_path = f"{settings.OUTPUT_DIR}/render_{job_id}.mp4"
            
            # Update job as completed
            await db_manager.update_render_job(job_id, {
                'status': 'completed',
                'progress': 1.0,
                'output_path': output_path
            })
            
            # Send completion
            await ws_manager.send_render_complete(job_id, output_path)
            
    except Exception as e:
        # Update job as failed
        async with DatabaseManager() as db_manager:
            await db_manager.update_render_job(job_id, {
                'status': 'error',
                'progress': 0.0
            })
        
        # Send error
        await ws_manager.send_error(job_id, f"Render failed: {str(e)}")


@router.get("/render", response_model=List[RenderJobResponse])
async def list_render_jobs(
    session_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List render jobs, optionally filtered by session."""
    async with DatabaseManager() as db_manager:
        # This would need a proper query implementation
        # For now, return empty list
        return []
