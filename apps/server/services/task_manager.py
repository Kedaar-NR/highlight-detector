"""
Task manager for handling long-running detection and render jobs.
"""

import asyncio
import uuid
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum
import logging

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Task type enumeration."""
    DETECTION = "detection"
    RENDER = "render"


class Task:
    """Represents a background task."""
    
    def __init__(
        self,
        task_id: str,
        task_type: TaskType,
        session_id: str,
        payload: Dict[str, Any]
    ):
        self.id = task_id
        self.type = task_type
        self.session_id = session_id
        self.payload = payload
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.message = ""
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Any] = None
        self._cancelled = False
        self._task: Optional[asyncio.Task] = None
    
    def cancel(self):
        """Cancel the task."""
        self._cancelled = True
        if self._task and not self._task.done():
            self._task.cancel()
        self.status = TaskStatus.CANCELLED
    
    def is_cancelled(self) -> bool:
        """Check if task is cancelled."""
        return self._cancelled
    
    def update_progress(self, progress: float, message: str = ""):
        """Update task progress."""
        self.progress = progress
        self.message = message
    
    def set_error(self, error: str):
        """Set task error."""
        self.error = error
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
    
    def set_completed(self, result: Any = None):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.progress = 1.0
        self.result = result
        self.completed_at = datetime.utcnow()


class TaskManager:
    """Manages background tasks for detection and rendering."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._max_concurrent_tasks = 3
        self._semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
    
    async def start(self):
        """Start the task manager."""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Task manager started")
    
    async def stop(self):
        """Stop the task manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all pending tasks
        for task in self.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.cancel()
        
        # Wait for worker to finish
        if self._worker_task:
            await self._worker_task
        
        logger.info("Task manager stopped")
    
    async def submit_detection_task(
        self,
        session_id: str,
        media_file_path: str,
        mode: str,
        config: Dict[str, Any]
    ) -> str:
        """Submit a detection task."""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            task_type=TaskType.DETECTION,
            session_id=session_id,
            payload={
                "media_file_path": media_file_path,
                "mode": mode,
                "config": config
            }
        )
        
        self.tasks[task_id] = task
        logger.info(f"Submitted detection task {task_id} for session {session_id}")
        
        return task_id
    
    async def submit_render_task(
        self,
        session_id: str,
        event_ids: list,
        preset: Dict[str, Any],
        output_path: str
    ) -> str:
        """Submit a render task."""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            task_type=TaskType.RENDER,
            session_id=session_id,
            payload={
                "event_ids": event_ids,
                "preset": preset,
                "output_path": output_path
            }
        )
        
        self.tasks[task_id] = task
        logger.info(f"Submitted render task {task_id} for session {session_id}")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self.tasks.get(task_id)
        if task:
            task.cancel()
            return True
        return False
    
    def list_tasks(self, session_id: Optional[str] = None) -> list:
        """List tasks, optionally filtered by session."""
        tasks = list(self.tasks.values())
        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]
        return tasks
    
    async def _worker_loop(self):
        """Main worker loop for processing tasks."""
        while self._running:
            try:
                # Find pending tasks
                pending_tasks = [
                    task for task in self.tasks.values()
                    if task.status == TaskStatus.PENDING
                ]
                
                if pending_tasks:
                    # Process tasks concurrently up to the limit
                    tasks_to_process = pending_tasks[:self._max_concurrent_tasks]
                    
                    for task in tasks_to_process:
                        asyncio.create_task(self._process_task(task))
                
                # Wait before checking again
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_task(self, task: Task):
        """Process a single task."""
        async with self._semaphore:
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                
                if task.type == TaskType.DETECTION:
                    await self._run_detection_task(task)
                elif task.type == TaskType.RENDER:
                    await self._run_render_task(task)
                
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                logger.info(f"Task {task.id} was cancelled")
            except Exception as e:
                task.set_error(str(e))
                logger.error(f"Task {task.id} failed: {e}")
    
    async def _run_detection_task(self, task: Task):
        """Run a detection task."""
        # This would integrate with the actual detection pipeline
        # For now, we'll simulate the process
        
        stages = [
            ("ingest", "Ingesting media file...", 0.1),
            ("audio_features", "Extracting audio features...", 0.3),
            ("vision_features", "Analyzing visual content...", 0.5),
            ("fusion", "Combining features...", 0.7),
            ("classification", "Identifying highlights...", 0.9),
            ("complete", "Detection complete!", 1.0)
        ]
        
        for stage, message, progress in stages:
            if task.is_cancelled():
                return
            
            task.update_progress(progress, message)
            await asyncio.sleep(2)  # Simulate processing time
        
        task.set_completed({"events": []})
        logger.info(f"Detection task {task.id} completed")
    
    async def _run_render_task(self, task: Task):
        """Run a render task."""
        # This would integrate with the actual rendering pipeline
        # For now, we'll simulate the process
        
        stages = [
            ("preparing", "Preparing video segments...", 0.2),
            ("processing", "Processing video frames...", 0.5),
            ("encoding", "Encoding final video...", 0.8),
            ("finalizing", "Finalizing output...", 1.0)
        ]
        
        for stage, message, progress in stages:
            if task.is_cancelled():
                return
            
            task.update_progress(progress, message)
            await asyncio.sleep(3)  # Simulate processing time
        
        output_path = task.payload.get("output_path", f"output_{task.id}.mp4")
        task.set_completed({"output_path": output_path})
        logger.info(f"Render task {task.id} completed")
