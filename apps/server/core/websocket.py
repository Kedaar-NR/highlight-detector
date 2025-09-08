"""
WebSocket connection manager for real-time updates.
"""

import json
import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket
from datetime import datetime
import uuid

from .config import get_settings


class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "id": str(uuid.uuid4()),
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow()
        }
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSockets."""
        if not self.active_connections:
            return
        
        # Create a list of tasks for concurrent sending
        tasks = []
        disconnected = []
        
        for connection in self.active_connections:
            try:
                task = asyncio.create_task(connection.send_text(message))
                tasks.append((connection, task))
            except Exception as e:
                print(f"Error creating broadcast task: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
        
        # Wait for all tasks to complete
        for connection, task in tasks:
            try:
                await task
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")
                self.disconnect(connection)
    
    async def send_detection_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """Send detection progress update."""
        message = {
            "type": "progress",
            "session_id": session_id,
            "data": progress_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(json.dumps(message))
    
    async def send_new_event(self, session_id: str, event_data: Dict[str, Any]):
        """Send new highlight event."""
        message = {
            "type": "event",
            "session_id": session_id,
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(json.dumps(message))
    
    async def send_detection_complete(self, session_id: str, total_events: int):
        """Send detection completion notification."""
        message = {
            "type": "complete",
            "session_id": session_id,
            "data": {"total_events": total_events},
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(json.dumps(message))
    
    async def send_error(self, session_id: str, error_message: str):
        """Send error notification."""
        message = {
            "type": "error",
            "session_id": session_id,
            "data": {"message": error_message},
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(json.dumps(message))
    
    async def send_render_progress(self, job_id: str, progress_data: Dict[str, Any]):
        """Send render progress update."""
        message = {
            "type": "render_progress",
            "job_id": job_id,
            "data": progress_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(json.dumps(message))
    
    async def send_render_complete(self, job_id: str, output_path: str):
        """Send render completion notification."""
        message = {
            "type": "render_complete",
            "job_id": job_id,
            "data": {"output_path": output_path},
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(json.dumps(message))
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about all active connections."""
        info = []
        for websocket, metadata in self.connection_metadata.items():
            info.append({
                "id": metadata["id"],
                "connected_at": metadata["connected_at"].isoformat(),
                "last_ping": metadata["last_ping"].isoformat()
            })
        return info
    
    async def ping_all(self):
        """Send ping to all connections to check if they're alive."""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(json.dumps(ping_message))
        
        # Update last ping time
        for metadata in self.connection_metadata.values():
            metadata["last_ping"] = datetime.utcnow()
