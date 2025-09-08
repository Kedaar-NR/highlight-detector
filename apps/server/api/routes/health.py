"""
Health check API routes.
"""

from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...core.websocket import ConnectionManager
from ...core.config import get_settings
from ...models.schemas import HealthResponse

router = APIRouter()

# Global websocket manager instance
websocket_manager = ConnectionManager()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint."""
    settings = get_settings()
    
    # Check database connection
    database_connected = True
    try:
        # Simple query to test database connection
        await db.execute("SELECT 1")
    except Exception:
        database_connected = False
    
    return HealthResponse(
        status="healthy" if database_connected else "unhealthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        database_connected=database_connected,
        active_connections=websocket_manager.get_connection_count()
    )
