"""
File upload API routes.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import ffmpeg

from ...core.database import get_db, DatabaseManager
from ...core.config import get_settings
from ...models.schemas import UploadResponse, MediaFileResponse

router = APIRouter()


async def probe_media_file(file_path: str) -> dict:
    """Probe media file to extract metadata."""
    try:
        probe = ffmpeg.probe(file_path)
        
        # Find video stream
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )
        
        # Find audio stream
        audio_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
            None
        )
        
        if not video_stream:
            raise ValueError("No video stream found in file")
        
        # Extract metadata
        duration = float(probe['format']['duration'])
        size = int(probe['format']['size'])
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = eval(video_stream['r_frame_rate'])  # Convert fraction to float
        
        video_format = video_stream['codec_name']
        audio_format = audio_stream['codec_name'] if audio_stream else 'none'
        
        return {
            'duration': duration,
            'size': size,
            'width': width,
            'height': height,
            'fps': fps,
            'video_format': video_format,
            'audio_format': audio_format
        }
        
    except Exception as e:
        raise ValueError(f"Failed to probe media file: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a media file."""
    settings = get_settings()
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail="Only video files are supported"
        )
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    max_size = settings.MAX_FILE_SIZE_GB * 1024 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_GB}GB"
        )
    
    try:
        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.TEMP_DIR) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = upload_dir / filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Probe media file to extract metadata
        metadata = await probe_media_file(str(file_path))
        
        # Create media file record in database
        async with DatabaseManager() as db_manager:
            media_file_data = {
                'id': file_id,
                'name': file.filename,
                'path': str(file_path),
                'size': file_size,
                'duration': metadata['duration'],
                'resolution_width': metadata['width'],
                'resolution_height': metadata['height'],
                'fps': metadata['fps'],
                'audio_format': metadata['audio_format'],
                'video_format': metadata['video_format']
            }
            
            media_file = await db_manager.create_media_file(media_file_data)
            
            # Convert to response model
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
        
        return UploadResponse(
            success=True,
            media_file=media_file_response
        )
        
    except Exception as e:
        # Clean up uploaded file if it exists
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded file: {str(e)}"
        )


@router.delete("/upload/{file_id}")
async def delete_uploaded_file(
    file_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete an uploaded file and its database record."""
    settings = get_settings()
    
    async with DatabaseManager() as db_manager:
        # Get media file record
        media_file = await db_manager.get_media_file(file_id)
        if not media_file:
            raise HTTPException(
                status_code=404,
                detail="Media file not found"
            )
        
        # Delete file from filesystem
        file_path = Path(media_file.path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete database record
        await db_manager.session.delete(media_file)
        await db_manager.session.commit()
    
    return {"success": True, "message": "File deleted successfully"}
