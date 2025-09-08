#!/usr/bin/env python3
"""
Demo script for Highlight Detector

This script demonstrates the complete workflow:
1. Upload a sample video
2. Create a detection session
3. Run highlight detection
4. Export clips

Usage:
    python scripts/demo.py [video_path] [mode]
    
Examples:
    python scripts/demo.py sample.mp4 sports
    python scripts/demo.py podcast.mp4 podcast
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages"))

from apps.server.core.config import get_settings
from apps.server.core.database import init_db, DatabaseManager
from apps.server.services.task_manager import TaskManager
from apps.server.models.schemas import DetectionMode


class HighlightDetectorDemo:
    """Demo class for Highlight Detector."""
    
    def __init__(self):
        self.settings = get_settings()
        self.task_manager = TaskManager()
    
    async def run_demo(self, video_path: str, mode: str = "sports"):
        """Run the complete demo workflow."""
        print(f"Highlight Detector Demo")
        print(f"📹 Video: {video_path}")
        print(f"Mode: {mode}")
        print("=" * 50)
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"ERROR: Video file not found: {video_path}")
            return False
        
        try:
            # Initialize database
            print("🔧 Initializing database...")
            await init_db()
            
            # Start task manager
            print("Starting task manager...")
            await self.task_manager.start()
            
            # Step 1: Upload and analyze video
            print("\n📤 Step 1: Uploading and analyzing video...")
            media_file = await self._upload_video(video_path)
            print(f"SUCCESS: Video uploaded: {media_file['name']}")
            print(f"   Duration: {media_file['duration']:.1f}s")
            print(f"   Resolution: {media_file['resolution']['width']}x{media_file['resolution']['height']}")
            
            # Step 2: Create session
            print("\nStep 2: Creating detection session...")
            session = await self._create_session(media_file, mode)
            print(f"SUCCESS: Session created: {session['id']}")
            
            # Step 3: Run detection
            print("\n🔍 Step 3: Running highlight detection...")
            events = await self._run_detection(session['id'])
            print(f"SUCCESS: Detection complete! Found {len(events)} events")
            
            # Display events
            if events:
                print("\nDetected Events:")
                for i, event in enumerate(events, 1):
                    print(f"   {i}. {event['label']} ({event['confidence']:.1%})")
                    print(f"      Time: {event['start_time']:.1f}s - {event['end_time']:.1f}s")
                    print(f"      Category: {event['category']}")
            
            # Step 4: Export clips (simulation)
            print("\nStep 4: Exporting clips...")
            if events:
                await self._export_clips(session['id'], events[:3])  # Export first 3 events
                print("SUCCESS: Clips exported successfully!")
            else:
                print("INFO: No events to export")
            
            print("\nSUCCESS: Demo completed successfully!")
            return True
            
        except Exception as e:
            print(f"\nERROR: Demo failed: {e}")
            return False
        
        finally:
            # Cleanup
            await self.task_manager.stop()
    
    async def _upload_video(self, video_path: str) -> dict:
        """Simulate video upload and analysis."""
        # This would normally use ffmpeg to probe the video
        # For demo purposes, we'll create mock data
        
        import uuid
        import time
        
        file_id = str(uuid.uuid4())
        file_name = os.path.basename(video_path)
        
        # Mock video analysis
        media_file = {
            'id': file_id,
            'name': file_name,
            'path': video_path,
            'size': os.path.getsize(video_path),
            'duration': 300.0,  # Mock 5 minutes
            'resolution': {'width': 1920, 'height': 1080},
            'fps': 30.0,
            'audio_format': 'aac',
            'video_format': 'h264',
            'created_at': time.time()
        }
        
        # Save to database
        async with DatabaseManager() as db:
            await db.create_media_file(media_file)
        
        return media_file
    
    async def _create_session(self, media_file: dict, mode: str) -> dict:
        """Create a detection session."""
        import uuid
        import time
        
        session_id = str(uuid.uuid4())
        
        session_data = {
            'id': session_id,
            'media_file_id': media_file['id'],
            'mode': mode,
            'status': 'idle',
            'progress': 0.0,
            'metadata_json': None
        }
        
        async with DatabaseManager() as db:
            session = await db.create_session(session_data)
        
        return {
            'id': session.id,
            'media_file': media_file,
            'mode': mode,
            'status': session.status,
            'progress': session.progress
        }
    
    async def _run_detection(self, session_id: str) -> list:
        """Run highlight detection."""
        # Submit detection task
        task_id = await self.task_manager.submit_detection_task(
            session_id=session_id,
            media_file_path="mock_path",
            mode="sports",
            config={}
        )
        
        # Wait for task completion
        while True:
            task = self.task_manager.get_task(task_id)
            if not task:
                break
            
            if task.status.value == "completed":
                break
            elif task.status.value == "failed":
                raise Exception(f"Detection failed: {task.error}")
            
            # Simulate progress updates
            print(f"   Progress: {task.progress:.1%} - {task.message}")
            await asyncio.sleep(1)
        
        # Generate mock events
        events = [
            {
                'id': 'event_1',
                'start_time': 45.2,
                'end_time': 52.8,
                'confidence': 0.87,
                'label': 'Home Run',
                'category': 'sports_action',
                'features': {
                    'audio_peak': 0.9,
                    'motion_magnitude': 0.8,
                    'voice_activity': 0.7
                },
                'evidence': {
                    'classifier_logits': [0.1, 0.9, 0.2],
                    'top_features': ['audio_peak', 'motion_magnitude']
                }
            },
            {
                'id': 'event_2',
                'start_time': 128.5,
                'end_time': 135.1,
                'confidence': 0.76,
                'label': 'Big Play',
                'category': 'sports_action',
                'features': {
                    'audio_peak': 0.8,
                    'motion_magnitude': 0.7,
                    'voice_activity': 0.6
                },
                'evidence': {
                    'classifier_logits': [0.2, 0.8, 0.1],
                    'top_features': ['audio_peak', 'voice_activity']
                }
            },
            {
                'id': 'event_3',
                'start_time': 245.3,
                'end_time': 251.9,
                'confidence': 0.82,
                'label': 'Key Moment',
                'category': 'sports_action',
                'features': {
                    'audio_peak': 0.85,
                    'motion_magnitude': 0.75,
                    'voice_activity': 0.8
                },
                'evidence': {
                    'classifier_logits': [0.15, 0.85, 0.2],
                    'top_features': ['motion_magnitude', 'audio_peak']
                }
            }
        ]
        
        # Save events to database
        async with DatabaseManager() as db:
            for event in events:
                await db.create_highlight_event({
                    'id': event['id'],
                    'session_id': session_id,
                    'start_time': event['start_time'],
                    'end_time': event['end_time'],
                    'confidence': event['confidence'],
                    'label': event['label'],
                    'category': event['category'],
                    'features': event['features'],
                    'evidence': event['evidence']
                })
        
        return events
    
    async def _export_clips(self, session_id: str, events: list):
        """Export clips for selected events."""
        import uuid
        
        # Create render job
        job_id = await self.task_manager.submit_render_job(
            session_id=session_id,
            event_ids=[event['id'] for event in events],
            preset={
                'id': 'vertical',
                'name': 'Vertical (9:16)',
                'width': 1080,
                'height': 1920,
                'aspect_ratio': '9:16',
                'crop_strategy': 'motion_centroid'
            },
            output_path=f"output/demo_clips_{session_id}.mp4"
        )
        
        # Wait for render completion
        while True:
            task = self.task_manager.get_task(job_id)
            if not task:
                break
            
            if task.status.value == "completed":
                break
            elif task.status.value == "failed":
                raise Exception(f"Render failed: {task.error}")
            
            print(f"   Render progress: {task.progress:.1%} - {task.message}")
            await asyncio.sleep(1)


async def main():
    """Main demo function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/demo.py <video_path> [mode]")
        print("Modes: sports, podcast")
        sys.exit(1)
    
    video_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "sports"
    
    if mode not in ["sports", "podcast"]:
        print("ERROR: Invalid mode. Use 'sports' or 'podcast'")
        sys.exit(1)
    
    demo = HighlightDetectorDemo()
    success = await demo.run_demo(video_path, mode)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
