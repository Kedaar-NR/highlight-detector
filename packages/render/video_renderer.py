"""
Video rendering system for highlight clips.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from dataclasses import dataclass
import ffmpeg
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RenderPreset:
    """Video rendering preset configuration."""
    id: str
    name: str
    width: int
    height: int
    aspect_ratio: str
    crop_strategy: str
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    video_bitrate: str = "2M"
    audio_bitrate: str = "128k"
    fps: int = 30


@dataclass
class RenderJob:
    """Video rendering job configuration."""
    input_path: str
    output_path: str
    start_time: float
    end_time: float
    preset: RenderPreset
    fade_duration: float = 0.15
    add_watermark: bool = False
    watermark_path: Optional[str] = None
    add_captions: bool = False
    captions_path: Optional[str] = None


class VideoRenderer:
    """Main video rendering class."""
    
    def __init__(self, temp_dir: str = "./temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Default presets
        self.presets = {
            "vertical": RenderPreset(
                id="vertical",
                name="Vertical (9:16)",
                width=1080,
                height=1920,
                aspect_ratio="9:16",
                crop_strategy="motion_centroid"
            ),
            "square": RenderPreset(
                id="square",
                name="Square (1:1)",
                width=1080,
                height=1080,
                aspect_ratio="1:1",
                crop_strategy="center"
            ),
            "wide": RenderPreset(
                id="wide",
                name="Wide (16:9)",
                width=1920,
                height=1080,
                aspect_ratio="16:9",
                crop_strategy="center"
            )
        }
    
    def render_clip(self, job: RenderJob) -> Dict[str, Any]:
        """Render a single highlight clip."""
        try:
            logger.info(f"Rendering clip: {job.start_time}s - {job.end_time}s")
            
            # Create temporary files
            temp_video = self.temp_dir / f"temp_{job.preset.id}.mp4"
            temp_audio = self.temp_dir / f"temp_{job.preset.id}.wav"
            
            # Step 1: Extract and trim video
            self._extract_video_segment(job, temp_video)
            
            # Step 2: Extract and trim audio
            self._extract_audio_segment(job, temp_audio)
            
            # Step 3: Apply video transformations
            processed_video = self._process_video(temp_video, job)
            
            # Step 4: Combine video and audio
            self._combine_video_audio(processed_video, temp_audio, job.output_path, job)
            
            # Step 5: Add effects (watermark, captions)
            if job.add_watermark or job.add_captions:
                self._add_effects(job.output_path, job)
            
            # Cleanup temporary files
            self._cleanup([temp_video, temp_audio, processed_video])
            
            # Generate metadata
            metadata = self._generate_metadata(job)
            
            logger.info(f"Successfully rendered clip: {job.output_path}")
            return {
                "success": True,
                "output_path": job.output_path,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to render clip: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def render_multiple_clips(
        self,
        input_path: str,
        events: List[Dict[str, Any]],
        preset: RenderPreset,
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Render multiple highlight clips."""
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, event in enumerate(events):
            if progress_callback:
                progress_callback(i / len(events), f"Rendering clip {i+1}/{len(events)}")
            
            # Create render job
            job = RenderJob(
                input_path=input_path,
                output_path=str(output_path / f"highlight_{i+1:03d}.mp4"),
                start_time=event['start_time'],
                end_time=event['end_time'],
                preset=preset
            )
            
            # Render clip
            result = self.render_clip(job)
            result['event_id'] = event.get('id', f'event_{i}')
            results.append(result)
        
        if progress_callback:
            progress_callback(1.0, "All clips rendered")
        
        return results
    
    def _extract_video_segment(self, job: RenderJob, output_path: Path):
        """Extract video segment using ffmpeg."""
        duration = job.end_time - job.start_time
        
        try:
            (
                ffmpeg
                .input(job.input_path, ss=job.start_time, t=duration)
                .output(str(output_path), vcodec='libx264', acodec='copy')
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise Exception(f"Failed to extract video segment: {e}")
    
    def _extract_audio_segment(self, job: RenderJob, output_path: Path):
        """Extract audio segment using ffmpeg."""
        duration = job.end_time - job.start_time
        
        try:
            (
                ffmpeg
                .input(job.input_path, ss=job.start_time, t=duration)
                .output(str(output_path), acodec='pcm_s16le', ar=44100)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise Exception(f"Failed to extract audio segment: {e}")
    
    def _process_video(self, input_path: Path, job: RenderJob) -> Path:
        """Process video with transformations (crop, resize, etc.)."""
        output_path = self.temp_dir / f"processed_{job.preset.id}.mp4"
        
        # Get input video properties
        probe = ffmpeg.probe(str(input_path))
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        input_width = int(video_stream['width'])
        input_height = int(video_stream['height'])
        
        # Calculate crop parameters
        crop_params = self._calculate_crop_params(
            input_width, input_height, job.preset
        )
        
        # Build ffmpeg filter
        filters = []
        
        # Crop if needed
        if crop_params:
            filters.append(f"crop={crop_params['width']}:{crop_params['height']}:{crop_params['x']}:{crop_params['y']}")
        
        # Scale to target resolution
        filters.append(f"scale={job.preset.width}:{job.preset.height}")
        
        # Add fade effects
        if job.fade_duration > 0:
            duration = job.end_time - job.start_time
            filters.append(f"fade=t=in:st=0:d={job.fade_duration}")
            filters.append(f"fade=t=out:st={duration-job.fade_duration}:d={job.fade_duration}")
        
        filter_string = ",".join(filters)
        
        try:
            (
                ffmpeg
                .input(str(input_path))
                .video.filter(filter_string)
                .output(str(output_path), vcodec=job.preset.video_codec, vb=job.preset.video_bitrate)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise Exception(f"Failed to process video: {e}")
        
        return output_path
    
    def _calculate_crop_params(
        self,
        input_width: int,
        input_height: int,
        preset: RenderPreset
    ) -> Optional[Dict[str, int]]:
        """Calculate crop parameters for the given preset."""
        input_aspect = input_width / input_height
        target_aspect = preset.width / preset.height
        
        if abs(input_aspect - target_aspect) < 0.01:
            # No cropping needed
            return None
        
        if preset.crop_strategy == "center":
            return self._center_crop(input_width, input_height, preset)
        elif preset.crop_strategy == "motion_centroid":
            return self._motion_centroid_crop(input_width, input_height, preset)
        else:
            return self._center_crop(input_width, input_height, preset)
    
    def _center_crop(
        self,
        input_width: int,
        input_height: int,
        preset: RenderPreset
    ) -> Dict[str, int]:
        """Calculate center crop parameters."""
        target_aspect = preset.width / preset.height
        
        if input_width / input_height > target_aspect:
            # Crop width
            new_width = int(input_height * target_aspect)
            x = (input_width - new_width) // 2
            return {
                'width': new_width,
                'height': input_height,
                'x': x,
                'y': 0
            }
        else:
            # Crop height
            new_height = int(input_width / target_aspect)
            y = (input_height - new_height) // 2
            return {
                'width': input_width,
                'height': new_height,
                'x': 0,
                'y': y
            }
    
    def _motion_centroid_crop(
        self,
        input_width: int,
        input_height: int,
        preset: RenderPreset
    ) -> Dict[str, int]:
        """Calculate motion centroid crop parameters."""
        # For now, fall back to center crop
        # In a full implementation, this would analyze motion in the video
        # to determine the best crop region
        return self._center_crop(input_width, input_height, preset)
    
    def _combine_video_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: str,
        job: RenderJob
    ):
        """Combine processed video and audio."""
        try:
            video_input = ffmpeg.input(str(video_path))
            audio_input = ffmpeg.input(str(audio_path))
            
            # Add audio fade effects
            audio_filters = []
            if job.fade_duration > 0:
                duration = job.end_time - job.start_time
                audio_filters.append(f"afade=t=in:st=0:d={job.fade_duration}")
                audio_filters.append(f"afade=t=out:st={duration-job.fade_duration}:d={job.fade_duration}")
            
            if audio_filters:
                audio_input = audio_input.filter(';'.join(audio_filters))
            
            (
                ffmpeg
                .output(
                    video_input,
                    audio_input,
                    output_path,
                    vcodec=job.preset.video_codec,
                    acodec=job.preset.audio_codec,
                    vb=job.preset.video_bitrate,
                    ab=job.preset.audio_bitrate,
                    r=job.preset.fps
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise Exception(f"Failed to combine video and audio: {e}")
    
    def _add_effects(self, video_path: str, job: RenderJob):
        """Add effects like watermark and captions."""
        temp_path = self.temp_dir / "with_effects.mp4"
        
        try:
            input_video = ffmpeg.input(video_path)
            
            # Add watermark if specified
            if job.add_watermark and job.watermark_path:
                watermark = ffmpeg.input(job.watermark_path)
                input_video = ffmpeg.filter([input_video, watermark], 'overlay', 10, 10)
            
            # Add captions if specified
            if job.add_captions and job.captions_path:
                input_video = input_video.filter('subtitles', job.captions_path)
            
            (
                ffmpeg
                .output(input_video, str(temp_path))
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Replace original with processed version
            os.replace(str(temp_path), video_path)
            
        except ffmpeg.Error as e:
            raise Exception(f"Failed to add effects: {e}")
    
    def _cleanup(self, paths: List[Path]):
        """Clean up temporary files."""
        for path in paths:
            if path and path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {path}: {e}")
    
    def _generate_metadata(self, job: RenderJob) -> Dict[str, Any]:
        """Generate metadata for the rendered clip."""
        return {
            "duration": job.end_time - job.start_time,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "preset": {
                "id": job.preset.id,
                "name": job.preset.name,
                "resolution": f"{job.preset.width}x{job.preset.height}",
                "aspect_ratio": job.preset.aspect_ratio
            },
            "effects": {
                "fade_duration": job.fade_duration,
                "watermark": job.add_watermark,
                "captions": job.add_captions
            }
        }
    
    def get_preset(self, preset_id: str) -> Optional[RenderPreset]:
        """Get a rendering preset by ID."""
        return self.presets.get(preset_id)
    
    def list_presets(self) -> List[RenderPreset]:
        """List all available rendering presets."""
        return list(self.presets.values())
    
    def create_playlist(
        self,
        clips: List[Dict[str, Any]],
        output_path: str,
        title: str = "Highlight Playlist"
    ) -> Dict[str, Any]:
        """Create a playlist file for multiple clips."""
        playlist_data = {
            "title": title,
            "clips": []
        }
        
        for clip in clips:
            if clip.get('success', False):
                playlist_data["clips"].append({
                    "file": clip['output_path'],
                    "title": clip.get('title', 'Highlight'),
                    "duration": clip.get('metadata', {}).get('duration', 0),
                    "start_time": clip.get('metadata', {}).get('start_time', 0)
                })
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(playlist_data, f, indent=2)
        
        return {
            "success": True,
            "playlist_path": output_path,
            "clip_count": len(playlist_data["clips"])
        }
    
    def create_concatenated_video(
        self,
        clips: List[Dict[str, Any]],
        output_path: str,
        transition_duration: float = 0.5
    ) -> Dict[str, Any]:
        """Create a single video by concatenating multiple clips."""
        if not clips:
            return {"success": False, "error": "No clips provided"}
        
        # Create file list for ffmpeg concat
        file_list_path = self.temp_dir / "file_list.txt"
        
        with open(file_list_path, 'w') as f:
            for clip in clips:
                if clip.get('success', False):
                    f.write(f"file '{clip['output_path']}'\n")
        
        try:
            (
                ffmpeg
                .input(str(file_list_path), format='concat', safe=0)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Cleanup
            file_list_path.unlink()
            
            return {
                "success": True,
                "output_path": output_path,
                "clip_count": len([c for c in clips if c.get('success', False)])
            }
            
        except ffmpeg.Error as e:
            return {"success": False, "error": f"Failed to concatenate videos: {e}"}
