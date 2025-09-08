"""
Unit tests for FastAPI endpoints.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import os

# Add the server to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "server"))

from main import app


class TestHealthEndpoint:
    """Test cases for health check endpoint."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "database_connected" in data
        assert "active_connections" in data


class TestUploadEndpoint:
    """Test cases for file upload endpoint."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('apps.server.api.routes.upload.probe_media_file')
    @patch('apps.server.core.database.DatabaseManager')
    def test_upload_valid_file(self, mock_db, mock_probe):
        """Test uploading a valid video file."""
        # Mock probe response
        mock_probe.return_value = {
            'duration': 300.0,
            'size': 1024000,
            'width': 1920,
            'height': 1080,
            'fps': 30.0,
            'video_format': 'h264',
            'audio_format': 'aac'
        }
        
        # Mock database response
        mock_media_file = MagicMock()
        mock_media_file.id = "test-id"
        mock_media_file.name = "test.mp4"
        mock_media_file.path = "/tmp/test.mp4"
        mock_media_file.size = 1024000
        mock_media_file.duration = 300.0
        mock_media_file.resolution_width = 1920
        mock_media_file.resolution_height = 1080
        mock_media_file.fps = 30.0
        mock_media_file.audio_format = "aac"
        mock_media_file.video_format = "h264"
        mock_media_file.created_at = "2023-01-01T00:00:00"
        
        mock_db_instance = MagicMock()
        mock_db_instance.create_media_file.return_value = mock_media_file
        mock_db.return_value.__aenter__.return_value = mock_db_instance
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = self.client.post(
                    "/api/upload",
                    files={"file": ("test.mp4", f, "video/mp4")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "media_file" in data
            assert data["media_file"]["name"] == "test.mp4"
            
        finally:
            os.unlink(tmp_path)
    
    def test_upload_invalid_file_type(self):
        """Test uploading an invalid file type."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"not a video")
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, 'rb') as f:
                response = self.client.post(
                    "/api/upload",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
            assert "video" in data["detail"].lower()
            
        finally:
            os.unlink(tmp_path)
    
    def test_upload_file_too_large(self):
        """Test uploading a file that's too large."""
        # Create a large file (simulate)
        with patch('apps.server.api.routes.upload.MAX_FILE_SIZE_GB', 0.001):  # 1MB limit
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                # Write 2MB of data
                tmp_file.write(b"x" * (2 * 1024 * 1024))
                tmp_path = tmp_file.name
            
            try:
                with open(tmp_path, 'rb') as f:
                    response = self.client.post(
                        "/api/upload",
                        files={"file": ("large.mp4", f, "video/mp4")}
                    )
                
                assert response.status_code == 413
                data = response.json()
                assert "detail" in data
                assert "size" in data["detail"].lower()
                
            finally:
                os.unlink(tmp_path)


class TestSessionEndpoint:
    """Test cases for session management endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('apps.server.core.database.DatabaseManager')
    def test_create_session(self, mock_db):
        """Test creating a new session."""
        # Mock database response
        mock_session = MagicMock()
        mock_session.id = "session-123"
        mock_session.media_file_id = "file-123"
        mock_session.mode = "sports"
        mock_session.status = "idle"
        mock_session.progress = 0.0
        mock_session.created_at = "2023-01-01T00:00:00"
        mock_session.updated_at = "2023-01-01T00:00:00"
        
        mock_db_instance = MagicMock()
        mock_db_instance.create_session.return_value = mock_session
        mock_db.return_value.__aenter__.return_value = mock_db_instance
        
        session_data = {
            "media_file": {
                "id": "file-123",
                "name": "test.mp4",
                "path": "/tmp/test.mp4",
                "size": 1024000,
                "duration": 300.0,
                "resolution": {"width": 1920, "height": 1080},
                "fps": 30.0,
                "audio_format": "aac",
                "video_format": "h264",
                "created_at": "2023-01-01T00:00:00"
            },
            "mode": "sports"
        }
        
        response = self.client.post("/api/sessions", json=session_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "session-123"
        assert data["mode"] == "sports"
        assert data["status"] == "idle"
    
    @patch('apps.server.core.database.DatabaseManager')
    def test_get_session(self, mock_db):
        """Test getting a session by ID."""
        # Mock session
        mock_session = MagicMock()
        mock_session.id = "session-123"
        mock_session.media_file_id = "file-123"
        mock_session.mode = "sports"
        mock_session.status = "idle"
        mock_session.progress = 0.0
        mock_session.created_at = "2023-01-01T00:00:00"
        mock_session.updated_at = "2023-01-01T00:00:00"
        
        # Mock media file
        mock_media_file = MagicMock()
        mock_media_file.id = "file-123"
        mock_media_file.name = "test.mp4"
        mock_media_file.path = "/tmp/test.mp4"
        mock_media_file.size = 1024000
        mock_media_file.duration = 300.0
        mock_media_file.resolution_width = 1920
        mock_media_file.resolution_height = 1080
        mock_media_file.fps = 30.0
        mock_media_file.audio_format = "aac"
        mock_media_file.video_format = "h264"
        mock_media_file.created_at = "2023-01-01T00:00:00"
        
        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_instance.get_media_file.return_value = mock_media_file
        mock_db_instance.get_events_for_session.return_value = []
        mock_db.return_value.__aenter__.return_value = mock_db_instance
        
        response = self.client.get("/api/sessions/session-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "session-123"
        assert data["mode"] == "sports"
        assert "media_file" in data
        assert "events" in data
    
    @patch('apps.server.core.database.DatabaseManager')
    def test_get_session_not_found(self, mock_db):
        """Test getting a non-existent session."""
        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = None
        mock_db.return_value.__aenter__.return_value = mock_db_instance
        
        response = self.client.get("/api/sessions/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    @patch('apps.server.core.database.DatabaseManager')
    def test_start_detection(self, mock_db):
        """Test starting detection for a session."""
        # Mock session
        mock_session = MagicMock()
        mock_session.id = "session-123"
        mock_session.status = "idle"
        
        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_instance.update_session.return_value = mock_session
        mock_db.return_value.__aenter__.return_value = mock_db_instance
        
        response = self.client.post("/api/sessions/session-123/detect")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "started" in data["message"].lower()


class TestRenderEndpoint:
    """Test cases for render endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('apps.server.core.database.DatabaseManager')
    def test_create_render_job(self, mock_db):
        """Test creating a render job."""
        # Mock session
        mock_session = MagicMock()
        mock_session.id = "session-123"
        
        # Mock render job
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.session_id = "session-123"
        mock_job.status = "pending"
        mock_job.progress = 0.0
        mock_job.output_path = None
        mock_job.created_at = "2023-01-01T00:00:00"
        mock_job.updated_at = "2023-01-01T00:00:00"
        
        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_instance.create_render_job.return_value = mock_job
        mock_db.return_value.__aenter__.return_value = mock_db_instance
        
        job_data = {
            "session_id": "session-123",
            "event_ids": ["event-1", "event-2"],
            "preset": {
                "id": "vertical",
                "name": "Vertical (9:16)",
                "width": 1080,
                "height": 1920,
                "aspect_ratio": "9:16",
                "crop_strategy": "motion_centroid"
            }
        }
        
        response = self.client.post("/api/render", json=job_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "job-123"
        assert data["session_id"] == "session-123"
        assert data["status"] == "pending"
    
    @patch('apps.server.core.database.DatabaseManager')
    def test_get_render_job(self, mock_db):
        """Test getting a render job by ID."""
        # Mock render job
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.session_id = "session-123"
        mock_job.event_ids_json = '["event-1", "event-2"]'
        mock_job.preset_json = '{"id": "vertical", "name": "Vertical (9:16)"}'
        mock_job.status = "completed"
        mock_job.progress = 1.0
        mock_job.output_path = "/tmp/output.mp4"
        mock_job.created_at = "2023-01-01T00:00:00"
        mock_job.updated_at = "2023-01-01T00:00:00"
        
        mock_db_instance = MagicMock()
        mock_db_instance.get_render_job.return_value = mock_job
        mock_db.return_value.__aenter__.return_value = mock_db_instance
        
        response = self.client.get("/api/render/job-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "job-123"
        assert data["status"] == "completed"
        assert data["output_path"] == "/tmp/output.mp4"


class TestPresetsEndpoint:
    """Test cases for presets endpoint."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_get_presets(self):
        """Test getting output presets."""
        response = self.client.get("/api/presets")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3  # vertical, square, wide
        
        # Check that all presets have required fields
        for preset in data:
            assert "id" in preset
            assert "name" in preset
            assert "width" in preset
            assert "height" in preset
            assert "aspect_ratio" in preset
            assert "crop_strategy" in preset
        
        # Check specific presets
        preset_ids = [p["id"] for p in data]
        assert "vertical" in preset_ids
        assert "square" in preset_ids
        assert "wide" in preset_ids


if __name__ == "__main__":
    pytest.main([__file__])
