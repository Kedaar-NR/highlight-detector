"""
Configuration management for the Highlight Detector server.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
import yaml
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Hardware
    CUDA_VISIBLE_DEVICES: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    TORCH_DEVICE: str = Field(default="auto", env="TORCH_DEVICE")
    NUM_WORKERS: int = Field(default=4, env="NUM_WORKERS")
    CACHE_SIZE_GB: int = Field(default=2, env="CACHE_SIZE_GB")
    
    # Paths
    TEMP_DIR: str = Field(default="./temp", env="TEMP_DIR")
    CACHE_DIR: str = Field(default="./cache", env="CACHE_DIR")
    OUTPUT_DIR: str = Field(default="./output", env="OUTPUT_DIR")
    SAMPLE_DATA_DIR: str = Field(default="./data/samples", env="SAMPLE_DATA_DIR")
    
    # Model Configuration
    MODEL_CACHE_DIR: str = Field(default="./models", env="MODEL_CACHE_DIR")
    DEFAULT_MODEL_PATH: str = Field(default="./models/default.onnx", env="DEFAULT_MODEL_PATH")
    OCR_ENABLED: bool = Field(default=False, env="OCR_ENABLED")
    OCR_MODEL_PATH: str = Field(default="./models/ocr.onnx", env="OCR_MODEL_PATH")
    
    # Performance
    MAX_FILE_SIZE_GB: int = Field(default=10, env="MAX_FILE_SIZE_GB")
    DETECTION_TIMEOUT_SECONDS: int = Field(default=300, env="DETECTION_TIMEOUT_SECONDS")
    RENDER_TIMEOUT_SECONDS: int = Field(default=600, env="RENDER_TIMEOUT_SECONDS")
    MEMORY_LIMIT_GB: int = Field(default=8, env="MEMORY_LIMIT_GB")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    LOG_DIR: str = Field(default="./logs", env="LOG_DIR")
    
    # Development
    SEED: int = Field(default=42, env="SEED")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class ConfigManager:
    """Manages YAML configuration files."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[dict] = None
    
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        if self._config is None:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = {}
        return self._config
    
    def get_mode_config(self, mode: str) -> dict:
        """Get configuration for a specific detection mode."""
        config = self.load_config()
        return config.get('modes', {}).get(mode, {})
    
    def get_detection_config(self) -> dict:
        """Get detection configuration."""
        config = self.load_config()
        return config.get('detection', {})
    
    def get_output_presets(self) -> dict:
        """Get output presets configuration."""
        config = self.load_config()
        return config.get('output_presets', {})
    
    def get_audio_config(self) -> dict:
        """Get audio processing configuration."""
        config = self.load_config()
        return config.get('audio', {})
    
    def get_video_config(self) -> dict:
        """Get video processing configuration."""
        config = self.load_config()
        return config.get('video', {})


# Global instances
_settings: Optional[Settings] = None
_config_manager: Optional[ConfigManager] = None


def get_settings() -> Settings:
    """Get application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_config_manager() -> ConfigManager:
    """Get configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
