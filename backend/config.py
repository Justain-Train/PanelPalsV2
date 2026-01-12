"""
Configuration management for PanelPals V2 Backend

Handles environment variables, API keys, and application settings.
Section 5.2: Secure-by-default API design.
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration using Pydantic BaseSettings."""
    
    # Application Settings
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000"],
        description="CORS allowed origins"
    )
    
    # Google Vision API (Section 6: OCR Pipeline)
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(
        default="",
        description="Path to Google Cloud credentials JSON"
    )
    GOOGLE_VISION_MAX_BATCH_SIZE: int = Field(
        default=10,
        description="Max images per Vision API batch request"
    )
    
    # ElevenLabs API (Section 8: Text-to-Speech)
    ELEVENLABS_API_KEY: str = Field(
        default="",
        description="ElevenLabs API key"
    )
    ELEVENLABS_VOICE_ID: str = Field(
        default="",
        description="Default narrator voice ID for MVP"
    )
    ELEVENLABS_MAX_PARALLEL_REQUESTS: int = Field(
        default=5,
        description="Maximum parallel TTS requests to respect rate limits"
    )
    
    # Audio Processing (Section 9: Audio Stitching)
    AUDIO_PAUSE_DURATION_MS: int = Field(
        default=500,
        description="Pause duration between text bubbles in milliseconds"
    )
    AUDIO_OUTPUT_FORMAT: str = Field(
        default="mp3",
        description="Output audio format (changed from WAV to MP3)"
    )
    AUDIO_SAMPLE_RATE: int = Field(
        default=44100,
        description="Audio sample rate in Hz"
    )
    AUDIO_FFMPEG_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="Timeout for ffmpeg conversion operations in seconds"
    )
    AUDIO_STITCH_TOTAL_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Total timeout for entire audio stitching operation in seconds"
    )
    
    # Text Bubble Grouping (Section 6.1)
    BUBBLE_MAX_VERTICAL_GAP: int = Field(
        default=100,
        description="Maximum vertical gap (pixels) to group lines into same bubble"
    )
    BUBBLE_MAX_CENTER_SHIFT: int = Field(
        default=150,
        description="Maximum horizontal center shift for same bubble"
    )
    
    # Security & Rate Limiting (Section 5.2)
    MAX_IMAGE_SIZE_MB: int = Field(
        default=10,
        description="Maximum image upload size in megabytes"
    )
    MAX_IMAGES_PER_REQUEST: int = Field(
        default=50,
        description="Maximum images per batch request"
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=10,
        description="API rate limit per client per minute"
    )
    
    @property
    def GOOGLE_VISION_CONFIGURED(self) -> bool:
        """Check if Google Vision API is configured."""
        return bool(self.GOOGLE_APPLICATION_CREDENTIALS and 
                   os.path.exists(self.GOOGLE_APPLICATION_CREDENTIALS))
    
    @property
    def ELEVENLABS_CONFIGURED(self) -> bool:
        """Check if ElevenLabs API is configured."""
        return bool(self.ELEVENLABS_API_KEY and self.ELEVENLABS_VOICE_ID)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
