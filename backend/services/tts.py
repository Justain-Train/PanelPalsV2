"""
ElevenLabs TTS Service

Section 8: Text-to-Speech (ElevenLabs)
Generates audio from text using ElevenLabs API with parallel processing.
"""

import logging
import asyncio
from typing import List, Optional
from dataclasses import dataclass

from elevenlabs import generate, Voice, VoiceSettings
from elevenlabs.api import History

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """
    Structured TTS result for a single text-to-speech conversion.
    
    Section 8.1: Each text bubble → individual audio clip
    """
    text: str
    audio_bytes: bytes
    reading_order: int
    voice_id: str
    
    def __post_init__(self):
        """Validate audio bytes."""
        if not isinstance(self.audio_bytes, bytes):
            raise TypeError("audio_bytes must be bytes")


class ElevenLabsTTSService:
    """
    ElevenLabs TTS service.
    
    Section 8.1: TTS Strategy
    - Send text bubbles in parallel batches
    - Respect API rate limits
    - Each text bubble → individual audio clip
    """
    
    def __init__(self, api_key: Optional[str] = None, voice_id: Optional[str] = None):
        """
        Initialize ElevenLabs TTS service.
        
        Args:
            api_key: ElevenLabs API key (defaults to settings)
            voice_id: Default voice ID (defaults to settings)
        """
        self.api_key = api_key or settings.ELEVENLABS_API_KEY
        self.voice_id = voice_id or settings.ELEVENLABS_VOICE_ID
        
        if not settings.ELEVENLABS_CONFIGURED:
            logger.warning(
                "ElevenLabs API not configured. "
                "Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env"
            )
        else:
            logger.info(f"ElevenLabs TTS service initialized with voice: {self.voice_id}")
    
    def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        reading_order: int = 1
    ) -> TTSResult:
        """
        Generate speech from text (synchronous).
        
        Section 8.1: Each text bubble → individual audio clip
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID (defaults to instance voice_id)
            reading_order: Reading order position
            
        Returns:
            TTSResult with audio bytes
            
        Raises:
            ValueError: If TTS not configured or text is empty
        """
        if not self.api_key or not self.voice_id:
            raise ValueError(
                "ElevenLabs API not configured. "
                "Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env"
            )
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        voice_id = voice_id or self.voice_id
        
        logger.info(f"Generating speech for: '{text[:50]}...' (order: {reading_order})")
        
        try:
            # Generate audio using ElevenLabs
            # Use eleven_turbo_v2 model (free tier compatible)
            audio_bytes = generate(
                text=text,
                voice=Voice(
                    voice_id=voice_id,
                    settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75
                    )
                ),
                model="eleven_turbo_v2",  # Updated model for free tier compatibility
                api_key=self.api_key
            )
            
            # Convert generator to bytes if needed
            if not isinstance(audio_bytes, bytes):
                audio_bytes = b''.join(audio_bytes)
            
            logger.info(f"Generated {len(audio_bytes)} bytes of audio")
            
            return TTSResult(
                text=text,
                audio_bytes=audio_bytes,
                reading_order=reading_order,
                voice_id=voice_id
            )
            
        except Exception as e:
            logger.error(f"TTS generation failed for text '{text[:50]}...': {e}")
            raise
    
    async def generate_speech_async(
        self,
        text: str,
        voice_id: Optional[str] = None,
        reading_order: int = 1
    ) -> TTSResult:
        """
        Generate speech asynchronously (for parallel processing).
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID (defaults to instance voice_id)
            reading_order: Reading order position
            
        Returns:
            TTSResult with audio bytes
        """
        # Run synchronous generate in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_speech,
            text,
            voice_id,
            reading_order
        )
    
    async def generate_speech_batch(
        self,
        texts: List[str],
        voice_id: Optional[str] = None
    ) -> List[TTSResult]:
        """
        Generate speech for multiple texts in parallel.
        
        Section 8.1: Send text bubbles in parallel batches
        Section 8.1: Respect API rate limits
        
        Args:
            texts: List of text strings to convert (in reading order)
            voice_id: Voice ID (defaults to instance voice_id)
            
        Returns:
            List of TTSResult objects in reading order
            
        Raises:
            ValueError: If TTS not configured
        """
        if not self.api_key or not self.voice_id:
            raise ValueError("ElevenLabs API not configured")
        
        if not texts:
            return []
        
        logger.info(f"Generating speech for {len(texts)} text bubbles in parallel")
        
        # Create tasks with rate limiting
        max_parallel = settings.ELEVENLABS_MAX_PARALLEL_REQUESTS
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def rate_limited_generate(text: str, order: int) -> TTSResult:
            """Generate speech with rate limiting."""
            async with semaphore:
                return await self.generate_speech_async(text, voice_id, order)
        
        # Create tasks for all texts
        tasks = [
            rate_limited_generate(text, idx + 1)
            for idx, text in enumerate(texts)
        ]
        
        # Execute in parallel with rate limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Text {idx + 1} failed: {result}")
                # Could implement retry logic here
            else:
                valid_results.append(result)
        
        # Sort by reading order to ensure correct sequence
        valid_results.sort(key=lambda r: r.reading_order)
        
        logger.info(f"Successfully generated {len(valid_results)}/{len(texts)} audio clips")
        return valid_results
    
    def generate_speech_batch_sync(
        self,
        texts: List[str],
        voice_id: Optional[str] = None
    ) -> List[TTSResult]:
        """
        Synchronous wrapper for batch generation.
        
        Args:
            texts: List of text strings to convert
            voice_id: Voice ID (defaults to instance voice_id)
            
        Returns:
            List of TTSResult objects in reading order
        """
        return asyncio.run(self.generate_speech_batch(texts, voice_id))
