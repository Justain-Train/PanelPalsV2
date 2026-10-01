"""
ElevenLabs TTS Service

Section 8: Text-to-Speech (ElevenLabs)
Generates audio from text using ElevenLabs API with parallel processing.
"""

import logging
import asyncio
import json
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from dataclasses import dataclass

import requests
from elevenlabs import generate, Voice, VoiceSettings
from elevenlabs.api import History
from elevenlabs.api.error import APIError, AuthorizationError, RateLimitError

from backend.config import settings

logger = logging.getLogger(__name__)

# Base delay for exponential backoff between retries (1s, 2s, 4s, ... plus jitter)
RETRY_BASE_DELAY_SECONDS = 1.0


def _is_retryable(error: Exception) -> bool:
    """
    Whether a TTS failure is worth retrying.

    Retry: rate limiting / busy server / 5xx, network errors, and non-JSON
    error pages (the SDK fails parsing those with JSONDecodeError).
    Don't retry: quota exceeded (the SDK's RateLimitError is HTTP 401
    'quota_exceeded' - out of credits), authorization errors, and other API
    errors such as an invalid voice.

    ElevenLabs' exact status strings for "too many requests" aren't documented
    in this SDK version, so status text mentioning rate/concurrent/busy counts
    as retryable, as do numeric 429 and 5xx statuses.
    """
    if isinstance(error, (RateLimitError, AuthorizationError)):
        return False
    if isinstance(error, APIError):
        status = str(error.status).lower()
        if status.isdigit():
            return status == "429" or status.startswith("5")
        return any(word in status for word in ("rate", "concurrent", "busy"))
    return isinstance(error, (requests.exceptions.ConnectionError,
                              requests.exceptions.Timeout,
                              json.JSONDecodeError))


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
        configured = settings.ELEVENLABS_API_KEY
        # SecretStr in settings; plain str also accepted (tests, overrides)
        self.api_key = api_key or getattr(configured, "get_secret_value", lambda: configured)()
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
            audio_bytes = generate(
                text=text,
                voice=Voice(
                    voice_id="NNl6r8mD7vthiJatiJt1",
                    settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75
                    )
                ),
                model="eleven_v4_turbo",
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
        reading_order: int = 1,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> TTSResult:
        """
        Generate speech asynchronously (for parallel processing).

        Args:
            text: Text to convert to speech
            voice_id: Voice ID (defaults to instance voice_id)
            reading_order: Reading order position
            executor: Thread pool to run the blocking SDK call in. Defaults to
                asyncio's shared pool, which only has min(32, CPUs + 4) threads.

        Returns:
            TTSResult with audio bytes
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
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
            List of TTSResult objects in reading order (failed clips omitted;
            use generate_speech_batch_detailed to find out which)

        Raises:
            ValueError: If TTS not configured
        """
        results, _ = await self.generate_speech_batch_detailed(texts, voice_id)
        return results

    async def generate_speech_batch_detailed(
        self,
        texts: List[str],
        voice_id: Optional[str] = None
    ) -> Tuple[List[TTSResult], List[int]]:
        """
        Generate speech for multiple texts in parallel, with retries.

        Runs up to ELEVENLABS_MAX_PARALLEL_REQUESTS calls at once on a
        dedicated thread pool of that size. Retryable failures (rate limit,
        busy, 5xx, network) are retried up to TTS_MAX_RETRIES times with
        exponential backoff; the parallel slot is released while waiting.

        Args:
            texts: List of text strings to convert (in reading order)
            voice_id: Voice ID (defaults to instance voice_id)

        Returns:
            (results in reading order, 1-based reading orders that still
            failed after retries)

        Raises:
            ValueError: If TTS not configured
        """
        if not self.api_key or not self.voice_id:
            raise ValueError("ElevenLabs API not configured")

        if not texts:
            return [], []

        max_parallel = max(1, settings.ELEVENLABS_MAX_PARALLEL_REQUESTS)
        max_retries = max(0, settings.TTS_MAX_RETRIES)
        logger.info(
            f"Generating speech for {len(texts)} text bubbles, "
            f"{max_parallel} in parallel, up to {max_retries} retries each"
        )

        semaphore = asyncio.Semaphore(max_parallel)

        async def generate_with_retries(text: str, order: int, executor) -> TTSResult:
            for attempt in range(max_retries + 1):
                async with semaphore:
                    try:
                        return await self.generate_speech_async(text, voice_id, order, executor)
                    except Exception as e:
                        if attempt == max_retries or not _is_retryable(e):
                            raise
                        error = e
                # Back off outside the semaphore so other lines keep using the slot
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt) * (1 + random.random() * 0.5)
                logger.warning(
                    f"Text {order} attempt {attempt + 1} failed ({error}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

        # Dedicated pool so the parallel setting is actually reachable on small machines
        with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="tts") as executor:
            results = await asyncio.gather(
                *(generate_with_retries(text, idx + 1, executor) for idx, text in enumerate(texts)),
                return_exceptions=True
            )

        valid_results, failed = [], []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Text {idx + 1} failed: {result}")
                failed.append(idx + 1)
            else:
                valid_results.append(result)

        # Sort by reading order to ensure correct sequence
        valid_results.sort(key=lambda r: r.reading_order)

        logger.info(f"Successfully generated {len(valid_results)}/{len(texts)} audio clips")
        if failed:
            logger.warning(f"Missing audio for lines {failed} after retries")
        return valid_results, failed
    
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
