"""
Audio Stitching Service

Section 9: Audio Stitching
Combines individual audio clips into a single MP3 file with pauses.
"""

import logging
import io
from typing import List, Optional
from pydub import AudioSegment

from backend.config import settings
from backend.services.tts import TTSResult

logger = logging.getLogger(__name__)


class AudioStitchingTimeoutError(Exception):
    """Raised when audio stitching times out."""
    pass


class AudioStitcher:
    """
    Audio stitching engine for combining TTS clips.
    
    Section 9: Audio Stitching
    - Stitch all audio clips into one MP3 file
    - Insert pauses between text bubbles
    - Maintain original reading order
    """
    
    def __init__(self, pause_duration_ms: Optional[int] = None):
        """
        Initialize audio stitcher.
        
        Args:
            pause_duration_ms: Pause duration between clips in milliseconds
        """
        self.pause_duration_ms = pause_duration_ms or settings.AUDIO_PAUSE_DURATION_MS
        logger.info(f"AudioStitcher initialized with {self.pause_duration_ms}ms pause")
    
    def stitch_audio_clips(
        self,
        tts_results: List[TTSResult],
        output_format: str = "mp3"
    ) -> bytes:
        """
        Stitch multiple TTS results into a single audio file.
        
        Section 9: Stitch all audio clips into one MP3 file
        Section 9: Insert pauses between text bubbles
        Section 9: Maintain original reading order
        
        Args:
            tts_results: List of TTSResult objects
            output_format: Output format ('mp3' only supported currently)
            
        Returns:
            Stitched audio as bytes
            
        Raises:
            ValueError: If tts_results is empty or invalid
        """
        if not tts_results:
            raise ValueError("Cannot stitch empty list of TTS results")
        
        if output_format != "mp3":
            raise ValueError(f"Unsupported output format: {output_format}. Only 'mp3' is supported.")
        
        logger.info(f"Stitching {len(tts_results)} audio clips with {self.pause_duration_ms}ms pauses")
        
        # Sort by reading order to ensure correct sequence
        sorted_results = sorted(tts_results, key=lambda r: r.reading_order)
        
        # Start with empty audio
        combined_audio = AudioSegment.silent(duration=0)
        
        for idx, result in enumerate(sorted_results):
            logger.info(f"Processing clip {idx + 1}/{len(sorted_results)}: order={result.reading_order}")
            
            # Load MP3 audio from bytes
            try:
                audio_segment = AudioSegment.from_mp3(io.BytesIO(result.audio_bytes))
            except Exception as e:
                logger.error(f"Failed to load audio for clip {idx + 1}: {e}")
                raise
            
            # Add the audio clip
            combined_audio += audio_segment
            
            # Add pause after each clip except the last one
            if idx < len(sorted_results) - 1:
                silence = AudioSegment.silent(duration=self.pause_duration_ms)
                combined_audio += silence
        
        # Export to MP3 bytes
        output_buffer = io.BytesIO()
        combined_audio.export(output_buffer, format="mp3")
        output_bytes = output_buffer.getvalue()
        
        duration_seconds = len(combined_audio) / 1000.0  # pydub duration is in milliseconds
        logger.info(f"Stitched audio: {len(output_bytes)} bytes, duration: {duration_seconds:.2f}s")
        
        return output_bytes
    
    def stitch_audio_clips_to_file(
        self,
        tts_results: List[TTSResult],
        output_path: str,
        output_format: Optional[str] = None
    ) -> str:
        """
        Stitch audio clips and save to file.
        
        Args:
            tts_results: List of TTSResult objects
            output_path: Path to save the output file
            output_format: Output format (inferred from path if None)
            
        Returns:
            Path to the saved file
        """
        # Infer format from file extension if not provided
        if output_format is None:
            output_format = output_path.split('.')[-1].lower()
        
        audio_bytes = self.stitch_audio_clips(tts_results, output_format)
        
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        logger.info(f"Saved stitched audio to: {output_path}")
        return output_path
    
    def get_total_duration_ms(self, tts_results: List[TTSResult]) -> int:
        """
        Calculate total duration of stitched audio.
        
        Args:
            tts_results: List of TTSResult objects
            
        Returns:
            Total duration in milliseconds (including pauses)
        """
        if not tts_results:
            return 0
        
        total_duration_ms = 0
        
        for idx, result in enumerate(tts_results):
            # Load audio to get duration
            audio_segment = AudioSegment.from_mp3(io.BytesIO(result.audio_bytes))
            total_duration_ms += len(audio_segment)  # pydub duration is in milliseconds
            
            # Add pause duration (except after last clip)
            if idx < len(tts_results) - 1:
                total_duration_ms += self.pause_duration_ms
        
        logger.info(f"Total duration: {total_duration_ms}ms for {len(tts_results)} clips")
        
        return total_duration_ms
