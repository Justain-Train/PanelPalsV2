"""
Audio Stitching Service

Section 9: Audio Stitching
Combines individual audio clips into a single MP3 file with pauses.
"""

import logging
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from pydub import AudioSegment

from backend.config import settings
from backend.services.tts import TTSResult

logger = logging.getLogger(__name__)

# Each MP3 decode runs as its own ffmpeg process; threads just wait on it,
# so a small pool decodes several clips at once.
DECODE_WORKERS = min(8, os.cpu_count() or 1)


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
    
    def __init__(self, pause_duration_ms: Optional[int] = None, output_format: Optional[str] = None):
        """
        Initialize audio stitcher.
        
        Args:
            pause_duration_ms: Pause duration between clips in milliseconds
            output_format: Output format (currently only 'mp3' supported)
        """
        self.pause_duration_ms = pause_duration_ms or settings.AUDIO_PAUSE_DURATION_MS
        self.output_format = output_format or settings.AUDIO_OUTPUT_FORMAT
        logger.info(f"AudioStitcher initialized with {self.pause_duration_ms}ms pause and {self.output_format} format")

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
        output_bytes, _ = self.stitch_audio_clips_with_duration(tts_results, output_format)
        return output_bytes

    def stitch_audio_clips_with_duration(
        self,
        tts_results: List[TTSResult],
        output_format: str = "mp3"
    ) -> Tuple[bytes, int]:
        """
        Stitch TTS clips into one MP3 and return it with its duration.

        Each clip is decoded exactly once (in parallel, since every decode is
        an ffmpeg subprocess), converted to one common format, and joined in a
        single pass - repeated `+=` would copy the growing audio every time.

        Args:
            tts_results: List of TTSResult objects
            output_format: Output format ('mp3' only supported currently)

        Returns:
            (stitched audio bytes, duration in milliseconds including pauses)

        Raises:
            ValueError: If tts_results is empty or the format is unsupported
        """
        if not tts_results:
            raise ValueError("Cannot stitch empty list of TTS results")

        if output_format != "mp3":
            raise ValueError(f"Unsupported output format: {output_format}. Only 'mp3' is supported.")

        logger.info(f"Stitching {len(tts_results)} audio clips with {self.pause_duration_ms}ms pauses")

        # Sort by reading order to ensure correct sequence
        sorted_results = sorted(tts_results, key=lambda r: r.reading_order)

        def decode(indexed_result):
            idx, result = indexed_result
            logger.debug(f"Decoding clip {idx + 1}/{len(sorted_results)}: order={result.reading_order}")
            try:
                return AudioSegment.from_mp3(io.BytesIO(result.audio_bytes))
            except Exception as e:
                logger.error(f"Failed to load audio for clip {idx + 1}: {e}")
                raise

        # map() keeps input order; the first decode error propagates
        with ThreadPoolExecutor(max_workers=min(DECODE_WORKERS, len(sorted_results))) as pool:
            segments = list(pool.map(decode, enumerate(sorted_results)))

        # Common format (the first clip's) so raw frames can be joined directly
        first = segments[0]
        frame_rate, channels, sample_width = first.frame_rate, first.channels, first.sample_width

        def normalize(segment: AudioSegment) -> AudioSegment:
            if segment.frame_rate != frame_rate:
                segment = segment.set_frame_rate(frame_rate)
            if segment.channels != channels:
                segment = segment.set_channels(channels)
            if segment.sample_width != sample_width:
                segment = segment.set_sample_width(sample_width)
            return segment

        silence = normalize(AudioSegment.silent(duration=self.pause_duration_ms, frame_rate=frame_rate))

        parts = []
        for idx, segment in enumerate(segments):
            parts.append(normalize(segment).raw_data)
            if idx < len(segments) - 1:
                parts.append(silence.raw_data)

        combined_audio = AudioSegment(
            data=b"".join(parts),
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=channels
        )

        # Export to MP3 bytes
        output_buffer = io.BytesIO()
        combined_audio.export(output_buffer, format=output_format)
        output_bytes = output_buffer.getvalue()

        duration_ms = len(combined_audio)  # pydub duration is in milliseconds
        logger.info(f"Stitched audio: {len(output_bytes)} bytes, duration: {duration_ms / 1000:.2f}s")

        return output_bytes, duration_ms
    
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
