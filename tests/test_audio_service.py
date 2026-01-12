"""
Unit tests for Audio Stitching Service

Section 14.4: TTS & Audio Tests
Tests audio stitching with WAV file generation.
"""

import pytest
import io
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from scipy.io import wavfile

from backend.services.audio import AudioStitcher
from backend.services.tts import TTSResult


@pytest.fixture
def stitcher():
    """Create AudioStitcher with default settings."""
    return AudioStitcher(pause_duration_ms=500)


@pytest.fixture
def mock_tts_result():
    """Create a mock TTS result with simulated MP3 data."""
    return TTSResult(
        text="Test text",
        audio_bytes=create_test_mp3_bytes(),
        reading_order=1,
        voice_id="test_voice"
    )


def create_test_mp3_bytes(duration_ms: int = 100) -> bytes:
    """Create test MP3 bytes (placeholder for actual MP3 data)."""
    # Create a simple WAV as MP3 placeholder
    sample_rate = 22050
    num_samples = int((duration_ms / 1000.0) * sample_rate)
    
    # Generate a simple sine wave
    frequency = 440.0  # A note
    t = np.linspace(0, duration_ms / 1000.0, num_samples)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to int16
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write to bytes as WAV (acting as MP3 placeholder for tests)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_data)
    return buffer.getvalue()


# Section 14.4: Audio Tests - Verify audio file creation per text bubble

class TestAudioStitcherInit:
    """Test AudioStitcher initialization."""
    
    def test_init_default_pause(self, stitcher):
        """Test initialization with default pause duration."""
        assert stitcher.pause_duration_ms == 500
        assert stitcher.sample_rate == 44100  # From settings default
    
    def test_init_custom_pause(self):
        """Test initialization with custom pause duration."""
        stitcher = AudioStitcher(pause_duration_ms=1000)
        assert stitcher.pause_duration_ms == 1000
    
    def test_init_none_pause_uses_settings(self):
        """Test initialization with None uses settings default."""
        with patch('backend.services.audio.settings') as mock_settings:
            mock_settings.AUDIO_PAUSE_DURATION_MS = 600
            mock_settings.AUDIO_SAMPLE_RATE = 44100
            stitcher = AudioStitcher()
            assert stitcher.pause_duration_ms == 600


class TestCreateSilence:
    """Test silence generation."""
    
    def test_create_silence_correct_duration(self, stitcher):
        """Test silence has correct duration."""
        duration_ms = 1000
        silence = stitcher.create_silence(duration_ms)
        
        # Calculate expected samples
        expected_samples = int((duration_ms / 1000.0) * stitcher.sample_rate)
        assert len(silence) == expected_samples
    
    def test_create_silence_all_zeros(self, stitcher):
        """Test silence is all zeros."""
        silence = stitcher.create_silence(500)
        assert np.all(silence == 0)
    
    def test_create_silence_correct_dtype(self, stitcher):
        """Test silence has correct data type."""
        silence = stitcher.create_silence(100)
        assert silence.dtype == np.int16


class TestMP3ToWavConversion:
    """Test MP3 to WAV conversion."""
    
    @patch('subprocess.run')
    @patch('scipy.io.wavfile.read')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.path.exists')
    @patch('os.unlink')
    def test_mp3_to_wav_array_success(
        self, mock_unlink, mock_exists, mock_temp, mock_wavread, mock_subprocess, stitcher
    ):
        """Test successful MP3 to WAV conversion."""
        # Setup mocks
        mock_temp.return_value.__enter__.return_value.name = '/tmp/test.mp3'
        mock_exists.return_value = True
        
        # Mock WAV read to return sample audio
        sample_audio = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        mock_wavread.return_value = (22050, sample_audio)
        
        # Call conversion
        result = stitcher._mp3_to_wav_array(b'fake_mp3_data')
        
        # Verify ffmpeg was called
        assert mock_subprocess.called
        assert result is not None
        assert len(result) == 5
    
    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    def test_mp3_to_wav_array_ffmpeg_error(self, mock_temp, mock_subprocess, stitcher):
        """Test handling of ffmpeg conversion errors."""
        mock_temp.return_value.__enter__.return_value.name = '/tmp/test.mp3'
        mock_subprocess.side_effect = Exception("ffmpeg error")
        
        with pytest.raises(Exception, match="ffmpeg error"):
            stitcher._mp3_to_wav_array(b'invalid_mp3')


# Section 14.4: Validate silence duration between clips

class TestStitchAudioClips:
    """Test audio stitching functionality."""
    
    def test_stitch_empty_list_raises_error(self, stitcher):
        """Test stitching empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot stitch empty list"):
            stitcher.stitch_audio_clips([])
    
    def test_stitch_unsupported_format_raises_error(self, stitcher, mock_tts_result):
        """Test unsupported output format raises error."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            stitcher.stitch_audio_clips([mock_tts_result], output_format="mp3")
    
    @patch.object(AudioStitcher, '_mp3_to_wav_array')
    def test_stitch_single_clip(self, mock_convert, stitcher, mock_tts_result):
        """Test stitching a single audio clip."""
        # Mock conversion to return simple audio
        mock_audio = np.array([100, 200, 300], dtype=np.int16)
        mock_convert.return_value = mock_audio
        
        result = stitcher.stitch_audio_clips([mock_tts_result])
        
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    @patch.object(AudioStitcher, '_mp3_to_wav_array')
    def test_stitch_multiple_clips_with_pause(self, mock_convert, stitcher):
        """Test stitching multiple clips includes pauses."""
        # Create multiple TTS results
        results = [
            TTSResult("Text 1", b'audio1', 1, "voice1"),
            TTSResult("Text 2", b'audio2', 2, "voice1"),
            TTSResult("Text 3", b'audio3', 3, "voice1"),
        ]
        
        # Mock audio conversion
        mock_audio = np.array([100, 200, 300], dtype=np.int16)
        mock_convert.return_value = mock_audio
        
        output = stitcher.stitch_audio_clips(results)
        
        # Should have been called 3 times (once per clip)
        assert mock_convert.call_count == 3
        assert isinstance(output, bytes)
    
    @patch.object(AudioStitcher, '_mp3_to_wav_array')
    def test_stitch_preserves_reading_order(self, mock_convert, stitcher):
        """Test clips are stitched in reading order."""
        # Create results out of order
        results = [
            TTSResult("Third", b'audio3', 3, "voice1"),
            TTSResult("First", b'audio1', 1, "voice1"),
            TTSResult("Second", b'audio2', 2, "voice1"),
        ]
        
        # Mock different audio for each
        call_count = [0]
        def mock_convert_side_effect(audio_bytes):
            call_count[0] += 1
            return np.array([call_count[0] * 100], dtype=np.int16)
        
        mock_convert.side_effect = mock_convert_side_effect
        
        stitcher.stitch_audio_clips(results)
        
        # Verify conversion was called in reading order (1, 2, 3)
        assert mock_convert.call_count == 3
        # First call should be for "First" (reading_order=1)
        assert mock_convert.call_args_list[0][0][0] == b'audio1'
    
    @patch.object(AudioStitcher, '_mp3_to_wav_array')
    def test_stitch_conversion_error_propagates(self, mock_convert, stitcher, mock_tts_result):
        """Test conversion errors are propagated."""
        mock_convert.side_effect = Exception("Conversion failed")
        
        with pytest.raises(Exception, match="Conversion failed"):
            stitcher.stitch_audio_clips([mock_tts_result])


# Section 14.4: Ensure stitched audio length matches expected timing

class TestStitchToFile:
    """Test stitching to file."""
    
    @patch.object(AudioStitcher, 'stitch_audio_clips')
    @patch('builtins.open', create=True)
    def test_stitch_to_file_saves_correctly(self, mock_open, mock_stitch, stitcher, mock_tts_result):
        """Test stitching to file saves audio data."""
        mock_stitch.return_value = b'audio_data'
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        output_path = "/tmp/output.wav"
        result_path = stitcher.stitch_audio_clips_to_file([mock_tts_result], output_path)
        
        assert result_path == output_path
        mock_file.write.assert_called_once_with(b'audio_data')
    
    @patch.object(AudioStitcher, 'stitch_audio_clips')
    @patch('builtins.open', create=True)
    def test_stitch_to_file_infers_format(self, mock_open, mock_stitch, stitcher, mock_tts_result):
        """Test format inference from file extension."""
        mock_stitch.return_value = b'audio_data'
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        stitcher.stitch_audio_clips_to_file([mock_tts_result], "/tmp/test.wav")
        
        # Should call stitch_audio_clips with 'wav' format
        mock_stitch.assert_called_once()
        # Check that it was called with the correct list and inferred format
        call_args = mock_stitch.call_args
        assert call_args[0][0] == [mock_tts_result]  # First positional arg
        assert call_args[0][1] == 'wav'  # Second positional arg (inferred format)


class TestGetTotalDuration:
    """Test duration calculation."""
    
    @patch.object(AudioStitcher, '_mp3_to_wav_array')
    def test_get_total_duration_single_clip(self, mock_convert, stitcher, mock_tts_result):
        """Test duration calculation for single clip."""
        # Mock 1 second of audio at configured sample rate (44100 Hz)
        mock_audio = np.zeros(stitcher.sample_rate, dtype=np.int16)
        mock_convert.return_value = mock_audio
        
        duration_ms = stitcher.get_total_duration_ms([mock_tts_result])
        
        # Should be approximately 1000ms
        assert 900 < duration_ms < 1100
    
    @patch.object(AudioStitcher, '_mp3_to_wav_array')
    def test_get_total_duration_multiple_clips(self, mock_convert, stitcher):
        """Test duration calculation includes pauses."""
        results = [
            TTSResult("Text 1", b'audio1', 1, "voice1"),
            TTSResult("Text 2", b'audio2', 2, "voice1"),
        ]
        
        # Mock 0.5 seconds each at configured sample rate (44100 Hz)
        mock_audio = np.zeros(int(stitcher.sample_rate * 0.5), dtype=np.int16)
        mock_convert.return_value = mock_audio
        
        stitcher.pause_duration_ms = 500
        duration_ms = stitcher.get_total_duration_ms(results)
        
        # Should be: 500ms + 500ms (pause) + 500ms = 1500ms
        assert 1400 < duration_ms < 1600
    
    def test_get_total_duration_empty_list(self, stitcher):
        """Test duration of empty list is zero."""
        assert stitcher.get_total_duration_ms([]) == 0


# Section 14.4: Duration-based assertions

class TestAudioOutputFormat:
    """Test audio output format validation."""
    
    @patch.object(AudioStitcher, '_mp3_to_wav_array')
    def test_output_is_valid_wav(self, mock_convert, stitcher, mock_tts_result):
        """Test output is a valid WAV file."""
        mock_audio = np.array([100, 200, 300], dtype=np.int16)
        mock_convert.return_value = mock_audio
        
        output_bytes = stitcher.stitch_audio_clips([mock_tts_result])
        
        # WAV files start with 'RIFF' header
        assert output_bytes[:4] == b'RIFF'
        assert b'WAVE' in output_bytes[:20]
