"""Tests for treating Google Vision and ElevenLabs responses as untrusted."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from elevenlabs import Voice, VoiceSettings
from elevenlabs.api.error import APIError, RateLimitError

from backend.services import tts
from backend.services.audio import AudioStitcher
from backend.services.tts import TTSResult, _is_retryable, generate, is_mp3
from backend.services.vision import GoogleVisionOCRService, is_noise_token

MP3 = b"ID3" + b"\x00" * 100
VOICE = Voice(voice_id="abc123", settings=VoiceSettings(stability=0.5, similarity_boost=0.75))


class FakeResponse:
    def __init__(self, status=200, body=MP3, content_type="audio/mpeg"):
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self._body = body

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def call(response):
    with patch.object(tts.requests, "post", return_value=response) as post:
        return generate("Hello", VOICE, "eleven_v4_turbo", "key"), post


# ElevenLabs

@pytest.mark.unit
def test_generate_returns_mp3_with_timeout_and_no_redirects():
    audio, post = call(FakeResponse())
    assert audio == MP3
    kwargs = post.call_args.kwargs
    assert kwargs["allow_redirects"] is False
    assert kwargs["stream"] is True
    assert kwargs["timeout"][1] > 0
    assert post.call_args.args[0].startswith("https://api.elevenlabs.io/")


@pytest.mark.unit
def test_generate_rejects_redirect():
    with pytest.raises(APIError) as exc:
        call(FakeResponse(status=302, body=b""))
    assert not _is_retryable(exc.value)


@pytest.mark.unit
def test_generate_rejects_oversized_audio(monkeypatch):
    monkeypatch.setattr(tts.settings, "TTS_MAX_AUDIO_BYTES", 50)
    with pytest.raises(APIError, match="larger than 50"):
        call(FakeResponse())


@pytest.mark.unit
@pytest.mark.parametrize("response", [
    FakeResponse(content_type="text/html"),
    FakeResponse(body=b"<html>not audio</html>"),
])
def test_generate_rejects_non_audio(response):
    with pytest.raises(APIError) as exc:
        call(response)
    assert not _is_retryable(exc.value)


@pytest.mark.unit
def test_generate_maps_quota_error_like_the_sdk():
    body = json.dumps({"detail": {"status": "quota_exceeded", "message": "out of credits"}}).encode()
    with pytest.raises(RateLimitError):
        call(FakeResponse(status=401, body=body, content_type="application/json"))


@pytest.mark.unit
def test_generate_server_error_is_retryable():
    with pytest.raises(APIError) as exc:
        call(FakeResponse(status=503, body=b'{"error": "busy"}', content_type="application/json"))
    assert _is_retryable(exc.value)


@pytest.mark.unit
@pytest.mark.parametrize("voice_id", ["../../v1/user", "abc?x=1", ""])
def test_generate_rejects_unsafe_voice_id(voice_id):
    with patch.object(tts.requests, "post") as post, pytest.raises(ValueError):
        generate("Hello", Voice(voice_id=voice_id), "m", "key")
    post.assert_not_called()


@pytest.mark.unit
def test_generate_rejects_huge_text():
    with patch.object(tts.requests, "post") as post, pytest.raises(ValueError):
        generate("a" * 5000, VOICE, "m", "key")
    post.assert_not_called()


@pytest.mark.unit
def test_is_mp3():
    assert is_mp3(b"ID3\x04") and is_mp3(b"\xff\xfb\x90")
    assert not is_mp3(b"") and not is_mp3(b"RIFF....WAVE") and not is_mp3(b"\xff\x00")


# Audio stitching

@pytest.mark.unit
def test_stitcher_refuses_non_mp3_clip():
    clip = TTSResult(text="x", audio_bytes=b"#!/bin/sh\n", reading_order=1, voice_id="v")
    with pytest.raises(ValueError, match="not MP3"):
        AudioStitcher(pause_duration_ms=0).stitch_audio_clips_with_duration([clip])


# Google Vision

def vision_service(annotations):
    service = GoogleVisionOCRService.__new__(GoogleVisionOCRService)
    service.client = MagicMock()
    service.client.text_detection.return_value = SimpleNamespace(
        error=SimpleNamespace(message=""), text_annotations=[None] + annotations
    )
    return service


def word(text, x=10, y=10):
    v = [SimpleNamespace(x=x, y=y), SimpleNamespace(x=x + 20, y=y + 10)]
    return SimpleNamespace(description=text, bounding_poly=SimpleNamespace(vertices=v))


@pytest.mark.unit
def test_vision_call_has_timeout():
    service = vision_service([word("Hi")])
    service.detect_text(b"img")
    assert service.client.text_detection.call_args.kwargs["timeout"] > 0


@pytest.mark.unit
def test_vision_word_count_capped(monkeypatch):
    import backend.services.vision as vision
    monkeypatch.setattr(vision.settings, "OCR_MAX_WORDS_PER_IMAGE", 5)
    assert len(vision_service([word("Hi")] * 50).detect_text(b"img")) == 5


@pytest.mark.unit
def test_vision_coordinates_clamped():
    result = vision_service([word("Hi", x=-500, y=10**12)]).detect_text(b"img")[0]
    assert result.bounding_box.left == 0
    assert result.bounding_box.bottom <= 100_000


@pytest.mark.unit
def test_overlong_word_is_noise():
    assert is_noise_token("A" * 101)
    assert not is_noise_token("Aaaaaaaaargh!")
