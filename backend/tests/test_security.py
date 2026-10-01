"""Tests for API-key auth, rate limiting and request validation."""

import io

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from backend.config import Settings, settings
from backend.main import app
from backend.security import (
    ANONYMOUS,
    RateLimiter,
    enforce_request_limits,
    rate_limiter,
    require_api_key,
)


@pytest.fixture
def real_limits():
    """Use the real auth/rate-limit dependency (conftest disables it by default)."""
    app.dependency_overrides.pop(enforce_request_limits, None)
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def png():
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buf, "PNG")
    return buf.getvalue()


def post(client, headers=None, chapter_id="ch1"):
    return client.post("/process/chapter", data={"chapter_id": chapter_id},
                       files=[("images", ("p.png", io.BytesIO(png()), "image/png"))], headers=headers or {})


# require_api_key

@pytest.mark.unit
def test_valid_key_accepted(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", [SecretStr("good-key")])
    assert require_api_key("good-key") == "good-key"


@pytest.mark.unit
@pytest.mark.parametrize("key", [None, "", "wrong-key", "good-ke"])
def test_missing_or_wrong_key_rejected(monkeypatch, key):
    monkeypatch.setattr(settings, "API_KEYS", [SecretStr("good-key")])
    with pytest.raises(HTTPException) as exc:
        require_api_key(key)
    assert exc.value.status_code == 401


@pytest.mark.unit
def test_no_keys_open_only_in_debug(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", [])
    monkeypatch.setattr(settings, "DEBUG", True)
    assert require_api_key(None) == ANONYMOUS
    monkeypatch.setattr(settings, "DEBUG", False)
    with pytest.raises(HTTPException) as exc:
        require_api_key(None)
    assert exc.value.status_code == 503


@pytest.mark.unit
def test_api_keys_parse_from_env_and_stay_masked(monkeypatch):
    monkeypatch.setenv("API_KEYS", '["k1", "k2"]')
    s = Settings()
    assert [k.get_secret_value() for k in s.API_KEYS] == ["k1", "k2"]
    assert "k1" not in repr(s.API_KEYS)


@pytest.mark.unit
def test_elevenlabs_key_masked():
    s = Settings(ELEVENLABS_API_KEY="sk_secret_value")
    assert "sk_secret_value" not in repr(s)
    assert s.ELEVENLABS_API_KEY.get_secret_value() == "sk_secret_value"


# RateLimiter

@pytest.mark.unit
def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check("client", 3)
    with pytest.raises(HTTPException) as exc:
        limiter.check("client", 3)
    assert exc.value.status_code == 429
    assert int(exc.value.headers["Retry-After"]) >= 1
    limiter.check("other-client", 3)          # separate budget per client


@pytest.mark.unit
def test_rate_limiter_window_expires(monkeypatch):
    import backend.security as sec
    now = [1000.0]
    monkeypatch.setattr(sec.time, "monotonic", lambda: now[0])
    limiter = RateLimiter()
    limiter.check("c", 1)
    now[0] += 61
    limiter.check("c", 1)                     # previous hit fell out of the window


@pytest.mark.unit
def test_rate_limiter_zero_disables():
    limiter = RateLimiter()
    for _ in range(100):
        limiter.check("c", 0)


# Through the real endpoint

@pytest.mark.integration
def test_endpoint_requires_key(real_limits, monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", [SecretStr("good-key")])
    client = TestClient(app)
    assert post(client).status_code == 401
    assert post(client, {"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.integration
def test_endpoint_rate_limited_per_key(real_limits, monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", [SecretStr("good-key")])
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 2)
    client = TestClient(app)
    # Auth passes; these fail later (bad chapter id), but each still counts against the limit
    codes = [post(client, {"X-API-Key": "good-key"}, chapter_id="bad id!").status_code for _ in range(3)]
    assert codes == [400, 400, 429]


@pytest.mark.integration
@pytest.mark.parametrize("chapter_id", ["나혼렙-1", 'a"; filename="x.exe', "x\r\nSet-Cookie: s=1", "", "a" * 65])
def test_unsafe_chapter_id_rejected_before_processing(chapter_id):
    response = post(TestClient(app), chapter_id=chapter_id)
    assert response.status_code in (400, 422)


@pytest.mark.integration
def test_oversized_image_dimensions_rejected(monkeypatch):
    import backend.routers.process as process
    monkeypatch.setattr(process.settings, "MAX_IMAGE_PIXELS", 100)   # the 20x20 test image is 400 px
    response = post(TestClient(app))
    assert response.status_code == 400
    assert "too large" in response.json()["detail"]


@pytest.mark.unit
def test_client_error_detail_hides_exception_outside_debug(monkeypatch):
    import backend.routers.process as process
    monkeypatch.setattr(process.settings, "DEBUG", False)
    assert process.client_error_detail("OCR processing failed", RuntimeError("quota: 0 credits")) == "OCR processing failed"
    monkeypatch.setattr(process.settings, "DEBUG", True)
    assert "quota" in process.client_error_detail("OCR processing failed", RuntimeError("quota: 0 credits"))
