"""Unit tests for the per-request stage timer."""

import pytest

from backend.services import timing
from backend.services.timing import StageTimer


@pytest.fixture
def fake_clock(monkeypatch):
    """Controllable perf_counter so durations are exact."""
    now = [100.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: now[0])
    return now


@pytest.mark.unit
def test_laps_record_time_since_previous_lap(fake_clock):
    timer = StageTimer()
    fake_clock[0] += 0.5
    assert timer.lap("read") == pytest.approx(0.5)
    fake_clock[0] += 2.0
    timer.lap("ocr")
    assert timer.stages == {"read": pytest.approx(0.5), "ocr": pytest.approx(2.0)}
    assert timer.total == pytest.approx(2.5)


@pytest.mark.unit
def test_repeated_stage_accumulates(fake_clock):
    timer = StageTimer()
    fake_clock[0] += 1.0
    timer.lap("tts")
    fake_clock[0] += 0.5
    timer.lap("tts")
    assert timer.stages["tts"] == pytest.approx(1.5)


@pytest.mark.unit
def test_summary_and_server_timing_header(fake_clock):
    timer = StageTimer()
    fake_clock[0] += 0.25
    timer.lap("read")
    fake_clock[0] += 1.5
    timer.lap("ocr")
    assert timer.summary() == "read=0.25s ocr=1.50s total=1.75s"
    assert timer.server_timing_header() == "read;dur=250, ocr;dur=1500, total;dur=1750"


@pytest.mark.unit
def test_no_laps_still_reports_total(fake_clock):
    timer = StageTimer()
    fake_clock[0] += 3.0
    assert timer.summary() == "total=3.00s"
    assert timer.server_timing_header() == "total;dur=3000"
