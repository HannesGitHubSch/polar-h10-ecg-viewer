"""
Tests for Lab Streaming Layer (LSL) integration.
"""

import time

import pytest

pylsl = pytest.importorskip("pylsl")

from polar_ecg.lsl_out import LslOutletManager
from polar_ecg.models import DeviceHR, Event, RawFrame, RPeak
from polar_ecg.session import Session
from polar_ecg.settings import Settings
from polar_ecg.synth import synth_ecg, synth_frames
from polar_ecg.timing import SampleClock


def test_lsl_outlet_manager_lifecycle():
    mgr = LslOutletManager(device_name="Test_Polar", device_id="TEST_001")
    assert mgr.is_active

    clock = SampleClock(fs_nominal=130.0)
    sig, _ = synth_ecg(duration_s=2.0, fs=130.0)
    frames = synth_frames(sig, fs=130.0, chunk_size=73)

    for f in frames:
        timed = clock.process(f)
        mgr.push_frame(timed)

    mgr.push_event(Event(t_session_s=0.5, host_utc_ns=time.time_ns(), label="test_marker", source="test"))
    mgr.push_peak(RPeak(sample_index=130.0, t_session_s=1.0, voltage_uv=1200.0, rri_ms=800.0, bpm=75.0))
    mgr.push_device_hr(DeviceHR(rx_utc_ns=time.time_ns(), hr_bpm=74, rr_ms=(810.0,)))

    mgr.close()
    assert not mgr.is_active


def test_lsl_stream_resolution_and_pull():
    import uuid
    unique_id = f"test_{uuid.uuid4().hex[:8]}"
    mgr = LslOutletManager(device_name="Test_H10", device_id=unique_id)
    assert mgr.is_active

    clock = SampleClock(fs_nominal=130.0)
    sig, _ = synth_ecg(duration_s=1.0, fs=130.0)
    frames = synth_frames(sig, fs=130.0, chunk_size=73)

    # Resolve ECG stream
    streams = pylsl.resolve_byprop("source_id", f"ecg_{unique_id}", timeout=3.0)
    assert len(streams) >= 1

    inlet = pylsl.StreamInlet(streams[0])
    inlet.open_stream(timeout=2.0)

    # Push frames after inlet is connected
    for f in frames:
        timed = clock.process(f)
        mgr.push_frame(timed)

    chunk, timestamps = inlet.pull_chunk(timeout=2.0, max_samples=100)
    assert len(chunk) > 0
    assert len(timestamps) == len(chunk)
    # Check that timestamps are monotonically increasing
    for i in range(len(timestamps) - 1):
        assert timestamps[i + 1] > timestamps[i]

    inlet.close_stream()
    mgr.close()


def test_session_with_lsl():
    settings = Settings(LSL_ENABLED=True)
    session = Session(settings=settings, enable_lsl=True)

    sig, _ = synth_ecg(duration_s=0.5)
    raw = RawFrame(device_last_ns=1_000_000_000, voltages_uv=sig[:65], rx_mono_ns=time.monotonic_ns())

    # Before streaming, lsl is None
    assert session.lsl is None

    # Simulate start streaming
    session._is_streaming = True
    session._lsl = LslOutletManager(device_name="SimSession", device_id="SIM_LSL")
    assert session.lsl.is_active

    session.on_raw_frame(raw)
    session.add_event("lsl_event")
    session.on_device_hr(DeviceHR(rx_utc_ns=time.time_ns(), hr_bpm=72, rr_ms=(833.0,)))

    session._lsl.close()
    assert not session.lsl.is_active
