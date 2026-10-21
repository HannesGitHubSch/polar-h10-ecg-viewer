"""
Unit tests for ClockSync and SampleClock timing algorithms.
"""

import numpy as np
import pytest

from polar_ecg.synth import synth_ecg, synth_frames
from polar_ecg.timing import ClockSync, SampleClock


def test_timing_continuous_frames():
    clock = SampleClock(fs_nominal=130.0)
    sig, _ = synth_ecg(duration_s=5.0, fs=130.0, seed=1)
    frames = synth_frames(sig, fs=130.0, chunk_size=73, jitter_ms=10.0, drop_rate=0.0)

    total_samples = 0
    for frame in frames:
        timed = clock.process(frame)
        assert timed.first_index == total_samples
        assert timed.gap_before == 0
        assert len(timed.voltages_uv) == len(frame.voltages_uv)
        assert len(timed.t_session_s) == len(frame.voltages_uv)
        assert len(timed.device_ns) == len(frame.voltages_uv)
        assert len(timed.host_utc_ns) == len(frame.voltages_uv)
        total_samples += len(frame.voltages_uv)

    assert clock.total_missing == 0


def test_timing_jitter_free_sample_intervals():
    clock = SampleClock(fs_nominal=130.0)
    sig, _ = synth_ecg(duration_s=3.0, fs=130.0, seed=2)
    # Heavy host arrival jitter (10ms to 90ms)
    frames = synth_frames(sig, fs=130.0, chunk_size=73, jitter_ms=80.0, drop_rate=0.0)

    all_t_session = []
    for frame in frames:
        timed = clock.process(frame)
        all_t_session.extend(timed.t_session_s)

    dt = np.diff(all_t_session)
    expected_dt = 1.0 / 130.0
    # Jitter should not affect dt; deviation should be < 1 microsecond
    np.testing.assert_allclose(dt, expected_dt, atol=1e-5)


def test_timing_dropped_frame_gap_detection():
    clock = SampleClock(fs_nominal=130.0)
    sig, _ = synth_ecg(duration_s=5.0, fs=130.0, seed=3)
    frames = synth_frames(sig, fs=130.0, chunk_size=73, jitter_ms=10.0, drop_rate=0.0)

    # Process frame 0, 1
    clock.process(frames[0])
    t1 = clock.process(frames[1])
    assert t1.gap_before == 0

    # Drop frame 2 (73 samples), process frame 3
    t3 = clock.process(frames[3])
    assert t3.gap_before == len(frames[2].voltages_uv)  # 73 samples
    assert t3.first_index == len(frames[0].voltages_uv) + len(frames[1].voltages_uv) + len(frames[2].voltages_uv)
    assert clock.total_missing == 73


def test_timing_device_clock_reset_discontinuity():
    clock = SampleClock(fs_nominal=130.0)
    sig, _ = synth_ecg(duration_s=10.0, fs=130.0, seed=4)
    # Reset device clock backwards at 5s
    frames = synth_frames(sig, fs=130.0, chunk_size=73, jitter_ms=5.0, reset_at_s=5.0)

    all_t_session = []
    discontinuity_seen = False

    for frame in frames:
        timed = clock.process(frame)
        if timed.gap_before == -1:
            discontinuity_seen = True
        all_t_session.extend(timed.t_session_s)

    assert discontinuity_seen, "Clock reset was not detected as discontinuity"
    # Even across clock reset, session timeline MUST remain monotonic
    diffs = np.diff(all_t_session)
    assert np.all(diffs > 0), "Session time went backwards or stalled"


def test_timing_fs_estimation():
    # Nominal 130 Hz, true 130.05 Hz
    true_fs = 130.05
    clock = SampleClock(fs_nominal=130.0)
    sig, _ = synth_ecg(duration_s=20.0, fs=true_fs, seed=5)
    frames = synth_frames(sig, fs=true_fs, chunk_size=73, jitter_ms=10.0)

    timed = None
    for frame in frames:
        timed = clock.process(frame)

    assert timed is not None
    # Estimated fs should converge within ±0.05 Hz of true_fs
    assert abs(timed.fs_est - true_fs) < 0.05


def test_timing_clock_sync_bounds():
    sync = ClockSync()
    anchor_mono = 100_000_000_000
    anchor_utc = 1_700_000_000_000_000_000
    sync.set_anchors(anchor_mono, anchor_utc)

    # Device timestamps: 0, 1e9, 2e9 ns (1s intervals)
    # Transmission delay min is 10ms, max is 50ms
    sync.update(device_ns=0, rx_mono_ns=anchor_mono + 25_000_000)
    sync.update(device_ns=1_000_000_000, rx_mono_ns=anchor_mono + 1_000_000_000 + 10_000_000)  # min delay
    sync.update(device_ns=2_000_000_000, rx_mono_ns=anchor_mono + 2_000_000_000 + 40_000_000)

    # Min latency offset is 10ms
    utc_mapped = sync.to_utc(np.array([1_000_000_000], dtype=np.int64))[0]
    expected_utc = anchor_utc + 1_000_000_000 + 10_000_000
    assert abs(utc_mapped - expected_utc) <= 1_000_000  # Within 1ms


def test_timing_reset():
    clock = SampleClock(fs_nominal=130.0)
    sig, _ = synth_ecg(duration_s=2.0, fs=130.0, seed=6)
    frames = synth_frames(sig, fs=130.0, chunk_size=73)

    clock.process(frames[0])
    assert clock.total_missing == 0
    clock.reset()

    # After reset, next frame should start at first_index=0 again
    t_new = clock.process(frames[0])
    assert t_new.first_index == 0
    assert t_new.t_session_s[0] == pytest.approx(0.0, abs=1e-6)
