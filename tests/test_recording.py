"""
Unit tests for Recorder: CSV v2 writer, sidecar metadata, and auxiliary files.
"""

import json
from pathlib import Path

import numpy as np

from polar_ecg.models import DeviceHR, DeviceInfo, Event, RPeak, TimedFrame
from polar_ecg.recording import Recorder, RecordingSummary
from polar_ecg.settings import Settings


def make_timed_frame(
    first_index: int,
    n_samples: int = 5,
    gap_before: int = 0,
    base_t: float = 0.0,
    base_utc_ns: int = 1_700_000_000_000_000_000,
) -> TimedFrame:
    voltages = np.array([100.0 + i for i in range(n_samples)], dtype=np.float32)
    device_ns = np.array([base_utc_ns + i * 7_692_308 for i in range(n_samples)], dtype=np.int64)
    t_session = np.array([base_t + i * (1.0 / 130.0) for i in range(n_samples)], dtype=np.float64)
    host_utc = device_ns.copy()
    return TimedFrame(
        first_index=first_index,
        voltages_uv=voltages,
        device_ns=device_ns,
        t_session_s=t_session,
        host_utc_ns=host_utc,
        packet_rx_utc_ns=base_utc_ns + 15_000_000,
        gap_before=gap_before,
        fs_est=130.0,
    )


def test_recorder_lifecycle_and_files(tmp_path: Path):
    rec = Recorder(Settings())
    base_path = tmp_path / "ECG_TEST_20261021_120000"
    dev_info = DeviceInfo(name="Polar H10 ABCD", address="00:11:22:33:44:55")

    rec.start(base_path=str(base_path), participant_id="P001", device_info=dev_info)
    assert rec.is_recording

    # Check that partial CSV and meta.json exist
    partial_csv = tmp_path / "ECG_TEST_20261021_120000.csv.partial"
    meta_json = tmp_path / "ECG_TEST_20261021_120000_meta.json"
    assert partial_csv.exists()
    assert meta_json.exists()

    # Submit a frame
    frame1 = make_timed_frame(first_index=0, n_samples=5, base_t=0.0)
    rec.submit_frame(frame1)

    # Add an event
    rec.add_event(Event(t_session_s=0.02, host_utc_ns=1_700_000_000_000_000_000, label="stim_start"))

    # Submit second frame with a gap
    frame2 = make_timed_frame(first_index=10, n_samples=5, gap_before=5, base_t=0.0769)
    rec.submit_frame(frame2)

    # Submit a detected peak
    rec.submit_peak(
        RPeak(
            sample_index=2.0,
            t_session_s=0.015,
            voltage_uv=1200.0,
            rri_ms=800.0,
            bpm=75.0,
            flags=frozenset(["searchback"]),
        )
    )

    # Submit native HR
    rec.submit_hr(
        DeviceHR(
            rx_utc_ns=1_700_000_000_000_000_000,
            hr_bpm=74,
            rr_ms=(812.5,),
        )
    )

    summary = rec.stop()
    assert not rec.is_recording
    assert isinstance(summary, RecordingSummary)
    assert summary.samples_written == 10
    assert summary.samples_missing == 5

    # Check that partial is renamed to final .csv
    final_csv = tmp_path / "ECG_TEST_20261021_120000.csv"
    assert final_csv.exists()
    assert not partial_csv.exists()

    # Check beats CSV and HR CSV exist
    beats_csv = tmp_path / "ECG_TEST_20261021_120000_beats.csv"
    hr_csv = tmp_path / "ECG_TEST_20261021_120000_hr.csv"
    assert beats_csv.exists()
    assert hr_csv.exists()

    # Verify CSV v2 content
    content = final_csv.read_text(encoding="utf-8").strip().splitlines()
    header = content[0].split(",")
    expected_header = [
        "sample_index",
        "t_session_s",
        "device_time_ns",
        "host_time_utc",
        "packet_rx_utc",
        "ecg_uv",
        "gap_before",
        "event",
    ]
    assert header == expected_header
    assert len(content) == 11  # Header + 10 samples

    # Check that event was stamped
    events_found = [line for line in content[1:] if "stim_start" in line]
    assert len(events_found) == 1

    # Check meta.json
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["format"] == "polar-ecg-csv"
    assert meta["format_version"] == 2
    assert meta["participant_id"] == "P001"
    assert meta["device"]["name"] == "Polar H10 ABCD"
    assert meta["samples_written"] == 10
    assert meta["samples_missing"] == 5
    assert len(meta["events"]) == 1
    assert meta["events"][0]["label"] == "stim_start"


def test_recorder_error_callback(tmp_path: Path):
    errors = []
    rec = Recorder(Settings(), error_callback=lambda err: errors.append(err))

    # Invalid path containing illegal characters
    rec.start(base_path="Z:\\invalid_drive_dir_nonexistent\\file", participant_id="P1")
    frame = make_timed_frame(0)
    rec.submit_frame(frame)
    rec.stop()

    assert len(errors) > 0, "Error callback was not notified on writer error"
