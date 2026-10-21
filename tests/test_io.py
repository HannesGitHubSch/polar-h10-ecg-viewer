"""
Unit tests for loading v1 and v2 recording formats.
"""

import json
from pathlib import Path

import numpy as np

from polar_ecg.analysis.io import Recording, load_recording


def test_load_v1_recording(tmp_path: Path):
    v1_csv = tmp_path / "ECG_P001_20260213_153045.csv"
    # Example format from repository README
    v1_content = """participant_id,sample_datetime,packet_received_datetime,polar_timestamp_s,ecg_voltage_uv
P001,2026-02-13T15:30:45.123456,2026-02-13T15:30:45.885456,0.000000,450.25
P001,2026-02-13T15:30:45.131148,2026-02-13T15:30:45.885456,0.007692,460.50
P001,2026-02-13T15:30:45.138840,2026-02-13T15:30:45.885456,0.015385,475.00
"""
    v1_csv.write_text(v1_content, encoding="utf-8")

    rec = load_recording(v1_csv)
    assert isinstance(rec, Recording)
    assert rec.format_version == 1
    assert rec.participant_id == "P001"
    assert len(rec.df) == 3

    # Check normalized columns
    assert "sample_index" in rec.df.columns
    assert "t_session_s" in rec.df.columns
    assert "ecg_uv" in rec.df.columns
    assert "host_time_utc" in rec.df.columns
    assert "gap_before" in rec.df.columns
    assert "event" in rec.df.columns

    np.testing.assert_array_equal(rec.df["sample_index"], [0, 1, 2])
    np.testing.assert_allclose(rec.df["t_session_s"], [0.0, 1.0 / 130.0, 2.0 / 130.0], atol=1e-5)
    np.testing.assert_allclose(rec.df["ecg_uv"], [450.25, 460.50, 475.00])


def test_load_v2_recording(tmp_path: Path):
    v2_csv = tmp_path / "ECG_P002_20261021_120000.csv"
    v2_content = """sample_index,t_session_s,device_time_ns,host_time_utc,packet_rx_utc,ecg_uv,gap_before,event
0,0.000000,1000000000000,2026-10-21T10:00:00.000000+00:00,2026-10-21T10:00:00.020000+00:00,120.50,0,
1,0.007692,1000007692308,2026-10-21T10:00:00.007692+00:00,2026-10-21T10:00:00.020000+00:00,135.20,0,event_a
"""
    v2_csv.write_text(v2_content, encoding="utf-8")

    # Write sidecar
    meta_json = tmp_path / "ECG_P002_20261021_120000_meta.json"
    meta_dict = {
        "format": "polar-ecg-csv",
        "format_version": 2,
        "participant_id": "P002",
        "fs_nominal": 130.0,
        "samples_written": 2,
    }
    meta_json.write_text(json.dumps(meta_dict), encoding="utf-8")

    rec = load_recording(v2_csv)
    assert rec.format_version == 2
    assert rec.participant_id == "P002"
    assert len(rec.df) == 2
    assert rec.df["event"].iloc[1] == "event_a"
    assert rec.meta["fs_nominal"] == 130.0
