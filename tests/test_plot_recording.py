"""
Tests for plot_recording analysis script and overwrite protection.
"""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # Headless backend for tests

from polar_ecg.analysis.plot_recording import analyze_and_plot
from polar_ecg.recording import Recorder
from polar_ecg.settings import Settings
from polar_ecg.synth import synth_ecg, synth_frames
from polar_ecg.timing import SampleClock


def generate_test_recording(tmp_path: Path, filename: str = "ECG_TEST_20261021_120000") -> Path:
    base_path = tmp_path / filename
    rec = Recorder(Settings())
    rec.start(base_path=str(base_path), participant_id="P_TEST")

    clock = SampleClock(130.0)
    sig, _ = synth_ecg(duration_s=5.0, fs=130.0, seed=1)
    frames = synth_frames(sig, fs=130.0, chunk_size=73)

    for f in frames:
        timed = clock.process(f)
        rec.submit_frame(timed)

    summary = rec.stop()
    return summary.file_path


def test_plot_recording_generation(tmp_path: Path):
    csv_path = generate_test_recording(tmp_path)
    out_png = tmp_path / "custom_plot.png"

    fig = analyze_and_plot(csv_path, out_path=out_png, show=False)
    assert fig is not None
    assert out_png.exists()
    assert out_png.stat().st_size > 1000


def test_plot_recording_overwrite_protection(tmp_path: Path):
    csv_path = generate_test_recording(tmp_path)
    # Attempt to overwrite the input CSV with the plot
    with pytest.raises(ValueError, match="matches input CSV path"):
        analyze_and_plot(csv_path, out_path=csv_path, show=False)


def test_plot_recording_empty_file_guard(tmp_path: Path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("sample_index,t_session_s,ecg_uv,host_time_utc,gap_before,event\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty or has fewer than 2 samples"):
        analyze_and_plot(empty_csv, show=False)
