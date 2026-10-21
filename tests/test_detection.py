"""
Unit tests for streaming and offline R-peak detection.
"""

import numpy as np
import pytest

from polar_ecg.detection import RPeakDetector, detect_offline
from polar_ecg.settings import Settings
from polar_ecg.synth import synth_ecg


def evaluate_detection(detected_indices: list[float], true_indices: np.ndarray, tol_samples: int = 6):
    """
    Calculate sensitivity (recall) and positive predictive value (precision)
    given true and detected peak sample indices within a tolerance window.
    """
    if len(detected_indices) == 0:
        return 0.0, 0.0, 0.0

    det_arr = np.array(detected_indices)
    matched_true = set()
    matched_det = set()
    errors = []

    for det_i, det in enumerate(det_arr):
        dists = np.abs(true_indices - det)
        closest_idx = np.argmin(dists)
        if dists[closest_idx] <= tol_samples:
            matched_det.add(det_i)
            matched_true.add(closest_idx)
            errors.append(dists[closest_idx])

    tp = len(matched_true)
    fn = len(true_indices) - tp
    fp = len(det_arr) - len(matched_det)

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    median_err_samples = np.median(errors) if errors else float("inf")

    return se, ppv, median_err_samples


def test_detection_clean_ecg():
    fs = 130.0
    sig, true_r = synth_ecg(duration_s=60.0, fs=fs, hr=75.0, noise_uv=5.0, wander_uv=50.0, seed=10)

    # Discard warm-up period (first 3s)
    warmup_idx = int(3.0 * fs)
    eval_true_r = true_r[true_r >= warmup_idx]

    peaks = detect_offline(sig, fs=fs)
    eval_peaks = [p.sample_index for p in peaks if p.sample_index >= warmup_idx]

    se, ppv, med_err = evaluate_detection(eval_peaks, eval_true_r, tol_samples=6)
    assert se >= 0.99, f"Sensitivity too low: {se:.3f}"
    assert ppv >= 0.99, f"PPV too low: {ppv:.3f}"
    # Error in ms (1 sample at 130Hz is ~7.69ms; parabolic interpolation should give sub-sample accuracy)
    med_err_ms = (med_err / fs) * 1000.0
    assert med_err_ms < 4.0, f"Timing error too high: {med_err_ms:.2f} ms"


def test_detection_noise_and_wander():
    fs = 130.0
    # Heavy baseline wander (300uV) and noise
    sig, true_r = synth_ecg(
        duration_s=60.0, fs=fs, hr=70.0, noise_uv=25.0, wander_uv=300.0, mains_hz=50.0, seed=11
    )

    warmup_idx = int(3.0 * fs)
    eval_true_r = true_r[true_r >= warmup_idx]

    peaks = detect_offline(sig, fs=fs)
    eval_peaks = [p.sample_index for p in peaks if p.sample_index >= warmup_idx]

    se, ppv, _ = evaluate_detection(eval_peaks, eval_true_r, tol_samples=7)
    assert se >= 0.98, f"Sensitivity under noise too low: {se:.3f}"
    assert ppv >= 0.98, f"PPV under noise too low: {ppv:.3f}"


def test_detection_inverted_polarity():
    fs = 130.0
    # Inverted lead (negative R wave)
    sig, true_r = synth_ecg(duration_s=40.0, fs=fs, hr=80.0, invert=True, seed=12)

    warmup_idx = int(3.0 * fs)
    eval_true_r = true_r[true_r >= warmup_idx]

    peaks = detect_offline(sig, fs=fs)
    eval_peaks = [p.sample_index for p in peaks if p.sample_index >= warmup_idx]

    se, ppv, _ = evaluate_detection(eval_peaks, eval_true_r, tol_samples=6)
    assert se >= 0.98, f"Inverted sensitivity too low: {se:.3f}"
    assert ppv >= 0.98, f"Inverted PPV too low: {ppv:.3f}"


def test_detection_streaming_matches_offline():
    fs = 130.0
    sig, _ = synth_ecg(duration_s=30.0, fs=fs, hr=72.0, seed=13)

    # Offline
    offline_peaks = detect_offline(sig, fs=fs)

    # Streaming in chunks of 73 samples
    detector = RPeakDetector(fs=fs, settings=Settings())
    streaming_peaks = []
    chunk_size = 73

    for i in range(0, len(sig), chunk_size):
        chunk = sig[i : i + chunk_size]
        new_peaks = detector.process(
            first_index=i,
            x_uv=chunk,
            index_to_t=lambda idx: idx / fs,
        )
        streaming_peaks.extend(new_peaks)

    # Ignore first 3 seconds (warm-up)
    off_filt = [p.sample_index for p in offline_peaks if p.sample_index > 3.0 * fs]
    stream_filt = [p.sample_index for p in streaming_peaks if p.sample_index > 3.0 * fs]

    assert len(off_filt) == len(stream_filt)
    for p_off, p_str in zip(off_filt, stream_filt):
        assert abs(p_off - p_str) < 0.2


def test_detection_restart_regression():
    fs = 130.0
    sig, _true_r = synth_ecg(duration_s=10.0, fs=fs, hr=75.0, seed=14)
    detector = RPeakDetector(fs=fs, settings=Settings())

    # Stream 5 seconds
    mid_idx = int(5.0 * fs)
    for i in range(0, mid_idx, 73):
        chunk = sig[i : min(i + 73, mid_idx)]
        detector.process(first_index=i, x_uv=chunk, index_to_t=lambda idx: idx / fs)

    # Restart stream
    detector.reset()

    # Stream from index 0 again
    restarted_peaks = []
    for i in range(0, mid_idx, 73):
        chunk = sig[i : min(i + 73, mid_idx)]
        new_peaks = detector.process(first_index=i, x_uv=chunk, index_to_t=lambda idx: idx / fs)
        restarted_peaks.extend(new_peaks)

    # Must detect peaks in the restarted stream after warm-up
    warmup_idx = int(2.5 * fs)
    valid_restarted = [p for p in restarted_peaks if p.sample_index > warmup_idx]
    assert len(valid_restarted) > 2, "No peaks detected after restart!"


def test_detection_gap_notification():
    fs = 130.0
    sig, _ = synth_ecg(duration_s=15.0, fs=fs, hr=75.0, seed=15)
    detector = RPeakDetector(fs=fs, settings=Settings())

    # Process first 5 seconds
    gap_idx = int(5.0 * fs)
    for i in range(0, gap_idx, 73):
        chunk = sig[i : min(i + 73, gap_idx)]
        detector.process(first_index=i, x_uv=chunk, index_to_t=lambda idx: idx / fs)

    # Notify gap of 5 seconds (650 missing samples)
    detector.notify_gap()

    # Process from 10 seconds onwards
    resumed_peaks = []
    resume_start = gap_idx + 650
    for i in range(resume_start, len(sig) + 650, 73):
        orig_i = i - 650
        chunk = sig[orig_i : min(orig_i + 73, len(sig))]
        new_peaks = detector.process(first_index=i, x_uv=chunk, index_to_t=lambda idx: idx / fs)
        resumed_peaks.extend(new_peaks)

    # First peak after gap must have flag 'after_gap' and rri_ms must be None
    assert len(resumed_peaks) > 0
    first_after_gap = resumed_peaks[0]
    assert "after_gap" in first_after_gap.flags
    assert first_after_gap.rri_ms is None


def test_detection_neurokit_agreement():
    nk2 = pytest.importorskip("neurokit2")
    fs = 130.0
    sig, _ = synth_ecg(duration_s=30.0, fs=fs, hr=70.0, noise_uv=10.0, wander_uv=100.0, seed=16)

    # Neurokit2 detection
    _, rpeaks_nk = nk2.ecg_peaks(sig, sampling_rate=int(fs))
    nk_r_indices = rpeaks_nk["ECG_R_Peaks"]

    # Our detection
    our_peaks = detect_offline(sig, fs=fs)
    our_indices = [p.sample_index for p in our_peaks]

    # Evaluate overlap after 3s warmup
    warmup_idx = int(3.0 * fs)
    nk_filt = nk_r_indices[nk_r_indices >= warmup_idx]
    our_filt = [idx for idx in our_indices if idx >= warmup_idx]

    se, ppv, _ = evaluate_detection(our_filt, nk_filt, tol_samples=6)
    assert se >= 0.95
    assert ppv >= 0.95
