"""
Synthetic ECG signal and packet generation for testing and simulation.
"""

import numpy as np

from polar_ecg.models import RawFrame


def synth_ecg(
    duration_s: float = 60.0,
    fs: float = 130.0,
    hr: float = 70.0,
    hrv_ms: float = 40.0,
    noise_uv: float = 20.0,
    wander_uv: float = 300.0,
    mains_hz: float = 50.0,
    invert: bool = False,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic ECG signal with realistic P-Q-R-S-T morphology,
    heart rate variability, baseline wander, mains noise, and sensor noise.

    Returns:
        (voltages_uv, true_r_indices)
    """
    rng = np.random.default_rng(seed)
    total_samples = int(duration_s * fs)
    t = np.arange(total_samples) / fs

    # Generate beat times with HRV
    mean_rr_s = 60.0 / hr
    std_rr_s = hrv_ms / 1000.0

    beat_times = []
    current_t = 0.5  # Start after brief pause
    while current_t < duration_s - 0.2:
        beat_times.append(current_t)
        # Sample next RR interval (clamped to physiological range)
        rr = rng.normal(mean_rr_s, std_rr_s)
        rr = np.clip(rr, 0.35, 1.8)
        current_t += rr

    beat_times = np.array(beat_times)
    true_r_indices = np.round(beat_times * fs).astype(int)
    true_r_indices = true_r_indices[true_r_indices < total_samples]

    # Gaussian wave components: (amplitude_uV, width_s, offset_s)
    components = [
        (150.0, 0.045, -0.16),   # P wave
        (-150.0, 0.020, -0.04),  # Q wave
        (1200.0, 0.025, 0.00),   # R peak
        (-350.0, 0.025, 0.035),  # S wave
        (250.0, 0.070, 0.22),    # T wave
    ]

    ecg_clean = np.zeros(total_samples, dtype=np.float64)
    for r_idx in true_r_indices:
        r_time = r_idx / fs
        for amp, width, offset in components:
            center = r_time + offset
            # Truncate Gaussian at 5 * sigma (relative contribution < 0.0003)
            half_width = 5.0 * width
            i_start = max(0, int((center - half_width) * fs))
            i_end = min(total_samples, int((center + half_width) * fs) + 1)
            if i_start < i_end:
                t_slice = t[i_start:i_end]
                wave = amp * np.exp(-0.5 * ((t_slice - center) / width) ** 2)
                ecg_clean[i_start:i_end] += wave

    if invert:
        ecg_clean = -ecg_clean

    # Baseline wander (respiration ~0.25 Hz + slow postural sway ~0.05 Hz)
    wander = (
        wander_uv * 0.7 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
        + wander_uv * 0.3 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))
    )

    # Mains interference (50 Hz / 60 Hz)
    mains = 15.0 * np.sin(2 * np.pi * mains_hz * t) if mains_hz > 0 else 0.0

    # High frequency / sensor noise
    noise = rng.normal(0, noise_uv, size=total_samples) if noise_uv > 0 else 0.0

    ecg_total = ecg_clean + wander + mains + noise
    return ecg_total.astype(np.float32), true_r_indices


def synth_frames(
    x_uv: np.ndarray,
    fs: float = 130.0,
    chunk_size: int = 73,
    start_device_ns: int = 1_000_000_000_000,
    start_mono_ns: int = 50_000_000_000,
    jitter_ms: float = 20.0,
    drop_rate: float = 0.0,
    reset_at_s: float | None = None,
    seed: int | None = 42,
) -> list[RawFrame]:
    """
    Slice continuous signal into RawFrame packets mimicking Polar H10 BLE transmission.
    """
    rng = np.random.default_rng(seed)
    frames = []
    n_samples = len(x_uv)
    dt_ns = round(1e9 / fs)

    device_time_offset = 0
    reset_applied = False

    for i in range(0, n_samples, chunk_size):
        chunk = x_uv[i : i + chunk_size]
        if len(chunk) == 0:
            continue

        chunk_end_idx = i + len(chunk) - 1
        elapsed_s = chunk_end_idx / fs

        # Device clock reset check (jump backwards)
        if reset_at_s is not None and elapsed_s >= reset_at_s and not reset_applied:
            device_time_offset = -int(3 * 1e9)  # Jump backwards by 3 seconds
            reset_applied = True

        device_last_ns = start_device_ns + (chunk_end_idx * dt_ns) + device_time_offset

        # Packet arrival host time with latency jitter
        base_arrival_mono = start_mono_ns + (chunk_end_idx * dt_ns)
        jitter_ns = int(rng.uniform(5.0, 5.0 + jitter_ms) * 1e6)  # Minimum 5ms transmission latency
        rx_mono_ns = base_arrival_mono + jitter_ns

        # Simulated frame drop
        if drop_rate > 0 and rng.uniform(0, 1) < drop_rate:
            continue

        frames.append(
            RawFrame(
                device_last_ns=device_last_ns,
                voltages_uv=chunk.copy(),
                rx_mono_ns=rx_mono_ns,
            )
        )

    return frames
