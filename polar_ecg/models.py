"""
Data models for Polar H10 ECG streaming, timing, detection, and events.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawFrame:
    """
    Raw ECG packet received from device or simulator.

    Attributes:
        device_last_ns: Polar device timestamp (nanoseconds) for the LAST sample in this packet.
        voltages_uv: 1D array of ECG voltages in microvolts (µV).
        rx_mono_ns: Host monotonic time (nanoseconds) recorded immediately at packet arrival.
    """

    device_last_ns: int
    voltages_uv: np.ndarray
    rx_mono_ns: int


@dataclass(frozen=True)
class TimedFrame:
    """
    ECG packet with synchronized timestamps and gap metadata.

    Attributes:
        first_index: Monotonically increasing continuous sample index for the first sample in packet.
        voltages_uv: 1D array of ECG voltages in µV (float32).
        device_ns: 1D array of per-sample device timestamps (int64).
        t_session_s: 1D array of elapsed session time in seconds derived from device clock.
        host_utc_ns: 1D array of per-sample host UTC timestamps (int64, ns since Unix epoch).
        packet_rx_utc_ns: Host UTC timestamp of packet arrival.
        gap_before: Number of missing samples immediately preceding this frame (0 = none, -1 = discontinuity).
        fs_est: Running least-squares estimate of effective sampling rate (Hz).
    """

    first_index: int
    voltages_uv: np.ndarray
    device_ns: np.ndarray
    t_session_s: np.ndarray
    host_utc_ns: np.ndarray
    packet_rx_utc_ns: int
    gap_before: int
    fs_est: float


@dataclass(frozen=True)
class RPeak:
    """
    Detected R-peak event.

    Attributes:
        sample_index: Continuous sample index (can be fractional from parabolic interpolation).
        t_session_s: Elapsed session time in seconds.
        voltage_uv: ECG voltage at peak in µV.
        rri_ms: RR interval in milliseconds relative to preceding detected peak (or None).
        bpm: Smoothed heart rate in beats per minute (or None).
        flags: Detection flags, e.g. {"searchback", "suspect_missed", "rejected_range", "after_gap"}.
    """

    sample_index: float
    t_session_s: float
    voltage_uv: float
    rri_ms: float | None
    bpm: float | None
    flags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DeviceHR:
    """
    Standard Bluetooth Heart Rate measurement from Polar H10.

    Attributes:
        rx_utc_ns: Host UTC timestamp when packet was received (ns).
        hr_bpm: Instantaneous heart rate reported by device firmware.
        rr_ms: Tuple of RR intervals in milliseconds (resolution 1/1024 s ≈ 0.98 ms).
    """

    rx_utc_ns: int
    hr_bpm: int
    rr_ms: tuple[float, ...] = ()


@dataclass(frozen=True)
class Event:
    """
    Experimental or system marker event.

    Attributes:
        t_session_s: Elapsed session time in seconds corresponding to event.
        host_utc_ns: Host UTC timestamp of event (ns).
        label: Event description or marker string (e.g. "baseline_start", "stimulus_1").
        source: Source of event ("ui", "key", "system", "lsl").
    """

    t_session_s: float
    host_utc_ns: int
    label: str
    source: str = "ui"


@dataclass(frozen=True)
class DeviceInfo:
    """
    Discovered BLE device metadata.
    """

    name: str
    address: str
    rssi: int | None = None
