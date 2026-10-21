"""
Clock synchronization and continuous sample index / timing calculation.
"""

import logging
import time
from collections import deque

import numpy as np

from polar_ecg.models import RawFrame, TimedFrame

logger = logging.getLogger(__name__)


class ClockSync:
    """
    Maps device clock nanoseconds onto host UTC nanoseconds using the
    minimum-latency lower envelope over a sliding window.
    """

    def __init__(self, window_size: int = 64):
        self._window_size = window_size
        self._offsets: deque[int] = deque(maxlen=window_size)
        self._anchor_mono_ns: int | None = None
        self._anchor_utc_ns: int | None = None

    def set_anchors(self, anchor_mono_ns: int, anchor_utc_ns: int) -> None:
        """Explicitly set reference anchors (useful in tests or session start)."""
        self._anchor_mono_ns = anchor_mono_ns
        self._anchor_utc_ns = anchor_utc_ns

    def reset(self) -> None:
        """Clear state and offsets."""
        self._offsets.clear()
        self._anchor_mono_ns = None
        self._anchor_utc_ns = None

    def update(self, device_ns: int, rx_mono_ns: int) -> None:
        """Record packet arrival offset: rx_mono_ns - device_ns."""
        if self._anchor_mono_ns is None or self._anchor_utc_ns is None:
            self._anchor_mono_ns = time.monotonic_ns()
            self._anchor_utc_ns = time.time_ns()

        offset_k = rx_mono_ns - device_ns
        self._offsets.append(offset_k)

    def mono_to_utc(self, mono_ns: int) -> int:
        """Convert a host monotonic timestamp to UTC nanoseconds."""
        if self._anchor_mono_ns is None or self._anchor_utc_ns is None:
            self._anchor_mono_ns = time.monotonic_ns()
            self._anchor_utc_ns = time.time_ns()
        return self._anchor_utc_ns + (mono_ns - self._anchor_mono_ns)

    def to_utc(self, device_ns_arr: np.ndarray) -> np.ndarray:
        """
        Map an array of device timestamps to host UTC nanoseconds
        via the minimum latency envelope offset.
        """
        if not self._offsets:
            # Fallback if called before any update
            now_mono = time.monotonic_ns()
            now_utc = time.time_ns()
            return device_ns_arr + (now_utc - now_mono)

        min_offset = min(self._offsets)
        host_mono = device_ns_arr + min_offset
        return self._anchor_utc_ns + (host_mono - self._anchor_mono_ns)


class SampleClock:
    """
    Maintains a continuous sample index, detects dropped frames and clock
    discontinuities, estimates effective sampling rate, and attaches
    monotonically increasing session timestamps.
    """

    def __init__(
        self,
        fs_nominal: float = 130.0,
        fs_tol: float = 0.02,
        sync: ClockSync | None = None,
    ):
        self._fs_nominal = float(fs_nominal)
        self._fs_tol = float(fs_tol)
        self._sync = sync if sync is not None else ClockSync()

        self._first_frame = True
        self._prev_device_last_ns: int = 0
        self._prev_rx_mono_ns: int = 0
        self._current_index: int = 0
        self._total_missing: int = 0
        self._fs_est: float = self._fs_nominal

        # (sample_index, device_ns) points for least-squares fs estimation
        self._contiguous_points: deque[tuple[int, int]] = deque(maxlen=120)

        self._t0_device_ns: int = 0
        self._segment_offset_s: float = 0.0
        self._last_t_session: float = 0.0

    @property
    def total_missing(self) -> int:
        """Total missing samples detected so far across all gaps."""
        return self._total_missing

    @property
    def fs_estimated(self) -> float:
        """Current estimated effective sampling rate in Hz."""
        return self._fs_est

    def reset(self) -> None:
        """Reset all clock state (called on session start or stream restart)."""
        self._first_frame = True
        self._prev_device_last_ns = 0
        self._prev_rx_mono_ns = 0
        self._current_index = 0
        self._total_missing = 0
        self._fs_est = self._fs_nominal
        self._contiguous_points.clear()
        self._t0_device_ns = 0
        self._segment_offset_s = 0.0
        self._last_t_session = 0.0
        self._sync.reset()

    def process(self, raw: RawFrame) -> TimedFrame:
        """
        Process a RawFrame and produce a TimedFrame with gap metadata and timestamps.
        """
        n = len(raw.voltages_uv)
        if n == 0:
            raise ValueError("RawFrame contains 0 samples")

        T = raw.device_last_ns
        self._sync.update(T, raw.rx_mono_ns)

        if self._first_frame:
            p_nom_ns = 1e9 / self._fs_nominal
            gap_before = 0
            self._current_index = 0
            self._t0_device_ns = round(T - (n - 1) * p_nom_ns)
            self._segment_offset_s = 0.0
            self._first_frame = False
            self._contiguous_points.append((n - 1, T))
        else:
            dt_ns = T - self._prev_device_last_ns
            max_gap_ns = 60 * 1_000_000_000  # 60 seconds threshold for discontinuity

            if dt_ns <= 0 or dt_ns > max_gap_ns:
                # Discontinuity (device clock reset, wrap, or reconnect after long gap)
                logger.warning(
                    "Device clock discontinuity detected: dt_ns=%d. Starting new segment.",
                    dt_ns,
                )
                gap_before = -1
                p_nom_ns = 1e9 / self._fs_nominal
                self._segment_offset_s = self._last_t_session + (1.0 / self._fs_nominal)
                self._t0_device_ns = round(T - (n - 1) * p_nom_ns)
                self._contiguous_points.clear()
                self._current_index += 1
                self._contiguous_points.append((self._current_index + n - 1, T))
            else:
                expected_samples = (dt_ns / 1e9) * self._fs_est
                missing = round(expected_samples) - n
                if missing >= 1:
                    gap_before = missing
                    self._total_missing += missing
                    self._current_index += missing
                    self._contiguous_points.clear()
                    logger.warning("Packet gap detected: %d missing samples.", missing)
                else:
                    gap_before = 0

                self._contiguous_points.append((self._current_index + n - 1, T))

                # Update fs_est via least squares over contiguous window
                if len(self._contiguous_points) >= 10:
                    pts = np.array(self._contiguous_points, dtype=np.float64)
                    idx_arr = pts[:, 0]
                    t_arr = pts[:, 1]
                    # Slope dt/didx gives period in ns
                    slope, _ = np.polyfit(idx_arr, t_arr, 1)
                    if slope > 0:
                        cand_fs = 1e9 / slope
                        min_fs = self._fs_nominal * (1.0 - self._fs_tol)
                        max_fs = self._fs_nominal * (1.0 + self._fs_tol)
                        self._fs_est = float(np.clip(cand_fs, min_fs, max_fs))

        self._prev_device_last_ns = T
        self._prev_rx_mono_ns = raw.rx_mono_ns

        first_idx = self._current_index
        p_eff_ns = 1e9 / self._fs_est
        indices = np.arange(n, dtype=np.float64)
        device_ns = np.round(T - (n - 1 - indices) * p_eff_ns).astype(np.int64)

        t_session_s = ((device_ns - self._t0_device_ns) / 1e9) + self._segment_offset_s
        self._last_t_session = float(t_session_s[-1])
        self._current_index += n

        host_utc_ns = self._sync.to_utc(device_ns)
        packet_rx_utc_ns = self._sync.mono_to_utc(raw.rx_mono_ns)

        return TimedFrame(
            first_index=first_idx,
            voltages_uv=raw.voltages_uv.astype(np.float32),
            device_ns=device_ns,
            t_session_s=t_session_s,
            host_utc_ns=host_utc_ns,
            packet_rx_utc_ns=packet_rx_utc_ns,
            gap_before=gap_before,
            fs_est=self._fs_est,
        )
