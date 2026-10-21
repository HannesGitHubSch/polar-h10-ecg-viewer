"""
Streaming and offline Pan-Tompkins style R-peak detector with adaptive thresholds,
fiducial parabolic interpolation, polarity auto-detection, and physiologically
guarded state updates.
"""

from collections import deque
from collections.abc import Callable

import numpy as np
from scipy import signal

from polar_ecg.models import RPeak
from polar_ecg.settings import Settings


class RPeakDetector:
    """
    Online/streaming QRS detector operating on continuous sample indices.
    """

    def __init__(self, fs: float, settings: Settings | None = None):
        self._fs = float(fs)
        self._cfg = settings if settings is not None else Settings()

        # Bandpass filter design (Butterworth 2nd order 5-15 Hz)
        low, high = self._cfg.QRS_BANDPASS_HZ
        nyq = self._fs / 2.0
        low_norm = min(max(low / nyq, 0.01), 0.95)
        high_norm = min(max(high / nyq, low_norm + 0.05), 0.98)
        self._sos = signal.butter(2, [low_norm, high_norm], btype="bandpass", output="sos")

        # Derivative filter kernel: 5-point centered derivative
        self._deriv_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 8.0

        # Moving window integration (MWI) width: ~150 ms
        self._mwi_len = max(3, round(0.150 * self._fs))
        self._mwi_kernel = np.ones(self._mwi_len, dtype=np.float64) / self._mwi_len

        # Refractory period in samples (default 250 ms)
        self._refractory_samples = round((self._cfg.RPEAK_REFRACTORY_MS / 1000.0) * self._fs)
        self._rr_min_ms = self._cfg.RR_MIN_MS
        self._rr_max_ms = self._cfg.RR_MAX_MS
        self._bpm_smoothing_beats = self._cfg.BPM_SMOOTHING_BEATS

        # Internal state
        self.reset()

    def reset(self) -> None:
        """Fully reset detector state, thresholds, and buffers."""
        self._sos_zi = signal.sosfilt_zi(self._sos) * 0.0

        # Carry buffers for continuous streaming across chunks
        self._bandpass_carry = np.zeros(0, dtype=np.float32)
        self._deriv_carry = np.zeros(0, dtype=np.float32)
        self._squared_carry = np.zeros(0, dtype=np.float32)

        # MWI lookahead buffer: list of float values
        self._mwi_buffer: list[float] = []
        self._mwi_first_idx = 0

        # Raw signal circular buffer for fiducial peak location: (sample_index, voltage_uv)
        self._raw_buf_capacity = round(4.0 * self._fs)
        self._raw_indices = deque(maxlen=self._raw_buf_capacity)
        self._raw_voltages = deque(maxlen=self._raw_buf_capacity)

        # Thresholds and learning
        self._spki = 0.0
        self._npki = 0.0
        self._thr1 = 0.0
        self._thr2 = 0.0
        self._is_initialized = False
        self._polarity = 1.0  # +1 for normal R peak, -1 for inverted
        self._init_candidates: list[float] = []  # MWI amplitudes during warm-up
        self._init_samples_seen = 0

        # Beat tracking
        self._last_accepted_idx: float | None = None
        self._last_cand_idx: int = -999999
        self._prev_qrs_slope: float = 1.0
        self._rr_history: deque[float] = deque(maxlen=8)
        self._accepted_rris: deque[float] = deque(maxlen=self._bpm_smoothing_beats * 2)
        self._after_gap = False

    def notify_gap(self) -> None:
        """
        Notify that a gap or discontinuity occurred in incoming data.
        Resets filters and carry state so gaps don't distort future samples.
        """
        self._sos_zi = signal.sosfilt_zi(self._sos) * 0.0
        self._bandpass_carry = np.zeros(0, dtype=np.float32)
        self._deriv_carry = np.zeros(0, dtype=np.float32)
        self._squared_carry = np.zeros(0, dtype=np.float32)
        self._mwi_buffer.clear()
        self._mwi_first_idx = 0
        self._last_accepted_idx = None
        self._last_cand_idx = -999999
        self._after_gap = True

    def process(
        self,
        first_index: int,
        x_uv: np.ndarray,
        index_to_t: Callable[[float], float],
    ) -> list[RPeak]:
        """
        Process a continuous chunk of raw ECG voltages starting at first_index.
        Returns a list of newly detected RPeak objects.
        """
        n = len(x_uv)
        if n == 0:
            return []

        # Store in raw ring buffer
        for i in range(n):
            self._raw_indices.append(first_index + i)
            self._raw_voltages.append(float(x_uv[i]))

        # 1. Bandpass filter
        bp_chunk, self._sos_zi = signal.sosfilt(self._sos, x_uv, zi=self._sos_zi)
        bp_full = (
            np.concatenate((self._bandpass_carry, bp_chunk))
            if len(self._bandpass_carry) > 0
            else bp_chunk
        )

        # 2. Derivative filter
        deriv = np.convolve(bp_full, self._deriv_kernel, mode="valid")
        self._bandpass_carry = bp_full[-4:].copy()

        # 3. Squaring
        squared = deriv**2

        # 4. Moving Window Integration (MWI)
        sq_full = (
            np.concatenate((self._squared_carry, squared))
            if len(self._squared_carry) > 0
            else squared
        )
        mwi = np.convolve(sq_full, self._mwi_kernel, mode="valid")
        self._squared_carry = sq_full[-(self._mwi_len - 1) :].copy()

        # Alignment: mwi start index relative to first_index
        mwi_delay = 2 + (self._mwi_len // 2)
        new_mwi_start_idx = first_index + mwi_delay

        if not self._mwi_buffer:
            self._mwi_first_idx = new_mwi_start_idx

        self._mwi_buffer.extend(mwi.tolist())

        # 5. Peak finding with lookahead on MWI buffer
        lookahead = self._refractory_samples
        if len(self._mwi_buffer) < lookahead * 2:
            return []

        peaks_in_buf, _ = signal.find_peaks(self._mwi_buffer, distance=self._refractory_samples)

        confirmed_candidates = []
        keep_from = 0

        for p_idx in peaks_in_buf:
            # Must have at least `lookahead` samples ahead of it to be confirmed
            if p_idx < len(self._mwi_buffer) - lookahead:
                abs_sample_idx = self._mwi_first_idx + p_idx
                amp = self._mwi_buffer[p_idx]
                confirmed_candidates.append((abs_sample_idx, amp))
                keep_from = max(keep_from, p_idx - lookahead)

        if keep_from > 0:
            self._mwi_buffer = self._mwi_buffer[keep_from:]
            self._mwi_first_idx += keep_from
        elif len(self._mwi_buffer) > int(4.0 * self._fs):
            # Prevent unbounded buffer growth if no peaks are confirmed for a long time
            drop_n = len(self._mwi_buffer) - int(2.0 * self._fs)
            self._mwi_buffer = self._mwi_buffer[drop_n:]
            self._mwi_first_idx += drop_n

        # 6. Initial learning phase (first 2.0 seconds)
        if not self._is_initialized:
            self._init_samples_seen += n
            self._init_candidates.extend(c[1] for c in confirmed_candidates)

            if self._init_samples_seen >= int(2.0 * self._fs):
                self._finish_initialization()

            return []

        # 7. Adaptive threshold decay if beats have been missed (prevents lock-out after artifacts)
        if self._last_accepted_idx is not None:
            avg_rr_samples = (
                np.mean(self._rr_history) * self._fs / 1000.0
                if len(self._rr_history) >= 2
                else 0.8 * self._fs
            )
            samples_since_last = (first_index + n) - self._last_accepted_idx
            if samples_since_last > 1.66 * avg_rr_samples:
                # Smooth decay of signal peak estimate towards noise floor
                self._spki = max(self._spki * 0.95, self._npki * 1.5, 1e-3)
                self._update_thresholds()

            # Hard fallback reset if no beat detected for > 3.0 seconds
            if samples_since_last > 3.0 * self._fs and len(self._mwi_buffer) >= int(self._fs):
                recent_mwi = self._mwi_buffer[-int(2.0 * self._fs) :]
                p90 = float(np.percentile(recent_mwi, 90))
                p20 = float(np.percentile(recent_mwi, 20))
                self._spki = max(p90 * 0.8, 1e-3)
                self._npki = min(max(p20, 1e-4), self._spki * 0.3)
                self._update_thresholds()

        # 8. Evaluate candidates
        detected_peaks: list[RPeak] = []

        for cand_idx, cand_mwi in confirmed_candidates:
            # Refractory period check
            if cand_idx - self._last_cand_idx < self._refractory_samples:
                continue

            time_since_prev = (
                cand_idx - self._last_accepted_idx if self._last_accepted_idx is not None else 99999
            )

            # Check threshold
            is_qrs = False
            is_searchback = False

            if cand_mwi >= self._thr1:
                is_qrs = True
            else:
                avg_rr_samples = (
                    np.mean(self._rr_history) * self._fs / 1000.0
                    if len(self._rr_history) >= 2
                    else 0.8 * self._fs
                )
                if time_since_prev > 1.66 * avg_rr_samples and cand_mwi >= self._thr2:
                    is_qrs = True
                    is_searchback = True

            if not is_qrs:
                # Update noise threshold (bounded by signal ceiling)
                self._npki = 0.875 * self._npki + 0.125 * min(cand_mwi, self._spki * 0.4)
                self._update_thresholds()
                continue

            # 9. Fiducial refinement on raw signal
            refined_idx, peak_voltage, cand_slope = self._find_raw_fiducial(cand_idx)
            if refined_idx is None:
                continue

            # 10. RR gating & T-wave discrimination before updating state
            flags = set()
            if is_searchback:
                flags.add("searchback")

            rri_ms: float | None = None
            bpm: float | None = None

            if self._last_accepted_idx is not None:
                rri_ms = ((refined_idx - self._last_accepted_idx) / self._fs) * 1000.0

                if rri_ms < self._rr_min_ms:
                    # Too close to previous beat (refractory period)
                    continue

                # T-wave discrimination: candidate occurring within 0.52 * expected_rr (up to 520 ms) after QRS with flat slope
                expected_rr_ms = (
                    np.median(self._rr_history)
                    if len(self._rr_history) >= 2
                    else 800.0
                )
                twave_window_ms = max(360.0, min(520.0, 0.52 * expected_rr_ms))
                if rri_ms < twave_window_ms and self._prev_qrs_slope > 0:
                    if cand_slope < 0.55 * self._prev_qrs_slope:
                        # Classify as T-wave: update noise floor, do not accept as QRS
                        self._npki = 0.875 * self._npki + 0.125 * min(cand_mwi, self._spki * 0.4)
                        self._update_thresholds()
                        continue

                # Cap single-beat MWI contribution to prevent huge artifact lock-outs
                capped_mwi = min(cand_mwi, self._spki * 2.5) if self._spki > 0 else cand_mwi

                if rri_ms > self._rr_max_ms:
                    flags.add("rejected_range")
                    self._last_accepted_idx = refined_idx
                    self._last_cand_idx = cand_idx
                else:
                    if len(self._rr_history) >= 4 and rri_ms > 1.66 * np.median(list(self._rr_history)[-8:]):
                        flags.add("suspect_missed")
                    self._last_accepted_idx = refined_idx
                    self._last_cand_idx = cand_idx
                    self._rr_history.append(rri_ms)
                    self._accepted_rris.append(rri_ms)
                    self._spki = 0.875 * self._spki + 0.125 * capped_mwi
                    self._prev_qrs_slope = 0.875 * self._prev_qrs_slope + 0.125 * cand_slope
                    self._update_thresholds()

                if self._accepted_rris:
                    bpm = 60000.0 / float(np.median(self._accepted_rris))
            else:
                self._last_accepted_idx = refined_idx
                self._last_cand_idx = cand_idx
                if self._after_gap:
                    flags.add("after_gap")
                    self._after_gap = False
                capped_mwi = min(cand_mwi, self._spki * 2.5) if self._spki > 0 else cand_mwi
                self._spki = 0.875 * self._spki + 0.125 * capped_mwi
                self._prev_qrs_slope = cand_slope
                self._update_thresholds()

            t_session = index_to_t(refined_idx)
            detected_peaks.append(
                RPeak(
                    sample_index=refined_idx,
                    t_session_s=t_session,
                    voltage_uv=peak_voltage,
                    rri_ms=rri_ms,
                    bpm=bpm,
                    flags=frozenset(flags),
                )
            )

        return detected_peaks

    def _update_thresholds(self) -> None:
        """Update detection thresholds THR1 and THR2 with noise ceiling."""
        self._npki = min(self._npki, 0.45 * self._spki)
        self._thr1 = self._npki + 0.25 * (self._spki - self._npki)
        self._thr2 = 0.5 * self._thr1

    def _finish_initialization(self) -> None:
        """Initialize thresholds and polarity from initial candidate pool."""
        self._is_initialized = True
        if self._init_candidates:
            # Use upper percentile of MWI peaks to estimate QRS peak amplitude
            qrs_est = float(np.percentile(self._init_candidates, 80))
            noise_est = float(np.percentile(self._init_candidates, 20))
            self._spki = max(qrs_est * 0.7, 1e-3)
            self._npki = max(noise_est, self._spki * 0.05)
        else:
            self._spki = 1.0
            self._npki = 0.1

        self._update_thresholds()

        # Polarity check: inspect raw voltages in buffer
        if len(self._raw_voltages) > 0:
            raw_arr = np.array(self._raw_voltages)
            med = np.median(raw_arr)
            pos_dev = np.percentile(raw_arr, 99) - med
            neg_dev = med - np.percentile(raw_arr, 1)
            # If negative excursion is larger, inverted lead
            self._polarity = -1.0 if neg_dev > 1.2 * pos_dev else 1.0

    def _find_raw_fiducial(self, cand_mwi_idx: int) -> tuple[float | None, float, float]:
        """
        Locate R-peak in the raw ECG signal around the MWI candidate index.
        Uses parabolic interpolation for fractional sample precision.
        Returns (best_sample_idx, peak_voltage, max_slope).
        """
        search_start = cand_mwi_idx - int(0.25 * self._fs)
        search_end = cand_mwi_idx + int(0.05 * self._fs)

        if not self._raw_indices:
            return None, 0.0, 0.0

        buf_first = self._raw_indices[0]
        buf_last = self._raw_indices[-1]

        # Clamp search window to available raw buffer rather than dropping edge candidates
        search_start = max(buf_first, search_start)
        search_end = min(buf_last, search_end)

        if search_start >= search_end:
            return None, 0.0, 0.0

        raw_idx_arr = np.array(self._raw_indices)
        raw_vol_arr = np.array(self._raw_voltages)

        mask = (raw_idx_arr >= search_start) & (raw_idx_arr <= search_end)
        sub_indices = raw_idx_arr[mask]
        sub_voltages = raw_vol_arr[mask]

        if len(sub_voltages) < 3:
            return None, 0.0, 0.0

        oriented_voltages = sub_voltages * self._polarity
        max_pos = np.argmax(oriented_voltages)

        best_sample_idx = float(sub_indices[max_pos])
        peak_voltage = float(sub_voltages[max_pos])

        # Compute max slope near the peak for T-wave discrimination
        slope_w = oriented_voltages[max(0, max_pos - 5) : min(len(oriented_voltages), max_pos + 6)]
        cand_slope = float(np.max(np.abs(np.diff(slope_w)))) if len(slope_w) > 1 else 1.0

        if 0 < max_pos < len(sub_voltages) - 1:
            y1 = float(oriented_voltages[max_pos - 1])
            y2 = float(oriented_voltages[max_pos])
            y3 = float(oriented_voltages[max_pos + 1])
            denom = y1 - 2.0 * y2 + y3
            if abs(denom) > 1e-6:
                delta = 0.5 * (y1 - y3) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
                best_sample_idx += delta
                v1 = float(sub_voltages[max_pos - 1])
                v2 = float(sub_voltages[max_pos])
                v3 = float(sub_voltages[max_pos + 1])
                peak_voltage = v2 - 0.25 * (v1 - v3) * delta

        return best_sample_idx, peak_voltage, cand_slope


    def flush(self, index_to_t: Callable[[float], float]) -> list[RPeak]:
        """
        Flush remaining candidates in buffer at end of recording/stream.
        """
        if not self._mwi_buffer:
            return []

        peaks_in_buf, _ = signal.find_peaks(self._mwi_buffer, distance=self._refractory_samples)
        detected = []
        for p_idx in peaks_in_buf:
            abs_sample_idx = self._mwi_first_idx + p_idx
            amp = self._mwi_buffer[p_idx]
            if amp >= self._thr1:
                refined_idx, peak_voltage, _ = self._find_raw_fiducial(abs_sample_idx)
                if refined_idx is not None:
                    if self._last_accepted_idx is not None:
                        rri_ms = ((refined_idx - self._last_accepted_idx) / self._fs) * 1000.0
                        if rri_ms < self._rr_min_ms:
                            continue
                    else:
                        rri_ms = None
                    t_session = index_to_t(refined_idx)
                    detected.append(
                        RPeak(
                            sample_index=refined_idx,
                            t_session_s=t_session,
                            voltage_uv=peak_voltage,
                            rri_ms=rri_ms,
                            bpm=None,
                        )
                    )
        self._mwi_buffer.clear()
        return detected


def detect_offline(
    x_uv: np.ndarray,
    fs: float = 130.0,
    settings: Settings | None = None,
) -> list[RPeak]:
    """
    Offline R-peak detection on an entire recording using the streaming detector.
    """
    cfg = settings if settings is not None else Settings()
    detector = RPeakDetector(fs=fs, settings=cfg)

    chunk_size = 73
    all_peaks: list[RPeak] = []

    for i in range(0, len(x_uv), chunk_size):
        chunk = x_uv[i : i + chunk_size]
        peaks = detector.process(
            first_index=i,
            x_uv=chunk,
            index_to_t=lambda idx: idx / fs,
        )
        all_peaks.extend(peaks)

    # Flush end of signal
    final_peaks = detector.flush(index_to_t=lambda idx: idx / fs)
    all_peaks.extend(final_peaks)

    return all_peaks
