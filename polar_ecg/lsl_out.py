"""
Lab Streaming Layer (LSL) integration for Polar H10 ECG, Markers, and RR outlets.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polar_ecg.models import DeviceHR, Event, RPeak, TimedFrame

logger = logging.getLogger(__name__)

try:
    import pylsl

    HAS_PYLSL = True
except ImportError:
    pylsl = None
    HAS_PYLSL = False


class LslOutletManager:
    """
    Manages streaming Polar H10 data to the Lab Streaming Layer (LSL).

    Exposes up to three synchronized streams:
    1. PolarH10_ECG: type 'ECG', 1-channel, nominal 130 Hz, float32 in microvolts.
    2. PolarH10_Markers: type 'Markers', 1-channel irregular string events.
    3. PolarH10_RR: type 'RR', 2-channel irregular float32 [rr_interval_ms, instant_bpm].
    """

    def __init__(self, device_name: str = "Polar H10", device_id: str = "polar_h10") -> None:
        self.device_name = device_name
        self.device_id = device_id.replace(":", "_").replace(" ", "_")

        self.ecg_outlet: Any = None
        self.markers_outlet: Any = None
        self.rr_outlet: Any = None
        self._lsl_t0: float | None = None

        if not HAS_PYLSL:
            logger.warning("pylsl is not installed; Lab Streaming Layer (LSL) output is disabled.")
            return

        self._init_outlets()

    @property
    def is_active(self) -> bool:
        return HAS_PYLSL and self.ecg_outlet is not None

    def _init_outlets(self) -> None:
        try:
            # Calibrate LSL clock offset against Python time.monotonic
            self._lsl_clock_offset = pylsl.local_clock() - time.monotonic()

            # 1. ECG Outlet
            ecg_info = pylsl.StreamInfo(
                name="PolarH10_ECG",
                type="ECG",
                channel_count=1,
                nominal_srate=130.0,
                channel_format=pylsl.cf_float32,
                source_id=f"ecg_{self.device_id}",
            )
            desc = ecg_info.desc()
            channels = desc.append_child("channels")
            ch = channels.append_child("channel")
            ch.append_child_value("label", "ECG")
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "ECG")
            desc.append_child_value("manufacturer", "Polar")
            desc.append_child_value("model", self.device_name)
            self.ecg_outlet = pylsl.StreamOutlet(ecg_info)

            # 2. Markers Outlet
            markers_info = pylsl.StreamInfo(
                name="PolarH10_Markers",
                type="Markers",
                channel_count=1,
                nominal_srate=pylsl.IRREGULAR_RATE,
                channel_format=pylsl.cf_string,
                source_id=f"markers_{self.device_id}",
            )
            self.markers_outlet = pylsl.StreamOutlet(markers_info)

            # 3. RR / Heart Rate Outlet
            rr_info = pylsl.StreamInfo(
                name="PolarH10_RR",
                type="RR",
                channel_count=2,
                nominal_srate=pylsl.IRREGULAR_RATE,
                channel_format=pylsl.cf_float32,
                source_id=f"rr_{self.device_id}",
            )
            rr_desc = rr_info.desc()
            rr_channels = rr_desc.append_child("channels")
            ch_rr = rr_channels.append_child("channel")
            ch_rr.append_child_value("label", "RR")
            ch_rr.append_child_value("unit", "milliseconds")
            ch_bpm = rr_channels.append_child("channel")
            ch_bpm.append_child_value("label", "BPM")
            ch_bpm.append_child_value("unit", "beats_per_minute")
            self.rr_outlet = pylsl.StreamOutlet(rr_info)

            logger.info("Initialized LSL outlets for %s (source_id=%s)", self.device_name, self.device_id)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize LSL outlets: %s", exc)
            self.ecg_outlet = None
            self.markers_outlet = None
            self.rr_outlet = None

    def push_frame(self, frame: TimedFrame) -> None:
        """Push a batch of continuous ECG samples with precise LSL timestamps."""
        if self.ecg_outlet is None or len(frame.voltages_uv) == 0:
            return

        if self._lsl_t0 is None:
            self._lsl_t0 = pylsl.local_clock() - float(frame.t_session_s[0])

        timestamps = (self._lsl_t0 + frame.t_session_s).tolist()
        samples = [[float(v)] for v in frame.voltages_uv]
        self.ecg_outlet.push_chunk(samples, timestamp=timestamps)

    def push_event(self, event: Event) -> None:
        """Push an experimental marker to the Markers outlet."""
        if self.markers_outlet is None:
            return

        ts = (self._lsl_t0 + event.t_session_s) if self._lsl_t0 is not None else pylsl.local_clock()
        self.markers_outlet.push_sample([str(event.label)], timestamp=ts)

    def push_peak(self, peak: RPeak) -> None:
        """Push detected R-peak information [rri_ms, bpm] to RR outlet."""
        if self.rr_outlet is None:
            return

        rri = float(peak.rri_ms) if peak.rri_ms is not None else 0.0
        bpm = float(peak.bpm) if peak.bpm is not None else 0.0
        ts = (self._lsl_t0 + peak.t_session_s) if self._lsl_t0 is not None else pylsl.local_clock()
        self.rr_outlet.push_sample([rri, bpm], timestamp=ts)

    def push_device_hr(self, hr: DeviceHR) -> None:
        """Push native Polar HR measurement to RR outlet."""
        if self.rr_outlet is None:
            return

        ts = pylsl.local_clock()
        bpm = float(hr.hr_bpm)
        if hr.rr_ms:
            for rr in hr.rr_ms:
                self.rr_outlet.push_sample([float(rr), bpm], timestamp=ts)
        else:
            self.rr_outlet.push_sample([0.0, bpm], timestamp=ts)

    def close(self) -> None:
        """Cleanly close outlets."""
        self.ecg_outlet = None
        self.markers_outlet = None
        self.rr_outlet = None
        self._lsl_t0 = None
