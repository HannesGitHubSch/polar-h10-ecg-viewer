"""
Threaded CSV v2 recorder with metadata sidecar, auxiliary event/HR/beat channels,
and atomic file finalization.
"""

import csv
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from polar_ecg.models import DeviceHR, DeviceInfo, Event, RPeak, TimedFrame
from polar_ecg.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingSummary:
    """Summary statistics produced upon finishing a recording session."""

    file_path: Path
    samples_written: int
    samples_missing: int
    discontinuities: int
    duration_s: float
    events_count: int


def ns_to_iso_utc(ns: int) -> str:
    """Convert nanoseconds since Unix epoch to ISO-8601 UTC string with offset."""
    sec = ns / 1e9
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    return dt.isoformat()


class Recorder:
    """
    Manages background file writing for ECG sessions.
    Decoupled from GUI thread through a thread-safe queue.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        error_callback: Callable[[str], None] | None = None,
    ):
        self._cfg = settings if settings is not None else Settings()
        self._error_callback = error_callback

        self._is_recording = False
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None

        self._base_path: Path | None = None
        self._participant_id: str = ""
        self._device_info: DeviceInfo | None = None

        # Statistics
        self._samples_written = 0
        self._samples_missing = 0
        self._discontinuities = 0
        self._start_time_mono = 0.0
        self._start_utc_str = ""
        self._events_list: list[dict] = []
        self._last_fs_est = self._cfg.ECG_FS_NOMINAL

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(
        self,
        base_path: str | Path,
        participant_id: str,
        device_info: DeviceInfo | None = None,
    ) -> None:
        """Start a new recording session."""
        if self._is_recording:
            logger.warning("Recorder.start called while already recording")
            return

        self._base_path = Path(base_path)
        self._participant_id = participant_id
        self._device_info = device_info

        self._samples_written = 0
        self._samples_missing = 0
        self._discontinuities = 0
        self._events_list = []
        self._start_time_mono = time.monotonic()
        self._start_utc_str = datetime.now(timezone.utc).isoformat()
        self._is_recording = True
        self._ready_event = threading.Event()

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Start writer thread
        self._worker_thread = threading.Thread(
            target=self._writer_loop,
            name="polar_ecg_recorder",
            daemon=True,
        )
        self._worker_thread.start()
        # Wait for file creation / initialization
        self._ready_event.wait(timeout=2.0)

    def submit_frame(self, frame: TimedFrame) -> None:
        """Enqueue a timed ECG frame for writing."""
        if self._is_recording:
            self._queue.put(("FRAME", frame))

    def submit_peak(self, peak: RPeak) -> None:
        """Enqueue an R-peak detection."""
        if self._is_recording:
            self._queue.put(("PEAK", peak))

    def submit_hr(self, hr: DeviceHR) -> None:
        """Enqueue a device native heart rate sample."""
        if self._is_recording and self._cfg.RECORD_DEVICE_HR:
            self._queue.put(("HR", hr))

    def add_event(self, event: Event) -> None:
        """Enqueue an experimental or system event."""
        if self._is_recording:
            self._queue.put(("EVENT", event))

    def stop(self) -> RecordingSummary | None:
        """Stop recording, flush buffers, finalize files, and return summary."""
        if not self._is_recording:
            return None

        self._is_recording = False
        self._queue.put(("STOP", None))

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None

        duration_s = time.monotonic() - self._start_time_mono
        final_csv_path = self._base_path.with_name(f"{self._base_path.name}.csv")

        return RecordingSummary(
            file_path=final_csv_path,
            samples_written=self._samples_written,
            samples_missing=self._samples_missing,
            discontinuities=self._discontinuities,
            duration_s=duration_s,
            events_count=len(self._events_list),
        )

    def _writer_loop(self) -> None:
        """Background file writing loop."""
        partial_csv_path = self._base_path.with_name(f"{self._base_path.name}.csv.partial")
        final_csv_path = self._base_path.with_name(f"{self._base_path.name}.csv")
        meta_json_path = self._base_path.with_name(f"{self._base_path.name}_meta.json")
        beats_csv_path = self._base_path.with_name(f"{self._base_path.name}_beats.csv")
        hr_csv_path = self._base_path.with_name(f"{self._base_path.name}_hr.csv")

        csv_file = None
        beats_file = None
        hr_file = None

        pending_events: list[Event] = []

        try:
            # Ensure parent folder exists
            partial_csv_path.parent.mkdir(parents=True, exist_ok=True)

            # 1. Initialize metadata sidecar
            initial_meta = {
                "format": "polar-ecg-csv",
                "format_version": 2,
                "software_version": "0.2.0",
                "participant_id": self._participant_id,
                "device": {
                    "name": self._device_info.name if self._device_info else "Unknown",
                    "address": self._device_info.address if self._device_info else "Unknown",
                },
                "fs_nominal": self._cfg.ECG_FS_NOMINAL,
                "fs_estimated": self._cfg.ECG_FS_NOMINAL,
                "resolution_bits": self._cfg.ECG_RESOLUTION,
                "started_utc": self._start_utc_str,
                "stopped_utc": None,
                "samples_written": 0,
                "samples_missing": 0,
                "discontinuities": 0,
                "events": [],
                "timing_notes": (
                    "host_time_utc maps device clock onto host clock using minimum-latency envelope. "
                    "Relative timing follows device clock."
                ),
            }
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(initial_meta, f, indent=2)

            # 2. Open CSV files
            csv_file = open(partial_csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "sample_index",
                "t_session_s",
                "device_time_ns",
                "host_time_utc",
                "packet_rx_utc",
                "ecg_uv",
                "gap_before",
                "event",
            ])

            beats_file = open(beats_csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
            beats_writer = csv.writer(beats_file)
            beats_writer.writerow([
                "sample_index",
                "t_session_s",
                "host_time_utc",
                "voltage_uv",
                "rri_ms",
                "bpm",
                "flags",
            ])

            if self._cfg.RECORD_DEVICE_HR:
                hr_file = open(hr_csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
                hr_writer = csv.writer(hr_file)
                hr_writer.writerow(["host_time_utc", "hr_bpm", "rr_ms"])

            # Signal that files are initialized and open
            self._ready_event.set()

            last_flush = time.monotonic()

            while True:
                try:
                    item_type, data = self._queue.get(timeout=0.2)
                except queue.Empty:
                    # Periodic flush
                    now = time.monotonic()
                    if now - last_flush >= self._cfg.RECORDING_FLUSH_INTERVAL_S:
                        if csv_file:
                            csv_file.flush()
                        if beats_file:
                            beats_file.flush()
                        if hr_file:
                            hr_file.flush()
                        last_flush = now
                    continue

                if item_type == "STOP":
                    break

                elif item_type == "EVENT":
                    event: Event = data
                    pending_events.append(event)
                    self._events_list.append({
                        "t_session_s": event.t_session_s,
                        "host_time_utc": ns_to_iso_utc(event.host_utc_ns),
                        "label": event.label,
                        "source": event.source,
                    })

                elif item_type == "PEAK":
                    peak: RPeak = data
                    peak_utc = ns_to_iso_utc(int(time.time_ns()))
                    flags_str = "|".join(sorted(peak.flags)) if peak.flags else ""
                    beats_writer.writerow([
                        f"{peak.sample_index:.2f}",
                        f"{peak.t_session_s:.4f}",
                        peak_utc,
                        f"{peak.voltage_uv:.2f}",
                        f"{peak.rri_ms:.2f}" if peak.rri_ms is not None else "",
                        f"{peak.bpm:.1f}" if peak.bpm is not None else "",
                        flags_str,
                    ])

                elif item_type == "HR":
                    hr: DeviceHR = data
                    hr_utc = ns_to_iso_utc(hr.rx_utc_ns)
                    for rr in (hr.rr_ms if hr.rr_ms else (None,)):
                        hr_writer.writerow([
                            hr_utc,
                            hr.hr_bpm,
                            f"{rr:.2f}" if rr is not None else "",
                        ])

                elif item_type == "FRAME":
                    frame: TimedFrame = data
                    self._last_fs_est = frame.fs_est
                    if frame.gap_before > 0:
                        self._samples_missing += frame.gap_before
                    elif frame.gap_before == -1:
                        self._discontinuities += 1

                    rx_utc_str = ns_to_iso_utc(frame.packet_rx_utc_ns)
                    n_samples = len(frame.voltages_uv)

                    # Map pending events to closest sample index in this frame
                    event_stamps: dict[int, list[str]] = {}
                    if pending_events:
                        remaining_events = []
                        for ev in pending_events:
                            # If event occurred before or inside this frame
                            if ev.t_session_s <= frame.t_session_s[-1] or len(frame.t_session_s) == 1:
                                dists = [abs(t - ev.t_session_s) for t in frame.t_session_s]
                                closest_idx = int(frame.first_index + dists.index(min(dists)))
                                event_stamps.setdefault(closest_idx, []).append(ev.label)
                            else:
                                remaining_events.append(ev)
                        pending_events = remaining_events

                    rows = []
                    for i in range(n_samples):
                        s_idx = frame.first_index + i
                        gap = frame.gap_before if i == 0 else 0
                        ev_str = "|".join(event_stamps.get(s_idx, []))
                        rows.append([
                            s_idx,
                            f"{frame.t_session_s[i]:.6f}",
                            frame.device_ns[i],
                            ns_to_iso_utc(frame.host_utc_ns[i]),
                            rx_utc_str,
                            f"{frame.voltages_uv[i]:.2f}",
                            gap,
                            ev_str,
                        ])

                    csv_writer.writerows(rows)
                    self._samples_written += n_samples

                    # Periodic flush check
                    now = time.monotonic()
                    if now - last_flush >= self._cfg.RECORDING_FLUSH_INTERVAL_S:
                        csv_file.flush()
                        last_flush = now

            # Clean shutdown of files
            for f in (csv_file, beats_file, hr_file):
                if f:
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                    f.close()

            # Rename .csv.partial to .csv
            if partial_csv_path.exists():
                if final_csv_path.exists():
                    final_csv_path.unlink()
                partial_csv_path.rename(final_csv_path)

            # Update final metadata sidecar
            stopped_utc_str = datetime.now(timezone.utc).isoformat()
            final_meta = {
                "format": "polar-ecg-csv",
                "format_version": 2,
                "software_version": "0.2.0",
                "participant_id": self._participant_id,
                "device": {
                    "name": self._device_info.name if self._device_info else "Unknown",
                    "address": self._device_info.address if self._device_info else "Unknown",
                },
                "fs_nominal": self._cfg.ECG_FS_NOMINAL,
                "fs_estimated": round(self._last_fs_est, 4),
                "resolution_bits": self._cfg.ECG_RESOLUTION,
                "started_utc": self._start_utc_str,
                "stopped_utc": stopped_utc_str,
                "samples_written": self._samples_written,
                "samples_missing": self._samples_missing,
                "discontinuities": self._discontinuities,
                "events": self._events_list,
                "timing_notes": (
                    "host_time_utc maps device clock onto host clock using minimum-latency envelope. "
                    "Relative timing follows device clock."
                ),
            }
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(final_meta, f, indent=2)

        except Exception as exc:
            logger.exception("Recorder encountered error in writer thread")
            self._ready_event.set()
            if self._error_callback:
                self._error_callback(str(exc))
        finally:
            for f in (csv_file, beats_file, hr_file):
                if f and not f.closed:
                    f.close()
