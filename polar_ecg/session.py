"""
Core Session controller managing the stream pipeline, gap-aware clock,
R-peak detector, threaded recorder, connection watchdog, and automatic reconnection.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

from polar_ecg.detection import RPeakDetector
from polar_ecg.lsl_out import LslOutletManager
from polar_ecg.models import DeviceHR, DeviceInfo, Event, RawFrame, RPeak, TimedFrame
from polar_ecg.recording import Recorder, RecordingSummary
from polar_ecg.settings import Settings
from polar_ecg.sources import Source
from polar_ecg.timing import SampleClock

logger = logging.getLogger(__name__)


class Session:
    """
    Coordinates data flow from Source -> SampleClock -> RPeakDetector -> Recorder.
    Runs completely decoupled from Qt widgets on an asyncio event loop.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        enable_lsl: bool | None = None,
        on_frame_ready: Callable[[TimedFrame], None] | None = None,
        on_peak: Callable[[RPeak], None] | None = None,
        on_device_hr: Callable[[DeviceHR], None] | None = None,
        on_state: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        self._cfg = settings if settings is not None else Settings()
        self._enable_lsl = enable_lsl if enable_lsl is not None else self._cfg.LSL_ENABLED
        self._lsl: LslOutletManager | None = None
        self._on_frame_ready = on_frame_ready
        self._on_peak = on_peak
        self._on_device_hr = on_device_hr
        self._on_state = on_state
        self._on_status = on_status
        self._on_error = on_error

        # Pipeline components
        self._clock = SampleClock(fs_nominal=self._cfg.ECG_FS_NOMINAL)
        self._detector = RPeakDetector(fs=self._cfg.ECG_FS_NOMINAL, settings=self._cfg)
        self._recorder = Recorder(settings=self._cfg, error_callback=self._handle_recorder_error)

        self._source: Source | None = None
        self._device_info: DeviceInfo | None = None

        # State tracking
        self._is_streaming = False
        self._is_reconnecting = False
        self._last_frame_mono = 0.0
        self._last_t_session = 0.0
        self._signal_is_stalled = False
        self._watchdog_task: asyncio.Task | None = None

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def is_recording(self) -> bool:
        return self._recorder.is_recording

    @property
    def clock(self) -> SampleClock:
        return self._clock

    @property
    def lsl(self) -> LslOutletManager | None:
        return self._lsl

    def set_source(self, source: Source, device_info: DeviceInfo | None = None) -> None:
        """Set or swap the active data source."""
        self._source = source
        self._device_info = device_info

    async def start_streaming(self) -> None:
        """Reset pipeline and begin streaming from the configured source."""
        if self._source is None:
            raise RuntimeError("Cannot start streaming: no source configured")

        if self._is_streaming:
            return

        self._clock.reset()
        self._detector.reset()
        self._last_frame_mono = time.monotonic()
        self._signal_is_stalled = False

        if self._enable_lsl and self._lsl is None:
            dev_name = self._device_info.name if self._device_info else "Polar H10"
            dev_addr = self._device_info.address if self._device_info else "polar_h10"
            self._lsl = LslOutletManager(device_name=dev_name, device_id=dev_addr)

        await self._source.start()
        self._is_streaming = True

        if self._on_state:
            self._on_state("STREAMING")
        if self._on_status:
            self._on_status("Streaming ECG data...")

        # Start watchdog
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def stop_streaming(self) -> None:
        """Stop data streaming and cancel watchdog."""
        if not self._is_streaming:
            return

        self._is_streaming = False

        if self._lsl:
            self._lsl.close()
            self._lsl = None

        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

        if self._source:
            await self._source.stop()

        if self._on_state:
            self._on_state("CONNECTED")
        if self._on_status:
            self._on_status("Streaming stopped")

    def start_recording(self, base_path: str | Path, participant_id: str) -> None:
        """Start recording raw ECG and auxiliary channels to disk."""
        self._recorder.start(
            base_path=base_path,
            participant_id=participant_id,
            device_info=self._device_info,
        )
        self.add_event("recording_start", source="system")
        if self._on_status:
            self._on_status(f"Recording to {base_path}...")

    def stop_recording(self) -> RecordingSummary | None:
        """Stop recording, finalize files, and return summary."""
        self.add_event("recording_stop", source="system")
        summary = self._recorder.stop()
        if self._on_status:
            self._on_status("Recording stopped and finalized.")
        return summary

    def add_event(self, label: str, source: str = "ui") -> None:
        """Register an event marker placed at current session time."""
        # Calculate t_session corresponding to now
        now_mono = time.monotonic()
        dt_since_frame = now_mono - self._last_frame_mono if self._last_frame_mono > 0 else 0.0
        event_t = self._last_t_session + dt_since_frame

        event = Event(
            t_session_s=round(event_t, 4),
            host_utc_ns=time.time_ns(),
            label=label,
            source=source,
        )
        self._recorder.add_event(event)
        if self._lsl:
            self._lsl.push_event(event)
        logger.info("Event added: %s at t=%.2fs (%s)", label, event_t, source)

    def on_raw_frame(self, raw: RawFrame) -> None:
        """Callback for incoming raw ECG frame."""
        self._last_frame_mono = time.monotonic()

        # Check if recovering from a stall
        if self._signal_is_stalled:
            self._signal_is_stalled = False
            self.add_event("signal_restored", source="system")
            if self._on_status:
                self._on_status("Signal restored.")

        # 1. Continuous clock & gap detection
        timed = self._clock.process(raw)
        self._last_t_session = float(timed.t_session_s[-1])

        # If gap or discontinuity occurred, notify detector
        if timed.gap_before != 0:
            self._detector.notify_gap()
            if timed.gap_before > 0:
                logger.warning("Packet gap of %d samples detected", timed.gap_before)
            else:
                logger.warning("Clock discontinuity detected")

        # 2. R-peak detection
        t_func = lambda idx: float(timed.t_session_s[0]) + (idx - timed.first_index) / self._clock.fs_estimated
        peaks = self._detector.process(
            first_index=timed.first_index,
            x_uv=timed.voltages_uv,
            index_to_t=t_func,
        )

        # 3. Submit to recorder if active
        if self._recorder.is_recording:
            self._recorder.submit_frame(timed)
            for p in peaks:
                self._recorder.submit_peak(p)

        # 4. Push to LSL if active
        if self._lsl:
            self._lsl.push_frame(timed)
            for p in peaks:
                self._lsl.push_peak(p)

        # 5. Notify UI
        if self._on_frame_ready:
            self._on_frame_ready(timed)

        if self._on_peak:
            for p in peaks:
                self._on_peak(p)

    def on_device_hr(self, hr: DeviceHR) -> None:
        """Callback for native Polar HR / RR measurement."""
        if self._recorder.is_recording:
            self._recorder.submit_hr(hr)
        if self._lsl:
            self._lsl.push_device_hr(hr)
        if self._on_device_hr:
            self._on_device_hr(hr)

    def on_source_disconnect(self, reason: str) -> None:
        """Callback when source connection is severed."""
        logger.warning("Source disconnected: %s", reason)
        if self._is_streaming and not self._is_reconnecting:
            asyncio.create_task(self._reconnect_loop())

    async def _watchdog_loop(self) -> None:
        """Periodic watchdog checking for signal stalls."""
        check_interval = max(0.05, min(0.5, self._cfg.DATA_WATCHDOG_S / 2.0))
        while self._is_streaming:
            await asyncio.sleep(check_interval)
            now = time.monotonic()
            if (now - self._last_frame_mono) > self._cfg.DATA_WATCHDOG_S and not self._signal_is_stalled:
                self._signal_is_stalled = True
                self.add_event("signal_lost", source="system")
                msg = "Signal stalled: no ECG data received"
                logger.warning(msg)
                if self._on_status:
                    self._on_status(msg)

    async def _reconnect_loop(self) -> None:
        """Automatic reconnection loop with exponential backoff."""
        if self._is_reconnecting or not self._cfg.AUTO_RECONNECT:
            return

        self._is_reconnecting = True
        if self._on_state:
            self._on_state("RECONNECTING")

        backoffs = self._cfg.RECONNECT_BACKOFF_S
        reconnected = False

        for attempt, delay in enumerate(backoffs, start=1):
            msg = f"Connection lost. Reconnecting (attempt {attempt}/{len(backoffs)} in {delay}s)..."
            logger.info(msg)
            if self._on_status:
                self._on_status(msg)

            await asyncio.sleep(delay)

            try:
                if self._source:
                    await self._source.start()
                    reconnected = True
                    break
            except Exception as e:  # noqa: BLE001
                logger.debug("Reconnection attempt %d failed: %s", attempt, e)

        self._is_reconnecting = False

        if reconnected:
            self._detector.reset()
            self.add_event("reconnected", source="system")
            logger.info("Successfully reconnected to device")
            if self._on_state:
                self._on_state("STREAMING")
            if self._on_status:
                self._on_status("Reconnected. Streaming resumed.")
        else:
            logger.error("All reconnection attempts failed.")
            self._is_streaming = False
            if self._on_state:
                self._on_state("ERROR")
            if self._on_status:
                self._on_status("Reconnection failed. Check device and battery.")

    def _handle_recorder_error(self, err_msg: str) -> None:
        logger.error("Recorder error: %s", err_msg)
        if self._on_error:
            self._on_error(f"Recording error: {err_msg}")

    async def shutdown(self) -> None:
        """Clean shutdown of session, streams, and recording."""
        if self._recorder.is_recording:
            try:
                self.stop_recording()
            except Exception as e:  # noqa: BLE001
                logger.error("Error stopping recorder during shutdown: %s", e)

        try:
            await self.stop_streaming()
        except Exception as e:  # noqa: BLE001
            logger.error("Error stopping streaming during shutdown: %s", e)
