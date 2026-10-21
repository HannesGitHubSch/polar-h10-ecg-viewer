"""
Data sources abstraction: BLE hardware device, synthetic real-time generator, and CSV replay.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from bleak.backends.device import BLEDevice
from polar_python.models import ECGData, HRData

from polar_ecg.analysis.io import load_recording
from polar_ecg.device import PolarH10
from polar_ecg.models import DeviceHR, RawFrame
from polar_ecg.synth import synth_ecg, synth_frames

logger = logging.getLogger(__name__)


class Source(Protocol):
    """Protocol for ECG data sources."""

    on_frame: Callable[[RawFrame], None]
    on_hr: Callable[[DeviceHR], None]
    on_disconnect: Callable[[str], None]

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def disconnect(self) -> None:
        ...


class DeviceSource:
    """
    Live Polar H10 BLE hardware data source.
    """

    def __init__(
        self,
        h10: PolarH10,
        ble_device: BLEDevice,
        on_frame: Callable[[RawFrame], None],
        on_hr: Callable[[DeviceHR], None],
        on_disconnect: Callable[[str], None],
        sample_rate: int = 130,
        stream_hr: bool = True,
    ):
        self.h10 = h10
        self.ble_device = ble_device
        self.on_frame = on_frame
        self.on_hr = on_hr
        self.on_disconnect = on_disconnect
        self.sample_rate = sample_rate
        self.stream_hr = stream_hr

    def _ecg_callback(self, data: ECGData) -> None:
        # Stamp host arrival monotonic time immediately
        rx_mono_ns = time.monotonic_ns()
        voltages = np.array(data.data, dtype=np.float32)
        frame = RawFrame(
            device_last_ns=data.timestamp,
            voltages_uv=voltages,
            rx_mono_ns=rx_mono_ns,
        )
        self.on_frame(frame)

    def _hr_callback(self, data: HRData) -> None:
        rx_utc_ns = time.time_ns()
        hr = DeviceHR(
            rx_utc_ns=rx_utc_ns,
            hr_bpm=data.heartrate,
            rr_ms=tuple(data.rr_intervals),
        )
        self.on_hr(hr)

    async def start(self) -> None:
        if not self.h10.is_connected:
            await self.h10.connect(
                self.ble_device,
                on_disconnect=lambda: self.on_disconnect("BLE connection lost"),
            )
        await self.h10.start_ecg(self._ecg_callback, sample_rate=self.sample_rate)
        if self.stream_hr:
            try:
                await self.h10.start_hr(self._hr_callback)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not start HR stream on Polar device: %s", e)

    async def stop(self) -> None:
        await self.h10.stop_all()

    async def disconnect(self) -> None:
        await self.h10.disconnect()


class SimulatedSource:
    """
    Synthetic real-time ECG generator for testing and demonstration without hardware.
    """

    def __init__(
        self,
        on_frame: Callable[[RawFrame], None],
        on_hr: Callable[[DeviceHR], None],
        on_disconnect: Callable[[str], None],
        duration_s: float = 3600.0,
        speed: float = 1.0,
        hr: float = 72.0,
        fs: float = 130.0,
        noise_uv: float = 15.0,
        wander_uv: float = 150.0,
        jitter_ms: float = 15.0,
        drop_rate: float = 0.0,
    ):
        self.on_frame = on_frame
        self.on_hr = on_hr
        self.on_disconnect = on_disconnect
        self.duration_s = duration_s
        self.speed = max(0.01, speed)
        self.hr = hr
        self.fs = fs
        self.noise_uv = noise_uv
        self.wander_uv = wander_uv
        self.jitter_ms = jitter_ms
        self.drop_rate = drop_rate

        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._generator_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def disconnect(self) -> None:
        await self.stop()

    async def _generator_loop(self) -> None:
        sig, _true_r = synth_ecg(
            duration_s=self.duration_s,
            fs=self.fs,
            hr=self.hr,
            noise_uv=self.noise_uv,
            wander_uv=self.wander_uv,
        )
        frames = synth_frames(
            sig,
            fs=self.fs,
            chunk_size=73,
            jitter_ms=self.jitter_ms,
            drop_rate=self.drop_rate,
        )

        dt_per_frame_s = (73.0 / self.fs) / self.speed
        last_hr_time = time.monotonic()

        for frame in frames:
            if not self._running:
                break

            # Override frame rx_mono_ns with current real monotonic time
            now_mono = time.monotonic_ns()
            stamped_frame = RawFrame(
                device_last_ns=frame.device_last_ns,
                voltages_uv=frame.voltages_uv,
                rx_mono_ns=now_mono,
            )
            self.on_frame(stamped_frame)

            # Emit periodic simulated HR (once per second)
            now = time.monotonic()
            if now - last_hr_time >= (1.0 / self.speed):
                last_hr_time = now
                self.on_hr(
                    DeviceHR(
                        rx_utc_ns=time.time_ns(),
                        hr_bpm=round(self.hr),
                        rr_ms=(round(60000.0 / self.hr, 2),),
                    )
                )

            await asyncio.sleep(dt_per_frame_s)


class ReplaySource:
    """
    Replay source streaming recorded CSV v1 or v2 files through the live pipeline.
    """

    def __init__(
        self,
        csv_path: str | Path,
        on_frame: Callable[[RawFrame], None],
        on_hr: Callable[[DeviceHR], None],
        on_disconnect: Callable[[str], None],
        speed: float = 1.0,
        chunk_size: int = 73,
    ):
        self.csv_path = Path(csv_path)
        self.on_frame = on_frame
        self.on_hr = on_hr
        self.on_disconnect = on_disconnect
        self.speed = max(0.01, speed)
        self.chunk_size = chunk_size

        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._replay_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def disconnect(self) -> None:
        await self.stop()

    async def _replay_loop(self) -> None:
        rec = load_recording(self.csv_path)
        df = rec.df
        n = len(df)
        if n == 0:
            return

        voltages = df["ecg_uv"].values

        fs = 130.0
        dt_ns = round(1e9 / fs)
        start_dev_ns = 1_000_000_000_000

        for i in range(0, n, self.chunk_size):
            if not self._running:
                break

            chunk_voltages = voltages[i : i + self.chunk_size]
            chunk_n = len(chunk_voltages)
            chunk_end_idx = i + chunk_n - 1

            now_mono = time.monotonic_ns()
            dev_last_ns = start_dev_ns + (chunk_end_idx * dt_ns)

            frame = RawFrame(
                device_last_ns=dev_last_ns,
                voltages_uv=chunk_voltages.astype(np.float32),
                rx_mono_ns=now_mono,
            )
            self.on_frame(frame)

            sleep_s = (chunk_n / fs) / self.speed
            await asyncio.sleep(sleep_s)
