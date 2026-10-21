"""
Unit tests for Session pipeline: streaming, recording, watchdog, and reconnection.
"""

import asyncio
import time
from pathlib import Path

import numpy as np

from polar_ecg.models import DeviceInfo, RawFrame
from polar_ecg.session import Session
from polar_ecg.settings import Settings
from polar_ecg.sources import SimulatedSource


def test_session_simulated_streaming_and_recording(tmp_path: Path):
    async def run_session():
        settings = Settings(DATA_WATCHDOG_S=2.0)
        received_frames = []
        received_peaks = []
        status_msgs = []

        session = Session(
            settings=settings,
            on_frame_ready=lambda f: received_frames.append(f),
            on_peak=lambda p: received_peaks.append(p),
            on_status=lambda s: status_msgs.append(s),
        )

        # Create fast simulated source (20x speed)
        source = SimulatedSource(
            on_frame=session.on_raw_frame,
            on_hr=session.on_device_hr,
            on_disconnect=session.on_source_disconnect,
            duration_s=30.0,
            speed=20.0,
            hr=75.0,
        )
        session.set_source(source, DeviceInfo("Simulated Polar H10", "SIM:00:11"))

        # Start streaming
        await session.start_streaming()
        assert session.is_streaming

        # Start recording
        rec_path = tmp_path / "ECG_TEST_SESSION"
        session.start_recording(rec_path, participant_id="P_SESSION")
        assert session.is_recording

        # Run for 0.8s real-time (~16s simulated ECG data)
        await asyncio.sleep(0.8)

        # Add an event
        session.add_event("marker_1", source="test")

        # Run another 0.4s (~8s simulated data)
        await asyncio.sleep(0.4)

        # Stop recording and streaming
        summary = session.stop_recording()
        await session.stop_streaming()

        assert not session.is_streaming
        assert not session.is_recording
        assert summary is not None
        assert summary.samples_written > 2000
        assert len(received_frames) > 0
        assert len(received_peaks) > 10

        # Check recorded files
        final_csv = tmp_path / "ECG_TEST_SESSION.csv"
        meta_json = tmp_path / "ECG_TEST_SESSION_meta.json"
        assert final_csv.exists()
        assert meta_json.exists()

    asyncio.run(run_session())


def test_session_watchdog_detects_stall():
    async def run_watchdog():
        settings = Settings(DATA_WATCHDOG_S=0.15)  # Fast watchdog for test
        statuses = []

        session = Session(
            settings=settings,
            on_status=lambda s: statuses.append(s),
        )

        class StalledSource:
            def __init__(self, on_frame):
                self.on_frame = on_frame

            async def start(self):
                raw = RawFrame(1000, np.zeros(73, dtype=np.float32), time.monotonic_ns())
                self.on_frame(raw)

            async def stop(self):
                pass

        session.set_source(StalledSource(session.on_raw_frame))
        await session.start_streaming()

        # Wait for watchdog to trigger (0.35s > 0.15s threshold)
        await asyncio.sleep(0.35)
        await session.stop_streaming()

        stall_alerts = [s for s in statuses if "stalled" in s.lower() or "lost" in s.lower()]
        assert len(stall_alerts) > 0, f"Watchdog did not report stall in {statuses}"

    asyncio.run(run_watchdog())
