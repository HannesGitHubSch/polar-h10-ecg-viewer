"""
Application entry point and CLI bootstrapping for Polar H10 ECG Viewer.
"""

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from polar_ecg.logging_setup import setup_logging
from polar_ecg.models import DeviceInfo
from polar_ecg.settings import load_settings
from polar_ecg.sources import ReplaySource, SimulatedSource
from polar_ecg.ui.main_window import MainWindow


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Polar H10 Real-time ECG Viewer & Recorder")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated synthetic ECG data")
    parser.add_argument("--replay", type=str, default=None, help="Replay a recorded CSV file")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation / replay speed multiplier")
    parser.add_argument("--lsl", action="store_true", help="Enable Lab Streaming Layer (LSL) outlet")
    parser.add_argument("--log-level", type=str, default=None, help="Logging level (DEBUG, INFO, etc.)")
    parser.add_argument("--autostart", action="store_true", help="Automatically begin streaming on startup")
    parser.add_argument("--record", type=str, default=None, help="Pre-set recording path")
    parser.add_argument("--participant", type=str, default="P001", help="Participant ID")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds then automatically close (useful for testing)",
    )

    args = parser.parse_args()

    # Load settings
    settings = load_settings()
    log_level = args.log_level if args.log_level else settings.LOG_LEVEL
    setup_logging(log_level)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    custom_source = None
    custom_device_info = None

    if args.simulate:
        custom_device_info = DeviceInfo("Simulated Polar H10", "SIM:00:00:00:00:00")
        custom_source = SimulatedSource(
            on_frame=lambda f: None,  # Will be bound by session in MainWindow
            on_hr=lambda hr: None,
            on_disconnect=lambda r: None,
            speed=args.speed,
        )

    elif args.replay:
        replay_path = Path(args.replay)
        custom_device_info = DeviceInfo(f"Replay: {replay_path.name}", "REPLAY:00:00")
        custom_source = ReplaySource(
            csv_path=replay_path,
            on_frame=lambda f: None,
            on_hr=lambda hr: None,
            on_disconnect=lambda r: None,
            speed=args.speed,
        )

    window = MainWindow(
        settings=settings,
        custom_source=custom_source,
        custom_device_info=custom_device_info,
    )
    window.show()

    # Bind source callbacks now that session is created
    if custom_source is not None:
        custom_source.on_frame = window.session.on_raw_frame
        custom_source.on_hr = window.session.on_device_hr
        custom_source.on_disconnect = window.session.on_source_disconnect

    if args.participant:
        window.participant_id_input.setText(args.participant)

    if args.record:
        rec_path = Path(args.record).with_suffix("")
        window.recording_base_path = rec_path

    if args.autostart or args.simulate or args.replay:
        # Schedule auto-start after event loop begins
        QTimer.singleShot(100, window._on_toggle_streaming)

    if args.record and (args.autostart or args.simulate or args.replay):
        QTimer.singleShot(300, window._on_toggle_recording)

    if args.duration is not None:
        def _auto_quit():
            if window.session.is_recording:
                window.session.stop_recording()
            window.close()

        duration_ms = round(args.duration * 1000)
        QTimer.singleShot(duration_ms, _auto_quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
