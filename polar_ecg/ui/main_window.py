"""
Main application window for Polar H10 ECG visualization and recording.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from polar_ecg.device import PolarH10
from polar_ecg.models import DeviceHR, DeviceInfo, RPeak, TimedFrame
from polar_ecg.session import Session
from polar_ecg.settings import Settings
from polar_ecg.sources import DeviceSource
from polar_ecg.ui.async_runner import AsyncRunner
from polar_ecg.ui.plots import BpmPlotWidget, EcgPlotWidget
from polar_ecg.ui.state import ENABLED_CONTROLS, AppState

logger = logging.getLogger(__name__)


class BridgeSignals(QObject):
    """Qt signal bridge for thread-safe session callbacks."""

    frame_ready = pyqtSignal(object)  # TimedFrame
    peak_detected = pyqtSignal(object)  # RPeak
    device_hr = pyqtSignal(object)  # DeviceHR
    state_changed = pyqtSignal(str)  # AppState string
    status_msg = pyqtSignal(str)  # Status message
    error_msg = pyqtSignal(str)  # Error message


class MainWindow(QMainWindow):
    """
    Main graphical interface for Polar H10 ECG streaming, visualization, and recording.
    """

    def __init__(
        self,
        settings: Settings,
        custom_source: Any | None = None,
        custom_device_info: DeviceInfo | None = None,
        enable_lsl: bool = False,
    ):
        super().__init__()
        self.cfg = settings
        self.custom_source = custom_source
        self.custom_device_info = custom_device_info
        self.enable_lsl = enable_lsl

        # Thread-safe runner & bridge signals
        self.runner = AsyncRunner()
        self.runner.start()
        self.bridge = BridgeSignals()

        # Discovered devices cache: name -> (DeviceInfo, BLEDevice)
        self.discovered_devices: dict[str, tuple[DeviceInfo, Any]] = {}
        self.h10_backend = PolarH10()

        # Core session
        self.session = Session(
            settings=self.cfg,
            enable_lsl=self.enable_lsl,
            on_frame_ready=lambda f: self.bridge.frame_ready.emit(f),
            on_peak=lambda p: self.bridge.peak_detected.emit(p),
            on_device_hr=lambda hr: self.bridge.device_hr.emit(hr),
            on_state=lambda s: self.bridge.state_changed.emit(s),
            on_status=lambda m: self.bridge.status_msg.emit(m),
            on_error=lambda e: self.bridge.error_msg.emit(e),
        )

        # Pre-configured source if simulated or replay
        if self.custom_source is not None:
            self.session.set_source(self.custom_source, self.custom_device_info)

        # UI state
        self.current_state = AppState.IDLE
        self.recording_base_path: Path | None = None
        self.last_frame_mono = 0.0
        self.last_peak_mono = 0.0
        self.last_session_t = 0.0

        self._init_ui()
        self._connect_signals()

        # Display refresh timer (e.g. 30 FPS)
        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self._on_display_tick)
        self.display_timer.start(round(1000.0 / self.cfg.DISPLAY_FPS))

        # Apply initial state
        if self.custom_source is not None:
            self.apply_state(AppState.CONNECTED)
        else:
            self.apply_state(AppState.IDLE)

    def _init_ui(self) -> None:
        self.setWindowTitle("Polar H10 ECG Viewer & Recorder")
        self.resize(1280, 820)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. Alert Banner (hidden by default; shown on signal loss/reconnect)
        self.banner = QFrame()
        self.banner.setStyleSheet("background-color: #D32F2F; border-radius: 4px; padding: 6px;")
        banner_layout = QHBoxLayout(self.banner)
        self.banner_label = QLabel("⚠ Signal Lost: Reconnecting to Polar H10...")
        self.banner_label.setStyleSheet("color: white; font-weight: bold; font-size: 11pt;")
        banner_layout.addWidget(self.banner_label)
        self.banner.setVisible(False)
        main_layout.addWidget(self.banner)

        # 2. Control Panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        # 3. ECG Plot Panel
        ecg_group = QGroupBox("ECG Signal (Real-time)")
        ecg_layout = QVBoxLayout(ecg_group)

        ecg_toolbar = QHBoxLayout()
        self.rpeak_toggle = QCheckBox("Show R-peaks")
        self.rpeak_toggle.setChecked(self.cfg.SHOW_RPEAKS_DEFAULT)
        self.rpeak_toggle.toggled.connect(self._on_toggle_rpeaks)
        ecg_toolbar.addWidget(self.rpeak_toggle)

        ecg_toolbar.addSpacing(20)
        self.missing_label = QLabel("Missing samples: 0")
        self.missing_label.setStyleSheet("color: #666;")
        ecg_toolbar.addWidget(self.missing_label)

        ecg_toolbar.addStretch()
        ecg_layout.addLayout(ecg_toolbar)

        self.ecg_plot_widget = pg.PlotWidget()
        self.ecg_plot = EcgPlotWidget(self.ecg_plot_widget, self.cfg)
        ecg_layout.addWidget(self.ecg_plot_widget)
        main_layout.addWidget(ecg_group, stretch=3)

        # 4. BPM Plot Panel
        bpm_group = QGroupBox("Heart Rate (BPM)")
        bpm_layout = QVBoxLayout(bpm_group)

        bpm_toolbar = QHBoxLayout()
        self.add_avg_btn = QPushButton(f"Add Average Line ({int(self.cfg.BPM_AVERAGE_WINDOW_S)}s)")
        self.add_avg_btn.clicked.connect(self._on_add_average_line)
        bpm_toolbar.addWidget(self.add_avg_btn)

        self.clear_avg_btn = QPushButton("Clear All Lines")
        self.clear_avg_btn.clicked.connect(self._on_clear_average_lines)
        bpm_toolbar.addWidget(self.clear_avg_btn)

        bpm_toolbar.addStretch()

        self.bpm_label = QLabel("Detected BPM: --")
        self.bpm_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.bpm_label.setStyleSheet(f"color: {self.cfg.COLOR_BPM_LINE};")
        bpm_toolbar.addWidget(self.bpm_label)

        bpm_toolbar.addSpacing(20)
        self.device_hr_label = QLabel("Polar HR: --")
        self.device_hr_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.device_hr_label.setStyleSheet("color: #0288D1;")
        bpm_toolbar.addWidget(self.device_hr_label)

        bpm_toolbar.addSpacing(15)
        self.overlay_hr_toggle = QCheckBox("Overlay Polar HR")
        self.overlay_hr_toggle.setChecked(False)
        self.overlay_hr_toggle.toggled.connect(self._on_toggle_overlay_hr)
        bpm_toolbar.addWidget(self.overlay_hr_toggle)

        bpm_layout.addLayout(bpm_toolbar)

        self.bpm_plot_widget = pg.PlotWidget()
        self.bpm_plot = BpmPlotWidget(self.bpm_plot_widget, self.cfg)
        # Link X-axes of both plots so they scroll identically
        self.bpm_plot_widget.setXLink(self.ecg_plot_widget)
        bpm_layout.addWidget(self.bpm_plot_widget)
        main_layout.addWidget(bpm_group, stretch=2)

        # Status Bar
        self.statusBar().showMessage("Ready - Select or scan for a Polar H10 device")

    def _create_control_panel(self) -> QGroupBox:
        panel = QGroupBox("Controls")
        layout = QHBoxLayout(panel)

        # Scan & Device selection
        layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(200)
        layout.addWidget(self.device_combo)

        self.scan_btn = QPushButton("Scan Devices")
        self.scan_btn.clicked.connect(self._on_scan)
        layout.addWidget(self.scan_btn)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        layout.addWidget(self.disconnect_btn)

        self.stream_btn = QPushButton("Start Streaming")
        self.stream_btn.clicked.connect(self._on_toggle_streaming)
        layout.addWidget(self.stream_btn)

        layout.addSpacing(15)

        # Recording setup
        layout.addWidget(QLabel("Participant ID:"))
        self.participant_id_input = QLineEdit()
        self.participant_id_input.setPlaceholderText("P001")
        self.participant_id_input.setMaximumWidth(80)
        layout.addWidget(self.participant_id_input)

        self.select_loc_btn = QPushButton("Select Save Location")
        self.select_loc_btn.clicked.connect(self._on_select_save_location)
        layout.addWidget(self.select_loc_btn)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.record_btn.clicked.connect(self._on_toggle_recording)
        layout.addWidget(self.record_btn)

        layout.addSpacing(15)

        # Event Markers
        layout.addWidget(QLabel("Event:"))
        self.event_input = QLineEdit()
        self.event_input.setPlaceholderText("e.g. baseline")
        self.event_input.setMaximumWidth(120)
        layout.addWidget(self.event_input)

        self.mark_btn = QPushButton("Mark")
        self.mark_btn.clicked.connect(self._on_mark_event)
        layout.addWidget(self.mark_btn)

        layout.addStretch()
        return panel

    def _connect_signals(self) -> None:
        self.bridge.frame_ready.connect(self._handle_frame_ready)
        self.bridge.peak_detected.connect(self._handle_peak_detected)
        self.bridge.device_hr.connect(self._handle_device_hr)
        self.bridge.state_changed.connect(self._handle_state_changed)
        self.bridge.status_msg.connect(self._handle_status_msg)
        self.bridge.error_msg.connect(self._handle_error_msg)
        self.runner.signals.error.connect(self._handle_async_error)

    def apply_state(self, state: AppState) -> None:
        """
        Enforce UI control enabled/disabled status strictly from the state table.
        """
        self.current_state = state
        enabled = ENABLED_CONTROLS.get(state, set())

        self.scan_btn.setEnabled("scan" in enabled)
        self.device_combo.setEnabled("device_combo" in enabled)
        self.connect_btn.setEnabled("connect" in enabled)
        self.disconnect_btn.setEnabled("disconnect" in enabled)
        self.stream_btn.setEnabled("stream" in enabled)
        self.participant_id_input.setEnabled("participant_id" in enabled)
        self.select_loc_btn.setEnabled("select_location" in enabled)
        self.record_btn.setEnabled("record" in enabled)
        self.mark_btn.setEnabled("mark" in enabled)
        self.rpeak_toggle.setEnabled("rpeak_toggle" in enabled)
        self.add_avg_btn.setEnabled("add_avg" in enabled)
        self.clear_avg_btn.setEnabled("clear_avg" in enabled)

        # Update button text/styles
        if state == AppState.SCANNING:
            self.scan_btn.setText("Scanning...")
        else:
            self.scan_btn.setText("Scan Devices")

        if state == AppState.CONNECTING:
            self.connect_btn.setText("Connecting...")
        elif state in (AppState.CONNECTED, AppState.STREAMING, AppState.RECONNECTING):
            self.connect_btn.setText("Connected")
        else:
            self.connect_btn.setText("Connect")

        if state == AppState.STREAMING:
            self.stream_btn.setText("Stop Streaming")
            self.stream_btn.setStyleSheet("background-color: #FF9800; color: white;")
        else:
            self.stream_btn.setText("Start Streaming")
            self.stream_btn.setStyleSheet("")

        # Alert banner
        if state == AppState.RECONNECTING:
            self.banner.setVisible(True)
            self.banner_label.setText("⚠ Connection lost! Automatically reconnecting to Polar H10...")
        elif state == AppState.ERROR:
            self.banner.setVisible(True)
            self.banner_label.setText("✗ Device disconnected. Reconnection failed. Check Bluetooth and battery.")
        else:
            self.banner.setVisible(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle numeric shortcut keys (1-9) and Space for rapid event markers."""
        key = event.key()
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            num = key - Qt.Key.Key_0
            label = f"event_{num}"
            self._trigger_event(label, source="key")
            event.accept()
        elif key == Qt.Key.Key_Space:
            text = self.event_input.text().strip()
            label = text if text else "marker"
            self._trigger_event(label, source="key")
            event.accept()
        else:
            super().keyPressEvent(event)

    def _trigger_event(self, label: str, source: str = "ui") -> None:
        if self.session.is_streaming:
            self.session.add_event(label, source=source)
            self.ecg_plot.add_marker(self.last_session_t, label=label, color="#4CAF50")
            self.statusBar().showMessage(f"Marked event: '{label}' at t={self.last_session_t:.2f}s")

    # --- Slots & Handlers ---

    def _on_scan(self) -> None:
        self.apply_state(AppState.SCANNING)
        self.statusBar().showMessage("Scanning for Polar BLE devices...")

        def _on_scan_done(found_devices: list[tuple[DeviceInfo, Any]]):
            self.discovered_devices.clear()
            self.device_combo.clear()

            if found_devices:
                for info, ble in found_devices:
                    label = f"{info.name} ({info.address})"
                    self.discovered_devices[label] = (info, ble)
                    self.device_combo.addItem(label)
                self.apply_state(AppState.READY)
                self.statusBar().showMessage(f"Found {len(found_devices)} Polar device(s)")
            else:
                self.device_combo.addItem("No Polar devices found")
                self.apply_state(AppState.IDLE)
                self.statusBar().showMessage("No Polar devices found.")

        self.runner.submit(
            self.h10_backend.scan(timeout=self.cfg.SCAN_TIMEOUT_S),
            op_name="scan_devices",
            on_done=_on_scan_done,
        )

    def _on_connect(self) -> None:
        selected_label = self.device_combo.currentText()
        if selected_label not in self.discovered_devices:
            self.statusBar().showMessage("Please select a discovered Polar device")
            return

        info, ble_device = self.discovered_devices[selected_label]
        self.apply_state(AppState.CONNECTING)
        self.statusBar().showMessage(f"Connecting to {info.name}...")

        # Create device source and set on session
        source = DeviceSource(
            h10=self.h10_backend,
            ble_device=ble_device,
            on_frame=self.session.on_raw_frame,
            on_hr=self.session.on_device_hr,
            on_disconnect=self.session.on_source_disconnect,
            sample_rate=int(self.cfg.ECG_FS_NOMINAL),
            stream_hr=self.cfg.RECORD_DEVICE_HR,
        )
        self.session.set_source(source, info)

        def _on_connected(_):
            self.apply_state(AppState.CONNECTED)
            self.statusBar().showMessage(f"Connected to {info.name}")

        self.runner.submit(
            self.h10_backend.connect(
                ble_device,
                on_disconnect=lambda: self.session.on_source_disconnect("BLE disconnect"),
                timeout=self.cfg.CONNECT_TIMEOUT_S,
            ),
            op_name="connect_device",
            on_done=_on_connected,
        )

    async def _async_disconnect(self) -> None:
        await self.session.disconnect()
        await self.h10_backend.disconnect()

    def _on_disconnect(self) -> None:
        self.statusBar().showMessage("Disconnecting...")

        def _on_disconnected(_):
            if self.discovered_devices:
                self.apply_state(AppState.READY)
            else:
                self.apply_state(AppState.IDLE)
            self.statusBar().showMessage("Disconnected.")

        self.runner.submit(
            self._async_disconnect(),
            op_name="disconnect_device",
            on_done=_on_disconnected,
        )

    def _on_toggle_streaming(self) -> None:
        if not self.session.is_streaming:
            self.ecg_plot.clear()
            self.bpm_plot.clear()
            self.last_peak_mono = 0.0
            self.bpm_label.setText("Detected BPM: --")
            self.device_hr_label.setText("Polar HR: --")

            def _on_started(_):
                self.apply_state(AppState.STREAMING)

            self.runner.submit(
                self.session.start_streaming(),
                op_name="start_streaming",
                on_done=_on_started,
            )
        else:
            def _on_stopped(_):
                self.apply_state(AppState.CONNECTED)

            self.runner.submit(
                self.session.stop_streaming(),
                op_name="stop_streaming",
                on_done=_on_stopped,
            )

    def _on_select_save_location(self) -> None:
        pid = self.participant_id_input.text().strip()
        if not pid:
            self.statusBar().showMessage("Please enter a Participant ID first")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"ECG_{pid}_{ts}.csv"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Save Location for ECG Recording",
            default_name,
            "CSV Files (*.csv)",
        )
        if file_path:
            p = Path(file_path)
            # Base path without extension
            self.recording_base_path = p.with_suffix("")
            self.select_loc_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.select_loc_btn.setText("Location Selected ✓")
            self.statusBar().showMessage(f"Save location: {file_path}")

    def _on_toggle_recording(self) -> None:
        if not self.session.is_recording:
            pid = self.participant_id_input.text().strip()
            if not pid:
                self.statusBar().showMessage("Please enter Participant ID before recording")
                return

            if not self.recording_base_path:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.recording_base_path = Path.cwd() / "recordings" / f"ECG_{pid}_{ts}"

            self.session.start_recording(self.recording_base_path, participant_id=pid)
            self.record_btn.setText("Stop Recording")
            self.record_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
            self.ecg_plot.add_marker(self.last_session_t, "Recording Start", color="#4CAF50")
            self.statusBar().showMessage(f"🔴 Recording: {self.recording_base_path}.csv")
        else:
            summary = self.session.stop_recording()
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.ecg_plot.add_marker(self.last_session_t, "Recording Stop", color="#F44336")
            self.recording_base_path = None
            self.select_loc_btn.setText("Select Save Location")
            self.select_loc_btn.setStyleSheet("")
            if summary:
                self.statusBar().showMessage(
                    f"Saved {summary.samples_written} samples ({summary.samples_missing} gaps) to {summary.file_path.name}"
                )

    def _on_mark_event(self) -> None:
        text = self.event_input.text().strip()
        label = text if text else "marker"
        self._trigger_event(label, source="ui")
        self.event_input.clear()

    def _on_toggle_rpeaks(self, checked: bool) -> None:
        self.ecg_plot.set_rpeaks_visible(checked)

    def _on_toggle_overlay_hr(self, checked: bool) -> None:
        self.bpm_plot.set_device_hr_visible(checked)

    def _on_add_average_line(self) -> None:
        avg = self.bpm_plot.add_average_line()
        if avg is not None:
            self.statusBar().showMessage(f"Added average line: {avg:.1f} BPM")
        else:
            self.statusBar().showMessage("Not enough BPM history for average line")

    def _on_clear_average_lines(self) -> None:
        self.bpm_plot.clear_average_lines()
        self.statusBar().showMessage("Cleared average lines")

    def _handle_frame_ready(self, frame: TimedFrame) -> None:
        self.last_frame_mono = time.monotonic()
        self.last_session_t = float(frame.t_session_s[-1])
        self.ecg_plot.add_timed_frame(frame)
        self.missing_label.setText(f"Missing samples: {self.session.clock.total_missing}")

    def _handle_peak_detected(self, peak: RPeak) -> None:
        self.last_peak_mono = time.monotonic()
        self.ecg_plot.add_peak(peak)
        self.bpm_plot.add_peak(peak)
        if peak.bpm is not None:
            self.bpm_label.setText(f"Detected BPM: {peak.bpm:.1f}")

    def _handle_device_hr(self, hr: DeviceHR) -> None:
        self.device_hr_label.setText(f"Polar HR: {hr.hr_bpm}")
        self.bpm_plot.add_device_hr(self.last_session_t, hr)

    def _handle_state_changed(self, state_str: str) -> None:
        try:
            state = AppState[state_str]
            self.apply_state(state)
        except KeyError:
            pass

    def _handle_status_msg(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def _handle_error_msg(self, msg: str) -> None:
        self.statusBar().showMessage(f"Error: {msg}")
        QMessageBox.warning(self, "Session Warning", msg)

    def _handle_async_error(self, op: str, err: str) -> None:
        self.statusBar().showMessage(f"Failed {op}: {err}")
        QMessageBox.critical(self, "Bluetooth Operation Error", f"Operation '{op}' failed:\n{err}")
        if self.discovered_devices:
            self.apply_state(AppState.READY)
        else:
            self.apply_state(AppState.IDLE)

    def _on_display_tick(self) -> None:
        """Executed at DISPLAY_FPS to smoothly redraw plotting curves."""
        if self.session.is_streaming:
            # Current time projects smoothly from last frame arrival
            now_mono = time.monotonic()
            dt = now_mono - self.last_frame_mono if self.last_frame_mono > 0 else 0.0
            smooth_session_t = self.last_session_t + dt

            # If no heart beat detected for > 3.0s, indicate staleness
            if self.last_peak_mono > 0 and (now_mono - self.last_peak_mono) > 3.0:
                self.bpm_label.setText("Detected BPM: --")

            self.ecg_plot.redraw(smooth_session_t)
            self.bpm_plot.redraw(smooth_session_t)

    def closeEvent(self, event) -> None:
        """Handle application close with safe shutdown and confirmation."""
        import os

        is_offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        if self.session.is_recording and not is_offscreen and self.isVisible():
            reply = QMessageBox.question(
                self,
                "Active Recording",
                "A recording session is currently active. Stop and save before exiting?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        self.display_timer.stop()
        self.runner.shutdown(session=self.session, timeout=4.0)
        event.accept()
