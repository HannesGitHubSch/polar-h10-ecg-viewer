"""
Polar ECG Viewer - Main Application
Real-time ECG visualization with R-peak detection and BPM monitoring
"""

import sys
import asyncio
import time
import csv
from datetime import datetime
from typing import Optional, List
from collections import deque
from threading import Thread

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QGroupBox, QLineEdit,
    QFileDialog
)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject, Qt
import pyqtgraph as pg

from polar_backend import PolarH10Handler, ECGSample, RPeak

try:
    from config import (
        DISPLAY_WINDOW_SECONDS, ECG_BUFFER_SECONDS, BPM_HISTORY_POINTS,
        BPM_AVERAGE_WINDOW_SECONDS, BPM_Y_AXIS_PADDING, SHOW_RPEAKS_DEFAULT,
        ENABLE_SMOOTH_SCROLLING, SCROLL_UPDATE_INTERVAL_MS,
        SMOOTH_SAMPLE_INSERTION, SAMPLES_PER_FRAME,
        COLOR_ECG_LINE, COLOR_RPEAK_MARKER, COLOR_BPM_LINE, COLOR_AVERAGE_LINE
    )
except ImportError:
    # Fallback values
    DISPLAY_WINDOW_SECONDS = 20
    ECG_BUFFER_SECONDS = 30
    BPM_HISTORY_POINTS = 500
    BPM_AVERAGE_WINDOW_SECONDS = 30
    BPM_Y_AXIS_PADDING = 10
    SHOW_RPEAKS_DEFAULT = False
    ENABLE_SMOOTH_SCROLLING = True
    SCROLL_UPDATE_INTERVAL_MS = 30
    SMOOTH_SAMPLE_INSERTION = True
    SAMPLES_PER_FRAME = 5
    COLOR_ECG_LINE = '#2E86AB'
    COLOR_RPEAK_MARKER = '#FF0000'
    COLOR_BPM_LINE = '#A23B72'
    COLOR_AVERAGE_LINE = '#F18F01'


class SignalBridge(QObject):
    """Qt signals for thread-safe communication"""
    ecg_data = pyqtSignal(list)  # List[ECGSample]
    rpeak_detected = pyqtSignal(object)  # RPeak
    connection_status = pyqtSignal(str)  # Status message
    devices_found = pyqtSignal(list)  # List[str]


class PolarECGViewer(QMainWindow):
    """Main application window for ECG viewing"""
    
    def __init__(self):
        super().__init__()
        
        # Calculate buffer sizes from config
        ecg_buffer_size = int(130 * ECG_BUFFER_SECONDS)  # 130 Hz sample rate
        
        # Data buffers (displayed data)
        self.ecg_times = deque(maxlen=ecg_buffer_size)
        self.ecg_voltages = deque(maxlen=ecg_buffer_size)
        self.rpeak_times = deque(maxlen=100)
        self.rpeak_voltages = deque(maxlen=100)
        self.bpm_times = deque(maxlen=BPM_HISTORY_POINTS)
        self.bpm_values = deque(maxlen=BPM_HISTORY_POINTS)
        
        # Pending samples buffer (for smooth insertion)
        self.pending_samples = deque()  # Samples waiting to be displayed
        self.enable_smooth_scrolling = ENABLE_SMOOTH_SCROLLING
        self.smooth_sample_insertion = SMOOTH_SAMPLE_INSERTION
        
        # Time tracking for scrolling
        self.streaming_start_time = None
        self.is_streaming_active = False
        self.last_sample_time = 0.0  # Our own timestamp counter
        
        # Mapping from Polar timestamps to our timestamps (for R-peak matching)
        self.timestamp_mapping = {}  # {polar_timestamp: our_timestamp}
        
        # Average lines management
        self.bpm_window_seconds = BPM_AVERAGE_WINDOW_SECONDS
        self.average_lines = []  # List of InfiniteLine objects
        
        # R-peak visibility
        self.show_rpeaks = SHOW_RPEAKS_DEFAULT
        
        # Recording state
        self.is_recording = False
        self.recording_file = None
        self.recording_csv_writer = None
        self.participant_id = ""
        self.recording_save_path = None  # Pre-selected save location
        
        # Recording visual markers
        self.recording_markers = []  # List of InfiniteLine objects for start/stop markers
        
        # Backend
        self.signals = SignalBridge()
        self.polar_handler: Optional[PolarH10Handler] = None
        self.asyncio_thread: Optional[Thread] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Setup UI
        self.init_ui()
        
        # Setup smooth scrolling timer if enabled
        if self.enable_smooth_scrolling:
            self.scroll_timer = QTimer()
            self.scroll_timer.timeout.connect(self.update_time_based_scroll)
            self.scroll_timer.start(SCROLL_UPDATE_INTERVAL_MS)  # e.g., 30ms = ~33 FPS
        
        # Connect signals
        self.signals.ecg_data.connect(self.update_ecg_plot)
        self.signals.rpeak_detected.connect(self.update_rpeak)
        self.signals.connection_status.connect(self.update_status)
        self.signals.devices_found.connect(self.update_device_list)
        
        # Start asyncio thread immediately
        self.start_asyncio_thread()
        
        # Link X-axes of both plots
        self.bpm_plot.setXLink(self.ecg_plot)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Polar ECG Viewer - Teaching Demo")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # ECG Plot
        ecg_group = QGroupBox("ECG Signal (Real-time)")
        ecg_layout = QVBoxLayout()
        
        # ECG controls
        ecg_controls = QHBoxLayout()
        self.rpeak_toggle = QCheckBox("Show R-peaks")
        self.rpeak_toggle.setChecked(SHOW_RPEAKS_DEFAULT)
        self.rpeak_toggle.stateChanged.connect(self.toggle_rpeak_visibility)
        ecg_controls.addWidget(self.rpeak_toggle)
        ecg_controls.addStretch()
        ecg_layout.addLayout(ecg_controls)
        
        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setLabel('left', 'Voltage', units='µV')
        self.ecg_plot.setLabel('bottom', 'Time', units='s')
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ecg_plot.enableAutoRange(axis='y', enable=True)  # Auto-scale Y
        self.ecg_plot.setAutoVisible(y=True)  # Auto-range to visible data
        
        # ECG curve
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen(color=COLOR_ECG_LINE, width=2))
        
        # R-peaks scatter
        self.rpeak_scatter = pg.ScatterPlotItem(
            size=10, 
            brush=pg.mkBrush(COLOR_RPEAK_MARKER + 'CC'),  # Add alpha
            symbol='o'
        )
        self.rpeak_scatter.setVisible(SHOW_RPEAKS_DEFAULT)
        self.ecg_plot.addItem(self.rpeak_scatter)
        
        ecg_layout.addWidget(self.ecg_plot)
        ecg_group.setLayout(ecg_layout)
        layout.addWidget(ecg_group, stretch=3)
        
        # BPM Plot
        bpm_group = QGroupBox("Heart Rate (BPM)")
        bpm_layout = QVBoxLayout()
        
        # BPM controls
        bpm_controls = QHBoxLayout()
        
        # Average line buttons
        self.add_avg_btn = QPushButton(f"Add Average Line ({BPM_AVERAGE_WINDOW_SECONDS}s)")
        self.add_avg_btn.clicked.connect(self.add_average_line)
        self.add_avg_btn.setEnabled(False)  # Enable after streaming starts
        bpm_controls.addWidget(self.add_avg_btn)
        
        self.clear_avg_btn = QPushButton("Clear All Lines")
        self.clear_avg_btn.clicked.connect(self.clear_average_lines)
        self.clear_avg_btn.setEnabled(False)
        bpm_controls.addWidget(self.clear_avg_btn)
        
        bpm_controls.addStretch()
        
        self.bpm_label = QLabel("Current BPM: --")
        self.bpm_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        bpm_controls.addWidget(self.bpm_label)
        
        bpm_layout.addLayout(bpm_controls)
        
        self.bpm_plot = pg.PlotWidget()
        self.bpm_plot.setLabel('left', 'BPM')
        self.bpm_plot.setLabel('bottom', 'Time', units='s')
        self.bpm_plot.showGrid(x=True, y=True, alpha=0.3)
        self.bpm_plot.setYRange(40, 180)  # Initial range, will be updated dynamically
        # Don't use auto-range for Y since we manage it manually with padding
        
        # BPM curve
        self.bpm_curve = self.bpm_plot.plot(
            pen=pg.mkPen(color=COLOR_BPM_LINE, width=2),
            symbol='o',
            symbolSize=5,
            symbolBrush=pg.mkBrush(COLOR_BPM_LINE + 'AA')  # Add alpha
        )
        
        bpm_layout.addWidget(self.bpm_plot)
        bpm_group.setLayout(bpm_layout)
        layout.addWidget(bpm_group, stretch=2)
        
        # Status bar
        self.statusBar().showMessage("Ready - Click 'Scan Devices' to start")
        
    def create_control_panel(self) -> QWidget:
        """Create the control panel with buttons and device selection"""
        panel = QGroupBox("Controls")
        layout = QHBoxLayout()
        
        # Device selection
        layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("No devices found")
        layout.addWidget(self.device_combo)
        
        # Scan button
        self.scan_btn = QPushButton("Scan Devices")
        self.scan_btn.clicked.connect(self.scan_devices)
        layout.addWidget(self.scan_btn)
        
        # Connect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_device)
        self.connect_btn.setEnabled(False)
        layout.addWidget(self.connect_btn)
        
        # Start/Stop button
        self.stream_btn = QPushButton("Start Streaming")
        self.stream_btn.clicked.connect(self.toggle_streaming)
        self.stream_btn.setEnabled(False)
        layout.addWidget(self.stream_btn)
        
        layout.addSpacing(20)
        
        # Recording controls - two-step process
        layout.addWidget(QLabel("Participant ID:"))
        self.participant_id_input = QLineEdit()
        self.participant_id_input.setPlaceholderText("e.g., P001")
        self.participant_id_input.setMaximumWidth(100)
        layout.addWidget(self.participant_id_input)
        
        self.select_location_btn = QPushButton("Select Save Location")
        self.select_location_btn.clicked.connect(self.select_save_location)
        self.select_location_btn.setEnabled(False)
        self.select_location_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        layout.addWidget(self.select_location_btn)
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.record_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        layout.addWidget(self.record_btn)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def scan_devices(self):
        """Scan for Polar devices"""
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")
        self.statusBar().showMessage("Scanning for Polar devices...")
        
        # Schedule async scanning
        asyncio.run_coroutine_threadsafe(
            self._scan_devices_async(),
            self.event_loop
        )
        
    async def _scan_devices_async(self):
        """Async device scanning"""
        handler = PolarH10Handler(None, None)
        devices = await handler.scan_devices(timeout=5.0)
        
        if devices:
            self.signals.devices_found.emit(devices)
            self.signals.connection_status.emit(f"Found {len(devices)} device(s)")
        else:
            self.signals.devices_found.emit([])
            self.signals.connection_status.emit("No Polar devices found")
            
    def update_device_list(self, devices: List[str]):
        """Update device combo box with found devices"""
        self.device_combo.clear()
        if devices:
            self.device_combo.addItems(devices)
            self.connect_btn.setEnabled(True)
        else:
            self.device_combo.addItem("No devices found")
            self.connect_btn.setEnabled(False)
            
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan Devices")
        
    def connect_device(self):
        """Connect to selected device"""
        device_name = self.device_combo.currentText()
        
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Connecting...")
        self.statusBar().showMessage(f"Connecting to {device_name}...")
        
        # Create handler with callbacks
        self.polar_handler = PolarH10Handler(
            ecg_callback=self._ecg_callback,
            rpeak_callback=self._rpeak_callback,
            sample_rate=130
        )
        
        # Connect in background thread
        asyncio.run_coroutine_threadsafe(
            self._connect_async(device_name),
            self.event_loop
        )
        
    async def _connect_async(self, device_name: str):
        """Async device connection"""
        success = await self.polar_handler.connect(device_name)
        
        if success:
            self.signals.connection_status.emit(f"Connected to {device_name}")
        else:
            self.signals.connection_status.emit(f"Failed to connect to {device_name}")
            
    def update_status(self, message: str):
        """Update status message"""
        self.statusBar().showMessage(message)
        
        if "Connected" in message:
            self.connect_btn.setText("Connected")
            self.stream_btn.setEnabled(True)
        elif "Failed" in message:
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("Connect")
            
    def select_save_location(self):
        """Select save location for recording (before starting)"""
        participant_id = self.participant_id_input.text().strip()
        if not participant_id:
            self.statusBar().showMessage("Please enter a Participant ID first")
            return
        
        # Open file dialog
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"ECG_{participant_id}_{timestamp}.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Save Location for ECG Recording",
            default_filename,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            # User cancelled
            return
        
        # Store the path
        self.recording_save_path = file_path
        
        # Update UI
        self.select_location_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.select_location_btn.setText("Location Selected ✓")
        self.record_btn.setEnabled(True)
        self.statusBar().showMessage(f"Ready to record to: {file_path}")
    
    def toggle_recording(self):
        """Start or stop recording ECG data to CSV"""
        if not self.is_recording:
            # Start recording
            if not self.recording_save_path:
                self.statusBar().showMessage("Please select save location first")
                return
            
            participant_id = self.participant_id_input.text().strip()
            if not participant_id:
                self.statusBar().showMessage("Please enter a Participant ID")
                return
            
            try:
                # Open the pre-selected file immediately
                self.recording_file = open(self.recording_save_path, 'w', newline='')
                self.recording_csv_writer = csv.writer(self.recording_file)
                # Write header
                self.recording_csv_writer.writerow([
                    'participant_id',
                    'sample_datetime',
                    'packet_received_datetime',
                    'polar_timestamp_s',
                    'ecg_voltage_uv'
                ])
                self.recording_file.flush()
                
                self.participant_id = participant_id
                self.is_recording = True
                
                # Add visual marker to plots
                self.add_recording_marker("Recording started", color='#4CAF50')
                
                # Update UI
                self.record_btn.setText("Stop Recording")
                self.record_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
                self.participant_id_input.setEnabled(False)
                self.select_location_btn.setEnabled(False)
                self.statusBar().showMessage(f"🔴 Recording to: {self.recording_save_path}")
                
            except Exception as e:
                self.statusBar().showMessage(f"Error starting recording: {e}")
                
        else:
            # Stop recording
            if self.recording_file:
                self.recording_file.close()
                self.recording_file = None
                self.recording_csv_writer = None
            
            self.is_recording = False
            
            # Add visual marker to plots
            self.add_recording_marker("Recording stopped", color='#f44336')
            
            # Update UI
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
            self.record_btn.setEnabled(False)
            self.participant_id_input.setEnabled(True)
            self.select_location_btn.setEnabled(True)
            self.select_location_btn.setText("Select Save Location")
            self.select_location_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
            self.recording_save_path = None
            self.statusBar().showMessage("Recording stopped")
    
    def add_recording_marker(self, label_text, color='#4CAF50'):
        """Add a vertical marker line to plots at current time"""
        if not self.is_streaming_active:
            return
        
        # Get current time relative to streaming start
        current_time = time.time() - self.streaming_start_time
        
        # Create marker line for ECG plot
        ecg_marker = pg.InfiniteLine(
            pos=current_time,
            angle=90,
            pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
            label=label_text,
            labelOpts={'position': 0.95, 'color': color, 'fill': (255, 255, 255, 150)}
        )
        self.ecg_plot.addItem(ecg_marker)
        
        # Create marker line for BPM plot
        bpm_marker = pg.InfiniteLine(
            pos=current_time,
            angle=90,
            pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
            label=label_text,
            labelOpts={'position': 0.95, 'color': color, 'fill': (255, 255, 255, 150)}
        )
        self.bpm_plot.addItem(bpm_marker)
        
        # Store markers
        self.recording_markers.append((ecg_marker, bpm_marker))
    
    def toggle_streaming(self):
        """Start or stop ECG streaming"""
        if not self.polar_handler:
            return
            
        if not self.polar_handler.is_streaming:
            # Start streaming
            self.stream_btn.setEnabled(False)
            self.stream_btn.setText("Starting...")
            
            asyncio.run_coroutine_threadsafe(
                self._start_streaming_async(),
                self.event_loop
            )
        else:
            # Stop streaming
            asyncio.run_coroutine_threadsafe(
                self.polar_handler.stop_streaming(),
                self.event_loop
            )
            self.stream_btn.setText("Start Streaming")
            self.statusBar().showMessage("Streaming stopped")
            
    async def _start_streaming_async(self):
        """Async streaming start"""
        success = await self.polar_handler.start_streaming()
        
        if success:
            self.signals.connection_status.emit("Streaming ECG data...")
            # Update button text on main thread
            self.stream_btn.setText("Stop Streaming")
            self.stream_btn.setEnabled(True)
            self.select_location_btn.setEnabled(True)  # Enable recording setup
        else:
            self.signals.connection_status.emit("Failed to start streaming")
            self.stream_btn.setEnabled(True)
            
    def _ecg_callback(self, samples: List[ECGSample]):
        """Callback for ECG data from backend"""
        self.signals.ecg_data.emit(samples)
        
    def _rpeak_callback(self, rpeak: RPeak):
        """Callback for R-peak detection from backend"""
        self.signals.rpeak_detected.emit(rpeak)
        
    def update_ecg_plot(self, samples: List[ECGSample]):
        """Receive new ECG samples and add to pending buffer"""
        
        # Write to CSV if recording is active
        if self.is_recording and self.recording_csv_writer and samples:
            # Get packet receive time (same for all samples in this packet)
            packet_receive_time = samples[-1].system_receive_time if samples else None
            
            for sample in samples:
                # Write raw data with proper timestamps
                self.recording_csv_writer.writerow([
                    self.participant_id,
                    sample.system_receive_time.isoformat() if sample.system_receive_time else '',
                    packet_receive_time.isoformat() if packet_receive_time else '',
                    f"{sample.timestamp:.6f}",  # Polar internal timestamp
                    f"{sample.voltage:.2f}"     # ECG voltage in µV
                ])
            # Flush periodically to ensure data is written
            self.recording_file.flush()
        
        current_real_time = time.time()
        
        if not self.is_streaming_active:
            self.is_streaming_active = True
            self.streaming_start_time = current_real_time
            # Start our timestamp counter at current time (relative to start)
            self.last_sample_time = 0.0
        
        # Current time relative to start
        relative_time = current_real_time - self.streaming_start_time
        
        # Calculate time per sample (1/130 Hz = ~0.00769 seconds)
        time_per_sample = 1.0 / 130.0
        
        # If we're behind real time, jump forward to catch up
        # This keeps ECG timestamps synchronized with the scrolling X-axis
        if self.last_sample_time < relative_time - 2.0:  # More than 2 seconds behind
            print(f"⚠ Catching up: jumping from {self.last_sample_time:.1f}s to {relative_time - 1.0:.1f}s")
            self.last_sample_time = relative_time - 1.0  # Jump to 1s ago
            # Clear pending buffer to start fresh
            cleared = len(self.pending_samples)
            self.pending_samples.clear()
            if cleared > 0:
                print(f"  Cleared {cleared} pending samples")
        
        adjusted_samples = []
        for i, sample in enumerate(samples):
            # Assign timestamp based on last_sample_time counter
            new_timestamp = self.last_sample_time + (i * time_per_sample)
            
            # Store mapping from Polar timestamp to our timestamp
            self.timestamp_mapping[sample.timestamp] = new_timestamp
            
            adjusted_sample = ECGSample(
                timestamp=new_timestamp,
                voltage=sample.voltage,
                system_receive_time=sample.system_receive_time
            )
            adjusted_samples.append(adjusted_sample)
        
        # Update last_sample_time for next batch
        if samples:
            self.last_sample_time = adjusted_samples[-1].timestamp + time_per_sample
        
        # Clean up old mappings (keep only last 30 seconds worth)
        if len(self.timestamp_mapping) > 5000:  # ~38 seconds at 130 Hz
            # Remove oldest entries
            sorted_keys = sorted(self.timestamp_mapping.keys())
            for key in sorted_keys[:1000]:
                del self.timestamp_mapping[key]
        
        if self.smooth_sample_insertion:
            # Add to pending buffer for gradual insertion
            self.pending_samples.extend(adjusted_samples)
        else:
            # Direct insertion
            for sample in adjusted_samples:
                self.ecg_times.append(sample.timestamp)
                self.ecg_voltages.append(sample.voltage)
            self.ecg_curve.setData(list(self.ecg_times), list(self.ecg_voltages))
    
    def update_time_based_scroll(self):
        """
        Update plot with time-based scrolling (like a paper strip chart).
        X-axis moves continuously with real time, samples are added as they arrive.
        """
        if not self.is_streaming_active:
            return
        
        # Calculate current time (relative to streaming start)
        current_time = time.time() - self.streaming_start_time if self.streaming_start_time else 0
        
        # Debug: Print buffer status occasionally
        if hasattr(self, '_last_debug_time'):
            if current_time - self._last_debug_time > 5.0:
                print(f"Debug: pending={len(self.pending_samples)}, ecg_times={len(self.ecg_times)}, " +
                      f"current_time={current_time:.1f}s")
                if len(self.ecg_times) > 0:
                    print(f"  ECG time range: [{min(self.ecg_times):.1f}, {max(self.ecg_times):.1f}]")
                self._last_debug_time = current_time
        else:
            self._last_debug_time = current_time
        
        # Insert pending samples gradually for smoothness
        # Adaptive drain rate: drain faster if buffer is growing
        if self.smooth_sample_insertion and self.pending_samples:
            pending_count = len(self.pending_samples)
            
            # Adaptive samples per frame based on buffer size
            if pending_count > 500:
                # Way too many - drain aggressively
                samples_to_add = min(50, pending_count)
            elif pending_count > 200:
                # Too many - drain fast
                samples_to_add = min(20, pending_count)
            elif pending_count > 100:
                # A bit too many - drain moderately
                samples_to_add = min(10, pending_count)
            else:
                # Normal - use configured rate
                samples_to_add = min(SAMPLES_PER_FRAME, pending_count)
            
            for _ in range(samples_to_add):
                sample = self.pending_samples.popleft()
                self.ecg_times.append(sample.timestamp)
                self.ecg_voltages.append(sample.voltage)
        
        # Update ECG plot data
        if len(self.ecg_times) > 0:
            self.ecg_curve.setData(list(self.ecg_times), list(self.ecg_voltages))
        
        # Update R-peak voltages by finding matching ECG samples
        if len(self.rpeak_times) > 0 and len(self.ecg_times) > 0:
            updated_voltages = []
            updated_times = []
            
            for rpeak_time, rpeak_voltage in zip(self.rpeak_times, self.rpeak_voltages):
                # Find closest ECG sample
                best_voltage = rpeak_voltage  # Keep old if not found
                min_diff = float('inf')
                
                for ecg_time, ecg_voltage in zip(self.ecg_times, self.ecg_voltages):
                    time_diff = abs(ecg_time - rpeak_time)
                    if time_diff < min_diff and time_diff < 0.02:  # Within 20ms
                        min_diff = time_diff
                        best_voltage = ecg_voltage
                
                updated_times.append(rpeak_time)
                updated_voltages.append(best_voltage)
            
            # Update rpeak_voltages in place
            self.rpeak_voltages.clear()
            self.rpeak_voltages.extend(updated_voltages)
            
            # Update scatter plot
            if self.show_rpeaks:
                self.rpeak_scatter.setData(updated_times, updated_voltages)
        
        # Update BPM Y-axis with padding (only consider visible values)
        if len(self.bpm_values) > 5 and len(self.bpm_times) > 5:
            # Only use BPM values within visible time window
            x_min = current_time - DISPLAY_WINDOW_SECONDS
            x_max = current_time
            
            visible_bpms = [
                bpm for bpm_time, bpm in zip(self.bpm_times, self.bpm_values)
                if x_min <= bpm_time <= x_max
            ]
            
            if len(visible_bpms) > 2:
                min_bpm = min(visible_bpms)
                max_bpm = max(visible_bpms)
                self.bpm_plot.setYRange(
                    min_bpm - BPM_Y_AXIS_PADDING, 
                    max_bpm + BPM_Y_AXIS_PADDING, 
                    padding=0
                )
        
        # Scroll X-axis with real time (this is the key!)
        # Always show the window [current_time - DISPLAY_WINDOW_SECONDS, current_time]
        if current_time > 1.0:  # Start scrolling after 1 second
            x_min = current_time - DISPLAY_WINDOW_SECONDS
            x_max = current_time
            self.ecg_plot.setXRange(x_min, x_max, padding=0)
            # BPM plot follows automatically due to linked X-axis
                
    def update_rpeak(self, rpeak: RPeak):
        """Update display with new R-peak detection"""
        # Convert Polar timestamp to our timestamp using mapping
        polar_timestamp = rpeak.timestamp
        
        # Find closest match in our timestamp mapping
        our_timestamp = None
        if polar_timestamp in self.timestamp_mapping:
            our_timestamp = self.timestamp_mapping[polar_timestamp]
        else:
            # Find closest timestamp in mapping
            min_diff = float('inf')
            for polar_ts, our_ts in self.timestamp_mapping.items():
                diff = abs(polar_ts - polar_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    our_timestamp = our_ts
                    if diff < 0.001:  # Within 1ms is good enough
                        break
        
        if our_timestamp is None:
            # No mapping available yet, skip this R-peak
            return
        
        # Store R-peak info - we'll find the voltage later when ECG data catches up
        self.rpeak_times.append(our_timestamp)
        # Initially use 0, will be updated in scroll timer
        self.rpeak_voltages.append(0)
            
        # Update BPM plot
        if rpeak.bpm is not None:
            self.bpm_times.append(our_timestamp)
            self.bpm_values.append(rpeak.bpm)
            
            # Update BPM curve
            self.bpm_curve.setData(list(self.bpm_times), list(self.bpm_values))
            
            # Update current BPM label
            self.bpm_label.setText(f"Current BPM: {rpeak.bpm:.1f}")
                
            # Enable average line button once we have enough data
            if len(self.bpm_times) >= 10:
                self.add_avg_btn.setEnabled(True)
                
    def add_average_line(self):
        """Add a static average line at current 30s average BPM"""
        if len(self.bpm_times) < 2:
            self.statusBar().showMessage("Not enough BPM data yet")
            return
            
        # Calculate average of last 30 seconds
        current_time = self.bpm_times[-1]
        cutoff_time = current_time - self.bpm_window_seconds
        
        recent_bpms = [
            bpm for time, bpm in zip(self.bpm_times, self.bpm_values)
            if time >= cutoff_time
        ]
        
        if len(recent_bpms) < 2:
            self.statusBar().showMessage("Not enough recent data (need 30s)")
            return
            
        avg_bpm = np.mean(recent_bpms)
        
        # Create new static line
        avg_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(color=COLOR_AVERAGE_LINE, width=2, style=Qt.PenStyle.DashLine),
            label=f'{avg_bpm:.1f} BPM',
            labelOpts={'position': 0.95, 'color': COLOR_AVERAGE_LINE}
        )
        avg_line.setValue(avg_bpm)
        self.bpm_plot.addItem(avg_line)
        self.average_lines.append(avg_line)
        
        self.statusBar().showMessage(f"Added average line: {avg_bpm:.1f} BPM (last {self.bpm_window_seconds}s)")
        self.clear_avg_btn.setEnabled(True)
        
    def clear_average_lines(self):
        """Remove all average lines from the BPM plot"""
        for line in self.average_lines:
            self.bpm_plot.removeItem(line)
        self.average_lines.clear()
        self.clear_avg_btn.setEnabled(False)
        self.statusBar().showMessage("Cleared all average lines")
        
    def toggle_rpeak_visibility(self, state):
        """Toggle R-peak markers visibility"""
        self.show_rpeaks = bool(state)
        self.rpeak_scatter.setVisible(self.show_rpeaks)
            
    def start_asyncio_thread(self):
        """Start background thread for asyncio event loop"""
        def run_loop():
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_forever()
            
        self.asyncio_thread = Thread(target=run_loop, daemon=True)
        self.asyncio_thread.start()
        
        # Wait for event loop to be ready (max 1 second)
        timeout = 1.0
        start = time.time()
        while self.event_loop is None and (time.time() - start) < timeout:
            time.sleep(0.01)
        
        if self.event_loop is None:
            raise RuntimeError("Failed to start asyncio event loop")
        
    def closeEvent(self, event):
        """Clean up when closing the application"""
        # Stop scrolling timer
        if hasattr(self, 'scroll_timer'):
            self.scroll_timer.stop()
        
        # Close recording file if open
        if self.recording_file:
            self.recording_file.close()
            print("Recording file closed")
        
        if self.polar_handler:
            if self.event_loop and self.event_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.polar_handler.disconnect(),
                    self.event_loop
                )
                
        if self.event_loop:
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = PolarECGViewer()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()