"""
Configuration file for Polar ECG Viewer
All parameters here can be modified to customize runtime behavior.
"""

# Device
SCAN_TIMEOUT_S = 5.0
CONNECT_TIMEOUT_S = 15.0
DATA_WATCHDOG_S = 3.0
AUTO_RECONNECT = True
RECONNECT_BACKOFF_S = (1, 2, 4, 8, 15)
RECORD_DEVICE_HR = True  # Also record H10 native HR / RR stream

# Signal
ECG_FS_NOMINAL = 130.0  # Polar H10 ECG nominal sampling rate (130 Hz only)
ECG_RESOLUTION = 14     # bits

# Detection
QRS_BANDPASS_HZ = (5.0, 15.0)  # Butterworth bandpass cutoff frequencies in Hz
RPEAK_REFRACTORY_MS = 250      # Minimum refractory period in ms
RR_MIN_MS = 273                # Min physiologically plausible RR in ms (~220 BPM)
RR_MAX_MS = 2000               # Max physiologically plausible RR in ms (~30 BPM)
BPM_SMOOTHING_BEATS = 3        # Number of recent accepted RR intervals to average for BPM

# Display
DISPLAY_WINDOW_S = 10          # Visible time window in plots (seconds)
DISPLAY_DELAY_S = 0.6          # Fixed display delay behind newest sample for smooth scrolling
DISPLAY_FPS = 30               # GUI plot refresh rate (frames per second)
ECG_BUFFER_S = 30              # Total buffer size in memory (seconds)
BPM_HISTORY_POINTS = 500       # Maximum number of BPM points kept in plot history
SHOW_RPEAKS_DEFAULT = False    # Whether R-peak markers are shown by default
BPM_Y_AXIS_PADDING = 10        # Vertical margin around visible BPM min/max (±10 BPM)
BPM_AVERAGE_WINDOW_S = 10      # Window in seconds for "Add Average Line" calculation

# UI Colors (hex)
COLOR_ECG_LINE = "#2E86AB"
COLOR_RPEAK_MARKER = "#FF0000"
COLOR_BPM_LINE = "#A23B72"
COLOR_AVERAGE_LINE = "#F18F01"

# Recording
RECORDING_FLUSH_INTERVAL_S = 1.0  # Periodic disk flush interval in seconds

# Lab Streaming Layer (LSL)
LSL_ENABLED = False

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
