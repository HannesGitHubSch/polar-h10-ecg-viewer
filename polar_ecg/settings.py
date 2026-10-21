"""
Application settings definitions and loader.
Single source of defaults with configuration validation.
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    # Device
    SCAN_TIMEOUT_S: float = 5.0
    CONNECT_TIMEOUT_S: float = 15.0
    DATA_WATCHDOG_S: float = 3.0
    AUTO_RECONNECT: bool = True
    RECONNECT_BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 15.0)
    RECORD_DEVICE_HR: bool = True

    # Signal
    ECG_FS_NOMINAL: float = 130.0
    ECG_RESOLUTION: int = 14

    # Detection
    QRS_BANDPASS_HZ: tuple[float, float] = (5.0, 15.0)
    RPEAK_REFRACTORY_MS: float = 250.0
    RR_MIN_MS: float = 273.0
    RR_MAX_MS: float = 2000.0
    BPM_SMOOTHING_BEATS: int = 3

    # Display
    DISPLAY_WINDOW_S: float = 10.0
    DISPLAY_DELAY_S: float = 0.6
    DISPLAY_FPS: int = 30
    ECG_BUFFER_S: float = 30.0
    BPM_HISTORY_POINTS: int = 500
    SHOW_RPEAKS_DEFAULT: bool = False
    BPM_Y_AXIS_PADDING: float = 10.0
    BPM_AVERAGE_WINDOW_S: float = 10.0

    # Colors
    COLOR_ECG_LINE: str = "#2E86AB"
    COLOR_RPEAK_MARKER: str = "#FF0000"
    COLOR_BPM_LINE: str = "#A23B72"
    COLOR_AVERAGE_LINE: str = "#F18F01"

    # Recording
    RECORDING_FLUSH_INTERVAL_S: float = 1.0

    # LSL
    LSL_ENABLED: bool = False

    # Debug
    LOG_LEVEL: str = "INFO"


def validate_settings(s: Settings) -> None:
    """Validate settings constraints."""
    if s.ECG_FS_NOMINAL <= 0:
        raise ValueError(f"ECG_FS_NOMINAL must be positive, got {s.ECG_FS_NOMINAL}")
    if s.RR_MIN_MS >= s.RR_MAX_MS:
        raise ValueError(f"RR_MIN_MS ({s.RR_MIN_MS}) must be < RR_MAX_MS ({s.RR_MAX_MS})")
    if s.RPEAK_REFRACTORY_MS <= 0:
        raise ValueError("RPEAK_REFRACTORY_MS must be > 0")
    if s.DISPLAY_WINDOW_S <= 0:
        raise ValueError("DISPLAY_WINDOW_S must be > 0")
    if s.DISPLAY_FPS <= 0:
        raise ValueError("DISPLAY_FPS must be > 0")
    if s.QRS_BANDPASS_HZ[0] <= 0 or s.QRS_BANDPASS_HZ[1] <= s.QRS_BANDPASS_HZ[0]:
        raise ValueError(f"Invalid QRS_BANDPASS_HZ band: {s.QRS_BANDPASS_HZ}")


def load_settings(custom_config: Any = None) -> Settings:
    """
    Load settings from root config.py or provided object/module,
    validate, and return a frozen Settings instance.
    """
    params = {}
    default_fields = {f.name: f.type for f in Settings.__dataclass_fields__.values()}

    cfg = custom_config
    if cfg is None:
        try:
            cfg = importlib.import_module("config")
        except ImportError:
            logger.debug("No local config.py found; using default settings")
            cfg = None

    if cfg is not None:
        attrs = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
        for key, val in attrs.items():
            if key in default_fields:
                params[key] = val
            else:
                logger.warning("Unrecognized configuration key in config: %s", key)

    settings = Settings(**params)
    validate_settings(settings)
    return settings
