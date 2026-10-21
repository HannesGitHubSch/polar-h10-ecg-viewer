"""
Application states and control enabling rules for UI consistency.
"""

from enum import Enum, auto


class AppState(Enum):
    """Finite states of the ECG Viewer application."""

    IDLE = auto()
    SCANNING = auto()
    READY = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    STREAMING = auto()
    RECONNECTING = auto()
    ERROR = auto()


# Map each state to the set of controls that may be enabled
ENABLED_CONTROLS: dict[AppState, set[str]] = {
    AppState.IDLE: {"scan", "participant_id", "select_location"},
    AppState.SCANNING: set(),
    AppState.READY: {"scan", "device_combo", "connect", "participant_id", "select_location"},
    AppState.CONNECTING: set(),
    AppState.CONNECTED: {"scan", "device_combo", "stream", "disconnect", "participant_id", "select_location"},
    AppState.STREAMING: {
        "stream",
        "disconnect",
        "participant_id",
        "select_location",
        "record",
        "mark",
        "rpeak_toggle",
        "add_avg",
        "clear_avg",
    },
    AppState.RECONNECTING: {"disconnect", "record", "mark"},
    AppState.ERROR: {"scan", "device_combo", "connect", "disconnect"},
}

