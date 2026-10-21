"""
UI smoke tests using pytest-qt and offscreen Qt platform.
"""

from pathlib import Path

from polar_ecg.models import DeviceInfo
from polar_ecg.settings import Settings
from polar_ecg.sources import SimulatedSource
from polar_ecg.ui.main_window import MainWindow
from polar_ecg.ui.state import AppState


def test_ui_window_creation(qtbot):
    settings = Settings()
    window = MainWindow(settings)
    qtbot.addWidget(window)
    window.show()

    assert window.isVisible()
    assert window.windowTitle() == "Polar H10 ECG Viewer & Recorder"
    assert window.current_state == AppState.IDLE
    window.close()


def test_ui_state_enable_rules(qtbot):
    settings = Settings()
    window = MainWindow(settings)
    qtbot.addWidget(window)

    window.apply_state(AppState.IDLE)
    assert window.scan_btn.isEnabled()
    assert not window.connect_btn.isEnabled()
    assert not window.stream_btn.isEnabled()

    window.apply_state(AppState.READY)
    assert window.scan_btn.isEnabled()
    assert window.connect_btn.isEnabled()

    window.apply_state(AppState.CONNECTED)
    assert window.stream_btn.isEnabled()
    assert window.disconnect_btn.isEnabled()

    window.apply_state(AppState.STREAMING)
    assert window.stream_btn.isEnabled()
    assert window.record_btn.isEnabled()
    assert window.mark_btn.isEnabled()
    window.close()


def test_ui_simulated_stream_smoke(qtbot, tmp_path: Path):
    settings = Settings()
    custom_info = DeviceInfo("Simulated Polar H10", "SIM:00:11")
    custom_source = SimulatedSource(
        on_frame=lambda f: None,
        on_hr=lambda hr: None,
        on_disconnect=lambda r: None,
        duration_s=10.0,
        speed=10.0,
    )

    window = MainWindow(
        settings=settings,
        custom_source=custom_source,
        custom_device_info=custom_info,
    )
    qtbot.addWidget(window)
    custom_source.on_frame = window.session.on_raw_frame
    custom_source.on_hr = window.session.on_device_hr
    custom_source.on_disconnect = window.session.on_source_disconnect

    window.show()

    # Toggle streaming
    window._on_toggle_streaming()
    qtbot.waitUntil(lambda: window.session.is_streaming, timeout=2000)
    assert window.session.is_streaming

    # Wait for ~0.5s real time (5s simulated data)
    qtbot.wait(600)

    # Check that points were added to buffers and curve
    assert len(window.ecg_plot.ecg_buffer) > 100
    assert len(window.bpm_plot.bpm_buffer) > 0

    # Stop streaming and close
    window._on_toggle_streaming()
    qtbot.waitUntil(lambda: not window.session.is_streaming, timeout=2000)
    assert not window.session.is_streaming

    window.close()
