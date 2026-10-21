"""
High-performance real-time pyqtgraph plotting widgets powered by RingBuffer.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from polar_ecg.models import DeviceHR, RPeak, TimedFrame
from polar_ecg.ringbuffer import RingBuffer
from polar_ecg.settings import Settings


class EcgPlotWidget:
    """
    Real-time rolling ECG voltage plot with R-peak markers and events.
    """

    def __init__(self, plot_widget: pg.PlotWidget, settings: Settings):
        self.widget = plot_widget
        self.cfg = settings

        self.widget.setLabel("left", "Voltage", units="µV")
        self.widget.setLabel("bottom", "Time", units="s")
        self.widget.showGrid(x=True, y=True, alpha=0.3)
        self.widget.enableAutoRange(axis="y", enable=True)
        self.widget.setAutoVisible(y=True)

        # 30-second circular buffer for continuous ECG waveform
        buffer_capacity = round(self.cfg.ECG_BUFFER_S * self.cfg.ECG_FS_NOMINAL)
        self.ecg_buffer = RingBuffer(capacity=buffer_capacity)

        # Peak markers circular buffer
        self.peak_buffer = RingBuffer(capacity=400)

        # Plot items
        self.ecg_curve = self.widget.plot(
            pen=pg.mkPen(color=self.cfg.COLOR_ECG_LINE, width=1.8),
            name="ECG",
        )
        self.peak_scatter = pg.ScatterPlotItem(
            size=9,
            brush=pg.mkBrush(self.cfg.COLOR_RPEAK_MARKER + "DD"),
            symbol="o",
            pen=pg.mkPen(color="#FFFFFF", width=0.5),
        )
        self.peak_scatter.setVisible(self.cfg.SHOW_RPEAKS_DEFAULT)
        self.widget.addItem(self.peak_scatter)

        self.markers: list[pg.InfiniteLine] = []

    def clear(self) -> None:
        """Reset buffers and clear plot data."""
        self.ecg_buffer.clear()
        self.peak_buffer.clear()
        self.ecg_curve.clear()
        self.peak_scatter.clear()
        for m in self.markers:
            self.widget.removeItem(m)
        self.markers.clear()

    def set_rpeaks_visible(self, visible: bool) -> None:
        """Toggle R-peak scatter plot visibility."""
        self.peak_scatter.setVisible(visible)

    def add_timed_frame(self, frame: TimedFrame) -> None:
        """Store incoming samples in circular buffer."""
        self.ecg_buffer.append_many(frame.t_session_s, frame.voltages_uv)

        # Add visual indicator for data gaps
        if frame.gap_before > 0:
            self.add_marker(
                t_session=float(frame.t_session_s[0]),
                label=f"Gap ({frame.gap_before})",
                color="#FFA000",
            )
        elif frame.gap_before == -1:
            self.add_marker(
                t_session=float(frame.t_session_s[0]),
                label="Reset",
                color="#D32F2F",
            )

    def add_peak(self, peak: RPeak) -> None:
        """Store detected R-peak location and voltage directly without searching."""
        self.peak_buffer.append(peak.t_session_s, peak.voltage_uv)

    def add_marker(self, t_session: float, label: str, color: str = "#4CAF50") -> None:
        """Add vertical marker line at specific session timestamp."""
        marker = pg.InfiniteLine(
            pos=t_session,
            angle=90,
            pen=pg.mkPen(color=color, width=1.5, style=Qt.PenStyle.DashLine),
            label=label,
            labelOpts={"position": 0.95, "color": color, "fill": (255, 255, 255, 160)},
        )
        self.widget.addItem(marker)
        self.markers.append(marker)

    def redraw(self, current_session_t: float) -> None:
        """Update plot curves and scroll X-axis to match real time."""
        if len(self.ecg_buffer) == 0:
            return

        x_ecg, y_ecg = self.ecg_buffer.view()
        self.ecg_curve.setData(x_ecg, y_ecg)

        if self.peak_scatter.isVisible() and len(self.peak_buffer) > 0:
            x_peak, y_peak = self.peak_buffer.view()
            self.peak_scatter.setData(x_peak, y_peak)

        # Paper-strip chart scrolling with fixed delay
        t_delayed = current_session_t - self.cfg.DISPLAY_DELAY_S
        if t_delayed > 0.5:
            x_min = t_delayed - self.cfg.DISPLAY_WINDOW_S
            x_max = t_delayed
            self.widget.setXRange(x_min, x_max, padding=0.0)


class BpmPlotWidget:
    """
    Heart rate BPM plot with auto-scaling padding and average lines.
    """

    def __init__(self, plot_widget: pg.PlotWidget, settings: Settings):
        self.widget = plot_widget
        self.cfg = settings

        self.widget.setLabel("left", "Heart Rate", units="BPM")
        self.widget.setLabel("bottom", "Time", units="s")
        self.widget.showGrid(x=True, y=True, alpha=0.3)
        self.widget.setYRange(40, 180, padding=0.0)

        self.bpm_buffer = RingBuffer(capacity=self.cfg.BPM_HISTORY_POINTS)

        self.bpm_curve = self.widget.plot(
            pen=pg.mkPen(color=self.cfg.COLOR_BPM_LINE, width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(self.cfg.COLOR_BPM_LINE + "CC"),
            name="Detected BPM",
        )

        # Optional overlay of Polar device native HR / RR-derived BPM
        self.device_hr_buffer = RingBuffer(capacity=self.cfg.BPM_HISTORY_POINTS)
        self.device_hr_curve = self.widget.plot(
            pen=pg.mkPen(color="#0288D1", width=1.5, style=Qt.PenStyle.DashLine),
            symbol="t",
            symbolSize=6,
            symbolBrush=pg.mkBrush("#0288D1"),
            name="Polar Device HR",
        )
        self.device_hr_curve.setVisible(False)

        self.average_lines: list[pg.InfiniteLine] = []

    def clear(self) -> None:
        self.bpm_buffer.clear()
        self.bpm_curve.clear()
        self.device_hr_buffer.clear()
        self.device_hr_curve.clear()
        self.clear_average_lines()

    def set_device_hr_visible(self, visible: bool) -> None:
        self.device_hr_curve.setVisible(visible)

    def add_peak(self, peak: RPeak) -> None:
        """Record instantaneous smoothed BPM at peak time."""
        if peak.bpm is not None:
            self.bpm_buffer.append(peak.t_session_s, peak.bpm)

    def add_device_hr(self, t_session: float, hr: DeviceHR) -> None:
        """Record native Polar HR / RR in overlay buffer."""
        if hr.rr_ms:
            for rr in hr.rr_ms:
                if rr > 0:
                    bpm = 60000.0 / rr
                    self.device_hr_buffer.append(t_session, bpm)
        elif hr.hr_bpm > 0:
            self.device_hr_buffer.append(t_session, float(hr.hr_bpm))

    def add_average_line(self) -> float | None:
        """Calculate recent average BPM and plot a horizontal dashed line."""
        if len(self.bpm_buffer) < 2:
            return None

        t_arr, bpm_arr = self.bpm_buffer.view()
        latest_t = t_arr[-1]
        cutoff_t = latest_t - self.cfg.BPM_AVERAGE_WINDOW_S

        mask = t_arr >= cutoff_t
        if not np.any(mask):
            return None

        avg_bpm = float(np.mean(bpm_arr[mask]))

        line = pg.InfiniteLine(
            pos=avg_bpm,
            angle=0,
            movable=False,
            pen=pg.mkPen(color=self.cfg.COLOR_AVERAGE_LINE, width=1.8, style=Qt.PenStyle.DashLine),
            label=f"{avg_bpm:.1f} BPM",
            labelOpts={"position": 0.95, "color": self.cfg.COLOR_AVERAGE_LINE},
        )
        self.widget.addItem(line)
        self.average_lines.append(line)
        return avg_bpm

    def clear_average_lines(self) -> None:
        for line in self.average_lines:
            self.widget.removeItem(line)
        self.average_lines.clear()

    def redraw(self, current_session_t: float) -> None:
        """Update BPM curve and adjust Y-range to visible range with padding."""
        if len(self.bpm_buffer) > 0:
            t_arr, bpm_arr = self.bpm_buffer.view()
            self.bpm_curve.setData(t_arr, bpm_arr)
        else:
            t_arr, bpm_arr = np.array([]), np.array([])

        if self.device_hr_curve.isVisible() and len(self.device_hr_buffer) > 0:
            t_dev, bpm_dev = self.device_hr_buffer.view()
            self.device_hr_curve.setData(t_dev, bpm_dev)

        if len(self.bpm_buffer) == 0:
            return

        # Dynamic vertical range scaling bounded by visible window
        t_delayed = current_session_t - self.cfg.DISPLAY_DELAY_S
        x_min = t_delayed - self.cfg.DISPLAY_WINDOW_S
        x_max = t_delayed

        visible_mask = (t_arr >= x_min) & (t_arr <= x_max)
        if np.any(visible_mask):
            vis_bpms = bpm_arr[visible_mask]
            min_bpm = float(np.min(vis_bpms))
            max_bpm = float(np.max(vis_bpms))
            pad = self.cfg.BPM_Y_AXIS_PADDING
            self.widget.setYRange(max(20, min_bpm - pad), min(240, max_bpm + pad), padding=0.0)
