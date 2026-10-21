# Real-time Polar H10 ECG Viewer & Recorder

A research-grade real-time ECG visualization, recording, and analysis suite for the **Polar H10** heart rate monitor. Designed for psychophysiology research, affective computing, clinical demonstrations, and real-time biofeedback protocols.

![Polar ECG Viewer Interface](assets/screenshot-interface.png)

---

## Key Highlights

- **Jitter-Free Dual-Clock Timing:** Decouples sensor crystal timing from Bluetooth transfer latency. Delivers mathematically uniform 130 Hz sampling and microsecond-level time synchronization.
- **Continuous Gap-Aware Sample Clock:** Detects dropped BLE packets and sensor resets, tracking exact missing sample counts without distorting physiological timebases.
- **Research-Grade QRS Detection:** Real-time Pan-Tompkins detector with dynamic refractory blanking, automatic polarity detection, and lookahead fiducial alignment.
- **Thread-Safe Reactive Architecture:** PyQt6 GUI decoupled from an asyncio BLE core via an event runner. Zero GUI freezes or cross-thread race conditions.
- **Fail-Safe Recording (CSV v2):** Atomic `.csv.partial` writing with JSON metadata sidecars, auxiliary beat/HR files, and overwrite protection.
- **Lab Streaming Layer (LSL):** Native synchronization outlets for ECG waveforms (130 Hz, microvolts), experimental markers, and RR intervals.
- **Hardware-Free Simulation & Replay:** Built-in synthetic ECG engine and CSV replay modes for development, testing, and continuous integration.

---

## Requirements & Installation

- **Python:** 3.10, 3.11, 3.12, or 3.13
- **Operating System:** Windows 10/11, macOS (12+), or Linux (with BlueZ 5.50+)
- **Hardware:** Polar H10 chest strap with Bluetooth Low Energy (BLE 4.0+)

### Setup

```bash
# Clone repository
git clone https://github.com/<owner>/polar-h10-ecg-viewer.git
cd polar-h10-ecg-viewer

# Recommended: create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# or: .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install package and dependencies
pip install -e .

# Optional: install development and research dependencies (neurokit2, pytest, ruff)
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Launch the Application

```bash
python main.py
```

### 2. Live Workflow
1. Click **Scan Devices** to discover nearby Polar straps.
2. Select your device from the drop-down and click **Connect**.
3. Click **Start Streaming** to view the live ECG and BPM plots.
4. **Disconnect:** You can safely disconnect and re-scan at any time without restarting the application.
5. **Auto-Reconnect:** If the Bluetooth signal drops, a red alert banner appears and the application automatically attempts reconnection using an exponential backoff schedule while preserving the ongoing recording.

### 3. Recording & Experimental Markers
1. Enter a **Participant ID** (e.g. `P001`).
2. (Optional) Click **Select Save Location** to specify a custom destination.
3. Click **Start Recording** (or press Space). Visual green markers indicate recording onset.
4. **Mark Events:** Type a label in the Event box and click **Mark** (or press keyboard hotkeys **`1` through `9`**) to insert timestamped experimental markers into the recording and LSL stream.
5. Click **Stop Recording**. The files are atomically finalized.

---

## Command-Line Interface (CLI)

The application supports multiple CLI flags for headless testing, research automation, and simulation:

```bash
# Run with synthetic ECG data (no Polar hardware required)
python main.py --simulate

# Stream a pre-recorded session through the real-time interface
python main.py --replay recordings/ECG_P001_20261021.csv --speed 1.0

# Broadcast real-time streams to Lab Streaming Layer (LSL)
python main.py --lsl

# Automated headless recording for testing or experimental scripting
python main.py --simulate --autostart --record scratch/session.csv --participant TEST --duration 30
```

| Flag | Argument | Description |
|---|---|---|
| `--simulate` | *None* | Runs the synthetic ECG generator with realistic P-Q-R-S-T waves, HRV, and noise. |
| `--replay` | `FILE` | Replays a CSV v1 or v2 recording file in real-time. |
| `--speed` | `FLOAT` | Replay or simulation speed multiplier (e.g., `2.0` for 2x speed). |
| `--lsl` | *None* | Enables Lab Streaming Layer outlets (`PolarH10_ECG`, `PolarH10_Markers`, `PolarH10_RR`). |
| `--autostart`| *None* | Automatically begins streaming upon startup. |
| `--record` | `PATH` | Pre-sets the base recording path. |
| `--participant` | `ID` | Pre-sets the participant identifier (default: `P001`). |
| `--duration` | `SECONDS` | Automatically stops recording and quits after `N` seconds. |
| `--log-level` | `LEVEL` | Sets logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Data Specifications (CSV v2)

When recording, the application generates a synchronized quartet of files:

```
recordings/
├── ECG_P001_20261021_120000.csv         # Primary continuous ECG sample data
├── ECG_P001_20261021_120000_meta.json   # Session metadata sidecar
├── ECG_P001_20261021_120000_beats.csv  # Detected R-peaks and RR intervals
└── ECG_P001_20261021_120000_hr.csv     # Polar H10 native heart rate & hardware RR
```

### 1. Primary ECG File (`.csv`)
Atomic file write with crash protection (`.csv.partial` renamed upon finalization).

```csv
sample_index,t_session_s,device_time_ns,host_time_utc,packet_rx_utc,ecg_uv,gap_before,event
0,0.000000,1000000000,2026-10-21T12:00:00.000000+00:00,2026-10-21T12:00:00.561234+00:00,12.50,0,recording_start
1,0.007692,1007692308,2026-10-21T12:00:00.007692+00:00,2026-10-21T12:00:00.561234+00:00,15.20,0,
```

- `sample_index`: Monotonically increasing continuous sample integer.
- `t_session_s`: Elapsed time in seconds since the start of streaming (jitter-free).
- `device_time_ns`: Sensor internal timestamp in nanoseconds.
- `host_time_utc`: Wall-clock UTC time derived via minimum-latency envelope mapping.
- `packet_rx_utc`: Host UTC arrival timestamp of the containing BLE packet.
- `ecg_uv`: Calibrated ECG potential in microvolts ($\mu\text{V}$).
- `gap_before`: Number of missing samples immediately preceding this sample (`0` if contiguous, `> 0` if packet dropped, `-1` on clock reset).
- `event`: Event marker label if an event coincided with this sample (e.g., `cue`, `recording_start`).

### 2. Metadata Sidecar (`_meta.json`)
Self-documenting session configuration and data hygiene audit:
- Participant ID, hardware device name and MAC address.
- Nominal ($130.0\text{ Hz}$) and estimated effective sampling rate.
- Total samples written, total missing samples, and clock discontinuities.
- Start and stop UTC timestamps and full event marker log.

### 3. Detected Beats (`_beats.csv`)
- `sample_index`, `t_session_s`, `host_time_utc`, `voltage_uv`, `rri_ms`, `bpm`, `flags`.

### 4. Native Hardware HR (`_hr.csv`)
- `host_time_utc`, `hr_bpm`, `rr_ms` (raw hardware intervals reported directly by Polar firmware).

---

## Analysis & Diagnostics Tool

The included `plot_recording.py` utility visualizes and validates recorded files:

```bash
# Visualize a recording
python plot_recording.py recordings/ECG_P001_20261021_120000.csv

# Non-interactive generation (generates PNG without opening a window)
python plot_recording.py recordings/ECG_P001_20261021_120000.csv --no-show --out plot.png

# Compare against Polar native hardware RR intervals
python plot_recording.py recordings/ECG_P001_20261021_120000.csv --compare-device-rr

# Use NeuroKit2 for offline QRS validation
python plot_recording.py recordings/ECG_P001_20261021_120000.csv --detector neurokit
```

![Example Recording Plot](assets/example-recording-plot.png)

### Safe File Handling
The tool features strict **overwrite protection**: it refuses to write output plots to paths that collide with raw data, preventing data corruption.

### Backward Compatibility
`plot_recording.py` and `polar_ecg.analysis.io.load_recording()` automatically detect whether a file is in legacy **CSV v1** or modern **CSV v2** format and parse it seamlessly.

---

## Lab Streaming Layer (LSL)

When `--lsl` is enabled (or `LSL_ENABLED = True` in settings), the application publishes three standardized synchronized streams:

1. **`PolarH10_ECG`** (`type=ECG`):
   - 1 channel, 130 Hz nominal, `float32`.
   - Units: microvolts ($\mu\text{V}$).
   - Sample timestamps aligned with `pylsl.local_clock()`.
2. **`PolarH10_Markers`** (`type=Markers`):
   - Irregular string markers for experimental triggers, recording milestones, and session state changes.
3. **`PolarH10_RR`** (`type=RR`):
   - 2 channels (`float32`): `[RR_ms, BPM]`.

Compatible with OpenViBE, LabRecorder, PsychoPy, Neuropype, and standard EEG/multimodal acquisition pipelines.

---

## Architecture & Timing Model

### The Polar H10 BLE Latency Problem
The Polar H10 samples ECG at 130 Hz internally and buffers samples into chunks of 73 samples before transmitting them via BLE notification approximately every $561.5\text{ ms}$. 

Due to Bluetooth radio scheduling, operating system kernel interrupts, and USB/PCIe polling:
- Arrival times of packets on the host jitter by $\pm 15$ to $50\text{ ms}$.
- Back-calculating sample timestamps directly from the host arrival time (as done in legacy software) induces artificial $\pm 50\text{ ms}$ jitter into every packet boundary, corrupting frequency-domain HRV ($LF/HF$ ratios).

### The Dual-Clock Solution
1. **Device Internal Timebase (`SampleClock`):**
   The relative spacing of samples within and across packets is dictated by the high-stability quartz oscillator on the Polar H10. `SampleClock` reconstructs a continuous sample index and continuous session seconds $t_{\text{session}}$:
   $$\Delta t = \frac{1}{f_{s,\text{estimated}}}$$
2. **Convex Latency Envelope (`ClockSync`):**
   To align with host wall-clock time without introducing jitter, a rolling linear regression estimates sensor clock drift while maintaining a minimum-latency lower bound:
   $$t_{\text{host}} = \alpha \cdot t_{\text{device}} + \beta$$
   This guarantees monotonic, smooth absolute timestamps while preserving the millisecond-precise shape of the cardiac signal.

---

## Configuration

Settings are centralized in `polar_ecg/settings.py` (with override support via environment variables or a local `config.py`):

```python
# Signal & Timing
ECG_FS_NOMINAL = 130.0              # Nominal sensor rate (Hz)
ECG_RESOLUTION = 14                 # Sensor ADC resolution (bits)

# QRS Detection
QRS_BANDPASS_HZ = (5.0, 15.0)       # Bandpass filter cutoff (Hz)
RPEAK_REFRACTORY_MS = 250.0         # Physiological refractory blanking (ms)
BPM_SMOOTHING_BEATS = 3             # Smoothing window for instantaneous BPM

# Real-time Display
DISPLAY_WINDOW_S = 10.0             # Horizontal time window (seconds)
DISPLAY_DELAY_S = 0.6               # Display lag buffer for jitter-free scrolling
DISPLAY_FPS = 30                    # GUI refresh rate

# Watchdog & Reconnection
DATA_WATCHDOG_S = 3.0               # Signal stall timeout (seconds)
AUTO_RECONNECT = True               # Automatic retry on disconnect
```

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **No devices discovered during Scan** | Bluetooth disabled or strap asleep | Ensure Bluetooth is ON on host. Moisten the strap electrodes with water or saline. Snap sensor securely into strap. |
| **Permission denied on Linux** | BlueZ / D-Bus permissions | Ensure user is in `bluetooth` or `netdev` group, or configure `/etc/bluetooth/` permissions. |
| **No ECG data after Connect** | Poor skin electrode impedance | Thoroughly wet the rubber pads; tighten chest strap. Wait 3–5 seconds for sensor to calibrate. |
| **BPM reads `--`** | Low signal amplitude or noisy baseline | Ensure sensor is centered over sternum/V1–V2. Toggle "Show R-peaks" to inspect detector alignment. |
| **Plot looks jerky on slow machine** | Display refresh workload | In `settings.py`, lower `DISPLAY_FPS` from `30` to `20` or reduce `ECG_BUFFER_S`. |

---

## Project Structure

```
polar-h10-ecg-viewer/
├── main.py                          # Primary application entry point (shim)
├── plot_recording.py                # Analysis & plotting entry point (shim)
├── pyproject.toml                   # Build system and package configuration
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt             # Development and test dependencies
│
├── polar_ecg/                       # Core modular package
│   ├── app.py                       # CLI parser and application bootstrapping
│   ├── config.py                    # Backwards-compatible configuration bridge
│   ├── settings.py                  # Settings definitions and loader
│   ├── models.py                    # Immutable dataclasses (RawFrame, TimedFrame, RPeak)
│   ├── ringbuffer.py                # Lockless circular NumPy buffer
│   ├── synth.py                     # High-performance synthetic ECG generator
│   ├── timing.py                    # SampleClock and ClockSync dual-clock engine
│   ├── detection.py                 # Pan-Tompkins real-time & offline QRS detector
│   ├── device.py                    # Bleak Bluetooth client with disconnect handling
│   ├── sources.py                   # Data sources (DeviceSource, SimulatedSource, ReplaySource)
│   ├── session.py                   # Core orchestrator and reconnection watchdog
│   ├── recording.py                 # Threaded CSV v2 file writer and metadata manager
│   ├── lsl_out.py                   # Lab Streaming Layer (LSL) outlet integration
│   │
│   ├── ui/                          # PyQt6 user interface
│   │   ├── state.py                 # AppState machine and control rules
│   │   ├── async_runner.py          # Thread-safe asyncio / Qt bridge
│   │   ├── plots.py                 # Real-time pyqtgraph rolling waveform widgets
│   │   └── main_window.py           # Main window layout and event handlers
│   │
│   └── analysis/                    # Post-hoc analysis tools
│       ├── io.py                    # Universal loader for CSV v1 and v2 formats
│       └── plot_recording.py        # Diagnostic plotting and validation engine
│
└── tests/                           # Comprehensive test suite
    ├── test_timing.py               # Clock continuity, jitter, and drift tests
    ├── test_detection.py            # QRS detection, noise tolerance, and NeuroKit2 agreement
    ├── test_ringbuffer.py           # Circular buffer wrapping and slicing tests
    ├── test_recording.py            # File writer, metadata, and error handling tests
    ├── test_io.py                   # Format v1 and v2 reader compatibility tests
    ├── test_plot_recording.py       # Diagnostic plotting and overwrite protection tests
    ├── test_device.py               # Bleak wrapper and scan filter tests
    ├── test_session.py              # Session lifecycle, streaming, and watchdog tests
    ├── test_lsl.py                  # Lab Streaming Layer outlet and resolution tests
    └── test_ui_smoke.py             # Headless PyQt6 GUI smoke tests
```

---

## Acknowledgments & License

- Original concept and implementation by **[zHElEARN](https://github.com/zHElEARN)**.
- Overhauled and expanded into a modular research framework by the open-source community.
- Built on top of [polar-python](https://github.com/JuPfu/polar-python), [bleak](https://github.com/hbldh/bleak), [PyQt6](https://www.riverbankcomputing.com/software/pyqt/), [pyqtgraph](https://www.pyqtgraph.org/), and [pylsl](https://github.com/labstreaminglayer/liblsl-Python).

Distributed under the **MIT License**. See `LICENSE` for details.
