"""
Visual sanity check and diagnostic plotting tool for recorded ECG sessions.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polar_ecg.analysis.io import Recording, load_recording
from polar_ecg.detection import detect_offline

logger = logging.getLogger(__name__)


def analyze_and_plot(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    detector: str = "builtin",
    compare_device_rr: bool = False,
    show: bool = True,
) -> plt.Figure:
    """
    Load an ECG recording, perform R-peak detection, compute metrics,
    and generate a 3-panel diagnostic visualization.
    """
    in_path = Path(csv_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Determine safe output path
    if out_path is not None:
        target_out = Path(out_path).resolve()
    else:
        target_out = in_path.with_name(f"{in_path.stem}_plot.png").resolve()

    if target_out == in_path:
        raise ValueError(
            f"Target output path '{target_out}' matches input CSV path. "
            "Refusing to overwrite input data file!"
        )

    # Load recording
    rec: Recording = load_recording(in_path)
    df = rec.df

    if len(df) < 2:
        raise ValueError(f"Recording in {in_path} is empty or has fewer than 2 samples.")

    t_s = df["t_session_s"].values
    voltages = df["ecg_uv"].values
    duration = float(t_s[-1] - t_s[0])
    effective_fs = len(df) / duration if duration > 0 else 130.0

    print(f"\n✓ Loaded {len(df)} samples from {in_path.name}")
    print(f"  Format version: v{rec.format_version}")
    print(f"  Participant ID: {rec.participant_id}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Effective sample rate: {effective_fs:.2f} Hz")

    # Gap and missing sample statistics
    missing_samples = int(df["gap_before"][df["gap_before"] > 0].sum()) if "gap_before" in df.columns else 0
    discontinuities = int((df["gap_before"] == -1).sum()) if "gap_before" in df.columns else 0
    print(f"  Missing samples: {missing_samples}")
    if discontinuities > 0:
        print(f"  Clock discontinuities: {discontinuities}")

    # Detect R-peaks
    if detector == "neurokit":
        try:
            import neurokit2 as nk

            _, rpeaks_nk = nk.ecg_peaks(voltages, sampling_rate=round(effective_fs))
            peak_indices = rpeaks_nk["ECG_R_Peaks"]
            peak_times = t_s[peak_indices]
            peak_voltages = voltages[peak_indices]
        except ImportError:
            print("⚠ neurokit2 not installed; falling back to builtin detector")
            detector = "builtin"

    if detector == "builtin":
        rpeaks = detect_offline(voltages, fs=effective_fs)
        peak_indices = [round(p.sample_index) for p in rpeaks if 0 <= round(p.sample_index) < len(t_s)]
        peak_times = np.array([p.t_session_s for p in rpeaks])
        peak_voltages = np.array([p.voltage_uv for p in rpeaks])

    print(f"\n💓 R-peak Detection ({detector}):")
    print(f"  Found {len(peak_times)} R-peaks")

    # Compute BPM from detected R-peaks
    bpm_times = []
    bpm_values = []
    if len(peak_times) >= 2:
        rr_intervals_s = np.diff(peak_times)
        instant_bpm = 60.0 / rr_intervals_s
        # Filter physiological range for summary
        valid = (instant_bpm >= 30) & (instant_bpm <= 220)
        valid_bpms = instant_bpm[valid]
        if len(valid_bpms) > 0:
            print(f"  Mean Heart Rate: {np.mean(valid_bpms):.1f} BPM")
            print(f"  BPM range: [{np.min(valid_bpms):.1f}, {np.max(valid_bpms):.1f}]")
        bpm_times = peak_times[1:]
        bpm_values = instant_bpm

    # Create figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Panel 1: Full ECG trace with R-peaks and gap / event markers
    ax1 = axes[0]
    ax1.plot(t_s, voltages, color="#2E86AB", linewidth=0.6, alpha=0.85, label="ECG")
    if len(peak_times) > 0:
        ax1.scatter(peak_times, peak_voltages, color="red", s=25, zorder=5, alpha=0.7, label="R-peaks")

    # Show gap markers if present
    if "gap_before" in df.columns:
        gap_rows = df[df["gap_before"] > 0]
        for _, row in gap_rows.iterrows():
            ax1.axvline(row["t_session_s"], color="orange", linestyle="--", alpha=0.7)

    # Show event markers if present
    if "event" in df.columns:
        ev_rows = df[df["event"].str.len() > 0]
        for _, row in ev_rows.iterrows():
            ax1.axvline(row["t_session_s"], color="#4CAF50", linestyle="-.", alpha=0.9)
            ax1.text(row["t_session_s"], np.max(voltages) * 0.9, f" {row['event']}", color="#2E7D32", fontsize=9)

    ax1.set_ylabel("ECG (µV)", fontsize=11)
    ax1.set_title(
        f"ECG Recording - Participant {rec.participant_id} ({in_path.name})",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Session Time (s)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Heart rate over time
    ax2 = axes[1]
    if len(bpm_values) > 0:
        ax2.plot(bpm_times, bpm_values, color="#A23B72", linewidth=1.5, marker="o", markersize=3, label="ECG BPM")
        valid_bpms = np.array(bpm_values)[(np.array(bpm_values) >= 30) & (np.array(bpm_values) <= 220)]
        if len(valid_bpms) > 0:
            mean_bpm = float(np.mean(valid_bpms))
            ax2.axhline(mean_bpm, color="#F18F01", linestyle="--", linewidth=1.5, label=f"Mean: {mean_bpm:.1f} BPM")
            ax2.set_ylim(max(30, np.min(valid_bpms) - 10), min(220, np.max(valid_bpms) + 10))

    # Compare device native HR if requested
    if compare_device_rr and rec.hr_df is not None and not rec.hr_df.empty:
        hr_df = rec.hr_df
        if "rr_ms" in hr_df.columns and not hr_df["rr_ms"].isna().all():
            device_rrs = hr_df["rr_ms"].dropna().values.astype(float)
            device_bpms = 60000.0 / device_rrs
            # Approximate time points
            dev_t = np.linspace(t_s[0], t_s[-1], len(device_bpms))
            ax2.plot(dev_t, device_bpms, color="#00B4D8", linestyle=":", linewidth=1.5, label="H10 Native RR BPM")

    ax2.set_ylabel("Heart Rate (BPM)", fontsize=11)
    ax2.set_xlabel("Session Time (s)", fontsize=10)
    ax2.set_title("Heart Rate Over Time", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Voltage distribution histogram
    ax3 = axes[2]
    ax3.hist(voltages, bins=100, color="#4CAF50", alpha=0.7, edgecolor="black")
    mean_v = float(np.mean(voltages))
    std_v = float(np.std(voltages))
    ax3.axvline(mean_v, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.1f} µV")
    ax3.axvline(mean_v + std_v, color="orange", linestyle=":", linewidth=1.2, label=f"±1 SD: {std_v:.1f} µV")
    ax3.axvline(mean_v - std_v, color="orange", linestyle=":", linewidth=1.2)
    ax3.set_xlabel("Voltage (µV)", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("ECG Voltage Distribution", fontsize=12)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    # Save figure
    target_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_out, dpi=150, bbox_inches="tight")
    print(f"✓ Saved diagnostic plot to: {target_out}")

    if show:
        print("\nDisplaying interactive plot (close window to exit)...")
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize Polar H10 ECG recording CSV.")
    parser.add_argument("csv_path", type=str, help="Path to recorded ECG CSV file")
    parser.add_argument("--out", type=str, default=None, help="Output PNG file path")
    parser.add_argument(
        "--detector",
        choices=["builtin", "neurokit"],
        default="builtin",
        help="QRS detection algorithm to use (default: builtin)",
    )
    parser.add_argument(
        "--compare-device-rr",
        action="store_true",
        help="Compare detected BPM against Polar H10 native RR stream if available",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive window")

    args = parser.parse_args()

    try:
        analyze_and_plot(
            csv_path=args.csv_path,
            out_path=args.out,
            detector=args.detector,
            compare_device_rr=args.compare_device_rr,
            show=not args.no_show,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
