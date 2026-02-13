#!/usr/bin/env python3
"""
Plot recorded ECG data from CSV file
Simple sanity check for recorded sessions
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks


def load_recording(csv_path):
    """Load ECG recording from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} samples from {csv_path}")
        print(f"  Participant ID: {df['participant_id'].iloc[0]}")
        
        # Parse datetime
        df['sample_datetime'] = pd.to_datetime(df['sample_datetime'])
        
        # Calculate duration
        duration = (df['sample_datetime'].iloc[-1] - df['sample_datetime'].iloc[0]).total_seconds()
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {len(df)/duration:.1f} Hz")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        sys.exit(1)


def detect_rpeaks_and_bpm(df, sample_rate=130):
    """Detect R-peaks and calculate BPM"""
    voltages = df['ecg_voltage_uv'].values
    
    # Apply moving average filter
    window = 5
    voltages_filtered = np.convolve(voltages, np.ones(window)/window, mode='same')
    
    # Calculate adaptive threshold
    signal_mean = np.mean(voltages_filtered)
    signal_std = np.std(voltages_filtered)
    height_threshold = signal_mean + 0.5 * signal_std
    
    # Find peaks
    peaks, properties = find_peaks(
        voltages_filtered,
        height=height_threshold,
        distance=int(sample_rate * 0.3),  # Min 300ms between peaks
        prominence=signal_std * 0.2
    )
    
    print(f"\n💓 R-peak Detection:")
    print(f"  Found {len(peaks)} R-peaks")
    
    if len(peaks) < 2:
        return None, None, None
    
    # Calculate BPM from R-peaks
    peak_times = df['polar_timestamp_s'].iloc[peaks].values
    rr_intervals = np.diff(peak_times)  # in seconds
    bpm_values = 60.0 / rr_intervals  # convert to BPM
    bpm_times = peak_times[1:]  # BPM values are between peaks
    
    # Filter physiological range
    valid_mask = (bpm_values >= 30) & (bpm_values <= 220)
    bpm_values = bpm_values[valid_mask]
    bpm_times = bpm_times[valid_mask]
    
    mean_bpm = np.mean(bpm_values)
    print(f"  Mean BPM: {mean_bpm:.1f}")
    print(f"  BPM range: [{bpm_values.min():.1f}, {bpm_values.max():.1f}]")
    
    return peaks, bpm_times, bpm_values


def check_timestamps(df):
    """Validate timestamp consistency"""
    print("\n📊 Timestamp Analysis:")
    
    # Check for gaps
    time_diffs = df['sample_datetime'].diff().dt.total_seconds()
    expected_interval = 1.0 / 130.0  # 130 Hz
    
    # Find large gaps (>2x expected interval)
    large_gaps = time_diffs[time_diffs > expected_interval * 2]
    if len(large_gaps) > 0:
        print(f"  ⚠ Found {len(large_gaps)} gaps in data")
        print(f"    Largest gap: {large_gaps.max():.3f}s")
    else:
        print(f"  ✓ No significant gaps detected")
    
    # Check sample interval statistics
    intervals = time_diffs[1:]  # Skip first NaN
    print(f"  Mean interval: {intervals.mean():.6f}s (expected: {expected_interval:.6f}s)")
    print(f"  Std interval: {intervals.std():.6f}s")


def plot_ecg(df, peaks=None, bpm_times=None, bpm_values=None):
    """Create ECG visualization"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Get participant ID for title
    participant_id = df['participant_id'].iloc[0]
    start_time = df['sample_datetime'].iloc[0]
    
    # 1. Full ECG trace with R-peaks
    ax1 = axes[0]
    ax1.plot(df['polar_timestamp_s'], df['ecg_voltage_uv'], 
             color='#2E86AB', linewidth=0.5, alpha=0.8, label='ECG')
    
    # Mark R-peaks
    if peaks is not None:
        ax1.scatter(df['polar_timestamp_s'].iloc[peaks], 
                   df['ecg_voltage_uv'].iloc[peaks],
                   color='red', s=30, zorder=5, alpha=0.6, label='R-peaks')
    
    ax1.set_ylabel('ECG (µV)', fontsize=11)
    ax1.set_title(f'ECG Recording - Participant {participant_id} - {start_time.strftime("%Y-%m-%d %H:%M:%S")}', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. BPM over time
    ax2 = axes[1]
    if bpm_times is not None and bpm_values is not None and len(bpm_values) > 0:
        ax2.plot(bpm_times, bpm_values, 
                 color='#A23B72', linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        # Add mean line
        mean_bpm = np.mean(bpm_values)
        ax2.axhline(mean_bpm, color='#F18F01', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_bpm:.1f} BPM')
        
        ax2.set_ylabel('Heart Rate (BPM)', fontsize=11)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_title('Heart Rate Over Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_ylim(40, 180)
    else:
        ax2.text(0.5, 0.5, 'No R-peaks detected\n(check electrode contact)', 
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax2.transAxes)
        ax2.set_ylabel('Heart Rate (BPM)', fontsize=11)
        ax2.set_title('Heart Rate Over Time', fontsize=12)
    
    # 3. ECG voltage distribution
    ax3 = axes[2]
    ax3.hist(df['ecg_voltage_uv'], bins=100, color='#4CAF50', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('ECG Voltage (µV)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('ECG Voltage Distribution', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_voltage = df['ecg_voltage_uv'].mean()
    std_voltage = df['ecg_voltage_uv'].std()
    ax3.axvline(mean_voltage, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_voltage:.1f} µV')
    ax3.axvline(mean_voltage + std_voltage, color='orange', linestyle=':', linewidth=1.5, label=f'±1 SD: {std_voltage:.1f} µV')
    ax3.axvline(mean_voltage - std_voltage, color='orange', linestyle=':', linewidth=1.5)
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    return fig


def print_stats(df):
    """Print recording statistics"""
    print("\n📈 ECG Statistics:")
    print(f"  Mean: {df['ecg_voltage_uv'].mean():.2f} µV")
    print(f"  Std:  {df['ecg_voltage_uv'].std():.2f} µV")
    print(f"  Min:  {df['ecg_voltage_uv'].min():.2f} µV")
    print(f"  Max:  {df['ecg_voltage_uv'].max():.2f} µV")
    print(f"  Range: {df['ecg_voltage_uv'].max() - df['ecg_voltage_uv'].min():.2f} µV")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python plot_recording.py <path_to_csv>")
        print("\nExample:")
        print("  python plot_recording.py ECG_P001_20260213_153045.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Load data
    df = load_recording(csv_path)
    
    # Validate timestamps
    check_timestamps(df)
    
    # Print statistics
    print_stats(df)
    
    # Detect R-peaks and calculate BPM
    peaks, bpm_times, bpm_values = detect_rpeaks_and_bpm(df)
    
    # Create plots
    print("\n🎨 Generating plots...")
    fig = plot_ecg(df, peaks, bpm_times, bpm_values)
    
    # Save figure
    output_path = csv_path.replace('.csv', '_plot.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    
    # Show interactive plot
    print("\n👁 Displaying interactive plot (close window to exit)...")
    plt.show()


if __name__ == "__main__":
    main()