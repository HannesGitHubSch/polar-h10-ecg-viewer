"""
Polar H10 Backend Handler
Manages BLE connection, ECG streaming, and R-peak detection
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, List
from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy.signal import find_peaks
from bleak import BleakScanner
from polar_python import PolarDevice, MeasurementSettings, SettingType, ECGData

try:
    from config import DEBUG_PEAK_DETECTION, BPM_SMOOTHING_WINDOW, ECG_BUFFER_SECONDS
except ImportError:
    # Fallback values if config not found
    DEBUG_PEAK_DETECTION = False
    BPM_SMOOTHING_WINDOW = 2
    ECG_BUFFER_SECONDS = 30


@dataclass
class ECGSample:
    """Single ECG data point with timestamp"""
    timestamp: float  # Polar internal timestamp (seconds)
    voltage: float    # in microvolts
    system_receive_time: Optional[datetime] = None  # System datetime when packet received
    

@dataclass
class RPeak:
    """Detected R-peak with timestamp and RR interval"""
    timestamp: float
    rri: Optional[float] = None  # RR interval in ms
    bpm: Optional[float] = None  # Instantaneous BPM


class PolarH10Handler:
    """
    Handles connection to Polar H10 and streams ECG data with R-peak detection
    """
    
    def __init__(self, 
                 ecg_callback: Callable[[List[ECGSample]], None],
                 rpeak_callback: Callable[[RPeak], None],
                 sample_rate: int = 130):
        """
        Initialize Polar H10 handler
        
        Args:
            ecg_callback: Callback function for ECG data chunks
            rpeak_callback: Callback function for detected R-peaks
            sample_rate: ECG sample rate in Hz (130 or 250)
        """
        self.ecg_callback = ecg_callback
        self.rpeak_callback = rpeak_callback
        self.sample_rate = sample_rate
        
        self.polar_device: Optional[PolarDevice] = None
        self.is_streaming = False
        self.start_time: Optional[float] = None
        self._first_data_received = False  # For debug output
        
        # R-peak detection parameters
        self.ecg_buffer = deque(maxlen=int(sample_rate * ECG_BUFFER_SECONDS))
        self.last_rpeak_time: Optional[float] = None
        self.last_debug_time: float = 0  # For throttling debug output
        
        # RRI smoothing for BPM calculation
        self.recent_rris = deque(maxlen=BPM_SMOOTHING_WINDOW)
        
        # Adaptive threshold parameters
        self.recent_peaks = deque(maxlen=8)  # Track recent peak heights
        self.peak_distance = int(sample_rate * 0.3)  # Min 300ms between R-peaks
        
        # Signal quality tracking
        self.signal_initialized = False
        
    async def scan_devices(self, timeout: float = 5.0) -> List[str]:
        """
        Scan for available Polar devices
        
        Returns:
            List of device names found
        """
        devices = await BleakScanner.discover(timeout=timeout)
        polar_devices = [d.name for d in devices if d.name and "Polar" in d.name]
        return polar_devices
        
    async def connect(self, device_name: str = "Polar H10") -> bool:
        """
        Connect to Polar H10 device
        
        Args:
            device_name: Name of device to connect to
            
        Returns:
            True if connection successful
        """
        # Find device
        device = await BleakScanner.find_device_by_filter(
            lambda bd, ad: bd.name and device_name in bd.name,
            timeout=10
        )
        
        if device is None:
            return False
            
        # Connect to device
        self.polar_device = PolarDevice(device, self._data_callback)
        await self.polar_device.__aenter__()
        
        return True
        
    async def start_streaming(self) -> bool:
        """
        Start ECG streaming
        
        Returns:
            True if streaming started successfully
        """
        if self.polar_device is None:
            return False
            
        # Configure ECG settings
        ecg_settings = MeasurementSettings(
            measurement_type="ECG",
            settings=[
                SettingType(type="SAMPLE_RATE", values=[self.sample_rate]),
                SettingType(type="RESOLUTION", values=[14]),
            ],
        )
        
        # Start streaming
        await self.polar_device.start_stream(ecg_settings)
        self.is_streaming = True
        self.start_time = time.time()
        
        return True
        
    async def stop_streaming(self):
        """Stop ECG streaming"""
        if self.polar_device and self.is_streaming:
            await self.polar_device.stop_stream("ECG")
            self.is_streaming = False
            
    async def disconnect(self):
        """Disconnect from device"""
        if self.polar_device:
            await self.polar_device.__aexit__(None, None, None)
            self.polar_device = None
            
    def _data_callback(self, data):
        """
        Internal callback for receiving data from Polar device
        
        Args:
            data: ECGData from polar-python
        """
        if not isinstance(data, ECGData):
            return
            
        # Convert to ECGSample objects with timestamps
        current_time = time.time()
        time_offset = current_time - self.start_time if self.start_time else 0
        
        # ECGData might have 'data' or be iterable directly
        ecg_values = None
        attr_used = None
        try:
            # Try different possible attribute names
            if hasattr(data, 'samples'):
                ecg_values = data.samples
                attr_used = 'samples'
            elif hasattr(data, 'data'):
                ecg_values = data.data
                attr_used = 'data'
            elif hasattr(data, 'ecg_data'):
                ecg_values = data.ecg_data
                attr_used = 'ecg_data'
            else:
                # Try to iterate directly
                ecg_values = list(data)
                attr_used = 'direct iteration'
                
            # Debug output on first successful read
            if not self._first_data_received and ecg_values:
                if DEBUG_PEAK_DETECTION:
                    print(f"✓ ECG data streaming - using attribute: {attr_used}")
                    print(f"  Received {len(ecg_values)} samples")
                self._first_data_received = True
                
        except Exception as e:
            if DEBUG_PEAK_DETECTION:
                print(f"Error accessing ECG data: {e}")
                print(f"ECGData attributes: {dir(data)}")
            return
        
        if not ecg_values:
            if DEBUG_PEAK_DETECTION:
                print("Warning: No ECG values found in data")
            return
        
        # Capture exact system datetime when this packet is received
        packet_receive_datetime = datetime.now()
        
        samples = []
        for i, voltage in enumerate(ecg_values):
            # Polar internal timestamp
            polar_timestamp = time_offset + (i / self.sample_rate)
            
            # Calculate system datetime for this sample:
            # The LAST sample gets the exact receive time
            # Earlier samples are back-calculated using sample rate
            samples_from_end = len(ecg_values) - 1 - i
            time_offset_from_last = samples_from_end / self.sample_rate
            sample_datetime = packet_receive_datetime - timedelta(seconds=time_offset_from_last)
            
            sample = ECGSample(
                timestamp=polar_timestamp, 
                voltage=float(voltage),
                system_receive_time=sample_datetime
            )
            samples.append(sample)
            
            # Add to buffer for R-peak detection
            self.ecg_buffer.append(sample)
            
        # Send ECG data to callback
        if self.ecg_callback:
            self.ecg_callback(samples)
            
        # Perform R-peak detection on buffer (checks periodically, not every time)
        self._detect_rpeaks()
        
    def _detect_rpeaks(self):
        """
        Detect R-peaks in the ECG buffer using improved adaptive algorithm
        """
        # Need enough data
        if len(self.ecg_buffer) < self.peak_distance * 2:
            return
            
        # Extract voltage values and timestamps
        voltages = np.array([s.voltage for s in self.ecg_buffer])
        timestamps = np.array([s.timestamp for s in self.ecg_buffer])
        
        # Only analyze the most recent 1.5 seconds to avoid reprocessing
        recent_window = int(self.sample_rate * 1.5)
        if len(voltages) > recent_window:
            analysis_start_idx = len(voltages) - recent_window
            voltages = voltages[analysis_start_idx:]
            timestamps = timestamps[analysis_start_idx:]
        else:
            analysis_start_idx = 0
        
        # Apply simple moving average filter to reduce noise
        window_size = 5
        if len(voltages) >= window_size:
            voltages_filtered = np.convolve(voltages, np.ones(window_size)/window_size, mode='same')
        else:
            voltages_filtered = voltages
            
        # Calculate signal statistics
        signal_std = np.std(voltages_filtered)
        signal_mean = np.mean(voltages_filtered)
        signal_min = np.min(voltages_filtered)
        signal_max = np.max(voltages_filtered)
        
        # Determine if we should print debug (needed early for threshold logic)
        current_time = time.time()
        should_debug = (current_time - self.last_debug_time) > 5.0  # Every 5s or when peaks found (updated later)
        
        # Initialize adaptive threshold if needed
        if not self.signal_initialized and len(voltages) >= self.sample_rate:
            # Use percentile-based threshold for initialization
            self.signal_initialized = True
            if DEBUG_PEAK_DETECTION:
                print(f"✓ R-peak detection initialized - mean: {signal_mean:.1f}µV, std: {signal_std:.1f}µV")
                print(f"  Signal range: {signal_min:.1f} to {signal_max:.1f}µV")
        
        # Adaptive threshold based on recent peaks or signal statistics
        if len(self.recent_peaks) >= 3:
            # Use median of recent peaks as reference, but cap it at signal max
            peak_median = np.median(self.recent_peaks)
            # Adaptive threshold should be reasonable relative to signal
            adaptive_threshold = peak_median * 0.6  # 60% of typical peak
            statistical_threshold = signal_mean + 0.5 * signal_std
            
            # Check if adaptive threshold is completely unreasonable (>2x signal max)
            # This means we have bad data in recent_peaks
            if adaptive_threshold > signal_max * 2.0:
                if should_debug and DEBUG_PEAK_DETECTION:
                    print(f"⚠ Resetting recent_peaks - threshold {adaptive_threshold:.0f}µV too high for signal max {signal_max:.0f}µV")
                self.recent_peaks.clear()
                height_threshold = statistical_threshold
                threshold_type = "reset"
            else:
                # Use the lower of adaptive or statistical (prevents runaway thresholds)
                # Also ensure it's not higher than 80% of current signal max
                max_reasonable = signal_max * 0.8
                height_threshold = min(adaptive_threshold, statistical_threshold, max_reasonable)
                
                if adaptive_threshold < statistical_threshold and adaptive_threshold < max_reasonable:
                    threshold_type = "adaptive"
                else:
                    threshold_type = f"capped ({adaptive_threshold:.0f}→{height_threshold:.0f})"
        else:
            # Not enough peaks yet - use statistical threshold
            height_threshold = signal_mean + 0.5 * signal_std
            threshold_type = "statistical"
            
        # Find peaks with relaxed prominence for better detection
        peaks, properties = find_peaks(
            voltages_filtered,
            height=height_threshold,
            distance=self.peak_distance,
            prominence=signal_std * 0.2  # Reduced from 0.3 for better sensitivity
        )
        
        # Update debug flag if peaks found
        should_debug = should_debug or len(peaks) > 0
        
        if should_debug and DEBUG_PEAK_DETECTION:
            self.last_debug_time = current_time
            # DEBUG: Print what find_peaks found
            print(f"\n--- R-peak Detection Check ---")
            print(f"Buffer: {len(self.ecg_buffer)} samples total, analyzing last {len(voltages)} samples")
            print(f"Analysis window: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s ({timestamps[-1]-timestamps[0]:.2f}s)")
            print(f"Signal range: [{signal_min:.1f}, {signal_max:.1f}]µV")
            print(f"Threshold: {height_threshold:.1f}µV ({threshold_type})")
            median_str = f"{np.median(self.recent_peaks):.1f}" if len(self.recent_peaks) > 0 else "N/A"
            print(f"Recent peaks tracked: {len(self.recent_peaks)} (median: {median_str})")
            last_peak_str = f"{self.last_rpeak_time:.2f}s" if self.last_rpeak_time else "None"
            print(f"Last R-peak was at: {last_peak_str}")
            print(f"Found {len(peaks)} candidate peaks in analysis window")
            if len(peaks) > 0:
                print(f"  Peak indices: {peaks[:5]}...")  # Show first 5
                print(f"  Peak heights: {[f'{voltages_filtered[p]:.1f}' for p in peaks[:5]]}")
                print(f"  Peak times: {[f'{timestamps[p]:.2f}' for p in peaks[:5]]}")
        
        # Only process peaks that are after the last detected one
        new_peaks = []
        current_last_time = self.last_rpeak_time  # Track within this check
        
        for peak_idx in peaks:
            peak_time = timestamps[peak_idx]
            
            # Check if this is a new peak (after last detected)
            time_since_last = (peak_time - current_last_time) if current_last_time else 999
            is_new = current_last_time is None or time_since_last > 0.25
            
            if should_debug and DEBUG_PEAK_DETECTION:
                print(f"  Peak at {peak_time:.2f}s (idx {peak_idx}): height={voltages_filtered[peak_idx]:.1f}µV, " +
                      f"time_since_last={time_since_last:.3f}s, is_new={is_new}")
            
            if is_new:
                new_peaks.append((peak_idx, peak_time))
                current_last_time = peak_time  # Update for next peak in same check
                
        if should_debug and DEBUG_PEAK_DETECTION:
            print(f"→ Processing {len(new_peaks)} NEW peaks")
                
        # Process new peaks
        for peak_idx, peak_time in new_peaks:
            peak_height = voltages_filtered[peak_idx]
            
            # Validate peak height is reasonable (not a huge outlier)
            # Should be within reasonable range of signal max
            if peak_height > signal_max * 1.5:
                if should_debug and DEBUG_PEAK_DETECTION:
                    print(f"✗ Peak rejected: height {peak_height:.1f}µV is outlier (>1.5x signal_max)")
                continue
            
            # Update recent peaks tracking (only if reasonable)
            self.recent_peaks.append(peak_height)
            
            # Calculate RR interval
            rri = None
            bpm = None
            if self.last_rpeak_time is not None:
                rri = (peak_time - self.last_rpeak_time) * 1000  # Convert to ms
                
                # Add to RRI buffer for smoothing
                if rri > 0:
                    self.recent_rris.append(rri)
                    
                    # Calculate BPM from smoothed RRI (average of recent RRIs)
                    if len(self.recent_rris) >= 1:
                        avg_rri = np.mean(self.recent_rris)
                        bpm = 60000 / avg_rri  # Convert to BPM
                    
            # Only emit if BPM is physiologically reasonable (30-220 BPM)
            if bpm is None or (30 <= bpm <= 220):
                bpm_str = f"{bpm:.1f}" if bpm is not None else "N/A"
                if DEBUG_PEAK_DETECTION:
                    print(f"✓ R-PEAK DETECTED at {peak_time:.2f}s: height={peak_height:.1f}µV, BPM={bpm_str}")
                
                rpeak = RPeak(
                    timestamp=peak_time,
                    rri=rri,
                    bpm=bpm
                )
                
                # Send to callback
                if self.rpeak_callback:
                    self.rpeak_callback(rpeak)
                    
                self.last_rpeak_time = peak_time
            else:
                if should_debug and DEBUG_PEAK_DETECTION:
                    print(f"✗ Peak rejected: BPM {bpm:.1f} out of range [30-220]")