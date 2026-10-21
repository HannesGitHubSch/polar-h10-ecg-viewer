"""
Bluetooth Low Energy interface for Polar H10 heart rate sensors using Bleak and polar-python.
"""

import asyncio
import logging
from collections.abc import Callable

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from polar_python.device import PolarDevice
from polar_python.models import ECGData, HRData

from polar_ecg.models import DeviceInfo

logger = logging.getLogger(__name__)


class _PolarDeviceWithDisconnect(PolarDevice):
    """
    Subclass of PolarDevice that injects a disconnected_callback into BleakClient
    to ensure unexpected connection drops are reported immediately.
    """

    def __init__(self, ble_device: BLEDevice, on_disconnect: Callable[[], None]):
        super().__init__(ble_device)
        # Note: polar-python creates an internal BleakClient. We replace it here
        # with our client configured with a disconnect notification callback.
        self._client = BleakClient(
            ble_device,
            disconnected_callback=lambda _client: on_disconnect(),
        )


class PolarH10:
    """
    High-level manager for Polar H10 BLE connection, ECG streaming,
    and Heart Rate / RR interval streaming.
    """

    def __init__(self):
        self._polar_device: _PolarDeviceWithDisconnect | None = None
        self._ble_device: BLEDevice | None = None
        self._is_streaming_ecg = False
        self._is_streaming_hr = False

    @property
    def is_connected(self) -> bool:
        if self._polar_device and self._polar_device._client:
            return bool(self._polar_device._client.is_connected)
        return False

    async def scan(self, timeout: float = 5.0) -> list[tuple[DeviceInfo, BLEDevice]]:
        """
        Scan for nearby Polar devices.
        Returns a list of (DeviceInfo, BLEDevice) tuples.
        """
        logger.info("Scanning for Polar devices (timeout=%.1fs)...", timeout)
        devices = await BleakScanner.discover(timeout=timeout)
        polar_devices = []
        for d in devices:
            if d.name and "Polar" in d.name:
                rssi = getattr(d, "rssi", None)
                info = DeviceInfo(name=d.name, address=d.address, rssi=rssi)
                polar_devices.append((info, d))

        logger.info("Found %d Polar device(s)", len(polar_devices))
        return polar_devices

    async def connect(
        self,
        ble_device: BLEDevice,
        on_disconnect: Callable[[], None],
        timeout: float = 15.0,
    ) -> None:
        """
        Connect to a specified BLEDevice.
        Directly uses discovered BLEDevice to avoid redundant scanning.
        """
        if self.is_connected:
            if self._ble_device and self._ble_device.address == ble_device.address:
                logger.info("Already connected to %s (%s)", ble_device.name, ble_device.address)
                return
            logger.info("Disconnecting previous connection before connecting to %s...", ble_device.name)
            await self.disconnect()

        if self._polar_device:
            try:
                await self._polar_device.disconnect()
            except Exception:
                pass
            self._polar_device = None

        logger.info("Connecting to %s (%s)...", ble_device.name, ble_device.address)
        self._ble_device = ble_device
        self._polar_device = _PolarDeviceWithDisconnect(ble_device, on_disconnect)

        await asyncio.wait_for(self._polar_device.connect(), timeout=timeout)
        logger.info("Successfully connected to %s", ble_device.name)

    async def start_ecg(
        self,
        cb: Callable[[ECGData], None],
        sample_rate: int = 130,
        resolution: int = 14,
    ) -> None:
        """
        Start raw ECG data streaming.
        Polar H10 supports sample_rate=130 and resolution=14.
        """
        if not self._polar_device or not self.is_connected:
            raise RuntimeError("Cannot start ECG: device is not connected")

        logger.info("Starting ECG stream (fs=%d Hz, res=%d bits)...", sample_rate, resolution)
        await self._polar_device.start_ecg_stream(
            ecg_callback=cb,
            sample_rate=sample_rate,
            resolution=resolution,
        )
        self._is_streaming_ecg = True

    async def start_hr(self, cb: Callable[[HRData], None]) -> None:
        """
        Start standard Bluetooth Heart Rate / RR interval streaming.
        Note: polar-python parses 8-bit HR and RR intervals in 1/1024 s units.
        """
        if not self._polar_device or not self.is_connected:
            raise RuntimeError("Cannot start HR: device is not connected")

        logger.info("Starting native HR / RR stream...")
        await self._polar_device.start_hr_stream(hr_callback=cb)
        self._is_streaming_hr = True

    async def stop_all(self) -> None:
        """
        Stop active ECG and HR streams, tolerating already closed connections.
        """
        if not self._polar_device:
            return

        if self._is_streaming_ecg:
            try:
                await self._polar_device.stop_ecg_stream()
            except Exception as e:  # noqa: BLE001
                logger.debug("Tolerated error stopping ECG stream: %s", e)
            self._is_streaming_ecg = False

        if self._is_streaming_hr:
            try:
                await self._polar_device.stop_hr_stream()
            except Exception as e:  # noqa: BLE001
                logger.debug("Tolerated error stopping HR stream: %s", e)
            self._is_streaming_hr = False

    async def disconnect(self) -> None:
        """Disconnect and release device resources."""
        await self.stop_all()
        if self._polar_device:
            try:
                await self._polar_device.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.debug("Tolerated error during disconnect: %s", e)
            self._polar_device = None
            self._ble_device = None
        logger.info("Device disconnected.")
