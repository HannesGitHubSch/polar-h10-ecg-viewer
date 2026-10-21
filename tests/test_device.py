"""
Unit tests for PolarH10 BLE device wrapper and BleakScanner filtering.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from bleak.backends.device import BLEDevice

from polar_ecg.device import PolarH10, _PolarDeviceWithDisconnect
from polar_ecg.models import DeviceInfo


def test_polar_device_has_client_attribute():
    dummy_device = BLEDevice("00:11:22:33:44:55", "Polar H10 12345678", None)
    disconnect_called = False

    def on_disconnect():
        nonlocal disconnect_called
        disconnect_called = True

    wrapped = _PolarDeviceWithDisconnect(dummy_device, on_disconnect)
    assert hasattr(wrapped, "_client")
    assert wrapped._client.address == "00:11:22:33:44:55"


def test_polar_h10_scan_filter():
    async def run_scan():
        polar_dev = BLEDevice("11:22:33:44:55:66", "Polar H10 ABCD1234", None)
        other_dev = BLEDevice("22:33:44:55:66:77", "Random BLE Device", None)
        unnamed_dev = BLEDevice("33:44:55:66:77:88", None, None)

        with patch("polar_ecg.device.BleakScanner.discover", new_callable=AsyncMock) as mock_discover:
            mock_discover.return_value = [polar_dev, other_dev, unnamed_dev]

            h10 = PolarH10()
            found = await h10.scan(timeout=1.0)

            assert len(found) == 1
            info, ble = found[0]
            assert isinstance(info, DeviceInfo)
            assert info.name == "Polar H10 ABCD1234"
            assert info.address == "11:22:33:44:55:66"
            assert ble == polar_dev

    asyncio.run(run_scan())


def test_polar_h10_connect_timeout():
    async def run_timeout():
        dummy_device = BLEDevice("00:11:22:33:44:55", "Polar H10", None)
        h10 = PolarH10()

        with patch.object(_PolarDeviceWithDisconnect, "connect", side_effect=TimeoutError), pytest.raises(TimeoutError):
            await h10.connect(dummy_device, on_disconnect=lambda: None, timeout=0.1)

    asyncio.run(run_timeout())
