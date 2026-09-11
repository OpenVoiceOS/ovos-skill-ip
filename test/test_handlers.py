"""Unit coverage for the handlers added/changed by issue #64.

Network calls (public-IP lookup) and the ``iwlist`` scan are mocked so these
tests run without internet access or a wireless adapter.
"""
import unittest
from unittest.mock import MagicMock, patch

import requests
from ovos_utils.fakebus import FakeBus

from ovos_skill_ip import IPSkill


def _make_skill():
    bus = FakeBus()
    skill = IPSkill()
    skill._startup(bus, "ovos-skill-ip.openvoiceos")
    skill.speak_dialog = MagicMock()
    return skill


class TestPublicIP(unittest.TestCase):
    def test_public_ip_success(self):
        skill = _make_skill()
        with patch("ovos_skill_ip.requests.get") as mock_get:
            mock_get.return_value = MagicMock(text="8.8.8.8")
            mock_get.return_value.raise_for_status = MagicMock()
            skill.handle_query_public_IP(MagicMock())
        skill.speak_dialog.assert_called_once()
        dialog, data = skill.speak_dialog.call_args[0][0], skill.speak_dialog.call_args[0][1]
        self.assertEqual(dialog, "my.public.ip")
        self.assertEqual(data["ip"], "8 dot 8 dot 8 dot 8")

    def test_public_ip_network_error_speaks_error_dialog(self):
        skill = _make_skill()
        with patch("ovos_skill_ip.requests.get",
                   side_effect=requests.exceptions.ConnectionError()):
            skill.handle_query_public_IP(MagicMock())
        skill.speak_dialog.assert_called_once_with("public.ip.error")

    def test_public_ip_timeout_speaks_error_dialog(self):
        skill = _make_skill()
        with patch("ovos_skill_ip.requests.get",
                   side_effect=requests.exceptions.Timeout()):
            skill.handle_query_public_IP(MagicMock())
        skill.speak_dialog.assert_called_once_with("public.ip.error")


class TestWifiSignal(unittest.TestCase):
    IWLIST_OUTPUT = (
        b'wlan0     Scan completed :\n'
        b'          Cell 01 - Address: AA:BB:CC:DD:EE:FF\n'
        b'                    ESSID:"MyNetwork"\n'
        b'                    Quality=70/70  Signal level=-40 dBm\n'
    )

    def test_scan_wifi_parses_essid_and_quality(self):
        skill = _make_skill()
        with patch("ovos_skill_ip.check_output", return_value=self.IWLIST_OUTPUT):
            ssid, quality = skill.scan_wifi()
        self.assertEqual(ssid, "MyNetwork")
        self.assertEqual(quality, "70/70")

    def test_wifi_signal_query_speaks_quality(self):
        skill = _make_skill()
        with patch("ovos_skill_ip.get_ifaces", return_value={"wlan0": "192.168.1.5"}), \
             patch("ovos_skill_ip.check_output", return_value=self.IWLIST_OUTPUT):
            skill.handle_wifi_signal_query(MagicMock())
        skill.speak_dialog.assert_called_once_with("wifi.signal", {"quality": "70/70"})

    def test_wifi_signal_query_falls_back_when_no_signal_data(self):
        skill = _make_skill()
        with patch("ovos_skill_ip.get_ifaces", return_value={"wlan0": "192.168.1.5"}), \
             patch("ovos_skill_ip.check_output", return_value=b"wlan0     No scan results\n"):
            skill.handle_wifi_signal_query(MagicMock())
        skill.speak_dialog.assert_called_once_with("ethernet.connection")

    def test_wifi_signal_query_no_network(self):
        skill = _make_skill()
        with patch("ovos_skill_ip.get_ifaces", return_value={}):
            skill.handle_wifi_signal_query(MagicMock())
        skill.speak_dialog.assert_called_once_with("no_network_connection")


if __name__ == "__main__":
    unittest.main()
