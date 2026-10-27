"""End-to-end intent-routing tests for ovos-skill-ip (en-US).

The ``what_ssid.intent`` and ``wifi_signal.intent`` handlers only register
when ``iwlist`` is present on the host; their routing tests skip without it.
"""
from shutil import which
from unittest import TestCase, skipUnless

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession
from ovoscope.routing import PIPELINE, TALLY, get_prototype_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"


class _IntentRoutingMixin:
    """Shared MiniCroft setup for intent routing."""

    @classmethod
    def setUpClass(cls):
        cls.minicroft = get_prototype_minicroft([SKILL_ID], lang=LANG)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "minicroft", None):
            cls.minicroft.stop()

    def _fired_types(self, utterance: str) -> list:
        session = Session(f"e2e-en_us-{hash(utterance)}")
        session.lang = LANG
        session.pipeline = PIPELINE
        capture = CaptureSession(self.minicroft)
        capture.capture(Message(
            "recognizer_loop:utterance",
            {"utterances": [utterance], "lang": LANG},
            {"session": session.serialize()},
        ), timeout=15)
        return [m.msg_type for m in capture.finish()]

    def _assert_intent(self, utterance: str, intent_name: str):
        TALLY.assert_routed(utterance, f"{SKILL_ID}:{intent_name}",
                            self._fired_types(utterance))

    def _assert_not_intent(self, utterance: str, intent_name: str):
        TALLY.assert_not_routed(utterance, f"{SKILL_ID}:{intent_name}",
                                self._fired_types(utterance))


class TestIPIntent(_IntentRoutingMixin, TestCase):
    """ip.intent — report the full IP address."""

    def test_whats_my_ip(self):
        self._assert_intent("what's my ip", "ip")

    def test_how_do_i_find_my_ip(self):
        self._assert_intent("how do I find my ip", "ip")

    def test_what_is_my_ip(self):
        self._assert_intent("what is my ip", "ip")

    def test_what_is_my_ip_address(self):
        self._assert_intent("what is my ip address", "ip")

    def test_whats_my_local_ip_address(self):
        self._assert_intent("what's my local ip address", "ip")

    def test_give_me_my_ip_address(self):
        self._assert_intent("give me my ip address", "ip")

    def test_tell_me_my_ip_address(self):
        self._assert_intent("tell me my ip address", "ip")

    def test_show_me_my_network_address(self):
        self._assert_intent("show me my network address", "ip")

    def test_public_ip_question_does_not_match_local_intent(self):
        self._assert_not_intent("what is my public ip", "ip")

    def test_external_ip_question_does_not_match_local_intent(self):
        self._assert_not_intent("what is my external ip address", "ip")


class TestPublicIPIntent(_IntentRoutingMixin, TestCase):
    """Publicip.intent — report the internet-facing (public) address."""

    def test_whats_my_public_ip(self):
        self._assert_intent("what is my public ip", "public_ip")

    def test_whats_my_external_ip_address(self):
        self._assert_intent("what is my external ip address", "public_ip")

    def test_tell_me_my_outside_ip(self):
        self._assert_intent("tell me my outside ip", "public_ip")


@skipUnless(which("iwlist"), "what.ssid.intent only registers when iwlist is present")
class TestWifiPhrasings(_IntentRoutingMixin, TestCase):
    """what.ssid.intent — additional phrasings for the wifi/network name."""

    def test_whats_my_wifi_called(self):
        self._assert_intent("what is my wifi called", "what_ssid.intent")

    def test_whats_my_wifi_called_again(self):
        self._assert_intent("what is my wifi called again", "what_ssid.intent")

    def test_can_you_please_tell_me_the_wifi_name(self):
        self._assert_intent("can you please tell me the wifi name",
                             "what_ssid.intent")

    def test_what_network_am_i_on(self):
        self._assert_intent("what network am I on", "what_ssid.intent")


@skipUnless(which("iwlist"), "wifi_signal.intent only registers when iwlist is present")
class TestWifiSignalIntent(_IntentRoutingMixin, TestCase):
    """wifi_signal.intent — report the wifi signal strength."""

    def test_whats_my_wifi_signal_strength(self):
        self._assert_intent("what's my wifi signal strength", "wifi_signal.intent")

    def test_how_good_is_my_wifi_connection(self):
        self._assert_intent("how good is my wifi connection", "wifi_signal.intent")


class TestLastIPDigitsIntent(_IntentRoutingMixin, TestCase):
    """last_ip_digits.intent — report only the trailing part of the address."""

    def test_last_digits_of_my_ip(self):
        self._assert_intent("what are the last digits of my ip", "last_ip_digits")

    def test_read_the_last_part_of_my_ip(self):
        self._assert_intent("read the last part of my ip", "last_ip_digits")

    def test_read_me_the_last_part_of_my_ip(self):
        self._assert_intent("read me the last part of my ip", "last_ip_digits")

    def test_final_digits_of_my_ip_address(self):
        self._assert_intent("tell me the final digits of my ip address", "last_ip_digits")
