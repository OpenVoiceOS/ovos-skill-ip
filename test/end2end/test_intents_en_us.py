"""End-to-end intent-routing tests for ovos-skill-ip (en-US).

Each case feeds an utterance through a MiniCroft stack running the default
pipeline (Adapt plus the Padatious-family plugins) and asserts it routes to
the expected handler. Coverage spans the plain address query (``ip_intent``)
across its verb and qualifier phrasings, and the trailing-part query
(``last_ip_digits_intent``). Both intents are registered from ``.intent`` files
and route via the Padatious-family pipeline; see
``test_intents_en_us_no_adapt.py`` for coverage that pins the pipeline to
exclude Adapt entirely.

The ``what_ssid.intent`` and ``wifi_signal.intent`` handlers only register
when ``iwlist`` is present on the host; their routing coverage is skipped
when it is absent, which is the case in CI.

Run: pytest test/end2end/ -v
"""
import time
from shutil import which
from unittest import TestCase, skipUnless

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import get_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"

PIPELINE = [
    "ovos-adapt-pipeline-plugin-high",
    "ovos-adapt-pipeline-plugin-medium",
    "ovos-adapt-pipeline-plugin-low",
    "ovos-padacioso-pipeline-plugin",
]


class _IntentRoutingMixin:
    """Shared MiniCroft setup for Adapt intent routing."""

    @classmethod
    def setUpClass(cls):
        cls.minicroft = get_minicroft([SKILL_ID])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "minicroft", None):
            cls.minicroft.stop()

    def _assert_intent(self, utterance: str, intent_name: str):
        intent_msg_type = f"{SKILL_ID}:{intent_name}"
        matched = []
        handler = lambda msg: matched.append(msg)
        self.minicroft.bus.on(intent_msg_type, handler)
        try:
            session = Session(f"e2e-en_us-{intent_name}-{hash(utterance)}")
            session.lang = LANG
            session.pipeline = PIPELINE
            self.minicroft.bus.emit(Message(
                "recognizer_loop:utterance",
                {"utterances": [utterance], "lang": LANG},
                {"session": session.serialize()},
            ))
            deadline = time.monotonic() + 15
            while not matched and time.monotonic() < deadline:
                time.sleep(0.2)
        finally:
            self.minicroft.bus.remove(intent_msg_type, handler)
        self.assertTrue(
            matched,
            f"{utterance!r} did not route to {intent_name}",
        )

    def _assert_not_intent(self, utterance: str, intent_name: str):
        intent_msg_type = f"{SKILL_ID}:{intent_name}"
        matched = []
        handler = lambda msg: matched.append(msg)
        self.minicroft.bus.on(intent_msg_type, handler)
        try:
            session = Session(f"e2e-en_us-not-{intent_name}-{hash(utterance)}")
            session.lang = LANG
            session.pipeline = PIPELINE
            self.minicroft.bus.emit(Message(
                "recognizer_loop:utterance",
                {"utterances": [utterance], "lang": LANG},
                {"session": session.serialize()},
            ))
            deadline = time.monotonic() + 5
            while not matched and time.monotonic() < deadline:
                time.sleep(0.2)
        finally:
            self.minicroft.bus.remove(intent_msg_type, handler)
        self.assertFalse(
            matched,
            f"{utterance!r} incorrectly routed to {intent_name}",
        )


class TestIPIntent(_IntentRoutingMixin, TestCase):
    """ip_intent — report the full IP address."""

    def test_whats_my_ip(self):
        self._assert_intent("what's my ip", "ip_intent")

    def test_how_do_i_find_my_ip(self):
        self._assert_intent("how do I find my ip", "ip_intent")

    def test_what_is_my_ip(self):
        self._assert_intent("what is my ip", "ip_intent")

    def test_whats_my_local_ip_address(self):
        self._assert_intent("what's my local ip address", "ip_intent")

    def test_give_me_my_ip_address(self):
        self._assert_intent("give me my ip address", "ip_intent")

    def test_tell_me_my_ip_address(self):
        self._assert_intent("tell me my ip address", "ip_intent")

    def test_show_me_my_network_address(self):
        self._assert_intent("show me my network address", "ip_intent")

    def test_public_ip_question_does_not_match_local_intent(self):
        self._assert_not_intent("what is my public ip", "ip_intent")

    def test_external_ip_question_does_not_match_local_intent(self):
        self._assert_not_intent("what is my external ip address", "ip_intent")


class TestPublicIPIntent(_IntentRoutingMixin, TestCase):
    """public_ip_intent — report the internet-facing (public) address."""

    def test_whats_my_public_ip(self):
        self._assert_intent("what is my public ip", "public_ip_intent")

    def test_whats_my_external_ip_address(self):
        self._assert_intent("what is my external ip address", "public_ip_intent")

    def test_tell_me_my_outside_ip(self):
        self._assert_intent("tell me my outside ip", "public_ip_intent")


@skipUnless(which("iwlist"), "what_ssid.intent only registers when iwlist is present")
class TestWifiPhrasings(_IntentRoutingMixin, TestCase):
    """what_ssid.intent — additional phrasings for the wifi/network name."""

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
    """last_ip_digits_intent — report only the trailing part of the address."""

    def test_last_digits_of_my_ip(self):
        self._assert_intent("what are the last digits of my ip", "last_ip_digits_intent")

    def test_read_the_last_part_of_my_ip(self):
        self._assert_intent("read the last part of my ip", "last_ip_digits_intent")

    def test_read_me_the_last_part_of_my_ip(self):
        self._assert_intent("read me the last part of my ip", "last_ip_digits_intent")

    def test_final_digits_of_my_ip_address(self):
        self._assert_intent("tell me the final digits of my ip address", "last_ip_digits_intent")
