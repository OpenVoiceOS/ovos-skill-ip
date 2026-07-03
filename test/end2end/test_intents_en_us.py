"""End-to-end intent-routing tests for ovos-skill-ip (en-US).

Each case feeds an utterance through a MiniCroft stack and asserts it routes
to the expected Adapt handler. Coverage spans the plain address query
(``IPIntent``) across its verb and qualifier phrasings, and the trailing-part
query (``LastIPDigitsIntent``).

The ``what.ssid.intent`` handler is intentionally not exercised here: it only
registers when ``iwlist`` is present on the host, which is not the case in CI.

Run: pytest test/end2end/ -v
"""
import time
from unittest import TestCase

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import get_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"

PIPELINE = [
    "ovos-adapt-pipeline-plugin-high",
    "ovos-adapt-pipeline-plugin-medium",
    "ovos-adapt-pipeline-plugin-low",
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


class TestIPIntent(_IntentRoutingMixin, TestCase):
    """IPIntent — report the full IP address."""

    def test_whats_my_ip(self):
        self._assert_intent("what's my ip", "IPIntent")

    def test_what_is_my_ip(self):
        self._assert_intent("what is my ip", "IPIntent")

    def test_whats_my_local_ip_address(self):
        self._assert_intent("what's my local ip address", "IPIntent")

    def test_tell_me_my_ip_address(self):
        self._assert_intent("tell me my ip address", "IPIntent")

    def test_show_me_my_network_address(self):
        self._assert_intent("show me my network address", "IPIntent")


class TestLastIPDigitsIntent(_IntentRoutingMixin, TestCase):
    """LastIPDigitsIntent — report only the trailing part of the address."""

    def test_last_digits_of_my_ip(self):
        self._assert_intent("what are the last digits of my ip", "LastIPDigitsIntent")

    def test_read_the_last_part_of_my_ip(self):
        self._assert_intent("read the last part of my ip", "LastIPDigitsIntent")

    def test_final_digits_of_my_ip_address(self):
        self._assert_intent("tell me the final digits of my ip address", "LastIPDigitsIntent")
