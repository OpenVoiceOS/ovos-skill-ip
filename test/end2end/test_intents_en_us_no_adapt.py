"""Adapt-less end-to-end intent-routing test for ovos-skill-ip (en-US).

Pins MiniCroft to the padacioso pipeline only (no Adapt plugins in the
stack) to prove ``IPIntent``/``LastIPDigitsIntent`` are registered as
``.intent`` (padatious-family) intents rather than Adapt intents.

Run: pytest test/end2end/test_intents_en_us_no_adapt.py -v
"""
import time
from unittest import TestCase

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import get_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"

PIPELINE = ["ovos-padacioso-pipeline-plugin"]


class TestNoAdaptIntentRouting(TestCase):
    """Route utterances with Adapt excluded from the pipeline."""

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
            session = Session(f"e2e-noadapt-{intent_name}-{hash(utterance)}")
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
            f"{utterance!r} did not route to {intent_name} via {PIPELINE}",
        )

    def test_what_is_my_ip_address(self):
        self._assert_intent("what is my ip address", "IPIntent")

    def test_last_digits_of_my_ip(self):
        self._assert_intent("what are the last digits of my ip", "LastIPDigitsIntent")
