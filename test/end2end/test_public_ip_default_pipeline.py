"""Public-ip questions route to public_ip.intent, and a plain ip question does not (en-US)."""
from unittest import TestCase

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession
from ovoscope.routing import PIPELINE, TALLY, get_prototype_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"


class TestPublicIPDefaultPipeline(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.minicroft = get_prototype_minicroft([SKILL_ID], lang=LANG)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "minicroft", None):
            cls.minicroft.stop()

    def _assert_intent(self, utterance: str, intent_name: str, timeout=15):
        session = Session(f"e2e-default-pipeline-{hash(utterance)}")
        session.lang = LANG
        session.pipeline = PIPELINE
        capture = CaptureSession(self.minicroft)
        capture.capture(Message(
            "recognizer_loop:utterance",
            {"utterances": [utterance], "lang": LANG},
            {"session": session.serialize()},
        ), timeout=timeout)
        TALLY.assert_routed(utterance, f"{SKILL_ID}:{intent_name}",
                            [m.msg_type for m in capture.finish()])

    def test_whats_my_public_ip_routes_to_public_ip(self):
        self._assert_intent("what is my public ip", "public_ip")

    def test_whats_my_external_ip_routes_to_public_ip(self):
        self._assert_intent("what's my external ip address", "public_ip")

    def test_bare_local_ip_question_unaffected(self):
        self._assert_intent("what is my ip", "ip")
