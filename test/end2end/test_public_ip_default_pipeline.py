"""Full default-pipeline routing test for the public-ip questions (en-US).

Live validation found "what is my public ip" answered by common_qa with a
docker-bridge address instead of PublicIPIntent. The root cause is pipeline
tier ordering: the default pipeline runs ``padatious-high`` before
``adapt-high``/``adapt-medium``, and before any high-priority fallback
(common_qa). PublicIPIntent's Adapt confidence for this exact phrasing
(``query``+``ip``+``public``, no ``address`` token) is only ~0.6 - below the
Adapt "high" tier cutoff, so it only ever matched at ``adapt-medium``, a
stage that runs AFTER ``ovos-fallback-pipeline-plugin-high`` (where common_qa
lives). A high-priority fallback claiming the utterance first meant
PublicIPIntent's own (correct) answer never got a chance to run.

PublicIPIntent.intent gives the mandated phrasings a padatious-family match that
resolves at ``padatious-high`` - the very first content-matching stage in the
default pipeline, before Adapt or any fallback ever sees the utterance.

The neural padatious tier trains in a background thread on first use; the
warm-up utterance in setUpClass forces training to finish before any
assertion fires (matching the "padaos compiling in background, serving last
compiled state in the meantime" log line seen on a cold container).
"""
import time
from unittest import TestCase

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import get_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"

# The real default `intents.pipeline` order (Configuration()["intents"]["pipeline"]):
# padatious-high and adapt-high both run before any fallback-high skill (e.g.
# common_qa) ever gets a turn.
PIPELINE = [
    "ovos-stop-pipeline-plugin-high",
    "ovos-converse-pipeline-plugin",
    "ovos-padatious-pipeline-plugin-high",
    "ovos-adapt-pipeline-plugin-high",
    "ovos-fallback-pipeline-plugin-high",
    "ovos-stop-pipeline-plugin-medium",
    "ovos-padatious-pipeline-plugin-medium",
    "ovos-adapt-pipeline-plugin-medium",
    "ovos-fallback-pipeline-plugin-medium",
    "ovos-fallback-pipeline-plugin-low",
]


class TestPublicIPDefaultPipeline(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.minicroft = get_minicroft([SKILL_ID])
        cls._warm_up_padatious()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "minicroft", None):
            cls.minicroft.stop()

    @classmethod
    def _warm_up_padatious(cls):
        session = Session("warmup")
        session.lang = LANG
        session.pipeline = list(PIPELINE)
        cls.minicroft.bus.emit(Message(
            "recognizer_loop:utterance",
            {"utterances": ["warmup utterance please ignore"], "lang": LANG},
            {"session": session.serialize()},
        ))
        time.sleep(2)

    def _assert_intent(self, utterance: str, intent_name: str, timeout=15):
        intent_msg_type = f"{SKILL_ID}:{intent_name}"
        matched = []
        handler = lambda msg: matched.append(msg)
        self.minicroft.bus.on(intent_msg_type, handler)
        try:
            session = Session(f"e2e-default-pipeline-{hash(utterance)}")
            session.lang = LANG
            session.pipeline = list(PIPELINE)
            self.minicroft.bus.emit(Message(
                "recognizer_loop:utterance",
                {"utterances": [utterance], "lang": LANG},
                {"session": session.serialize()},
            ))
            deadline = time.monotonic() + timeout
            while not matched and time.monotonic() < deadline:
                time.sleep(0.2)
        finally:
            self.minicroft.bus.remove(intent_msg_type, handler)
        self.assertTrue(
            matched,
            f"{utterance!r} did not route to {intent_name} - a "
            "high-priority fallback (e.g. common_qa) running before "
            "adapt-medium would otherwise have claimed it first",
        )

    def test_whats_my_public_ip_resolves_before_fallback_high(self):
        # PublicIPIntent.intent (padatious-high) must claim this, not the
        # lower-tier Adapt PublicIPIntent (adapt-medium, conf ~0.6).
        self._assert_intent("what is my public ip", "PublicIPIntent")

    def test_whats_my_external_ip_resolves_before_fallback_high(self):
        self._assert_intent("what's my external ip address", "PublicIPIntent")

    def test_bare_local_ip_question_unaffected(self):
        # negative: a plain local-ip question must still route to IPIntent,
        # not PublicIPIntent.intent, on the same pipeline.
        matched = []
        handler = lambda msg: matched.append(msg)
        self.minicroft.bus.on(f"{SKILL_ID}:PublicIPIntent", handler)
        try:
            session = Session("e2e-default-pipeline-local")
            session.lang = LANG
            session.pipeline = list(PIPELINE)
            self.minicroft.bus.emit(Message(
                "recognizer_loop:utterance",
                {"utterances": ["what is my ip"], "lang": LANG},
                {"session": session.serialize()},
            ))
            time.sleep(3)
        finally:
            self.minicroft.bus.remove(f"{SKILL_ID}:PublicIPIntent", handler)
        self.assertFalse(
            matched, "\"what is my ip\" incorrectly routed to PublicIPIntent.intent"
        )
