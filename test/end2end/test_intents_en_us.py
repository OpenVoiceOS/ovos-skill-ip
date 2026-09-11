"""End-to-end intent-routing tests for ovos-skill-ip (en-US).

Each case feeds an utterance through a MiniCroft stack running the default
pipeline (Adapt plus the Padatious-family plugins) and asserts it routes to
the expected handler AND that the handler speaks a dialog line from that
intent's own ``.dialog`` file. The expected lines are read directly from
the locale files on disk, independent of the skill handler under test,
so a handler that speaks the wrong dialog (or the right dialog for a
different intent) fails here even though it still emits a ``speak`` message
and still matches the correct intent name.

The ``what.ssid.intent`` and ``wifi_signal.intent`` handlers only register
when ``iwlist`` is present on the host; their routing coverage is skipped
when it is absent, which is the case in CI.

Run: pytest test/end2end/ -v
"""
import re
import time
from pathlib import Path
from shutil import which
from unittest import TestCase, skipUnless

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"
LOCALE_EN_US = Path(__file__).parent.parent.parent / "locale" / "en-US"

PIPELINE = [
    "ovos-adapt-pipeline-plugin-high",
    "ovos-adapt-pipeline-plugin-medium",
    "ovos-adapt-pipeline-plugin-low",
    "ovos-padacioso-pipeline-plugin",
]


def _dialog_lines(name: str) -> set:
    """Read dialog template lines from a shipped .dialog file at test time."""
    path = LOCALE_EN_US / f"{name}.dialog"
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _template_to_pattern(template: str) -> str:
    """Convert a dialog template to a regex pattern.
    
    Handles:
    - Placeholders like {var} → match any non-newline
    - Optional parts like [word] → word is optional
    """
    # Escape special regex chars except for the brackets we'll use
    escaped = re.escape(template)
    
    # Handle optional parts: \[text\] → (?:text)?
    escaped = re.sub(r"\\\[(.*?)\\\]", r"(?:\1)?", escaped)
    
    # Replace escaped placeholders with regex pattern for any non-newline
    escaped = re.sub(r"\\\{[^}]+\\\}", r"[^\\n]+", escaped)
    
    return escaped


def _spoken_matches_template(spoken: str, template: str) -> bool:
    """Check if a rendered spoken line matches a dialog template pattern.
    
    Converts template placeholders and optional parts to regex patterns that
    match rendered substitutions.
    """
    pattern = _template_to_pattern(template)
    return bool(re.fullmatch(pattern, spoken))


def _matches_any_template(spoken: str, templates: set) -> bool:
    """Check if a spoken line matches any of the given dialog templates."""
    return any(_spoken_matches_template(spoken, t) for t in templates)


# Read once, directly from the shipped dialog files -- never from a captured
# bus message -- so these sets are independent of the code under test.
MY_ADDRESS_IS_LINES = _dialog_lines("my_address_is")
MY_ADDRESS_ON_X_IS_Y_LINES = _dialog_lines("my_address_on_x_is_y")
MY_PUBLIC_IP_LINES = _dialog_lines("my.public.ip")
LAST_DIGITS_LINES = _dialog_lines("last_digits")
NO_NETWORK_LINES = _dialog_lines("no_network_connection")
PUBLIC_IP_ERROR_LINES = _dialog_lines("public.ip.error")

# Map intent name to the dialog line sets it may speak
_DIALOG_LINES = {
    "IPIntent": MY_ADDRESS_IS_LINES | MY_ADDRESS_ON_X_IS_Y_LINES | NO_NETWORK_LINES,
    "PublicIPIntent": MY_PUBLIC_IP_LINES | PUBLIC_IP_ERROR_LINES,
    "LastIPDigitsIntent": LAST_DIGITS_LINES | NO_NETWORK_LINES,
}


class _IntentRoutingMixin:
    """Shared MiniCroft setup for intent routing with effect assertions."""

    @classmethod
    def setUpClass(cls):
        cls.minicroft = get_minicroft([SKILL_ID])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "minicroft", None):
            cls.minicroft.stop()

    def _run(self, text: str):
        """Emit an utterance and capture all messages until completion."""
        session = Session("test-session")
        session.lang = LANG
        session.pipeline = PIPELINE
        utterance = Message(
            "recognizer_loop:utterance",
            {"utterances": [text], "lang": LANG},
            {"session": session.serialize(), "source": "A", "destination": "B"},
        )
        capture = CaptureSession(self.minicroft)
        capture.capture(utterance, timeout=30)
        return capture.finish()

    def _assert_intent(self, utterance: str, intent_name: str):
        """Assert an utterance routes to an intent and speaks expected dialog."""
        messages = self._run(utterance)
        types = [m.msg_type for m in messages]
        intent_msg_type = f"{SKILL_ID}:{intent_name}"
        self.assertIn(
            intent_msg_type,
            types,
            f"{utterance!r} did not route to {intent_name}",
        )

        # Extract spoken utterances from the captured messages
        spoken = [
            m.data.get("utterance", "")
            for m in messages
            if m.msg_type in ("speak", "ovos.utterance.speak")
        ]
        self.assertTrue(
            spoken,
            f"expected a spoken response for {utterance!r}, got types {types!r}",
        )

        # Assert at least one spoken line matches a template in the expected dialog set
        expected_lines = _DIALOG_LINES[intent_name]
        self.assertTrue(
            any(_matches_any_template(utt, expected_lines) for utt in spoken),
            f"expected one of {intent_name}.dialog's own lines to be spoken for "
            f"{utterance!r}, got {spoken!r}",
        )

    def _assert_not_intent(self, utterance: str, intent_name: str):
        """Assert an utterance does not route to an intent."""
        messages = self._run(utterance)
        types = [m.msg_type for m in messages]
        intent_msg_type = f"{SKILL_ID}:{intent_name}"
        self.assertNotIn(
            intent_msg_type,
            types,
            f"{utterance!r} incorrectly routed to {intent_name}",
        )


class TestDialogDisjointness(TestCase):
    """Dialog sets for each intent must be pairwise disjoint."""

    def test_dialog_sets_are_disjoint(self):
        """No dialog line should appear in multiple intent sets."""
        intent_sets = {
            "my_address_is": MY_ADDRESS_IS_LINES,
            "my_address_on_x_is_y": MY_ADDRESS_ON_X_IS_Y_LINES,
            "my.public.ip": MY_PUBLIC_IP_LINES,
            "last_digits": LAST_DIGITS_LINES,
            "no_network_connection": NO_NETWORK_LINES,
            "public.ip.error": PUBLIC_IP_ERROR_LINES,
        }

        names = sorted(intent_sets.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                overlap = intent_sets[name1] & intent_sets[name2]
                self.assertFalse(
                    overlap,
                    f"{name1} and {name2} share lines: {overlap}",
                )


class TestIPIntent(_IntentRoutingMixin, TestCase):
    """IPIntent — report the full IP address."""

    def test_what_is_my_ip(self):
        self._assert_intent("what is my ip", "IPIntent")

    def test_whats_my_ip(self):
        self._assert_intent("what's my ip", "IPIntent")

    def test_tell_me_my_ip_address(self):
        self._assert_intent("tell me my ip address", "IPIntent")

    def test_give_me_my_ip_address(self):
        self._assert_intent("give me my ip address", "IPIntent")

    def test_show_me_my_network_address(self):
        self._assert_intent("show me my network address", "IPIntent")

    def test_how_do_i_find_my_ip(self):
        self._assert_intent("how do i find my ip", "IPIntent")

    def test_whats_my_local_ip_address(self):
        self._assert_intent("what's my local ip address", "IPIntent")

    def test_public_ip_question_does_not_match_local_intent(self):
        self._assert_not_intent("what is my public ip", "IPIntent")

    def test_external_ip_question_does_not_match_local_intent(self):
        self._assert_not_intent("what is my external ip address", "IPIntent")


class TestPublicIPIntent(_IntentRoutingMixin, TestCase):
    """PublicIPIntent — report the public IP address."""

    def test_whats_my_public_ip(self):
        self._assert_intent("what's my public ip", "PublicIPIntent")

    def test_whats_my_external_ip_address(self):
        self._assert_intent("what is my external ip address", "PublicIPIntent")

    def test_tell_me_my_outside_ip(self):
        self._assert_intent("tell me my outside ip", "PublicIPIntent")


@skipUnless(which("iwlist"), "what.ssid.intent only registers when iwlist is present")
class TestWifiPhrasings(_IntentRoutingMixin, TestCase):
    """what.ssid.intent — additional phrasings for the wifi/network name."""

    def test_whats_my_wifi_called(self):
        self._assert_intent("what is my wifi called", "what.ssid.intent")

    def test_whats_my_wifi_called_again(self):
        self._assert_intent("what is my wifi called again", "what.ssid.intent")

    def test_can_you_please_tell_me_the_wifi_name(self):
        self._assert_intent("can you please tell me the wifi name",
                             "what.ssid.intent")

    def test_what_network_am_i_on(self):
        self._assert_intent("what network am I on", "what.ssid.intent")


@skipUnless(which("iwlist"), "wifi_signal.intent only registers when iwlist is present")
class TestWifiSignalIntent(_IntentRoutingMixin, TestCase):
    """wifi_signal.intent — report the wifi signal strength."""

    def test_whats_my_wifi_signal_strength(self):
        self._assert_intent("what's my wifi signal strength", "wifi_signal.intent")

    def test_how_good_is_my_wifi_connection(self):
        self._assert_intent("how good is my wifi connection", "wifi_signal.intent")


class TestLastIPDigitsIntent(_IntentRoutingMixin, TestCase):
    """LastIPDigitsIntent — report only the trailing part of the address."""

    def test_last_digits_of_my_ip(self):
        self._assert_intent("what are the last digits of my ip", "LastIPDigitsIntent")

    def test_read_the_last_part_of_my_ip(self):
        self._assert_intent("read the last part of my ip", "LastIPDigitsIntent")

    def test_read_me_the_last_part_of_my_ip(self):
        self._assert_intent("read me the last part of my ip", "LastIPDigitsIntent")

    def test_final_digits_of_my_ip_address(self):
        self._assert_intent("tell me the final digits of my ip address", "LastIPDigitsIntent")
