"""m2v-multilingual candidate-default gate for ovos-skill-ip (en-US).

Boots the skill under the candidate default engine -- the m2v multilingual
classifier (``OpenVoiceOS/ovos-m2v-intents-multi-128M-v5``) via ovoscope's
``get_m2v_minicroft`` -- and replays a representative slice of this skill's
own golden utterances (``golden_utterances.jsonl``). Padatious/Adapt stay
the skill's deterministic floor (see ``test_golden_utterances.py`` and
``test_public_ip_default_pipeline.py``); this gate validates the candidate
model that may replace them as the shipping default.

``PublicIPIntent`` hits a real network endpoint (api.ipify.org); the
outbound call is stubbed to a fixed address so the effect assertion checks
the skill's own dialog rendering, not network availability.

Each row asserts both routing (the classifier picks this skill's registered
intent id) and effect (the rendered ``speak`` text carries a real IP-shaped
answer, not the bare dialog name).

Run:
    uv run pytest test/end2end/test_m2v_gate.py -v
"""
from unittest.mock import MagicMock, patch

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_m2v_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"
STUBBED_PUBLIC_IP = "203.0.113.42"

# One representative utterance per intent, pulled from the skill's own golden set.
ROWS = [
    {"utterance": "what is my IP", "intent_label": "IPIntent"},
    {"utterance": "what are the last digits of my IP", "intent_label": "LastIPDigitsIntent"},
    {"utterance": "what are the final digits of my IP", "intent_label": "LastIPDigitsIntent"},
    {"utterance": "what is my public ip", "intent_label": "PublicIPIntent"},
]


@pytest.fixture(scope="module")
def minicroft():
    mc = get_m2v_minicroft(skill_ids=[SKILL_ID], lang=LANG)
    pipe = mc.intents.pipeline_plugins["ovos-m2v-pipeline"]
    pipe._ensure_model(background_ok=False)
    yield mc
    mc.stop()


def _capture(mc, text, session_id):
    session = Session(session_id)
    session.lang = LANG
    utterance = Message(
        "recognizer_loop:utterance",
        {"utterances": [text], "lang": LANG},
        {"session": session.serialize(), "source": "A", "destination": "B"},
    )
    capture = CaptureSession(mc)
    capture.capture(utterance, timeout=30)
    return capture.finish()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["utterance"])
def test_m2v_gate(minicroft, row):
    expected_intent = f"{SKILL_ID}:{row['intent_label']}"

    stub_response = MagicMock()
    stub_response.text = STUBBED_PUBLIC_IP
    stub_response.raise_for_status = lambda: None
    with patch("ovos_skill_ip.requests.get", return_value=stub_response):
        messages = _capture(minicroft, row["utterance"], f"m2v-{row['utterance']}")

    matched = [m for m in messages if m.msg_type == "ovos.intent.matched"]
    assert matched, (
        f"{row['utterance']!r}: expected ovos.intent.matched, got "
        f"{[m.msg_type for m in messages]!r}"
    )
    names = [m.data.get("intent_name") for m in matched]
    assert expected_intent in names, (
        f"{row['utterance']!r}: expected intent_name {expected_intent!r}, got {names!r}"
    )

    speaks = [m for m in messages if m.msg_type in ("speak", "ovos.utterance.speak")]
    assert speaks, f"{row['utterance']!r}: no speak message captured"
    spoken = speaks[0].data.get("utterance", "")
    assert spoken, f"{row['utterance']!r}: empty spoken text"

    if row["intent_label"] == "PublicIPIntent":
        # The stubbed public IP's digits, spoken with "dot" separators, must
        # actually reach speech -- not a generic error dialog.
        assert "4" in spoken and "2" in spoken and "dot" in spoken.lower(), (
            f"{row['utterance']!r}: stubbed public IP {STUBBED_PUBLIC_IP!r} "
            f"did not reach speech: {spoken!r}"
        )
    else:
        # Local-IP intents must speak actual digits, not just the dialog name.
        assert any(ch.isdigit() for ch in spoken), (
            f"{row['utterance']!r}: no digits in spoken IP answer: {spoken!r}"
        )
