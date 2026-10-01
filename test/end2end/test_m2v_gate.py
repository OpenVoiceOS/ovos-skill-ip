"""m2v-multilingual candidate-default gate for ovos-skill-ip (en-US).

Boots the skill under the candidate default engine -- the m2v multilingual
classifier (``OpenVoiceOS/ovos-m2v-intents-multi-128M-v5``) via ovoscope's
``get_m2v_minicroft`` -- and replays a slice of this skill's own golden
utterances. Padatious/Adapt stay the skill's deterministic floor (see
``test_golden_utterances_multilang.py`` and
``test_public_ip_default_pipeline.py``); this gate validates the candidate
model that may replace them as the shipping default.

Each row asserts both routing (the classifier picks this skill's registered
intent id) and effect (the rendered ``speak`` text carries a real IP-shaped
answer, not the bare dialog name).

The expected intent id per utterance comes from
``golden_utterances_en-US.jsonl``, the same gold the padatious suite reads,
so a resource rename moves both suites together.

The public-IP query reads a real network endpoint (api.ipify.org). The
outbound call is stubbed to a fixed address, so the effect assertion checks
the skill's own dialog rendering and not network availability.

Run:
    uv run pytest test/end2end/test_m2v_gate.py -v
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_m2v_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"
LANG = "en-US"
STUBBED_PUBLIC_IP = "203.0.113.42"
# How that address must appear in speech: the octets, "dot" between them.
SPOKEN_PUBLIC_IP = " dot ".join(STUBBED_PUBLIC_IP.split("."))

END2END_DIR = Path(__file__).parent

# The utterances this gate drives. ``stubbed_address`` marks the row whose
# answer must carry the stubbed public address rather than any local one.
ROWS = [
    {"utterance": "what is my IP"},
    {"utterance": "what are the last digits of my IP"},
    {"utterance": "what are the final digits of my IP"},
    {"utterance": "what is my public ip", "stubbed_address": True},
]


def _golden_labels(lang):
    """Map utterance -> registered intent name, read from the locale's gold."""
    path = END2END_DIR / f"golden_utterances_{lang}.jsonl"
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[row["utterance"]] = row["intent_label"]
    return labels


GOLDEN_LABELS = _golden_labels(LANG)


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


# func_only keeps the budget on one capture. The module fixture boots a core
# and loads a 128M model, which alone costs more than the per-row budget.
@pytest.mark.timeout(60, func_only=True)
@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["utterance"])
def test_m2v_gate(minicroft, row):
    utterance = row["utterance"]
    assert utterance in GOLDEN_LABELS, (
        f"{utterance!r} is not in golden_utterances_{LANG}.jsonl, so this gate "
        f"has no gold for it"
    )
    expected_intent = f"{SKILL_ID}:{GOLDEN_LABELS[utterance]}"

    stub_response = MagicMock()
    stub_response.text = STUBBED_PUBLIC_IP
    stub_response.raise_for_status = lambda: None
    with patch("ovos_skill_ip.requests.get", return_value=stub_response):
        messages = _capture(minicroft, utterance, f"m2v-{utterance}")

    matched = [m for m in messages if m.msg_type == "ovos.intent.matched"]
    assert matched, (
        f"{utterance!r}: expected ovos.intent.matched, got "
        f"{[m.msg_type for m in messages]!r}"
    )
    names = [m.data.get("intent_name") for m in matched]
    assert expected_intent in names, (
        f"{utterance!r}: expected intent_name {expected_intent!r}, got {names!r}"
    )

    speaks = [m for m in messages if m.msg_type in ("speak", "ovos.utterance.speak")]
    assert speaks, f"{utterance!r}: no speak message captured"
    spoken = speaks[0].data.get("utterance", "")
    assert spoken, f"{utterance!r}: empty spoken text"

    if row.get("stubbed_address"):
        # The stubbed address itself must reach speech, not merely some digits
        # it happens to contain: "4", "2" and "dot" are satisfied by any IPv4
        # that carries those digits.
        assert SPOKEN_PUBLIC_IP in " ".join(spoken.lower().split()), (
            f"{utterance!r}: stubbed public IP {STUBBED_PUBLIC_IP!r} "
            f"did not reach speech as {SPOKEN_PUBLIC_IP!r}: {spoken!r}"
        )
    else:
        # Local-IP intents must speak actual digits, not just the dialog name.
        assert any(ch.isdigit() for ch in spoken), (
            f"{utterance!r}: no digits in spoken IP answer: {spoken!r}"
        )
