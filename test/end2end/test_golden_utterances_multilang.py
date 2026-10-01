"""Multilingual golden-utterance end-to-end coverage for ovos-skill-ip.

Every locale under ``locale/`` gets its own
``golden_utterances_<lang>.jsonl``, and every file present runs, including
rows marked ``needs_manual``. Each row's utterance is a direct mechanical
expansion of one of that locale's own ``.intent`` padatious templates:
``(a|b|c)`` word-choice groups are resolved to one alternative,
``[optional]`` tokens (including an internal ``a|b`` choice) are kept
or dropped. No translation, no drafted prose.

The skill registers ``what_ssid`` and ``wifi_signal`` only when ``iwlist``
is on ``PATH``, so each locale class puts a stub ``iwlist`` first on
``PATH`` while its MiniCroft runs.

One ``MiniCroft`` is booted per locale (class-scoped, torn down after),
mirroring the other skills' multilang suites in this batch and
ovos-skill-date-time/test/end2end/test_intents_it_it.py on dev.

Run:
    uv run pytest test/end2end/test_golden_utterances_multilang.py -v
"""
import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovoscope import CaptureSession, get_minicroft

SKILL_ID = "ovos-skill-ip.openvoiceos"

PIPELINE = [
    "ovos-adapt-pipeline-plugin-high",
    "ovos-padatious-pipeline-plugin-high",
    "ovos-padacioso-pipeline-plugin-high",
    "ovos-adapt-pipeline-plugin-medium",
    "ovos-padacioso-pipeline-plugin-medium",
    "ovos-adapt-pipeline-plugin-low",
]

END2END_DIR = Path(__file__).parent

LANGS = sorted(
    p.stem.removeprefix("golden_utterances_")
    for p in END2END_DIR.glob("golden_utterances_*.jsonl")
)

NEGATIVE_UTTERANCES = [
    ("what's the weather like today", "en-US", "ovos-skill-weather.openvoiceos"),
    ("tell me a joke", "en-US", "ovos-skill-icanhazdadjokes.openvoiceos"),
    ("set the volume to 50 percent", "en-US", "ovos-skill-volume.openvoiceos"),
    ("what is your cpu usage", "en-US", None),
    ("tell me your kernel version", "en-US", None),
    ("what is my current location", "en-US", None),
    ("where is the international space station", "en-US", None),
]


def _load_rows(lang):
    path = END2END_DIR / f"golden_utterances_{lang}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _matched_names(mc, text, lang, session_id):
    session = Session(session_id)
    session.lang = lang
    session.pipeline = list(PIPELINE)
    utterance = Message(
        "recognizer_loop:utterance",
        {"utterances": [text], "lang": lang},
        {"session": session.serialize(), "source": "A", "destination": "B"},
    )
    capture = CaptureSession(mc, eof_msgs=["mycroft.skill.handler.start"])
    capture.capture(utterance, timeout=30)
    msgs = capture.finish()
    return [m.data.get("intent_name") for m in msgs if m.msg_type == "ovos.intent.matched"]


KNOWN_BUGS = {}


def _make_locale_test_case(lang):
    rows = _load_rows(lang)
    negatives = [n for n in NEGATIVE_UTTERANCES if n[1] == lang]

    class _LocaleGoldenCase(TestCase):
        LANG = lang

        @classmethod
        def setUpClass(cls):
            cls._stub_dir = tempfile.TemporaryDirectory()
            stub = Path(cls._stub_dir.name) / "iwlist"
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(0o755)
            cls._old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{cls._stub_dir.name}{os.pathsep}{cls._old_path}"
            cls.minicroft = get_minicroft([SKILL_ID], max_wait=180, lang=lang)

        @classmethod
        def tearDownClass(cls):
            if getattr(cls, "minicroft", None):
                cls.minicroft.stop()
            os.environ["PATH"] = cls._old_path
            cls._stub_dir.cleanup()

        def _check_row(self, row):
            expected = f"{SKILL_ID}:{row['intent_label']}"
            names = _matched_names(
                self.minicroft, row["utterance"], row["lang"],
                f"golden-{row['lang']}-{row['intent_label']}-{row['utterance']}",
            )
            matched = expected in names
            bug_key = (row["lang"], row["utterance"])
            if bug_key in KNOWN_BUGS and not matched:
                self.skipTest(f"known-bug: {KNOWN_BUGS[bug_key]}")
            self.assertTrue(
                matched,
                f"[{row['lang']}] {row['utterance']!r}: expected {expected!r}, "
                f"got {names!r}",
            )

        def _check_negative(self, text, source_skill):
            names = _matched_names(self.minicroft, text, lang, f"negative-{lang}-{text}")
            claimed = any((n or "").startswith(f"{SKILL_ID}:") for n in names)
            self.assertFalse(
                claimed, f"[{lang}] {text!r} was incorrectly claimed by {SKILL_ID}"
            )

    for i, row in enumerate(rows):
        def _test(self, row=row):
            self._check_row(row)
        _test.__name__ = f"test_golden_{i:03d}_{row['intent_label'].replace('.', '_')}"
        setattr(_LocaleGoldenCase, _test.__name__, _test)

    for i, (text, _lang, source_skill) in enumerate(negatives):
        def _neg_test(self, text=text, source_skill=source_skill):
            self._check_negative(text, source_skill)
        _neg_test.__name__ = f"test_negative_{i:03d}"
        setattr(_LocaleGoldenCase, _neg_test.__name__, _neg_test)

    _LocaleGoldenCase.__name__ = f"TestGolden_{lang.replace('-', '_')}"
    _LocaleGoldenCase.__qualname__ = _LocaleGoldenCase.__name__
    return _LocaleGoldenCase


for _lang in LANGS:
    _cls = _make_locale_test_case(_lang)
    globals()[_cls.__name__] = _cls
del _lang, _cls  # for-loop variables leak into module globals; without this
# deletion pytest also collects a spurious extra test class literally named
# "_cls" (bound to whichever locale ran last), which boots a second,
# redundant MiniCroft for that locale under a different collected name.


def test_every_shipping_locale_has_a_golden_file():
    golden = {p.stem.split("_", 2)[2] for p in END2END_DIR.glob("golden_utterances_*.jsonl")}
    locale_root = Path(__file__).parents[2] / "locale"
    shipping = {d.name for d in locale_root.iterdir()
                if d.is_dir() and any(d.rglob("*.intent"))}
    assert golden == shipping, f"golden files {sorted(golden ^ shipping)} differ from shipping locales"
