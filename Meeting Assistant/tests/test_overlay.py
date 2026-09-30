"""Drive the overlay's new follow-up paths with tkinter replaced by mock widgets (no display needed).

This proves the wiring (what gets shown, what becomes the anchor); it can't show how the window looks.
"""
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import bank as b
from conftest import fake_embed
from matcher import Match, Matcher as RealMatcher

LED = "Tell me about a time you led a team."


class StubListener:
    def __init__(self, cfg, on_text, on_status, on_transcript=None):
        self.on_text, self.on_status, self.on_transcript = on_text, on_status, on_transcript
        self.paused, self.device_name = threading.Event(), "Speakers"

    def start(self): pass
    def stop(self): pass


@pytest.fixture
def ov(monkeypatch, entries):
    tk = MagicMock()
    for widget in ("Tk", "Label", "Text", "Frame", "Button", "Entry", "Menubutton", "Menu", "Scrollbar"):
        getattr(tk, widget).side_effect = lambda *a, **k: MagicMock()   # every widget is its own mock
    root = tk.Tk.return_value
    root.winfo_fpixels.return_value, root.winfo_screenwidth.return_value, root.winfo_width.return_value = 96.0, 1920, 560
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.font", tk.font)
    sys.modules.pop("overlay", None)
    import overlay
    monkeypatch.setattr(overlay, "Listener", StubListener)
    monkeypatch.setattr(overlay, "Matcher", lambda: RealMatcher(embed=fake_embed))
    monkeypatch.setattr(overlay.bankmod, "load_config", lambda: {**b.DEFAULT_CONFIG, "active_banks": ["test"]})
    monkeypatch.setattr(overlay.bankmod, "active_entries", lambda cfg=None: entries)
    monkeypatch.setattr(overlay.Overlay, "_reload_banks", lambda self: None)   # don't start a loader thread
    o = overlay.Overlay()
    o._load_matcher()
    o.tk = tk
    yield o
    sys.modules.pop("overlay", None)


def shown(o):
    return o.question.configure.call_args.kwargs["text"]


def body(o):
    return "".join(c.args[1] for c in o.text.insert.call_args_list if len(c.args) > 1)


def buttons(o):
    return [c.kwargs["text"] for c in o.tk.Button.call_args_list if "text" in c.kwargs]


def test_a_new_question_is_shown_as_before(ov, entries):
    ov._on_heard(LED)
    assert shown(ov) == LED and ov.session.live_anchor().question == LED
    assert "At NovaCore I led a team" in body(ov)


def test_a_prepared_followup_shows_your_scripted_reply(ov):
    ov._on_heard(LED)
    ov._on_heard("How big was the team?")
    assert shown(ov) == "How big was the team?"
    assert ov.skeleton.configure.call_args.kwargs["text"] == f"↳ Follow-up to: {LED}"
    assert "Six people" in body(ov).split("At NovaCore")[-1] or body(ov).rstrip().endswith("and me.")
    assert ov.conf.configure.call_args.kwargs["text"].startswith("↳")


def test_a_generic_followup_shows_the_relevant_part_of_the_answer_and_buttons(ov):
    ov._on_heard(LED)
    ov.tk.Button.reset_mock()
    ov._on_heard("What was the result?")
    assert shown(ov) == "Follow-up: asking about the result"
    text = body(ov)
    assert "From your answer:" in text and ("uptime rose" in text or "two weeks early" in text)
    assert "At NovaCore I led a team" in text                       # the whole answer is still there to scroll to
    labels = buttons(ov)
    assert any(x.startswith("↩ Back to: Tell me about a time you led a team") for x in labels)
    assert any("How big was the team?" in x for x in labels)


def test_followup_buttons_work(ov):
    ov._on_heard(LED)
    ov.tk.Button.reset_mock()
    ov._on_heard("What was the result?")
    cmds = {c.kwargs["text"]: c.kwargs["command"] for c in ov.tk.Button.call_args_list if "command" in c.kwargs}
    next(cmd for label, cmd in cmds.items() if "How big was the team?" in label)()
    assert shown(ov) == "How big was the team?" and "Six people" in body(ov)
    next(cmd for label, cmd in cmds.items() if label.startswith("↩ Back to"))()
    assert shown(ov) == LED


def test_picking_an_answer_by_hand_makes_it_the_anchor(ov, entries):
    ov._show([Match(entries[1], 0.9)])                              # e.g. clicked "Also:" or used search
    assert ov.session.live_anchor().question == "Why should we hire you?"
    ov._on_heard("Tell me more about that.")
    assert ov.skeleton.configure.call_args.kwargs["text"] == "↳ Follow-up to: Why should we hire you?"


def test_old_behaviour_is_unchanged_for_non_questions_and_fragments(ov):
    ov._on_heard(LED)
    before = shown(ov)
    ov._on_heard("We are going to record this meeting today.")      # a statement: nothing new on screen
    assert shown(ov) == before
    ov._on_heard("Can you tell me about the")                       # an unfinished question waits to be glued on
    assert ov.pending is not None and "Can you tell me about the" in ov.pending[0]


def test_followups_off_restores_plain_matching(ov):
    ov.cfg["followups"] = False
    ov._on_heard(LED)
    ov._on_heard("What was the result?")
    assert "Follow-up" not in str(ov.skeleton.configure.call_args_list[-1])


def test_speech_confidence_travels_with_the_text(ov):
    while not ov.events.empty():
        ov.events.get_nowait()
    ov.listener.on_transcript(SimpleNamespace(confidence=0.42))     # what the Listener calls before on_text
    ov.listener.on_text("hello")
    kinds = [ov.events.get_nowait() for _ in range(2)]
    assert kinds[0] == ("confidence", 0.42) and kinds[1] == ("text", "hello")
    ov.events.put(("confidence", 0.42))
    ov._pump()
    assert ov._confidence == 0.42
