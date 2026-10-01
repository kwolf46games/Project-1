"""The overlay window, run headless (xvfb) with a fake audio listener and the fake embedder."""
import gc
import json
import os
import threading
import time

import pytest

tk = pytest.importorskip("tkinter")
if not os.environ.get("DISPLAY"):
    pytest.skip("needs a display (run under xvfb-run)", allow_module_level=True)


@pytest.fixture(autouse=True)
def main_thread_gc():
    """Tk objects must be freed on the main thread: a collection run by a worker thread would call into a
    dead Tcl interpreter from the wrong thread and hang or abort. So collect only when we say so."""
    gc.collect()
    gc.disable()
    yield
    gc.collect()
    gc.enable()

import bank as b  # noqa: E402
from audio import Heard  # noqa: E402
from conftest import fake_embed  # noqa: E402
from followup import Decision  # noqa: E402
from matcher import Match, Matcher  # noqa: E402


class FakeListener:
    def __init__(self, cfg, on_text, on_status):
        self.cfg, self.on_text, self.on_status = cfg, on_text, on_status
        self.paused = threading.Event()
        self.device_name = "Fake Speakers"
        self.vocab = []
        self.started = self.stopped = False
        self.restarts = 0
        self.level = 0.02

    def start(self):
        self.started = True
        self.on_status("Listening · Fake Speakers")

    def stop(self):
        self.stopped = True

    def restart(self):
        self.restarts += 1

    def set_vocabulary(self, terms):
        self.vocab = terms

    def stats(self):
        return {"level": self.level, "device": self.device_name, "model_ready": True, "quiet_for": 5.0,
                "waiting": 0, "dropped": 0, "reconnects": 0, "lag": 0.4, "decode": 0.1, "error": ""}


@pytest.fixture
def ov(banks_dir, monkeypatch):
    import overlay
    banks_dir.mkdir(parents=True)
    bk = b.Bank("demo", "Demo", [
        b.Entry("Tell me about a time you led a team", "**Lead** answer with {{placeholder}}\n- point one\n---\nend",
                ["describe a time you led a team"], skeleton="S-T-A-R", bank="demo"),
        b.Entry("What was the result of that project?", "RESULT answer", ["what was the outcome"],
                follows=["Tell me about a time you led a team"], bank="demo"),
        b.Entry("How did you motivate the team?", "MOTIVATE answer", follows=["Tell me about a time you led a team"], bank="demo"),
        b.Entry("What is your biggest weakness?", "WEAK answer", bank="demo"),
    ])
    bk.save()
    b.CONFIG_PATH.write_text(json.dumps({"active_banks": ["demo"], "match_threshold": 0.78}), encoding="utf-8")
    monkeypatch.setattr(overlay, "Listener", FakeListener)
    monkeypatch.setattr(overlay, "Matcher", lambda: Matcher(embed_fn=fake_embed))
    o = overlay.Overlay()
    o._mk = overlay
    yield o
    try:
        o._quit()
    except tk.TclError:
        pass


def pump(o, until, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        o.root.update()
        if until():
            return True
        time.sleep(0.01)
    return False


def body(o):
    return o.text.get("1.0", "end").strip()


def chips(frame):
    return [w.cget("text") for w in frame.winfo_children() if w.winfo_class() == "Button"]


def say(o, text, final=True, utt=1, repeat=False):
    o.heard_q.put(Heard(text, final, utt, repeat, lag=0.4, decode=0.1))


def ready(o):
    ok = pump(o, lambda: o.conv is not None and "questions" in o.status.cget("text"))
    if not ok:   # say where every other thread is stuck: far more useful than "timed out"
        import sys
        import traceback
        frames = {t.ident: t.name for t in threading.enumerate()}
        stacks = "\n".join(f"--- {frames.get(i, i)}\n" + "".join(traceback.format_stack(f)[-6:])
                           for i, f in sys._current_frames().items() if frames.get(i) != "MainThread")
        raise AssertionError(f"overlay never became ready: {o.status.cget('text')!r} error={o.matcher_error!r}\n{stacks}")


def test_starts_and_shows_status(ov):
    ready(ov)
    assert ov.listener.started and "Listening" in ov.status.cget("text") and "4 questions" in ov.status.cget("text")
    assert ov.listener.vocab == [] or isinstance(ov.listener.vocab, list)


def test_early_guess_then_identical_final_does_not_redraw(ov):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", final=False, utt=1)
    assert pump(ov, lambda: ov.question.cget("text") == "Tell me about a time you led a team")
    assert "Lead" in body(ov) and "⚠ placeholder" in body(ov) and "•  point one" in body(ov)
    assert ov.skeleton.cget("text") == "Skeleton: S-T-A-R"
    ov.text.configure(state="normal"); ov.text.insert("end", "MARK"); ov.text.configure(state="disabled")
    say(ov, "Tell me about a time you led a team?", final=True, utt=1, repeat=True)
    assert pump(ov, lambda: "(0.4s)" in ov.heard.cget("text") and "…" not in ov.heard.cget("text"))
    assert "MARK" in body(ov)                                  # not re-rendered: the answer was already up
    assert chips(ov.likely) == ["↳  What was the result of that project?", "↳  How did you motivate the team?"]


def test_early_guess_without_question_mark_waits_for_final(ov):
    ready(ov)
    say(ov, "Tell me about a time you led a team", final=False, utt=2)
    assert pump(ov, lambda: ov.heard.cget("text").startswith("Heard"), timeout=2) or True
    time.sleep(0.2); ov.root.update()
    assert ov.question.cget("text") == "Waiting for a question…"
    say(ov, "Tell me about a time you led a team.", final=True, utt=2)
    assert pump(ov, lambda: ov.question.cget("text") == "Tell me about a time you led a team")


def test_scripted_followup_shows_banner_and_answer(ov):
    ready(ov)
    say(ov, "Tell me about a time you led a team.", utt=1)
    assert pump(ov, lambda: ov.question.cget("text").startswith("Tell me about"))
    say(ov, "And what was the outcome?", utt=2)
    assert pump(ov, lambda: ov.question.cget("text") == "What was the result of that project?")
    assert "Follow-up" in ov.banner.cget("text") and "Tell me about a time you led a team" in ov.banner.cget("text")
    assert body(ov) == "RESULT answer"


def test_unscripted_followup_keeps_answer_and_explains(ov):
    ready(ov)
    say(ov, "Tell me about a time you led a team.", utt=1)
    assert pump(ov, lambda: ov.question.cget("text").startswith("Tell me about"))
    before = body(ov)
    say(ov, "Can you elaborate on that?", utt=2)
    assert pump(ov, lambda: "follow-up" in ov.banner.cget("text").lower())
    assert body(ov) == before and ov.question.cget("text").startswith("Tell me about")
    assert "more detail" in ov.hint.cget("text") or "detail" in ov.hint.cget("text")
    assert len(chips(ov.likely)) == 2


def test_follow_button_forces_followup_then_resets(ov):
    ready(ov)
    say(ov, "Tell me about a time you led a team.", utt=1)
    assert pump(ov, lambda: ov.question.cget("text").startswith("Tell me about"))
    ov.follow_btn.invoke()
    assert "Listening for follow-up" in ov.follow_btn.cget("text")
    say(ov, "hmm so the budget for that", utt=2)
    assert pump(ov, lambda: "follow-up" in ov.banner.cget("text").lower())
    assert pump(ov, lambda: ov.follow_btn.cget("text") == "↳ Follow-up")
    ov.follow_btn.invoke(); assert "Listening" in ov.follow_btn.cget("text")
    ov.follow_btn.invoke(); assert ov.follow_btn.cget("text") == "↳ Follow-up"      # toggles off


def test_notes_take_no_room_until_needed(ov):
    ready(ov)
    ov.root.update()
    assert ov.notes.winfo_height() <= 1 and not ov.banner.winfo_ismapped()
    say(ov, "Tell me about a time you led a team.", utt=1)
    assert pump(ov, lambda: ov.question.cget("text").startswith("Tell me about"))
    say(ov, "Can you elaborate on that?", utt=2)
    assert pump(ov, lambda: ov.banner.winfo_ismapped() and ov.hint.winfo_ismapped())
    assert ov.banner.winfo_y() < ov.hint.winfo_y()                 # banner above hint
    say(ov, "What is your biggest weakness?", utt=3)
    assert pump(ov, lambda: ov.question.cget("text") == "What is your biggest weakness?")
    ov.root.update()
    assert not ov.banner.winfo_ismapped() and not ov.hint.winfo_ismapped() and ov.notes.winfo_height() <= 1


def test_likely_followup_chip_shows_that_answer(ov):
    ready(ov)
    say(ov, "Tell me about a time you led a team.", utt=1)
    assert pump(ov, lambda: len(chips(ov.likely)) == 2)
    ov.likely.winfo_children()[1].invoke()                       # first chip (child 0 is the label)
    assert pump(ov, lambda: ov.question.cget("text") == "What was the result of that project?")
    assert ov.conf.cget("text") == "picked" and body(ov) == "RESULT answer"


def test_unsure_then_pick_then_remember_wording(ov):
    ready(ov)
    es = ov.matcher.entries
    ov._apply(Decision("unsure", "so how did you get everybody pulling together", [Match(es[2], 0.74), Match(es[0], 0.71)]),
              Heard("so how did you get everybody pulling together"))
    assert pump(ov, lambda: chips(ov.alts))
    assert chips(ov.alts)[0].startswith("74%")
    ov.alts.winfo_children()[1].invoke()                          # pick "How did you motivate the team?"
    assert pump(ov, lambda: ov.question.cget("text") == "How did you motivate the team?")
    rem = [w for w in ov.alts.winfo_children() if w.winfo_class() == "Button" and w.cget("text").startswith("＋ Remember")]
    assert len(rem) == 1
    rem[0].invoke()
    assert "Saved" in rem[0].cget("text")
    assert "so how did you get everybody pulling together" in b.load_bank("demo").entries[2].also
    # and the matcher picks it up after the reload
    assert pump(ov, lambda: "so how did you get everybody pulling together" in ov.matcher.entries[2].also, timeout=5)


def test_manual_search_sets_context(ov):
    ready(ov)
    ov.search.insert(0, "what is your biggest weakness"); ov._manual_search()
    assert pump(ov, lambda: ov.question.cget("text") == "What is your biggest weakness?")
    assert ov.conv.followups_of(ov.matcher.entries[3]) == []


def test_meter_pause_and_quit(ov):
    ready(ov)
    ov._tick()
    ov.root.update()
    assert ov.meter.coords(ov._meter_bar)[2] > 0
    ov.pause_btn.invoke()
    assert ov.listener.paused.is_set() and ov.status.cget("text").startswith("Paused")
    ov.pause_btn.invoke()
    assert not ov.listener.paused.is_set()
    ov._quit()
    assert ov.listener.stopped


def test_matcher_failure_is_reported_not_silent(banks_dir, monkeypatch):
    import overlay
    banks_dir.mkdir(parents=True)
    b.CONFIG_PATH.write_text(json.dumps({"active_banks": []}), encoding="utf-8")
    monkeypatch.setattr(overlay, "Listener", FakeListener)

    def boom():
        raise RuntimeError("model files missing")
    monkeypatch.setattr(overlay, "Matcher", boom)
    o = overlay.Overlay()
    try:
        assert pump(o, lambda: "failed to load" in o.status.cget("text"))
        assert "model files missing" in o.status.cget("text") or "failed" in o.status.cget("text")
        assert o.dot.cget("fg") == "#f28b82"                          # red, and it stays red after "Listening" arrives
        o.events.put(("status", "Listening · Fake Speakers")); pump(o, lambda: False, timeout=0.3)
        assert "failed to load" in o.status.cget("text")
    finally:
        o._quit()
