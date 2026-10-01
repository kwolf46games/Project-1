"""Drafted answers in the overlay: question beside the answer, streaming, gap filling, failures. Headless."""
import gc
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter")
if not os.environ.get("DISPLAY"):
    pytest.skip("needs a display (run under xvfb-run)", allow_module_level=True)

import bank as b  # noqa: E402
import generate as g  # noqa: E402
from conftest import fake_embed  # noqa: E402
from matcher import Matcher  # noqa: E402
from test_generate import FakeClient, _api_error  # noqa: E402
from test_overlay_ui import FakeListener, body, pump, ready, say  # noqa: E402

DRAFT = ["**I lead by listening first.** ", "On my last team I {{team size}} people ", "and kept everyone aligned."]
DRAFT_TEXT = "".join(DRAFT)


@pytest.fixture(autouse=True)
def main_thread_gc():
    gc.collect()
    gc.disable()
    yield
    gc.collect()
    gc.enable()


@pytest.fixture
def holder():
    return SimpleNamespace(client=FakeClient(pieces=DRAFT))


@pytest.fixture
def ov(banks_dir, monkeypatch, holder):
    import overlay
    banks_dir.mkdir(parents=True)
    b.Bank("demo", "Demo", [
        b.Entry("Tell me about a time you led a team", "**Lead** answer: I led {{N}} people.",
                ["describe a time you led a team"], bank="demo"),
        b.Entry("What was the result of that project?", "RESULT answer", ["what was the outcome"],
                follows=["Tell me about a time you led a team"], bank="demo"),
        b.Entry("What is your biggest weakness?", "WEAK answer", bank="demo"),
    ]).save()
    b.CONFIG_PATH.write_text(json.dumps({"active_banks": ["demo"], "match_threshold": 0.78, "generate": True}), encoding="utf-8")
    monkeypatch.setattr(overlay, "Listener", FakeListener)
    monkeypatch.setattr(overlay, "Matcher", lambda: Matcher(embed_fn=fake_embed))
    # the overlay builds its client once at start-up, so hand it a proxy that always forwards to the CURRENT fake
    proxy = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: holder.client.stream(**kw)))
    monkeypatch.setattr(overlay, "Generator", lambda cfg, on_event: g.Generator(cfg, on_event, client_factory=lambda: proxy))
    o = overlay.Overlay()
    yield o
    try:
        o._quit()
    except tk.TclError:
        pass


def drafted(o):
    return o.gen_state == "done"


def last_prompt(holder):
    return holder.client.calls[-1]["messages"][0]["content"]


def test_question_sits_beside_the_answer(ov):
    ready(ov)
    ov.root.update()
    assert ov.heard.winfo_rootx() + ov.heard.winfo_width() <= ov.text.winfo_rootx() + 2      # left of it, not above it
    top, bottom = ov.text.winfo_rooty(), ov.text.winfo_rooty() + ov.text.winfo_height()
    assert top - 40 <= ov.heard.winfo_rooty() <= bottom                                        # in the same band
    assert ov.right.winfo_rootx() > ov.left.winfo_rootx()


def test_prepared_answer_shows_at_once_then_the_draft_streams_in(ov, holder):
    holder.client = FakeClient(pieces=DRAFT, delay=0.15)
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: "Lead" in body(ov) and ov.gen_state == "drafting")                 # instant: the prepared one
    assert ov.view == "prepared" or ov.gen_text == ""
    assert pump(ov, lambda: ov.view == "generated" and ov.gen_text)                             # first words arrive
    assert pump(ov, lambda: drafted(ov))
    assert body(ov) == "I lead by listening first. On my last team I ⚠ team size people and kept everyone aligned."
    assert ov.gen_status.cget("text").startswith("✨ adapted from your prepared answer")
    assert "Tell me about a time you led a team" in ov.heard.cget("text")                      # the question, left of it
    assert ov.question.cget("text") == "Tell me about a time you led a team"                   # closest prepared question
    assert str(ov.tab_gen.cget("state")) == "normal" and ov.save_btn.cget("state") == "normal"


def test_prompt_carries_the_question_and_the_prepared_answer(ov, holder):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    p = last_prompt(holder)
    assert "<question>\nTell me about a time you led a team?\n</question>" in p
    assert "very close match" in p and "I led {{N}} people." in p
    assert holder.client.calls[0]["model"] == "claude-opus-5-5"


def test_tabs_flip_between_generated_and_prepared(ov):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    ov.tab_prep.invoke()
    assert "Lead answer: I led" in body(ov) and ov.view == "prepared"
    ov.tab_gen.invoke()
    assert body(ov).startswith("I lead by listening first") and ov.view == "generated"


def test_no_prepared_answer_fits_so_the_gap_is_filled(ov, holder):
    ready(ov)
    say(ov, "What is your experience with Kubernetes deployments and monitoring?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    assert ov.question.cget("text") == "No close prepared answer"
    assert ov.view == "generated" and body(ov).startswith("I lead by listening first")
    p = last_prompt(holder)
    assert "Kubernetes deployments" in p and "{{placeholder}}" not in p
    assert "None of the prepared answers" in p or "loosely related" in p or "related match" in p
    assert "no prepared answer matched" in ov.gen_status.cget("text") or "blended" in ov.gen_status.cget("text")


def test_statements_are_not_drafted(ov, holder):
    ready(ov)
    say(ov, "Thanks everyone for joining us today, we have a busy agenda.", utt=1)
    assert pump(ov, lambda: ov.heard.cget("text").startswith("Heard: “Thanks"))
    time.sleep(0.3)
    assert holder.client.calls == [] and ov.gen_state == "idle"


def test_early_guess_confirmed_by_the_final_costs_one_call(ov, holder):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", final=False, utt=1)
    assert pump(ov, lambda: drafted(ov))
    say(ov, "Tell me about a time you led a team?", final=True, utt=1, repeat=True)
    assert pump(ov, lambda: "(0.4s)" in ov.heard.cget("text"))
    time.sleep(0.3)
    assert len(holder.client.calls) == 1 and ov.gen_state == "done"


def test_a_different_final_supersedes_the_early_draft(ov, holder):
    holder.client = FakeClient(pieces=["Early ", "words "] * 20, delay=0.05)
    ready(ov)
    say(ov, "Tell me about a time you led a team?", final=False, utt=1)
    assert pump(ov, lambda: ov.gen_text.startswith("Early"))
    holder.client = FakeClient(pieces=["Weakness ", "answer."])
    say(ov, "What is your biggest weakness?", final=True, utt=1)
    assert pump(ov, lambda: ov.gen_state == "done" and ov.gen_text == "Weakness answer.")
    assert ov.question.cget("text") == "What is your biggest weakness?"
    assert "Early" not in body(ov)


def test_follow_up_is_drafted_from_the_previous_answer(ov, holder):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    say(ov, "Can you elaborate on that for me?", utt=2)
    assert pump(ov, lambda: "follow-up" in ov.banner.cget("text").lower())            # the screen has caught up...
    assert pump(ov, lambda: len(holder.client.calls) == 2 and drafted(ov) and ov._gen_job == 2)
    p = last_prompt(holder)
    assert "follow-up (asking for more detail) to an earlier question: Tell me about a time you led a team" in p
    assert "I led {{N}} people." in p
    # the parent answer stays up as the prepared one: a follow-up must not wipe the left pane
    assert ov.question.cget("text") == "Tell me about a time you led a team"
    ov.tab_prep.invoke()
    assert "Lead answer: I led" in body(ov)


def test_failure_leaves_the_prepared_answer_alone_and_stops_retrying(ov, holder):
    holder.client = FakeClient(pieces=["x"], fail=_api_error("AuthenticationError", 401))
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: ov.gen_state == "error")
    assert "key was rejected" in ov.gen_status.cget("text") and "Lead" in body(ov) and ov.view == "prepared"
    assert str(ov.tab_gen.cget("state")) == "disabled"
    say(ov, "What is your biggest weakness?", utt=2)
    assert pump(ov, lambda: ov.question.cget("text") == "What is your biggest weakness?")
    assert pump(ov, lambda: ov.gen_state in ("off", "error"))
    assert len(holder.client.calls) == 1 and "WEAK answer" in body(ov)       # no second request


def test_missing_package_is_explained_on_screen(banks_dir, monkeypatch):
    import overlay
    banks_dir.mkdir(parents=True)
    b.Bank("demo", "Demo", [b.Entry("What is your biggest weakness?", "WEAK answer", bank="demo")]).save()
    b.CONFIG_PATH.write_text(json.dumps({"active_banks": ["demo"], "generate": True, "generate_provider": "anthropic"}),
                             encoding="utf-8")
    monkeypatch.setattr(overlay, "Listener", FakeListener)
    monkeypatch.setattr(overlay, "Matcher", lambda: Matcher(embed_fn=fake_embed))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    o = overlay.Overlay()
    try:
        ready(o)
        say(o, "What is your biggest weakness?", utt=1)
        assert pump(o, lambda: "pip install anthropic" in o.gen_status.cget("text"))
        assert "WEAK answer" in body(o)
    finally:
        o._quit()


def test_draft_toggle_switches_it_off_and_remembers(ov, holder):
    ready(ov)
    ov.gen_btn.invoke()
    assert ov.gen_btn.cget("text") == "✨ Draft: off" and json.loads(b.CONFIG_PATH.read_text())["generate"] is False
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: "Lead" in body(ov))
    time.sleep(0.3)
    assert holder.client.calls == [] and str(ov.tab_gen.cget("state")) == "disabled"
    ov.gen_btn.invoke()
    assert ov.gen_btn.cget("text") == "✨ Draft: on"
    say(ov, "What is your biggest weakness?", utt=2)
    assert pump(ov, lambda: drafted(ov))


def test_regenerate_asks_again(ov, holder):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    holder.client = FakeClient(pieces=["Second ", "take."])
    ov.regen_btn.invoke()
    assert pump(ov, lambda: ov.gen_state == "done" and ov.gen_text == "Second take.")
    assert len(holder.client.calls) == 1 and body(ov) == "Second take."


def test_save_turns_the_draft_into_a_prepared_answer(ov, holder):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    ov.save_btn.invoke()
    assert "Saved to" in ov.gen_status.cget("text")
    saved = b.load_bank("demo").entries[-1]
    assert saved.question == "Tell me about a time you led a team?" and saved.tags == ["generated"]
    assert saved.response == DRAFT_TEXT.strip()
    assert ov.save_btn.cget("state") == "disabled"
    ov.gen_state = "done"; ov._refresh_gen_ui()
    ov._save_generated()                                                    # the same question twice is refused
    assert "already in the bank" in ov.gen_status.cget("text")
    assert len([e for e in b.load_bank("demo").entries if e.question.startswith("Tell me about a time")]) == 2


def test_picking_an_answer_by_hand_stops_the_draft(ov, holder):
    holder.client = FakeClient(pieces=["word "] * 80, delay=0.05)
    ready(ov)
    es = ov.matcher.entries
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: ov.gen_text)
    job = ov.gen._job
    ov._pick_entry(es[2])
    assert ov.gen._job != job                                    # the Generator itself was told to stop (no wasted tokens)
    assert pump(ov, lambda: ov.view == "prepared" and "WEAK answer" in body(ov))
    time.sleep(0.4)
    assert ov.gen_text == "" and ov.gen_state == "idle"


def test_groq_draft_streams_beside_the_prepared_answer_and_names_its_model(ov):
    sent = []

    def http(payload):
        sent.append(payload)
        for piece in ("**Listen first.** ", "Then {{your example}}."):
            yield json.dumps({"choices": [{"delta": {"content": piece}, "finish_reason": None}]})
        yield json.dumps({"choices": [], "usage": {"prompt_tokens": 600, "completion_tokens": 20}})
        yield "[DONE]"
    ov.gen = g.Generator(ov.cfg, lambda job, kind, data: ov.events.put(("gen", (job, kind, data))), http=http)
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    assert body(ov) == "Listen first. Then ⚠ your example."
    assert ov.gen_status.cget("text").endswith("llama-3.3-70b-versatile")
    assert sent[0]["messages"][0]["role"] == "system" and "led a team" in sent[0]["messages"][1]["content"]
    assert "Lead answer" in sent[0]["messages"][1]["content"] or "I led {{N}} people." in sent[0]["messages"][1]["content"]


def test_stream_safe_hides_unfinished_markers(ov):
    f = ov.__class__._stream_safe
    assert f("Say **this") == "Say this" and f("Say **this** now") == "Say **this** now"
    assert f("I led {{team") == "I led " and f("I led {{team}} fine") == "I led {{team}} fine"


def test_a_long_draft_does_not_freeze_the_window(ov, holder):
    holder.client = FakeClient(pieces=[f"word{i} " for i in range(400)], delay=0.002)
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    t0, ticks = time.monotonic(), 0
    while not drafted(ov) and time.monotonic() - t0 < 5:
        ov.root.update(); ticks += 1; time.sleep(0.005)
    assert drafted(ov) and body(ov).count("word") == 400 and ticks > 20
