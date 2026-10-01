"""Drafted answers in the overlay: question | prepared | generated side by side, streaming, failures. Headless."""
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

DRAFT = ["I lead by listening first. ", "On my last team, which was about six people, ", "I kept everyone aligned."]
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


def gbody(o):
    return o.gtext.get("1.0", "end").strip()


def last_prompt(holder):
    return holder.client.calls[-1]["messages"][0]["content"]


def test_question_prepared_and_generated_sit_side_by_side(ov):
    ready(ov)
    ov.root.update()
    left, mid, right = ov.heard, ov.text, ov.gtext
    assert left.winfo_rootx() + left.winfo_width() <= mid.winfo_rootx() + 2          # question, then prepared...
    assert mid.winfo_rootx() + mid.winfo_width() <= right.winfo_rootx() + 2          # ...then generated
    assert abs(mid.winfo_rooty() - right.winfo_rooty()) <= 2 and abs(mid.winfo_height() - right.winfo_height()) <= 2
    assert mid.winfo_ismapped() and right.winfo_ismapped()
    assert abs(mid.winfo_width() - right.winfo_width()) < 0.35 * mid.winfo_width()    # a fair share each


def test_both_answers_are_on_screen_at_the_same_time(ov):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    assert "Lead answer: I led" in body(ov)                                            # prepared (its own ⚠ highlight is fine)
    assert gbody(ov) == DRAFT_TEXT.strip()                                              # generated, in the next column
    assert ov.text.winfo_ismapped() and ov.gtext.winfo_ismapped()
    assert ov.prep_status.cget("text") == "from your bank: demo"
    assert not hasattr(ov, "tab_gen") and not hasattr(ov, "tab_prep")                  # no tabs to flip between


def test_prepared_shows_at_once_and_stays_while_the_draft_streams_in(ov, holder):
    holder.client = FakeClient(pieces=DRAFT, delay=0.15)
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: "Lead answer" in body(ov) and ov.gen_state == "drafting")
    assert pump(ov, lambda: gbody(ov).startswith("I lead by listening") and ov.gen_state == "drafting")   # partway through...
    assert "Lead answer: I led" in body(ov)                                             # ...both are visible
    assert pump(ov, lambda: drafted(ov))
    assert "Lead answer: I led" in body(ov)                                             # the prepared one never moved
    assert ov.gen_status.cget("text").startswith("adapted from your prepared answer")
    assert "Tell me about a time you led a team" in ov.heard.cget("text")
    assert ov.question.cget("text") == "Tell me about a time you led a team"
    assert ov.save_btn.cget("state") == "normal"


def test_prompt_carries_the_question_and_the_prepared_answer(ov, holder):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    p = last_prompt(holder)
    assert "<question>\nTell me about a time you led a team?\n</question>" in p
    assert "very close match" in p and "I led {{N}} people." in p
    assert holder.client.calls[0]["model"] == "claude-opus-5-5"


def test_the_draft_never_shows_bullets_bold_or_yellow_highlights(ov, holder):
    holder.client = FakeClient(pieces=["- I led **six** people\n", "- we shipped on time\n\n", "I learned that {{lesson}} matters."])
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    text = gbody(ov)
    assert text == "I led six people. we shipped on time.\nI learned that lesson matters."     # prose; one paragraph per line
    for bad in ("•", "- ", "**", "{{", "⚠"):
        assert bad not in text
    assert ov.gtext.tag_ranges("ph") == () and ov.gtext.tag_ranges("b") == ()           # nothing highlighted or bold
    assert "Lead answer" in body(ov) and ov.text.tag_ranges("ph")                       # the PREPARED answer keeps its own


def test_paragraphs_stay_paragraphs(ov, holder):
    holder.client = FakeClient(pieces=["First thought, said plainly.\n\n", "Second thought follows."])
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    assert gbody(ov).split("\n") == ["First thought, said plainly.", "Second thought follows."]


def test_no_prepared_answer_fits_so_the_gap_is_filled(ov, holder):
    ready(ov)
    say(ov, "What is your experience with Kubernetes deployments and monitoring?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    assert ov.question.cget("text") == "No close prepared answer"
    assert body(ov) == "No prepared answer matches this question." and not ov.text.tag_ranges("ph")
    assert gbody(ov) == DRAFT_TEXT.strip()
    p = last_prompt(holder)
    assert "Kubernetes deployments" in p
    assert "None of the prepared answers" in p or "loosely related" in p or "related match" in p
    assert "no prepared answer matched" in ov.gen_status.cget("text") or "blended" in ov.gen_status.cget("text")


def test_statements_are_not_drafted(ov, holder):
    ready(ov)
    say(ov, "Thanks everyone for joining us today, we have a busy agenda.", utt=1)
    assert pump(ov, lambda: ov.heard.cget("text").startswith("Heard: “Thanks"))
    time.sleep(0.3)
    assert holder.client.calls == [] and ov.gen_state == "idle"
    assert gbody(ov) == "A tailored answer will appear here for each question."


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
    assert "WEAK answer" in body(ov) and gbody(ov) == "Weakness answer."
    assert "Early" not in gbody(ov)


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
    assert ov.question.cget("text") == "Tell me about a time you led a team"            # the parent stays up as the prepared one
    assert "Lead answer: I led" in body(ov) and gbody(ov) == DRAFT_TEXT.strip()          # and both columns are still showing


def test_failure_leaves_the_prepared_answer_alone_and_stops_retrying(ov, holder):
    holder.client = FakeClient(pieces=["x"], fail=_api_error("AuthenticationError", 401))
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: ov.gen_state == "error")
    assert "key was rejected" in ov.gen_status.cget("text") and "key was rejected" in gbody(ov)
    assert "Lead answer" in body(ov)                                                    # the prepared answer is untouched
    say(ov, "What is your biggest weakness?", utt=2)
    assert pump(ov, lambda: ov.question.cget("text") == "What is your biggest weakness?")
    assert pump(ov, lambda: ov.gen_state in ("off", "error"))
    assert len(holder.client.calls) == 1 and "WEAK answer" in body(ov)                 # no second request


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
        assert "pip install anthropic" in gbody(o) and "WEAK answer" in body(o)
    finally:
        o._quit()


def test_draft_toggle_switches_it_off_and_remembers(ov, holder):
    ready(ov)
    ov.gen_btn.invoke()
    assert ov.gen_btn.cget("text") == "✨ Draft: off" and json.loads(b.CONFIG_PATH.read_text())["generate"] is False
    assert "switched off" in gbody(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: "Lead" in body(ov))
    time.sleep(0.3)
    assert holder.client.calls == [] and "switched off" in gbody(ov)
    ov.gen_btn.invoke()
    assert ov.gen_btn.cget("text") == "✨ Draft: on"
    say(ov, "What is your biggest weakness?", utt=2)
    assert pump(ov, lambda: drafted(ov)) and gbody(ov) == DRAFT_TEXT.strip()


def test_regenerate_asks_again(ov, holder):
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    holder.client = FakeClient(pieces=["Second ", "take."])
    ov.regen_btn.invoke()
    assert pump(ov, lambda: ov.gen_state == "done" and ov.gen_text == "Second take.")
    assert len(holder.client.calls) == 1 and gbody(ov) == "Second take." and "Lead answer" in body(ov)


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
    ov.gen_state = "done"
    ov._refresh_gen_ui()
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
    assert pump(ov, lambda: "WEAK answer" in body(ov))
    time.sleep(0.4)
    assert ov.gen_text == "" and ov.gen_state == "idle" and gbody(ov).startswith("A tailored answer")


def test_groq_draft_streams_beside_the_prepared_answer_and_names_its_model(ov):
    sent = []

    def http(payload):
        sent.append(payload)
        for piece in ("Listening first has always worked for me. ", "Then I check in with everyone."):
            yield json.dumps({"choices": [{"delta": {"content": piece}, "finish_reason": None}]})
        yield json.dumps({"choices": [], "usage": {"prompt_tokens": 600, "completion_tokens": 20}})
        yield "[DONE]"
    ov.gen = g.Generator(ov.cfg, lambda job, kind, data: ov.events.put(("gen", (job, kind, data))), http=http)
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    assert pump(ov, lambda: drafted(ov))
    assert gbody(ov) == "Listening first has always worked for me. Then I check in with everyone."
    assert "Lead answer" in body(ov)
    assert ov.gen_status.cget("text").endswith("llama-3.3-70b-versatile")
    assert sent[0]["messages"][0]["role"] == "system" and "led a team" in sent[0]["messages"][1]["content"]


def test_stream_safe_hides_unfinished_markers(ov):
    f = ov.__class__._stream_safe
    assert f("Say **this") == "Say this" and f("Say **this** now") == "Say **this** now"
    assert f("I led {{team") == "I led " and f("I led {{team}} fine") == "I led {{team}} fine"


def test_half_streamed_markers_never_flash_on_screen(ov, holder):
    holder.client = FakeClient(pieces=["I led {{te", "am size}} people and **kept ", "them** aligned."], delay=0.12)
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    seen = set()
    end = time.monotonic() + 5
    while time.monotonic() < end and not drafted(ov):
        ov.root.update()
        seen.add(gbody(ov))
        time.sleep(0.01)
    assert drafted(ov) and gbody(ov) == "I led team size people and kept them aligned."
    assert not any("{{" in t or "**" in t for t in seen)


def test_a_long_draft_does_not_freeze_the_window(ov, holder):
    holder.client = FakeClient(pieces=[f"word{i} " for i in range(400)], delay=0.002)
    ready(ov)
    say(ov, "Tell me about a time you led a team?", utt=1)
    t0, ticks = time.monotonic(), 0
    while not drafted(ov) and time.monotonic() - t0 < 5:
        ov.root.update()
        ticks += 1
        time.sleep(0.005)
    assert drafted(ov) and gbody(ov).count("word") == 400 and ticks > 20
