"""Drafts held to about 45-50 seconds, and a story is never told twice in a session (fake client, no network)."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

import bank as b
import generate as g
from followup import Cue
from story_samples import (BILLING_MISTAKE, CHECKOUT_TIMEOUTS, DASHBOARD, HACKATHON, LED_MIGRATION, MIGRATION_REWORDED,
                           MIGRATION_SHORT, MOTIVATION, MOTIVATION_AGAIN, SUPPORT_QUEUE, TICKETING_ROLLOUT)
from test_generate import FakeStream

THR = 0.78


def chunks(text, size=4):
    """A text as the little pieces a model streams: `size` words at a time, spaces kept."""
    words = text.split(" ")
    return [" ".join(words[i:i + size]) + (" " if i + size < len(words) else "") for i in range(0, len(words), size)]


class CountedStream(FakeStream):
    """A fake model stream that notes every piece the generator actually reads."""

    def __init__(self, parts, on_piece, **kw):
        super().__init__(parts, **kw)
        self.on_piece = on_piece

    @property
    def text_stream(self):
        for p in self.pieces:
            self.on_piece()
            yield p


class SeqClient:
    """Answers each call with the next prepared text (the last one repeats). Records every request.
    `fail_on`: call numbers that raise; `stops`: {call number: stop reason}."""

    def __init__(self, *texts, fail_on=(), stops=None):
        self.texts, self.calls, self.fail_on, self.stops, self.consumed = list(texts), [], set(fail_on), stops or {}, []
        self.messages = self

    def stream(self, **kw):
        n = len(self.calls)
        self.calls.append(kw)
        if n in self.fail_on:
            return FakeStream(["x"], fail=RuntimeError("boom"))
        text = self.texts[min(n, len(self.texts) - 1)]
        return CountedStream(chunks(text), lambda: self.consumed.append(n), stop=self.stops.get(n, "end_turn"))

    def prompt(self, n):
        return self.calls[n]["messages"][0]["content"]


class Harness:
    def __init__(self, *texts, cfg=None, **client_kw):
        self.client = SeqClient(*texts, **client_kw)
        self.events, self.cv = [], threading.Condition()
        self.gen = g.Generator({"generate": True, **(cfg or {})}, self._on, client_factory=lambda: self.client)

    def _on(self, job, kind, data):
        with self.cv:
            self.events.append((job, kind, data))
            self.cv.notify_all()

    def ask(self, question, utt=1, repeat_ok=False, matches=(), timeout=5):
        req = g.build_request(question, list(matches), thr=THR, utt=utt, words=self.gen.words)
        req.repeat_ok = repeat_ok
        job = self.gen.start(req)
        end = time.monotonic() + timeout
        with self.cv:
            while time.monotonic() < end:
                for j, k, d in self.events:
                    if j == job and k in ("done", "error"):
                        return job, [e for e in self.events if e[0] == job]
                self.cv.wait(0.05)
        raise AssertionError("no result: " + str(self.events[-3:]))

    def done(self, question, **kw):
        _, ev = self.ask(question, **kw)
        assert ev[-1][1] == "done", ev[-1]
        return ev[-1][2]

    def kinds(self, ev):
        return [k for _, k, _ in ev]


def words(n, sentence="That is how it went."):
    """About n words of plain sentences."""
    out, per = [], len(sentence.split())
    while len(" ".join(out).split()) + per <= n:
        out.append(sentence)
    return " ".join(out)


# ---------- length: about 45-50 seconds ----------

def test_the_default_length_is_about_46_seconds_and_never_over_about_51():
    gen = g.Generator({}, lambda *a: None)
    assert gen.seconds == 46 and gen.words == 106 and gen.max_words == 117
    assert 45 <= gen.words / g.WORDS_PER_SECOND <= 50 and gen.max_words / g.WORDS_PER_SECOND < 51.5


def test_length_follows_generate_seconds_within_sane_bounds():
    assert g.Generator({"generate_seconds": 30}, lambda *a: None).words == 69
    assert g.Generator({"generate_seconds": 1}, lambda *a: None).seconds == g.MIN_SECONDS
    assert g.Generator({"generate_seconds": 9999}, lambda *a: None).seconds == g.MAX_SECONDS
    assert g.Generator({"generate_seconds": "abc"}, lambda *a: None).seconds == g.DEFAULT_SECONDS
    assert g.Generator({"generate_seconds": None}, lambda *a: None).seconds == g.DEFAULT_SECONDS


def test_the_old_word_count_setting_no_longer_controls_the_length():
    """Existing config.json files carry generate_words: 150 from the previous version."""
    assert g.Generator({"generate_words": 150}, lambda *a: None).words == 106
    assert "generate_words" not in b.DEFAULT_CONFIG and b.DEFAULT_CONFIG["generate_seconds"] == 46


def test_the_model_is_told_the_length_in_seconds_and_words():
    gen = g.Generator({}, lambda *a: None)
    s = g.build_request("q?", [], thr=THR, words=gen.words).system[0]["text"]
    assert "about 46 seconds" in s and "about 106 words and never more than 117" in s
    assert "condense it rather than covering everything" in s
    assert "{seconds}" not in s and "{words}" not in s and "{max_words}" not in s
    assert "about a sentence each for the Result and the Reflection, so the Result is never cut off" in s


def test_fit_words_leaves_short_text_alone():
    assert g.fit_words("One two three.", 5) == ("One two three.", False)
    assert g.fit_words("", 5) == ("", False)
    assert g.fit_words("anything at all here", 0) == ("anything at all here", False)


def test_fit_words_cuts_at_the_last_sentence_that_fits():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve thirteen fourteen."
    assert g.fit_words(text, 12) == ("One two three four five. Six seven eight nine ten.", True)


def test_fit_words_does_not_mistake_a_decimal_or_an_abbreviation_for_a_sentence_end():
    text = "We grew revenue by 3.5 percent across the whole region over the following quarter with one more team member"
    cut, trimmed = g.fit_words(text, 10)
    assert trimmed and cut.endswith(".") and "3.5 percent" in cut and len(cut.split()) <= 10


def test_fit_words_cuts_mid_sentence_only_when_no_sentence_ends_in_time():
    text = "This one very long sentence just keeps going and going without ever stopping for breath, then ends."
    cut, trimmed = g.fit_words(text, 8)
    assert trimmed and cut == "This one very long sentence just keeps going." and len(cut.split()) == 8


def test_fit_words_keeps_paragraph_breaks():
    text = "First paragraph here.\n\nSecond paragraph is a bit longer than the first one is. Third bit."
    cut, trimmed = g.fit_words(text, 14)
    assert trimmed and cut == "First paragraph here.\n\nSecond paragraph is a bit longer than the first one is."


def test_a_draft_that_runs_long_is_stopped_and_cut_at_a_sentence():
    h = Harness(words(400))
    ev = h.ask("what is your weakness?")[1]
    done = ev[-1][2]
    assert ev[-1][1] == "done" and done["trimmed"] and done["words"] <= h.gen.max_words
    assert done["text"].endswith(".") and done["spoken"] <= 51
    shown = "".join(d for _, k, d in ev if k == "delta")
    assert len(shown.split()) <= h.gen.max_words + 4                         # what streamed to the screen stayed in bounds too
    assert len(h.client.consumed) < 40                                      # and it stopped reading: 400 words is 100 pieces


def test_a_draft_within_the_limit_is_not_touched():
    text = words(100)
    done = Harness(text).done("what is your weakness?")
    assert done["text"] == text and not done["trimmed"] and 40 <= done["spoken"] <= 46


def test_a_slightly_over_target_draft_is_kept_whole():
    text = words(112)                                                        # over the 106 target, under the 117 limit
    done = Harness(text).done("what is your weakness?")
    assert done["text"] == text and not done["trimmed"]


def test_a_shorter_target_shortens_the_cut():
    done = Harness(words(400), cfg={"generate_seconds": 30}).done("what is your weakness?")
    assert done["words"] <= 76 and done["trimmed"]


def test_groq_drafts_are_cut_to_length_too_and_the_stream_is_closed():
    sent = {"lines": 0, "closed": False}
    text = words(400)

    def http(payload):
        assert payload["max_tokens"] <= g.GROQ_MAX_TOKENS
        try:
            for piece in chunks(text):
                sent["lines"] += 1
                yield json.dumps({"choices": [{"delta": {"content": piece}}]})
            yield "[DONE]"
        finally:
            sent["closed"] = True
    events, done = [], threading.Event()
    gen = g.Generator({"generate": True}, lambda j, k, d: (events.append((k, d)), done.set() if k == "done" else None), http=http)
    gen.start(g.build_request("q?", [], thr=THR, words=gen.words))
    assert done.wait(5)
    final = [d for k, d in events if k == "done"][0]
    assert final["trimmed"] and final["words"] <= gen.max_words and final["text"].endswith(".")
    assert sent["lines"] < 40 and sent["closed"]


def test_the_status_estimate_counts_the_words_in_the_finished_text():
    done = Harness("One two three four five six seven eight nine ten.").done("q?")
    assert done["words"] == 10 and done["spoken"] == pytest.approx(10 / g.WORDS_PER_SECOND)


# ---------- not telling the same story twice ----------

def test_the_first_story_is_told_without_any_warning():
    h = Harness(LED_MIGRATION)
    done = h.done("Tell me about a time you led a team")
    assert "<already_told>" not in h.client.prompt(0) and not done["reworded"] and not done["similar"]
    assert len(h.gen.stories) == 1


def test_the_next_question_is_told_which_stories_were_already_used():
    h = Harness(LED_MIGRATION, HACKATHON)
    h.done("Tell me about a time you led a team", utt=1)
    h.done("Tell me about a time you handled conflict", utt=2)
    p = h.client.prompt(1)
    assert "<already_told>" in p and 'Asked "Tell me about a time you led a team"' in p
    assert "A while back at my last company we had to move our billing system" in p                # the story's opening
    assert "Do not retell any of them" in p
    assert p.index("<already_told>") < p.index(g.ASK) and p.rstrip().endswith(g.ASK)         # before the final instruction
    assert "<question>\nTell me about a time you handled conflict\n</question>" in p
    assert "<already_told>" not in h.client.prompt(0)


def test_the_instructions_to_the_model_tell_it_how_to_retell_a_used_story():
    s = g.build_request("q?", [], thr=THR).system[0]["text"]
    assert "never retell one of them" in s and "change the circumstances" in s and "nobody would recognise it" in s
    assert "unless its story was already used" in s                                          # the adapt-the-prepared-answer rule defers to it


def test_a_different_story_goes_through_with_one_call():
    h = Harness(LED_MIGRATION, HACKATHON)
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    assert len(h.client.calls) == 2 and "retry" not in h.kinds(ev)
    assert not ev[-1][2]["reworded"] and not ev[-1][2]["similar"] and ev[-1][2]["text"] == HACKATHON
    assert len(h.gen.stories) == 2


def test_a_story_that_repeats_is_rewritten_once_with_a_different_one():
    h = Harness(LED_MIGRATION, MIGRATION_SHORT, HACKATHON)
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    kinds = h.kinds(ev)
    assert "retry" in kinds and kinds.index("retry") < kinds.index("done")
    done = ev[-1][2]
    assert done["text"] == HACKATHON and done["reworded"] and not done["similar"]
    assert len(h.client.calls) == 3
    retry_prompt = h.client.prompt(2)
    assert "Your first attempt told the same story as this earlier answer" in retry_prompt
    assert "clearly a different story" in retry_prompt and retry_prompt.rstrip().endswith(g.ASK)
    assert "<already_told>" in retry_prompt                                                  # the full list is still there
    stored = [s.text for s in h.gen.stories.recall()]
    assert stored[0] == HACKATHON and MIGRATION_SHORT not in stored                          # only what was actually told is kept


def test_a_retelling_that_gets_past_the_opening_is_withdrawn_from_the_screen_when_caught():
    """Its words stream live; once the whole draft is seen to repeat, the screen is told to clear and a new draft follows."""
    h = Harness(LED_MIGRATION, MIGRATION_REWORDED, HACKATHON)
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    kinds = h.kinds(ev)
    i = kinds.index("retry")
    assert "delta" in kinds[:i]                                                              # it had started to show
    shown = "".join(d for _, k, d in ev[:i] if k == "delta")
    assert shown.startswith("At my previous job we were replacing the old invoicing tool")
    assert ev[i][2] == g.REWORD_NOTE and kinds[-1] == "done"
    assert "".join(d for _, k, d in ev[i + 1:] if k == "delta") == HACKATHON                 # the replacement, in full
    assert ev[-1][2]["text"] == HACKATHON and ev[-1][2]["reworded"] and not ev[-1][2]["similar"]


def test_a_repeated_opening_is_caught_before_it_is_shown():
    h = Harness(LED_MIGRATION, MIGRATION_SHORT, HACKATHON)
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    first_delta = next(d for _, k, d in ev if k == "delta")
    assert first_delta.startswith("In my second year")                                       # the repeat never reached the screen
    assert not any(k == "delta" and "billing" in d for _, k, d in ev)
    assert h.kinds(ev).index("retry") < h.kinds(ev).index("delta")
    # and the repeated draft was not read to the end: the gate stopped it after about GATE_WORDS words
    second_call_pieces = h.client.consumed.count(1)
    assert second_call_pieces <= g.GATE_WORDS // 4 + 3 and ev[-1][2]["reworded"]


def test_the_opening_is_not_held_back_when_nothing_has_been_told():
    h = Harness(LED_MIGRATION)
    _, ev = h.ask("Tell me about a time you led a team", utt=1)
    assert h.kinds(ev)[:2] == ["start", "delta"] and ev[1][2] == chunks(LED_MIGRATION)[0]


def test_the_opening_is_held_only_for_the_first_words_then_streams_live():
    h = Harness(LED_MIGRATION, HACKATHON)
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    deltas = [d for _, k, d in ev if k == "delta"]
    assert len(deltas) >= 2 and len(deltas[0].split()) >= g.GATE_WORDS - 4                   # the held words arrive together
    assert "".join(deltas) == HACKATHON[:len("".join(deltas))]


def test_if_the_rewrite_still_repeats_it_is_shown_and_marked():
    h = Harness(LED_MIGRATION, MIGRATION_SHORT, MIGRATION_SHORT.replace("At my last company", "At a previous company"))
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    done = ev[-1][2]
    assert done["similar"] and done["reworded"] and len(h.client.calls) == 3                 # one rewrite, never a loop
    assert done["text"].startswith("At a previous company")


def test_if_the_rewrite_fails_the_first_draft_is_still_delivered():
    h = Harness(LED_MIGRATION, MIGRATION_REWORDED, fail_on=(2,))
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    done = ev[-1][2]
    assert ev[-1][1] == "done" and done["similar"] and done["text"] == MIGRATION_REWORDED
    assert not h.gen.disabled_reason


def test_if_the_rewrite_is_declined_the_first_draft_is_still_delivered():
    h = Harness(LED_MIGRATION, MIGRATION_REWORDED, "", stops={2: "refusal"})
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    assert ev[-1][1] == "done" and ev[-1][2]["similar"] and ev[-1][2]["text"] == MIGRATION_REWORDED


def test_if_the_rewrite_fails_after_the_opening_was_held_back_the_error_is_shown():
    h = Harness(LED_MIGRATION, MIGRATION_SHORT, fail_on=(2,))
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Tell me about a time you handled conflict", utt=2)
    assert ev[-1][1] == "error"                                                              # nothing usable was ever produced


def test_a_follow_up_may_carry_on_with_the_same_story():
    h = Harness(LED_MIGRATION, MIGRATION_SHORT)
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("What was the result of that project?", utt=2, repeat_ok=True)
    assert "retry" not in h.kinds(ev) and "<already_told>" not in h.client.prompt(1)
    assert not ev[-1][2]["reworded"] and not ev[-1][2]["similar"] and len(h.client.calls) == 2


def test_building_a_request_for_a_follow_up_marks_it_as_one():
    parent = b.Entry("Tell me about a time you led a team", "I led six people.", bank="t")
    r = g.build_request("what was the outcome?", [], thr=THR, parent=parent, cue=Cue("outcome", 1.0, "x"))
    assert r.repeat_ok is True
    assert g.build_request("tell me about a time you failed", [], thr=THR).repeat_ok is False


def test_the_same_question_asked_again_gets_the_same_story_not_a_new_one():
    h = Harness(LED_MIGRATION, MIGRATION_SHORT)
    h.done("Tell me about a time you led a team", utt=1)
    _, ev = h.ask("Can you tell me about a time when you led a team?", utt=2)
    assert "retry" not in h.kinds(ev) and len(h.client.calls) == 2
    assert "<already_told>" not in h.client.prompt(1)


def test_regenerating_a_draft_is_not_held_against_itself_and_replaces_it():
    h = Harness(LED_MIGRATION, MIGRATION_SHORT)
    h.done("Tell me about a time you led a team", utt=7)
    done = h.done("Tell me about a time you led a team", utt=7)                              # same utterance, asked again for
    assert not done["reworded"] and len(h.client.calls) == 2 and "<already_told>" not in h.client.prompt(1)
    assert len(h.gen.stories) == 1 and h.gen.stories.recall()[0].text == MIGRATION_SHORT


def test_an_early_guess_and_its_final_wording_count_as_one_story():
    h = Harness(HACKATHON, LED_MIGRATION)
    h.done("Tell me about a time you led", utt=3)                                            # the early guess, a different story
    done = h.done("Tell me about a time you led a team", utt=3)                              # the final wording
    assert len(h.gen.stories) == 1 and h.gen.stories.recall()[0].text == LED_MIGRATION
    assert not done["reworded"] and "<already_told>" not in h.client.prompt(1)


def test_a_draft_that_was_replaced_before_it_finished_is_not_counted_as_told():
    events = []
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: FakeStream(chunks(LED_MIGRATION), delay=0.03)))
    gen = g.Generator({"generate": True}, lambda j, k, d: events.append((j, k, d)), client_factory=lambda: client)
    gen.start(g.build_request("Tell me about a time you led a team", [], thr=THR, utt=1))
    time.sleep(0.2)
    gen.cancel()
    time.sleep(0.5)
    assert len(gen.stories) == 0 and not any(k == "done" for _, k, _ in events)


def test_a_draft_replaced_at_the_last_moment_is_not_counted_as_told():
    """The newer question arrives just as the old draft finishes: the old one is neither shown nor remembered."""
    box = {}
    events = []

    class LateStream(FakeStream):
        def get_final_message(self):
            box["gen"].cancel()
            return super().get_final_message()
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: LateStream(chunks(LED_MIGRATION))))
    gen = box["gen"] = g.Generator({"generate": True}, lambda j, k, d: events.append(k), client_factory=lambda: client)
    gen.start(g.build_request("Tell me about a time you led a team", [], thr=THR, utt=1))
    time.sleep(0.5)
    assert len(gen.stories) == 0 and "done" not in events


def test_answers_that_are_not_stories_are_never_held_to_this():
    h = Harness(MOTIVATION, MOTIVATION_AGAIN)
    h.done("Why do you want this role?", utt=1)
    _, ev = h.ask("What attracted you to us?", utt=2)
    assert "retry" not in h.kinds(ev) and len(h.gen.stories) == 0 and "<already_told>" not in h.client.prompt(1)


def test_the_check_can_be_switched_off():
    h = Harness(LED_MIGRATION, MIGRATION_SHORT, cfg={"generate_avoid_repeats": False})
    h.done("Tell me about a time you led a team", utt=1)
    done = h.done("Tell me about a time you handled conflict", utt=2)
    assert len(h.client.calls) == 2 and "<already_told>" not in h.client.prompt(1)
    assert not done["reworded"] and not done["similar"] and len(h.gen.stories) == 0


def test_only_the_most_recent_stories_are_listed_to_the_model():
    h = Harness(MOTIVATION)
    told = [HACKATHON, BILLING_MISTAKE, DASHBOARD, TICKETING_ROLLOUT, CHECKOUT_TIMEOUTS, SUPPORT_QUEUE, LED_MIGRATION]
    for i, text in enumerate(told, 1):
        assert h.gen.stories.add(i, f"earlier question number {i}", text)
    h.done("Why do you want this role?", utt=99)
    p = h.client.prompt(0)
    assert p.count('Asked "') == 6 and "earlier question number 7" in p and "earlier question number 1" not in p


def test_a_new_generator_starts_with_no_stories():
    """Closing the program forgets everything: a fresh run begins empty."""
    h = Harness(LED_MIGRATION)
    h.done("Tell me about a time you led a team", utt=1)
    assert len(h.gen.stories) == 1
    assert len(g.Generator({"generate": True}, lambda *a: None).stories) == 0


def test_the_story_notes_are_added_after_the_cached_part_of_the_prompt():
    h = Harness(LED_MIGRATION, HACKATHON)
    h.done("Tell me about a time you led a team", utt=1)
    h.done("Tell me about a time you handled conflict", utt=2)
    assert h.client.calls[0]["system"] == h.client.calls[1]["system"]                        # same system text: still cacheable


def test_a_refusal_is_reported_as_a_message():
    _, ev = Harness("", stops={0: "refusal"}).ask("Tell me about a time you led a team", utt=1)
    assert ev[-1][1] == "error" and "declined" in ev[-1][2]
