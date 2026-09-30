from bank import Entry
from matcher import Matcher
from session import Session
from conftest import fake_embed

LED = "Tell me about a time you led a team."


def test_a_new_question_becomes_the_anchor(session):
    h = session.hear(LED)
    assert h.kind == "new" and h.primary.entry.question == LED and h.anchor.question == LED
    assert h.followup is None and h.is_question


def test_prepared_followup_is_recognised_and_answered(session):
    session.hear(LED)
    session.clock["t"] += 20
    h = session.hear("How big was the team?")
    assert h.kind == "followup" and h.followup.anchor.question == LED
    assert h.followup.prepared.question == "How big was the team?"
    assert h.followup.prepared.answer.startswith("Six people") and h.followup.focus == []


def test_reworded_prepared_followup_still_matches(session):
    session.hear(LED)
    h = session.hear("Okay, can you tell me more about that rollback plan?")
    assert h.kind == "followup" and h.followup.prepared.question == "Tell me more about the rollback plan."


def test_generic_followup_points_at_the_answer_and_lists_prepared_ones(session):
    session.hear(LED)
    h = session.hear("What was the result?")
    f = h.followup
    assert h.kind == "followup" and f.prepared is None and f.signal.kind == "outcome"
    assert any("uptime rose" in s or "two weeks early" in s for s in f.focus)
    assert [p.question for p in f.related] == ["How big was the team?", "Tell me more about the rollback plan."]


def test_bare_probe_is_a_followup(session):
    session.hear(LED)
    assert session.hear("Why?").kind == "followup"


def test_no_followup_without_a_live_anchor(session):
    assert session.hear("What was the result?").kind == "unmatched"         # nothing to be a follow-up to
    session.hear(LED)
    session.clock["t"] += 600                                                # long past the 240 s window
    assert session.live_anchor() is None
    assert session.hear("What was the result?").kind == "unmatched"


def test_prepared_followup_found_even_after_the_window_expires(session):
    session.hear(LED)
    session.clock["t"] += 600
    h = session.hear("Tell me more about the rollback plan.")
    assert h.kind == "followup" and h.followup.anchor.question == LED        # matched on its own wording


def test_a_clear_new_question_replaces_the_anchor(session):
    session.hear(LED)
    h = session.hear("Why should we hire you?")
    assert h.kind == "new" and h.anchor.question == "Why should we hire you?"
    assert session.hear("What was the result?").followup.anchor.question == "Why should we hire you?"


def test_followup_keeps_the_anchor_alive(session):
    session.hear(LED)
    for _ in range(3):
        session.clock["t"] += 200       # each step is inside the window; together they are not
        assert session.hear("Tell me more about that.").kind == "followup"


def test_cue_followup_does_not_hijack_a_question_that_matches_the_bank_exactly(entries):
    extra = entries + [Entry("What would you do differently in your last job?", "Plan earlier.")]
    m = Matcher(embed=fake_embed)
    m.load(extra)
    s = Session(m, {"match_threshold": 0.78, "followups": True})
    s.hear(LED)
    exact = s.hear("What would you do differently in your last job?")
    assert exact.kind == "new"
    s.hear(LED)
    probe = s.hear("What would you do differently?")                        # the same idea, but as a probe
    assert probe.kind == "followup" and probe.followup.signal.kind == "hindsight"


def test_followups_can_be_switched_off(matcher):
    s = Session(matcher, {"match_threshold": 0.78, "followups": False})
    s.hear(LED)
    h = s.hear("How big was the team?")
    assert h.kind != "followup" and h.followup is None


def test_statements_and_uncertainty(session):
    h = session.hear("We are going to record this meeting today.")
    assert h.kind == "statement" and not h.is_question and not h.uncertain
    session.hear(LED)
    assert session.hear(LED, confidence=0.4).uncertain
    assert not session.hear(LED, confidence=0.95).uncertain


def test_reset_and_reload_do_not_lose_track(session, matcher, entries):
    session.hear(LED)
    matcher.load(list(entries))                     # banks reloaded: new Entry objects, same questions
    assert session.hear("What was the result?").kind == "followup"
    session.reset()
    assert session.live_anchor() is None


def test_a_followup_under_the_current_answer_gets_a_head_start(matcher):
    """A prepared follow-up that only just misses the threshold is accepted when its answer is on the table."""
    text = "how large was the team then"
    score = matcher.match_followups(text, k=1)[0].score
    cfg = {"match_threshold": 0.99, "followup_threshold": score + 0.03, "followups": True}   # bonus is 0.06
    cold = Session(matcher, dict(cfg))
    assert cold.hear(text).kind != "followup"
    warm = Session(matcher, dict(cfg))
    warm.hear(LED)
    h = warm.hear(text)
    assert h.kind == "followup" and h.followup.prepared.question == "How big was the team?"


def test_a_better_bank_question_beats_a_similar_prepared_followup(entries):
    extra = entries + [Entry("How big was the team at your last job?", "Forty people.")]
    m = Matcher(embed=fake_embed)
    m.load(extra)
    s = Session(m, {"match_threshold": 0.78, "followups": True})
    s.hear(LED)
    h = s.hear("How big was the team at your last job?")
    assert h.kind == "new" and h.primary.entry.question == "How big was the team at your last job?"
