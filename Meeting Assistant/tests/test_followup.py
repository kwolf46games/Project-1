import pytest

import bank as b
from followup import Conversation, detect_followup


@pytest.mark.parametrize("text,kind", [
    ("Can you tell me more about that?", "elaborate"),
    ("Could you elaborate on that?", "elaborate"),
    ("Tell me more.", "elaborate"),
    ("Can you be more specific?", "elaborate"),
    ("What do you mean by that?", "elaborate"),
    ("Walk me through that.", "elaborate"),
    ("Go on.", "elaborate"),
    ("How so?", "elaborate"),
    ("Can you give me an example of that?", "example"),
    ("Do you have a specific example?", "example"),
    ("Can you think of another example?", "example"),
    ("Any examples?", "example"),
    ("What was the outcome?", "outcome"),
    ("And what was the result of that?", "outcome"),
    ("How did that turn out?", "outcome"),
    ("What did you learn from it?", "outcome"),
    ("So what happened next?", "outcome"),
    ("Why did you do that?", "reason"),
    ("Why is that?", "reason"),
    ("What made you choose that approach?", "reason"),
    ("How did you handle that?", "how"),
    ("What steps did you take?", "how"),
    ("What would you do differently?", "alternative"),
    ("Is there anything else?", "alternative"),
    ("What was the hardest part?", "challenge"),
    ("You mentioned a tight deadline. How tight was it?", "reference"),
    ("Going back to what you said about the team, who was on it?", "reference"),
    ("A quick follow up question: how big was the budget?", "explicit"),
])
def test_cues_detected(text, kind):
    cue = detect_followup(text)
    assert cue is not None and cue.kind == kind, (text, cue)


@pytest.mark.parametrize("text", [
    "Tell me about a time you led a team.",
    "What is your biggest weakness?",
    "Where do you see yourself in five years?",
    "Can you describe a situation where you handled conflict?",
    "Give me an example of a time you showed leadership.",
    "Thanks for joining us today.",
    "Our company has about two hundred employees and we are growing quickly in the region this year.",
    "Welcome, please have a seat.",
])
def test_not_followups(text):
    cue = detect_followup(text)
    assert cue is None or cue.strength < 0.8, (text, cue)


def test_weak_cues():
    assert detect_followup("And how many people were on it?").kind == "continuation"
    assert detect_followup("How big was that?").kind == "pronoun"
    assert detect_followup("How big was that?").strength < 0.6


# ---------- Conversation ----------

class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def build(matcher, **cfg):
    es = [
        b.Entry("Tell me about a time you led a team", "LEAD", ["describe a time you led a team"], bank="t"),
        b.Entry("How did you motivate the team?", "MOTIVATE", ["how did you keep the team motivated"],
                follows=["Tell me about a time you led a team"], bank="t"),
        b.Entry("What was the result of that project?", "RESULT", ["what was the outcome", "what was the result"],
                follows=["Tell me about a time you led a team"], bank="t"),
        b.Entry("Can you give me an example of that?", "GENERIC-EXAMPLE", ["give me an example", "do you have an example"],
                tags=["followup"], bank="t"),
        b.Entry("What is your biggest weakness?", "WEAK", bank="t"),
        b.Entry("Where do you see yourself in five years?", "FIVE", bank="t"),
    ]
    matcher.load(es)
    c = {"match_threshold": 0.78, "followups": True, "followup_window_seconds": 120, **cfg}
    clock = Clock()
    return Conversation(matcher, c, clock), es, clock


def test_plain_question_shows(matcher):
    conv, es, _ = build(matcher)
    d = conv.resolve("Tell me about a time you led a team.")
    assert d.action == "show" and d.matches[0].entry is es[0] and not d.is_followup
    assert {e.response for e in d.likely} == {"MOTIVATE", "RESULT"}


def test_scripted_followup_after_answer(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("Tell me about a time you led a team.")
    d = conv.resolve("And what was the outcome?")
    assert d.action == "show" and d.matches[0].entry is es[2]
    assert d.is_followup and d.parent is es[0] and "Follow-up" in d.banner


def test_generic_followup(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("Tell me about a time you led a team.")
    d = conv.resolve("Do you have an example?")
    assert d.action == "show" and d.matches[0].entry is es[3] and d.is_followup


def test_new_topic_beats_followup_guess(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("Tell me about a time you led a team.")
    d = conv.resolve("What is your biggest weakness?")
    assert d.action == "show" and d.matches[0].entry is es[4] and not d.is_followup


def test_unscripted_followup_keeps_answer_and_explains(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("Tell me about a time you led a team.")
    d = conv.resolve("Can you elaborate on that?")
    assert d.action == "followup" and d.parent is es[0] and d.cue.kind == "elaborate"
    assert "more detail" in d.banner and d.hint and len(d.likely) == 2


def test_context_expires(matcher):
    conv, es, clock = build(matcher)
    conv.resolve("Tell me about a time you led a team.")
    clock.t += 500
    assert conv.resolve("Can you elaborate on that?").action != "followup"


def test_no_context_no_followup(matcher):
    conv, es, _ = build(matcher)
    assert conv.resolve("Can you elaborate on that?").action != "followup"


def test_forced_followup_button(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("Tell me about a time you led a team.")
    d = conv.resolve("hmm and the budget for it", forced=True)
    assert d.action == "followup" and d.cue.kind == "forced" and d.parent is es[0]


def test_followups_can_be_switched_off(matcher):
    conv, es, _ = build(matcher, followups=False)
    conv.resolve("Tell me about a time you led a team.")
    d = conv.resolve("Can you elaborate on that?")
    assert d.action != "followup"
    d = conv.resolve("What was the outcome?")
    assert not d.is_followup


def test_followup_of_a_followup_chain(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("Tell me about a time you led a team.")
    conv.resolve("What was the outcome?")                 # now RESULT is on screen
    d = conv.resolve("How did you motivate the team?")     # sibling of RESULT, child of LEAD (older parent)
    assert d.action == "show" and d.matches[0].entry is es[1]


def test_early_guess_does_not_poison_state(matcher):
    conv, es, _ = build(matcher)
    d = conv.resolve("Tell me about a time", final=False)          # unfinished: must not set pending
    assert conv._pending is None
    d = conv.resolve("Tell me about a time you led a team.", final=True, utt=1)
    assert d.action == "show" and d.matches[0].entry is es[0]
    assert len(conv._recent) == 1


def test_final_replaces_early_guess_for_same_utterance(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("What is your biggest weakness?", final=False, utt=7)
    conv.resolve("Where do you see yourself in five years?", final=True, utt=7)
    assert [k[1] for k, _, _ in conv._recent] == ["where do you see yourself in five years?"]


def test_fragments_are_glued(matcher):
    conv, es, clock = build(matcher)
    d = conv.resolve("Tell me about")
    assert d.action in ("ignore", "unsure")
    clock.t += 2
    d = conv.resolve("a time you led a team")
    assert d.action == "show" and d.matches[0].entry is es[0]
    clock.t += 10                                          # too late to join
    d = conv.resolve("Tell me about")
    clock.t += 10
    d = conv.resolve("a time you led a team")
    assert d.heard == "a time you led a team"


def test_reload_keeps_context_by_name(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("Tell me about a time you led a team.")
    fresh = [b.Entry(e.question, e.response, list(e.also), list(e.tags), e.skeleton, e.bank, list(e.follows)) for e in es]
    matcher.load(fresh)
    conv.refresh()
    d = conv.resolve("And what was the outcome?")
    assert d.action == "show" and d.matches[0].entry is fresh[2] and d.parent is fresh[0]


def test_pick_by_hand_sets_context(matcher):
    conv, es, _ = build(matcher)
    kids = conv.pick(es[0])
    assert {e.response for e in kids} == {"MOTIVATE", "RESULT"}
    assert conv.resolve("Can you elaborate on that?").parent is es[0]


# ---------- exact control over the similarity numbers ----------

def zbuild(sims, **cfg):
    """Embedder where the utterance ("...outcome...") is one unit vector and each entry sits at a chosen
    cosine to it.  Entries use nonsense words so the lexical bonus can't interfere with the numbers."""
    import numpy as np
    from matcher import Matcher

    def embed(batch):
        out = []
        for i, t in enumerate(batch):
            v = np.zeros(8, dtype=np.float32)
            if "outcome" in t.lower():
                v[0] = 1.0
            elif t in sims:
                v[0], v[1] = sims[t], (1 - sims[t] ** 2) ** 0.5
            else:
                v[2 + (sum(map(ord, t)) % 6)] = 1.0
            out.append(v)
        return np.array(out)

    es = [b.Entry("Zorp alpha one", "PARENT", bank="t"),
          b.Entry("Quux beta two", "KID", follows=["Zorp alpha one"], bank="t"),
          b.Entry("Frob gamma three", "OTHER", bank="t")]
    m = Matcher(embed_fn=embed)
    m.load(es)
    conv = Conversation(m, {"match_threshold": 0.78, "followups": True, "followup_window_seconds": 120, **cfg}, Clock())
    conv.note(es[0])      # the parent answer is on screen
    return conv, es


def test_scripted_followup_gets_a_lower_bar_than_a_stranger():
    # KID sits at 0.74: under the 0.78 bar, over the relaxed 0.71 bar -- but only with a clear cue AND context
    conv, es = zbuild({"Quux beta two": 0.74})
    d = conv.resolve("What was the outcome?")
    assert d.action == "show" and d.matches[0].entry is es[1] and d.is_followup
    conv.reset()                                    # no context: the same words need the full bar
    d = conv.resolve("What was the outcome?")
    assert d.action != "show"


def test_confident_other_question_beats_followup_guess():
    conv, es = zbuild({"Quux beta two": 0.80, "Frob gamma three": 0.92})
    d = conv.resolve("What was the outcome?")
    assert d.matches[0].entry is es[2] and not d.is_followup      # 0.92 vs 0.80+0.08: the stranger wins


def test_boost_breaks_close_calls_in_favour_of_the_scripted_followup():
    conv, es = zbuild({"Quux beta two": 0.82, "Frob gamma three": 0.85})
    d = conv.resolve("What was the outcome?")
    assert d.matches[0].entry is es[1] and d.is_followup           # 0.82+0.08 > 0.85
    conv.reset()
    d = conv.resolve("What was the outcome?")
    assert d.matches[0].entry is es[2]                              # without context, 0.85 wins


def test_weak_cue_gets_a_smaller_nudge():
    conv, es = zbuild({"Quux beta two": 0.82, "Frob gamma three": 0.86})
    d = conv.resolve("What was the outcome?")        # clear cue: 0.82 + 0.08 beats 0.86
    assert d.matches[0].entry is es[1]
    conv.reset(); conv.note(es[0])
    d = conv.resolve("And the outcome of it?")       # continuation-strength cue: 0.82 + 0.032 does not
    assert d.matches[0].entry is es[2]


# ---------- early guesses (made while the speaker may still be talking) ----------

def test_early_guess_needs_a_clearly_complete_clearly_right_match(matcher):
    conv, es, _ = build(matcher)
    d = conv.resolve("Tell me about a time you led a team", final=False)       # no question mark: might be cut off
    assert d.action == "ignore" and not conv._recent
    d = conv.resolve("Tell me about a time you led a team?", final=False, utt=3)
    assert d.action == "show" and d.matches[0].entry is es[0]
    assert len(conv._recent) == 1


def test_held_back_early_guess_leaves_no_trace(matcher):
    conv, es, _ = build(matcher)
    conv.resolve("Can you elaborate on that?", final=False)                     # would be a 'followup' decision if final
    assert conv._pending is None and not conv._recent
