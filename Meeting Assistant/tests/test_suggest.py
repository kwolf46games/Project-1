import numpy as np
import pytest

import bank as b
from matcher import Matcher
from suggest import Suggestion, generate, suggest_phrasings, vet


def texts(q, **kw):
    return suggest_phrasings(q, **kw)


def test_basic_shape():
    out = generate("What is your biggest weakness?", limit=8)
    assert 4 <= len(out) <= 8
    assert {s.source for s in out} >= {"intent", "pattern", "synonym"}
    assert len({s.text.lower() for s in out}) == len(out)                    # no repeats
    assert all(len(s.text.split()) >= 2 for s in out)


def test_never_suggests_what_is_known():
    q = "What is your biggest weakness?"
    first = texts(q, limit=20)
    assert q not in first and "What is your biggest weakness" not in first
    known = first[:3]
    again = texts(q, existing=known, limit=20)
    assert not set(known) & set(again)
    assert "What's your biggest weakness?" not in texts(q, existing=["What's your biggest weakness"], limit=20)


def test_questions_end_with_question_marks_statements_do_not():
    for t in texts("What is your biggest weakness?", limit=20):
        asks = t.split()[0].lower() in {"what", "can", "could", "how", "why", "where", "would", "is", "do", "have"}
        if asks and not t.lower().startswith(("i'd like to know", "i'm curious", "tell me")):
            assert t.endswith("?"), t
        if t.lower().startswith(("tell me", "i'd like", "i'm curious", "describe", "walk me", "talk to me", "please")):
            assert not t.endswith("?"), t


@pytest.mark.parametrize("q,expect", [
    ("Tell me about yourself", "Can you introduce yourself?"),
    ("What is your biggest weakness?", "Can you tell me what your biggest weakness is?"),
    ("How do you handle conflict?", "Tell me how you handle conflict"),
    ("How do you handle conflict?", "How do you deal with conflict?"),
    ("Tell me about a time you led a team", "Describe a time you led a team"),
    ("Why did you choose this field?", "What made you choose this field?"),
    ("Do you have experience with SQL?", "Have you worked with SQL?"),
    ("What would you do if a client complained?", "If a client complained, what would you do?"),
    ("Can you walk me through your resume?", "Walk me through your resume"),
    ("Describe your leadership style", "How would you describe yourself as a leader?"),
])
def test_expected_wordings_present(q, expect):
    assert expect in texts(q, limit=30), texts(q, limit=30)


@pytest.mark.parametrize("q,banned", [
    ("Why did you leave your last job?", "What interests you about this role?"),
    ("Tell me about a time you led a team", "Describe your leadership style"),
    ("Tell me about a time you led a team", "Tell me about a time you worked on a team"),
    ("How would you explain an API to a non-technical person?", "Can you explain how you would explain an API to a non-technical person?"),
    ("Tell me about yourself", "Share yourself"),
    ("Tell me about yourself", "Walk me through yourself"),
])
def test_known_bad_suggestions_are_gone(q, banned):
    assert banned not in texts(q, limit=40)


def test_unknown_question_still_gets_structural_help_and_junk_gets_nothing():
    assert texts("What kind of projects do you enjoy?", limit=10)
    assert generate("Hi") == [] and generate("") == []


def test_no_ungrammatical_pronoun_inversions():
    for q in ["What is it like to work here?", "What is that about?", "What was there before?"]:
        assert not any("what it like to work here is" in t.lower() or "what that about is" in t.lower() for t in texts(q, limit=20))


# ---------- vet(), against the matcher ----------

def make(embed):
    m = Matcher(embed_fn=embed)
    es = [b.Entry("What is your biggest weakness?", "W", bank="t"),
          b.Entry("What is your biggest strength?", "S", bank="t"),
          b.Entry("Tell me about yourself", "Y", bank="t")]
    m.load(es)
    return m, es


def table_embed(table, default_axis=5):
    def embed(batch):
        out = np.zeros((len(batch), 8), dtype=np.float32)
        for i, t in enumerate(batch):
            v = table.get(t)
            if v is None:
                out[i, default_axis] = 1.0
            else:
                out[i, : len(v)] = v
        return out
    return embed


def test_vet_drops_distant_and_flags_rivals():
    W, S = [1, 0, 0], [0.8, 0.6, 0]                                  # two similar questions (cos 0.8)
    table = {
        "What is your biggest weakness?": W, "What is your biggest strength?": S, "Tell me about yourself": [0, 0, 1],
        "Close paraphrase of weakness": [0.99, 0.02, 0],             # clearly W, not S -> kept, no rival
        "Off topic wording": [0.3, 0, 0.95],                         # far from W -> dropped
        "Ambiguous wording": [0.85, 0.52, 0],                        # still >0.8 to W but closer to S -> rival
    }
    m, es = make(table_embed(table))
    sugg = [Suggestion("Close paraphrase of weakness", "pattern"), Suggestion("Off topic wording", "pattern"),
            Suggestion("Ambiguous wording", "intent")]
    out = vet(sugg, es[0], m, min_sim=0.8)
    names = {s.text: s for s in out}
    assert "Off topic wording" not in names
    good = names["Close paraphrase of weakness"]
    assert good.rival == "" and good.ok and good.sim > 0.99
    risky = names["Ambiguous wording"]
    assert risky.rival == "What is your biggest strength?" and not risky.ok and 0.8 <= risky.sim < 0.9


def test_vet_works_for_an_entry_not_yet_saved():
    table = {"What is your biggest weakness?": [1, 0, 0], "What is your biggest strength?": [0, 1, 0],
             "Tell me about yourself": [0, 0, 1], "Brand new question": [0, 0.5, 0.87],
             "Brand new wording": [0, 0.1, 0.99]}
    m, _ = make(table_embed(table))
    new = b.Entry("Brand new question", "", bank="t")                   # not in the matcher's index
    out = vet([Suggestion("Brand new wording", "pattern")], new, m)
    # the wording is closer to the existing "Tell me about yourself" than to the new question -> flagged
    assert out and out[0].rival == "Tell me about yourself"


def test_vet_empty():
    m, es = make(table_embed({}))
    assert vet([], es[0], m) == []
