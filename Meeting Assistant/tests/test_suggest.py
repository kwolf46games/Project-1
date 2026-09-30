import numpy as np

import suggest
from suggest import generate, suggest_phrasings
from textsim import lexical_similarity


def texts(question, existing=(), n=8):
    return [s.text for s in suggest_phrasings(question, existing, n=n)]


def test_compound_questions_are_split_and_not_mangled():
    got = texts("In your own words, what does AlphaSights do, and what does an Associate do?")
    assert "What does AlphaSights do?" in got and "What does an Associate do?" in got
    assert not any("does an Associate does" in g for g in got)
    got = texts("Tell me about yourself / walk me through your résumé")
    assert "Tell me about yourself." in got and "Walk me through your résumé." in got
    assert "Why client service?" in texts("Why client service? Why a client-facing, results-driven, commercial environment?")


def test_behavioural_questions_get_the_common_frames():
    got = texts("Tell me about a time you led a team.")
    assert any(g.startswith("Describe a situation where you led a team") for g in got)
    assert any("extra mile" in g for g in texts("Tell me about a time you went above and beyond."))


def test_or_split_reads_naturally():
    got = texts("Tell me about a time you worked under pressure, or a plan changed at the last minute.")
    assert "Tell me about a time when a plan changed at the last minute." in got
    assert not any("a plan changed at the last minute" in g and "a time" not in g for g in got)


def test_names_keep_their_capitals_and_label_prefixes_are_dropped():
    got = texts("College questions: why Penn State?")
    assert "What draws you to Penn State?" in got and not any("College" in g for g in got)
    assert any("AlphaSights" in g for g in texts("Why AlphaSights?"))


def test_digit_variants_but_no_pointless_contraction_swaps():
    assert "Where do you see yourself in 5 years?" in texts("Where do you see yourself in five years?")
    assert "What is a misconception about this role?" not in texts("What’s a misconception about this role?")


def test_curated_rewordings_for_common_questions():
    assert any("greatest weakness" in g for g in texts("What is your biggest weakness?"))
    hire = texts("Why should we hire you?")
    assert "Why should we choose you?" in hire
    assert not any("choose you" in g for g in texts("Why should we not hire you?"))   # must not flip the meaning


def test_existing_wordings_and_the_original_are_never_suggested():
    q = "Why should we hire you?"
    have = ["Why should we choose you?", "What sets you apart from other candidates?"]
    got = texts(q, have)
    assert q not in got and not set(have) & set(got)
    assert all(lexical_similarity(g, q) < 0.93 for g in got)


def test_respects_n_and_has_no_duplicates():
    got = texts("Tell me about a time you led a team.", n=3)
    assert len(got) == 3 == len(set(got))
    assert texts("Zzz?") == [] or isinstance(texts("Zzz?"), list)


def test_embedding_filter_drops_candidates_that_drift_in_meaning(monkeypatch):
    monkeypatch.setattr(suggest, "generate", lambda q: ["Why should we choose you?", "Why is the sky blue?",
                                                        "What sets you apart?"])

    def embed(ts):        # the "sky" candidate points somewhere else entirely
        return np.array([[0, 1.0] if "sky" in t else [1.0, 0.1 * i] for i, t in enumerate(ts)], dtype=np.float32)
    got = [s.text for s in suggest_phrasings("Why should we hire you?", n=5, embed=embed)]
    assert "Why is the sky blue?" not in got and got


def test_embedding_selection_prefers_variety(monkeypatch):
    monkeypatch.setattr(suggest, "generate", lambda q: ["Variant A?", "Variant A again?", "Variant B?"])
    vecs = {"Variant A?": [1, .1], "Variant A again?": [1, .1001], "Variant B?": [.85, .52]}
    embed = lambda ts: np.array([vecs.get(t, [1, 0]) for t in ts], dtype=np.float32)   # noqa: E731
    got = [s.text for s in suggest_phrasings("Original?", n=2, embed=embed)]
    assert got[0] == "Variant A?" and got[1] == "Variant B?"      # not the near-copy of the first pick


def test_embedding_selection_counts_existing_wordings_as_covered(monkeypatch):
    monkeypatch.setattr(suggest, "generate", lambda q: ["Variant A?", "Variant B?"])
    vecs = {"Variant A?": [1, .1], "Variant B?": [.85, .52], "Already have this?": [1, .1002]}
    embed = lambda ts: np.array([vecs.get(t, [1, 0]) for t in ts], dtype=np.float32)   # noqa: E731
    got = [s.text for s in suggest_phrasings("Original?", ["Already have this?"], n=1, embed=embed)]
    assert got == ["Variant B?"]


def test_generate_is_deterministic_and_nonempty_for_real_questions():
    assert generate("Tell me about a failure or setback.") == generate("Tell me about a failure or setback.")
    assert generate("Tell me about a failure or setback.")
