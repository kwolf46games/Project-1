import threading

import numpy as np

import bank as b
from matcher import Matcher, content_tokens, looks_like_question, strip_lead, tidy


def E(q, also=(), **kw):
    return b.Entry(question=q, response="resp:" + q, also=list(also), bank="t", **kw)


def test_tidy_removes_fillers_and_leadins():
    assert tidy("Um, great, thanks. So, tell me about yourself?") == "tell me about yourself?"
    assert tidy("what what is your biggest weakness?") == "what is your biggest weakness?"
    assert tidy("Okay. Right now, what are you looking for?") == "what are you looking for?"
    assert tidy("Do you know how to use Python?") == "Do you know how to use Python?"   # 'you know' kept


def test_strip_lead_never_empties():
    assert strip_lead("Okay") == "Okay"
    assert strip_lead("Well-known tools matter") == "Well-known tools matter"


def test_looks_like_question():
    yes = ["What is your weakness", "so how did you handle it", "Tell me about yourself", "Anything else?",
           "Great. Now, walk me through that project", "I was wondering how you approach testing"]
    no = ["However, I think so.", "Whole teams worked on it", "That sounds good", "Thanks for joining us"]
    for t in yes:
        assert looks_like_question(t), t
    for t in no:
        assert not looks_like_question(t), t


def test_content_tokens_expand_contractions():
    assert content_tokens("What's your biggest weakness?") == content_tokens("What is your biggest weakness")
    assert "the" not in content_tokens("tell me about the team")


def test_exact_and_preamble_match(matcher):
    es = [E("What is your biggest weakness?"), E("Tell me about a time you led a team"), E("Why do you want this job?")]
    matcher.load(es)
    top = matcher.match("What is your biggest weakness?")[0]
    assert top.entry is es[0] and top.score > 0.95
    top = matcher.match("Great, thanks for that. So, why do you want this job?")[0]
    assert top.entry is es[2] and top.score > 0.9


def test_lexical_bonus_separates_lookalikes():
    """Same embedding for everything -> only the word-overlap bonus can tell these apart."""
    m = Matcher(embed_fn=lambda texts: np.ones((len(texts), 8), dtype=np.float32))
    es = [E("Tell me about a time you succeeded"), E("Tell me about a time you failed")]
    m.load(es)
    assert m.match("Tell me about a time you failed")[0].entry is es[1]
    assert m.match("Tell me about a time you succeeded")[0].entry is es[0]


def test_bonus_never_lowers_score(matcher):
    es = [E("What is your biggest weakness?")]
    matcher.load(es)
    sc = matcher.score("What is your biggest weakness?")
    assert (sc.total >= sc.emb - 1e-6).all() and sc.total.max() <= 1.0


def test_boosts_change_rank_not_score(matcher):
    es = [E("How do you handle conflict?"), E("How do you handle pressure?")]
    matcher.load(es)
    base = matcher.match("how do you handle stress")
    j = matcher.index_of(es[1] if base[0].entry is es[0] else es[0])
    boosted = matcher.match("how do you handle stress", boosts={j: 0.5})
    assert boosted[0].entry is matcher.entries[j] and boosted[0].boost == 0.5
    assert abs(boosted[0].score - next(m.score for m in base if m.entry is boosted[0].entry)) < 1e-6


def test_reload_reuses_embeddings(matcher, embed_calls):
    es = [E("What is your biggest weakness?", ["what's your greatest weakness"]), E("Why this company?")]
    matcher.load(es)
    n = len(embed_calls)
    matcher.load(es + [E("Where do you see yourself in five years?")])
    assert len(embed_calls) == n + 1 and embed_calls[-1] == ["Where do you see yourself in five years?"]


def test_query_cache_skips_repeat_work(matcher, embed_calls):
    matcher.load([E("What is your biggest weakness?")])
    matcher.match("what is your biggest strength")
    n = len(embed_calls)
    matcher.match("what is your biggest strength")
    assert len(embed_calls) == n


def test_empty_and_unloaded(matcher):
    assert matcher.match("hello there") == []
    matcher.load([E("What is your biggest weakness?")])
    assert matcher.match("") == [] and matcher.match("um uh") == []


def test_concurrent_match_and_reload(matcher):
    es = [E(f"Question number {i} about topic {i}?") for i in range(30)]
    matcher.load(es)
    errs = []

    def hammer():
        try:
            for i in range(60):
                matcher.match(f"question number {i % 30} about topic")
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    def reload():
        try:
            for _ in range(10):
                matcher.load(es[: 20 + _])
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=hammer) for _ in range(3)] + [threading.Thread(target=reload)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs, errs


def test_bank_vocabulary_picks_names_and_jargon_not_answers():
    from matcher import bank_vocabulary
    es = [E("Tell me about your experience with Kubernetes and Terraform"),
          E("How do you test a REST API?", ["what do you know about GraphQL"]),
          E("Why do you want to work at Acme Corp?"),
          E("What is your biggest weakness?")]
    es[0].response = "I used Docker and Jenkins daily"          # answers are not mined
    v = bank_vocabulary(es)
    assert {"Kubernetes", "Terraform", "REST", "API", "GraphQL", "Acme", "Corp"} <= set(v)
    assert "Docker" not in v and "Tell" not in v and "What" not in v and len(v) <= 12
    assert bank_vocabulary([]) == []
