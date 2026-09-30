from bank import Entry
from matcher import Match, Matcher, candidates, looks_like_question, question_tails, split_sentences
from conftest import fake_embed


def test_match_ranks_the_right_question_first(matcher):
    top = matcher.match("Tell me about a time you led a team.", k=3)
    assert top[0].entry.question == "Tell me about a time you led a team."
    assert top[0].score > 0.95 and len(top) == 3
    assert top[0].score >= top[1].score >= top[2].score


def test_match_uses_other_wordings(matcher):
    top = matcher.match("what is your greatest weakness", k=1)[0]
    assert top.entry.question == "What is your biggest weakness?"
    assert top.phrasing == "What's your greatest weakness?"


def test_match_finds_the_question_inside_a_long_utterance(matcher):
    text = "Great, thanks for walking me through that. Now tell me about a failure or setback."
    assert matcher.match(text, k=1)[0].entry.question == "Tell me about a failure or setback."


def test_match_tolerates_transcription_noise(matcher):
    noisy = "tell me about a time you lead a team"
    assert matcher.match(noisy, k=1)[0].entry.question == "Tell me about a time you led a team."
    hit = matcher.match("Tell me about a failure or a set back.", k=1)[0]
    assert hit.entry.question == "Tell me about a failure or setback." and hit.lexical > 0.8


def test_lexical_bonus_is_bounded_and_score_capped(matcher):
    for m in matcher.match("Why should we hire you?", k=5):
        assert 0.0 <= m.score <= 1.0
        assert m.score - m.dense <= 0.0601


def test_negation_keeps_opposite_questions_apart(matcher):
    assert matcher.match("Why should we not hire you?", k=1)[0].entry.question == "Why should we not hire you?"
    assert matcher.match("Why should we hire you?", k=1)[0].entry.question == "Why should we hire you?"


def test_empty_inputs(matcher):
    assert matcher.match("") == [] and matcher.match("   ") == []
    empty = Matcher(embed=fake_embed)
    assert empty.match("anything") == [] and empty.match_followups("anything") == [] and empty.entries == []
    empty.load([])
    assert empty.match("anything") == []


def test_match_followups_finds_prepared_ones(matcher):
    top = matcher.match_followups("how big was the team", k=2)
    assert top[0].followup.question == "How big was the team?"
    assert top[0].entry.question == "Tell me about a time you led a team."
    assert top[0].score > 0.9


def test_reload_replaces_the_index(matcher):
    matcher.load([Entry("Why are you here?", "Because.")])
    assert [e.question for e in matcher.entries] == ["Why are you here?"]
    assert matcher.match_followups("how big was the team") == []


def test_match_stays_constructible_positionally():
    e = Entry("Q?", "A")
    m = Match(e, 0.5)
    assert m.phrasing == "" and m.lexical == 0.0


def test_looks_like_question():
    assert looks_like_question("Tell me about yourself.") and looks_like_question("Tell us about yourself")
    assert looks_like_question("Great, thanks. So what was the hardest part") and looks_like_question("anything?")
    assert looks_like_question("I was wondering how you handled it")
    assert not looks_like_question("However, we are still hiring.")
    assert not looks_like_question("Whatever happens, we will call you.")


def test_question_tails_and_candidates():
    text = "Thanks for that. Okay. Now tell me about a failure or setback."
    assert "tell me about a failure or setback." in question_tails(text)
    assert "tell me about a failure or setback." in [c.lower() for c in candidates(text)]
    assert split_sentences("One two. Three four? Five") == ["One two.", "Three four?"]
