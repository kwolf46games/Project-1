from transcript import (Vocabulary, Word, align_words, clean_text, is_hallucination, looks_incomplete,
                        mean_confidence)


def test_clean_text_removes_fillers_stutters_and_loops():
    assert clean_text("Um, so tell me, uh, about about yourself") == "So tell me, about yourself"
    assert clean_text("I I think we we should go") == "I think we should go"
    assert clean_text("the the the the the the") == "The"
    assert clean_text("Tell me about yourself .") == "Tell me about yourself."


def test_clean_text_keeps_real_words_containing_filler_letters():
    assert clean_text("Her umbrella is here") == "Her umbrella is here"


def test_hallucinations_dropped_but_real_speech_kept():
    assert is_hallucination("Thanks for watching!")
    assert is_hallucination("", 1.0)
    assert is_hallucination("Thank you.", duration=1.0, confidence=0.3)
    assert not is_hallucination("Thank you.", duration=1.0, confidence=0.95)
    assert is_hallucination("hello there", no_speech=0.9, avg_logprob=-1.5)
    assert not is_hallucination("Tell me about yourself.", duration=3, confidence=0.9)


def test_incomplete_detection():
    assert looks_incomplete("Tell me about a time when you")
    assert looks_incomplete("So, can you walk me through,")
    assert looks_incomplete("And then the")
    assert not looks_incomplete("Tell me about yourself.")
    assert not looks_incomplete("How are you?")
    assert not looks_incomplete("")


def test_vocabulary_extracts_names_not_formatting(entries):
    v = Vocabulary.from_entries(entries)
    assert "NovaCore" in v.terms and "SLA" in v.terms
    for junk in ("SCRIPT", "LIKELY", "SPOKEN", "ABOUT", "SAY", "THIS"):
        assert junk not in v.terms


def test_prompt_mixes_base_prompt_and_terms_within_budget():
    v = Vocabulary(["NovaCore", "SLA", "Orbitly"])
    p = v.prompt("BREAD")
    assert p.startswith("BREAD.") and "NovaCore" in p and "SLA" in p
    assert "BREAD" not in p.split(":", 1)[1]          # not repeated in the list
    assert len(Vocabulary(["X" + str(i) * 10 for i in range(200)]).prompt("", max_chars=100)) < 160
    assert Vocabulary().prompt("") == ""


def _words(text, prob=None):
    return [Word(t, prob=prob) for t in text.split()]


def test_correction_restores_case_and_joins_split_names():
    v = Vocabulary(["NovaCore", "Penn State"])
    out, changes = v.correct(_words("why nova core and novacore at penn state?", 0.9))
    assert " ".join(w.text for w in out) == "why NovaCore and NovaCore at Penn State?"
    assert changes


def test_correction_fixes_unsure_words_but_not_confident_ones():
    v = Vocabulary(["NovaCore", "Kubernetes"])
    unsure, _ = v.correct(_words("tell me about Novacor and cubernetes", 0.4))
    assert [w.text for w in unsure][-3:] == ["NovaCore", "and", "Kubernetes"]
    sure, changes = v.correct([Word("Novacor", prob=0.97)])
    assert sure[0].text == "Novacor" and not changes


def test_correction_leaves_ordinary_words_alone():
    v = Vocabulary(["NovaCore", "Orbitly", "Kubernetes"])
    text = "tell me about a score you improved at the core of your work"
    out, changes = v.correct(_words(text, 0.3))
    assert " ".join(w.text for w in out) == text and not changes


def test_correction_keeps_punctuation_and_confidence():
    v = Vocabulary(["NovaCore"])
    out, _ = v.correct([Word("Why", prob=.9), Word("novacore?", 1.0, 2.0, .4)])
    assert out[1].text == "NovaCore?" and out[1].prob == .4 and out[1].start == 1.0


def test_align_words_survives_removed_fillers():
    words = [Word("um", 0, .2, .3), Word("tell", .2, .4, .9), Word("me", .4, .5, .8), Word("more", .5, .8, .6)]
    out = align_words("Tell me more", words)
    assert [w.text for w in out] == ["Tell", "me", "more"]
    assert [w.prob for w in out] == [.9, .8, .6]
    assert round(mean_confidence(out), 2) == 0.77


def test_repair_never_swallows_neighbouring_words():
    v = Vocabulary(["AlphaSights", "Guidepoint", "TAMID"])
    out, _ = v.correct(_words("why alpha sights and tammid", 0.4))
    assert " ".join(w.text for w in out) == "why AlphaSights and TAMID"
    out, _ = v.correct(_words("what about guide point and alphasites", 0.4))
    assert " ".join(w.text for w in out) == "what about Guidepoint and AlphaSights"


def test_a_misheard_word_inside_a_longer_name_is_repaired_with_the_name():
    v = Vocabulary(["Smeal College", "Penn State"])
    out, _ = v.correct(_words("about the smile college of business at penn state", 0.3))
    assert " ".join(w.text for w in out) == "about the Smeal College of business at Penn State"
    out, _ = v.correct(_words("a smile on the college campus", 0.3))        # ordinary words are left alone
    assert " ".join(w.text for w in out) == "a smile on the college campus"


def test_ordinary_words_in_the_vocabulary_are_not_recapitalised():
    v = Vocabulary(["Head", "NovaCore", "Commercial Operations"])
    out, changes = v.correct(_words("use your head at novacore", 0.9))
    assert " ".join(w.text for w in out) == "use your head at NovaCore"


def test_vocabulary_ignores_headings_and_sentence_openers():
    text = ("**SCRIPT  ·  ABOUT 50 SECONDS SPOKEN** Yeah, I’m at Orbitly. Probably the best part is NovaCore. "
            "I’d say it works. Honestly it does. VALUE CHAIN YOU CAN TALK THROUGH")
    v = Vocabulary.from_entries([__import__("bank").Entry("Why Orbitly?", text)])
    assert "Orbitly" in v.terms and "NovaCore" in v.terms
    for junk in ("I", "I’m", "I’d", "Yeah", "Probably", "Honestly", "VALUE", "SPOKEN", "SECONDS", "THROUGH"):
        assert junk not in v.terms, junk
