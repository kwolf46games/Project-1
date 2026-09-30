from textsim import lexical_similarity as L, normalize, phonetic_key


def test_normalize_expands_contractions_and_strips_accents():
    assert normalize("What’s your résumé?") == "what your resume"
    assert normalize("Don't") == "do not"


def test_paraphrase_scores_high_and_different_question_scores_low():
    assert L("Tell me about a time you led a team.", "tell me about a time when you led a team") > 0.75
    assert L("Tell me about a time you led a team.", "Tell me about a time you faced rejection.") < 0.4


def test_split_and_joined_words_still_match():
    assert L("Why AlphaSights?", "why alpha sights") > 0.9
    assert L("Tell me about a failure or setback.", "tell me about a failure or a set back") > 0.9


def test_negation_is_not_blurred():
    assert L("Why should we hire you?", "Why should we not hire you?") < 0.6


def test_digits_and_words():
    assert L("Where do you see yourself in five years?", "where do you see yourself in 5 years") > 0.7


def test_phonetic_key_groups_soundalikes():
    assert phonetic_key("Kubernetes") == phonetic_key("cubernetes")
    assert phonetic_key("TAMID") == phonetic_key("tameed")
    assert phonetic_key("Smeal") != phonetic_key("meal")
