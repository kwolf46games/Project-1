import bank as b


def _blocks():
    return [(2, "Q10. Why should we hire you?"), (None, "a10"),
            (2, "Q11. Why should we not hire you?"), (3, "“Why should we not hire you?”"), (None, "a11"),
            (3, "Q12 What is your biggest weakness?"), (None, "a12"),
            (2, "Q13. Where do you see yourself in five years?"), (None, "a13"),
            (2, "Q15. College questions: why finance?         FIXXXX"), (None, "a15")]


def test_numbered_question_one_heading_level_down_is_its_own_entry():
    bank = b._blocks_to_bank(_blocks(), "t")
    assert [e.question for e in bank.entries][:3] == ["Why should we hire you?", "Why should we not hire you?",
                                                      "What is your biggest weakness?"]
    by_q = {e.question: e.response for e in bank.entries}
    assert "a12" not in by_q["Why should we not hire you?"] and by_q["What is your biggest weakness?"] == "a12"
    assert "Why should we not hire you?" in by_q["Why should we not hire you?"]   # an unnumbered restatement stays as body


def test_todo_tags_are_stripped_from_question_text_but_the_answer_is_kept():
    bank = b._blocks_to_bank(_blocks(), "t")
    assert bank.entries[-1].question == "College questions: why finance?" and bank.entries[-1].response == "a15"
    assert b._clean_question("Q3. Why us?  TODO") == "Why us?" and b._clean_question("Why FIXME now?") == "Why FIXME now?"


def test_unnumbered_deeper_headings_are_still_just_body_text():
    blocks = [(2, "Q1. Why should we hire you?"), (3, "Sub-heading"), (None, "x"), (2, "Q2. Why us?"), (None, "y"),
              (2, "Q3. Why now?"), (None, "z")]
    assert [e.question for e in b._blocks_to_bank(blocks, "t").entries] == ["Why should we hire you?", "Why us?", "Why now?"]
