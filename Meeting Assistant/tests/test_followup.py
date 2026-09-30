import bank as b
from followup import detect, focus_passages, prepared_followups, script_text


def test_prepared_followups_parsed_from_bank_text(entries):
    fus = prepared_followups(entries[0])
    assert [f.question for f in fus] == ["How big was the team?", "Tell me more about the rollback plan."]
    assert [f.seconds for f in fus] == [10, 20]
    assert fus[0].answer.startswith("Six people")
    assert fus[1].answer.endswith("Every step had an owner.")      # an answer that wrapped onto a second line


def test_request_style_followups_need_no_question_mark(entries):
    assert not prepared_followups(entries[0])[1].question.endswith("?")


def test_entry_without_followups_and_section_ends_at_next_header():
    e = b.Entry("Q?", "**LIKELY FOLLOW-UPS  ·  SAY THIS**\n\n**Why?**  ·  about 5 sec Because.\n\n"
                "**HIRING MANAGER’S NOTE** not a follow-up\n**Also?**  ·  about 5 sec nope")
    assert [f.question for f in prepared_followups(e)] == ["Why?"]
    assert prepared_followups(b.Entry("Q?", "just an answer")) == []


def test_script_text_is_only_what_you_would_say(entries):
    s = script_text(entries[0])
    assert s.startswith('"At NovaCore I led a team')
    assert "listening for" not in s and "How big was the team" not in s and "**" not in s


def test_detect_recognises_generic_probes():
    for text, kind in [("Tell me more about that.", "elaborate"), ("What was the result?", "outcome"),
                       ("What was your specific role?", "role"), ("Walk me through that number.", "number"),
                       ("How did people react?", "reaction"), ("What would you do differently?", "hindsight"),
                       ("Give me an example of that.", "example"), ("Can you elaborate?", "elaborate"),
                       ("What do you mean by that?", "clarify"), ("You mentioned a rollback plan. Why?", "back"),
                       ("And what was the outcome?", "outcome"), ("Were you the leader?", "role")]:
        sig = detect(text)
        assert sig and sig.kind == kind, (text, sig)


def test_detect_elliptical_questions():
    for text in ("Why?", "How so?", "And then?", "Such as?"):
        assert detect(text), text


def test_detect_leaves_standalone_questions_alone():
    for text in ("Tell me about yourself.", "Why should we hire you?", "Where do you see yourself in five years?",
                 "What is your biggest weakness?", "Tell me about a time you led a team.", "", "We start at noon."):
        assert not detect(text), text


def test_detect_reports_cues_and_label():
    sig = detect("Tell me more about that.")
    assert sig.cues and sig.label


def test_focus_passages_pick_the_relevant_sentences(entries):
    result = focus_passages(entries[0], "What was the result?", "outcome")
    assert any("uptime rose by 20%" in s or "finished two weeks early" in s for s in result)
    role = focus_passages(entries[0], "What was your role?", "role")
    assert any("I set the plan" in s or "I led a team" in s or "I wrote" in s for s in role)
    assert len(result) <= 2


def test_focus_passages_keep_spoken_order_and_handle_empty(entries):
    got = focus_passages(entries[0], "what was the result and what did you do", "outcome", k=3)
    script = script_text(entries[0])
    assert got == sorted(got, key=script.index)
    assert focus_passages(b.Entry("Q?", ""), "anything", "") == []
