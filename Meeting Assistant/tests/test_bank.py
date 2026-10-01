import csv

import pytest

import bank as b

OLD_FORMAT = """# Practice questions

## Tell me about yourself
also: walk me through your background | who are you
tags: intro
skeleton: past -> present -> future

I'm a **developer** with {{N}} years of experience.
Second paragraph.

## What is your biggest weakness?

Plain answer, no metadata.
"""


def test_old_banks_load_and_resave_unchanged(banks_dir):
    banks_dir.mkdir(parents=True)
    (banks_dir / "old.md").write_text(OLD_FORMAT, encoding="utf-8")
    bk = b.load_bank("old")
    assert bk.title == "Practice questions" and len(bk.entries) == 2
    first = bk.entries[0]
    assert first.also == ["walk me through your background", "who are you"] and first.tags == ["intro"]
    assert first.follows == [] and not first.is_generic_followup
    assert "I'm a **developer**" in first.response and "Second paragraph." in first.response
    bk.save()
    again = b.load_bank("old")
    assert [(e.question, e.also, e.tags, e.skeleton, e.response, e.follows) for e in again.entries] == \
           [(e.question, e.also, e.tags, e.skeleton, e.response, e.follows) for e in bk.entries]
    assert "follows:" not in (banks_dir / "old.md").read_text(encoding="utf-8")


def test_follows_round_trip_through_markdown(banks_dir):
    bk = b.Bank("t", "T", [
        b.Entry("Tell me about yourself", "Y", bank="t"),
        b.Entry("Why that school?", "S", follows=["Tell me about yourself"], bank="t"),
        b.Entry("Can you give an example?", "E", ["for instance?"], follows=["A | B".split(" | ")[0], "B"], tags=["Follow-up"], bank="t"),
    ])
    bk.save()
    text = (banks_dir / "t.md").read_text(encoding="utf-8")
    assert "follows: Tell me about yourself\n" in text and "follows: A | B\n" in text
    again = b.load_bank("t")
    assert again.entries[1].follows == ["Tell me about yourself"]
    assert again.entries[2].follows == ["A", "B"] and again.entries[2].also == ["for instance?"]
    assert again.entries[2].is_generic_followup                       # "Follow-up" tag, any spelling/case


@pytest.mark.parametrize("tag", ["followup", "FollowUp", "follow-up", "Follow Up", "follow-ups"])
def test_generic_followup_tag_spellings(tag):
    assert b.Entry("Q", tags=["intro", tag]).is_generic_followup
    assert not b.Entry("Q", tags=["intro"]).is_generic_followup


def test_metadata_order_does_not_matter_when_reading():
    text = "## Q one\nskeleton: s\nfollows: P\ntags: t\nalso: a | b\n\nbody\n"
    e = b.parse_markdown(text, "x").entries[0]
    assert (e.skeleton, e.follows, e.tags, e.also, e.response) == ("s", ["P"], ["t"], ["a", "b"], "body")


def test_csv_import_with_follows_column(tmp_path):
    f = tmp_path / "q.csv"
    with f.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Question", "Answer", "Other phrasings", "Follow-up to", "Tags"])
        w.writerow(["Tell me about yourself", "A1", "who are you", "", "intro"])
        w.writerow(["Why that school?", "A2", "", "Tell me about yourself", ""])
    bk = b.import_file(f, "q")
    assert [e.question for e in bk.entries] == ["Tell me about yourself", "Why that school?"]
    assert bk.entries[1].follows == ["Tell me about yourself"] and bk.entries[0].follows == []
    assert bk.entries[0].also == ["who are you"] and bk.entries[0].tags == ["intro"]


def test_csv_import_without_headers_still_works(tmp_path):
    f = tmp_path / "plain.csv"
    f.write_text("Why this job?,Because.\nWhat is your goal?,To grow.\n", encoding="utf-8")
    bk = b.import_file(f, "plain")
    assert [(e.question, e.response, e.follows) for e in bk.entries] == [("Why this job?", "Because.", []), ("What is your goal?", "To grow.", [])]


def test_xlsx_import_with_follows(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Question", "Response", "Parent question"])
    ws.append(["Tell me about yourself", "A1", None])
    ws.append(["Why that school?", "A2", "Tell me about yourself"])
    f = tmp_path / "q.xlsx"
    wb.save(f)
    bk = b.import_file(f, "q")
    assert bk.entries[1].follows == ["Tell me about yourself"]


def test_markdown_with_follows_is_recognised_as_this_apps_format(tmp_path):
    f = tmp_path / "q.md"
    f.write_text("## A?\n\nans\n\n## B?\nfollows: A?\n\nans2\n", encoding="utf-8")
    bk = b.import_file(f, "q")
    assert bk.entries[1].follows == ["A?"] and bk.entries[0].response == "ans"


def test_rename_question_relinks_followups_case_insensitively():
    bk = b.Bank("t", entries=[b.Entry("Tell me about yourself"), b.Entry("Why?", follows=["tell me ABOUT yourself", "Other"])])
    bk.rename_question("Tell me about yourself", "Introduce yourself")
    assert bk.entries[1].follows == ["Introduce yourself", "Other"]


def test_teach_phrasing(banks_dir):
    banks_dir.mkdir(parents=True)
    b.Bank("t", "T", [b.Entry("What is your biggest weakness?", "W", bank="t")]).save()
    entry = b.load_bank("t").entries[0]
    assert b.teach_phrasing(entry, "  what do you struggle   with the most ") is True
    assert entry.also == ["what do you struggle with the most"]                       # whitespace tidied, in-memory copy updated
    assert b.load_bank("t").entries[0].also == ["what do you struggle with the most"]
    assert b.teach_phrasing(entry, "What do you struggle with the most") is False        # known, any case
    assert b.teach_phrasing(entry, "WHAT IS YOUR BIGGEST WEAKNESS?") is False            # same as the question
    assert b.teach_phrasing(entry, "   ") is False
    b.Bank("t", "T", []).save()
    with pytest.raises(KeyError):
        b.teach_phrasing(entry, "something new")                                         # question was deleted meanwhile


def test_config_defaults_fill_in_new_keys(banks_dir):
    banks_dir.parent.mkdir(parents=True, exist_ok=True)
    b.CONFIG_PATH.write_text('{"active_banks": ["x"], "whisper_model": "base.en", "whisper_prompt": "BREAD"}', encoding="utf-8")
    cfg = b.load_config()
    assert cfg["active_banks"] == ["x"] and cfg["whisper_prompt"] == "BREAD"
    assert cfg["followups"] is True and cfg["early_silence_seconds"] == 0.35 and cfg["reconnect_idle_seconds"] == 45
    assert cfg["match_threshold"] == 0.78 and cfg["silence_seconds"] == 0.7              # untouched defaults


def test_slug_and_names_unchanged(banks_dir):
    assert b.slug("My Bank #1!") == "my-bank-1"
    with pytest.raises(ValueError):
        b.slug("!!!")
