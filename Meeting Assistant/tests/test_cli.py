import app
import bank as b
import matcher
from conftest import BANK_MD, fake_embed


def _patch_matcher(monkeypatch, entries):
    real = matcher.Matcher
    monkeypatch.setattr(matcher, "Matcher", lambda: real(embed=fake_embed))
    monkeypatch.setattr(b, "active_entries", lambda cfg=None: entries)
    monkeypatch.setattr(b, "load_config", lambda: {**b.DEFAULT_CONFIG, "active_banks": ["test"]})


def test_test_command_reports_followups(monkeypatch, capsys, entries):
    _patch_matcher(monkeypatch, entries)
    app.main(["test", "How big was the team?", "--after", "Tell me about a time you led a team."])
    out = capsys.readouterr().out
    assert "FOLLOW-UP" in out and "Tell me about a time you led a team." in out and "Six people" in out
    app.main(["test", "What was the result?", "--after", "Tell me about a time you led a team."])
    out = capsys.readouterr().out
    assert "FOLLOW-UP (asking about the result)" in out and "from your answer:" in out


def test_test_command_still_reports_plain_matches(monkeypatch, capsys, entries):
    _patch_matcher(monkeypatch, entries)
    app.main(["test", "Why should we hire you?"])
    out = capsys.readouterr().out
    assert "question-like: True" in out and "NEW QUESTION: Why should we hire you?" in out and "FOLLOW-UP" not in out


def _bank_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(app, "_suggest_embedder", lambda: None)
    (tmp_path / "test.md").write_text(BANK_MD, encoding="utf-8")


def test_q_suggest_previews_then_applies(tmp_path, monkeypatch, capsys):
    _bank_dir(tmp_path, monkeypatch)
    app.main(["q", "suggest", "test", "1"])
    out = capsys.readouterr().out
    assert "Nothing saved" in out and "  + " in out
    assert b.load_bank("test").entries[0].also == []
    app.main(["q", "suggest", "test", "Why should we hire you?", "-n", "2", "--apply"])
    saved = b.load_bank("test").entries[1]
    assert len(saved.also) == 2 and saved.question == "Why should we hire you?"
    assert "LIKELY FOLLOW-UPS" in saved.response                 # the answer and its follow-ups are untouched


def test_banks_suggest_only_touches_thin_questions(tmp_path, monkeypatch, capsys):
    _bank_dir(tmp_path, monkeypatch)
    app.main(["banks", "suggest", "test", "--min", "1", "--apply"])
    bk = b.load_bank("test")
    assert bk.entries[3].also == ["What's your greatest weakness?"]    # already had one: skipped
    assert all(e.also for i, e in enumerate(bk.entries) if i != 3 and e.question != "Why should we not hire you?")
    assert "Added" in capsys.readouterr().out
