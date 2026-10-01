import wave
from types import SimpleNamespace

import numpy as np
import pytest

import app
import audio
import bank as b
import matcher as matcher_mod
from conftest import fake_embed


@pytest.fixture
def cli(banks_dir, monkeypatch):
    banks_dir.mkdir(parents=True)
    b.Bank("demo", "Demo", [
        b.Entry("Tell me about a time you led a team", "LEAD", ["describe a time you led a team"], bank="demo"),
        b.Entry("What was the result of that project?", "RESULT", ["what was the outcome"],
                follows=["Tell me about a time you led a team"], bank="demo"),
        b.Entry("What is your biggest weakness?", "WEAK", bank="demo"),
    ]).save()
    b.set_active("demo", True)
    monkeypatch.setattr(matcher_mod, "_load_fastembed", lambda: fake_embed)
    return banks_dir


def test_banks_show_marks_followups(cli, capsys):
    app.main(["banks", "show", "demo"])
    out = capsys.readouterr().out
    assert "(follows: Tell me about a time you led a team)" in out and "  3. What is your biggest weakness?" in out


def test_q_add_and_edit_with_follows_and_rename(cli, capsys):
    app.main(["q", "add", "demo", "--question", "How did you motivate them?", "--response", "R", "--also", "keep them motivated",
              "--follows", "Tell me about a time you led a team"])
    assert b.load_bank("demo").entries[-1].follows == ["Tell me about a time you led a team"]
    app.main(["q", "edit", "demo", "1", "--question", "Describe a time you led a team"])
    bk = b.load_bank("demo")
    assert bk.entries[0].question == "Describe a time you led a team"
    assert all(e.follows == ["Describe a time you led a team"] for e in bk.entries if e.follows)     # relinked
    app.main(["q", "edit", "demo", "2", "--follows", ""])
    assert b.load_bank("demo").entries[1].follows == []


def test_test_command_plain_and_as_followup(cli, capsys):
    app.main(["test", "What is your biggest weakness?"])
    out = capsys.readouterr().out
    assert "follow-up wording: none" in out and "overlay would: show the answer" in out
    app.main(["test", "And what was the outcome?", "--after", "led a team"])
    out = capsys.readouterr().out
    assert "pretending this answer is on screen: Tell me about a time you led a team" in out
    assert "follow-up wording: outcome" in out and "(as a follow-up)" in out and "Follow-up" in out
    app.main(["test", "Can you elaborate on that?", "--after", "led a team"])
    out = capsys.readouterr().out
    assert "keep the current answer and flag a follow-up" in out and "more detail" in out
    with pytest.raises(SystemExit) as e:
        app.main(["test", "x", "--after", "no such question"])
    assert "matches 0 active questions" in str(e.value)


def test_suggest_command_lists_and_applies(cli, capsys):
    app.main(["suggest", "demo", "weakness"])
    out = capsys.readouterr().out
    assert "What is your biggest weakness?" in out and "  + " in out and "--apply" in out
    before = b.load_bank("demo").entries[2].also[:]
    assert before == []
    app.main(["suggest", "demo", "weakness", "--apply", "--no-check", "--max", "3"])
    out = capsys.readouterr().out
    assert "Added 3 phrasings" in out
    assert len(b.load_bank("demo").entries[2].also) == 3
    app.main(["suggest", "demo", "--no-check", "--apply", "--max", "2"])                     # whole bank
    assert all(len(e.also) >= 2 for e in b.load_bank("demo").entries)


def test_suggest_survives_missing_model(cli, capsys, monkeypatch):
    def boom():
        raise RuntimeError("no model files")
    monkeypatch.setattr(matcher_mod, "_load_fastembed", boom)
    app.main(["suggest", "demo", "weakness"])
    out = capsys.readouterr().out
    assert "couldn't load the model" in out and "What is your biggest weakness?" in out


def test_transcribe_command_runs_the_whole_chain_on_a_wav(cli, capsys, monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    rate = 44100
    t = np.arange(int(rate * 2.0))
    speech = (rng.standard_normal(len(t)) * 0.05 * (0.6 + 0.4 * np.sin(t / 900))).astype(np.float32)
    pcm = np.concatenate([np.zeros(rate // 2), speech, np.zeros(rate)]) 
    stereo = np.repeat((pcm * 32767).astype(np.int16)[:, None], 2, axis=1)
    f = tmp_path / "clip.wav"
    with wave.open(str(f), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate); w.writeframes(stereo.tobytes())

    class Model:
        def transcribe(self, a, **kw):
            assert kw["language"] == "en" and len(a) > 16000
            return iter([SimpleNamespace(text="What is your biggest weakness?", no_speech_prob=0.0)]), None

    def load(self):
        self.model = Model()
    monkeypatch.setattr(audio.Listener, "load_model", load)
    app.main(["transcribe", str(f), "--match"])
    out = capsys.readouterr().out
    assert "What is your biggest weakness?" in out and "-> " in out and "1 utterance found" in out


def test_transcribe_rejects_bad_files(cli, tmp_path):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"not a wav")
    with pytest.raises(SystemExit) as e:
        app.main(["transcribe", str(bad)])
    assert "ffmpeg" in str(e.value)
    f8 = tmp_path / "eight.wav"
    with wave.open(str(f8), "wb") as w:
        w.setnchannels(1); w.setsampwidth(1); w.setframerate(8000); w.writeframes(bytes(100))
    with pytest.raises(SystemExit) as e:
        app.main(["transcribe", str(f8)])
    assert "16-bit" in str(e.value)


class FakeListener:
    script = {}

    def __init__(self, cfg, on_text, on_status):
        self.on_text, self.on_status = on_text, on_status
        self.s = dict(level=0.05, device="Fake Speakers", model_ready=True, quiet_for=0, waiting=0, dropped=0,
                      reconnects=0, lag=0.4, decode=0.1, error="")
        self.s.update(self.script.get("stats", {}))

    def start(self):
        if self.script.get("text"):
            self.on_text(audio.Heard(self.script["text"], True, 1, False, 0.42, 0.1))

    def stop(self):
        pass

    def stats(self):
        return self.s


@pytest.mark.parametrize("script,expect", [
    ({"text": "what is your name"}, "OK: audio is captured"),
    ({"stats": {"level": 0.0}}, "no sound reached the capture"),
    ({"stats": {"device": ""}}, "no audio output could be opened"),
    ({"stats": {"model_ready": False}}, "speech model didn't finish loading"),
    ({"stats": {"error": "Couldn't open audio output for capture: boom"}}, "PROBLEM: Couldn't open audio output"),
    ({}, "Audio arrived but nothing was transcribed"),
])
def test_audiotest_diagnoses_each_stage(cli, capsys, monkeypatch, script, expect):
    FakeListener.script = script
    monkeypatch.setattr(audio, "Listener", FakeListener)
    app.main(["audiotest", "--seconds", "1"])
    assert expect.lower() in capsys.readouterr().out.lower()
