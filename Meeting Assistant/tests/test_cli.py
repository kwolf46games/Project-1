import sys
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


# ---------- generate ----------

import generate as gen  # noqa: E402

RealGenerator = gen.Generator


def _fake_generator(monkeypatch, pieces=("**Own it.** ", "I {{verb}} well."), fail=None):
    from test_generate import FakeClient
    client = FakeClient(pieces=list(pieces), fail=fail)
    monkeypatch.setattr(gen, "Generator", lambda cfg, on_event: RealGenerator(cfg, on_event, client_factory=lambda: client))
    return client


@pytest.fixture
def gen_cli(cli, monkeypatch):
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gen, "KEY_FILE", cli / "no_key.txt")
    monkeypatch.setattr(gen, "GROQ_KEY_FILE", cli / "no_groq_key.txt")
    monkeypatch.setattr(gen, "PROFILE_FILE", cli / "no_profile.md")
    return cli


def test_generate_prints_the_streamed_draft_and_timing(gen_cli, capsys, monkeypatch):
    client = _fake_generator(monkeypatch)
    app.main(["generate", "Tell me about a time you led a team?"])
    out = capsys.readouterr().out
    assert "(adapted from your prepared answer; anthropic: claude-opus-5-5)" in out and "**Own it.** I {{verb}} well." in out
    assert "tokens in" in out and "first words after" in out
    sent = client.calls[0]["messages"][0]["content"]
    assert "led a team" in sent and "LEAD" in sent


def test_generate_show_prompt_discloses_what_is_sent(gen_cli, capsys, monkeypatch):
    _fake_generator(monkeypatch)
    app.main(["generate", "What is your biggest weakness?", "--show-prompt"])
    out = capsys.readouterr().out
    assert "=== SYSTEM ===" in out and "Never invent personal facts" in out
    assert "=== MESSAGE ===" in out and "<question>\nWhat is your biggest weakness?\n</question>" in out and "WEAK" in out


def test_generate_as_a_followup(gen_cli, capsys, monkeypatch):
    client = _fake_generator(monkeypatch)
    app.main(["generate", "Can you elaborate on that?", "--after", "led a team"])
    p = client.calls[0]["messages"][0]["content"]
    assert "follow-up (asking for more detail) to an earlier question: Tell me about a time you led a team" in p


def test_generate_needs_a_question(gen_cli, monkeypatch):
    _fake_generator(monkeypatch)
    with pytest.raises(SystemExit) as e:
        app.main(["generate"])
    assert "Give the question" in str(e.value)


def test_generate_explains_a_missing_package(gen_cli, monkeypatch):
    monkeypatch.setattr(gen, "Generator", RealGenerator)               # the real one, with no client injected
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    with pytest.raises(SystemExit) as e:
        app.main(["generate", "Tell me about yourself?"])
    assert "pip install anthropic" in str(e.value)


def test_generate_check_reports_success_and_failure(gen_cli, capsys, monkeypatch):
    _fake_generator(monkeypatch, pieces=["OK"])
    app.main(["generate", "--check"])
    out = capsys.readouterr().out
    assert "model: claude-opus-5-5" in out and "OK: the API key and connection work." in out
    from test_generate import _api_error
    _fake_generator(monkeypatch, fail=_api_error("AuthenticationError", 401))
    app.main(["generate", "--check"])
    out = capsys.readouterr().out
    assert "key was rejected" in out and "OK: the API key" not in out


# ---------- generate with Groq ----------

def _groq_generator(monkeypatch, pieces=("Own ", "it."), models=None):
    import json

    def http(payload):
        for piece in pieces:
            yield json.dumps({"choices": [{"delta": {"content": piece}, "finish_reason": None}]})
        yield json.dumps({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 4}})
        yield "[DONE]"
    sent = []

    class Fake(RealGenerator):
        def list_models(self):
            return models or []
    monkeypatch.setattr(gen, "Generator", lambda cfg, on_event: Fake(cfg, on_event, http=lambda p: (sent.append(p), http(p))[1]))
    return sent


def test_generate_with_a_groq_key_uses_groq(gen_cli, capsys, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    http_calls = []

    def http(payload):
        import json
        http_calls.append(payload)
        yield json.dumps({"choices": [{"delta": {"content": "Own it."}, "finish_reason": "stop"}]})
        yield "[DONE]"
    monkeypatch.setattr(gen, "Generator", lambda cfg, on_event: RealGenerator(cfg, on_event, http=http))
    app.main(["generate", "Tell me about a time you led a team?"])
    out = capsys.readouterr().out
    assert "groq: llama-3.3-70b-versatile" in out and "Own it." in out
    assert http_calls[0]["messages"][0]["role"] == "system" and "led a team" in http_calls[0]["messages"][1]["content"]


def test_generate_models_lists_groq_models(gen_cli, capsys, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    _groq_generator(monkeypatch, models=[("llama-3.3-70b-versatile", 131072), ("openai/gpt-oss-20b", 65536)])
    app.main(["generate", "--models"])
    out = capsys.readouterr().out
    assert "llama-3.3-70b-versatile" in out and "131072" in out and "generate_model" in out


def test_generate_models_needs_groq(gen_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    _fake_generator(monkeypatch)
    with pytest.raises(SystemExit) as e:
        app.main(["generate", "--models"])
    assert "lists Groq's models" in str(e.value)


def test_generate_check_with_groq_and_without_a_key(gen_cli, capsys, monkeypatch):
    monkeypatch.setattr(gen, "Generator", RealGenerator)
    app.main(["generate", "--check"])
    out = capsys.readouterr().out
    assert "PROBLEM:" in out and "groq_key.txt" in out and "OK: the API key" not in out     # tells you where to put one
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    _groq_generator(monkeypatch, pieces=("OK",))
    app.main(["generate", "--check"])
    out = capsys.readouterr().out
    assert "provider: groq" in out and "key: GROQ_API_KEY" in out and "OK: the API key and connection work." in out


# ---------- setkey ----------

def _type_key(monkeypatch, key):
    import getpass
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": key)


def test_setkey_saves_the_groq_key_and_tests_it(gen_cli, capsys, monkeypatch):
    _type_key(monkeypatch, '  "gsk_secret123"  ')
    _groq_generator(monkeypatch, pieces=("OK",))
    app.main(["setkey"])
    assert gen.GROQ_KEY_FILE.read_text(encoding="utf-8") == "gsk_secret123\n"             # tidied: spaces and quotes removed
    out = capsys.readouterr().out
    assert "Saved to no_groq_key.txt" in out and "never uploaded" in out and "gsk_secret123" not in out
    assert "OK: the API key and connection work." in out                                    # tested straight away


def test_setkey_for_anthropic(gen_cli, capsys, monkeypatch):
    _type_key(monkeypatch, "sk-ant-api03-abc")
    _fake_generator(monkeypatch, pieces=["OK"])
    app.main(["setkey", "anthropic"])
    assert gen.KEY_FILE.read_text(encoding="utf-8") == "sk-ant-api03-abc\n"


@pytest.mark.parametrize("typed,provider,msg", [
    ("sk-ant-wrongone", "groq", "doesn't look like a Groq key"),
    ("gsk_wrongone", "anthropic", "doesn't look like a Anthropic key"),
    ("   ", "groq", "No key entered"),
])
def test_setkey_refuses_the_wrong_kind_of_key_and_saves_nothing(gen_cli, monkeypatch, typed, provider, msg):
    _type_key(monkeypatch, typed)
    with pytest.raises(SystemExit) as e:
        app.main(["setkey", provider])
    assert msg in str(e.value) and "Nothing was saved" in str(e.value)
    assert not gen.GROQ_KEY_FILE.exists() and not gen.KEY_FILE.exists()
    assert not typed.strip() or typed.strip() not in str(e.value)                         # never echoed back
