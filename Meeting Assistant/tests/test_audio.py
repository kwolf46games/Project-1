import numpy as np

import audio
import bank as b
from transcript import Vocabulary


class W:
    def __init__(self, word, start, end, probability):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class Seg:
    def __init__(self, text, probs=None, no_speech_prob=0.0, avg_logprob=-0.2):
        self.text, self.no_speech_prob, self.avg_logprob = text, no_speech_prob, avg_logprob
        toks = text.split()
        probs = probs or [0.95] * len(toks)
        self.words = [W(" " + t, i * .3, i * .3 + .25, p) for i, (t, p) in enumerate(zip(toks, probs))]


class FakeModel:
    """Returns the next scripted list of segments for each transcribe() call, recording what it was given."""
    def __init__(self, *scripts):
        self.scripts, self.calls = list(scripts), []

    def transcribe(self, audio_, **opts):
        self.calls.append((len(audio_), opts))
        return iter(self.scripts.pop(0)), None


def listener(model, vocab=None, **cfg):
    heard, transcripts = [], []
    c = {**b.DEFAULT_CONFIG, "active_banks": [], "whisper_prompt": "BREAD", **cfg}
    lst = audio.Listener(c, heard.append, lambda s: None, transcripts.append)
    lst.model = model
    lst._vocab = vocab or Vocabulary()
    lst._vocab_sig = lst._bank_signature()          # keep the injected vocabulary; don't reload from disk
    return lst, heard, transcripts


def clip(seconds):
    return np.zeros(int(audio.TARGET_RATE * seconds), dtype=np.float32)


def test_transcribe_uses_wide_beam_word_confidence_and_vocabulary_prompt():
    m = FakeModel([Seg("Why AlphaSights?")])
    lst, heard, _ = listener(m, Vocabulary(["AlphaSights", "TAMID"]))
    tr = lst._transcribe(clip(2))
    _, opts = m.calls[0]
    assert opts["beam_size"] == 5 and opts["word_timestamps"] is True and "without_timestamps" not in opts
    assert opts["initial_prompt"].startswith("BREAD.") and "AlphaSights" in opts["initial_prompt"]
    assert tr.text == "Why AlphaSights?" and tr.confidence == 0.95 and not tr.incomplete


def test_config_can_restore_the_old_fast_greedy_behaviour():
    m = FakeModel([Seg("Tell me about yourself.")])
    lst, *_ = listener(m, beam_size=1, word_confidence=False)
    lst._transcribe(clip(2))
    _, opts = m.calls[0]
    assert opts["beam_size"] == 1 and opts["without_timestamps"] is True and "word_timestamps" not in opts


def test_misheard_names_are_repaired_only_where_the_model_was_unsure():
    m = FakeModel([Seg("Why Alphasight at Tammid?", probs=[.9, .35, .9, .3])])
    lst, *_ = listener(m, Vocabulary(["AlphaSights", "TAMID"]))
    tr = lst._transcribe(clip(2))
    assert tr.text == "Why AlphaSights at TAMID?"
    assert ("Alphasight", "AlphaSights") in tr.corrections and tr.words[1].prob == .35


def test_junk_from_silence_is_dropped():
    lst, heard, _ = listener(FakeModel([Seg("Thanks for watching!")], [Seg("Um.")],
                                       [Seg("hello", no_speech_prob=.95, avg_logprob=-1.4)]))
    for _ in range(3):
        assert lst._transcribe(clip(1)) is None


def test_fillers_and_stutters_are_cleaned_before_anything_sees_the_text():
    lst, *_ = listener(FakeModel([Seg("Um, tell tell me about yourself.")]))
    assert lst._transcribe(clip(2)).text == "Tell me about yourself."


def test_complete_utterance_is_emitted_at_once_with_transcript_first():
    order = []
    lst, *_ = listener(FakeModel([Seg("Tell me about yourself.")]))
    lst.on_transcript = lambda tr: order.append(("transcript", tr.text))
    lst.on_text = lambda t: order.append(("text", t))
    lst._finish(clip(2))
    assert order == [("transcript", "Tell me about yourself."), ("text", "Tell me about yourself.")]
    assert lst._held is None


def test_question_cut_off_by_a_pause_is_joined_with_its_second_half():
    m = FakeModel([Seg("Tell me about a time when you")], [Seg("Tell me about a time when you led a team.")])
    lst, heard, transcripts = listener(m)
    lst._finish(clip(2))
    assert heard == [] and lst._held is not None           # held, not emitted as half a question
    lst._finish(clip(1.5))
    assert heard == ["Tell me about a time when you led a team."] and len(transcripts) == 1
    assert m.calls[1][0] == len(clip(2)) + int(audio.TARGET_RATE * 0.2) + len(clip(1.5))   # one joined clip
    assert lst._held is None


def test_held_utterance_is_released_after_the_wait_but_not_while_someone_is_speaking():
    lst, heard, _ = listener(FakeModel([Seg("Tell me about a time when you")]), continuation_seconds=1.5)
    lst._finish(clip(2))
    deadline = lst._held[2]
    lst._release_held(False, now=deadline - 0.1)
    lst._release_held(True, now=deadline + 5)                # speech in progress: keep waiting
    assert heard == []
    lst._release_held(False, now=deadline + 0.1)
    assert heard == ["Tell me about a time when you"] and lst._held is None


def test_holding_can_be_turned_off():
    lst, heard, _ = listener(FakeModel([Seg("Tell me about a time when you")]), continuation_seconds=0)
    lst._finish(clip(2))
    assert heard == ["Tell me about a time when you"]


def test_held_text_is_not_lost_if_the_joined_clip_turns_out_to_be_noise():
    lst, heard, _ = listener(FakeModel([Seg("Tell me about a time when you")], [Seg("Thanks for watching!")]))
    lst._finish(clip(2))
    lst._finish(clip(1))
    assert heard == ["Tell me about a time when you"]


def test_vocabulary_follows_the_active_banks_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "BANKS_DIR", tmp_path)
    (tmp_path / "one.md").write_text("## Why NovaCore?\n\nBecause NovaCore ships.\n", encoding="utf-8")
    lst, *_ = listener(FakeModel(), active_banks=["one"])
    lst._vocab_sig = None
    lst.refresh_vocabulary()
    assert "NovaCore" in lst._vocab.terms
    (tmp_path / "one.md").write_text("## Why Orbitly?\n\nBecause Orbitly ships.\n", encoding="utf-8")
    import os, time
    os.utime(tmp_path / "one.md", (time.time() + 5, time.time() + 5))
    lst.refresh_vocabulary()
    assert "Orbitly" in lst._vocab.terms and "NovaCore" not in lst._vocab.terms
    lst.cfg["use_vocabulary"] = False
    lst.refresh_vocabulary()
    assert not lst._vocab


def test_broken_bank_does_not_stop_listening(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "BANKS_DIR", tmp_path)
    lst, *_ = listener(FakeModel(), Vocabulary(["Keep"]), active_banks=["missing"])
    lst._vocab_sig = None
    lst.refresh_vocabulary()                                   # no such bank: must not raise
    assert isinstance(lst._vocab, Vocabulary)
