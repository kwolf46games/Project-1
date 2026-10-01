"""Listener end to end: real threads, a fake sound device (pyaudiowpatch) and a fake Whisper model."""
import inspect
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

import audio
from audio import Heard, Listener, Transcriber, is_hallucination

RNG = np.random.default_rng(7)
CFG = {"whisper_model": "base.en", "silence_seconds": 0.7, "early_silence_seconds": 0.35,
       "reconnect_idle_seconds": 0, "output_device": "", "whisper_prompt": ""}


def voice16(sec, amp=0.05):
    n = int(sec * 16000)
    return (RNG.standard_normal(n) * amp * (0.6 + 0.4 * np.sin(np.arange(n) / 700))).astype(np.float32)


def to_device_bytes(x16, rate=48000, channels=2):
    up = np.repeat(x16, rate // 16000)                      # crude upsample is fine: the resampler removes the images
    pcm = (np.clip(up, -1, 1) * 32767).astype(np.int16)
    return np.repeat(pcm[:, None], channels, axis=1).tobytes()


# ---------- fake sound device ----------

class FakeDevice:
    """Plays a script of audio into the stream callback in real time, like a loopback endpoint."""

    def __init__(self, rate=48000, channels=2):
        self.rate, self.channels = rate, channels
        self.script: list[bytes] = []
        self.opens = 0
        self.fail_opens = 0
        self.active = True
        self.silence_zeros = False
        self.lock = threading.Lock()

    def play(self, x16):
        raw = to_device_bytes(x16, self.rate, self.channels)
        step = int(self.rate * 0.03) * 2 * self.channels
        with self.lock:
            self.script += [raw[i:i + step] for i in range(0, len(raw), step)]

    def silence(self, sec):
        self.play(np.zeros(int(sec * 16000), dtype=np.float32))


class FakeStream:
    def __init__(self, dev, cb):
        self.dev, self.cb, self.closed = dev, cb, False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        step = int(self.dev.rate * 0.03) * 2 * self.dev.channels
        while not self.closed:
            time.sleep(0.03)
            with self.dev.lock:
                chunk = self.dev.script.pop(0) if self.dev.script else None
            if chunk is None and self.dev.silence_zeros:
                chunk = bytes(step)
            if chunk is not None:
                self.cb(chunk, int(self.dev.rate * 0.03), None, 0)

    def is_active(self):
        return self.dev.active and not self.closed

    def stop_stream(self):
        self.closed = True

    def close(self):
        self.closed = True


@pytest.fixture
def device(monkeypatch):
    dev = FakeDevice()
    mod = types.ModuleType("pyaudiowpatch")
    mod.paInt16, mod.paContinue, mod.paWASAPI = 8, 0, 13

    class PyAudio:
        def get_loopback_device_info_generator(self):
            yield {"index": 3, "name": "Speakers (Fake) [Loopback]", "maxInputChannels": dev.channels,
                   "defaultSampleRate": float(dev.rate)}

        def get_host_api_info_by_type(self, t):
            return {"defaultOutputDevice": 1}

        def get_device_info_by_index(self, i):
            return {"name": "Speakers (Fake)"}

        def open(self, **kw):
            dev.opens += 1
            if dev.fail_opens > 0:
                dev.fail_opens -= 1
                raise OSError("device unavailable")
            return FakeStream(dev, kw["stream_callback"])

        def terminate(self):
            pass
    mod.PyAudio = PyAudio
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)
    return dev


class FakeModel:
    """Stands in for faster_whisper.WhisperModel: answers by audio length, records every call."""

    def __init__(self, text_for=None, delay=0.05):
        self.calls = []
        self.delay = delay
        self.text_for = text_for or (lambda sec, n: f"what is your biggest weakness number {n}?")
        self.fail_on: set[int] = set()
        self.n = 0
        self.lock = threading.Lock()

    def transcribe(self, audio_in, **kw):
        with self.lock:
            self.n += 1
            n = self.n
        self.calls.append((len(audio_in) / 16000, kw))
        time.sleep(self.delay)
        if n in self.fail_on:
            raise RuntimeError("decoder blew up")
        text = self.text_for(len(audio_in) / 16000, n)
        return iter([SimpleNamespace(text=text, no_speech_prob=0.0)] if text else []), None


def make_listener(model=None, cfg=None):
    heard: list[Heard] = []
    status: list[str] = []
    lock = threading.Lock()

    def on_text(h):
        with lock:
            heard.append(h)
    lis = Listener({**CFG, **(cfg or {})}, on_text, status.append)
    lis.model = model or FakeModel()
    return lis, heard, status


def wait_for(cond, timeout=6.0, step=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return False


# ---------- tests ----------

def test_utterance_flows_through_with_early_text(device):
    lis, heard, status = make_listener()
    lis.start()
    try:
        assert wait_for(lambda: any(s.startswith("Listening") for s in status))
        device.silence(0.3)
        device.play(voice16(2.0))
        device.silence(1.2)
        assert wait_for(lambda: any(h.final for h in heard))
        early = [h for h in heard if not h.final]
        final = [h for h in heard if h.final]
        assert len(early) == 1 and len(final) == 1
        assert final[0].repeat and final[0].text == early[0].text and early[0].utt == final[0].utt
        assert early[0].lag < 0.8 and final[0].lag < 1.2           # early guess ~0.35 s + fake 0.05 s decode
        assert lis.model.n == 1                                     # the commit reused the early decode: ONE decode
    finally:
        lis.stop()


def test_resumed_speech_gets_a_second_decode(device):
    lis, heard, status = make_listener(FakeModel(text_for=lambda sec, n: f"heard {sec:.1f} seconds"))
    lis.start()
    try:
        device.silence(0.2)
        device.play(voice16(1.5))
        device.silence(0.5)               # long enough for an early guess, too short to end the utterance
        device.play(voice16(1.5))
        device.silence(1.2)
        assert wait_for(lambda: any(h.final for h in heard))
        assert [(h.final, h.repeat) for h in heard] == [(False, False), (False, False), (True, True)]
        assert len({h.utt for h in heard}) == 1
        first, second, last = (float(h.text.split()[1]) for h in heard)
        assert first < 2.3 and second > 3.5 and last == second       # the final is the second guess, reused
        assert lis.model.n == 2                                      # two decodes in total, never three
    finally:
        lis.stop()


def test_decode_failure_on_early_guess_is_redone_at_commit(device):
    model = FakeModel()
    model.fail_on = {1}
    lis, heard, status = make_listener(model)
    lis.start()
    try:
        device.silence(0.2)
        device.play(voice16(2.0))
        device.silence(1.2)
        assert wait_for(lambda: any(h.final for h in heard))
        assert any("Couldn't transcribe" in s for s in status)
        assert model.n == 2                                          # early attempt failed, commit decoded again
    finally:
        lis.stop()


def test_listener_survives_callback_exceptions(device):
    boom = {"n": 0}
    got = []

    def on_text(h):
        boom["n"] += 1
        if boom["n"] == 1:
            raise ValueError("ui bug")
        got.append(h)
    status = []
    lis = Listener(dict(CFG), on_text, status.append)
    lis.model = FakeModel()
    lis.start()
    try:
        for _ in range(2):
            device.silence(0.2)
            device.play(voice16(1.2))
            device.silence(1.1)
        assert wait_for(lambda: len(got) >= 1, timeout=8)
        assert any("Couldn't use what was heard" in s for s in status)
    finally:
        lis.stop()


def test_device_failure_then_recovery(device):
    device.fail_opens = 2
    lis, heard, status = make_listener()
    lis.start()
    try:
        assert wait_for(lambda: any("Couldn't open audio output" in s for s in status))
        assert wait_for(lambda: any(s.startswith("Listening") for s in status), timeout=8)
        assert device.opens == 3
        device.silence(0.2)
        device.play(voice16(1.5))
        device.silence(1.1)
        assert wait_for(lambda: any(h.final for h in heard))
    finally:
        lis.stop()


def test_restart_reopens_device_without_reloading_model(device):
    lis, heard, status = make_listener()
    lis.start()
    try:
        assert wait_for(lambda: device.opens == 1 and lis.stats()["device"])
        model_before = lis.model
        lis.restart()
        assert wait_for(lambda: device.opens == 2)
        assert lis.model is model_before and lis.reconnects >= 1
        device.silence(0.2)
        device.play(voice16(1.5))
        device.silence(1.1)
        assert wait_for(lambda: any(h.final for h in heard))
    finally:
        lis.stop()


def test_dead_stream_reconnects(device):
    lis, heard, status = make_listener()
    lis.start()
    try:
        assert wait_for(lambda: device.opens == 1)
        device.active = False
        assert wait_for(lambda: any("reconnecting" in s for s in status))
        device.active = True
        assert wait_for(lambda: device.opens >= 2, timeout=8)
        assert wait_for(lambda: status[-1].startswith("Listening"), timeout=8)
    finally:
        lis.stop()


def test_pause_ignores_audio(device):
    lis, heard, status = make_listener()
    lis.start()
    try:
        assert wait_for(lambda: device.opens == 1)
        lis.paused.set()
        device.silence(0.2)
        device.play(voice16(2.0))
        device.silence(1.5)
        time.sleep(2.5)
        assert heard == [] and lis.model.n == 0
        lis.paused.clear()
        device.silence(0.2)
        device.play(voice16(1.5))
        device.silence(1.1)
        assert wait_for(lambda: any(h.final for h in heard))
    finally:
        lis.stop()


def test_idle_reconnect_backs_off(device):
    lis, heard, status = make_listener(cfg={"reconnect_idle_seconds": 1})
    lis.start()
    try:
        assert wait_for(lambda: device.opens >= 2, timeout=6)       # ~1 s of silence -> quiet re-open
        t0 = time.monotonic()
        n = device.opens
        assert wait_for(lambda: device.opens > n, timeout=8)
        assert time.monotonic() - t0 > 1.4                          # second gap is longer than the first (backoff)
        assert lis._idle_after >= 2
    finally:
        lis.stop()


def test_no_idle_reconnect_while_audio_flows(device):
    device.silence_zeros = True                                     # some drivers deliver zeros instead of nothing
    lis, heard, status = make_listener(cfg={"reconnect_idle_seconds": 1})
    lis.start()
    try:
        assert wait_for(lambda: device.opens == 1)
        end = time.monotonic() + 2.5
        while time.monotonic() < end:
            device.play(voice16(0.3, 0.1))                          # real sound keeps arriving
            time.sleep(0.3)
        assert device.opens == 1
    finally:
        lis.stop()


def test_44k_mono_device(device):
    device.rate, device.channels = 44100, 1
    lis, heard, status = make_listener()
    # device bytes helper upsamples by integer factors only; 44100/16000 is not one, so build with scipy
    scipy_signal = pytest.importorskip("scipy.signal")
    lis.start()
    try:
        x = voice16(2.0)
        y = scipy_signal.resample_poly(x, 441, 160)
        pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
        step = int(44100 * 0.03) * 2
        with device.lock:
            device.script += [pcm[i:i + step] for i in range(0, len(pcm), step)]
        z = (np.zeros(44100) * 0).astype(np.int16).tobytes()
        with device.lock:
            device.script += [z[i:i + step] for i in range(0, len(z), step)]
        assert wait_for(lambda: any(h.final for h in heard), timeout=8)
        # the audio the model received should have the right duration (2 s of speech plus pre-roll/pad)
        secs = lis.model.calls[0][0]
        assert 2.0 <= secs <= 2.7
    finally:
        lis.stop()


def test_model_load_failure_then_retry(device, monkeypatch):
    lis = Listener(dict(CFG), lambda h: None, (status := []).append)
    state = {"fail": True}

    def load():
        if state["fail"]:
            raise RuntimeError("no internet and no cached model")
        lis.model = FakeModel()
    monkeypatch.setattr(lis, "load_model", load)
    lis.start()
    try:
        assert wait_for(lambda: any("Speech model failed to load" in s for s in status))
        assert wait_for(lambda: any(s.startswith("Listening") or s.startswith("Speech model failed") for s in status))
        state["fail"] = False
        lis.restart()
        assert wait_for(lambda: lis.stats()["model_ready"], timeout=5)
        assert wait_for(lambda: status[-1].startswith("Listening"))
    finally:
        lis.stop()


def test_queue_overflow_drops_instead_of_blocking(device):
    lis, heard, status = make_listener()
    # never start the consumer: fill the queue the way the callback would
    for _ in range(audio.MAX_QUEUE_CHUNKS + 50):
        try:
            lis._raw.put_nowait(b"\x00\x00")
        except Exception:
            lis.dropped += 1
    assert lis.dropped == 50


def test_vocabulary_goes_into_the_prompt():
    lis, _, _ = make_listener(cfg={"whisper_prompt": "BREAD"})
    assert lis._prompt() == "BREAD"
    lis.set_vocabulary(["Kubernetes", "SQL"])
    assert lis._prompt() == "BREAD Terms: Kubernetes, SQL."
    lis.set_vocabulary([f"T{i}" for i in range(40)])
    assert len(lis.vocabulary) == 12 and len(lis._prompt()) <= 400


# ---------- Transcriber ----------

class ScriptModel:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def transcribe(self, a, **kw):
        self.calls.append(kw)
        r = self.results.pop(0)
        return iter([SimpleNamespace(text=t, no_speech_prob=p) for t, p in r]), None


def test_transcriber_basic_and_hallucination_filter():
    tr = Transcriber(ScriptModel([[("What is your name?", 0.0)], [("Thank you.", 0.0)]]))
    a = voice16(1.0)
    assert tr.decode(a, True, 1.0) == "What is your name?"
    assert tr.decode(a, True, 1.0) == ""


def test_transcriber_retries_without_vad_once_for_finals_only():
    m = ScriptModel([[], [("how would you handle conflict", 0.1)]])
    tr = Transcriber(m)
    assert tr.decode(voice16(1.5), True, 1.5) == "how would you handle conflict"
    assert [c["vad_filter"] for c in m.calls] == [True, False]
    m = ScriptModel([[]])
    assert Transcriber(m).decode(voice16(1.5), False, 1.5) == ""           # early guesses never retry
    assert len(m.calls) == 1
    m = ScriptModel([[], [("music la la la", 0.9)]])
    assert Transcriber(m).decode(voice16(1.5), True, 1.5) == ""            # whisper itself says: not speech
    m = ScriptModel([[]])
    assert Transcriber(m).decode(voice16(0.5), True, 0.5) == ""            # too short to bother
    assert len(m.calls) == 1


def test_transcriber_falls_back_when_option_unsupported():
    class Old:
        calls = []

        def transcribe(self, a, language, beam_size, vad_filter, condition_on_previous_text, without_timestamps,
                       initial_prompt):
            self.calls.append(1)
            return iter([SimpleNamespace(text="hello there friend", no_speech_prob=0)]), None
    m = Old()
    assert Transcriber(m).decode(voice16(1.0), True, 1.0) == "hello there friend"


def test_prompt_echo_and_loops_filtered():
    assert is_hallucination("BREAD", prompt="BREAD")
    assert is_hallucination("Terms: Kubernetes, SQL.", prompt="Terms: Kubernetes, SQL.")
    assert is_hallucination("you you you you you you you you")
    assert not is_hallucination("Tell me about Kubernetes", prompt="Terms: Kubernetes, SQL.")
    assert not is_hallucination("Thank you for coming in today, tell me about yourself", "", 3.0)
    assert is_hallucination("Thank you.", "", 1.0) and not is_hallucination("Thank you.", "", 4.0)


def test_kwargs_match_installed_faster_whisper():
    fw = pytest.importorskip("faster_whisper")
    params = set(inspect.signature(fw.WhisperModel.transcribe).parameters)
    seen = {}

    class Spy:
        def transcribe(self, a, **kw):
            seen.update(kw)
            return iter([]), None
    Transcriber(Spy(), lambda: "x").decode(voice16(1.0), True, 1.0)
    assert set(seen) <= params, set(seen) - params
    from faster_whisper.vad import VadOptions   # the VAD options this app relies on exist
    assert {"threshold", "min_silence_duration_ms", "speech_pad_ms"} <= set(VadOptions.__dataclass_fields__)
    init = set(inspect.signature(fw.WhisperModel.__init__).parameters)
    assert {"device", "compute_type", "cpu_threads", "download_root", "local_files_only"} <= init
