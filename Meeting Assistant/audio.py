"""Capture system audio output (WASAPI loopback), cut it into utterances, transcribe locally.

Only what your speakers/headphones play is captured, so your own mic never gets in.
Audio and transcripts stay in memory; nothing is written to disk.

Three threads, so a slow transcription can never make the audio path drop or stall:

    PortAudio callback ──▶ capture thread ──▶ speech worker ──▶ on_text(Heard)
    (just queues bytes)    resample + find     faster-whisper,
                           utterance edges     one job at a time

Speed: when the speaker pauses for `early_silence_seconds` the utterance so far is transcribed
straight away.  If they really were finished (the usual case) that text is simply reused when the
full `silence_seconds` pass, so the answer is ready that much sooner at no extra cost; if they
carry on, the early text is superseded by the final one.
"""
from __future__ import annotations

import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bank import ROOT

MODELS_DIR = ROOT / "models"
TARGET_RATE = 16000
FRAME_SEC = 0.03
MIN_VOICED_SEC = 0.30         # shorter blips are clicks and dings, not speech
MIN_EARLY_VOICED_SEC = 0.90   # don't start an early transcription on a few syllables
MAX_UTTERANCE_SEC = 25.0
MIN_RMS = 0.004               # quietest speech the gate will notice (about -48 dBFS)
RETRY_NO_VAD_SEC = 1.0        # if the speech detector finds nothing in this much sound, double-check once
MAX_QUEUE_CHUNKS = 400        # ~12 s of audio the callback may have waiting before it starts dropping


# ---------- devices ----------

def list_loopback_devices() -> list[dict]:
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        return list(p.get_loopback_device_info_generator())
    finally:
        p.terminate()


def _pick_loopback(p, name_hint: str = "") -> dict:
    import pyaudiowpatch as pyaudio
    loopbacks = list(p.get_loopback_device_info_generator())
    if not loopbacks:
        raise RuntimeError("No WASAPI loopback devices found.")
    if name_hint:
        for d in loopbacks:
            if name_hint.lower() in d["name"].lower():
                return d
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    for d in loopbacks:
        if speakers["name"] in d["name"]:
            return d
    return loopbacks[0]


# ---------- signal helpers ----------

def to_mono(raw: bytes, channels: int) -> np.ndarray:
    """Interleaved int16 bytes -> mono float32 in [-1, 1]. Tolerates a ragged tail."""
    x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16)
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).astype(np.float32).mean(axis=1)
    else:
        x = x.astype(np.float32)
    return x / 32768.0


class StreamResampler:
    """Chunk-by-chunk sample-rate conversion to 16 kHz (polyphase windowed-sinc, numpy only).

    Unlike resampling each chunk separately, filter state carries across chunks, so there are no
    clicks at chunk edges and nothing above 8 kHz folds back into the speech band.  The output
    matches scipy.signal.resample_poly (delayed by a fixed ~0.6 ms)."""

    def __init__(self, rate_in: int, rate_out: int = TARGET_RATE) -> None:
        rate_in, rate_out = int(round(rate_in)), int(rate_out)
        g = math.gcd(rate_in, rate_out)
        self.up, self.down = rate_out // g, rate_in // g
        self.passthrough = self.up == self.down == 1
        if self.passthrough:
            return
        m = max(self.up, self.down)
        half = 10 * m
        t = np.arange(-half, half + 1, dtype=np.float64)
        h = np.sinc(t / m) * np.kaiser(2 * half + 1, 5.0)
        h *= self.up / h.sum()                                   # unity gain at DC after zero-stuffing
        self.taps = -(-len(h) // self.up)                        # taps per output sample
        h = np.concatenate([h, np.zeros(self.taps * self.up - len(h))])
        self.filt = h.reshape(self.taps, self.up).T.astype(np.float32).copy()   # [phase, tap]
        self.delay = half // self.down                           # output samples of group delay
        self._buf = np.zeros(self.taps - 1, dtype=np.float32)    # history, zero-padded before the stream
        self._off = -(self.taps - 1)                             # global index of _buf[0]
        self._k = 0                                              # next output sample to produce
        self._tap_ix = np.arange(self.taps)[None, :]

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return x.astype(np.float32, copy=False)
        if len(x):
            self._buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)])
        total = self._off + len(self._buf)                       # input samples seen so far
        k_max = (total * self.up - 1) // self.down               # last output whose newest input has arrived
        if k_max < self._k:
            return np.zeros(0, dtype=np.float32)
        m = np.arange(self._k, k_max + 1, dtype=np.int64) * self.down
        base, phase = m // self.up, m % self.up
        window = self._buf[base[:, None] - self._tap_ix - self._off]
        y = np.einsum("kt,kt->k", self.filt[phase], window)
        self._k = k_max + 1
        drop = (self._k * self.down) // self.up - (self.taps - 1) - self._off
        if drop > 0:
            self._buf, self._off = self._buf[drop:], self._off + drop
        return y.astype(np.float32, copy=False)


def prepare_for_asr(audio: np.ndarray) -> np.ndarray:
    """Quiet streams transcribe poorly; lift them to a healthy level (never loud ones, never > ~21 dB)."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if 1e-4 < peak < 0.3:
        audio = np.clip(audio * min(0.6 / peak, 12.0), -1.0, 1.0)
    return audio.astype(np.float32, copy=False)


_HALLUCINATIONS = {
    "thank you", "thanks", "thanks for watching", "thank you for watching", "thank you very much", "you", "bye",
    "bye bye", "goodbye", "see you next time", "please subscribe", "subscribe", "okay", "ok", "oh", "uh", "um",
    "hmm", "mm", "mhm", "the end", "so", "yeah", "i", "a", "the",
}


def is_hallucination(text: str, prompt: str = "", voiced_sec: float = 0.0) -> bool:
    """Whisper invents text on near-silence ("Thank you.", the prompt itself, repeating loops). Catch it."""
    t = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return True
    if t in _HALLUCINATIONS and voiced_sec < 2.5:
        return True
    p = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", prompt.lower())).strip()
    words = t.split()
    if p and (t == p or (len(words) >= 2 and t in p)):
        return True
    return len(words) >= 6 and len(set(words)) / len(words) < 0.3


# ---------- cutting speech into utterances ----------

@dataclass
class Event:
    kind: str                 # "spec" early guess | "final" | "commit" (final == the last "spec")
    utt: int
    audio: np.ndarray | None  # None for "commit"
    voiced_sec: float
    quiet_sec: float          # how long the speaker had been quiet when this fired


class Segmenter:
    """Finds where utterances start and end in a 16 kHz mono stream (no threads, no clock of its own).

    Silence is counted in audio time, so a burst of buffered audio can't fool it; `idle_sec` adds
    wall-clock quiet for the case where loopback stops delivering anything while nothing plays."""

    PRE_FRAMES = 10   # keep this much before the first loud frame so word onsets aren't clipped
    PAD_FRAMES = 6    # and this much after the last

    def __init__(self, silence_sec: float = 0.7, early_sec: float = 0.35, max_sec: float = MAX_UTTERANCE_SEC,
                 frame_sec: float = FRAME_SEC, rate: int = TARGET_RATE) -> None:
        self.silence_sec = max(0.2, float(silence_sec))
        self.early_sec = float(early_sec) if 0 < early_sec < self.silence_sec else 0.0
        self.max_sec, self.frame_sec = max_sec, frame_sec
        self.frame_len = int(rate * frame_sec)
        self.floor = 0.003    # running estimate of the background level
        self.level = 0.0      # smoothed recent loudness, for the meter
        self.utt = 0
        self._buf = np.zeros(0, dtype=np.float32)
        self._reset([])

    def _reset(self, pre: list) -> None:
        self.speech: list[np.ndarray] = []
        self.pre = pre[-self.PRE_FRAMES:]
        self.voiced = 0           # loud frames in this utterance
        self.trail = 0            # quiet frames since the last loud one
        self.last_voiced = 0      # frames up to and including the last loud one
        self.spec_voiced = -1     # `voiced` when the early guess was requested

    @property
    def in_speech(self) -> bool:
        return bool(self.speech)

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._reset([])

    def _audio(self) -> np.ndarray:
        return np.concatenate(self.speech[: self.last_voiced + self.PAD_FRAMES])

    def feed(self, samples: np.ndarray, idle_sec: float = 0.0) -> list[Event]:
        if len(samples):
            self._buf = np.concatenate([self._buf, samples])
        n = self.frame_len
        events: list[Event] = []
        while len(self._buf) >= n:
            frame, self._buf = self._buf[:n], self._buf[n:]
            rms = float(np.sqrt(np.mean(frame * frame)))
            self.level = max(rms, self.level * 0.92)
            on = max(MIN_RMS, self.floor * 3.0)
            off = max(MIN_RMS * 0.6, self.floor * 2.0)   # once talking, a dip must be deeper to count as quiet
            voiced = rms > (off if self.speech else on)
            if not voiced:
                self.floor += (rms - self.floor) * (0.1 if rms < self.floor else 0.005)
                self.floor = min(0.05, max(0.0005, self.floor))
            if voiced:
                if not self.speech:
                    self.utt += 1
                    self.speech = list(self.pre)
                self.speech.append(frame)
                self.voiced += 1
                self.trail = 0
                self.last_voiced = len(self.speech)
            elif self.speech:
                self.speech.append(frame)
                self.trail += 1
            else:
                self.pre = (self.pre + [frame])[-self.PRE_FRAMES:]
            events += self._due(0.0)      # per frame: a burst of buffered audio must not merge two utterances
        return events + self._due(idle_sec)

    def _due(self, idle_sec: float) -> list[Event]:
        if not self.speech:
            return []
        quiet = self.trail * self.frame_sec + max(0.0, idle_sec)
        voiced_sec = self.voiced * self.frame_sec
        if quiet >= self.silence_sec:
            out = []
            if voiced_sec >= MIN_VOICED_SEC:
                if self.spec_voiced == self.voiced:
                    out.append(Event("commit", self.utt, None, voiced_sec, quiet))
                else:
                    out.append(Event("final", self.utt, self._audio(), voiced_sec, quiet))
            self._reset(self.speech)
            return out
        if (self.early_sec and quiet >= self.early_sec and voiced_sec >= MIN_EARLY_VOICED_SEC
                and self.spec_voiced != self.voiced):
            self.spec_voiced = self.voiced
            return [Event("spec", self.utt, self._audio(), voiced_sec, quiet)]
        if len(self.speech) * self.frame_sec >= self.max_sec:
            ev = Event("final", self.utt, np.concatenate(self.speech), voiced_sec, 0.0)
            self._reset(self.speech)   # carry the tail over so a word cut by the split isn't lost
            return [ev]
        return []


# ---------- speech to text ----------

@dataclass
class Heard:
    text: str
    final: bool = True      # False: an early guess made during a pause, may be superseded
    utt: int = 0
    repeat: bool = False    # final text identical to the early guess already delivered
    lag: float = 0.0        # seconds from the speaker going quiet until this text was ready
    decode: float = 0.0     # seconds the transcription itself took


class Transcriber:
    """faster-whisper with the settings this app has always used, plus guards."""

    def __init__(self, model, prompt: Callable[[], str] = lambda: "") -> None:
        self.model, self.prompt = model, prompt

    def _run(self, audio: np.ndarray, vad: bool) -> list:
        kw = dict(language="en", beam_size=1, vad_filter=vad, condition_on_previous_text=False,
                  without_timestamps=True, initial_prompt=self.prompt() or None,
                  temperature=0.0)   # no retries at higher temperature: bounded time, repeatable text
        try:
            segments, _ = self.model.transcribe(audio, **kw)
        except TypeError:   # a faster-whisper too old for one of the options: use the original set
            kw.pop("temperature")
            segments, _ = self.model.transcribe(audio, **kw)
        return list(segments)

    def decode(self, audio: np.ndarray, final: bool = True, voiced_sec: float = 0.0) -> str:
        audio = prepare_for_asr(audio)
        text = " ".join(s.text.strip() for s in self._run(audio, vad=True)).strip()
        if not text and final and voiced_sec >= RETRY_NO_VAD_SEC:
            # The detector inside Whisper can miss quiet or narrow-band speech the level gate caught.
            # Ask once without it, trusting only segments Whisper itself believes are speech.
            segs = self._run(audio, vad=False)
            text = " ".join(s.text.strip() for s in segs if getattr(s, "no_speech_prob", 0.0) < 0.5).strip()
        text = re.sub(r"\s+", " ", text)
        return "" if is_hallucination(text, self.prompt(), voiced_sec) else text


@dataclass
class _Job:
    kind: str
    utt: int
    audio: np.ndarray | None
    voiced_sec: float
    quiet_end: float   # monotonic time the speaker went quiet


class _JobBoard:
    """FIFO of transcription jobs where a newer job for an utterance replaces its waiting early guesses."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._jobs: list[_Job] = []

    def put(self, job: _Job) -> None:
        with self._cv:
            if job.kind in ("spec", "final"):
                self._jobs = [j for j in self._jobs if not (j.utt == job.utt and j.kind == "spec")]
            self._jobs.append(job)
            self._cv.notify()

    def get(self, timeout: float) -> _Job | None:
        with self._cv:
            if not self._jobs:
                self._cv.wait(timeout)
            return self._jobs.pop(0) if self._jobs else None

    def newer_waiting(self, utt: int) -> bool:
        with self._cv:
            return any(j.utt == utt and j.kind in ("spec", "final") for j in self._jobs)

    def __len__(self) -> int:
        with self._cv:
            return len(self._jobs)


class Listener:
    """Background threads: loopback audio -> utterances -> faster-whisper -> on_text(Heard)."""

    def __init__(self, cfg: dict, on_text: Callable[[Heard], None], on_status: Callable[[str], None]):
        self.cfg = cfg
        self.on_text = on_text
        self.on_status = on_status
        self.paused = threading.Event()
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._raw: queue.Queue[bytes] = queue.Queue(maxsize=MAX_QUEUE_CHUNKS)
        self._board = _JobBoard()
        self._threads: list[threading.Thread] = []
        self._seg: Segmenter | None = None
        self.model = None
        self.device_name = ""
        self.vocabulary: list[str] = []
        self._error = ""
        self._model_ready = False
        self._last_sound = time.monotonic()
        self._last_spec: tuple[int, np.ndarray, str | None] | None = None
        self._early_text: dict[int, str] = {}
        self._idle_after = 0.0
        self.dropped = 0
        self.reconnects = 0
        self.last_lag = 0.0
        self.last_decode = 0.0

    # ----- lifecycle -----
    def start(self) -> None:
        """Start whichever of the two worker threads isn't running."""
        self._stop.clear()
        alive = {t.name for t in self._threads if t.is_alive()}
        for name, body in (("capture", self._capture_loop), ("speech", self._asr_loop)):
            if name not in alive:
                if name == "speech":
                    self._error = "" if self._error.startswith("Speech model") else self._error
                t = threading.Thread(target=self._guard, args=(body,), name=name, daemon=True)
                self._threads = [x for x in self._threads if x.name != name] + [t]
                t.start()

    def stop(self) -> None:
        self._stop.set()
        self._reconnect.set()   # wakes anything waiting

    def restart(self) -> None:
        """Re-open the audio device (e.g. after switching headphones) and retry anything that failed.
        The speech model stays loaded."""
        self.start()
        self._reconnect.set()

    def set_vocabulary(self, terms: list[str]) -> None:
        """Names and jargon from the active banks; whisper is told to expect them."""
        self.vocabulary = list(terms)[:12]

    def stats(self) -> dict:
        seg = self._seg
        return {"level": seg.level if seg else 0.0, "device": self.device_name, "model_ready": self._model_ready,
                "quiet_for": time.monotonic() - self._last_sound, "waiting": len(self._board),
                "dropped": self.dropped, "reconnects": self.reconnects,
                "lag": self.last_lag, "decode": self.last_decode, "error": self._error}

    def _prompt(self) -> str:
        parts = [str(self.cfg.get("whisper_prompt") or "").strip()]
        if self.vocabulary:
            parts.append("Terms: " + ", ".join(self.vocabulary) + ".")
        return " ".join(p for p in parts if p)[:400]

    def _announce(self) -> None:
        if self._error:
            self.on_status(self._error)
        elif not self._model_ready:
            self.on_status(f"Loading speech model '{self.cfg['whisper_model']}'…")
        elif not self.device_name:
            self.on_status("Opening audio…")
        else:
            self.on_status(f"Listening · {self.device_name}")

    def _guard(self, fn: Callable[[], None]) -> None:
        """Run a thread body; if it escapes with an exception, say so instead of dying silently."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self._error = f"Audio engine stopped ({type(e).__name__}: {e}). Press Reconnect."
            self._announce()

    # ----- speech model -----
    def load_model(self) -> None:
        if self.model is not None:
            return
        from faster_whisper import WhisperModel
        name = self.cfg["whisper_model"]
        MODELS_DIR.mkdir(exist_ok=True)
        threads = max(2, min(8, os.cpu_count() or 4))

        def make(**extra):
            return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads,
                                download_root=str(MODELS_DIR), **extra)
        self._announce()
        try:                # cached copy first: instant, and works with no internet
            self.model = make(local_files_only=True)
        except Exception:  # noqa: BLE001 - not downloaded yet (or an older faster-whisper)
            self.on_status(f"Downloading speech model '{name}' (first run only)…")
            self.model = make()
        self._warm_up()

    def _warm_up(self) -> None:
        """The first transcription is slow (lazy start-up); do it now instead of on the first question."""
        try:
            t = np.arange(TARGET_RATE, dtype=np.float32) / TARGET_RATE
            tone = (0.1 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)
            tr = Transcriber(self.model)
            tr._run(tone, vad=False)
            tr._run(tone, vad=True)
        except Exception:  # noqa: BLE001 - a warm-up hiccup must never stop real work
            pass

    # ----- speech worker -----
    def _asr_loop(self) -> None:
        try:
            self.load_model()
        except Exception as e:  # noqa: BLE001
            self._error = f"Speech model failed to load: {e}"
            self._announce()
            return
        self._model_ready = True
        self._announce()
        tr = Transcriber(self.model, self._prompt)
        while not self._stop.is_set():
            job = self._board.get(0.2)
            if job is None:
                continue
            try:
                self._handle(tr, job)
            except Exception as e:  # noqa: BLE001 - one bad utterance must not end the session
                self.on_status(f"Couldn't transcribe that ({type(e).__name__}: {e})")

    def _handle(self, tr: Transcriber, job: _Job) -> None:
        if job.kind == "commit":
            held = self._last_spec if self._last_spec and self._last_spec[0] == job.utt else None
            if held is None:
                return
            _, audio, text = held
            if text is None:   # the early attempt failed: do it properly now
                text = self._timed_decode(tr, audio, True, job)
        elif job.kind == "spec":
            try:
                text = self._timed_decode(tr, job.audio, False, job)
            except Exception:
                self._last_spec = (job.utt, job.audio, None)   # so the commit can redo it
                raise
            self._last_spec = (job.utt, job.audio, text)
            if text and not self._board.newer_waiting(job.utt):   # else a longer version is already queued
                self._early_text[job.utt] = text
                self._deliver(Heard(text, False, job.utt, False, self.last_lag, self.last_decode))
            return
        else:
            text = self._timed_decode(tr, job.audio, True, job)
        early = self._early_text.pop(job.utt, None)
        if text:
            self._deliver(Heard(text, True, job.utt, early == text, self.last_lag, self.last_decode))
        self._early_text = {u: t for u, t in self._early_text.items() if u > job.utt}

    def _timed_decode(self, tr: Transcriber, audio: np.ndarray, final: bool, job: _Job) -> str:
        t0 = time.monotonic()
        text = tr.decode(audio, final=final, voiced_sec=job.voiced_sec)
        done = time.monotonic()
        self.last_decode, self.last_lag = done - t0, done - job.quiet_end
        return text

    def _deliver(self, heard: Heard) -> None:
        try:
            self.on_text(heard)
        except Exception as e:  # noqa: BLE001 - a bug in the listener of the text must not stop listening
            self.on_status(f"Couldn't use what was heard ({type(e).__name__}: {e})")

    # ----- capture -----
    def _capture_loop(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            self._reconnect.clear()
            try:
                outcome = self._capture_once()
                backoff = 0.5
                if outcome == "stop":
                    return
            except Exception as e:  # noqa: BLE001
                self.device_name = ""
                self._error = f"Couldn't open audio output for capture: {e}"
                self._announce()
                self._reconnect.wait(backoff)   # a press of Reconnect cuts the wait short
                backoff = min(backoff * 2, 8.0)
                continue
            self.reconnects += 1

    def _capture_once(self) -> str:
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        stream = None
        try:
            dev = _pick_loopback(p, self.cfg.get("output_device", ""))
            channels, rate = int(dev["maxInputChannels"]), int(dev["defaultSampleRate"])
            while True:   # drop audio left over from the previous device
                try:
                    self._raw.get_nowait()
                except queue.Empty:
                    break

            def callback(data, frames, time_info, status):
                try:
                    self._raw.put_nowait(data)
                except queue.Full:
                    self.dropped += 1
                return (None, pyaudio.paContinue)

            stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                            input_device_index=dev["index"], frames_per_buffer=int(rate * FRAME_SEC),
                            stream_callback=callback)
            self.device_name = dev["name"].replace(" [Loopback]", "")
            self._error = ""
            self._announce()
            return self._pump(stream, channels, rate)
        finally:
            for closer in (lambda: stream.stop_stream(), lambda: stream.close(), p.terminate):
                try:
                    closer()
                except Exception:  # noqa: BLE001
                    pass

    def _pump(self, stream, channels: int, rate: int) -> str:
        """Feed captured audio through resampler and segmenter until told to stop or reconnect."""
        res = StreamResampler(rate)
        seg = self._seg = Segmenter(float(self.cfg["silence_seconds"]), float(self.cfg.get("early_silence_seconds", 0.35)))
        idle_limit = float(self.cfg.get("reconnect_idle_seconds", 45) or 0)
        self._idle_after = self._idle_after or idle_limit
        last_data = self._last_sound = time.monotonic()
        while not self._stop.is_set():
            if self._reconnect.is_set():
                return "reconnect"
            try:
                raw = self._raw.get(timeout=0.05)
            except queue.Empty:
                raw = None
            now = time.monotonic()
            if self.paused.is_set():
                seg.reset()
                last_data = self._last_sound = now
                continue
            x = np.zeros(0, dtype=np.float32)
            if raw is not None:
                last_data = now
                x = res.process(to_mono(raw, channels))
            events = seg.feed(x, idle_sec=0.0 if raw is not None else now - last_data)
            if seg.level > 0.0005:
                self._last_sound = now
                self._idle_after = idle_limit
            for ev in events:
                self._board.put(_Job(ev.kind, ev.utt, ev.audio, ev.voiced_sec, now - ev.quiet_sec))
            try:
                alive = stream.is_active()
            except Exception:  # noqa: BLE001
                alive = False
            if not alive:
                self._error = "Audio device stopped; reconnecting…"
                self._announce()
                return "reconnect"
            if idle_limit and not seg.in_speech and now - self._last_sound >= self._idle_after:
                # Nothing has played for a long while. Quietly re-open: if the default output device
                # changed (headphones, Bluetooth) this finds the new one. Back off if it didn't.
                self._idle_after = min(self._idle_after * 2, 300.0)
                self._last_sound = now
                return "reconnect"
        return "stop"
