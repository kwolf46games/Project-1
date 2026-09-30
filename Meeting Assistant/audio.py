"""Capture system audio output (WASAPI loopback), cut it into utterances, transcribe locally.

Only what your speakers/headphones play is captured, so your own mic never gets in.
Audio and transcripts stay in memory; nothing is written to disk.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

from bank import ROOT

MODELS_DIR = ROOT / "models"
TARGET_RATE = 16000
FRAME_SEC = 0.03
MIN_UTTERANCE_SEC = 0.6
MAX_UTTERANCE_SEC = 25.0


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


def _to_mono_16k(raw: bytes, channels: int, rate: int) -> np.ndarray:
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate == TARGET_RATE:
        return x
    if rate % TARGET_RATE == 0:  # e.g. 48k -> 16k: block-average doubles as a low-pass
        f = rate // TARGET_RATE
        n = len(x) // f * f
        return x[:n].reshape(-1, f).mean(axis=1)
    n_out = int(len(x) * TARGET_RATE / rate)
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


class Listener:
    """Background thread: loopback audio -> energy-based segmentation -> faster-whisper."""

    def __init__(self, cfg: dict, on_text: Callable[[str], None], on_status: Callable[[str], None]):
        self.cfg = cfg
        self.on_text = on_text
        self.on_status = on_status
        self.paused = threading.Event()
        self._stop = threading.Event()
        self._chunks: queue.Queue[np.ndarray] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.model = None
        self.device_name = ""

    # ----- lifecycle -----
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def restart(self) -> None:
        """Re-open the audio device (e.g. after switching headphones)."""
        self.stop()
        if self._thread:
            self._thread.join(timeout=3)
        self._stop.clear()
        self._chunks = queue.Queue()
        self.start()

    def load_model(self) -> None:
        if self.model is not None:
            return
        from faster_whisper import WhisperModel
        name = self.cfg["whisper_model"]
        self.on_status(f"Loading speech model '{name}' (first run downloads it)…")
        MODELS_DIR.mkdir(exist_ok=True)
        self.model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=8,
                                  download_root=str(MODELS_DIR))

    # ----- worker -----
    def _run(self) -> None:
        import pyaudiowpatch as pyaudio
        try:
            self.load_model()
        except Exception as e:  # noqa: BLE001
            self.on_status(f"Speech model failed to load: {e}")
            return
        p = pyaudio.PyAudio()
        try:
            dev = _pick_loopback(p, self.cfg.get("output_device", ""))
            self.device_name = dev["name"].replace(" [Loopback]", "")
            channels, rate = int(dev["maxInputChannels"]), int(dev["defaultSampleRate"])

            def callback(data, frames, time_info, status):
                self._chunks.put(_to_mono_16k(data, channels, rate))
                return (None, pyaudio.paContinue)

            stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                            input_device_index=dev["index"], frames_per_buffer=int(rate * FRAME_SEC),
                            stream_callback=callback)
        except Exception as e:  # noqa: BLE001
            p.terminate()
            self.on_status(f"Couldn't open audio output for capture: {e}")
            return

        self.on_status(f"Listening · {self.device_name}")
        try:
            self._segment_loop()
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _segment_loop(self) -> None:
        frame_len = int(TARGET_RATE * FRAME_SEC)
        buf = np.zeros(0, dtype=np.float32)      # unframed leftovers
        speech: list[np.ndarray] = []            # current utterance
        pre_roll: list[np.ndarray] = []          # a little audio before speech starts
        noise_floor = 0.003
        last_voice = 0.0
        silence_sec = float(self.cfg["silence_seconds"])

        while not self._stop.is_set():
            try:
                buf = np.concatenate([buf, self._chunks.get(timeout=0.1)])
            except queue.Empty:
                pass  # loopback delivers nothing while the system is silent; the clock still runs

            while len(buf) >= frame_len:
                frame, buf = buf[:frame_len], buf[frame_len:]
                if self.paused.is_set():
                    continue
                rms = float(np.sqrt(np.mean(frame ** 2)))
                voiced = rms > max(0.008, noise_floor * 3.0)
                if not voiced:
                    noise_floor = 0.995 * noise_floor + 0.005 * rms
                if voiced:
                    if not speech:
                        speech.extend(pre_roll)
                    speech.append(frame)
                    last_voice = time.monotonic()
                elif speech:
                    speech.append(frame)
                else:
                    pre_roll = (pre_roll + [frame])[-10:]

            if self.paused.is_set():
                speech, pre_roll = [], []
                continue

            dur = len(speech) * FRAME_SEC
            ended = speech and time.monotonic() - last_voice >= silence_sec
            if ended or dur >= MAX_UTTERANCE_SEC:
                audio = np.concatenate(speech)
                speech, pre_roll = [], []
                if dur >= MIN_UTTERANCE_SEC:
                    self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> None:
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=True,
            condition_on_previous_text=False, without_timestamps=True,
            initial_prompt=self.cfg.get("whisper_prompt") or None,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            self.on_text(text)
