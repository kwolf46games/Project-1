"""Resampler and segmenter: the signal-processing half of audio.py, tested on synthetic audio."""
import numpy as np
import pytest

from audio import Segmenter, StreamResampler, prepare_for_asr, to_mono

scipy_signal = pytest.importorskip("scipy.signal")
RNG = np.random.default_rng(1)


# ---------- resampler ----------

def reference(x, rate):
    from math import gcd
    g = gcd(rate, 16000)
    return scipy_signal.resample_poly(x, 16000 // g, rate // g).astype(np.float32)


@pytest.mark.parametrize("rate", [48000, 44100, 96000, 32000, 22050, 24000, 8000])
def test_matches_scipy(rate):
    x = (RNG.standard_normal(rate) * 0.1).astype(np.float32)
    r = StreamResampler(rate)
    y = r.process(x)
    ref = reference(x, rate)
    d = r.delay
    n = min(len(y) - d, len(ref)) - 2 * d       # compare away from the edges (scipy zero-pads differently)
    assert n > 1000
    err = np.max(np.abs(y[d + d: d + d + n] - ref[d: d + n]))
    assert err < 2e-4, err


@pytest.mark.parametrize("rate", [48000, 44100])
def test_chunking_does_not_change_output(rate):
    x = (RNG.standard_normal(rate * 2) * 0.1).astype(np.float32)
    whole = StreamResampler(rate).process(x)
    r = StreamResampler(rate)
    parts, i = [], 0
    while i < len(x):
        step = int(RNG.integers(1, 5000))      # includes tiny and odd chunks
        parts.append(r.process(x[i:i + step]))
        i += step
    pieces = np.concatenate(parts)
    assert len(pieces) == len(whole)
    assert np.max(np.abs(pieces - whole)) < 1e-6


@pytest.mark.parametrize("rate", [48000, 44100])
def test_alias_rejection_and_passband(rate):
    t = np.arange(rate) / rate
    in_band = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5
    ultrasonic = np.sin(2 * np.pi * 10500 * t).astype(np.float32) * 0.5     # folds to 5.5 kHz if not filtered
    y = StreamResampler(rate).process(in_band)[400:-400]
    assert abs(np.sqrt(np.mean(y ** 2)) - 0.5 / np.sqrt(2)) < 0.01          # unity gain in the speech band
    z = StreamResampler(rate).process(ultrasonic)[400:-400]
    assert np.sqrt(np.mean(z ** 2)) < 0.005                                  # >35 dB down


def test_output_length_and_passthrough():
    r = StreamResampler(44100)
    n = sum(len(r.process(np.zeros(441, dtype=np.float32))) for _ in range(1000))
    assert abs(n - 160000) <= 2 * r.taps
    p = StreamResampler(16000)
    x = RNG.standard_normal(100).astype(np.float32)
    assert p.passthrough and np.array_equal(p.process(x), x)
    assert len(StreamResampler(48000).process(np.zeros(0, dtype=np.float32))) == 0


def test_to_mono_edge_cases():
    st = (np.array([[1000, 3000], [-2000, 2000]], dtype=np.int16)).tobytes()
    assert np.allclose(to_mono(st, 2), [2000 / 32768, 0.0])
    assert len(to_mono(st + b"\x01", 2)) == 2                      # stray byte
    assert len(to_mono(st[:6], 2)) == 1                            # ragged frame
    assert len(to_mono(b"", 2)) == 0


def test_prepare_for_asr_levels():
    quiet = np.full(100, 0.01, dtype=np.float32)
    assert 0.1 < np.max(prepare_for_asr(quiet)) <= 0.13            # capped at 12x
    mid = np.full(100, 0.1, dtype=np.float32)
    assert abs(np.max(prepare_for_asr(mid)) - 0.6) < 1e-6
    loud = np.full(100, 0.6, dtype=np.float32)
    assert np.array_equal(prepare_for_asr(loud), loud)
    assert np.array_equal(prepare_for_asr(np.zeros(10, dtype=np.float32)), np.zeros(10))


# ---------- segmenter ----------

def voice(sec, amp=0.05):
    n = int(sec * 16000)
    return (RNG.standard_normal(n) * amp * (0.6 + 0.4 * np.sin(np.arange(n) / 700))).astype(np.float32)


def quiet(sec, amp=0.0):
    return (RNG.standard_normal(int(sec * 16000)) * amp).astype(np.float32)


def run(parts, early=0.35, silence=0.7, chunk=480, tail=2.0, **kw):
    """Feed `parts` in real-time sized chunks. Returns [(audio_time_s, Event)], audio_time = end of the chunk."""
    seg = Segmenter(silence, early, **kw)
    x = np.concatenate(parts + [quiet(tail)])
    out = []
    for i in range(0, len(x), chunk):
        for ev in seg.feed(x[i:i + chunk]):
            out.append(((i + chunk) / 16000, ev))
    return out, seg


def kinds(out):
    return [(e.kind, e.utt) for _, e in out]


def test_one_utterance_early_then_commit():
    out, _ = run([quiet(0.5), voice(2.0)])
    assert kinds(out) == [("spec", 1), ("commit", 1)]
    t_spec, spec = out[0]
    t_commit, _ = out[1]
    assert 2.5 + 0.30 <= t_spec <= 2.5 + 0.45            # speech ends at 2.5 s; early guess ~0.35 s later
    assert 2.5 + 0.65 <= t_commit <= 2.5 + 0.80          # commit at the full 0.7 s
    assert 2.0 + 0.3 <= len(spec.audio) / 16000 <= 2.0 + 0.3 + 0.3   # speech + pre-roll + pad, not the whole silence


def test_pause_then_resume_supersedes_early_guess():
    out, _ = run([voice(1.5), quiet(0.5), voice(1.5)])
    # the 0.5 s mid-sentence pause earns an early guess; the real end earns a second one, then the commit
    assert kinds(out) == [("spec", 1), ("spec", 1), ("commit", 1)]
    first, second = out[0][1].audio, out[1][1].audio
    assert len(first) / 16000 < 2.3 and len(second) / 16000 > 3.4     # the later guess holds both halves


def test_short_blips_ignored():
    out, _ = run([quiet(0.5), voice(0.1, 0.2), quiet(1.5)])
    assert out == []


def test_short_phrase_gets_a_final_without_early_guess():
    out, _ = run([voice(0.6)])                           # voiced < 0.9 s: no speculation, but still speech
    assert kinds(out) == [("final", 1)]


def test_two_utterances_in_order():
    out, _ = run([voice(1.5), quiet(1.2), voice(1.6)])
    assert kinds(out) == [("spec", 1), ("commit", 1), ("spec", 2), ("commit", 2)]


def test_quiet_speech_is_heard():
    """Speech at rms ~0.006 sat below the old fixed 0.008 gate and was never transcribed."""
    out, _ = run([quiet(1.0, 0.0003), voice(2.0, 0.009)])
    assert [k for k, _ in kinds(out)] == ["spec", "commit"]


def test_steady_hiss_is_ignored_then_speech_found():
    out, seg = run([quiet(6.0, 0.002), voice(2.0, 0.05)])
    assert kinds(out) == [("spec", 1), ("commit", 1)]
    assert out[0][0] > 6.0


def test_hysteresis_keeps_utterance_together():
    dip = quiet(0.15, 0.0038)    # a soft consonant cluster / breath: between the off and on thresholds
    out, _ = run([quiet(1.0, 0.0003), voice(1.5), dip, voice(1.5)])
    assert [k for k, _ in kinds(out)] == ["spec", "commit"] and out[0][1].utt == 1


def test_forced_split_keeps_everything():
    out, _ = run([voice(31.0)])
    assert kinds(out) == [("final", 1), ("spec", 2), ("commit", 2)]      # 25 s cut, then the 6 s remainder
    first, rest = out[0][1].audio, out[1][1].audio                        # (a commit reuses the spec's audio)
    assert len(first) / 16000 <= 25.5
    assert (len(first) + len(rest)) / 16000 >= 31.0                       # nothing lost (the overlap adds a little)


def test_wall_clock_idle_when_loopback_goes_quiet():
    seg = Segmenter(0.7, 0.35)
    evs = []
    x = voice(2.0)
    for i in range(0, len(x), 480):
        evs += seg.feed(x[i:i + 480])
    assert evs == []
    evs = seg.feed(np.zeros(0, dtype=np.float32), idle_sec=0.2)
    assert evs == []
    evs = seg.feed(np.zeros(0, dtype=np.float32), idle_sec=0.4)
    assert [e.kind for e in evs] == ["spec"]
    assert seg.feed(np.zeros(0, dtype=np.float32), idle_sec=0.5) == []        # no second guess for the same audio
    evs = seg.feed(np.zeros(0, dtype=np.float32), idle_sec=0.8)
    assert [e.kind for e in evs] == ["commit"]


def test_burst_of_buffered_audio_does_not_merge_utterances():
    seg = Segmenter(0.7, 0.35)
    x = np.concatenate([voice(1.5), quiet(1.0), voice(1.5), quiet(1.5)])
    evs = seg.feed(x)                                     # everything at once, as after a stall
    assert [(e.kind, e.utt) for e in evs] == [("spec", 1), ("commit", 1), ("spec", 2), ("commit", 2)]


def test_early_guess_can_be_turned_off():
    out, _ = run([voice(2.0)], early=0)
    assert kinds(out) == [("final", 1)]


def test_uneven_chunk_sizes():
    seg = Segmenter()
    x = np.concatenate([voice(2.0), quiet(2.0)])
    evs, i = [], 0
    while i < len(x):
        step = int(RNG.integers(1, 3000))
        evs += seg.feed(x[i:i + step])
        i += step
    assert [e.kind for e in evs] == ["spec", "commit"]


def test_reset_discards_partial_utterance():
    seg = Segmenter()
    seg.feed(voice(1.0))
    assert seg.in_speech
    seg.reset()
    assert not seg.in_speech and seg.feed(quiet(2.0)) == []
