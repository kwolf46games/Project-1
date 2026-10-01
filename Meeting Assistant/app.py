"""Interview answer overlay: CLI entry point.

  it run                                  launch the live overlay
  it manage                               open the bank manager window
  it banks list                           show banks (* = active)
  it banks add NAME [--from FILE]         new empty bank, or import .md/.csv/.xlsx
  it banks remove NAME [-y]               delete a bank
  it banks use NAME [NAME ...]            choose which banks the overlay matches against
  it banks show NAME                      list a bank's questions with numbers
  it q add BANK                           add a question (prompts for text)
  it q edit BANK N|TEXT [--question ...]  edit a question's fields (--follows "Parent question" links a follow-up)
  it q remove BANK N|TEXT [-y]            remove a question
  it test "some question"                 see what the matcher would pick
  it test "what was the outcome?" --after "Tell me about yourself"
                                          ...as a follow-up to an answer that was just shown
  it suggest BANK [N|TEXT] [--apply]      suggest extra phrasings for each question (--apply adds the safe ones)
  it generate "a question"                draft an answer from your banks (needs the anthropic package + an API key)
  it generate --check                     live check of your API key and connection
  it devices                              list audio outputs that can be captured
  it audiotest [--seconds 10]             check live that audio arrives and is turned into text
  it transcribe FILE.wav [--match]        run a recording through the same audio -> text -> answer chain
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import bank as b


def _confirm(msg: str, yes: bool) -> bool:
    return yes or input(f"{msg} [y/N] ").strip().lower() == "y"


def _multiline(prompt: str) -> str:
    print(f"{prompt} (finish with a line containing only a period '.')")
    lines = []
    while (line := input()) != ".":
        lines.append(line)
    return "\n".join(lines).strip()


def cmd_banks(a) -> None:
    if a.action == "list":
        cfg = b.load_config()
        names = b.bank_names()
        if not names:
            print("No banks yet. Create one with: it banks add NAME [--from FILE]")
        for n in names:
            bk = b.load_bank(n)
            star = "*" if n in cfg["active_banks"] else " "
            print(f" {star} {n:<28} {len(bk.entries):>3} questions   {bk.title}")
    elif a.action == "add":
        bk = b.add_bank(a.name, Path(a.source) if a.source else None, a.title or "")
        print(f"Created '{bk.name}' with {len(bk.entries)} questions -> {bk.path}")
        if a.source and not bk.entries:
            print("No questions were found in that file. See README for the layouts that import.")
    elif a.action == "remove":
        bk = b.load_bank(a.name)
        if _confirm(f"Delete bank '{a.name}' ({len(bk.entries)} questions)? It moves to banks/deleted and can be restored.", a.yes):
            b.remove_bank(a.name)
            print(f"Removed '{a.name}'.")
    elif a.action == "use":
        b.use_banks(a.names)
        print("Active banks:", ", ".join(a.names))
    elif a.action == "show":
        bk = b.load_bank(a.name)
        for i, e in enumerate(bk.entries, 1):
            tags = f"  [{', '.join(e.tags)}]" if e.tags else ""
            fu = f"   (follows: {' | '.join(e.follows)})" if e.follows else ""
            print(f"{i:>3}. {e.question}{tags}{fu}")


def cmd_q(a) -> None:
    bk = b.load_bank(a.bank)
    split = lambda s, sep: [p.strip() for p in s.split(sep) if p.strip()]
    if a.action == "add":
        q = a.question or input("Question: ").strip()
        resp = a.response or _multiline("Response")
        also = split(a.also or input("Other phrasings, separated by | (optional): "), "|")
        entry = b.Entry(q, resp, also, split(a.tags or "", ","), a.skeleton or "", bk.name,
                        split(a.follows or "", "|"))
        bk.entries.append(entry)
        bk.save()
        print(f"Added #{len(bk.entries)} to '{bk.name}'.")
    elif a.action == "edit":
        e = bk.find(a.target)
        if not any([a.question, a.response, a.also, a.tags, a.skeleton]) and a.follows is None:
            print(f"Editing: {e.question}\nPress Enter to keep a field as is.")
            a.question = input("Question: ").strip()
            a.also = input(f"Other phrasings [{' | '.join(e.also)}]: ").strip()
            if input("Replace the response? [y/N] ").strip().lower() == "y":
                a.response = _multiline("New response")
        if a.question:
            bk.rename_question(e.question, a.question)   # keep other questions' "follows:" links working
            e.question = a.question
        if a.follows is not None:
            e.follows = split(a.follows, "|")
        if a.response:
            e.response = a.response
        if a.also:
            e.also = split(a.also, "|")
        if a.tags:
            e.tags = split(a.tags, ",")
        if a.skeleton:
            e.skeleton = a.skeleton
        bk.save()
        print(f"Saved: {e.question}")
    elif a.action == "remove":
        e = bk.find(a.target)
        if _confirm(f"Remove '{e.question}' from '{bk.name}'?", a.yes):
            bk.entries.remove(e)
            bk.save()
            print("Removed.")


def cmd_test(a) -> None:
    from followup import Conversation, detect_followup
    from matcher import Matcher, looks_like_question
    m = Matcher()
    entries = b.active_entries()
    m.load(entries)
    cfg = b.load_config()
    thr = cfg["match_threshold"]
    conv = Conversation(m, cfg)
    if a.after:
        hits = [e for e in entries if a.after.lower() in e.question.lower()]
        if len(hits) != 1:
            raise KeyError(f"--after '{a.after}' matches {len(hits)} active questions; be more specific.")
        conv.note(hits[0])
        print(f"pretending this answer is on screen: {hits[0].question}")
    cue = detect_followup(a.text)
    print(f"question-like: {looks_like_question(a.text)}   threshold: {thr:.2f}   "
          f"follow-up wording: {cue.kind if cue else 'none'}")
    for r in m.match(a.text, k=5):
        print(f"  {r.score:.3f}  {r.entry.question}")
    d = conv.resolve(a.text, forced=a.follow)
    shown = {"show": "show the answer", "unsure": "say 'Not sure' and list the closest",
             "followup": "keep the current answer and flag a follow-up", "ignore": "do nothing"}[d.action]
    print(f"overlay would: {shown}")
    if d.action == "show":
        print(f"  -> {d.matches[0].entry.question}" + ("   (as a follow-up)" if d.is_followup else ""))
    if d.banner:
        print(f"  {d.banner}")
    if d.hint:
        print(f"  {d.hint}")


def cmd_suggest(a) -> None:
    import suggest as sg
    bk = b.load_bank(a.bank)
    targets = [bk.find(a.target)] if a.target else bk.entries
    matcher = None
    if not a.no_check:
        try:
            from matcher import Matcher
            matcher = Matcher()
            matcher.load(b.active_entries())
        except Exception as e:  # noqa: BLE001
            print(f"(couldn't load the model to check suggestions against your other questions: {e}\n"
                  " showing them unchecked; use --no-check to skip this step)\n")
    added = 0
    for e in targets:
        items = sg.generate(e.question, e.phrasings(), limit=a.max)
        if matcher:
            items = sg.vet(items, e, matcher)
        if not items:
            continue
        print(f"{e.question}")
        for s in items:
            mark = "x" if s.rival else "+"
            note = f"   (would match “{s.rival}” instead)" if s.rival else f"   ({s.sim:.0%})" if s.sim is not None else ""
            print(f"  {mark} {s.text}{note}")
        if a.apply:
            for s in items:
                if s.ok:
                    e.also.append(s.text)
                    added += 1
    if a.apply:
        if added:
            bk.save()
        print(f"\nAdded {added} phrasing{'s' if added != 1 else ''} to '{bk.name}'." if added else "\nNothing to add.")
    else:
        print("\n+ = safe to add, x = could put the wrong answer up.  Add the + ones with --apply, or tick them in "
              "'Manage banks'.")


def cmd_generate(a) -> None:
    import threading

    import generate as gen
    cfg = b.load_config()
    done = threading.Event()
    started = time.monotonic()
    state = {"first": None, "error": ""}

    def on_event(job, kind, data) -> None:
        if kind == "start":
            print(f"({data})\n")
        elif kind == "delta":
            if state["first"] is None:
                state["first"] = time.monotonic() - started
            print(data, end="", flush=True)
        elif kind == "done":
            print(f"\n\n[{data['seconds']:.1f}s total, first words after {data['first']:.1f}s; "
                  f"{data['input_tokens']} tokens in, {data['output_tokens']} out"
                  + (f", {data['cached_tokens']} from cache" if data["cached_tokens"] else "")
                  + (", cut short" if data["truncated"] else "") + "]")
            done.set()
        elif kind == "error":
            state["error"] = data
            print(f"\n{data}")
            done.set()
    g = gen.Generator(cfg, on_event)
    if a.check:
        ok, why = g.status()
        key = ("ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY")
               else f"{gen.KEY_FILE.name}" if gen.KEY_FILE.exists() else "none found (the SDK will try its own sources)")
        print(f"package: {'ok' if ok or 'package' not in why else 'MISSING - ' + why}\nkey: {key}\nmodel: {g.model}")
        if not ok:
            print(f"PROBLEM: {why}")
            return
        req = gen.Request([{"type": "text", "text": "Reply with the single word OK."}],
                          [{"role": "user", "content": "ping"}], "connection check", "ping")
        g.cfg = {**cfg, "generate_max_tokens": 200}
        g.start(req)
        done.wait(40)
        print("OK: the API key and connection work." if done.is_set() and not state["error"] else "")
        return
    if not a.text:
        raise ValueError("Give the question to answer, e.g.  it generate \"Tell me about a time you led a team\"")
    ok, why = g.status()
    if not ok:
        raise ValueError(why)
    from followup import Conversation
    from matcher import Matcher
    m = Matcher()
    entries = b.active_entries()
    m.load(entries)
    conv = Conversation(m, cfg)
    if a.after:
        hits = [e for e in entries if a.after.lower() in e.question.lower()]
        if len(hits) != 1:
            raise KeyError(f"--after '{a.after}' matches {len(hits)} active questions; be more specific.")
        conv.note(hits[0])
    d = conv.resolve(a.text)
    matches = d.matches or m.match(a.text, k=3)
    req = gen.build_request(a.text, matches, thr=float(cfg["match_threshold"]), words=g.words,
                            max_matches=int(cfg.get("generate_matches", 3)), profile=g.profile,
                            parent=d.parent if d.is_followup else None, cue=d.cue if d.is_followup else None,
                            recent=conv.recent(2))
    if a.show_prompt:
        print("=== SYSTEM ===")
        for blk in req.system:
            print(blk["text"], "\n")
        print("=== MESSAGE ===")
        print(req.messages[0]["content"], "\n=== END ===\n")
    g.start(req)
    done.wait(90)


def cmd_devices(_a) -> None:
    from audio import list_loopback_devices
    for d in list_loopback_devices():
        print(f"  {d['name'].replace(' [Loopback]', '')}   ({int(d['maxInputChannels'])} ch, {float(d['defaultSampleRate']):.0f} Hz)")
    print('\nTo pin one, set "output_device" in config.json to part of its name.')


def cmd_audiotest(a) -> None:
    import math
    import time
    from audio import Listener
    cfg = b.load_config()
    heard = []

    def on_text(h) -> None:
        heard.append(h)
        print(f"\r  {'heard' if h.final else 'early'}  ({h.lag:.2f}s after you stopped, {h.decode:.2f}s to transcribe): {h.text}")
    lis = Listener(cfg, on_text, lambda s: print(f"\r  status: {s}"))
    print(f"Listening for {a.seconds} seconds. Play some speech through the speakers or headphones you use for\n"
          "meetings (a video with someone talking is perfect).  Ctrl+C stops early.\n")
    lis.start()
    peak, t0, st = 0.0, time.monotonic(), {}
    try:
        while time.monotonic() - t0 < a.seconds:
            time.sleep(0.2)
            st = lis.stats()
            peak = max(peak, st["level"])
            bar = "#" * int(max(0.0, min(1.0, (20 * math.log10(max(st["level"], 1e-5)) + 55) / 50)) * 30)
            print(f"\r  level [{bar:<30}]", end="", flush=True)
    except KeyboardInterrupt:
        pass
    lis.stop()
    st = lis.stats()
    db = 20 * math.log10(max(peak, 1e-5))
    print(f"\n\ndevice: {st['device'] or '(none opened)'}   speech model ready: {st['model_ready']}   "
          f"loudest: {db:.0f} dBFS   utterances heard: {sum(1 for h in heard if h.final)}   "
          f"dropped audio chunks: {st['dropped']}   reconnects: {st['reconnects']}")
    if st["error"]:
        print(f"PROBLEM: {st['error']}")
    elif not st["model_ready"]:
        print("PROBLEM: the speech model didn't finish loading. First run needs internet to download it.")
    elif not st["device"]:
        print("PROBLEM: no audio output could be opened. Run 'devices' and set output_device in config.json.")
    elif peak < 0.0005:
        print("PROBLEM: no sound reached the capture. Is the meeting audio playing through that device?  "
              "(Windows sound settings -> Output; or set output_device in config.json.)")
    elif not heard:
        print("Audio arrived but nothing was transcribed. Was it speech?  Music and effects are ignored on purpose.")
    else:
        print("OK: audio is captured and turned into text.")


def cmd_transcribe(a) -> None:
    import wave

    import numpy as np
    from audio import Listener, Segmenter, StreamResampler, Transcriber, to_mono
    try:
        w = wave.open(a.file, "rb")
    except (wave.Error, EOFError, FileNotFoundError) as e:
        raise ValueError(f"Can't read {a.file}: {e}. Use a 16-bit PCM .wav "
                         "(convert with: ffmpeg -i in.mp4 -ac 1 -ar 16000 out.wav)")
    with w:
        if w.getsampwidth() != 2:
            raise ValueError("Need a 16-bit PCM .wav (convert with: ffmpeg -i in.mp4 -ac 1 -ar 16000 out.wav)")
        channels, rate, raw = w.getnchannels(), w.getframerate(), w.readframes(w.getnframes())
    x = StreamResampler(rate).process(to_mono(raw, channels))
    x = np.concatenate([x, np.zeros(3 * 16000, dtype=np.float32)])
    cfg = b.load_config()
    lis = Listener(cfg, lambda h: None, lambda s: print(f"  {s}"))
    lis.load_model()
    tr = Transcriber(lis.model, lis._prompt)
    matcher = None
    if a.match:
        from matcher import Matcher
        matcher = Matcher()
        matcher.load(b.active_entries())
    seg = Segmenter(float(cfg["silence_seconds"]), 0.0)
    done = 0.0
    for i in range(0, len(x), 480):
        for ev in seg.feed(x[i:i + 480]):
            t0 = time.monotonic()
            text = tr.decode(ev.audio, True, ev.voiced_sec)
            took = time.monotonic() - t0
            print(f"[{(i / 16000):6.1f}s]  {text or '(no speech)'}   ({took:.2f}s to transcribe)")
            if matcher and text:
                best = matcher.match(text, k=1)
                if best:
                    print(f"           -> {best[0].score:.0%}  {best[0].entry.question}")
            done += 1
    print(f"\n{int(done)} utterance{'s' if done != 1 else ''} found in {len(x) / 16000 - 3:.1f}s of audio.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="it", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("run")
    sub.add_parser("manage")
    sub.add_parser("devices")
    t = sub.add_parser("test")
    t.add_argument("text")
    t.add_argument("--after", help="part of the question whose answer is on screen (to test a follow-up)")
    t.add_argument("--follow", action="store_true", help="treat it as a follow-up even if it doesn't sound like one")
    x = sub.add_parser("suggest")
    x.add_argument("bank")
    x.add_argument("target", nargs="?", help="question number or part of its text (default: every question)")
    x.add_argument("--apply", action="store_true", help="add the safe suggestions to the bank")
    x.add_argument("--max", type=int, default=5, help="suggestions per question (default 5)")
    x.add_argument("--no-check", action="store_true", help="skip checking against your other questions (faster)")
    x = sub.add_parser("generate")
    x.add_argument("text", nargs="?", help="the question to answer")
    x.add_argument("--after", help="part of the question whose answer is on screen (to draft a follow-up)")
    x.add_argument("--show-prompt", action="store_true", help="print exactly what is sent to the API")
    x.add_argument("--check", action="store_true", help="test the API key and connection with a tiny request")
    x = sub.add_parser("audiotest")
    x.add_argument("--seconds", type=int, default=10)
    x = sub.add_parser("transcribe")
    x.add_argument("file")
    x.add_argument("--match", action="store_true", help="also show which bank question each utterance matches")

    bp = sub.add_parser("banks").add_subparsers(dest="action", required=True)
    bp.add_parser("list")
    x = bp.add_parser("add"); x.add_argument("name"); x.add_argument("--from", dest="source"); x.add_argument("--title")
    x = bp.add_parser("remove"); x.add_argument("name"); x.add_argument("-y", "--yes", action="store_true")
    x = bp.add_parser("use"); x.add_argument("names", nargs="+")
    x = bp.add_parser("show"); x.add_argument("name")

    qp = sub.add_parser("q").add_subparsers(dest="action", required=True)
    for act in ("add", "edit", "remove"):
        x = qp.add_parser(act)
        x.add_argument("bank")
        if act != "add":
            x.add_argument("target", help="question number (from 'banks show') or part of its text")
        if act == "remove":
            x.add_argument("-y", "--yes", action="store_true")
        else:
            for f in ("question", "response", "also", "follows", "tags", "skeleton"):
                x.add_argument(f"--{f}")

    a = p.parse_args(argv)
    try:
        if a.cmd == "manage":
            from manager import BankManager
            BankManager().run()
        elif a.cmd in (None, "run"):
            from overlay import Overlay
            Overlay().run()
        else:
            {"banks": cmd_banks, "q": cmd_q, "test": cmd_test, "devices": cmd_devices, "suggest": cmd_suggest,
             "generate": cmd_generate, "audiotest": cmd_audiotest, "transcribe": cmd_transcribe}[a.cmd](a)
    except (KeyError, ValueError) as e:
        sys.exit(f"Error: {e.args[0] if e.args else e}")


if __name__ == "__main__":
    main()
