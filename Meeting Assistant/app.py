"""Interview answer overlay: CLI entry point.

  it run                                  launch the live overlay
  it manage                               open the bank manager window
  it banks list                           show banks (* = active)
  it banks add NAME [--from FILE]         new empty bank, or import .md/.csv/.xlsx
  it banks remove NAME [-y]               delete a bank
  it banks use NAME [NAME ...]            choose which banks the overlay matches against
  it banks show NAME                      list a bank's questions with numbers
  it banks suggest NAME [--apply]         suggest other wordings for questions that have few
  it q add BANK                           add a question (prompts for text)
  it q edit BANK N|TEXT [--question ...]  edit a question's fields
  it q remove BANK N|TEXT [-y]            remove a question
  it q suggest BANK N|TEXT [--apply]      suggest other ways an interviewer might word a question
  it test "some question"                 see what the matcher would pick
  it test "Why?" --after "Tell me about a time you led a team."
                                          ...and how it would treat a follow-up to that question
  it devices                              list audio outputs that can be captured
"""
from __future__ import annotations

import argparse
import sys
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
            print(f"{i:>3}. {e.question}{tags}")
    elif a.action == "suggest":
        bk = b.load_bank(a.name)
        embed = _suggest_embedder()
        added = 0
        for i, e in enumerate(bk.entries, 1):
            if len(e.also) >= a.min:
                continue
            new = _suggest_for(e, a.n, embed)
            print(f"{i:>3}. {e.question}")
            for s in new:
                print(f"       + {s}")
            if a.apply and new:
                e.also += new
                added += len(new)
        if a.apply:
            bk.save()
            print(f"\nAdded {added} wordings to '{bk.name}'.")
        else:
            print("\nNothing saved. Add --apply to keep these (you can edit them afterwards in the manager).")


def _suggest_embedder():
    """The matcher's embedding model, used to drop suggestions that drift in meaning. None if unavailable."""
    try:
        from matcher import Matcher
        return Matcher().embed_texts
    except Exception as e:  # noqa: BLE001 - suggestions still work on wording alone
        print(f"(meaning check unavailable: {e})")
        return None


def _suggest_for(entry: b.Entry, n: int, embed) -> list[str]:
    from suggest import suggest_phrasings
    return [s.text for s in suggest_phrasings(entry.question, entry.also, n=n, embed=embed)]


def cmd_q(a) -> None:
    bk = b.load_bank(a.bank)
    split = lambda s, sep: [p.strip() for p in s.split(sep) if p.strip()]
    if a.action == "add":
        q = a.question or input("Question: ").strip()
        resp = a.response or _multiline("Response")
        also = split(a.also or input("Other phrasings, separated by | (optional): "), "|")
        entry = b.Entry(q, resp, also, split(a.tags or "", ","), a.skeleton or "", bk.name)
        bk.entries.append(entry)
        bk.save()
        print(f"Added #{len(bk.entries)} to '{bk.name}'.")
    elif a.action == "edit":
        e = bk.find(a.target)
        if not any([a.question, a.response, a.also, a.tags, a.skeleton]):
            print(f"Editing: {e.question}\nPress Enter to keep a field as is.")
            a.question = input("Question: ").strip()
            a.also = input(f"Other phrasings [{' | '.join(e.also)}]: ").strip()
            if input("Replace the response? [y/N] ").strip().lower() == "y":
                a.response = _multiline("New response")
        if a.question:
            e.question = a.question
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
    elif a.action == "suggest":
        e = bk.find(a.target)
        new = _suggest_for(e, a.n, _suggest_embedder())
        print(f"{e.question}\n  already: {' | '.join(e.also) or '(no other wordings yet)'}")
        for s in new:
            print(f"  + {s}")
        if not new:
            print("  Nothing new to suggest: this one is well covered.")
        elif a.apply:
            e.also += new
            bk.save()
            print(f"Added {len(new)} wordings.")
        else:
            print("Nothing saved. Add --apply to keep these.")


def cmd_test(a) -> None:
    from matcher import Matcher, looks_like_question
    from session import Session
    m = Matcher()
    m.load(b.active_entries())
    cfg = b.load_config()
    thr = cfg["match_threshold"]
    print(f"question-like: {looks_like_question(a.text)}   threshold: {thr:.2f}")
    for r in m.match(a.text, k=5):
        print(f"  {r.score:.3f}  {r.entry.question}")
    session = Session(m, cfg)
    for earlier in a.after or []:
        session.hear(earlier)
    h = session.hear(a.text)
    if a.after:
        print(f"\nafter: {' / '.join(a.after)}")
    if h.kind == "followup":
        f = h.followup
        print(f"-> FOLLOW-UP ({f.signal.label or 'matches one you prepared'}) to: {f.anchor.question}")
        if f.prepared:
            print(f"   prepared ({f.score:.3f}): {f.prepared.question}\n   say: {f.prepared.answer[:160]}")
        for line in f.focus:
            print(f"   from your answer: {line}")
        if h.primary:
            print(f"   (could also be a new question: {h.primary.entry.question})")
    elif h.kind == "new":
        print(f"-> NEW QUESTION: {h.primary.entry.question}")
    else:
        print("-> no confident match")
    if h.uncertain:
        print("   (close call)")


def cmd_devices(_a) -> None:
    from audio import list_loopback_devices
    for d in list_loopback_devices():
        print(f"  {d['name'].replace(' [Loopback]', '')}")
    print('\nTo pin one, set "output_device" in config.json to part of its name.')


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="it", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("run")
    sub.add_parser("manage")
    sub.add_parser("devices")
    t = sub.add_parser("test")
    t.add_argument("text")
    t.add_argument("--after", action="append", help="a question heard just before (repeatable), to test follow-ups")

    bp = sub.add_parser("banks").add_subparsers(dest="action", required=True)
    bp.add_parser("list")
    x = bp.add_parser("add"); x.add_argument("name"); x.add_argument("--from", dest="source"); x.add_argument("--title")
    x = bp.add_parser("remove"); x.add_argument("name"); x.add_argument("-y", "--yes", action="store_true")
    x = bp.add_parser("use"); x.add_argument("names", nargs="+")
    x = bp.add_parser("show"); x.add_argument("name")
    x = bp.add_parser("suggest"); x.add_argument("name")
    x.add_argument("--min", type=int, default=3, help="only questions with fewer than this many other wordings")
    x.add_argument("-n", type=int, default=5, help="wordings to suggest per question")
    x.add_argument("--apply", action="store_true", help="save the suggestions into the bank")

    qp = sub.add_parser("q").add_subparsers(dest="action", required=True)
    for act in ("add", "edit", "remove", "suggest"):
        x = qp.add_parser(act)
        x.add_argument("bank")
        if act != "add":
            x.add_argument("target", help="question number (from 'banks show') or part of its text")
        if act == "remove":
            x.add_argument("-y", "--yes", action="store_true")
        elif act == "suggest":
            x.add_argument("-n", type=int, default=5, help="wordings to suggest")
            x.add_argument("--apply", action="store_true", help="save the suggestions into the question")
        else:
            for f in ("question", "response", "also", "tags", "skeleton"):
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
            {"banks": cmd_banks, "q": cmd_q, "test": cmd_test, "devices": cmd_devices}[a.cmd](a)
    except (KeyError, ValueError) as e:
        sys.exit(f"Error: {e.args[0] if e.args else e}")


if __name__ == "__main__":
    main()
