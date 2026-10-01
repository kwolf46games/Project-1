"""Always-on-top overlay: shows the best-matching prepared response for what was just asked."""
from __future__ import annotations

import ctypes

WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011

def prevent_window_capture(hwnd):
    """Applies the WDA_EXCLUDEFROMCAPTURE flag to the given HWND."""
    user32 = ctypes.windll.user32
    result = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    return result != 0



import gc
import math
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import bank as bankmod
from audio import Heard, Listener
from followup import STRONG_MARGIN, Conversation, Decision
from generate import Generator, build_request, plain_prose
from matcher import Match, Matcher, bank_vocabulary, looks_like_question
from theme import ACCENT, BAD, BG, FG, GOOD, HOVER_BG, MUTED, PANEL, PLACEHOLDER_BG, WARN, enable_dpi_awareness

FOLLOW_ARM_SEC = 20  # how long the "↳ Follow-up" button stays armed


class Overlay:
    def __init__(self) -> None:
        self.cfg = bankmod.load_config()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.matcher: Matcher | None = None
        self.conv: Conversation | None = None
        self.current: list[Match] = []
        self.heard_q: queue.Queue[Heard | None] = queue.Queue()
        self._early_shown: set[int] = set()   # utterances whose answer an early guess already put on screen
        self._forced_until = 0.0              # "↳ Follow-up" button armed until this time
        self._unsure_text = ""                # what was heard when we weren't sure (for "Remember this wording")
        self.matcher_error = ""               # why matching is unavailable (shown instead of "Listening")
        self._timers: list[str] = []
        # drafted ("generated") answer shown beside the prepared one
        self.gen: Generator | None = None
        self._prepared_md = ""
        self._prepared_note = ""              # shown in the prepared column when nothing prepared fits
        self.gen_text = ""
        self.gen_state = "idle"               # idle | drafting | done | error | off
        self.gen_note = ""
        self.gen_summary = ""
        self.gen_seconds = 0.0
        self.gen_model = ""
        self.gen_truncated = False
        self._gen_floor = 0                   # events from jobs up to this id are stale
        self._gen_job = 0
        self._gen_seen = 0
        self._last_req = None
        self._render_pending = False
        self._gen_warned = ""

        self._reload_lock = threading.Lock()

        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.title("Interview Answer Overlay")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", float(self.cfg["opacity"]))
        self.scale = self.root.winfo_fpixels("1i") / 96.0
        w = min(int(1240 * self.scale), self.root.winfo_screenwidth() - 48)
        h = int(760 * self.scale)
        self.root.geometry(f"{w}x{h}+{self.root.winfo_screenwidth() - w - 24}+40")
        self.root.minsize(int(760 * self.scale), int(420 * self.scale))

        self.base = tkfont.Font(family="Segoe UI", size=self.cfg["font_size"])
        self.bold = tkfont.Font(family="Segoe UI", size=self.cfg["font_size"], weight="bold")
        self.small = tkfont.Font(family="Segoe UI", size=max(9, self.cfg["font_size"] - 3))
        self.title_f = tkfont.Font(family="Segoe UI Semibold", size=self.cfg["font_size"] + 2)

        self._build()
        self.gen = Generator(self.cfg, lambda job, kind, data: self.events.put(("gen", (job, kind, data))))
        if self.gen.status()[0]:
            self.gen.warm()
        self._refresh_gen_ui()
        self._render_gen()
        self.listener = Listener(self.cfg, self.heard_q.put, lambda s: self.events.put(("status", s)))
        self._reload_banks()
        self._match_thread = threading.Thread(target=self._match_loop, daemon=True)
        self._match_thread.start()
        self.listener.start()
        self._timers = [self.root.after(20, self._pump), self.root.after(250, self._tick)]
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ---------- layout ----------
    def _btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, bg=PANEL, fg=FG, activebackground=HOVER_BG,
                         activeforeground=FG, relief="flat", bd=0, padx=8, pady=3, font=self.small,
                         cursor="hand2")

    def _build(self) -> None:
        status_row = tk.Frame(self.root, bg=BG)
        status_row.pack(fill="x", padx=10, pady=(8, 2))
        self.dot = tk.Label(status_row, text="●", fg=WARN, bg=BG, font=self.small)
        self.dot.pack(side="left")
        self.status = tk.Label(status_row, text="Starting…", fg=MUTED, bg=BG, font=self.small, anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.quiet = tk.Label(status_row, text="", fg=MUTED, bg=BG, font=self.small)
        self.quiet.pack(side="right", padx=(4, 0))
        mw, mh = int(64 * self.scale), int(8 * self.scale)
        self.meter = tk.Canvas(status_row, width=mw, height=mh, bg=PANEL, highlightthickness=0)
        self.meter.pack(side="right", padx=(4, 0))
        self._meter_bar = self.meter.create_rectangle(0, 0, 0, mh, fill=GOOD, width=0)
        self._meter_w = mw

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=10, pady=(0, 4))
        self._btn(top, "A+", lambda: self._font(1)).pack(side="right", padx=2)
        self._btn(top, "A−", lambda: self._font(-1)).pack(side="right", padx=2)
        self.banks_btn = tk.Menubutton(top, text="Banks ▾", bg=PANEL, fg=FG, activebackground=HOVER_BG,
                                       activeforeground=FG, relief="flat", font=self.small, padx=8, pady=3)
        self.banks_menu = tk.Menu(self.banks_btn, tearoff=False, postcommand=self._fill_banks_menu)
        self.banks_btn["menu"] = self.banks_menu
        self.banks_btn.pack(side="right", padx=2)
        self._btn(top, "↻ Audio", self._restart_audio).pack(side="right", padx=2)
        self.pause_btn = self._btn(top, "❚❚ Pause", self._toggle_pause)
        self.pause_btn.pack(side="right", padx=2)
        self.follow_btn = self._btn(top, "↳ Follow-up", self._toggle_follow)
        self.follow_btn.pack(side="right", padx=2)
        self.gen_btn = self._btn(top, "✨ Draft: on", self._toggle_gen)
        self.gen_btn.pack(side="right", padx=2)

        search = tk.Frame(self.root, bg=BG)
        search.pack(side="bottom", fill="x", padx=10, pady=(2, 10))
        tk.Label(search, text="Search", fg=MUTED, bg=BG, font=self.small).pack(side="left")
        self.search = tk.Entry(search, bg=PANEL, fg=FG, insertbackground=FG, relief="flat", font=self.small)
        self.search.pack(side="left", fill="x", expand=True, padx=6, ipady=3)
        self.alts = tk.Frame(self.root, bg=BG)
        self.alts.pack(side="bottom", fill="x", padx=10, pady=(6, 2))
        self.likely = tk.Frame(self.root, bg=BG)
        self.likely.pack(side="bottom", fill="x", padx=10, pady=(4, 0))

        # two panes: what was asked (left) beside the answer (right); drag the divider to taste
        self.paned = tk.PanedWindow(self.root, orient="horizontal", bg=BG, bd=0, sashwidth=int(6 * self.scale),
                                    sashrelief="flat", opaqueresize=True)
        self.paned.pack(fill="both", expand=True, padx=10)
        self.left = tk.Frame(self.paned, bg=BG)
        self.mid = tk.Frame(self.paned, bg=BG)
        self.right = tk.Frame(self.paned, bg=BG)
        self.paned.add(self.left, minsize=int(190 * self.scale), width=int(290 * self.scale), stretch="never")
        self.paned.add(self.mid, minsize=int(230 * self.scale), stretch="always")
        self.paned.add(self.right, minsize=int(230 * self.scale), stretch="always")

        tk.Label(self.left, text="THEY ASKED", fg=MUTED, bg=BG, font=self.small, anchor="w").pack(fill="x", padx=(2, 8))
        self.heard = tk.Label(self.left, text="Heard: —", fg=FG, bg=BG, font=self.title_f,
                              anchor="nw", justify="left", wraplength=280)
        self.heard.pack(fill="x", padx=(2, 8), pady=(2, 10))

        tk.Label(self.left, text="CLOSEST PREPARED QUESTION", fg=MUTED, bg=BG, font=self.small, anchor="w").pack(
            fill="x", padx=(2, 8))
        head = tk.Frame(self.left, bg=BG)
        head.pack(fill="x", padx=(2, 8), pady=(2, 2))
        self.conf = tk.Label(head, text="", fg=BG, bg=MUTED, font=self.small, padx=6)
        self.conf.pack(side="right", anchor="n")
        self.question = tk.Label(head, text="Waiting for a question…", fg=FG, bg=BG, font=self.bold,
                                 anchor="w", justify="left", wraplength=220)
        self.question.pack(side="left", fill="x", expand=True)

        self.notes = tk.Frame(self.left, bg=BG)   # follow-up banner + hint; takes no room while empty
        self.notes.pack(fill="x", padx=(2, 8))
        self.banner = tk.Label(self.notes, text="", fg=WARN, bg=BG, font=self.small, anchor="w",
                               justify="left", wraplength=280)
        self.hint = tk.Label(self.notes, text="", fg=MUTED, bg=BG, font=self.small, anchor="w",
                             justify="left", wraplength=280)

        self.skeleton = tk.Label(self.left, text="", fg=ACCENT, bg=BG, font=self.small, anchor="w",
                                 justify="left", wraplength=280)
        self.skeleton.pack(fill="x", padx=(2, 8), pady=(0, 6))

        # both answer columns share one layout (header, answer, status line) so they line up when read side by side
        rbar = tk.Frame(self.right, bg=BG)
        rbar.pack(fill="x", pady=(0, 4))
        tk.Label(rbar, text="✨ GENERATED ANSWER", fg=MUTED, bg=BG, font=self.small, anchor="w").pack(side="left", padx=(4, 0))
        self.save_btn = self._btn(rbar, "＋ Save", self._save_generated)
        self.save_btn.pack(side="right", padx=(2, 0))
        self.regen_btn = self._btn(rbar, "↻", self._regenerate)
        self.regen_btn.pack(side="right", padx=2)
        mbar = tk.Frame(self.mid, bg=BG)
        mbar.pack(fill="x", pady=(0, 4))
        tk.Label(mbar, text="PREPARED ANSWER", fg=MUTED, bg=BG, font=self.small, anchor="w").pack(side="left", padx=(4, 0))
        rbar.update_idletasks()
        mbar.configure(height=rbar.winfo_reqheight())      # same header height on both sides
        mbar.pack_propagate(False)
        self.gen_status = tk.Label(self.right, text="", fg=MUTED, bg=BG, font=self.small, anchor="w",
                                   justify="left", wraplength=300)
        self.gen_status.pack(side="bottom", fill="x", padx=4, pady=(4, 0))
        self.prep_status = tk.Label(self.mid, text="", fg=MUTED, bg=BG, font=self.small, anchor="w",
                                    justify="left", wraplength=300)
        self.prep_status.pack(side="bottom", fill="x", padx=4, pady=(4, 0))
        self.text = self._answer_box(self.mid)
        self.gtext = self._answer_box(self.right)

        self.search.bind("<Return>", lambda e: self._manual_search())
        self.search.bind("<Escape>", lambda e: self.search.delete(0, "end"))
        self.root.bind("<Control-f>", lambda e: self.search.focus_set())
        self.root.bind("<Configure>", self._rewrap)

    def _answer_box(self, parent) -> tk.Text:
        body = tk.Frame(parent, bg=PANEL)
        body.pack(fill="both", expand=True)
        sb = tk.Scrollbar(body)
        sb.pack(side="right", fill="y")
        t = tk.Text(body, wrap="word", bg=PANEL, fg=FG, font=self.base, relief="flat", bd=0,
                    padx=12, pady=10, spacing2=3, spacing3=8, yscrollcommand=sb.set,
                    insertbackground=FG, cursor="arrow")
        t.pack(fill="both", expand=True)
        sb.config(command=t.yview)
        t.tag_configure("b", font=self.bold)
        t.tag_configure("ph", background=PLACEHOLDER_BG, foreground="#ffe7a3")
        t.tag_configure("rule", foreground=MUTED, justify="center")
        t.tag_configure("dim", foreground=MUTED, font=self.small)
        t.configure(state="disabled")
        return t

    def _set_notes(self, banner: str = "", hint: str = "") -> None:
        self.banner.configure(text=banner)
        self.hint.configure(text=hint)
        self.banner.pack_forget()
        self.hint.pack_forget()
        if banner:
            self.banner.pack(fill="x")
        if hint:
            self.hint.pack(fill="x")
        if not (banner or hint):
            self.notes.configure(height=1)   # Tk keeps a frame's old size once its last child leaves

    def _rewrap(self, _e=None) -> None:
        w = max(160, self.left.winfo_width() - 18)
        self.heard.configure(wraplength=w)
        self.banner.configure(wraplength=w)
        self.hint.configure(wraplength=w)
        self.skeleton.configure(wraplength=w)
        self.question.configure(wraplength=max(120, w - int(70 * self.scale)))
        self.gen_status.configure(wraplength=max(160, self.right.winfo_width() - 18))
        self.prep_status.configure(wraplength=max(160, self.mid.winfo_width() - 18))

    # ---------- events ----------
    def _pump(self) -> None:
        try:
            while True:
                kind, val = self.events.get_nowait()
                if kind == "status":
                    self._set_status(str(val))
                elif kind == "heard":
                    self._set_heard(val.text, val)
                elif kind == "decision":
                    self._apply(*val)
                elif kind == "reload":
                    self._reload_banks()
                elif kind == "matcher_ready":
                    self.matcher_error = ""
                    self._set_status(self.status.cget("text"))
                elif kind == "matcher_failed":
                    self.matcher_error = str(val)
                    self._set_status(self.status.cget("text"))
                elif kind == "forced_off":
                    self._forced_style(False)
                elif kind == "gen_req":
                    self._on_gen_req(*val)
                elif kind == "gen":
                    self._on_gen(*val)
        except queue.Empty:
            pass
        self._timers[0] = self.root.after(20, self._pump)

    def _tick(self) -> None:
        """A few times a second: the level meter, and a hint if nothing has been heard for a long while."""
        st = self.listener.stats()
        level = st["level"] if not self.listener.paused.is_set() else 0.0
        frac = min(1.0, max(0.0, (20 * math.log10(max(level, 1e-5)) + 55) / 50))
        self.meter.coords(self._meter_bar, 0, 0, int(self._meter_w * frac), self.meter.winfo_height())
        live = st["model_ready"] and st["device"] and not self.listener.paused.is_set() and not st["error"]
        self.quiet.configure(text=f"no sound for {int(st['quiet_for'])}s" if live and st["quiet_for"] > 30 else "")
        self._timers[1] = self.root.after(250, self._tick)

    def _set_status(self, s: str) -> None:
        if self.matcher_error:   # nothing works without the matcher, so this outranks "Listening"
            self.status.configure(text=self.matcher_error)
            self.dot.configure(fg=BAD)
            return
        n = len(self.matcher.entries) if self.matcher else 0
        failed = any(w in s.lower() for w in ("fail", "couldn't", "stopped", "error"))
        suffix = f"  ·  {n} questions" if self.matcher else "  ·  loading bank…"
        self.status.configure(text=s.split("  ·  ")[0] + ("" if failed else suffix))
        color = BAD if failed else (
            MUTED if self.listener.paused.is_set() else GOOD if s.startswith("Listening") else WARN)
        self.dot.configure(fg=color)

    def _load_matcher(self) -> None:
        try:
            if self.matcher is None:
                self.matcher = Matcher()
            self.cfg["active_banks"] = bankmod.load_config()["active_banks"]
            entries = bankmod.active_entries(self.cfg)
            self.matcher.load(entries)
            if self.conv is None:
                self.conv = Conversation(self.matcher, self.cfg)
            else:
                self.conv.refresh()
            self.listener.set_vocabulary(bank_vocabulary(entries))
        except Exception as e:  # noqa: BLE001
            self.events.put(("matcher_failed", f"Question matcher failed to load: {e}"))
            return
        self.events.put(("matcher_ready", None))

    # ----- matching (own thread: never blocks the window or the audio) -----
    def _match_loop(self) -> None:
        while True:
            h = self.heard_q.get()
            if h is None:
                return
            try:
                self._process(h)
            except Exception as e:  # noqa: BLE001 - keep listening whatever one lookup does
                self.events.put(("status", f"Couldn't match that ({type(e).__name__}: {e})"))

    def _process(self, h: Heard) -> None:
        conv = self.conv
        self._early_shown = {u for u in self._early_shown if u >= h.utt}
        if conv is None:
            self.events.put(("heard", h))
            return
        if h.final and h.repeat and h.utt in self._early_shown:   # the early guess already put this answer up
            self._early_shown.discard(h.utt)
            self.events.put(("heard", h))
            return
        forced = h.final and time.monotonic() < self._forced_until
        d = conv.resolve(h.text, final=h.final, forced=forced, utt=h.utt)
        if forced:
            self._forced_until = 0.0
            self.events.put(("forced_off", None))
        if not h.final and d.action == "show":
            self._early_shown.add(h.utt)
        self.events.put(("decision", (d, h)))
        self._consider_generation(d, h)

    def _consider_generation(self, d: Decision, h: Heard) -> None:
        """Start drafting an answer for what was just asked, from the closest prepared answers."""
        gen = self.gen
        if gen is None or self.conv is None:
            return
        text = (d.heard or h.text).strip()
        eligible = (h.final or d.action == "show") and len(text.split()) >= 4 and (
            d.action in ("show", "followup", "unsure") or d.cue is not None or looks_like_question(text))
        if not eligible:
            if d.action == "show":
                gen.cancel()            # a different answer is up now; stop drafting for the old question
            return
        ok, why = gen.status()
        if not ok:
            if self.cfg.get("generate", True) and why != self._gen_warned:
                self._gen_warned = why
                self.events.put(("gen", (0, "unavailable", why)))
            return
        thr = float(self.cfg["match_threshold"])
        parent = d.parent if d.is_followup else None
        req = build_request(text, d.matches, thr=thr, words=gen.words, max_matches=int(self.cfg.get("generate_matches", 3)),
                            profile=gen.profile, parent=parent, cue=d.cue if parent else None,
                            recent=self.conv.recent(2))
        # a confident answer can't be a half-heard fragment; anything else waits a beat in case more is coming
        delay = 0.5 if d.action in ("ignore", "unsure") and not text.endswith("?") else 0.0
        job = gen.start(req, delay=delay)
        # a follow-up with no scripted answer keeps the previous (parent) answer up, so it still counts as "prepared"
        self.events.put(("gen_req", (job, req, d.action in ("show", "followup"))))

    def _set_heard(self, text: str, h: Heard | None = None) -> None:
        tail = ""
        if h is not None:
            tail = ("" if h.final else " …") + (f"   ({h.lag:.1f}s)" if h.lag else "")
        self.heard.configure(text=f"Heard: “{text}”{tail}")

    def _apply(self, d: Decision, h: Heard) -> None:
        self._set_heard(d.heard or h.text, h)
        if d.action == "show":
            self._unsure_text = ""
            self._show(d.matches, d)
        elif d.action == "unsure":
            self._unsure_text = d.heard
            self._show_alts(d.matches, label="Not sure. Closest:")
        elif d.action == "followup":
            self._show_followup(d)

    def _manual_search(self) -> None:
        q = self.search.get().strip()
        if q and self.matcher:
            matches = self.matcher.match(q)
            if matches:
                self._unsure_text = ""
                if self.conv:
                    self.conv.pick(matches[0].entry)
                self._show(matches)

    # ---------- rendering ----------
    def _show(self, matches: list[Match], d: Decision | None = None, manual: bool = False) -> None:
        self.current = matches
        m = matches[0]
        e = m.entry
        self.question.configure(text=e.question)
        thr = float(self.cfg["match_threshold"])
        if manual:
            self.conf.configure(text="picked", bg=MUTED)
        else:
            color = GOOD if m.score >= thr + STRONG_MARGIN else WARN if m.score >= thr else BAD
            self.conf.configure(text=f"{m.score:.0%}", bg=color)
        self._set_notes(d.banner if d else "", d.hint if d else "")
        self.skeleton.configure(text=f"Skeleton: {e.skeleton}" if e.skeleton else "")
        self._prepared_md, self._prepared_note = e.response, ""
        self.prep_status.configure(text=f"from your bank: {e.bank}" if e.bank else "")
        self._clear_gen()
        if d is None and self.gen:
            self.gen.cancel()           # picked by hand: the user has decided, stop drafting
        self._render_prepared()
        self._show_alts(matches[1:], label="Also:")
        self._show_likely(d.likely if d else (self.conv.followups_of(e) if self.conv else []))

    def _show_followup(self, d: Decision) -> None:
        """Heard a follow-up with no scripted answer: keep the current answer up and say what is being asked."""
        self._set_notes(d.banner, d.hint)
        self._show_likely(d.likely)
        self._show_alts(d.matches, label="Closest follow-ups:")

    def _show_likely(self, entries) -> None:
        for w in self.likely.winfo_children():
            w.destroy()
        if not entries:
            return
        tk.Label(self.likely, text="Likely follow-ups:", fg=MUTED, bg=BG, font=self.small).pack(anchor="w")
        for e in entries[:4]:
            q = e.question if len(e.question) < 70 else e.question[:67] + "…"
            b = self._btn(self.likely, f"↳  {q}", lambda e=e: self._pick_entry(e))
            b.configure(anchor="w", justify="left")
            b.pack(fill="x", pady=1)

    def _show_alts(self, matches: list[Match], label: str) -> None:
        for w in self.alts.winfo_children():
            w.destroy()
        if not matches:
            return
        tk.Label(self.alts, text=label, fg=MUTED, bg=BG, font=self.small).pack(anchor="w")
        for m in matches[:3]:
            q = m.entry.question if len(m.entry.question) < 70 else m.entry.question[:67] + "…"
            b = self._btn(self.alts, f"{m.score:.0%}  {q}", lambda m=m: self._pick(m))
            b.configure(anchor="w", justify="left")
            b.pack(fill="x", pady=1)

    def _pick(self, m: Match) -> None:
        others = [x for x in self.current if x.entry is not m.entry] if self.current else []
        if self.conv:
            self.conv.pick(m.entry)
        self._set_notes()
        self._show([m, *others][:3])
        self._offer_remember(m.entry)

    def _pick_entry(self, entry) -> None:
        if self.conv:
            self.conv.pick(entry)
        self._unsure_text = ""
        self._show([Match(entry, 1.0)], manual=True)

    def _offer_remember(self, entry) -> None:
        """After choosing the right answer from a 'Not sure' list: offer to learn the wording that was heard."""
        text = self._unsure_text
        if not text:
            return
        short = text if len(text) < 48 else text[:45] + "…"
        btn = self._btn(self.alts, f"＋ Remember “{short}” as another way to ask this",
                        lambda: self._remember(entry, text, btn))
        btn.configure(anchor="w", justify="left")
        btn.pack(fill="x", pady=(6, 0))

    def _remember(self, entry, text: str, btn: tk.Button) -> None:
        try:
            added = bankmod.teach_phrasing(entry, text)
        except (KeyError, OSError) as e:
            btn.configure(text=f"Couldn't save that: {e}", state="disabled")
            return
        self._unsure_text = ""
        btn.configure(text="Saved. It will match next time." if added else "Already known.", state="disabled")
        if added:
            self._reload_banks()

    # ---------- drafted answers ----------
    def _clear_gen(self) -> None:
        """A different question's answer is up: forget the draft and ignore events from older jobs."""
        self._gen_floor = self._gen_seen
        self.gen_text, self.gen_state, self.gen_note, self.gen_summary = "", "idle", "", ""
        self.gen_truncated = False
        self._refresh_gen_ui()
        self._render_gen()

    def _begin_gen(self, job: int) -> None:
        self._gen_job = self._gen_seen = max(self._gen_seen, job)
        self.gen_text, self.gen_state, self.gen_note = "", "drafting", ""
        self.gen_truncated = False

    def _on_gen_req(self, job: int, req, has_prepared: bool) -> None:
        if job <= self._gen_floor:
            return
        if job > self._gen_job:
            self._begin_gen(job)
        self._last_req = req
        self.gen_summary = req.summary
        if not has_prepared:      # nothing prepared fits: don't leave the last question's answer on screen
            self.question.configure(text="No close prepared answer")
            self.conf.configure(text="", bg=BG)
            self.skeleton.configure(text="")
            self._prepared_md, self._prepared_note = "", "No prepared answer matches this question."
            self.prep_status.configure(text="")
            self._render_prepared()
        self._refresh_gen_ui()
        self._render_gen()

    def _on_gen(self, job: int, kind: str, data) -> None:
        if kind == "unavailable":
            self.gen_state, self.gen_note = "off", str(data)
            self._refresh_gen_ui()
            self._render_gen()
            return
        if job <= self._gen_floor:
            return
        if job > self._gen_job:
            self._begin_gen(job)
        self._gen_seen = max(self._gen_seen, job)
        if kind == "start":
            self.gen_summary, self.gen_state = str(data), "drafting"
        elif kind == "delta":
            self.gen_text += str(data)
            self._schedule_render()
        elif kind == "done":
            self.gen_text = (data.get("text") or self.gen_text).strip()
            self.gen_state, self.gen_seconds = "done", float(data.get("seconds", 0.0))
            self.gen_model = str(data.get("model", ""))
            self.gen_truncated = bool(data.get("truncated"))
            self._render_gen()
        elif kind == "error":
            self.gen_state, self.gen_note = "error", str(data)
            self._render_gen()
        self._refresh_gen_ui()

    def _schedule_render(self) -> None:
        if not self._render_pending:
            self._render_pending = True
            self.root.after(70, self._flush_render)

    def _flush_render(self) -> None:
        self._render_pending = False
        self._render_gen()

    def _render_prepared(self) -> None:
        if self._prepared_md:
            self._render(self._prepared_md)
        else:
            self._fill(self.text, [(self._prepared_note, "dim")] if self._prepared_note else [])

    def _render_gen(self) -> None:
        """The drafted answer: plain paragraphs, never bullets, bold or highlighted placeholders."""
        if self.gen_text:
            text = self._stream_safe(self.gen_text) if self.gen_state == "drafting" else self.gen_text
            paras = [p.strip() for p in plain_prose(text).split("\n") if p.strip()]
            self._fill(self.gtext, [(p, ()) for p in paras])
        elif self.gen_state == "drafting":
            self._fill(self.gtext, [("Drafting an answer from your prepared answers…", "dim")])
        elif self.gen_state in ("error", "off") and self.gen_note:
            self._fill(self.gtext, [(self.gen_note, "dim")])
        elif not self.cfg.get("generate", True):
            self._fill(self.gtext, [("Drafted answers are switched off. Press “✨ Draft: off” at the top to turn them on.", "dim")])
        else:
            self._fill(self.gtext, [("A tailored answer will appear here for each question.", "dim")])

    def _fill(self, box: tk.Text, paragraphs) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        for text, tag in paragraphs:
            box.insert("end", text + "\n", tag)
        box.configure(state="disabled")
        box.yview_moveto(0)

    @staticmethod
    def _stream_safe(md: str) -> str:
        """Half-streamed text may end inside **bold** or {{a placeholder}}; hide the unfinished marker."""
        if md.count("**") % 2:
            md = md[: md.rfind("**")] + md[md.rfind("**") + 2:]
        if md.rfind("{{") > md.rfind("}}"):
            md = md[: md.rfind("{{")]
        return md

    def _refresh_gen_ui(self) -> None:
        """Status line and the Regenerate / Save buttons for the current draft state."""
        wanted = bool(self.cfg.get("generate", True))
        usable = wanted and self.gen_state != "off" and (self.gen is None or self.gen.status()[0])
        self.gen_btn.configure(text="✨ Draft: on" if wanted else "✨ Draft: off")
        self.regen_btn.configure(state="normal" if (usable and self._last_req is not None
                                                    and self.gen_state in ("done", "error")) else "disabled")
        self.save_btn.configure(state="normal" if (self.gen_state == "done" and self.gen_text) else "disabled")
        color, text = MUTED, ""
        if self.gen_state == "drafting":
            text = "drafting…" + (f"  ·  {self.gen_summary}" if self.gen_summary else "")
        elif self.gen_state == "done":
            text = (f"{self.gen_summary}  ·  {self.gen_seconds:.1f}s" + (f"  ·  {self.gen_model}" if self.gen_model else "")
                    + ("  ·  cut short" if self.gen_truncated else ""))
        elif self.gen_state in ("error", "off") and self.gen_note:
            color, text = WARN, self.gen_note
        self.gen_status.configure(text=text, fg=color)

    def _toggle_gen(self) -> None:
        self.cfg["generate"] = not self.cfg.get("generate", True)
        self._gen_warned = ""
        if not self.cfg["generate"]:
            if self.gen:
                self.gen.cancel()
            self._clear_gen()
        self._save_cfg()
        self._refresh_gen_ui()
        self._render_gen()

    def _regenerate(self) -> None:
        if self.gen and self._last_req is not None:
            job = self.gen.start(self._last_req)
            self._on_gen_req(job, self._last_req, True)

    def _save_generated(self) -> None:
        """Keep a good draft: it becomes a normal prepared answer, matched by voice from now on."""
        if not (self.gen_text and self._last_req):
            return
        req = self._last_req
        name = req.sources[0].bank if req.sources else (self.cfg.get("active_banks") or [""])[0]
        try:
            bk = bankmod.load_bank(name)
        except KeyError:
            self.gen_status.configure(text="No active bank to save into.", fg=WARN)
            return
        q = req.question.strip()
        if any(e.question.strip().lower() == q.lower() for e in bk.entries):
            self.gen_status.configure(text="That question is already in the bank.", fg=WARN)
            return
        bk.entries.append(bankmod.Entry(question=q, response=self.gen_text, tags=["generated"], bank=bk.name))
        try:
            bk.save()
        except OSError as e:
            self.gen_status.configure(text=f"Couldn't save: {e}", fg=WARN)
            return
        self.save_btn.configure(state="disabled")
        self.gen_status.configure(text=f"Saved to “{bk.title or bk.name}”. Edit it in Manage banks.", fg=GOOD)
        self._reload_banks()

    def _render(self, md: str) -> None:
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for line in md.splitlines():
            if line.strip() == "---":
                t.insert("end", "─" * 30 + "\n", "rule")
                continue
            if line.startswith("- "):
                line = "•  " + line[2:]
            for tok in re.split(r"(\*\*.+?\*\*|\{\{.+?\}\})", line):
                if tok.startswith("**") and tok.endswith("**"):
                    self._insert_ph_aware(tok[2:-2], ("b",))
                elif tok.startswith("{{"):
                    t.insert("end", "⚠ " + tok[2:-2], ("ph",))
                elif tok:
                    t.insert("end", tok)
            t.insert("end", "\n")
        t.configure(state="disabled")
        t.yview_moveto(0)

    def _insert_ph_aware(self, s: str, tags: tuple) -> None:
        for tok in re.split(r"(\{\{.+?\}\})", s):
            if tok.startswith("{{"):
                self.text.insert("end", "⚠ " + tok[2:-2], (*tags, "ph"))
            elif tok:
                self.text.insert("end", tok, tags)

    # ---------- controls ----------
    def _font(self, d: int) -> None:
        size = min(28, max(9, self.base.cget("size") + d))
        self.base.configure(size=size)
        self.bold.configure(size=size)
        self.small.configure(size=max(9, size - 3))
        self.title_f.configure(size=size + 2)
        self.cfg["font_size"] = size
        self._save_cfg()

    def _toggle_pause(self) -> None:
        if self.listener.paused.is_set():
            self.listener.paused.clear()
            self.pause_btn.configure(text="❚❚ Pause")
            self._set_status(f"Listening · {self.listener.device_name}")
        else:
            self.listener.paused.set()
            self.pause_btn.configure(text="▶ Resume")
            self._set_status("Paused")

    def _toggle_follow(self) -> None:
        """Arm (or disarm) 'treat the next thing heard as a follow-up to the answer on screen'."""
        if time.monotonic() < self._forced_until:
            self._forced_until = 0.0
            self._forced_style(False)
            return
        self._forced_until = time.monotonic() + FOLLOW_ARM_SEC
        self._forced_style(True)
        self.root.after(FOLLOW_ARM_SEC * 1000 + 100, self._expire_follow)

    def _expire_follow(self) -> None:
        if time.monotonic() >= self._forced_until:
            self._forced_style(False)

    def _forced_style(self, on: bool) -> None:
        self.follow_btn.configure(text="↳ Listening for follow-up…" if on else "↳ Follow-up",
                                  bg=ACCENT if on else PANEL, fg=BG if on else FG)

    def _restart_audio(self) -> None:
        self._set_status("Reconnecting audio…")
        threading.Thread(target=self.listener.restart, daemon=True).start()

    def _fill_banks_menu(self) -> None:
        self.banks_menu.delete(0, "end")
        self.banks_menu.add_command(label="Manage banks…", command=self._open_manager)
        self.banks_menu.add_separator()
        self.cfg["active_banks"] = bankmod.load_config()["active_banks"]
        names = bankmod.bank_names()
        if not names:
            self.banks_menu.add_command(label="(no banks yet)", state="disabled")
        for n in names:
            var = tk.BooleanVar(value=n in self.cfg["active_banks"])
            self.banks_menu.add_checkbutton(label=n, variable=var,
                                            command=lambda n=n, v=var: self._toggle_bank(n, v.get()))
        self.banks_menu.add_separator()
        self.banks_menu.add_command(label="Reload banks from disk", command=self._reload_banks)

    def _toggle_bank(self, name: str, on: bool) -> None:
        bankmod.set_active(name, on)
        self._reload_banks()

    def _open_manager(self) -> None:
        from manager import BankManager
        if getattr(self, "_manager", None) and self._manager.win.winfo_exists():
            self._manager.win.lift()
            return
        self._manager = BankManager(self.root, on_change=lambda: self.events.put(("reload", None)),
                                    matcher_provider=lambda: self.matcher)

    def _reload_banks(self) -> None:
        self._reload_gen = getattr(self, "_reload_gen", 0) + 1
        gen = self._reload_gen

        def work():
            with self._reload_lock:
                if gen == self._reload_gen:  # skip if a newer reload is already queued
                    self._load_matcher()
        threading.Thread(target=work, daemon=True).start()

    def _save_cfg(self) -> None:
        disk = bankmod.load_config()
        disk["font_size"] = self.cfg["font_size"]
        disk["generate"] = self.cfg.get("generate", True)
        bankmod.save_config(disk)

    def _quit(self) -> None:
        # Tk objects (fonts, variables) must be freed on this thread. If a worker thread dropped the last
        # reference, or the garbage collector ran there, Tcl would be called from the wrong thread and hang
        # or abort. So: stop the workers, wait for the one that holds the window, and keep the collector off.
        self.listener.stop()
        if self.gen:
            self.gen.cancel()
        self.heard_q.put(None)
        self._match_thread.join(timeout=2)
        gc.disable()
        for t in self._timers:
            try:
                self.root.after_cancel(t)
            except tk.TclError:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
