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
from matcher import Match, Matcher, bank_vocabulary
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
        self._reload_lock = threading.Lock()

        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.title("Interview Answer Overlay")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", float(self.cfg["opacity"]))
        self.scale = self.root.winfo_fpixels("1i") / 96.0
        w, h = int(560 * self.scale), int(760 * self.scale)
        self.root.geometry(f"{w}x{h}+{self.root.winfo_screenwidth() - w - 24}+40")
        self.root.minsize(int(380 * self.scale), int(420 * self.scale))

        self.base = tkfont.Font(family="Segoe UI", size=self.cfg["font_size"])
        self.bold = tkfont.Font(family="Segoe UI", size=self.cfg["font_size"], weight="bold")
        self.small = tkfont.Font(family="Segoe UI", size=max(9, self.cfg["font_size"] - 3))
        self.title_f = tkfont.Font(family="Segoe UI Semibold", size=self.cfg["font_size"] + 2)

        self._build()
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

        self.heard = tk.Label(self.root, text="Heard: —", fg=MUTED, bg=BG, font=self.small,
                              anchor="w", justify="left", wraplength=520)
        self.heard.pack(fill="x", padx=12)

        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=12, pady=(8, 2))
        self.conf = tk.Label(head, text="", fg=BG, bg=MUTED, font=self.small, padx=6)
        self.conf.pack(side="right", anchor="n")
        self.question = tk.Label(head, text="Waiting for a question…", fg=FG, bg=BG, font=self.title_f,
                                 anchor="w", justify="left", wraplength=450)
        self.question.pack(side="left", fill="x", expand=True)

        self.notes = tk.Frame(self.root, bg=BG)   # follow-up banner + hint; takes no room while empty
        self.notes.pack(fill="x", padx=12)
        self.banner = tk.Label(self.notes, text="", fg=WARN, bg=BG, font=self.small, anchor="w",
                               justify="left", wraplength=520)
        self.hint = tk.Label(self.notes, text="", fg=MUTED, bg=BG, font=self.small, anchor="w",
                             justify="left", wraplength=520)

        self.skeleton = tk.Label(self.root, text="", fg=ACCENT, bg=BG, font=self.small, anchor="w",
                                 justify="left", wraplength=520)
        self.skeleton.pack(fill="x", padx=12, pady=(0, 6))

        search = tk.Frame(self.root, bg=BG)
        search.pack(side="bottom", fill="x", padx=10, pady=(2, 10))
        tk.Label(search, text="Search", fg=MUTED, bg=BG, font=self.small).pack(side="left")
        self.search = tk.Entry(search, bg=PANEL, fg=FG, insertbackground=FG, relief="flat", font=self.small)
        self.search.pack(side="left", fill="x", expand=True, padx=6, ipady=3)
        self.alts = tk.Frame(self.root, bg=BG)
        self.alts.pack(side="bottom", fill="x", padx=10, pady=(6, 2))
        self.likely = tk.Frame(self.root, bg=BG)
        self.likely.pack(side="bottom", fill="x", padx=10, pady=(4, 0))

        body = tk.Frame(self.root, bg=PANEL)
        body.pack(fill="both", expand=True, padx=10)
        sb = tk.Scrollbar(body)
        sb.pack(side="right", fill="y")
        self.text = tk.Text(body, wrap="word", bg=PANEL, fg=FG, font=self.base, relief="flat", bd=0,
                            padx=12, pady=10, spacing2=3, spacing3=8, yscrollcommand=sb.set,
                            insertbackground=FG, cursor="arrow")
        self.text.pack(fill="both", expand=True)
        sb.config(command=self.text.yview)
        self.text.tag_configure("b", font=self.bold)
        self.text.tag_configure("ph", background=PLACEHOLDER_BG, foreground="#ffe7a3")
        self.text.tag_configure("rule", foreground=MUTED, justify="center")
        self.text.configure(state="disabled")

        self.search.bind("<Return>", lambda e: self._manual_search())
        self.search.bind("<Escape>", lambda e: self.search.delete(0, "end"))
        self.root.bind("<Control-f>", lambda e: self.search.focus_set())
        self.root.bind("<Configure>", self._rewrap)

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
        w = max(300, self.root.winfo_width() - 40)
        self.heard.configure(wraplength=w)
        self.banner.configure(wraplength=w)
        self.hint.configure(wraplength=w)
        self.skeleton.configure(wraplength=w)
        self.question.configure(wraplength=w - int(90 * self.scale))

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
        self._render(e.response)
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
        bankmod.save_config(disk)

    def _quit(self) -> None:
        # Tk objects (fonts, variables) must be freed on this thread. If a worker thread dropped the last
        # reference, or the garbage collector ran there, Tcl would be called from the wrong thread and hang
        # or abort. So: stop the workers, wait for the one that holds the window, and keep the collector off.
        self.listener.stop()
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
