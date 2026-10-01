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



import queue
import re
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import bank as bankmod
from audio import Listener
from matcher import Match, Matcher, looks_like_question
from theme import ACCENT, BAD, BG, FG, GOOD, HOVER_BG, MUTED, PANEL, PLACEHOLDER_BG, WARN, enable_dpi_awareness

JOIN_WINDOW_SEC = 4.0
STRONG_MARGIN = 0.06  # score >= threshold + this shows even if it doesn't sound like a question
GUESS_MARGIN = 0.08   # below threshold by up to this, a question gets "Not sure" suggestions


class Overlay:
    def __init__(self) -> None:
        self.cfg = bankmod.load_config()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.matcher: Matcher | None = None
        self.current: list[Match] = []
        self.pending: tuple[str, float] | None = None
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
        self.listener = Listener(self.cfg, lambda t: self.events.put(("text", t)),
                                 lambda s: self.events.put(("status", s)))
        self._reload_banks()
        self.listener.start()
        self.root.after(50, self._pump)
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

    def _rewrap(self, _e=None) -> None:
        w = max(300, self.root.winfo_width() - 40)
        self.heard.configure(wraplength=w)
        self.skeleton.configure(wraplength=w)
        self.question.configure(wraplength=w - int(90 * self.scale))

    # ---------- events ----------
    def _pump(self) -> None:
        try:
            while True:
                kind, val = self.events.get_nowait()
                if kind == "status":
                    self._set_status(str(val))
                elif kind == "text":
                    self._on_heard(str(val))
                elif kind == "reload":
                    self._reload_banks()
                elif kind == "matcher_ready":
                    self._set_status(self.status.cget("text"))
        except queue.Empty:
            pass
        self.root.after(50, self._pump)

    def _set_status(self, s: str) -> None:
        n = len(self.matcher.entries) if self.matcher else 0
        suffix = f"  ·  {n} questions" if self.matcher else "  ·  loading bank…"
        self.status.configure(text=s.split("  ·  ")[0] + ("" if "fail" in s.lower() else suffix))
        color = BAD if ("fail" in s.lower() or "couldn't" in s.lower()) else (
            MUTED if self.listener.paused.is_set() else GOOD if s.startswith("Listening") else WARN)
        self.dot.configure(fg=color)

    def _load_matcher(self) -> None:
        if self.matcher is None:
            self.matcher = Matcher()
        self.cfg["active_banks"] = bankmod.load_config()["active_banks"]
        self.matcher.load(bankmod.active_entries(self.cfg))
        self.events.put(("matcher_ready", None))

    def _on_heard(self, text: str) -> None:
        # A mid-question pause splits speech into fragments; glue an unmatched fragment
        # onto whatever follows it within a few seconds.
        now = time.monotonic()
        if self.pending and now - self.pending[1] <= JOIN_WINDOW_SEC:
            text = f"{self.pending[0]} {text}"
        self.pending = None
        self.heard.configure(text=f"Heard: “{text}”")
        if not self.matcher:
            return
        matches = self.matcher.match(text)
        if not matches:
            return
        thr = float(self.cfg["match_threshold"])
        best = matches[0].score
        is_q = looks_like_question(text)
        if best >= thr + STRONG_MARGIN or (best >= thr and is_q):
            self._show(matches)
            return
        self.pending = (text[-400:], now)
        if is_q and best >= thr - GUESS_MARGIN:
            self._show_alts(matches, label="Not sure. Closest:")

    def _manual_search(self) -> None:
        q = self.search.get().strip()
        if q and self.matcher:
            self._show(self.matcher.match(q))

    # ---------- rendering ----------
    def _show(self, matches: list[Match]) -> None:
        self.current = matches
        m = matches[0]
        e = m.entry
        self.question.configure(text=e.question)
        thr = float(self.cfg["match_threshold"])
        color = GOOD if m.score >= thr + STRONG_MARGIN else WARN if m.score >= thr else BAD
        self.conf.configure(text=f"{m.score:.0%}", bg=color)
        self.skeleton.configure(text=f"Skeleton: {e.skeleton}" if e.skeleton else "")
        self._render(e.response)
        self._show_alts(matches[1:], label="Also:")

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
        self._show([m, *others][:3])

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
        self.listener.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
