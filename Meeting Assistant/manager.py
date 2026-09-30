"""Bank manager window: add, import, rename, delete and restore banks; add/edit/delete questions;
and test how a spoken question would match. Opens standalone or from the overlay."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont
from typing import Callable

import bank as b
from theme import ACCENT, BAD, BG, FG, GOOD, HOVER_BG, MUTED, PANEL, SELECT_BG, WARN, enable_dpi_awareness

FILE_TYPES = [
    ("Question banks", "*.docx *.xlsx *.xlsm *.csv *.md *.txt"),
    ("Word document", "*.docx"), ("Excel", "*.xlsx *.xlsm"), ("CSV", "*.csv"),
    ("Markdown / text", "*.md *.txt"), ("All files", "*.*"),
]
IMPORT_HELP = (
    "I couldn't find any questions in that file.\n\n"
    "What works:\n"
    "• Word/Markdown: each question as a heading, with the answer underneath\n"
    "• Lines starting with “Q:” and “A:”\n"
    "• Questions on their own line ending in “?”, answer below\n"
    "• Excel/CSV: a Question column and an Answer (or Response) column, "
    "or just questions in column A and answers in column B"
)


class BankManager:
    def __init__(self, master: tk.Misc | None = None, on_change: Callable[[], None] | None = None,
                 matcher_provider: Callable[[], object] | None = None) -> None:
        self.on_change = on_change or (lambda: None)
        self.matcher_provider = matcher_provider
        self._own_matcher = None
        self._matcher_dirty = False
        if master is None:
            enable_dpi_awareness()
            self.win: tk.Tk | tk.Toplevel = tk.Tk()
        else:
            self.win = tk.Toplevel(master)
            self.win.attributes("-topmost", True)
        self.win.title("Manage Question Banks")
        self.win.configure(bg=BG)
        s = self.win.winfo_fpixels("1i") / 96.0
        self.win.geometry(f"{int(980 * s)}x{int(640 * s)}")
        self.win.minsize(int(760 * s), int(480 * s))
        self.f = tkfont.Font(family="Segoe UI", size=10)
        self.fb = tkfont.Font(family="Segoe UI Semibold", size=10)
        self.fh = tkfont.Font(family="Segoe UI Semibold", size=13)
        self.fs = tkfont.Font(family="Segoe UI", size=9)
        self.scale = s
        self._style()
        self._build()
        self.refresh_banks()

    # ---------- look ----------
    def _style(self) -> None:
        st = ttk.Style(self.win)
        st.theme_use("clam")
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, borderwidth=0,
                     rowheight=int(28 * self.scale), font=self.f)
        st.map("Treeview", background=[("selected", SELECT_BG)], foreground=[("selected", FG)])
        st.configure("Treeview.Heading", background=BG, foreground=MUTED, relief="flat", font=self.fs)
        st.map("Treeview.Heading", background=[("active", BG)])
        st.configure("Vertical.TScrollbar", background=PANEL, troughcolor=BG, borderwidth=0, arrowcolor=MUTED)

    def _btn(self, parent, text, cmd, primary=False):
        return tk.Button(parent, text=text, command=cmd, bg=ACCENT if primary else PANEL,
                         fg=BG if primary else FG, activebackground=HOVER_BG, activeforeground=FG,
                         relief="flat", bd=0, padx=10, pady=5, font=self.fb if primary else self.f, cursor="hand2")

    def _label(self, parent, text, font=None, fg=FG, **kw):
        return tk.Label(parent, text=text, bg=BG, fg=fg, font=font or self.f, anchor="w", justify="left", **kw)

    # ---------- layout ----------
    def _build(self) -> None:
        # bottom: tester
        test = tk.Frame(self.win, bg=PANEL)
        test.pack(fill="x", side="bottom", padx=14, pady=(0, 12))
        row = tk.Frame(test, bg=PANEL)
        row.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(row, text="Try a question", bg=PANEL, fg=FG, font=self.fb).pack(side="left")
        self.test_var = tk.StringVar()
        e = tk.Entry(row, textvariable=self.test_var, bg=BG, fg=FG, insertbackground=FG, relief="flat", font=self.f)
        e.pack(side="left", fill="x", expand=True, padx=8, ipady=4)
        e.bind("<Return>", lambda _e: self.run_test())
        self._btn(row, "Test", self.run_test).pack(side="left")
        self.test_out = tk.Frame(test, bg=PANEL)
        self.test_out.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(self.test_out, text="Type something a recruiter might say to see which answer the overlay would show.",
                 bg=PANEL, fg=MUTED, font=self.fs, anchor="w").pack(fill="x")

        root = tk.Frame(self.win, bg=BG)
        root.pack(fill="both", expand=True, padx=14, pady=12)

        # left: banks
        left = tk.Frame(root, bg=BG)
        left.pack(side="left", fill="y")
        self._label(left, "Your banks", self.fh).pack(fill="x")
        self._label(left, "Click the ✓ column to turn a bank on or off in the overlay.", self.fs, MUTED).pack(fill="x", pady=(0, 6))
        grid = tk.Frame(left, bg=BG)
        grid.pack(side="bottom", fill="x", pady=(8, 0))
        for i, (text, cmd, primary) in enumerate((
                ("＋ New bank", self.new_bank, True), ("Import file…", self.import_bank, True),
                ("Rename", self.rename_bank, False), ("Delete", self.delete_bank, False),
                ("Restore deleted…", self.restore_bank, False), ("Open folder", self.open_folder, False))):
            self._btn(grid, text, cmd, primary).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        grid.columnconfigure((0, 1), weight=1)
        self.banks = ttk.Treeview(left, columns=("on", "title", "n"), show="headings", selectmode="browse",
                                  height=12)
        for col, text, w, anchor in (("on", "✓", 36, "center"), ("title", "Bank", 210, "w"), ("n", "Questions", 80, "center")):
            self.banks.heading(col, text=text)
            self.banks.column(col, width=int(w * self.scale), anchor=anchor, stretch=col == "title")
        self.banks.pack(fill="y", expand=True)
        self.banks.bind("<<TreeviewSelect>>", lambda e: self.refresh_questions())
        self.banks.bind("<Button-1>", self._on_bank_click)
        self.banks.bind("<Double-1>", lambda e: self.rename_bank())
        self.banks.bind("<Delete>", lambda e: self.delete_bank())

        # right: questions
        right = tk.Frame(root, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self.q_title = self._label(right, "Questions", self.fh)
        self.q_title.pack(fill="x")
        bar = tk.Frame(right, bg=BG)
        bar.pack(fill="x", pady=(2, 6))
        self._label(bar, "Filter", self.fs, MUTED).pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self.refresh_questions())
        tk.Entry(bar, textvariable=self.filter_var, bg=PANEL, fg=FG, insertbackground=FG, relief="flat",
                 font=self.f).pack(side="left", fill="x", expand=True, padx=6, ipady=3)

        qbtns = tk.Frame(right, bg=BG)
        qbtns.pack(side="bottom", fill="x", pady=(8, 0))
        self._btn(qbtns, "＋ Add question", self.add_question, True).pack(side="left", padx=(0, 4))
        self._btn(qbtns, "Edit", self.edit_question).pack(side="left", padx=4)
        self._btn(qbtns, "Delete", self.delete_question).pack(side="left", padx=4)
        self._label(qbtns, "Tip: double-click a question to edit it.", self.fs, MUTED).pack(side="left", padx=10)
        qframe = tk.Frame(right, bg=BG)
        qframe.pack(fill="both", expand=True)
        self.questions = ttk.Treeview(qframe, columns=("q", "alts"), show="headings", selectmode="browse")
        self.questions.heading("q", text="Question")
        self.questions.heading("alts", text="Other phrasings")
        self.questions.column("q", width=int(420 * self.scale), stretch=True)
        self.questions.column("alts", width=int(110 * self.scale), anchor="center", stretch=False)
        sb = ttk.Scrollbar(qframe, orient="vertical", command=self.questions.yview)
        self.questions.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.questions.pack(side="left", fill="both", expand=True)
        self.questions.bind("<Double-1>", lambda e: self.edit_question())
        self.questions.bind("<Return>", lambda e: self.edit_question())
        self.questions.bind("<Delete>", lambda e: self.delete_question())

    # ---------- banks ----------
    def selected_bank(self) -> str | None:
        sel = self.banks.selection()
        return sel[0] if sel else None

    def refresh_banks(self, select: str | None = None) -> None:
        keep = select or self.selected_bank()
        active = set(b.load_config()["active_banks"])
        self.banks.delete(*self.banks.get_children())
        for n in b.bank_names():
            bk = b.load_bank(n)
            self.banks.insert("", "end", iid=n, values=("✓" if n in active else "", bk.title or n, len(bk.entries)))
        names = self.banks.get_children()
        if names:
            self.banks.selection_set(keep if keep in names else names[0])
            self.banks.see(self.banks.selection()[0])
        self.refresh_questions()

    def _on_bank_click(self, event) -> str | None:
        row = self.banks.identify_row(event.y)
        if row and self.banks.identify_column(event.x) == "#1":
            b.set_active(row, row not in b.load_config()["active_banks"])
            self.refresh_banks(select=row)
            self._changed()
            return "break"
        return None

    def new_bank(self) -> None:
        title = self._ask_name("New bank", "Name for the new bank:", "")
        if title:
            bk = b.add_bank(title, activate=True)
            self.refresh_banks(select=bk.name)
            self._changed()
            self.add_question()

    def import_bank(self) -> None:
        path = filedialog.askopenfilename(parent=self.win, title="Import a question bank", filetypes=FILE_TYPES)
        if not path:
            return
        path = Path(path)
        try:
            preview = b.import_file(path, "preview")
        except Exception as e:  # noqa: BLE001 - show any parse failure to the user
            messagebox.showerror("Import failed", f"Couldn't read {path.name}:\n\n{e}", parent=self.win)
            return
        if not preview.entries:
            messagebox.showerror("No questions found", IMPORT_HELP, parent=self.win)
            return
        default = path.stem.replace("_", " ").replace("-", " ").strip()
        title = self._ask_name("Name this bank", f"Found {len(preview.entries)} questions in {path.name}.\n\nName for this bank:", default)
        if not title:
            return
        preview.name, preview.title = b.slug(title), title
        for e in preview.entries:
            e.bank = preview.name
        preview.save()
        b.set_active(preview.name, True)
        self.refresh_banks(select=preview.name)
        self._changed()
        messagebox.showinfo("Imported", f"Added “{title}” with {len(preview.entries)} questions. It's switched on "
                            "in the overlay.\n\nTip: open a few questions and add other ways they might be asked; "
                            "that's what makes matching reliable.", parent=self.win)

    def rename_bank(self) -> None:
        name = self.selected_bank()
        if not name:
            return
        bk = b.load_bank(name)
        title = self._ask_name("Rename bank", "New name:", bk.title or name, current=name)
        if title and title != bk.title:
            new = b.rename_bank(name, title)
            self.refresh_banks(select=new)
            self._changed()

    def delete_bank(self) -> None:
        name = self.selected_bank()
        if not name:
            return
        bk = b.load_bank(name)
        if messagebox.askyesno("Delete bank", f"Delete “{bk.title or name}” ({len(bk.entries)} questions)?\n\n"
                               "You can bring it back with Restore deleted….", parent=self.win):
            b.remove_bank(name)
            self.refresh_banks()
            self._changed()

    def restore_bank(self) -> None:
        paths = b.deleted_banks()
        if not paths:
            messagebox.showinfo("Restore", "There are no deleted banks.", parent=self.win)
            return
        dlg = tk.Toplevel(self.win)
        dlg.title("Restore a deleted bank")
        dlg.configure(bg=BG)
        dlg.transient(self.win)
        dlg.grab_set()
        self._label(dlg, "Pick a bank to restore:", self.fb).pack(fill="x", padx=12, pady=(12, 6))
        lb = tk.Listbox(dlg, bg=PANEL, fg=FG, selectbackground=SELECT_BG, relief="flat", font=self.f,
                        height=min(10, len(paths)), width=48, activestyle="none")
        for p in paths:
            lb.insert("end", p.stem)
        lb.selection_set(0)
        lb.pack(fill="both", expand=True, padx=12)

        def do_restore():
            if lb.curselection():
                name = b.restore_bank(paths[lb.curselection()[0]])
                dlg.destroy()
                self.refresh_banks(select=name)
                self._changed()
        lb.bind("<Double-1>", lambda e: do_restore())
        row = tk.Frame(dlg, bg=BG)
        row.pack(fill="x", padx=12, pady=12)
        self._btn(row, "Restore", do_restore, True).pack(side="right")
        self._btn(row, "Cancel", dlg.destroy).pack(side="right", padx=6)

    def open_folder(self) -> None:
        b.BANKS_DIR.mkdir(exist_ok=True)
        os.startfile(b.BANKS_DIR)

    def _ask_name(self, title: str, prompt: str, default: str, current: str | None = None) -> str | None:
        while True:
            name = simpledialog.askstring(title, prompt, initialvalue=default, parent=self.win)
            if not name or not name.strip():
                return None
            try:
                s = b.slug(name)
            except ValueError as e:
                messagebox.showerror(title, str(e), parent=self.win)
                continue
            if s in b.bank_names() and s != current:
                messagebox.showerror(title, f"You already have a bank called “{name}”. Pick another name.", parent=self.win)
                default = name
                continue
            return name.strip()

    # ---------- questions ----------
    def refresh_questions(self) -> None:
        self.questions.delete(*self.questions.get_children())
        name = self.selected_bank()
        if not name:
            self.q_title.configure(text="No banks yet. Click “New bank” or “Import file…” to start.")
            return
        bk = b.load_bank(name)
        self.q_title.configure(text=f"Questions in “{bk.title or name}”")
        needle = self.filter_var.get().strip().lower()
        for i, e in enumerate(bk.entries):
            hay = " ".join([e.question, *e.also, e.response]).lower()
            if not needle or needle in hay:
                self.questions.insert("", "end", iid=str(i), values=(e.question, len(e.also) or "—"))

    def _selected_entry(self) -> tuple[b.Bank, int] | None:
        name, sel = self.selected_bank(), self.questions.selection()
        if not name or not sel:
            return None
        return b.load_bank(name), int(sel[0])

    def add_question(self) -> None:
        name = self.selected_bank()
        if not name:
            messagebox.showinfo("Add question", "Create or import a bank first.", parent=self.win)
            return

        def save(entry: b.Entry) -> None:
            bk = b.load_bank(name)
            bk.entries.append(entry)
            bk.save()
            self.refresh_banks(select=name)
            self.questions.selection_set(str(len(bk.entries) - 1))
            self.questions.see(str(len(bk.entries) - 1))
            self._changed()
        QuestionEditor(self, b.Entry(question=""), save, title="Add question")

    def edit_question(self) -> None:
        got = self._selected_entry()
        if not got:
            return
        bk, i = got

        def save(entry: b.Entry) -> None:
            fresh = b.load_bank(bk.name)
            fresh.entries[i] = entry
            fresh.save()
            self.refresh_questions()
            self.questions.selection_set(str(i))
            self._changed()
        QuestionEditor(self, bk.entries[i], save, title="Edit question")

    def delete_question(self) -> None:
        got = self._selected_entry()
        if not got:
            return
        bk, i = got
        if messagebox.askyesno("Delete question", f"Delete this question?\n\n“{bk.entries[i].question}”", parent=self.win):
            del bk.entries[i]
            bk.save()
            self.refresh_banks(select=bk.name)
            self._changed()

    # ---------- tester ----------
    def _matcher(self):
        if self.matcher_provider:
            m = self.matcher_provider()
            if m is not None:
                return m
        if self._own_matcher is None:
            from matcher import Matcher
            self._own_matcher = Matcher()
            self._matcher_dirty = True
        if self._matcher_dirty:
            self._own_matcher.load(b.active_entries())
            self._matcher_dirty = False
        return self._own_matcher

    def run_test(self) -> None:
        text = self.test_var.get().strip()
        if not text:
            return
        self._show_test([("Matching… (the first test loads the model, about 10 seconds)", None)])

        def work():
            try:
                res = self._matcher().match(text, k=3)
                self.win.after(0, lambda: self._show_test(res, text))
            except Exception as e:  # noqa: BLE001
                self.win.after(0, lambda: self._show_test([(f"Test failed: {e}", None)]))
        threading.Thread(target=work, daemon=True).start()

    def _show_test(self, results, text: str = "") -> None:
        for w in self.test_out.winfo_children():
            w.destroy()
        if results and isinstance(results[0], tuple):
            tk.Label(self.test_out, text=results[0][0], bg=PANEL, fg=MUTED, font=self.fs, anchor="w").pack(fill="x")
            return
        if not results:
            tk.Label(self.test_out, text="No banks are switched on, so nothing can match.", bg=PANEL, fg=WARN,
                     font=self.fs, anchor="w").pack(fill="x")
            return
        thr = float(b.load_config()["match_threshold"])
        for rank, m in enumerate(results):
            color = GOOD if m.score >= thr + 0.06 else WARN if m.score >= thr else BAD
            row = tk.Frame(self.test_out, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{m.score:.0%}", bg=color, fg=BG, font=self.fs, width=5).pack(side="left")
            label = ("Overlay would show: " if rank == 0 and m.score >= thr else "") + m.entry.question
            tk.Label(row, text=label, bg=PANEL, fg=FG if rank == 0 else MUTED, font=self.f, anchor="w").pack(
                side="left", fill="x", expand=True, padx=8)
            tk.Button(row, text="＋ Teach it this wording", bg=BG, fg=ACCENT, relief="flat", bd=0, font=self.fs,
                      cursor="hand2", activebackground=HOVER_BG, activeforeground=FG,
                      command=lambda m=m: self._teach(m.entry, text)).pack(side="right")
        if results[0].score < thr:
            tk.Label(self.test_out, text="No confident match. If one of these is the right answer, click "
                     "“Teach it this wording” so it matches next time.", bg=PANEL, fg=WARN, font=self.fs,
                     anchor="w").pack(fill="x", pady=(4, 0))

    def _teach(self, entry: b.Entry, text: str) -> None:
        if not messagebox.askyesno("Teach this wording", f"Add “{text}” as another way to ask:\n\n{entry.question}",
                                   parent=self.win):
            return
        bk = b.load_bank(entry.bank)
        for e in bk.entries:
            if e.question == entry.question:
                if text not in e.also:
                    e.also.append(text)
                break
        bk.save()
        self.refresh_questions()
        self._changed()
        self._show_test([(f"Added “{text}” as another way to ask: {entry.question}", None)])

    def _changed(self) -> None:
        self._matcher_dirty = True
        self.on_change()

    def run(self) -> None:
        self.win.mainloop()


class QuestionEditor:
    """Dialog for one question: wording, other phrasings, skeleton, tags, response."""

    def __init__(self, mgr: BankManager, entry: b.Entry, on_save: Callable[[b.Entry], None], title: str):
        self.mgr, self.entry, self.on_save = mgr, entry, on_save
        s = mgr.scale
        d = self.dlg = tk.Toplevel(mgr.win)
        d.title(title)
        d.configure(bg=BG)
        d.transient(mgr.win)
        d.geometry(f"{int(720 * s)}x{int(720 * s)}")
        d.minsize(int(560 * s), int(560 * s))
        d.grab_set()
        f, fs, fb = mgr.f, mgr.fs, mgr.fb

        body = tk.Frame(d, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        def label(text, hint=""):
            row = tk.Frame(body, bg=BG)
            row.pack(fill="x", pady=(8, 2))
            tk.Label(row, text=text, bg=BG, fg=FG, font=fb).pack(side="left")
            if hint:
                tk.Label(row, text=hint, bg=BG, fg=MUTED, font=fs).pack(side="left", padx=8)

        def entry_box(value):
            e = tk.Entry(body, bg=PANEL, fg=FG, insertbackground=FG, relief="flat", font=f)
            e.insert(0, value)
            e.pack(fill="x", ipady=4)
            return e

        def text_box(value, height, expand=False):
            t = tk.Text(body, bg=PANEL, fg=FG, insertbackground=FG, relief="flat", font=f, height=height,
                        wrap="word", padx=8, pady=6, undo=True)
            t.insert("1.0", value)
            t.pack(fill="both", expand=expand)
            return t

        label("Question")
        self.q = entry_box(entry.question)
        label("Other ways it might be asked", "one per line; more = better matching")
        self.also = text_box("\n".join(entry.also), 4)
        mgr._btn(body, "Suggest wordings", self.suggest).pack(anchor="w", pady=(4, 0))
        label("Response", "**bold** = line to land · {{text}} = placeholder to fix")
        self.resp = text_box(entry.response, 12, expand=True)
        label("Skeleton", "optional one-line outline shown above the answer")
        self.skel = entry_box(entry.skeleton)
        label("Tags", "optional, comma-separated")
        self.tags = entry_box(", ".join(entry.tags))

        row = tk.Frame(d, bg=BG)
        row.pack(fill="x", padx=16, pady=(0, 14))
        mgr._btn(row, "Save", self.save, True).pack(side="right")
        mgr._btn(row, "Cancel", d.destroy).pack(side="right", padx=6)
        d.bind("<Control-s>", lambda e: self.save())
        d.bind("<Escape>", lambda e: d.destroy())
        self.q.focus_set()

    def suggest(self) -> None:
        """Offer reworded versions of the question to add under 'Other ways it might be asked'."""
        from suggest import suggest_phrasings
        q = self.q.get().strip()
        if not q:
            messagebox.showinfo("Suggest wordings", "Type the question first.", parent=self.dlg)
            return
        have = [line.strip() for line in self.also.get("1.0", "end").splitlines() if line.strip()]
        found = suggest_phrasings(q, have, n=8)
        if not found:
            messagebox.showinfo("Suggest wordings", "No new wordings to suggest. This one is well covered.",
                                parent=self.dlg)
            return
        mgr = self.mgr
        dlg = tk.Toplevel(self.dlg)
        dlg.title("Suggested wordings")
        dlg.configure(bg=BG)
        dlg.transient(self.dlg)
        dlg.grab_set()
        tk.Label(dlg, text="Keep the ones an interviewer might really say:", bg=BG, fg=FG, font=mgr.fb).pack(
            anchor="w", padx=12, pady=(12, 6))
        lb = tk.Listbox(dlg, selectmode="extended", bg=PANEL, fg=FG, selectbackground=SELECT_BG, relief="flat",
                        font=mgr.f, height=min(10, len(found)), width=64, activestyle="none", exportselection=False)
        for s in found:
            lb.insert("end", s.text)
        lb.selection_set(0, "end")
        lb.pack(fill="both", expand=True, padx=12)

        def add() -> None:
            picked = [found[i].text for i in lb.curselection()]
            if picked:
                current = self.also.get("1.0", "end").strip()
                self.also.delete("1.0", "end")
                self.also.insert("1.0", "\n".join(([current] if current else []) + picked))
            dlg.destroy()
        row = tk.Frame(dlg, bg=BG)
        row.pack(fill="x", padx=12, pady=12)
        mgr._btn(row, "Add selected", add, True).pack(side="right")
        mgr._btn(row, "Cancel", dlg.destroy).pack(side="right", padx=6)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def save(self) -> None:
        q = self.q.get().strip()
        if not q:
            messagebox.showerror("Missing question", "Type the question first.", parent=self.dlg)
            return
        self.on_save(b.Entry(
            question=q,
            response=self.resp.get("1.0", "end").strip(),
            also=[line.strip() for line in self.also.get("1.0", "end").splitlines() if line.strip()],
            tags=[t.strip() for t in self.tags.get().split(",") if t.strip()],
            skeleton=self.skel.get().strip(),
            bank=self.entry.bank,
        ))
        self.dlg.destroy()


if __name__ == "__main__":
    BankManager().run()
