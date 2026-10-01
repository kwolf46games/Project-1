"""Bank manager dialogs, headless: follow-up field, suggestions, bulk suggestions, rename linking."""
import gc
import os
import time

import pytest

tk = pytest.importorskip("tkinter")
if not os.environ.get("DISPLAY"):
    pytest.skip("needs a display (run under xvfb-run)", allow_module_level=True)


@pytest.fixture(autouse=True)
def main_thread_gc():
    """Tk objects must be freed on the main thread: a collection run by a worker thread would call into a
    dead Tcl interpreter from the wrong thread and hang or abort. So collect only when we say so."""
    gc.collect()
    gc.disable()
    yield
    gc.collect()
    gc.enable()

import bank as b  # noqa: E402
import manager  # noqa: E402
from conftest import fake_embed  # noqa: E402
from matcher import Matcher  # noqa: E402


def track(monkeypatch, name):
    """Record every instance of manager.<name> that gets created, so tests can drive it directly."""
    cls = getattr(manager, name)
    made, orig = [], cls.__init__

    def init(self, *a, **k):
        made.append(self)
        orig(self, *a, **k)
    monkeypatch.setattr(cls, "__init__", init)
    return made


@pytest.fixture
def mgr(banks_dir, monkeypatch):
    banks_dir.mkdir(parents=True)
    b.Bank("demo", "Demo", [
        b.Entry("Tell me about a time you led a team", "LEAD", ["describe a time you led a team"], bank="demo"),
        b.Entry("What was the result of that project?", "RESULT", follows=["Tell me about a time you led a team"], bank="demo"),
        b.Entry("Can you give me an example?", "EX", tags=["followup"], bank="demo"),
        b.Entry("What is your biggest weakness?", "WEAK", bank="demo"),
    ]).save()
    b.set_active("demo", True)
    for name in ("showinfo", "showerror"):
        monkeypatch.setattr(manager.messagebox, name, lambda *a, **k: None)
    monkeypatch.setattr(manager.messagebox, "askyesno", lambda *a, **k: True)
    m = manager.BankManager()
    shared = Matcher(embed_fn=fake_embed)
    shared.load(b.active_entries())
    monkeypatch.setattr(m, "_matcher", lambda: shared)
    yield m
    m.win.after_cancel(m._drain_id)
    m.win.destroy()


def pump(m, until=None, timeout=4.0):
    """Run the window's events until `until()` is true; with no condition, just settle for a moment."""
    end = time.monotonic() + (0.25 if until is None else timeout)
    while time.monotonic() < end:
        m.win.update()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until() if until is not None else True


def labels(w):
    out, stack = [], [w]
    while stack:
        x = stack.pop()
        if x.winfo_class() == "Label":
            out.append(x.cget("text"))
        stack += x.winfo_children()
    return out


def edit(mgr, monkeypatch, index):
    made = track(monkeypatch, "QuestionEditor")
    mgr.refresh_banks("demo")
    mgr.questions.selection_set(str(index))
    mgr.edit_question()
    pump(mgr)
    return made[-1]


def test_try_a_question_reports_a_model_failure(mgr, monkeypatch):
    """The 'Try a question' box used to hit a NameError instead of saying why it failed."""
    def boom():
        raise RuntimeError("model files missing")
    monkeypatch.setattr(mgr, "_matcher", boom)
    mgr.test_var.set("What is your biggest weakness?")
    mgr.run_test()
    assert pump(mgr, lambda: any("Test failed: model files missing" in t for t in labels(mgr.test_out)))


def test_try_a_question_shows_results(mgr):
    mgr.test_var.set("What is your biggest weakness?")
    mgr.run_test()
    assert pump(mgr, lambda: any("Overlay would show: What is your biggest weakness?" in t for t in labels(mgr.test_out)))


def test_list_shows_followup_column(mgr):
    mgr.refresh_banks("demo")
    col = {mgr.questions.item(i, "values")[0]: mgr.questions.item(i, "values")[2] for i in mgr.questions.get_children()}
    assert col["What was the result of that project?"] == "Tell me about a time you led a team"
    assert col["Can you give me an example?"] == "any answer"
    assert col["What is your biggest weakness?"] == "—"


def test_follows_choices_exclude_the_question_itself(mgr, monkeypatch):
    ed = edit(mgr, monkeypatch, 0)
    values = ed.follows.cget("values")
    assert "What is your biggest weakness?" in values and "Tell me about a time you led a team" not in values


def test_follows_round_trips_including_two_parents(mgr, monkeypatch):
    ed = edit(mgr, monkeypatch, 3)
    ed.follows.set("What was the result of that project? | Tell me about a time you led a team")
    ed.save()
    assert b.load_bank("demo").entries[3].follows == ["What was the result of that project?",
                                                      "Tell me about a time you led a team"]


def test_renaming_a_question_relinks_its_followups(mgr, monkeypatch):
    ed = edit(mgr, monkeypatch, 0)
    ed.q.delete(0, "end")
    ed.q.insert(0, "Describe a time you led a team")
    ed.save()
    saved = b.load_bank("demo")
    assert saved.entries[0].question == "Describe a time you led a team"
    assert saved.entries[1].follows == ["Describe a time you led a team"]


def test_empty_question_is_refused(mgr, monkeypatch):
    ed = edit(mgr, monkeypatch, 0)
    ed.q.delete(0, "end")
    ed.save()
    assert b.load_bank("demo").entries[0].question == "Tell me about a time you led a team"


def test_suggest_button_adds_ticked_wordings(mgr, monkeypatch):
    dialogs = track(monkeypatch, "SuggestDialog")
    ed = edit(mgr, monkeypatch, 3)
    ed.suggest()
    d = dialogs[-1]
    assert len(d.vars) >= 5
    assert pump(mgr, lambda: any(t.startswith("Checked") for t in labels(d.dlg)))      # vetted in the background
    texts = [s.text for _, s in d.vars]
    assert any("weakness" in t.lower() for t in texts)
    assert all(sug.sim is not None and sug.sim >= 0.8 for _, sug in d.vars)             # vetting dropped the distant ones
    ticked = [s.text for v, s in d.vars if v.get()]
    d.add()
    pump(mgr)
    also = ed.also.get("1.0", "end").strip().splitlines()
    assert also == ticked and 1 <= len(also) <= 5


def test_rivals_start_unticked(mgr, monkeypatch):
    """A wording that is just as close to a different question must not be pre-ticked."""
    from suggest import Suggestion
    dialogs = track(monkeypatch, "SuggestDialog")
    entry = b.Entry("What is your biggest weakness?", bank="demo")
    manager.SuggestDialog(mgr, mgr.win, "t", [Suggestion("Fine one", "pattern", 0.9), Suggestion("Risky one", "pattern", 0.85, "Other?")],
                          entry, lambda texts: None)
    d = dialogs[-1]
    d.items = [Suggestion("Fine one", "pattern", 0.9), Suggestion("Risky one", "pattern", 0.85, "Other?")]
    d._render()
    assert [v.get() for v, _ in d.vars] == [True, False]


def test_suggest_needs_a_question_first(mgr, monkeypatch):
    dialogs = track(monkeypatch, "SuggestDialog")
    ed = edit(mgr, monkeypatch, 0)
    ed.q.delete(0, "end")
    ed.suggest()
    assert dialogs == []


def test_bulk_suggest_tick_untick_and_add(mgr, monkeypatch):
    bulks = track(monkeypatch, "BulkSuggestDialog")
    mgr.refresh_banks("demo")
    mgr.suggest_all()
    d = bulks[-1]
    tree = d.tree
    assert pump(mgr, lambda: len(tree.get_children()) > 4)
    assert pump(mgr, lambda: any(t.startswith("Checked") for t in labels(d.dlg)), timeout=8)
    heads = [i for i in tree.get_children() if i.startswith("q")]
    assert tree.item(heads[0], "values")[1] in [e.question for e in d.bank.entries]    # (vetting may drop a question's rows)
    pump(mgr)

    def click(iid, col="on"):
        bbox = tree.bbox(iid, col)
        ev = type("Ev", (), {"x": bbox[0] + 3, "y": bbox[1] + 3})
        d._click(ev)

    on_rows = [i for i in tree.get_children() if i.startswith("s") and tree.set(i, "on") == "✓"]
    assert on_rows
    victim = on_rows[0]
    victim_text = d.rows[victim][1].text
    click(victim)
    assert tree.set(victim, "on") == "" and d.touched[(d.rows[victim][0], victim_text)] is False
    d._fill([])                                                           # re-rendering with nothing must not crash
    assert d.rows == {}
    d._vetted([(i, __import__("suggest").generate(e.question, e.phrasings(), limit=6)) for i, e in enumerate(d.bank.entries)])
    assert d.touched[(d.rows[victim][0], victim_text)] is False and d.rows[victim][2] is False   # user's choice survived
    # clicking a question's own row switches its whole group on, then off again
    other = next(h for h in [i for i in tree.get_children() if i.startswith("q")] if int(h[1:]) != d.rows[victim][0])
    group = [k for k, (i, _, _) in d.rows.items() if i == int(other[1:])]
    for _ in range(2):
        click(other, "text")
        states = {d.rows[k][2] for k in group}
        assert len(states) == 1
    assert states == {False}
    click(other, "text")
    assert all(d.rows[k][2] for k in group)
    d.add()
    pump(mgr)
    saved = b.load_bank("demo")
    added = [t for e in saved.entries for t in e.also]
    assert victim_text not in added
    assert len(added) >= 3 and all(len(t.split()) >= 2 for t in added)
    assert all(d.rows[k][1].text in added for k in group)                 # the group we switched on was added
