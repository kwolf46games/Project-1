"""Follow-up questions, and the decision about what to show for each thing heard.

A follow-up ("Can you give an example of that?", "What was the outcome?", "Why did you do that?")
can't be understood on its own: it only means something next to the answer just shown.  So:

* detect_followup() spots the wording of a follow-up (pure text, no models);
* Conversation remembers which answers were just shown, prefers follow-ups scripted for them
  (a question with "follows: <question>", or tagged "followup" for generic ones) and, when nothing
  scripted fits, keeps the current answer on screen with a hint about what is being asked.

A confident match to a *different* question always wins over a guessed follow-up, so a new topic
never gets forced into the old one.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from bank import Entry
from matcher import Match, Matcher, looks_like_question, split_sentences

JOIN_WINDOW_SEC = 4.0   # an unmatched fragment is glued onto what follows within this long
STRONG_MARGIN = 0.06    # score >= threshold + this shows even if it doesn't sound like a question
GUESS_MARGIN = 0.08     # below threshold by up to this, a question gets "Not sure" suggestions
FOLLOW_BOOST = 0.08     # ranking weight for follow-ups scripted for the answer just shown
GENERIC_BOOST = 0.04    # ... and for generic follow-ups ("tags: followup")
FOLLOW_RELAX = 0.07     # how far the threshold drops for a scripted follow-up
MAX_CUE_WORDS = 12      # follow-ups are short; longer speech is a statement or a new question

# kind -> (what is being asked, coaching hint)
KINDS: dict[str, tuple[str, str]] = {
    "elaborate": ("more detail", "They want more detail: expand on the key points of your answer."),
    "example": ("an example", "They want a specific example: pick one concrete case (situation, action, result)."),
    "outcome": ("the result", "They're asking how it turned out: lead with the outcome, then what you learned."),
    "reason": ("why", "They're asking why: explain your reasoning and what you weighed up."),
    "how": ("how you did it", "They're asking how: walk through the steps you took."),
    "alternative": ("what else / what you'd change", "They're asking what you'd change or add."),
    "challenge": ("the hardest part", "They're asking about the hardest part: name the obstacle and how you got past it."),
    "reference": ("something you said", "They're referring back to something you said earlier."),
    "continuation": ("what comes next", "They're continuing from your last answer."),
    "pronoun": ("about that", "This seems to refer to your last answer."),
    "explicit": ("follow-up", "They called it a follow-up: stay on the same topic."),
    "forced": ("follow-up", "Treated as a follow-up to your last answer."),
}

_WH = r"(?:what|why|how|when|where|who|which|did|do|does|would|could|can|were|was|is|are|have|has|had)"
_EX_REST = (r"(?=\s*(?:of (?:that|this|it|those|these|what you|how you|how that|when that|when you)"
            r"|to (?:illustrate|help|show|back)|please|\?|$))")

# (kind, strength, regex) over lower-cased words and "?"; first hit wins, so order = priority
_CUES: list[tuple[str, float, re.Pattern]] = [(k, s, re.compile(p)) for k, s, p in [
    ("explicit", 1.0, r"\bfollow ?up question\b|\bquick follow ?up\b|\bfollowing up on\b|\bto follow up on\b"),
    ("example", 1.0, r"\banother (?:example|instance|time|situation|case)\b"),
    ("example", 1.0, r"\b(?:an?|one|some|any|a specific|a concrete|a real) (?:specific |concrete |real )?"
                     r"(?:examples?|instances?)\b" + _EX_REST),
    ("example", 1.0, r"\bfor (?:example|instance) ?\?|\bexamples? ?\?$"),
    ("elaborate", 1.0, r"\b(?:tell|say|share) (?:me |us )?(?:a (?:little|bit) )?more\b"
                       r"(?: (?:about (?:that|this|it|those|what you|how you|why)|on (?:that|this|it)))?(?: please)? ?\??$"),
    ("elaborate", 1.0, r"\b(?:elaborate|expand (?:on|upon)|clarify|go (?:deeper|further)|dig (?:deeper|into (?:that|it))|"
                       r"unpack (?:that|it))\b|\bmore (?:detail|details|specifics|context)\b"
                       r"|\bbe (?:a bit |a little )?more (?:specific|precise|detailed|concrete)\b"),
    ("elaborate", 1.0, r"\bwhat do you mean\b|\bwhat does that mean\b|\bhow so\b|\bin what way\b|\bsuch as what\b"
                       r"|\blike what\b|\bmeaning what\b|\bwalk me through (?:that|it|this|how that)\b"
                       r"|\bbreak (?:that|it|this) down\b"),
    ("elaborate", 1.0, r"^(?:go on|keep going|continue|please continue|and|and\?)$"),
    ("outcome", 1.0, r"\bwhat (?:was|were|is) (?:the|your) (?:end )?(?:result|results|outcome|impact|effect|takeaway|"
                     r"lesson|learnings?)\b|\bhow did (?:that|it|things|everything) (?:go|turn out|end|work out|pan out|play out)\b"
                     r"|\bwhat happened (?:next|then|after(?: that)?|in the end|as a result)\b"
                     r"|\bwhat did you learn\b|\bdid (?:that|it) (?:work|help|succeed|pay off)\b"
                     r"|\bhow (?:did|do) you (?:measure|know|track)\b|\bwhat came (?:of|out of) (?:that|it)\b"
                     r"|\band then what\b|\bso what happened\b|\bwas (?:it|that) (?:successful|a success|worth it)\b"),
    ("reason", 0.8, r"\bwhy (?:did|do|would) you (?:do|choose|pick|decide|go|take|use|approach|handle|say|think|feel) "
                    r"(?:that|it|this|so|the way|things)\b|\bwhy (?:is|was) that\b|\bwhy not\b|\bhow come\b"
                    r"|\bwhat made you (?:do|choose|pick|decide|go|take|say)\b|\bwhat (?:was|is) your "
                    r"(?:reasoning|rationale|thinking|thought process|logic)\b|\bwhat led you to\b|\bwhat motivated\b"
                    r"|^why ?\??$|^why (?:so|not)\b"),
    ("how", 0.8, r"\bhow (?:did|do|would|will) you (?:handle|deal with|approach|tackle|manage|solve|overcome|go about|fix|"
                 r"address|respond to|react to) (?:that|it|this|those|them)\b|\bhow did you do (?:that|it)\b"
                 r"|\bwhat steps did you take\b|\bwhat (?:was|is) your (?:role|approach|part|contribution|process)"
                 r"(?: in (?:that|it|this))?\b|\bwhat did you do (?:about (?:that|it)|then|next|there|in that situation)\b"
                 r"|\bwhat was your (?:response|reaction)\b"),
    ("alternative", 0.8, r"\bwhat (?:would|could) you (?:have )?(?:do|done|change)(?: it| that)? differently\b"
                         r"|\bwould you do (?:it|that|anything) differently\b|\bis there anything (?:else|you would change)\b"
                         r"|\banything else\b|\bwhat else\b|\bany other (?:examples?|ways|options|thoughts)\b"
                         r"|\bwhat would you change\b|\bwhat if (?:that|it|they|the)\b|\bhow would you improve (?:it|that|this)\b"),
    ("challenge", 0.8, r"\bwhat (?:was|were|is) the (?:hardest|toughest|biggest|most (?:difficult|challenging)|worst) "
                       r"(?:part|challenge|thing|obstacle|moment)\b|\bwhat (?:obstacles|challenges|difficulties) "
                       r"(?:did you|were there)\b|\bwhat went wrong\b|\bwhat was (?:hard|difficult|challenging)\b"),
]]
_REFERENCE = re.compile(r"\byou (?:mentioned|said|talked about|brought up|described|noted|referred to)\b|\bearlier you\b"
                        r"|\bgoing back to\b|\bback to (?:what|the)\b|\bcoming back to\b|\byou just (?:said|mentioned)\b"
                        r"|\bon (?:that|this) (?:note|point|topic)\b|\brelated to (?:that|this)\b|\bbuilding on (?:that|this)\b")
_CONTINUATION = re.compile(r"^(?:and|so|but|then|also|now|and so|(?:okay|ok|alright|right) and)\s+" + _WH + r"\b")
_PRONOUN = re.compile(r"\b(?:that|those|them|this|these|there)\b|\bit\b")


@dataclass(frozen=True)
class Cue:
    kind: str
    strength: float   # 1.0 unmistakable ... 0.5 weak (needs context to count)
    phrase: str

    @property
    def effect(self) -> float:
        """How much context is allowed to bend the matching: full for clear cues, less for weak ones."""
        return 1.0 if self.strength >= 0.8 else 0.6 if self.strength >= 0.6 else 0.4

    @property
    def what(self) -> str:
        return KINDS[self.kind][0]

    @property
    def hint(self) -> str:
        return KINDS[self.kind][1]


def _plain(text: str) -> str:
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^a-z0-9'? ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def detect_followup(text: str) -> Cue | None:
    """Does this sound like a follow-up to whatever was just discussed?  Pure wording check."""
    sents = split_sentences(text) or [text]
    span = " ".join(sents[-2:])
    t = _plain(span)
    words = len(t.replace("?", " ").split())
    if not t:
        return None
    last = _plain(sents[-1])

    if words <= 40 and looks_like_question(text) and _REFERENCE.search(t):
        return Cue("reference", 0.8, _REFERENCE.search(t).group(0))
    if words <= MAX_CUE_WORDS:
        for kind, strength, rx in _CUES:
            m = rx.search(t)
            if m:
                return Cue(kind, strength, m.group(0).strip())
    m = _CONTINUATION.match(last)
    if m and len(last.split()) <= MAX_CUE_WORDS + 4:
        return Cue("continuation", 0.6, m.group(0))
    if words <= MAX_CUE_WORDS + 2 and looks_like_question(text):
        m = _PRONOUN.search(t)
        if m:
            return Cue("pronoun", 0.5, m.group(0))
    return None


# ---------- what to show ----------

@dataclass
class Decision:
    action: str                              # "show" | "unsure" | "followup" | "ignore"
    heard: str = ""
    matches: list[Match] = field(default_factory=list)
    cue: Cue | None = None
    parent: Entry | None = None              # the answer a follow-up refers to
    banner: str = ""
    hint: str = ""
    likely: list[Entry] = field(default_factory=list)   # follow-ups scripted for what is now on screen
    is_followup: bool = False


def _ekey(e: Entry) -> tuple[str, str]:
    return (e.bank, e.question.strip().lower())


def _qkey(text: str) -> str:
    return " ".join(text.strip().lower().split())


class Conversation:
    """Remembers the answers just shown and decides what each new utterance means."""

    def __init__(self, matcher: Matcher, cfg: dict, clock: Callable[[], float] = time.monotonic) -> None:
        self.matcher, self.cfg, self.clock = matcher, cfg, clock
        self._lock = threading.RLock()
        self._recent: list[tuple[tuple[str, str], float, int | None]] = []   # (entry key, time, utterance id)
        self._pending: tuple[str, float] | None = None
        self._kids: dict[str, list[int]] = {}
        self._generic: list[int] = []
        self.refresh()

    # ----- bank bookkeeping -----
    def refresh(self) -> None:
        """Call after the matcher loads banks: rebuilds the follow-up links."""
        with self._lock:
            entries = self.matcher.entries
            kids: dict[str, list[int]] = {}
            for i, e in enumerate(entries):
                for parent in e.follows:
                    kids.setdefault(_qkey(parent), []).append(i)
            self._kids = kids
            self._generic = [i for i, e in enumerate(entries) if e.is_generic_followup]

    def followups_of(self, entry: Entry, limit: int = 5) -> list[Entry]:
        """Follow-ups scripted for `entry` (never itself)."""
        with self._lock:
            es = self.matcher.entries
            return [es[i] for i in self._kids.get(_qkey(entry.question), []) if es[i] is not entry][:limit]

    def note(self, entry: Entry, utt: int | None = None) -> None:
        """Record that `entry` is now on screen (shown by voice, or picked by hand)."""
        with self._lock:
            if utt is not None and self._recent and self._recent[-1][2] == utt:
                self._recent.pop()   # the final transcript replaces this utterance's earlier guess
            self._recent.append((_ekey(entry), self.clock(), utt))
            del self._recent[:-6]

    def _recent_entries(self, now: float) -> list[Entry]:
        """Entries shown within the follow-up window, newest first (current objects, no duplicates)."""
        window = float(self.cfg.get("followup_window_seconds", 120))
        by_key = {}
        for e in self.matcher.entries:
            by_key.setdefault(_ekey(e), e)
        out: list[Entry] = []
        for key, t, _ in reversed(self._recent):
            e = by_key.get(key)
            if e is not None and now - t <= window and all(e is not x for x in out):
                out.append(e)
        return out[:3]

    def recent(self, limit: int = 2) -> list[Entry]:
        """The answers shown lately (within the follow-up window), newest first."""
        with self._lock:
            return self._recent_entries(self.clock())[:limit]

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._pending = None

    # ----- the decision -----
    def resolve(self, text: str, final: bool = True, forced: bool = False, utt: int | None = None) -> Decision:
        """What should the overlay do about `text`?  `final=False` is an early guess made before the
        speaker has certainly finished: it never changes the fragment-joining state."""
        with self._lock:
            now = self.clock()
            heard = text.strip()
            if self._pending and now - self._pending[1] <= JOIN_WINDOW_SEC:
                heard = f"{self._pending[0]} {heard}"
            if not self.matcher.entries or not heard:
                return Decision("ignore", heard)
            thr = float(self.cfg["match_threshold"])
            is_q = looks_like_question(heard)
            cue = detect_followup(heard) if self.cfg.get("followups", True) else None
            if forced:
                cue = Cue("forced", 1.0, "") if cue is None else cue
            recent = self._recent_entries(now) if cue else []
            parent = recent[0] if recent else None
            follow = bool(parent) and cue is not None

            sc = self.matcher.score(heard)
            if sc is None:
                return Decision("ignore", heard)

            boosts: dict[int, float] = {}
            if follow:
                eff = cue.effect
                for rank, anc in enumerate(recent[:2]):          # older answers count for less
                    for i in self._kids.get(_qkey(anc.question), []):
                        if self.matcher.entries[i] is not anc:
                            boosts[i] = max(boosts.get(i, 0.0), FOLLOW_BOOST * eff * (1.0 if rank == 0 else 0.5))
                for i in self._generic:
                    boosts[i] = max(boosts.get(i, 0.0), GENERIC_BOOST * eff)
            matches = self.matcher.rank(sc, 5, boosts)
            if not matches:
                return Decision("ignore", heard)
            best = matches[0]
            in_space = best.boost > 0
            relax = FOLLOW_RELAX * cue.effect if (in_space and cue) else 0.0
            eff_thr = thr - relax
            question_like = is_q or cue is not None

            if final:
                show_ok = best.score >= eff_thr + STRONG_MARGIN or (best.score >= eff_thr and question_like)
            else:   # an early guess made during a pause: act only if it is clearly complete and clearly right
                show_ok = best.score >= eff_thr + STRONG_MARGIN and heard.rstrip().endswith("?")
            if show_ok:
                if final:
                    self._pending = None
                self.note(best.entry, utt)
                d = Decision("show", heard, matches[:3], cue if in_space else None,
                             parent if in_space else None, is_followup=in_space)
                if in_space and parent is not None:
                    d.banner = f"↳ Follow-up ({cue.what}) to: {parent.question}"
                d.likely = self.followups_of(best.entry)
                return d

            if not final:
                return Decision("ignore", heard)   # early guesses never leave a trace unless they show an answer
            self._pending = (heard[-400:], now)
            if follow and (forced or cue.strength >= 0.6):
                # Nothing scripted fits, but this is plainly a follow-up: keep the answer on screen and say what's asked.
                near = [m for m in matches if m.boost > 0][:3]
                return Decision("followup", heard, near, cue, parent,
                                banner=f"↳ Sounds like a follow-up ({cue.what}) to: {parent.question}",
                                hint=cue.hint, likely=self.followups_of(parent), is_followup=True)
            if is_q and best.score >= thr - GUESS_MARGIN:
                return Decision("unsure", heard, matches[:3], cue)
            return Decision("ignore", heard, matches[:3], cue)

    def pick(self, entry: Entry) -> list[Entry]:
        """The user chose `entry` by hand. Returns its scripted follow-ups."""
        self.note(entry)
        with self._lock:
            self._pending = None
        return self.followups_of(entry)
