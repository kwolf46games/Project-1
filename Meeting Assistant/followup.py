"""Follow-up questions: recognising them, tying them to the answer they probe, and finding what to say.

An interviewer's follow-up usually can't be matched on its own ("What was the result?", "Why?", "Tell me more").
Three pieces work together:

  prepared_followups(entry)  the follow-ups you wrote under an answer ("LIKELY FOLLOW-UPS  ·  SAY THIS")
  detect(text)               cue-based check for "this is a follow-up to what was just discussed", and what kind
  focus_passages(...)        which sentences of the anchor answer best address a follow-up nobody prepared for
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from textsim import lexical_similarity, normalize

# ---------- prepared follow-ups written into a bank entry ----------

_FU_HEADER = re.compile(r"^\*\*\s*LIKELY FOLLOW-?UPS\b.*\*\*\s*$", re.I)
_FU_LINE = re.compile(r"^\*\*(?P<q>[^*]+?)\*\*\s*[·•|\-]\s*(?:about\s+)?(?P<sec>\d+)\s*sec\w*\.?\s*(?P<a>.*)$", re.I)
_SCRIPT_MARK = re.compile(r"\*\*SCRIPT\b[^*]*\*\*\s*", re.I)


@dataclass
class PreparedFollowup:
    question: str
    answer: str
    seconds: int = 0


def prepared_followups(entry) -> list[PreparedFollowup]:
    """Follow-ups under a 'LIKELY FOLLOW-UPS' header, written as `**Question**  ·  about 30 sec  Answer…`.

    Questions may be phrased as requests ("Tell me more about…") so a trailing '?' is not required."""
    out: list[PreparedFollowup] = []
    in_section = False
    for raw in entry.response.splitlines():
        line = raw.strip()
        if _FU_HEADER.match(line):
            in_section = True
            continue
        if not in_section or not line:
            continue
        m = _FU_LINE.match(line)
        if m:
            out.append(PreparedFollowup(m["q"].strip(), m["a"].strip(), int(m["sec"])))
        elif line.startswith("**"):
            in_section = False            # the next header (a new question, a note) ends the section
        elif out:
            out[-1].answer = f"{out[-1].answer} {line}".strip()  # an answer that wrapped onto another line
    return out


def script_text(entry) -> str:
    """The answer itself: what you'd say, without the coaching note above it or the follow-ups below it."""
    kept = []
    for line in entry.response.splitlines():
        if _FU_HEADER.match(line.strip()):
            break
        kept.append(line)
    text = "\n".join(kept)
    m = _SCRIPT_MARK.search(text)
    return (text[m.end():] if m else text).replace("**", "").strip()


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.?!])[\"”’']?\s+", flat) if len(s.split()) >= 3]


# ---------- recognising a follow-up ----------

KIND_LABEL = {
    "elaborate": "wants more detail", "clarify": "asking what you meant", "example": "wants an example",
    "role": "asking about your role", "number": "asking about a number", "outcome": "asking about the result",
    "reaction": "asking how people reacted", "challenge": "asking about the hard part",
    "hindsight": "asking what you'd change or learned", "why": "asking why", "how": "asking how",
    "more": "asking for another / something else", "back": "referring back to what you said",
    "probe": "follow-up",
}

# (kind, weight, pattern) on normalised text; weights combine as independent evidence (noisy-or)
_RULES: list[tuple[str, float, re.Pattern]] = [(k, w, re.compile(p)) for k, w, p in (
    ("elaborate", .75, r"\b(?:tell|say) (?:me|us) (?:a (?:little |bit )?)?more\b"),
    ("elaborate", .75, r"\b(?:elaborate|expand|unpack|go deeper|go into (?:more )?detail|dig (?:deeper|into))\b"),
    ("elaborate", .65, r"\bwalk (?:me|us) through (?:that|this|it|the (?:process|steps|details?))\b"),
    ("elaborate", .55, r"\b(?:explain|describe|clarify|break down) (?:that|this|it|how)\b"),
    ("clarify", .6, r"\bwhat do you mean\b|\bwhat (?:does|did) that mean\b|\bhow so\b|\bcan you clarify\b"),
    ("example", .7, r"\b(?:give|share) (?:me|us) (?:a|an|one|another|some)(?: (?:specific|concrete|real|quick))? "
                    r"(?:example|instance)s?\b|\bfor (?:example|instance)\b|\bspecific (?:time|example|instance)\b"
                    r"|\bexamples? of (?:that|this|it|what)\b"),
    ("role", .7, r"\bwhat (?:was|is) your (?:specific |exact |actual |own )?(?:role|part|contribution|responsibility)\b"
                 r"|\bwhat did you (?:personally |actually |specifically )?(?:do|contribute)\b"
                 r"|\bwere you (?:the|a) (?:leader|lead|one)\b|\bwho was (?:responsible|involved)\b"),
    ("number", .7, r"\bwalk (?:me|us) through (?:that|the|those) numbers?\b|\b(?:that|the|those) numbers?\b"
                   r"|\bwhat (?:percent|percentage|size|scale)\b|\bhow did you (?:measure|calculate|quantify)\b"
                   r"|\bhow (?:many|much|long|big|large) (?:was|were|did|is)\b.*\b(?:that|it|this)\b"),
    ("outcome", .7, r"\bwhat (?:was|were) the (?:result|results|outcome|impact|end result)\b"
                    r"|\bhow did (?:that|it|this) (?:turn out|go|end|work out)\b|\bwhat happened (?:next|after|then)\b"
                    r"|\bdid (?:that|it) work\b|\bwhat came of (?:that|it)\b"),
    ("reaction", .7, r"\bhow did (?:people|they|he|she|the \w+|your \w+|everyone|everybody|others?) "
                     r"(?:react|respond|feel|take (?:it|that|the news))\b|\bwhat was (?:their|his|her|the) "
                     r"(?:reaction|response)\b"),
    ("challenge", .65, r"\bwhat (?:was|were) the (?:hardest|toughest|most (?:difficult|challenging)|biggest "
                       r"(?:challenge|obstacle))\b|\b(?:something|anything) (?:go|going|went) wrong\b"
                       r"|\bwhat (?:went wrong|obstacles)\b|\bany (?:obstacles|challenges|pushback)\b"),
    ("hindsight", .7, r"\bwhat would you do differently\b|\bin hindsight\b|\blooking back\b|\bwhat did you learn\b"
                      r"|\bwould you do (?:it|that|anything) (?:again|differently)\b"
                      r"|\bif you (?:could|had to) do (?:it|that|this) again\b"),
    ("why", .5, r"\bwhy did you (?:do|choose|decide|pick|go with|take|say) (?:that|this|it|so)\b|\bwhy (?:that|this|so)\b"
                r"|\bwhat made you (?:do|choose|decide|pick|go) (?:that|this|it|with)\b|\bhow come\b"),
    ("how", .55, r"\bhow (?:did|would) you (?:do|handle|approach|manage|deal with|get|go about) (?:that|this|it)\b"),
    ("more", .55, r"\b(?:what|how) about (?:the|your|that|this|when|if)\b|\b(?:any|anything) (?:other|else)\b"
                  r"|\b(?:another|other) (?:example|time|one|weakness|story)\b|\bwhat else\b|\bwhat if\b"),
    ("back", .6, r"\byou (?:mentioned|said|talked about|brought up|were saying|just said|described)\b"
                 r"|\b(?:going|circling|coming) back\b|\bgo back to\b|\bearlier you\b|\bon that (?:note|point|topic)\b"
                 r"|\b(?:following|follow) up\b|\b(?:building|expanding) on that\b|\bback to what you\b"),
)]
_LEAD = re.compile(r"^(?:and|so|but|then|also|okay so|ok so|right so|and so)\b")
_WH = re.compile(r"^(?:what|why|how|when|where|which|who|did|do|does|were|was|is|are|can|could|would)\b")
_PRONOUN = re.compile(r"\b(?:that|this|those|it|they|them)\b")
_STILL = re.compile(r"\b(?:anyway|still|even so|even then)\b")


@dataclass
class Signal:
    score: float = 0.0                        # 0..1: how sure this is a follow-up to the previous topic
    kind: str = ""                            # key of KIND_LABEL, "" when score is 0
    cues: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return KIND_LABEL.get(self.kind, "")

    def __bool__(self) -> bool:
        return self.score >= FOLLOWUP_CUE_THRESHOLD


FOLLOWUP_CUE_THRESHOLD = 0.5


def detect(text: str) -> Signal:
    """Score how much `text` reads as a follow-up to whatever was just discussed, from its wording alone."""
    n = normalize(text)
    if not n:
        return Signal()
    hits: list[tuple[float, str, str]] = []
    for kind, weight, pat in _RULES:
        m = pat.search(n)
        if m:
            hits.append((weight, kind, m.group()))
    words = n.split()
    structural: list[tuple[float, str]] = []
    if _LEAD.match(n) and len(words) > 2:
        structural.append((.25, "starts with a connector"))
    if len(words) <= 2 and (_WH.match(n) or text.strip().endswith("?")):
        structural.append((.55, "one- or two-word question"))
    elif len(words) <= 4 and (_WH.match(n) or text.strip().endswith("?")):
        structural.append((.3, "very short question"))
    if len(words) <= 10 and _PRONOUN.search(n) and _WH.match(n):
        structural.append((.2, "refers to 'that/it'"))
    if _STILL.search(n):
        structural.append((.2, "'anyway/still'"))
    if not hits and not structural:
        return Signal()
    miss = 1.0
    for w, *_ in hits:
        miss *= 1 - w
    for w, _ in structural:
        miss *= 1 - w
    kind = max(hits)[1] if hits else "probe"
    return Signal(min(0.97, 1 - miss), kind, [c for *_, c in sorted(hits, reverse=True)] + [c for _, c in structural])


# ---------- what to say when nothing was prepared ----------

_KIND_WORDS: dict[str, tuple[str, ...]] = {
    "outcome": ("result", "outcome", "led to", "resulted", "grew", "growth", "increase", "reduced", "saved", "signed",
                "won", "first place", "offer", "delivered", "closed", "%", "percent"),
    "number": ("%", "$", "percent", "million", "thousand", "hundred", "projects", "bids", "clients", "people",
               "crews", "meetings", "weeks", "months", "years"),
    "role": ("i ran", "i led", "i built", "i set", "i made", "my job", "my role", "my part", "i was responsible",
             "i owned", "i handled", "i managed", "i did", "i wrote", "i created", "i sat", "i worked"),
    "reaction": ("pushed back", "react", "thanked", "feedback", "agreed", "concern", "resist", "supportive", "upset"),
    "challenge": ("hardest", "difficult", "challenge", "problem", "issue", "behind", "pressure", "obstacle", "wrong",
                  "last minute", "tight", "deadline", "conflict", "struggle"),
    "hindsight": ("learned", "differently", "next time", "now i", "lesson", "realized", "would have", "since then"),
    "why": ("because", "so that", "reason", "decided", "chose", "wanted", "which is why"),
    "how": ("first", "then", "started", "approach", "process", "step", "so i"),
    "example": ("for example", "for instance", "one time", "once", "specifically", "when i"),
}


def focus_passages(entry, text: str, kind: str = "", k: int = 2,
                   embed: Callable[[list[str]], np.ndarray] | None = None) -> list[str]:
    """Sentences of the entry's answer that best address a follow-up, in the order they're spoken."""
    sents = _sentences(script_text(entry))
    if not sents:
        return []
    words = _KIND_WORDS.get(kind, ())
    dense = np.zeros(len(sents))
    if embed is not None:
        v = np.asarray(embed([text] + sents), dtype=np.float32)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        dense = np.clip((v[1:] @ v[0] - 0.5) / 0.4, 0, 1)
    scored = []
    for i, s in enumerate(sents):
        low = s.lower()
        hits = sum(1 for w in words if w in low) + (1 if kind in ("number", "outcome") and re.search(r"[\d$%]", s) else 0)
        kw = min(1.0, hits / 2)
        lex = lexical_similarity(text, s)
        score = 0.4 * kw + 0.3 * lex + 0.3 * dense[i] if embed is not None else 0.5 * kw + 0.5 * lex
        scored.append((score, i))
    best = sorted(((sc, i) for sc, i in scored if sc >= 0.12), reverse=True)[:k]
    return [sents[i] for _, i in sorted(best, key=lambda x: x[1])]
