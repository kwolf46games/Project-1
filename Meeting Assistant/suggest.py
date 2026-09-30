"""Suggest other ways an interviewer might word a bank question.

Matching is only as good as the wordings a question is stored under (its `also:` list). This proposes new
ones from three sources, then keeps the few that add the most variety without changing the meaning:

  - splitting compound questions ("A? B?", "A / B", "A, and B") into the pieces that may be asked alone
  - rewriting into the other forms people use (imperative <-> question, digits <-> words,
    "tell me what X is", "what do you think X looks like" <-> "what would X look like")
  - curated rewordings of common interview questions
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from textsim import fingerprint, lexical_similarity, normalize, similarity

MIN_MEANING = 0.80     # embedding similarity to the original below which a rewording has drifted too far
TOO_SIMILAR = 0.93     # lexical similarity to an existing wording above which a suggestion adds nothing


@dataclass
class Suggestion:
    text: str
    score: float       # similarity to the original question (embedding if available, else lexical)


# ---------- curated rewordings of common questions (key: regex on the normalised question) ----------

_CURATED: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"tell me about yourself|walk me through your (?:resume|background|cv)",
     ("Walk me through your background.", "Tell me a bit about yourself.", "Can you introduce yourself?",
      "Give me a quick overview of your experience.", "Take me through your resume.")),
    (r"why should we hire you",
     ("Why should we choose you?", "What makes you the right person for this role?",
      "What sets you apart from other candidates?", "Why are you the best fit for this position?")),
    (r"why should we not hire you",
     ("Why shouldn't we hire you?", "What would be the biggest concern about hiring you?",
      "What is the strongest argument against hiring you?")),
    (r"(?:biggest|greatest) weakness",
     ("What would you say is your greatest weakness?", "What is one area where you need to improve?",
      "What do you struggle with most?", "Where do you feel you have the most room to grow?")),
    (r"(?:biggest|greatest) strength",
     ("What would you say is your greatest strength?", "What are you best at?",
      "What is the strongest skill you bring?")),
    (r"where do you see yourself in (?:five|5) years",
     ("Where do you see yourself in 5 years?", "Where would you like to be in five years?",
      "What are your long-term career goals?", "What does your career look like in five years?")),
    (r"why do you want to work (?:here|at|for)",
     ("What draws you to this company?", "What interests you about working here?",
      "Why are you interested in this role?")),
    (r"what motivates you", ("What drives you?", "What keeps you motivated?", "What gets you excited about work?")),
    (r"(?:any|do you have) questions for (?:us|me)",
     ("Is there anything you'd like to ask us?", "What questions do you have for us?")),
    (r"why (?:are you leaving|did you leave)",
     ("What made you decide to leave your last role?", "What prompted you to leave?")),
    (r"(?:proudest|greatest) (?:accomplishment|achievement)",
     ("What accomplishment are you most proud of?", "Tell me about something you've achieved that you're proud of.")),
    (r"tell me about (?:a )?(?:failure|setback)",
     ("Tell me about a time you failed.", "Describe a time things didn't go as planned.",
      "What is a mistake you learned from?")),
    (r"difficult (?:client|customer)",
     ("Describe a time you dealt with a difficult client.", "Tell me about a challenging client.",
      "How have you handled a tough customer?")),
    (r"critical feedback", ("Tell me about a time you received constructive criticism.",
                            "Describe a time someone gave you tough feedback.")),
    (r"learn(?:ed)? something new quickly", ("Tell me about a time you had to pick something up fast.",
                                              "Describe a time you had to get up to speed quickly.")),
    (r"\bconflict\b", ("Tell me about a time you disagreed with a teammate.",
                       "Describe a conflict on a team and how you handled it.")),
    (r"rejection", ("Tell me about a time someone told you no.", "Describe a time you kept going after being turned down.")),
    (r"competing priorities", ("Tell me about a time you had too much on your plate.",
                               "How did you handle competing deadlines?")),
    (r"led a team|lead a team", ("Tell me about a time you were in charge of a team.",
                                  "Describe a time you took the lead on a group project.")),
    (r"took initiative|take initiative", ("Tell me about a time you took ownership without being asked.",
                                          "Describe a time you went out of your way to start something.")),
    (r"above and beyond", ("Tell me about a time you went the extra mile.",
                           "Describe a time you did more than was expected of you.")),
    (r"persuade|influence a decision", ("Tell me about a time you convinced someone to see it your way.",
                                        "Describe a time you changed someone's mind.")),
    (r"under pressure|plan changed", ("Tell me about a time you had to work under a tight deadline.",
                                      "Describe a time when plans changed suddenly.")),
    (r"organi[sz]ed one", ("Tell me about a time you kept a group on track.",
                           "Describe a time you took charge of the logistics.")),
)

_NUMBERS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "twelve": "12"}
_WH = r"(what|why|how|where|when|which|who)"


# ---------- generation ----------

def _cap(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:]


def _q(s: str) -> str:
    return _cap(s).rstrip(".?! ") + "?"


def _imp(s: str) -> str:
    return _cap(s).rstrip(".?! ") + "."


def _units(question: str) -> list[str]:
    """The question with labels stripped, plus the pieces of a compound question."""
    q = question.strip().replace("’", "'").replace("“", '"').replace("”", '"')
    q = re.sub(r"^[A-Za-z][A-Za-z ]{2,30}:\s+(?=\S)", "", q)                      # "College questions: why X?"
    q = re.sub(r"^(?:in your own words|so|okay|ok|now|and)[, ]+", "", q, flags=re.I)
    units = [q]
    pieces: list[str] = []
    for part in re.split(r"\s+/\s+|(?<=[?!])\s+", q):
        pieces += re.split(rf",\s+(?:and|or)\s+(?={_WH}\b)", part, flags=re.I)
    pieces = [p for p in (x for x in pieces if x) if p != q]
    if len(pieces) > 1:
        units += [_q(p) if re.match(rf"{_WH}\b|(?:can|could|do|did|would|have|are) ", p, re.I) else _imp(p)
                  for p in pieces]
    m = re.match(r"^(tell me about|describe|give me an example of|talk about)\s+(.+?),\s+or\s+((?:a|an|the|your|their)\s+.+)$",
                 q.rstrip(".?! "), re.I)
    if m:  # "Tell me about a conflict, or a teammate who..." -> one request per noun phrase
        left, right = m.group(2), m.group(3)
        if re.match(r"a time\b", left, re.I) and not re.search(r"\b(?:who|that|which)\b", right, re.I):
            right = "a time when " + right  # "...a time X, or a plan changed" -> "a time when a plan changed"
        units += [_imp(f"{m.group(1)} {left}"), _imp(f"{m.group(1)} {right}")]
    return list(dict.fromkeys(units))


def _indirect(text: str) -> list[str]:
    """'What is X' -> 'what X is', 'Where do you see...' -> 'where you see...': for 'Can you tell me ...'."""
    body = text.rstrip(".?! ")
    m = re.match(rf"^{_WH} (is|are|was|were) (.+)$", body, re.I)
    if m and m.group(1).lower() != "how":
        return [f"{m.group(1).lower()} {m.group(3)} {m.group(2).lower()}"]
    m = re.match(rf"^{_WH} (do|does|should|would|could|can|will) (you|we|they|i|he|she) (.+)$", body, re.I)
    if m:
        aux = "" if m.group(2).lower() in ("do", "does") else f" {m.group(2).lower()}"
        return [f"{m.group(1).lower()} {m.group(3)}{aux} {m.group(4)}"]
    m = re.match(r"^what does ([^,]+?) do$", body, re.I)
    if m:
        return [f"what {m.group(1)} does"]
    return []


def _rules(unit: str) -> list[str]:
    out: list[str] = []
    orig = unit.strip().rstrip(".?! ")
    is_q = unit.strip().endswith("?")
    fin = _q if is_q else _imp

    # behavioural frame: "tell me about a time you X"
    m = re.match(r"^(?:tell me about|describe|give me an example of|talk about|walk me through)\s+"
                 r"(?:a time|the time|a situation|an instance|a moment)(?:\s+(?:when|where|that))?\s+(.+)$", orig, re.I)
    if m:
        x = m.group(1)
        out += [_imp(f"tell me about a time when {x}"), _q(f"can you tell me about a time {x}"),
                _imp(f"describe a situation where {x}"), _imp(f"give me an example of a time {x}"),
                _imp(f"walk me through a time when {x}"), _q(f"can you share an experience where {x}"),
                _imp(f"I'd like to hear about a time when {x}")]
        if x.lower().startswith("you "):
            out.append(_q(f"have you ever been in a situation where {x}"))
    else:
        m = re.match(r"^(?:tell me about|describe|talk about|give me an example of)\s+(.+)$", orig, re.I)
        if m:
            x = m.group(1)
            out += [_q(f"can you tell me about {x}"), _imp(f"describe {x}"), _imp(f"give me an example of {x}"),
                    _q(f"could you talk about {x}"), _imp(f"I'd love to hear about {x}")]
        m = re.match(r"^walk me through\s+(.+)$", orig, re.I)
        if m:
            x = m.group(1)
            out += [_q(f"can you walk me through {x}"), _imp(f"take me through {x}"), _imp(f"talk me through {x}")]

    # "Why X?" where X is a noun phrase ("Why AlphaSights?", "Why finance?")
    m = re.match(r"^why\s+(?!do\b|did\b|should\b|would\b|could\b|can\b|will\b|are\b|is\b|was\b|were\b|have\b|"
                 r"has\b|not\b|this\b|that\b|it\b)(.+)$", orig, re.I)
    if m:
        x = m.group(1)
        out += [_q(f"why are you interested in {x}"), _q(f"what draws you to {x}"),
                _q(f"what interests you about {x}"), _q(f"why did you choose {x}")]

    m = re.match(r"^how is (.+?) different from (.+)$", orig, re.I)
    if m:
        a, b = m.groups()
        out += [_q(f"how does {a} differ from {b}"), _q(f"what sets {a} apart from {b}"),
                _q(f"what makes {a} different from {b}")]

    m = re.match(r"^what do you think (.+?) (looks|is|feels) like$", orig, re.I)
    if m:
        x, v = m.group(1), m.group(2).lower()
        verb = {"looks": "look", "is": "be", "feels": "feel"}[v]
        out += [_q(f"what would {x} {verb} like"), _q(f"what do you imagine {x} {v} like"),
                _q(f"can you describe what you think {x} {v} like")]

    m = re.match(r"^what would you do differently(.*)$", orig, re.I)
    if m:
        rest = m.group(1)
        out += [_q(f"if you could do it over, what would you do differently{rest}"),
                _q(f"looking back, what would you change{rest}")]

    m = re.match(r"^(.+?) resonates with you$", orig, re.I)
    if m:
        out += [_q(f"{m.group(1)} stands out to you"), _q(f"{m.group(1)} speaks to you")]

    # a bare noun phrase: "Favorite class?" -> "What is your favorite class?"
    if re.match(r"^(?:favou?rite|biggest|greatest|proudest|best|worst|hardest|toughest)\b", orig, re.I) \
            and len(orig.split()) <= 4:
        out += [_q(f"what is your {orig}"), _q(f"what was your {orig}"), _q(f"what would you say is your {orig}")]

    for ind in _indirect(orig):
        out += [_q(f"can you tell me {ind}"),
                _q(f"could you explain {ind}") if ind.startswith("what ") else _imp(f"I'd like to know {ind}")]

    # digits <-> words (speech models print either)
    for w, d in _NUMBERS.items():
        for old, new in ((w, d), (d, w)):
            if re.search(rf"\b{old}\b", orig, re.I):
                out.append(fin(re.sub(rf"\b{old}\b", new, orig, flags=re.I)))
                break
    return out


def _curated(unit: str) -> list[str]:
    n = normalize(unit)
    out: list[str] = []
    for pat, alts in _CURATED:
        if re.search(pat, n) and not (pat.startswith("why should we hire") and "not" in n.split()):
            out += list(alts)
    return out


def generate(question: str) -> list[str]:
    """Every candidate rewording, best-sourced first (pieces and curated, then rewrites). Not yet filtered."""
    units = _units(question)
    cands: list[str] = []
    cands += units[1:]                                   # pieces of a compound question
    for u in units:
        cands += _curated(u)
    for u in (units[1:] if len(units) > 1 else units):   # rewrite the pieces of a compound, not the whole
        cands += _rules(u)
    return list(dict.fromkeys(c for c in cands if len(c.split()) >= 2))


# ---------- selection ----------

def suggest_phrasings(question: str, existing: Iterable[str] = (), n: int = 5,
                      embed: Callable[[list[str]], np.ndarray] | None = None) -> list[Suggestion]:
    """Up to `n` new wordings for `question`, skipping ones already stored in `existing`.

    With `embed`, candidates that drift in meaning are dropped and the rest are chosen to differ from each
    other; without it, variety is judged on wording alone."""
    have = [question, *existing]
    have_prints = [fingerprint(h) for h in have]
    cands = [c for c in generate(question)
             if all(similarity(fingerprint(c), hp) < TOO_SIMILAR for hp in have_prints)]
    if not cands:
        return []
    if embed is not None:
        v = np.asarray(embed([question, *existing, *cands]), dtype=np.float32)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        k = 1 + len(list(existing))
        cv = v[k:]
        meaning = cv @ v[0]
        alive = [i for i, m in enumerate(meaning) if m >= MIN_MEANING]
        covered = list(v[1:k])                 # wordings already stored count as covered
        picked: list[int] = []
        while alive and len(picked) < n:
            def value(i: int) -> float:        # max marginal relevance: stay on meaning, avoid repeating what's covered
                redundancy = max((float(cv[i] @ c) for c in covered), default=0.0)
                return 0.3 * float(meaning[i]) - 0.7 * redundancy - 0.001 * i   # earlier (better-sourced) wins ties
            best = max(alive, key=value)
            picked.append(best)
            alive.remove(best)
            covered.append(cv[best])
        return [Suggestion(cands[i], float(meaning[i])) for i in picked]
    picked_prints = list(have_prints)
    out: list[Suggestion] = []
    for c in cands:
        fp = fingerprint(c)
        if max(similarity(fp, p) for p in picked_prints) >= 0.85:
            continue
        out.append(Suggestion(c, lexical_similarity(c, question)))
        picked_prints.append(fp)
        if len(out) == n:
            break
    return out
