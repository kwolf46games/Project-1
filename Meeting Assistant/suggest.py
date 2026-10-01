"""Suggest extra ways a question might be asked (the "also:" phrasings), entirely offline.

More phrasings make matching more reliable, but thinking them up is tedious.  Candidates come from
three places, mixed so you see variety rather than eight near-copies:

* intent    - well-known question types ("biggest weakness", "tell me about a conflict"...) with the
              alternative wordings interviewers really use;
* pattern   - grammatical restructurings of *this* question ("Tell me about X" <-> "Can you walk me
              through X?", "What is X?" <-> "Can you tell me what X is?");
* synonym   - one safe word swap at a time (handle / deal with / manage, biggest / greatest...).

vet() then checks each candidate against the real matcher when it is available: it must stay
close to the original question, and it must not match some *other* question better (which would
make the overlay show the wrong answer).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from bank import Entry
from matcher import Matcher, _stem, content_tokens

REQUEST_VERBS = ("tell", "describe", "explain", "share", "walk", "give", "talk", "take", "show", "outline",
                 "summarize", "summarise", "discuss", "name", "list", "elaborate", "think of", "provide")
_INTERROGATIVE = re.compile(r"^(what|why|how|when|where|which|who|whose|can|could|would|will|do|does|did|have|has|is|are"
                            r"|was|were|should|any|anything|is there|are there)\b", re.I)


@dataclass
class Suggestion:
    text: str
    source: str                  # "intent" | "pattern" | "synonym"
    sim: float | None = None     # closeness to the original question (set by vet)
    rival: str = ""              # another question this wording matches better (set by vet)

    @property
    def ok(self) -> bool:
        return not self.rival and (self.sim is None or self.sim > 0)


# ---------- known question types ----------
# Each entry: groups of alternative words that must ALL be represented (a word from every group),
# then wordings.  Words are compared after the matcher's light stemming.
_INTENTS: list[tuple] = [   # (word groups, wordings[, words that rule it out])
    ([["tell", "introduce", "walk", "describe"], ["yourself", "background", "resume", "cv", "career"]], [
        "Tell me about yourself", "Can you introduce yourself?", "Walk me through your background",
        "Give me a quick overview of who you are", "Why don't you start by telling me about yourself?",
        "Tell me a little about your background"]),
    ([["strength", "strong"]], [
        "What are your greatest strengths?", "What would you say you're best at?", "What are you really good at?",
        "What do you consider your biggest strength?", "What strengths would you bring to this role?"]),
    ([["weakness", "weak", "flaw", "improve", "improvement"]], [
        "What is your biggest weakness?", "What's one weakness you're working on?",
        "What would you say is your greatest area for improvement?", "Tell me about a weakness you have",
        "Where do you think you need to improve?", "What do you struggle with the most?"]),
    ([["why"], ["company", "organization", "organisation", "us", "here"],
      ["want", "interest", "interested", "apply", "applying", "attract", "appeal", "excite", "choose", "join", "work"]], [
        "What attracts you to this company?", "Why do you want to work here?", "Why are you interested in this company?",
        "What made you apply to us?", "What interests you about working for us?"],
     ["leave", "left", "leaving", "quit"]),
    ([["why"], ["role", "position", "job", "opening", "opportunity"],
      ["interest", "interested", "apply", "applying", "attract", "appeal", "excite", "want", "choose"]], [
        "What interests you about this role?", "Why are you interested in this position?",
        "What made you apply for this job?", "What attracted you to this opportunity?"],
     ["leave", "left", "leaving", "quit", "last", "previous", "current", "former"]),
    ([["why"], ["hire", "choose", "pick", "select"], ["you"]], [
        "What makes you the right person for this job?", "Why are you the best candidate?",
        "What would you bring to our team?", "What sets you apart from other candidates?",
        "Why should we choose you over other candidates?"]),
    ([["see", "where", "future", "goal"], ["year", "years", "future", "long-term", "career"]], [
        "Where do you see yourself in five years?", "What are your long-term career goals?",
        "Where would you like to be in five years?", "What do you want your career to look like in five years?",
        "What are your career goals?"]),
    ([["salary", "compensation", "pay", "wage", "money"]], [
        "What are your salary expectations?", "What kind of compensation are you looking for?",
        "What salary range do you have in mind?", "How much are you looking to make?"]),
    ([["leave", "leaving", "left", "quit", "looking"], ["job", "company", "position", "role", "employer", "search"]], [
        "Why are you looking to leave your current job?", "Why did you leave your last job?",
        "What made you decide to leave your previous role?", "What's prompting your job search?"]),
    ([["conflict", "disagreement", "disagree", "argument", "clash"]], [
        "Tell me about a time you had a conflict at work", "Describe a disagreement with a coworker and how you handled it",
        "Have you ever had a conflict with a teammate?", "How do you handle conflict at work?",
        "Give me an example of a time you disagreed with someone"]),
    ([["fail", "failure", "failed", "mistake", "error", "wrong"]], [
        "Tell me about a time you failed", "Describe a mistake you made and what you learned from it",
        "What's the biggest mistake you've made at work?", "Can you share a time something didn't go as planned?",
        "Tell me about a failure and how you handled it"]),
    ([["lead", "leader", "leadership", "led"], ["style", "approach", "philosophy"]], [
        "Describe your leadership style", "How would you describe yourself as a leader?", "What kind of leader are you?",
        "What's your approach to leading a team?"]),
    ([["lead", "leader", "led", "leading"], ["time", "example", "situation", "experience", "ever", "occasion", "instance"]], [
        "Tell me about a time you led a team", "Give me an example of leading a project",
        "Have you ever had to lead a group?", "Describe a situation where you took the lead"],
     ["style", "approach", "philosophy"]),
    ([["team", "teamwork", "collaborate", "collaboration", "cooperate"]], [
        "Tell me about a time you worked on a team", "How do you work with others?",
        "Describe your experience working in a team", "Give me an example of successful teamwork",
        "What role do you usually play on a team?"],
     ["lead", "led", "leader", "leading", "leadership"]),
    ([["pressure", "stress", "stressful", "overwhelmed"]], [
        "How do you handle pressure?", "How do you deal with stress at work?", "Tell me about a time you worked under pressure",
        "How do you cope with tight deadlines?", "Describe a high-pressure situation and how you handled it"]),
    ([["achievement", "accomplishment", "proud", "success", "accomplish", "achieve"]], [
        "What is your greatest accomplishment?", "What achievement are you most proud of?",
        "Tell me about something you're proud of at work", "What's your biggest professional success?",
        "Describe a time you achieved something significant"]),
    ([["challenge", "difficult", "obstacle", "hardest", "toughest", "tough"], ["time", "situation", "overcome", "faced", "face", "problem"]], [
        "Tell me about a challenge you overcame", "Describe a difficult situation at work and how you dealt with it",
        "What's the hardest problem you've had to solve?", "Give me an example of overcoming an obstacle",
        "Tell me about a tough situation you faced"]),
    ([["prioritize", "prioritise", "priority", "priorities", "organize", "organise", "multitask", "multiple"]], [
        "How do you prioritize your work?", "How do you manage competing deadlines?", "How do you organize your time?",
        "What's your approach to managing multiple tasks?"]),
    ([["question"], ["us", "me", "you", "ask", "have"]], [
        "Do you have any questions for us?", "Is there anything you'd like to ask us?",
        "What questions do you have for me?", "Anything you want to know about the role or the company?"]),
    ([["start", "available", "availability", "notice", "begin"]], [
        "When could you start?", "What's your availability?", "How soon can you begin?", "What's your notice period?"]),
    ([["motivate", "motivation", "motivated", "drive", "drives", "inspire", "inspires"]], [
        "What motivates you?", "What drives you at work?", "What keeps you motivated?",
        "What gets you excited about your work?"]),
    ([["feedback", "criticism", "critique", "criticize"]], [
        "How do you handle criticism?", "How do you respond to feedback?", "Tell me about a time you received tough feedback",
        "How do you take constructive criticism?"]),
    ([["environment", "culture", "workplace", "style"], ["work", "prefer", "ideal", "thrive", "best"]], [
        "What kind of work environment do you prefer?", "How would you describe your work style?",
        "What's your ideal workplace?", "What type of culture do you thrive in?"]),
    ([["decision", "decide", "decisions"]], [
        "Tell me about a tough decision you had to make", "Describe a time you had to make a difficult decision",
        "How do you make decisions?", "Walk me through how you make important decisions"]),
    ([["problem"], ["solve", "solving", "solved", "approach", "process"]], [
        "Tell me about a time you solved a difficult problem", "How do you approach problem solving?",
        "Describe your problem-solving process", "Walk me through how you solve problems"]),
    ([["hobby", "hobbies", "interest", "interests", "free", "outside"], ["work", "time", "life"]], [
        "What do you do in your free time?", "What are your hobbies?", "What do you like to do outside of work?"]),
    ([["know", "familiar", "learn", "learned", "research"], ["company", "us", "organization", "organisation"]], [
        "What do you know about us?", "How familiar are you with our company?",
        "What have you learned about our organization?", "Tell me what you know about the company"]),
    ([["above", "extra", "exceed", "exceeded"], ["beyond", "mile", "expectation", "expectations"]], [
        "Describe a time you went the extra mile", "Give me an example of exceeding expectations",
        "When have you done more than was asked of you?"]),
    ([["change", "adapt", "adaptable", "flexible", "flexibility", "adjust"]], [
        "How do you handle change?", "Tell me about a time you had to adapt quickly", "How adaptable are you?",
        "Describe a time you adjusted to a major change"]),
]

_SYNONYMS: list[tuple[str, list[str]]] = [
    ("handle", ["deal with", "manage", "approach", "respond to"]),
    ("deal with", ["handle", "manage"]),
    ("biggest", ["greatest", "main", "top"]),
    ("greatest", ["biggest", "top"]),
    ("describe", ["explain"]),
    ("challenge", ["difficulty", "obstacle"]),
    ("mistake", ["error", "misstep"]),
    ("goal", ["objective", "aim"]),
    ("goals", ["objectives", "aims"]),
    ("job", ["role", "position"]),
    ("role", ["position", "job"]),
    ("company", ["organization", "employer"]),
    ("coworker", ["colleague", "teammate"]),
    ("colleague", ["coworker", "teammate"]),
    ("boss", ["manager", "supervisor"]),
    ("manager", ["supervisor", "boss"]),
    ("conflict", ["disagreement"]),
    ("weakness", ["area for improvement", "shortcoming"]),
    ("strength", ["strong point"]),
    ("improve", ["get better at"]),
    ("experience with", ["background in", "exposure to"]),
    ("work with", ["collaborate with", "partner with"]),
    ("tell me about", ["talk to me about"]),
    ("difficult", ["challenging", "tough", "hard"]),
    ("project", ["initiative", "piece of work"]),
    ("quickly", ["fast", "rapidly"]),
    ("your approach to", ["how you approach", "your way of handling"]),
]


# ---------- helpers ----------

def _key(text: str) -> str:
    return " ".join(content_tokens(text.replace("?", " "))) + "|" + re.sub(r"[^a-z0-9]+", "", text.lower())


def _cap(s: str) -> str:
    s = s.strip()
    return s[:1].upper() + s[1:]


def _low(s: str) -> str:
    """Lower-case the first word unless it is I or an acronym."""
    first = s.split(" ", 1)[0]
    return s if first == "I" or first.isupper() and len(first) > 1 else s[:1].lower() + s[1:]


def _finish(s: str, question: bool = False) -> str:
    """Tidy spacing and capitals; a wording keeps the "?" it was written with."""
    asked = s.rstrip().endswith("?")
    s = _cap(re.sub(r"\s+", " ", s).strip().rstrip("?.!").strip())
    return s + "?" if asked or (question and _INTERROGATIVE.match(s)) else s


def _starts_with_verb(rest: str) -> bool:
    return rest.lower().startswith(REQUEST_VERBS)


def _patterns(b: str) -> list[str]:
    out: list[str] = []
    add = out.append

    wrappers: list[str] = []     # polite re-wordings of the same request: useful, but least different, so last
    m = re.match(r"(?i)^(can|could|would|will) you (please )?(.+)$", b)
    if m and _starts_with_verb(m.group(3)):
        rest = m.group(3)
        add(_cap(rest))
        wrappers += [f"{aux.capitalize()} you {rest}?" for aux in ("can", "could") if aux != m.group(1).lower()]
        wrappers += [f"I'd like you to {rest}", f"Would you be able to {rest}?"]
    elif re.match(r"(?i)^(tell|describe|explain|share|walk|give|talk|take|show|outline|summari[sz]e|discuss|name|list)\b", b):
        low = _low(b)
        wrappers += [f"Can you {low}?", f"Could you {low}?", f"Please {low}", f"I'd like you to {low}"]

    m = re.match(r"(?i)^tell me about (.+)$", b)
    if m:
        r = m.group(1)
        if re.match(r"(?i)^an? ", r):
            add(f"Describe {r}")
            add(f"Give me an example of {r}")
            add(f"Walk me through {r}")
            add(f"Can you share {r} with me?")
        else:
            if r.lower() != "yourself":
                add(f"Walk me through {r}")
            add(f"I'd like to hear about {r}")
            add(f"Talk to me about {r}")
            add(f"Describe {r}")
            add(f"What can you tell me about {r}?")
    m = re.match(r"(?i)^describe (.+)$", b)
    if m:
        r = m.group(1)
        add(f"Tell me about {r}")
        add(f"Walk me through {r}")
        add(f"Can you describe {r}?")
        if r.lower().startswith("your "):
            add(f"How would you describe {r}?")
    m = re.match(r"(?i)^walk me through (.+)$", b)
    if m:
        r = m.group(1)
        add(f"Take me through {r}")
        add(f"Talk me through {r}")
        add(f"Tell me about {r}")
        add(f"Can you walk me through {r}?")
    m = re.match(r"(?i)^give me an? (?:example|instance) of (.+)$", b)
    if m:
        r = m.group(1)
        add(f"Tell me about {r}")
        add(f"Describe {r}")
        add(f"Can you give me an example of {r}?")

    m = re.match(r"(?i)^(what|why|how|where|when|which|who)((?: [\w'-]+){0,6}?) do (you|they|we) (.+)$", b)
    if m:
        wh, mid, subj, rest = m.group(1).lower(), m.group(2), m.group(3).lower(), m.group(4)
        clause = f"{wh}{mid} {subj} {rest}"
        add(f"Tell me {clause}")
        add(f"Can you tell me {clause}?")
        add(f"I'd like to know {clause}")
        add(f"I'm curious {clause}")
        if wh == "how":
            add(f"How would {subj} {rest}?")
    m = re.match(r"(?i)^how would (you|they|we) (.+)$", b)
    if m:
        subj, rest = m.group(1).lower(), m.group(2)
        add(f"How do {subj} {rest}?")
        add(f"Tell me how {subj} would {rest}")
        add(f"Walk me through how {subj} would {rest}")
        if not rest.lower().startswith("explain"):
            add(f"Can you explain how {subj} would {rest}?")

    m = re.match(r"(?i)^(what|who|where|when|which) (is|are|was|were) (.+)$", b)
    if m and not re.match(r"(?i)^(it|there|that|this|he|she|they)\b", m.group(3)) and len(m.group(3).split()) >= 2:
        wh, cop, r = m.group(1).lower(), m.group(2).lower(), m.group(3)
        inverted = f"{wh} {r} {cop}"
        add(f"Can you tell me {inverted}?")
        add(f"Tell me {inverted}")
        add(f"I'd like to know {inverted}")
        add(f"I'm curious {inverted}")
        if cop == "is" and wh == "what":
            add(f"What's {r}?")

    m = re.match(r"(?i)^why did you (.+)$", b)
    if m:
        add(f"What made you {m.group(1)}?")
        add(f"What led you to {m.group(1)}?")
    m = re.match(r"(?i)^why do you (want to .+)$", b)
    if m:
        add(f"What makes you {m.group(1)}?")
    m = re.match(r"(?i)^why are you interested in (.+)$", b)
    if m:
        add(f"What interests you about {m.group(1)}?")
        add(f"What attracts you to {m.group(1)}?")
    m = re.match(r"(?i)^why are you applying (?:for|to) (.+)$", b)
    if m:
        add(f"What made you apply for {m.group(1)}?")

    m = re.match(r"(?i)^do you have (?:any )?(?:experience|exposure|background) (?:with|in|using) (.+)$", b)
    if m:
        r = m.group(1)
        add(f"Have you worked with {r}?")
        add(f"What experience do you have with {r}?")
        add(f"Tell me about your experience with {r}")
        add(f"How familiar are you with {r}?")
    m = re.match(r"(?i)^what experience do you have (?:with|in|using) (.+)$", b)
    if m:
        r = m.group(1)
        add(f"Do you have experience with {r}?")
        add(f"Tell me about your experience with {r}")
        add(f"Have you worked with {r}?")
        add(f"How familiar are you with {r}?")
    m = re.match(r"(?i)^have you (?:ever )?worked with (.+)$", b)
    if m:
        add(f"Do you have experience with {m.group(1)}?")
        add(f"What experience do you have with {m.group(1)}?")

    m = re.match(r"(?i)^what would you do if (.+)$", b)
    if m:
        r = m.group(1)
        add(f"How would you handle it if {r}?")
        add(f"How would you respond if {r}?")
        add(f"If {r}, what would you do?")
        add(f"Suppose {r}. What would you do?")
    return out + wrappers


def _synonyms(b: str) -> list[str]:
    out = []
    for word, alts in _SYNONYMS:
        for m in re.finditer(r"(?i)\b" + re.escape(word) + r"\b", b):
            for alt in alts:
                swapped = b[:m.start()] + (alt if b[m.start()].islower() else _cap(alt)) + b[m.end():]
                out.append(swapped)
            break
    return out


def _intents(b: str) -> list[str]:
    stems = {_stem(w) for w in re.findall(r"[a-z]+(?:-[a-z]+)?", b.lower())}
    out = []
    for groups, wordings, *rest in _INTENTS:
        unless = rest[0] if rest else []
        if all(any(_stem(w) in stems for w in g) for g in groups) and not any(_stem(w) in stems for w in unless):
            out += wordings
    return out


def generate(question: str, existing: Iterable[str] = (), limit: int = 8) -> list[Suggestion]:
    """Up to `limit` new wordings of `question`, mixed across sources, never repeating what's known."""
    base = re.sub(r"\s+", " ", question.strip().replace("’", "'")).rstrip("?.! ")
    if len(base.split()) < 2:
        return []
    seen = {_key(question)} | {_key(e) for e in existing}
    asked = question.strip().endswith("?")
    pools = {"intent": _intents(base), "pattern": _patterns(base), "synonym": [t + ("?" if asked else "") for t in _synonyms(base)]}
    queues: dict[str, list[str]] = {}
    for src, items in pools.items():
        q = []
        for text in items:
            text = _finish(text)
            k = _key(text)
            if k not in seen and len(text.split()) >= 2:
                seen.add(k)
                q.append(text)
        queues[src] = q
    out: list[Suggestion] = []
    order = ["intent", "pattern", "synonym"]
    while len(out) < limit and any(queues.values()):
        for src in order:
            if queues[src] and len(out) < limit:
                out.append(Suggestion(queues[src].pop(0), src))
    return out


def suggest_phrasings(question: str, existing: Iterable[str] = (), limit: int = 8) -> list[str]:
    return [s.text for s in generate(question, existing, limit)]


# ---------- checking against the real matcher ----------

def vet(suggestions: list[Suggestion], entry: Entry, matcher: Matcher, min_sim: float = 0.80) -> list[Suggestion]:
    """Keep wordings that still mean `entry`'s question and don't belong to a different one better.
    Sets .sim and .rival; wordings below `min_sim` are dropped."""
    if not suggestions:
        return []
    sims = matcher.similarity(entry.question, [s.text for s in suggestions])
    out = []
    for s, sim in zip(suggestions, sims):
        if sim < min_sim:
            continue
        s.sim = round(float(sim), 3)
        rival = matcher.rival(s.text, entry, margin=sim)
        if rival is not None:
            s.rival = rival.question
        out.append(s)
    return out
