"""Turning raw speech-model output into text the matcher can trust.

- clean_text: drop fillers, stutters and repetition loops the speech model sometimes emits
- is_hallucination: drop the "Thanks for watching" junk Whisper invents from near-silence
- looks_incomplete: spot an utterance cut off mid-sentence so the listener can wait for the rest
- Vocabulary: names/jargon from your banks, used to bias the speech model and to repair words it misheard
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from textsim import normalize, phonetic_key


@dataclass
class Word:
    text: str
    start: float = 0.0
    end: float = 0.0
    prob: float | None = None  # speech-model confidence 0..1; None when unknown


@dataclass
class Transcript:
    text: str                                   # cleaned and corrected: what the matcher should see
    raw: str = ""                               # exactly what the speech model produced
    words: list[Word] = field(default_factory=list)
    confidence: float | None = None             # mean word confidence, None when unknown
    duration: float = 0.0
    incomplete: bool = False                    # sounds cut off mid-sentence
    corrections: list[tuple[str, str]] = field(default_factory=list)  # (heard, replaced with)

    def weak_words(self, below: float = 0.5) -> list[Word]:
        return [w for w in self.words if w.prob is not None and w.prob < below]


# ---------- cleanup ----------

_FILLER = re.compile(r"(?<![\w'-])(?:um+|uh+|uhm+|erm+|er|hmm+|mm+|ah+)(?![\w'-])[,.]?\s*", re.I)
_STUTTER = re.compile(r"\b(\w+)(?:[\s,]+\1\b)+", re.I)
_REAL_DOUBLES = frozenset({"that", "had", "is", "was", "no", "very", "really", "bye", "so", "ha", "yeah"})  # can be deliberate
_LOOP = re.compile(r"\b((?:\w+\s+){0,3}\w+)(?:[\s,.]+\1\b){2,}", re.I)
_HALLUCINATIONS = (
    "thanks for watching", "thank you for watching", "please subscribe", "subscribe to the channel",
    "subtitles by", "transcription by", "like and subscribe", "see you in the next video",
)
_SHORT_NOISE = frozenset({"thank you", "thanks", "you", "bye", "bye bye", "okay", "oh", "so", "yeah", "hmm", "the end"})


def clean_text(text: str) -> str:
    t = _FILLER.sub("", text)
    t = _LOOP.sub(r"\1", t)
    t = _STUTTER.sub(lambda m: m.group() if m.group(1).lower() in _REAL_DOUBLES else m.group(1), t)
    t = re.sub(r"\s+([,.?!;:])", r"\1", t)
    t = re.sub(r"^[\s,.;:-]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def is_hallucination(text: str, duration: float = 0.0, confidence: float | None = None,
                     no_speech: float = 0.0, avg_logprob: float = 0.0) -> bool:
    """True for output that is almost certainly invented from silence or noise."""
    n = normalize(text)
    if not n:
        return True
    if any(h in n for h in _HALLUCINATIONS):
        return True
    if no_speech > 0.7 and avg_logprob < -0.9:
        return True
    return n in _SHORT_NOISE and duration < 2.5 and confidence is not None and confidence < 0.6


_OPEN_ENDINGS = frozenset("""
a an the and or but so if of to in on at for with about from by as that which who whom whose when where while because
than then also your my our their his her its this these those is are was were be been am do does did have has had can
could would should will might must not just like into onto over under between through during
""".split())


_OPEN_PRONOUNS = frozenset({"you", "we", "i", "they", "he", "she", "it"})


def looks_incomplete(text: str) -> bool:
    """True when the text stops mid-thought ("...a time when you", "...and,") rather than finishing a sentence."""
    t = text.strip()
    if not t:
        return False
    if t.endswith(("?", "!", ".")) and not t.endswith(("...", "…")):
        return False
    if t.endswith((",", "...", "…", "-", "—", ":", ";")):
        return True
    words = normalize(t).split()
    if not words:
        return False
    # a bare pronoun ends real sentences too ("Thank you"), so only trust it once the utterance has some length
    return words[-1] in _OPEN_ENDINGS or (len(words) >= 5 and words[-1] in _OPEN_PRONOUNS)


# ---------- vocabulary ----------

_HEADER_BOLD = re.compile(r"\*\*[^a-z*\n]{3,}\*\*")      # "**SCRIPT  ·  ABOUT 70 SECONDS SPOKEN**"
_NOT_TERMS = frozenset({"OK", "AM", "PM", "US", "ID", "TV", "AI", "IT", "OR", "AND", "THE", "TO", "SO", "NO", "YES",
                        "NOTE", "STAR", "ALL", "NEW", "NAME", "TBD", "TODO", "FIXME"})
_CALENDAR = frozenset("""monday tuesday wednesday thursday friday saturday sunday january february march april may june
july august september october november december""".split())
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&'’-]*")
_SENTENCE = re.compile(r"(?<=[.?!])[\"”’')\]]*\s+|\n+|\s·\s")
_PRONOUN_I = re.compile(r"I(?:['’](?:m|d|ve|ll))?")


def _strip_markup(text: str) -> str:
    t = _HEADER_BOLD.sub(" ", text)
    return re.sub(r"\*\*|\{\{|\}\}|_{2,}", " ", t)


def _shaped(w: str) -> bool:
    """CamelCase or an ACRONYM: looks like a name wherever it sits in a sentence."""
    return w.isalpha() and (bool(re.search(r"[a-z][A-Z]", w)) or (w.isupper() and 2 <= len(w) <= 8)) \
        and w.upper() not in _NOT_TERMS


def _allcaps(w: str) -> bool:
    return len(w) >= 2 and w.replace("’", "").replace("'", "").isupper()


def _proper_nouns(sentences: list[str], lower_seen: set[str]):
    """Names in `sentences`: runs of capitalised words, ignoring the capital that merely starts a sentence, and
    single capitalised words only when the text never writes them in lower case (so 'Yeah', 'Probably' don't count)."""
    from textsim import STOPWORDS
    for sent in sentences:
        run: list[str] = []
        prev_end = 0

        def flush():
            words = list(run)
            run.clear()
            while words and words[0].lower() in STOPWORDS:
                words.pop(0)
            while words and words[-1].lower() in STOPWORDS:
                words.pop()
            if sum(_allcaps(w) for w in words) >= 2:
                return                            # "SECONDS SPOKEN": a heading, not a name
            for w in words:
                if _shaped(w) and len(words) > 1:
                    yield w                       # an acronym inside a longer name is a term of its own
            if len(words) == 1:
                w = words[0]
                if _shaped(w) or (len(w) >= 4 and w.lower() not in lower_seen and w.lower() not in _CALENDAR):
                    yield w
            elif len(words) > 1:
                yield " ".join(words)

        for i, m in enumerate(_WORD.finditer(sent)):
            w = m.group()
            starts_sentence = i == 0
            is_name = w[0].isupper() and not _PRONOUN_I.fullmatch(w) and not (starts_sentence and not _shaped(w))
            if is_name and run and sent[prev_end:m.start()].strip() == "":
                run.append(w)
            else:
                yield from flush()
                run.clear()
                if is_name:
                    run.append(w)
            prev_end = m.end()
        yield from flush()


class Vocabulary:
    """Names and jargon that will be spoken in the meeting, taken from the active banks."""

    def __init__(self, terms: list[str] | None = None) -> None:
        self.terms: list[str] = list(dict.fromkeys(terms or []))
        # exact-match case restoration only for things that are clearly names, so "head" stays "head"
        self._exact = {t.lower(): t for t in self.terms if " " in t or _shaped(t) or len(t) >= 6}
        self._fuzzy = [t for t in self.terms if " " not in t and len(t) >= 5]
        self._phrases = [t for t in self.terms if " " in t and len(t.replace(" ", "")) >= 8]
        self._keys = {t: phonetic_key(t) for t in self._fuzzy + self._phrases}

    def __bool__(self) -> bool:
        return bool(self.terms)

    @classmethod
    def from_entries(cls, entries, limit: int = 60) -> "Vocabulary":
        """Rank names by how often they appear, counting questions (what the other side says) triple."""
        from followup import prepared_followups, script_text  # local: followup must not import this module
        sources: list[tuple[str, int]] = []
        for e in entries:
            sources += [(" ".join(e.phrasings()), 3), (" ".join(e.tags), 2), (script_text(e), 1)]
            for f in prepared_followups(e):
                sources += [(f.question, 3), (f.answer, 1)]
        split = [([s for s in _SENTENCE.split(_strip_markup(text)) if s.strip()], w) for text, w in sources]
        lower_seen = {w.lower() for sents, _ in split for s in sents for w in _WORD.findall(s) if w[0].islower()}
        score: Counter[str] = Counter()
        for sents, weight in split:
            for term in _proper_nouns(sents, lower_seen):
                score[term] += weight
        # a lone ALL-CAPS word must recur (or be something they ask about) to be an acronym rather than a heading
        return cls([t for t, n in score.most_common() if n >= 2 or not t.isupper()][:limit])

    def prompt(self, base: str = "", max_chars: int = 400) -> str:
        """A short priming sentence for the speech model: your own prompt plus the likeliest names."""
        listed, size = [], 0
        for t in self.terms:
            if t.lower() in base.lower() or any(t.lower() in x.lower() for x in listed):
                continue
            if size + len(t) + 2 > max_chars:
                break
            listed.append(t)
            size += len(t) + 2
        parts = [base.strip().rstrip(".")] if base.strip() else []
        if listed:
            parts.append("Names and terms that may come up: " + ", ".join(listed))
        return ". ".join(parts) + ("." if parts else "")

    # ----- repair -----
    def correct(self, words: list[Word]) -> tuple[list[Word], list[tuple[str, str]]]:
        """Repair words the speech model likely misheard, using the vocabulary.

        Case-only fixes always apply. Sound-alike/spelling fixes apply only to words the model was unsure
        about (or all words when confidence is unknown), so real words it heard clearly are left alone."""
        if not self.terms:
            return words, []
        out: list[Word] = []
        changes: list[tuple[str, str]] = []
        i = 0
        while i < len(words):
            w = words[i]
            lead, core, trail = _split_punct(w.text)
            if not core:
                out.append(w)
                i += 1
                continue
            # multi-word exact phrase ("penn state") and split words ("alpha sights" -> AlphaSights)
            hit = self._match_span(words, i)
            if hit:
                term, n = hit
                last = words[i + n - 1]
                _, _, tail = _split_punct(last.text)
                probs = [x.prob for x in words[i:i + n] if x.prob is not None]
                new = Word(lead + term + tail, w.start, last.end, min(probs) if probs else None)
                if n > 1 or core != term:
                    changes.append((" ".join(x.text for x in words[i:i + n]), new.text))
                out.append(new)
                i += n
                continue
            fixed = self._fix_one(core, w.prob)
            if fixed and fixed != core:
                changes.append((w.text, lead + fixed + trail))
                out.append(Word(lead + fixed + trail, w.start, w.end, w.prob))
            else:
                out.append(w)
            i += 1
        return out, changes

    def correct_text(self, text: str) -> str:
        words, _ = self.correct([Word(t) for t in text.split()])
        return " ".join(w.text for w in words)

    def _match_span(self, words: list[Word], i: int) -> tuple[str, int] | None:
        """A term starting at words[i]: an exact (case-insensitive) name of 1-3 words, or a name the speech model
        split in two or three ("alpha sights", "guide point") that is unsure about those words."""
        spans = []
        for n in (3, 2, 1):
            if i + n > len(words):
                continue
            chunk = words[i:i + n]
            if any(_split_punct(x.text)[2] for x in chunk[:-1]):
                continue  # only the last word of a span may carry trailing punctuation
            cores = [_split_punct(x.text)[1] for x in chunk]
            if all(cores):
                spans.append((n, chunk, cores))
        for n, _, cores in spans:
            joined = " ".join(cores).lower()
            if joined in self._exact:
                return self._exact[joined], n
            squashed = "".join(cores).lower()
            if n > 1 and squashed in self._exact and " " not in self._exact[squashed]:
                return self._exact[squashed], n
        for n, chunk, cores in sorted((x for x in spans if x[0] > 1), key=lambda x: x[0]):
            squashed = "".join(cores).lower()
            if not all(x.prob is None or x.prob < 0.75 for x in chunk):
                continue
            pool = self._fuzzy + [t for t in self._phrases if len(t.split()) == n]   # a name of n words for n words
            for t in pool:
                flat = t.replace(" ", "")
                if abs(len(squashed) - len(flat)) <= 2 and self._near(squashed, t):  # no room for a stray extra word
                    return t, n
        return None

    def _fix_one(self, core: str, prob: float | None) -> str | None:
        low = core.lower()
        if low in self._exact:
            return self._exact[low]
        if len(core) < 4 or (prob is not None and prob >= 0.75):
            return None
        best, best_sim = None, 0.0
        for t in self._fuzzy:
            if self._near(low, t):
                sim = SequenceMatcher(None, low, t.lower()).ratio()
                if sim > best_sim:
                    best, best_sim = t, sim
        return best

    def _near(self, low: str, term: str) -> bool:
        sim = SequenceMatcher(None, low, term.lower().replace(" ", "")).ratio()
        return sim >= 0.86 or (sim >= 0.6 and phonetic_key(low) == self._keys.get(term, phonetic_key(term)))


def _split_punct(token: str) -> tuple[str, str, str]:
    m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", token)
    return (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")


def align_words(text: str, words: list[Word]) -> list[Word]:
    """Give each token of `text` the timing/confidence of the speech-model word it came from.

    `text` may have had fillers or repeats removed, so align by content rather than position."""
    tokens = text.split()
    src = [normalize(w.text) for w in words]
    dst = [normalize(t) for t in tokens]
    out = [Word(t) for t in tokens]
    for blk in SequenceMatcher(None, src, dst, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            w = words[blk.a + k]
            out[blk.b + k] = Word(tokens[blk.b + k], w.start, w.end, w.prob)
    return out


def mean_confidence(words: list[Word]) -> float | None:
    probs = [w.prob for w in words if w.prob is not None]
    return sum(probs) / len(probs) if probs else None
