"""Matching of heard speech against bank questions.

Local ONNX sentence embeddings do the heavy lifting.  Two things around them make speech
transcripts match better:

* the transcript is tidied first (fillers, "Great, thanks. So..." lead-ins) so the question
  itself is what gets compared, and
* a small, bounded lexical bonus (IDF-weighted word overlap) separates look-alike questions
  such as "...a time you failed" / "...a time you succeeded" that embeddings score almost equally.
  The bonus only ever adds to the semantic score, so it can't make a good semantic match worse.

Embeddings are cached, so reloading banks and re-scoring repeated sentences is cheap.
"""
from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from bank import ROOT, Entry

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODELS_DIR = ROOT / "models"

LEX_BONUS = 0.08       # most the lexical signal can add to a semantic score
LEX_FLOOR = 0.60       # word-overlap below this adds nothing
MAX_CANDIDATES = 5     # distinct pieces of one utterance that get embedded
QUERY_CACHE = 512

QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "which", "who", "whose", "whom", "tell me", "walk me", "talk me",
    "describe", "give me", "can you", "could you", "would you", "do you", "did you", "have you", "had you",
    "are you", "were you", "will you", "is there", "are there", "was there", "were there", "is it", "was it",
    "is that", "is this", "does", "did", "has", "share", "explain", "talk about", "talk to me",
    "i'd love to hear", "i would love to hear", "i'd like to hear", "i would like to hear", "i want to hear",
    "i'd like to know", "i would like to know", "i was wondering", "i'm curious", "i am curious",
    "let's talk about", "let us talk about", "take me through", "help me understand", "what's", "how'd",
    "how's", "where's", "who's", "what about", "how about", "why don't", "don't you", "haven't you",
    "if you", "imagine", "suppose", "let me ask", "one more", "next question", "last question",
    "final question", "anything", "any other", "any examples", "so tell me", "so what", "so how", "so why",
    "and what", "and how", "and why", "now what", "now how", "now tell me",
)
_QSTART = re.compile(r"^(?:" + "|".join(re.escape(s) for s in QUESTION_STARTS) + r")\b", re.I)

# "Great, thanks. So, now..." -- throat-clearing that says nothing about which question is coming.
_LEAD = re.compile(
    r"^(?:(?:okay|ok|alright|all right|right|so|well|now|and|but|then|great|good|perfect|awesome|excellent|"
    r"cool|nice|sure|thanks|thank you|thank you very much|got it|i see|gotcha|interesting|wonderful|"
    r"fantastic|that's great|that's good|that makes sense|makes sense|yeah|yes|yep|mhm|uh-huh|mm-hmm)"
    r"(?=[\s,.!?;:]|$)[\s,.!;:]*)+", re.I)
_FILLERS = re.compile(r"\b(?:u+m+|u+h+|uhm+|er+m*|ah+|hmm+|mm+)\b[,.]?\s*|\b(?:you know|i mean),\s*", re.I)
_REPEAT = re.compile(r"\b(\w+(?:\s\w+)?)(?:\s+\1\b)+", re.I)   # "what what is" / "how do how do you"

_CONTRACTIONS = {
    "what's": "what is", "who's": "who is", "how's": "how is", "where's": "where is", "that's": "that is",
    "there's": "there is", "it's": "it is", "let's": "let us", "you're": "you are", "they're": "they are",
    "we're": "we are", "i'm": "i am", "you've": "you have", "i've": "i have", "we've": "we have",
    "they've": "they have", "you'd": "you would", "i'd": "i would", "we'd": "we would", "you'll": "you will",
    "i'll": "i will", "we'll": "we will", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "how'd": "how did", "what'd": "what did",
}
_STOP = frozenset(
    "a an the of to and or in on at for with is are was were be been it its that this as by from do does did "
    "you your me my i can could would will about please".split())


@dataclass
class Match:
    entry: Entry
    score: float
    boost: float = 0.0   # extra weight given by context (follow-ups); ranking uses score + boost


@dataclass
class Scores:
    """Every entry's score for one utterance, before ranking."""
    entries: list[Entry]
    total: np.ndarray    # semantic + lexical bonus, per entry
    emb: np.ndarray      # semantic part alone
    lex: np.ndarray      # best word-overlap (0..1)
    pieces: list[str]    # what was actually compared


# ---------- text helpers ----------

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 2]


def strip_lead(text: str) -> str:
    """Drop leading throat-clearing ("Okay, so...") but never the whole sentence."""
    out = _LEAD.sub("", text.strip(), count=1).strip()
    return out if out else text.strip()


def _drop_lead(text: str) -> str:
    return _LEAD.sub("", text.strip(), count=1).strip()   # may be empty: "Great, thanks." says nothing


def tidy(text: str) -> str:
    """Speech transcript -> cleaner text: curly quotes, fillers, stutters, per-sentence lead-ins."""
    t = text.replace("’", "'").replace("‘", "'")
    t = _FILLERS.sub("", t)
    t = _REPEAT.sub(r"\1", t)
    parts = re.split(r"(?<=[.?!])\s+", t.strip())
    return " ".join(p for p in (_drop_lead(p) for p in parts) if p).strip()


def looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    sents = split_sentences(t) or [t]
    return any(_QSTART.match(strip_lead(s)) for s in sents)


def bank_vocabulary(entries: list[Entry], limit: int = 12) -> list[str]:
    """Names, acronyms and jargon in the questions (not the answers). Speech recognition is told to
    expect these, so "Kubernetes" doesn't come out as "cuber nettie's"."""
    seen: dict[str, list] = {}
    for e in entries:
        for phrase in e.phrasings():
            for i, w in enumerate(re.findall(r"[A-Za-z][A-Za-z0-9+#.]*(?:-[A-Za-z0-9]+)*", phrase)):
                w = w.rstrip(".")
                camel = bool(re.search(r"[a-z][A-Z]", w))
                special = (w.isupper() and len(w) >= 2) or camel or any(c.isdigit() or c in "+#" for c in w[1:])
                proper = i > 0 and w[0].isupper() and len(w) > 2 and w.lower() not in _STOP and w != "I"
                if special or proper:
                    seen.setdefault(w.lower(), [w, 0])[1] += 1
    ranked = sorted(seen.values(), key=lambda x: (-x[1], x[0].lower()))
    return [w for w, _ in ranked[:limit]]


def content_tokens(text: str) -> list[str]:
    """Lower-cased, contraction-expanded, lightly stemmed words without filler words."""
    out: list[str] = []
    for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'")):
        for x in _CONTRACTIONS.get(w, w).split():
            if x not in _STOP:
                out.append(_stem(x))
    return out


def _stem(w: str) -> str:
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("es") and not w.endswith("ses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _bonus(lex: np.ndarray) -> np.ndarray:
    return LEX_BONUS * np.clip((lex - LEX_FLOOR) / (1.0 - LEX_FLOOR), 0.0, 1.0) ** 2


# ---------- index ----------

@dataclass(frozen=True)
class _Index:
    entries: list[Entry]
    owner: np.ndarray                         # phrasing -> index into entries
    vecs: np.ndarray                          # (phrasings, dim), unit length
    idf: dict[str, float]
    idf_oov: float
    postings: dict[str, tuple[np.ndarray, float]]   # word -> (phrasings containing it, idf^2)
    pnorm: np.ndarray                         # lexical vector length per phrasing
    pos: dict[int, int]                       # id(entry) -> index into entries


_EMPTY = _Index([], np.zeros(0, dtype=int), np.zeros((0, 1), dtype=np.float32), {}, 1.0, {},
                np.zeros(0, dtype=np.float32), {})


def _load_fastembed() -> Callable[[list[str]], np.ndarray]:
    from fastembed import TextEmbedding
    MODELS_DIR.mkdir(exist_ok=True)
    try:   # cached copy first: starts instantly and works with no internet
        model = TextEmbedding(MODEL_NAME, cache_dir=str(MODELS_DIR), local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet (or an older fastembed): fetch it
        model = TextEmbedding(MODEL_NAME, cache_dir=str(MODELS_DIR))
    return lambda texts: np.array(list(model.embed(texts)), dtype=np.float32)


class Matcher:
    def __init__(self, embed_fn: Callable[[list[str]], np.ndarray] | None = None) -> None:
        """`embed_fn` (list of texts -> array of vectors) replaces the ONNX model; used by tests."""
        self._raw_embed = embed_fn or _load_fastembed()
        self._lock = threading.Lock()                       # one load at a time
        self._cache_lock = threading.Lock()
        self._phrase_cache: dict[str, np.ndarray] = {}      # bank phrasings: survive reloads
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._index: _Index = _EMPTY                        # swapped whole, so match() never sees a half-load
        self._raw_embed(["warm up"])                        # first call is slow; pay for it now

    @property
    def entries(self) -> list[Entry]:
        return self._index.entries

    def index_of(self, entry: Entry) -> int | None:
        return self._index.pos.get(id(entry))

    def _embed(self, texts: list[str], persistent: bool = False) -> np.ndarray:
        """Unit vectors for `texts`. Bank phrasings (`persistent`) stay cached across reloads;
        queries live in a small LRU. Results are gathered locally so a concurrent eviction can't bite."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        got: dict[str, np.ndarray] = {}
        with self._cache_lock:
            for t in texts:
                v = self._phrase_cache.get(t)
                if v is None:
                    v = self._query_cache.get(t)
                if v is not None:
                    got[t] = v
        missing = [t for t in dict.fromkeys(texts) if t not in got]
        if missing:
            v = np.asarray(self._raw_embed(missing), dtype=np.float32)
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            with self._cache_lock:
                store = self._phrase_cache if persistent else self._query_cache
                for t, row in zip(missing, v):
                    store[t] = got[t] = row
                while len(self._query_cache) > QUERY_CACHE:
                    self._query_cache.popitem(last=False)
        return np.stack([got[t] for t in texts])

    def load(self, entries: list[Entry]) -> None:
        with self._lock:
            texts: list[str] = []
            owner: list[int] = []
            for i, e in enumerate(entries):
                for p in e.phrasings():
                    p = p.strip()
                    if p:
                        texts.append(p)
                        owner.append(i)
            vecs = self._embed(texts, persistent=True) if texts else np.zeros((0, 1), dtype=np.float32)

            toks = [set(content_tokens(t)) for t in texts]
            df: dict[str, int] = {}
            for s in toks:
                for w in s:
                    df[w] = df.get(w, 0) + 1
            n = max(1, len(texts))
            idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
            members: dict[str, list[int]] = {}
            for j, s in enumerate(toks):
                for w in s:
                    members.setdefault(w, []).append(j)
            postings = {w: (np.array(ix, dtype=int), idf[w] ** 2) for w, ix in members.items()}
            pnorm = np.array([math.sqrt(sum(idf[w] ** 2 for w in s)) for s in toks], dtype=np.float32)
            self._index = _Index(entries, np.array(owner, dtype=int), vecs, idf, math.log(n + 1) + 1.0,
                                 postings, pnorm, {id(e): i for i, e in enumerate(entries)})

    # ---------- scoring ----------
    def _pieces(self, text: str) -> list[str]:
        clean = tidy(text)
        sents = split_sentences(clean) or ([clean] if clean else [])
        if not sents:
            return []
        pieces = [clean, sents[-1], " ".join(sents[-2:])]
        pieces += [s for s in sents if looks_like_question(s)][-3:]
        out = [p for p in dict.fromkeys(pieces) if len(p.split()) >= 2 or len(sents) == 1]
        return out[:MAX_CANDIDATES]

    def _lexical(self, ix: _Index, piece: str) -> np.ndarray:
        words = set(content_tokens(piece))
        acc = np.zeros(len(ix.pnorm), dtype=np.float32)
        if not words:
            return acc
        qnorm_sq = 0.0
        for w in words:
            qnorm_sq += ix.idf.get(w, ix.idf_oov) ** 2
            hit = ix.postings.get(w)
            if hit is not None:
                acc[hit[0]] += hit[1]
        denom = np.sqrt(qnorm_sq) * ix.pnorm
        return np.where(denom > 0, acc / np.maximum(denom, 1e-9), 0.0).astype(np.float32)

    def score(self, text: str) -> Scores | None:
        """Score every entry against what was said (best phrasing, best piece of the utterance)."""
        ix = self._index
        if not ix.entries or not len(ix.owner) or not text.strip():
            return None
        pieces = self._pieces(text)
        if not pieces:
            return None
        q = self._embed(pieces)
        sims = q @ ix.vecs.T                                   # (pieces, phrasings)
        lex = np.stack([self._lexical(ix, p) for p in pieces])
        total_ph = (sims + _bonus(lex)).max(axis=0)
        emb_ph, lex_ph = sims.max(axis=0), lex.max(axis=0)
        n = len(ix.entries)
        total = np.full(n, -1.0, dtype=np.float32)
        emb = np.full(n, -1.0, dtype=np.float32)
        lx = np.zeros(n, dtype=np.float32)
        np.maximum.at(total, ix.owner, total_ph)
        np.maximum.at(emb, ix.owner, emb_ph)
        np.maximum.at(lx, ix.owner, lex_ph)
        return Scores(ix.entries, np.minimum(total, 1.0), emb, lx, pieces)

    def rank(self, sc: Scores, k: int = 3, boosts: Mapping[int, float] | None = None) -> list[Match]:
        """Top `k` of already-computed scores. `boosts` maps entry index -> extra ranking weight
        (context such as "this follows the answer on screen"); `Match.score` stays the raw score."""
        extra = np.zeros_like(sc.total)
        for i, b in (boosts or {}).items():
            if 0 <= i < len(extra):
                extra[i] = b
        order = np.argsort(-(sc.total + extra), kind="stable")[:k]
        return [Match(sc.entries[i], float(sc.total[i]), float(extra[i])) for i in order]

    def match(self, text: str, k: int = 3, boosts: Mapping[int, float] | None = None) -> list[Match]:
        """Best `k` entries for the utterance."""
        sc = self.score(text)
        return self.rank(sc, k, boosts) if sc is not None else []

    # ---------- helpers for suggesting phrasings ----------
    def similarity(self, a: str, others: list[str]) -> list[float]:
        """Cosine similarity of `a` to each of `others` (meaning only, no word-overlap bonus)."""
        if not others:
            return []
        v = self._embed([a, *others])
        return (v[1:] @ v[0]).tolist()

    def rival(self, text: str, entry: Entry, margin: float = 0.0) -> Entry | None:
        """A *different* loaded question that `text` matches about as well as (or better than) `entry`'s
        own question, i.e. a wording that would put the wrong answer on screen. `margin` is how well the
        text is known to match `entry` when `entry` itself isn't loaded."""
        sc = self.score(text)
        if sc is None:
            return None
        own_key = (entry.bank, entry.question.strip().lower())
        own, best, best_score = margin, None, -1.0
        for e, t in zip(sc.entries, sc.total):
            if (e.bank, e.question.strip().lower()) == own_key:
                own = max(own, float(t))
            elif t > best_score:
                best, best_score = e, float(t)
        return best if best is not None and best_score > own - 0.02 else None
