"""Semantic matching of heard speech against bank questions (local ONNX embeddings).

Dense embeddings find the meaning; a small lexical bonus rescues matches whose names or key words the
speech model garbled. Prepared follow-ups written under each answer are indexed alongside the questions.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bank import ROOT, Entry
from followup import PreparedFollowup, prepared_followups
from textsim import Fingerprint, fingerprint, similarity
from transcript import clean_text

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODELS_DIR = ROOT / "models"

QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "which", "who", "tell me", "tell us", "walk me", "walk us", "talk me",
    "describe", "give me", "can you", "could you", "would you", "do you", "did you", "have you",
    "are you", "is there", "share", "explain", "talk about", "i'd love to hear", "i would love to hear",
    "let's talk about", "take me through", "help me understand", "what's", "how'd", "i'm curious",
    "i was wondering", "can i ask", "may i ask", "would you mind", "anything else", "so what", "and what",
)
_STARTS_RE = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in sorted(QUESTION_STARTS, key=len, reverse=True))
                        + r")\b", re.I)
_LEAD_IN = re.compile(r"^(?:(?:so|okay|ok|alright|all right|great|thanks|thank you|awesome|perfect|cool|yeah|right|"
                      r"well|now|and|next)\b[,.!]?\s+)+", re.I)

LEX_BONUS = 0.06   # most a perfect lexical match can add to a dense score
LEX_FLOOR = 0.70   # lexical similarity below this adds nothing
LEX_TOP = 8        # only the best few phrasings by dense score get the (slower) lexical check


@dataclass
class Match:
    entry: Entry
    score: float
    phrasing: str = ""      # which wording of the entry matched best
    dense: float = 0.0      # embedding similarity alone
    lexical: float = 0.0    # word-level similarity (0 when it wasn't checked)


@dataclass
class FollowupMatch:
    entry: Entry            # the answer this follow-up was written under
    followup: PreparedFollowup
    score: float
    dense: float = 0.0
    lexical: float = 0.0


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 2]


def looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    return any(_STARTS_RE.match(_LEAD_IN.sub("", s)) for s in (split_sentences(t) or [t]))


def question_tails(text: str, limit: int = 3) -> list[str]:
    """Where a question starts partway through speech ("Great, thanks. Now tell me about..."), the text from there on."""
    starts = [m.start() for m in _STARTS_RE.finditer(text) if m.start() > 0]
    tails = [text[s:] for s in starts[-limit:]]
    stripped = _LEAD_IN.sub("", text)
    if stripped != text:
        tails.append(stripped)
    return tails


def candidates(text: str) -> list[str]:
    """Different cuts of an utterance to match: the whole, its trailing sentences, each question, and tails."""
    sents = split_sentences(text) or [text]
    cands = {text, sents[-1], " ".join(sents[-2:])}
    cands |= {s for s in sents if looks_like_question(s)}
    cands |= set(question_tails(text))
    return sorted(c for c in (x.strip() for x in cands) if c)


@dataclass(frozen=True)
class _Index:
    entries: list[Entry]
    owner: np.ndarray                       # phrasing -> entry number
    vecs: np.ndarray
    phrasings: list[str]
    prints: list[Fingerprint]
    fu_owner: np.ndarray                    # prepared follow-up -> entry number
    fu_items: list[PreparedFollowup]
    fu_vecs: np.ndarray
    fu_prints: list[Fingerprint]


_EMPTY = _Index([], np.zeros(0, dtype=int), np.zeros((0, 1), dtype=np.float32), [], [],
                np.zeros(0, dtype=int), [], np.zeros((0, 1), dtype=np.float32), [])


class Matcher:
    def __init__(self, embed: Callable[[list[str]], np.ndarray] | None = None) -> None:
        if embed is None:
            from fastembed import TextEmbedding
            MODELS_DIR.mkdir(exist_ok=True)
            model = TextEmbedding(MODEL_NAME, cache_dir=str(MODELS_DIR))
            embed = lambda texts: np.array(list(model.embed(texts)))  # noqa: E731
        self._embed_fn = embed
        self._lock = threading.Lock()  # one load at a time
        self._cache: OrderedDict[str, tuple[list[str], np.ndarray]] = OrderedDict()
        self._cache_lock = threading.Lock()
        # swapped in as one object so match() never sees a half-loaded state
        self._index: _Index = _EMPTY

    @property
    def entries(self) -> list[Entry]:
        return self._index.entries

    def _embed(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self._embed_fn(texts), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Unit-length embeddings, for callers (suggestions, focus passages) that need the same model."""
        return self._embed(texts)

    def load(self, entries: list[Entry]) -> None:
        with self._lock:
            texts, owner = [], []
            for i, e in enumerate(entries):
                for p in e.phrasings():
                    texts.append(p)
                    owner.append(i)
            fu_items, fu_owner = [], []
            for i, e in enumerate(entries):
                for f in prepared_followups(e):
                    fu_items.append(f)
                    fu_owner.append(i)
            everything = texts + [f.question for f in fu_items]
            vecs = self._embed(everything) if everything else np.zeros((0, 1), dtype=np.float32)
            n = len(texts)
            self._index = _Index(
                entries, np.array(owner, dtype=int), vecs[:n], texts, [fingerprint(t) for t in texts],
                np.array(fu_owner, dtype=int), fu_items, vecs[n:], [fingerprint(f.question) for f in fu_items])
            with self._cache_lock:
                self._cache.clear()

    def _query(self, text: str) -> tuple[list[str], np.ndarray]:
        with self._cache_lock:
            hit = self._cache.get(text)
            if hit:
                self._cache.move_to_end(text)
                return hit
        cands = candidates(text)
        out = (cands, self._embed(cands))
        with self._cache_lock:
            self._cache[text] = out
            while len(self._cache) > 16:
                self._cache.popitem(last=False)
        return out

    def _score(self, text: str, vecs: np.ndarray, prints: list[Fingerprint]):
        """Per stored phrase: (combined score, dense score, lexical score) against the best cut of `text`."""
        cands, q = self._query(text)
        dense = (q @ vecs.T).max(axis=0)
        lex = np.zeros_like(dense)
        cprints = [fingerprint(c) for c in cands]
        for j in np.argsort(-dense)[:LEX_TOP]:
            lex[j] = max(similarity(cp, prints[j]) for cp in cprints)
        bonus = LEX_BONUS * np.clip(lex - LEX_FLOOR, 0, None) / (1 - LEX_FLOOR)
        return np.minimum(1.0, dense + bonus), dense, lex

    def match(self, text: str, k: int = 3) -> list[Match]:
        """Score each entry by its best phrasing against the utterance and its trailing sentences
        (so a preamble like 'Great, thanks. So next...' doesn't dilute it)."""
        idx = self._index
        text = clean_text(text)
        if not idx.entries or not text:
            return []
        score, dense, lex = self._score(text, idx.vecs, idx.prints)
        out, seen = [], set()
        for j in np.argsort(-score):
            i = int(idx.owner[j])
            if i in seen:
                continue
            seen.add(i)
            out.append(Match(idx.entries[i], float(score[j]), idx.phrasings[j], float(dense[j]), float(lex[j])))
            if len(out) == k:
                break
        return out

    def match_followups(self, text: str, k: int = 3) -> list[FollowupMatch]:
        """Score the prepared follow-ups (under every active answer) against what was heard."""
        idx = self._index
        text = clean_text(text)
        if not idx.fu_items or not text:
            return []
        score, dense, lex = self._score(text, idx.fu_vecs, idx.fu_prints)
        return [FollowupMatch(idx.entries[int(idx.fu_owner[j])], idx.fu_items[j], float(score[j]),
                              float(dense[j]), float(lex[j])) for j in np.argsort(-score)[:k]]
