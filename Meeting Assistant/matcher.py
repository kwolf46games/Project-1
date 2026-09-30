"""Semantic matching of heard speech against bank questions (local ONNX embeddings)."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np

from bank import ROOT, Entry

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODELS_DIR = ROOT / "models"

QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "which", "who", "tell me", "walk me", "talk me",
    "describe", "give me", "can you", "could you", "would you", "do you", "did you", "have you",
    "are you", "is there", "share", "explain", "talk about", "i'd love to hear", "i would love to hear",
    "let's talk about", "take me through", "help me understand", "what's", "how'd",
)


@dataclass
class Match:
    entry: Entry
    score: float


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 2]


def looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    return any(s.startswith(QUESTION_STARTS) for s in (x.lower() for x in split_sentences(t) or [t]))


class Matcher:
    def __init__(self) -> None:
        from fastembed import TextEmbedding
        MODELS_DIR.mkdir(exist_ok=True)
        self.model = TextEmbedding(MODEL_NAME, cache_dir=str(MODELS_DIR))
        self._lock = threading.Lock()  # one load at a time
        # (entries, owner, vecs) swapped in as one tuple so match() never sees a half-loaded state
        self._index: tuple[list[Entry], np.ndarray, np.ndarray] = ([], np.zeros(0, dtype=int), np.zeros((0, 1)))

    @property
    def entries(self) -> list[Entry]:
        return self._index[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        v = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def load(self, entries: list[Entry]) -> None:
        with self._lock:
            texts, owner = [], []
            for i, e in enumerate(entries):
                for p in e.phrasings():
                    texts.append(p)
                    owner.append(i)
            vecs = self._embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
            self._index = (entries, np.array(owner, dtype=int), vecs)

    def match(self, text: str, k: int = 3) -> list[Match]:
        """Score each entry by its best phrasing against the utterance and its trailing
        sentences (so a preamble like 'Great, thanks. So next...' doesn't dilute it)."""
        entries, owner, vecs = self._index
        if not entries or not text.strip():
            return []
        sents = split_sentences(text) or [text]
        candidates = {text, sents[-1], " ".join(sents[-2:])}
        candidates |= {s for s in sents if looks_like_question(s)}
        q = self._embed(sorted(candidates))
        sims = (q @ vecs.T).max(axis=0)  # best candidate per phrasing
        best = np.full(len(entries), -1.0, dtype=np.float32)
        np.maximum.at(best, owner, sims)
        order = np.argsort(-best)[:k]
        return [Match(entries[i], float(best[i])) for i in order]
