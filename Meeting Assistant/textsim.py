"""Small dependency-free text similarity helpers.

Speech-to-text mangles names and jargon ("alpha sights", "Tammid"), which embeddings alone can miss.
These give matching a cheap second opinion that survives spacing, spelling and word-form noise.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import sqrt

# Function words and interview boilerplate. The wh-words and negations are deliberately kept:
# "why" vs "what", and "hire you" vs "not hire you", are exactly what must not be blurred.
STOPWORDS = frozenset("""
a an the and or but if so to of in on at for with about from by as is are was were be been being am do does did
have has had can could would should will may might must you your yours we our us they their them he she it its i me
my mine this that these those there here tell give walk talk describe share explain please just really kind sort
lot bit little
""".split())
NEGATIONS = frozenset({"not", "no", "never", "without"})

_CONTRACTIONS = (("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'d", " would"),
                 ("'m", " am"), ("'s", ""))


def normalize(text: str) -> str:
    """Lowercase, strip accents, expand contractions, reduce to letters/digits separated by single spaces."""
    t = text.replace("’", "'").replace("‘", "'")  # before the ascii pass, which would otherwise drop them
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii").lower()
    for old, new in _CONTRACTIONS:
        t = t.replace(old, new)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def content_tokens(text: str) -> list[str]:
    return [_stem(w) for w in normalize(text).split() if w not in STOPWORDS]


def _trigrams(s: str) -> Counter:
    s = f"  {s} "
    return Counter(s[i:i + 3] for i in range(len(s) - 2))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (sqrt(sum(v * v for v in a.values())) * sqrt(sum(v * v for v in b.values())))


@dataclass(frozen=True)
class Fingerprint:
    """Precomputed form of a phrase so repeated comparisons are cheap."""
    tokens: tuple[str, ...]
    grams: Counter
    negated: bool


def fingerprint(text: str) -> Fingerprint:
    toks = content_tokens(text)
    neg = any(w in NEGATIONS for w in normalize(text).split())
    return Fingerprint(tuple(toks), _trigrams("".join(toks)), neg)


def _best(tok: str, pool: tuple[str, ...]) -> float:
    best = 0.0
    for other in pool:
        if tok == other:
            return 1.0
        r = SequenceMatcher(None, tok, other).ratio()
        if r > best:
            best = r
    return best if best >= 0.8 else 0.0


def similarity(a: Fingerprint, b: Fingerprint) -> float:
    """0..1 lexical similarity of two phrases, tolerant of spelling noise and word splits/joins.

    Half is fuzzy content-word overlap, half is character-trigram cosine over the squashed content
    words (so "alpha sights" ~ "AlphaSights"). A negation mismatch caps the result."""
    if not a.tokens or not b.tokens:
        return 0.0
    p = sum(_best(t, b.tokens) for t in a.tokens) / len(a.tokens)
    r = sum(_best(t, a.tokens) for t in b.tokens) / len(b.tokens)
    f = 2 * p * r / (p + r) if p + r else 0.0
    sim = 0.5 * f + 0.5 * _cosine(a.grams, b.grams)
    whole = SequenceMatcher(None, "".join(a.tokens), "".join(b.tokens)).ratio()  # catches split/joined words
    if whole >= 0.75:
        sim = max(sim, 0.95 * whole)
    return sim * 0.6 if a.negated != b.negated else sim


def lexical_similarity(a: str, b: str) -> float:
    return similarity(fingerprint(a), fingerprint(b))


# ---------- sound-alike keys ----------

_PHONETIC_RULES = (("ph", "f"), ("ck", "k"), ("gh", ""), ("wh", "w"), ("sch", "sk"), ("tch", "ch"), ("dg", "j"),
                   ("kn", "n"), ("wr", "r"), ("ps", "s"), ("x", "ks"), ("q", "k"), ("z", "s"), ("v", "f"))


def phonetic_key(word: str) -> str:
    """Crude sound-alike key: equal keys mean 'probably heard the same' (kubernetes ~ cubernetes)."""
    w = re.sub(r"[^a-z]", "", normalize(word))
    for old, new in _PHONETIC_RULES:
        w = w.replace(old, new)
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    head, tail = w[:1], re.sub(r"[aeiouyhw]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", head + tail)
