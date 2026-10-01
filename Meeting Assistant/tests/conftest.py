"""Shared test setup: import path, a stand-in theme module, a fake embedder, a temp banks folder."""
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

if "theme" not in sys.modules:   # the real theme.py lives with the app; tests only need the names
    try:
        import theme  # noqa: F401
    except ImportError:
        t = types.ModuleType("theme")
        for name, val in dict(ACCENT="#8ab4f8", BAD="#f28b82", BG="#1e1e1e", FG="#e8eaed", GOOD="#81c995",
                              HOVER_BG="#3c4043", MUTED="#9aa0a6", PANEL="#292a2d", PLACEHOLDER_BG="#5f4b00",
                              SELECT_BG="#3b4b6b", WARN="#fdd663").items():
            setattr(t, name, val)
        t.enable_dpi_awareness = lambda: None
        sys.modules["theme"] = t

from matcher import content_tokens  # noqa: E402


def fake_embed(texts):
    """Bag-of-stems hashed into 384 dims: similar wording -> similar vectors. Plumbing tests only;
    it says nothing about how the real sentence model scores."""
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for r, t in enumerate(texts):
        toks = content_tokens(t)
        for w in toks:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            out[r, h % 384] += 1.0
        for a, b in zip(toks, toks[1:]):
            h = int(hashlib.md5((a + " " + b).encode()).hexdigest(), 16)
            out[r, h % 384] += 0.5
        if not toks:
            out[r, 0] = 1.0
    return out


@pytest.fixture
def embed_calls():
    return []


@pytest.fixture
def matcher(embed_calls):
    from matcher import Matcher

    def counting(texts):
        embed_calls.append(list(texts))
        return fake_embed(texts)
    return Matcher(embed_fn=counting)


@pytest.fixture
def banks_dir(tmp_path, monkeypatch):
    import bank
    monkeypatch.setattr(bank, "BANKS_DIR", tmp_path / "banks")
    monkeypatch.setattr(bank, "DELETED_DIR", tmp_path / "banks" / "deleted")
    monkeypatch.setattr(bank, "CONFIG_PATH", tmp_path / "config.json")
    return tmp_path / "banks"
