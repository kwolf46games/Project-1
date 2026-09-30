"""Shared test setup: a small synthetic bank in the real file format, and a stand-in embedder.

The stand-in embeds by words and character trigrams, so it can prove the plumbing (routing, thresholds,
repair) offline. It cannot judge meaning the way the real model does; see `it test` for that.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bank as b  # noqa: E402
from matcher import Matcher  # noqa: E402
from session import Session  # noqa: E402

BANK_MD = """# Test bank

## Tell me about a time you led a team.
tags: leadership

**What I'm listening for: **a clear story.

**SCRIPT  ·  ABOUT 60 SECONDS SPOKEN** "At NovaCore I led a team of six on a migration. I set the plan and ran the weekly check-in. The client was nervous about downtime, so I wrote a rollback plan. We finished two weeks early. Afterwards uptime rose by 20% and support tickets fell by half."

**LIKELY FOLLOW-UPS  ·  SAY THIS**

**How big was the team?**   ·   about 10 sec Six people: four engineers, one analyst and me.
**Tell me more about the rollback plan.**   ·   about 20 sec It was a one-page checklist with a clear trigger for rolling back.
Every step had an owner.

## Why should we hire you?

**SCRIPT  ·  ABOUT 30 SECONDS SPOKEN** "Because I ship, and I communicate."

**LIKELY FOLLOW-UPS  ·  SAY THIS**

**What makes you different from other candidates?**   ·   about 15 sec I close the loop with clients.

## Why should we not hire you?

**SCRIPT  ·  ABOUT 30 SECONDS SPOKEN** "I haven't done cold outreach at volume."

## What is your biggest weakness?
also: What's your greatest weakness?

**SCRIPT  ·  ABOUT 30 SECONDS SPOKEN** "I do too much myself instead of delegating."

## Tell me about a failure or setback.

**SCRIPT  ·  ABOUT 30 SECONDS SPOKEN** "We missed an SLA at NovaCore. I owned it and fixed the SLA alerting."
"""


def fake_embed(texts):
    """Hash words + character trigrams into a fixed-size vector."""
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t.lower()
        toks = [w for w in "".join(c if c.isalnum() else " " for c in t).split()]
        for w in toks:
            out[i, zlib.crc32(b"w:" + w.encode()) % 384] += 1.0
        s = "  " + " ".join(toks) + " "
        for j in range(len(s) - 2):
            out[i, zlib.crc32(b"g:" + s[j:j + 3].encode()) % 384] += 0.4
    return out


@pytest.fixture
def entries():
    return b.parse_markdown(BANK_MD, "test").entries


@pytest.fixture
def matcher(entries):
    m = Matcher(embed=fake_embed)
    m.load(entries)
    return m


@pytest.fixture
def session(matcher):
    clock = {"t": 1000.0}
    s = Session(matcher, {"match_threshold": 0.78, "followups": True, "followup_window_seconds": 240},
                clock=lambda: clock["t"])
    s.clock = clock
    return s
