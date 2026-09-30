"""One conversation: remembers which answer is on the table so follow-up questions can be recognised.

The overlay feeds each transcribed utterance to Session.hear() and shows what comes back:

    heard = session.hear(text, confidence=transcript.confidence)
    if heard.kind == "new":         show heard.primary.entry                      (a fresh question)
    elif heard.kind == "followup":  show heard.followup.prepared (if any) or heard.followup.focus,
                                    with heard.followup.anchor as the answer being probed
    elif heard.kind == "unmatched": show heard.matches as "did you mean…" candidates
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from bank import Entry, load_config
from followup import PreparedFollowup, Signal, detect, focus_passages, prepared_followups
from matcher import Match, Matcher, looks_like_question
from transcript import clean_text

ANCHOR_BONUS = 0.06        # extra credit for a follow-up written under the answer being discussed
AMBIGUOUS_MARGIN = 0.04    # a cue-flagged follow-up only loses to a bank match that clears the threshold by this much
WORDY_MATCH = 0.85         # ...and only when the heard words really are that bank question's words


@dataclass
class FollowupHit:
    anchor: Entry                                   # the answer being probed
    signal: Signal                                  # what about the wording says "follow-up"
    prepared: PreparedFollowup | None = None        # the follow-up you wrote, when one matched
    score: float = 0.0                              # similarity to that prepared follow-up
    focus: list[str] = field(default_factory=list)  # sentences of the anchor answer that address it
    related: list[PreparedFollowup] = field(default_factory=list)  # anchor's prepared follow-ups, closest first


@dataclass
class Heard:
    text: str
    is_question: bool
    kind: str                                       # "new" | "followup" | "unmatched" | "statement"
    matches: list[Match] = field(default_factory=list)   # standalone matches against the banks, best first
    primary: Match | None = None                    # best standalone match, only if it clears the threshold
    followup: FollowupHit | None = None
    anchor: Entry | None = None                     # the answer on the table after this utterance
    uncertain: bool = False                         # close call, or the speech model was unsure
    confidence: float | None = None


def _key(e: Entry | None) -> tuple[str, str] | None:
    return (e.bank, e.question) if e else None


class Session:
    def __init__(self, matcher: Matcher, cfg: dict | None = None, clock: Callable[[], float] = time.monotonic):
        self.matcher = matcher
        self.cfg = cfg if cfg is not None else load_config()
        self._clock = clock
        self._anchor: Entry | None = None
        self._anchor_at = 0.0

    def reset(self) -> None:
        self._anchor, self._anchor_at = None, 0.0

    def live_anchor(self, now: float | None = None) -> Entry | None:
        """The answer under discussion, or None once it has gone stale."""
        now = self._clock() if now is None else now
        window = float(self.cfg.get("followup_window_seconds", 240))
        return self._anchor if self._anchor is not None and now - self._anchor_at <= window else None

    def hear(self, text: str, confidence: float | None = None, now: float | None = None) -> Heard:
        now = self._clock() if now is None else now
        text = clean_text(text)
        thr = float(self.cfg.get("match_threshold", 0.78))
        matches = self.matcher.match(text, k=3)
        top = matches[0] if matches else None
        primary = top if top and top.score >= thr else None
        is_q = looks_like_question(text)
        anchor = self.live_anchor(now)

        hit = self._followup(text, anchor, primary, top, thr) if self.cfg.get("followups", True) else None
        if hit:
            kind, is_q = "followup", True
            self._anchor, self._anchor_at = hit.anchor, now
        elif primary:
            kind = "new"
            self._anchor, self._anchor_at = primary.entry, now
        else:
            kind = "unmatched" if is_q else "statement"

        margin = matches[0].score - matches[1].score if len(matches) > 1 else 1.0
        uncertain = kind != "statement" and (
            (kind == "new" and margin < 0.03) or (confidence is not None and confidence < 0.6))
        return Heard(text, is_q, kind, matches, primary, hit, self._anchor if kind != "statement" else anchor,
                     uncertain, confidence)

    def _followup(self, text: str, anchor: Entry | None, primary: Match | None, top: Match | None,
                  thr: float) -> FollowupHit | None:
        sig = detect(text)
        fms = self.matcher.match_followups(text, k=10_000)
        anchor_key = _key(anchor)

        def eff(m) -> float:
            return m.score + (ANCHOR_BONUS if _key(m.entry) == anchor_key else 0.0)

        fu_thr = float(self.cfg.get("followup_threshold", thr))
        best = max(fms, key=eff, default=None)
        # 1. a follow-up you prepared: it beats a bank question unless that question matches even better
        if best and eff(best) >= fu_thr and eff(best) >= (primary.score if primary else 0.0):
            return self._hit(best.entry, text, sig, fms, best)
        # 2. wording that only makes sense relative to the last answer ("Why?", "Tell me more", "What was the result?")
        if anchor is not None and sig:
            strong_standalone = primary is not None and primary.score >= thr + AMBIGUOUS_MARGIN \
                and primary.lexical >= WORDY_MATCH
            if not strong_standalone:
                return self._hit(anchor, text, sig, fms, None)
        return None

    def _hit(self, anchor: Entry, text: str, sig: Signal, fms, best) -> FollowupHit:
        mine = [m for m in fms if _key(m.entry) == _key(anchor)]
        related = [m.followup for m in mine] or prepared_followups(anchor)
        focus: list[str] = []
        if best is None:  # nothing prepared: point at the part of the anchor answer that addresses it
            focus = focus_passages(anchor, text, sig.kind, embed=self.matcher.embed_texts)
        return FollowupHit(anchor, sig, best.followup if best else None, best.score if best else 0.0, focus, related)
