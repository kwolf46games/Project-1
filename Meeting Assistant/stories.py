"""Remember which stories the drafted answers have already told, so the same one is not told twice.

The language model is told about the stories used so far (prevention), and every finished draft is also
checked against them here (safety net).  "The same story" is judged by the specifics a retelling keeps: the
nouns, the numbers and the turns of events, with the words every interview answer shares ("team", "learned",
"project") left out.  That is a word-level comparison, so a story retold in completely different terms can slip
past it; the instruction to the model is the first line of defence and this is the second.

Nothing is written to disk: closing the program forgets every story, and the next session starts clean.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

# A new draft is "the same story" when it shares at least this many of its distinctive words with an earlier one
# and they make up at least this share of the shorter of the two.  Different stories on the same theme, and even
# in the same business area, measure 0.12 or less on hand-written examples; a retelling measures 0.23 and up.
SAME_SHARE = 0.18
SAME_COUNT = 5
# The opening of a draft (its first sentences) is compared as it streams in, so a repeat is caught before it is
# shown.  Here the share is of the opening's own words, which is a stricter question.
OPEN_SHARE = 0.40
OPEN_COUNT = 4
OPEN_WORDS = 30
SAME_QUESTION = 0.75          # two questions this alike are the same question asked again, not a new one
DIGEST_WORDS = 60
KEEP = 12                     # stories remembered; older ones fall off

_STOP = frozenset("""a an the of to and or in on at for with is are was were be been being it its that this these those as by
from do does did done you your me my mine i you'd i'd i've i'm we we'd we've we're our us they them their he she his her
can could would will should may might must shall about please so but if then than too very just also really quite pretty
much many some any all each both more most other another such own same only even still not no nor yes there here when where
while after before during since until once again up down out off over under into onto through between among across what which
who whom whose how why because though although however thing things way ways kind lot lots bit little got get gets getting
going goes went gone make makes made making take takes took taken taking come came comes coming put puts one two three first
second next last back ever never always often sometimes already yet now today had has have having""".split())

# Words that turn up in nearly every interview story: they say what KIND of answer it is, not WHICH story.
_COMMON = frozenset("""team teams project projects work worked working works time times learned learn learning lesson lessons
result results challenge challenges situation task action reflection role company job people person someone everyone everything
anything something colleague colleagues manager important helped help helping needed need needs wanted want wants decided decide
tried try trying ended end ending started start starting realized realised knew know think thought felt feel looked look looking
turned turn asked ask told tell said say talked talk good great better best well big small new old long short different right able
sure clear clearly early later year years week weeks day days month months part parts goal goals plan plans together lead led
leading leader away spent join joined without added within took brought bring taught teach teaches group groups normal
set wrote write ran run keep kept found find built build called call gave give pulled pull change changed problem problems
issue issues question questions few fast common daily check""".split())

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SUFFIXES = ("ingly", "ing", "edly", "ed", "ies", "es", "ly", "s")
_PAST = frozenset("""was were had led ran built made took got went saw found felt knew thought told said gave came began wrote
met held kept brought sent spent left lost won chose stood set put paid sat spoke taught drew grew became decided did""".split())
_NOT_PAST = frozenset("need needed speed feed seed indeed proceed succeed exceed embed hundred weed breed".split())
_PERSON = frozenset("i we my our me us".split())


def _stem(w: str) -> str:
    if w.isdigit():
        return w
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":   # mapping -> mapp -> map
        w = w[:-1]
    return w


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def story_words(text: str) -> frozenset:
    """The distinctive words of a passage: stems of everything except filler and interview boilerplate."""
    out = set()
    for w in _words(text):
        if w in _STOP or w in _COMMON or (len(w) < 3 and not w.isdigit()):
            continue
        out.add(_stem(w))
    return frozenset(out)


def question_words(text: str) -> frozenset:
    return frozenset(_stem(w) for w in _words(text) if w not in _STOP and (len(w) > 2 or w.isdigit()))


def same_question(a: str, b: str) -> bool:
    qa, qb = question_words(a), question_words(b)
    return bool(qa and qb) and len(qa & qb) / len(qa | qb) >= SAME_QUESTION


def is_story(text: str) -> bool:
    """True when the passage tells something that happened (at least two sentences where I/we did something),
    as opposed to an opinion, a motivation or a technical explanation."""
    told = 0
    for sentence in _SENTENCE.split(text):
        ws = _words(sentence)
        if not _PERSON.intersection(ws):
            continue
        if any(w in _PAST or (w.endswith("ed") and len(w) > 3 and w not in _NOT_PAST) for w in ws):
            told += 1
    return told >= 2


def shared(a: frozenset, b: frozenset) -> tuple[int, float]:
    """(how many words they share, that count as a share of the smaller set)."""
    if not a or not b:
        return 0, 0.0
    n = len(a & b)
    return n, n / min(len(a), len(b))


def digest(text: str, words: int = DIGEST_WORDS) -> str:
    """The first few sentences: enough for the model to recognise the story and steer away from it."""
    out: list[str] = []
    count = 0
    for sentence in _SENTENCE.split(" ".join(text.split())):
        n = len(sentence.split())
        if out and count + n > words:
            break
        out.append(sentence)
        count += n
    return " ".join(out)


@dataclass
class Story:
    key: object                   # which utterance it answered; a draft for the same utterance replaces the old one
    question: str
    text: str
    words: frozenset = field(default_factory=frozenset)

    @property
    def digest(self) -> str:
        return digest(self.text)


class StoryLog:
    """The stories told so far in this run of the program, newest last."""

    def __init__(self, keep: int = KEEP) -> None:
        self.keep = keep
        self._stories: list[Story] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._stories)

    def clear(self) -> None:
        with self._lock:
            self._stories.clear()

    def _others(self, key, question: str) -> list[Story]:
        """Stories to compare against: not this utterance's own earlier draft, and not the answer to the very same
        question asked again (then repeating it is right, not a slip)."""
        return [s for s in self._stories
                if (key is None or s.key != key) and not (question and same_question(question, s.question))]

    def add(self, key, question: str, text: str) -> bool:
        """Note a finished draft. Only stories are kept; a draft for the same utterance replaces the earlier one."""
        if not is_story(text):
            self.forget(key)
            return False
        with self._lock:
            if key is not None:
                self._stories = [s for s in self._stories if s.key != key]
            self._stories.append(Story(key, question, text, story_words(text)))
            del self._stories[: -self.keep]
        return True

    def forget(self, key) -> None:
        if key is None:
            return
        with self._lock:
            self._stories = [s for s in self._stories if s.key != key]

    def recall(self, key=None, question: str = "", limit: int = 6) -> list[Story]:
        """What has already been told (newest first), for the model's instructions."""
        with self._lock:
            return list(reversed(self._others(key, question)))[:limit]

    def find(self, text: str, key=None, question: str = ""):
        """The earlier story that `text` retells, or None. Only a draft that is itself a story is checked."""
        if not is_story(text):
            return None
        mine = story_words(text)
        with self._lock:
            best, best_share = None, 0.0
            for s in self._others(key, question):
                n, share = shared(mine, s.words)
                if n >= SAME_COUNT and share >= SAME_SHARE and share > best_share:
                    best, best_share = s, share
            return best

    def opening(self, text: str, key=None, question: str = ""):
        """The earlier story the first sentences of a draft are already repeating, or None."""
        mine = story_words(text)
        if len(mine) < OPEN_COUNT:
            return None
        with self._lock:
            for s in reversed(self._others(key, question)):
                n = len(mine & s.words)
                if n >= OPEN_COUNT and n / len(mine) >= OPEN_SHARE:
                    return s
        return None
