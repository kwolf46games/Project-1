"""Recognising a story that was already told: same story retold vs a different story, on hand-written examples."""
import inspect
import itertools

import pytest

import stories as st
from story_samples import (CHECKOUT_TIMEOUTS, DASHBOARD, HACKATHON, LED_MIGRATION, MIGRATION_REWORDED, MIGRATION_SHORT,
                           MOTIVATION, MOTIVATION_AGAIN, OTHER_STORIES, SAME_STORY, SUPPORT_QUEUE, TICKETING_ROLLOUT)


def log_with(text, key=1, question="Tell me about a time you led a team"):
    log = st.StoryLog()
    log.add(key, question, text)
    return log


# ---------- what counts as a story ----------

@pytest.mark.parametrize("text", SAME_STORY + OTHER_STORIES)
def test_each_sample_is_recognised_as_a_story(text):
    assert st.is_story(text)


def test_opinions_and_motivation_are_not_stories():
    assert not st.is_story(MOTIVATION)
    assert not st.is_story("I'd say my biggest weakness is perfectionism. I'm working on it by setting time limits.")
    assert not st.is_story("A good test plan starts with risk. We prioritise by impact and likelihood.")
    assert not st.is_story("")


def test_a_single_past_event_is_not_yet_a_story():
    assert not st.is_story("Last year I led a migration. It is something I enjoy.")


def test_words_every_interview_story_shares_do_not_count():
    words = st.story_words("I learned a lot working on this team project, and the result taught me an important lesson.")
    assert words == frozenset()
    assert {"bill", "platform", "six", "weekend"} <= st.story_words(LED_MIGRATION)   # stems, not whole words


def test_word_endings_and_number_forms_are_folded_together():
    assert st.story_words("mapping the records") == st.story_words("we map record")
    assert "40" in st.story_words("it improved by 40 percent")


# ---------- the same story vs a different one ----------

@pytest.mark.parametrize("earlier,later", [(a, b) for a, b in itertools.permutations(SAME_STORY, 2)])
def test_the_same_story_retold_is_recognised(earlier, later):
    assert log_with(earlier).find(later, key=2, question="Tell me about a conflict") is not None


@pytest.mark.parametrize("later", OTHER_STORIES)
def test_a_different_story_is_not_mistaken_for_it(later):
    for earlier in SAME_STORY:
        assert log_with(earlier).find(later, key=2, question="Tell me about a conflict") is None


def test_different_stories_never_come_near_each_other_in_any_pairing():
    """The margin matters more than the verdict: nothing different should sit right at the threshold."""
    pool = OTHER_STORIES + [LED_MIGRATION]
    for a, b in itertools.combinations(pool, 2):
        n, share = st.shared(st.story_words(a), st.story_words(b))
        assert share < st.SAME_SHARE * 0.9 or n < st.SAME_COUNT, (a[:40], b[:40], n, share)


def test_retellings_clear_the_threshold_with_room_to_spare():
    mine = st.story_words(MIGRATION_REWORDED)
    for other in (LED_MIGRATION, MIGRATION_SHORT):
        n, share = st.shared(mine, st.story_words(other))
        assert n >= st.SAME_COUNT and share >= st.SAME_SHARE * 1.1


def test_a_few_stray_shared_words_in_a_long_answer_are_not_a_repeat():
    """Five shared words out of dozens is coincidence (the same trade, the same tools), not the same story."""
    earlier = " ".join([LED_MIGRATION, CHECKOUT_TIMEOUTS, SUPPORT_QUEUE])
    later = " ".join([HACKATHON, DASHBOARD, TICKETING_ROLLOUT, "We checked the billing platform, the finance mapping and the weekend."])
    n, share = st.shared(st.story_words(later), st.story_words(earlier))
    assert n >= st.SAME_COUNT and share < st.SAME_SHARE
    assert log_with(earlier).find(later, key=2, question="another") is None


def test_the_closest_earlier_story_is_the_one_reported():
    log = st.StoryLog()
    log.add(1, "q one", HACKATHON)
    log.add(2, "q two", MIGRATION_SHORT)
    assert log.find(LED_MIGRATION, key=3, question="q three").text == MIGRATION_SHORT


def test_a_motivation_answer_said_twice_is_never_flagged():
    log = st.StoryLog()
    assert not log.add(1, "why this role", MOTIVATION)
    assert log.find(MOTIVATION_AGAIN, key=2, question="what attracted you") is None
    assert len(log) == 0


# ---------- the opening, checked while the draft is still streaming ----------

def opening(text, n=st.OPEN_WORDS):
    return " ".join(text.split()[:n])


def test_a_repeated_opening_is_caught_early():
    log = log_with(LED_MIGRATION)
    assert log.opening(opening(MIGRATION_SHORT), key=2, question="another") is not None


@pytest.mark.parametrize("later", OTHER_STORIES)
def test_a_different_opening_is_left_alone(later):
    assert log_with(LED_MIGRATION).opening(opening(later), key=2, question="another") is None


def test_too_few_words_to_judge_is_not_a_repeat():
    assert log_with(LED_MIGRATION).opening("At my last company", key=2) is None


# ---------- the log itself ----------

def test_a_draft_for_the_same_utterance_replaces_the_earlier_one():
    log = st.StoryLog()
    log.add(1, "first guess", HACKATHON)
    log.add(1, "final wording", LED_MIGRATION)
    assert len(log) == 1 and log.recall()[0].text == LED_MIGRATION


def test_a_draft_is_never_compared_with_its_own_earlier_version():
    log = log_with(LED_MIGRATION, key=7)
    assert log.find(MIGRATION_SHORT, key=7, question="x") is None          # same utterance regenerated
    assert log.recall(key=7) == []


def test_the_same_question_asked_again_may_get_the_same_story():
    log = log_with(LED_MIGRATION, question="Tell me about a time you led a team")
    assert log.find(MIGRATION_SHORT, key=2, question="Can you tell me about a time when you led a team?") is None
    assert log.find(MIGRATION_SHORT, key=2, question="Tell me about a time you handled conflict") is not None
    assert log.recall(key=2, question="tell me about a time you led a team") == []


@pytest.mark.parametrize("a,b,same", [
    ("Tell me about a time you led a team", "Can you tell me about a time when you led a team?", True),
    ("Tell me about a time you led a team", "Tell me about a time you led a project", False),
    ("Tell me about a time you led a team", "Tell me about a time you handled a conflict", False),
    ("What is your biggest weakness?", "what's your greatest weakness", False),
    ("", "anything", False),
])
def test_same_question(a, b, same):
    assert st.same_question(a, b) is same


def test_recall_is_newest_first_limited_and_bounded():
    log = st.StoryLog(keep=3)
    for i, text in enumerate([HACKATHON, LED_MIGRATION, OTHER_STORIES[1], OTHER_STORIES[2]], 1):
        log.add(i, f"question number {i}", text)
    assert len(log) == 3                                                    # the oldest fell off
    assert [s.key for s in log.recall()] == [4, 3, 2]
    assert [s.key for s in log.recall(limit=2)] == [4, 3]


def test_forget_and_clear():
    log = st.StoryLog()
    log.add(1, "a b c", HACKATHON)
    log.add(2, "d e f", LED_MIGRATION)
    log.forget(1)
    log.forget(None)
    assert [s.key for s in log.recall()] == [2]
    log.clear()
    assert len(log) == 0


def test_a_draft_without_a_key_is_just_added():
    log = st.StoryLog()
    log.add(None, "one", HACKATHON)
    log.add(None, "two", LED_MIGRATION)
    assert len(log) == 2


def test_a_digest_is_the_opening_sentences_within_the_word_budget():
    d = st.digest(LED_MIGRATION)
    assert LED_MIGRATION.startswith(d) and d.endswith(".")
    assert len(d.split()) <= st.DIGEST_WORDS + 25                           # a long first sentence is kept whole
    assert st.digest("One short sentence.") == "One short sentence."


def test_a_story_is_remembered_in_memory_only():
    """Closing the program must forget every story, so nothing here may touch the disk."""
    source = inspect.getsource(st)
    for word in ("open(", "write_text", "read_text", "json", "pickle", "shelve", "sqlite"):
        assert word not in source, word
