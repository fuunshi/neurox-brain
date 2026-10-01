"""
Card selection: the caps and the front rule.

These are the tuning constants from the plan, and they are pinned because every
one of them is a silent failure rather than a loud one — a cap that stops
working produces a worse deck, not an error.
"""

from __future__ import annotations

from neurox_brain.nlp import analyse, selection
from neurox_brain.schemas import AnalysisOptions

PARAGRAPH = (
    "A binary search tree is a node-based binary tree that keeps its keys sorted. "
    "A binary search tree has a time complexity of logarithmic order when balanced. "
    "The binary search tree consists of nodes with a key and two child pointers. "
    "Inserting into a binary search tree follows the search path and then adds a leaf. "
    "The difference between a stack and a queue is the order of removal. "
    "A stack is used to evaluate arithmetic expressions in compilers."
)


def test_no_two_cards_share_a_front(parse):
    """
    Two cards with the same prompt are indistinguishable to a reader.

    The scheduler also treats them as independent items, so a duplicate front is
    two reviews of something the reader cannot tell apart.
    """
    document = parse(PARAGRAPH)
    cards = analyse.analyse(
        PARAGRAPH, None, AnalysisOptions(max_cards=12, include_cloze=True)
    ).cards

    fronts = [card.front for card in cards]
    assert len(fronts) == len(set(fronts)), fronts


def test_comparison_front_names_both_participants(parse):
    """A bare term cannot express a two-party fact — the front would misstate it."""
    document = parse(PARAGRAPH)
    cards = analyse.analyse(PARAGRAPH, None, AnalysisOptions(max_cards=12)).cards

    comparisons = [c for c in cards if c.kind == "COMPARISON"]
    assert comparisons, "no comparison card survived selection"
    assert all(selection.VS in c.front for c in comparisons)


def test_a_term_can_hold_several_facts_but_not_many(parse):
    """
    The per-term cap, which is what stops one topic becoming the whole deck.

    "binary search tree" appears in four of the six sentences above and legitimately
    has several distinct facts; it must still not take the deck.
    """
    cards = analyse.analyse(PARAGRAPH, None, AnalysisOptions(max_cards=12)).cards

    per_term: dict[str, int] = {}
    for card in cards:
        key = card.front.split(" — ")[0].split(f" {selection.VS} ")[0].lower()
        per_term[key] = per_term.get(key, 0) + 1

    assert per_term, "no cards produced"
    assert max(per_term.values()) <= selection.MAX_PER_TERM_STRONG


def test_cloze_is_off_by_default(parse):
    """A generated deck is facts, not sentences with holes in them."""
    cards = analyse.analyse(PARAGRAPH, None, AnalysisOptions(max_cards=12)).cards

    assert not any(card.kind == "CLOZE" for card in cards)


def test_zero_quiz_questions_skips_the_work(parse):
    """`max_quiz_questions=0` is a request for none, not an error."""
    result = analyse.analyse(
        PARAGRAPH, None, AnalysisOptions(max_cards=12, max_quiz_questions=0)
    )

    assert result.quiz == []


def test_a_property_of_front_is_suppressed(parse):
    """
    "The advantage of a linked list is its dynamic size" is not a definition.

    The copular pattern cannot tell the two apart, so left alone it fronted the
    card "advantage of a linked list" — a category rather than a subject, and
    the linked list never got the fact.
    """
    text = "The advantage of a linked list is its dynamic size."

    cards = analyse.analyse(text, None, AnalysisOptions(max_cards=10)).cards

    assert all(" of " not in card.front or not card.front.startswith("advantage")
               for card in cards), [c.front for c in cards]
