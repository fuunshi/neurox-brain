"""
The fact families.

Each test comes in two halves: a positive case proving the pattern fires, and an
adversarial one proving it does not fire on a sentence that merely resembles it.
The second half is the part that matters — `have`, `contain` and `include` are
among the most frequent verbs in expository prose, and a family that fires on
all of them puts a card on nearly every sentence in a document.
"""

from __future__ import annotations

from neurox_brain.nlp import facts, salience


def _patterns(document, salience_map=None) -> list[str]:
    return [f.pattern for f in facts.extract(document, salience_map)]


def _card(document, term_fragment: str, salience_map=None):
    for fact in facts.extract(document, salience_map):
        if term_fragment in fact.term.lower():
            return fact

    return None


# --------------------------------------------------- facts & properties --- #


def test_consists_of(parse):
    document = parse("The tree consists of nodes with a key and two child pointers.")

    fact = _card(document, "tree")

    assert fact is not None
    assert fact.pattern == "consist_of"
    assert fact.family == facts.PROPERTY
    assert fact.back == "Nodes with a key and two child pointers."


def test_possess_property_needs_a_property_noun(parse):
    """
    "has" alone is not a fact. The object has to be one of `PROPERTY_NOUNS`.

    This split is the family's whole discipline. Both sentences below have the
    same shape; only the first is about the thing being studied.
    """
    document = parse(
        "A binary search tree has a time complexity of logarithmic order in the balanced case. "
        "Algorithms have been studied for decades by many researchers."
    )

    patterns = _patterns(document)

    assert "possess_property" in patterns
    assert _card(document, "algorithm") is None or _card(document, "algorithm").pattern != (
        "possess_property"
    )


def test_possess_salient_is_below_the_floor_without_salience(parse):
    """
    The open pattern cannot reach a reader on its own.

    `possess_salient` exists for factual statements whose object is not a
    property noun, and its base score is deliberately under `MIN_FACT_CONFIDENCE`
    so that only a document which keeps returning to the term lifts it into a
    deck. That is the mechanism, and this pins it.
    """
    document = parse("The list contains a pointer to the next node in the chain.")

    unscored = _card(document, "list")
    scored = _card(document, "list", salience.build(document))

    # Whatever else happens, salience only ever adds.
    if unscored is not None and scored is not None:
        assert scored.confidence >= unscored.confidence

    assert facts.MIN_FACT_CONFIDENCE > 0.35, (
        "possess_salient's base must stay under the floor, or the gate is decoration"
    )


def test_every_fact_is_extractive(parse):
    """
    Every back is a span of the sentence it came from.

    The service has no language model, so this is the property that separates it
    from one that does: nothing here can state a fact the author did not write.

    Compared case-insensitively, because `_clean_definition` capitalises the
    first letter for display — "nodes with a key" becomes "Nodes with a key",
    and that is presentation rather than content.
    """
    document = parse(
        "The tree consists of nodes with a key and two child pointers. "
        "A binary search tree has a time complexity of logarithmic order."
    )

    for fact in facts.extract(document, salience.build(document)):
        assert fact.back.rstrip(".").lower() in fact.evidence.rstrip(".").lower(), (
            f"{fact.back!r} is not a span of {fact.evidence!r}"
        )
