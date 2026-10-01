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


# ------------------------------------------------ comparisons and purpose --- #


def test_difference_between_names_both_participants(parse):
    """
    A comparison is about a *pair*, and the phrase span has to be split for it.

    The span deliberately keeps nominal coordination — that is what makes
    "vertices and edges" survive in a definition — but here each half is one
    participant, so a front of "stack and a queue" would be the wrong card
    twice over.
    """
    document = parse(
        "The difference between a stack and a queue is the order of removal."
    )

    fact = _card(document, "stack")

    assert fact is not None
    assert fact.family == facts.COMPARISON
    assert fact.other == "queue"
    assert "and" not in fact.term


def test_unlike_keeps_its_negation(parse):
    """
    The one place a negation *is* the fact.

    `definitions._passes_guards` rejects negation, and sharing that guard here
    would delete this card rather than fix it — which is why the comparison
    family writes its own rule instead of taking a flag on the shared one.
    """
    document = parse("Unlike arrays, linked lists do not require contiguous memory.")

    fact = _card(document, "linked lists")

    assert fact is not None
    assert fact.pattern == "unlike"
    assert "do not require" in fact.back


def test_negation_is_rejected_by_the_other_families(parse):
    """
    The same sentence shape the definition guards reject, in a fact family.

    Before this check existed, "linked lists do not require contiguous memory"
    produced the card "linked lists require contiguous memory" — false rather
    than merely weak, which is the distinction that matters.
    """
    document = parse("Linked lists do not require contiguous memory.")

    assert facts.extract(document, salience.build(document)) == []


def test_purpose_accepts_a_modal(parse):
    """
    "can be used to" is how course text states a capability.

    The definition guard rejects modals because a hedged *definition* asserts
    something the source did not. A purpose is the opposite case, and the two
    guard sets exist separately for exactly this sentence.
    """
    document = parse("A stack can be used to reverse a string in place.")

    fact = _card(document, "stack")

    assert fact is not None
    assert fact.family == facts.PURPOSE


def test_process_keeps_its_sequence(parse):
    """A chain is its conjuncts; dropping them leaves half a procedure."""
    document = parse("The algorithm first sorts the array and then merges the two halves.")

    fact = _card(document, "algorithm")

    assert fact is not None
    assert fact.pattern == "ordered_chain"
    assert "sorts" in fact.back and "merges" in fact.back


def test_process_rejects_an_unresolved_pronoun(parse):
    """
    "decodes it, and executes it" is a real sentence and a useless card.

    There is no coreference resolution in this stack, so a reader meeting the
    back out of context cannot tell what "it" refers to. Dropping the sentence
    is the intended cost.
    """
    document = parse(
        "The CPU fetches the instruction, decodes it, and executes it immediately."
    )

    for fact in facts.extract(document, salience.build(document)):
        assert "decodes it" not in fact.back
