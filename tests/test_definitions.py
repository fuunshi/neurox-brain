"""
Definition extraction: the four patterns, the five guards, and the phrase span.

Written as characterisation tests before any behaviour changed, so that the
`_phrase_span` fix that follows is provably a fix and not a rewrite. Two of the
span tests below assert behaviour that is **known to be wrong** and say so;
they exist to pin the bug, and they are expected to be rewritten by the commit
that fixes it.
"""

from __future__ import annotations

from neurox_brain.nlp import definitions


def _attr_token(document):
    """The `attr` token of the first copular construction in a document."""
    return next(t for t in document.doc if t.dep_ == "attr")


def _terms(document) -> list[str]:
    return [d.term for d in definitions.extract(document)]


def _definition_of(document, term: str) -> str:
    return next(d.definition for d in definitions.extract(document) if d.term == term)


# --------------------------------------------------------------- patterns --- #


def test_copular_definition(parse):
    document = parse(
        "A stack is a linear data structure that follows last in first out order."
    )

    found = definitions.extract(document)

    assert [d.term for d in found] == ["stack"]
    assert found[0].pattern == "copular"
    assert found[0].definition.startswith("A linear data structure")


def test_relative_clause_is_kept_in_the_definition(parse):
    """
    A `relcl` is part of the noun phrase, not a second clause.

    It is the one dependent clause the phrase span deliberately keeps, and
    nothing in the fix may change that.
    """
    document = parse(
        "A queue is a linear data structure that follows first in first out order."
    )

    assert "first in first out" in _definition_of(document, "queue")


def test_cue_definition(parse):
    document = parse("A stack is defined as a linear data structure.")

    found = definitions.extract(document)

    assert [d.term for d in found] == ["stack"]
    assert found[0].pattern == "cue"
    assert found[0].definition == "A linear data structure."


def test_cue_with_a_gerund_complement_is_missed(parse):
    """
    **Known gap, not yet fixed.** "X is defined as <gerund>" yields nothing.

    Two separate reasons, and both would have to change: the gerund attaches as
    `pcomp` where `_object_of` only follows `prep -> pobj`, and even once found
    the nominal-head guard rejects it because a gerund's POS is `VERB`.

    Left as a gap deliberately rather than fixed alongside the span work. This
    is a *recall* change — it would add cards — and the tuning agreed for this
    pass is precision-first, so it wants its own commit and its own look at what
    it produces. Asserted here so the hole is recorded rather than rediscovered.
    """
    document = parse("Hashing is defined as mapping a key to an array position.")

    assert definitions.extract(document) == []


def test_appositive_definition(parse):
    document = parse(
        "A binary search tree, a node-based tree, keeps its keys in order."
    )

    found = [d for d in definitions.extract(document) if d.pattern == "appos"]

    assert found, "the appositive pattern did not fire"
    assert found[0].term == "binary search tree"


def test_hypernym_definition(parse):
    document = parse("A stack is a type of linear data structure.")

    found = definitions.extract(document)

    assert [d.pattern for d in found] == ["hypernym"]


# ----------------------------------------------------------------- guards --- #


def test_negation_is_rejected(parse):
    """`A stack is not a queue` defines nothing about a stack."""
    document = parse("A stack is not a linear data structure at all.")

    assert _terms(document) == []


def test_modal_is_rejected(parse):
    """
    A hedged definition asserts something the source did not.

    Note this guard is definition-specific: the purpose family must later
    *accept* "can be used to", which is why the two need separate guard sets.
    """
    document = parse("A stack could be a linear data structure of some kind.")

    assert _terms(document) == []


def test_expletive_subject_is_rejected(parse):
    """The front would be the word "there"."""
    document = parse("There is a linear data structure called a stack.")

    assert _terms(document) == []


def test_contrastive_complement_is_rejected(parse):
    """`another data structure` defines this one by what it is not."""
    document = parse("A stack is another linear data structure entirely.")

    assert _terms(document) == []


# ------------------------------------------------------------ phrase span --- #
#
# These four exercise `_phrase_span` directly, because that is where the
# truncation lives and a card-level assertion would not say which part broke.


def test_span_keeps_an_interior_conjunction(parse):
    """
    Coordination *inside* a noun phrase survives — by accident, today.

    The span is the enclosing range of the surviving tokens, so a dropped
    `conj` that sits before a later kept token is swallowed back in. This test
    documents that the good case works; the next one documents that it works
    for the wrong reason.
    """
    document = parse("A graph is a data structure of vertices and edges in computing.")
    span = definitions._phrase_span(_attr_token(document))

    assert "and edges" in span.text


def test_span_keeps_a_trailing_conjunction(parse):
    """
    **This was the bug.** A `conj` at the right edge used to be lost.

    `and` is `cc` and `edges` is `conj`. Because the span is the enclosing range
    over the *kept* tokens, an excluded token in the middle is swallowed back in
    and one at the end has nothing beyond it to be swallowed by — so `vertices`
    became the last kept token and the card back read "…consisting of vertices."
    The reader was never told that edges are a second thing a graph consists of,
    and the same truncated string was the correct answer and the distractor pool
    for every quiz built from that definition.

    Fixed by judging a conjunct on what it hangs off rather than dropping the
    label outright.
    """
    document = parse(
        "A graph is a non-linear data structure consisting of vertices and edges."
    )
    span = definitions._phrase_span(_attr_token(document))

    assert span.text == "a non-linear data structure consisting of vertices and edges"


def test_span_still_cuts_clause_coordination(parse):
    """
    The other half of the same exclusion, and it must survive the fix.

    Here the second clause is a `conj` too, so a fix that simply stops
    excluding `conj` would drag "and a queue is FIFO" into the definition of a
    stack. The distinction the fix has to make is the conjunct's head.
    """
    document = parse("A stack is a LIFO structure and a queue is FIFO.")
    span = definitions._phrase_span(_attr_token(document))

    assert "queue" not in span.text
    assert "FIFO" not in span.text
