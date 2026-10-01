"""
Salience: the sentence and term signals that card selection is gated on.

The important assertions here are not about scores being *right* — there is no
ground truth for "how central is this sentence" — but about the two things that
would be invisible if they broke: that summaries and cards come from one
ranking, and that term lookup survives the lemmatisation between a card's
display term and a keyword's key.
"""

from __future__ import annotations

from neurox_brain.nlp import salience, summarise, textrank

PARAGRAPH = (
    "A stack is a linear data structure that follows last in first out order. "
    "A queue is a linear data structure that follows first in first out order. "
    "A linked list is a linear data structure whose elements point to the next element. "
    "A binary tree is a hierarchical data structure where each node has at most two children. "
    "A graph is a non-linear data structure consisting of vertices and edges. "
    "Hashing is a technique that maps keys to array positions using a hash function. "
    "A stack is used to evaluate arithmetic expressions in compilers. "
    "The call stack of a running program is the clearest example of a stack."
)


def test_build_scores_every_sentence(parse):
    document = parse(PARAGRAPH)

    found = salience.build(document)

    assert len(found.sentences) == len(document.sentences)
    assert len(found.ranked) == len(document.sentences)


def test_scores_are_normalised_within_a_document(parse):
    """
    TextRank's scores are the stationary distribution of a random walk, so they
    sum to about 1. Pinned because a change that returned raw PageRank values
    would still *sort* the same way and would silently break any threshold
    expressed as a fraction.

    Needs at least `GRAPH_MIN_SENTENCES` sentences to reach the graph at all —
    the frequency fallback's scores are deliberately not normalised, and
    asserting this on a short document would be asserting the wrong thing.
    """
    long_text = " ".join(
        f"A {name} is a linear data structure that stores {name} values in order."
        for name in (
            "stack", "queue", "list", "tree", "graph", "heap",
            "set", "map", "trie", "table", "array", "buffer",
        )
    )
    document = parse(long_text)

    assert len(document.sentences) >= textrank.GRAPH_MIN_SENTENCES

    found = salience.build(document)

    assert abs(sum(found.sentences.values()) - 1.0) < 0.01


def test_short_documents_use_the_frequency_fallback(parse):
    """
    Below `GRAPH_MIN_SENTENCES`, both callers get frequency ranking.

    A three-sentence document has a graph too sparse for the walk to mean
    anything, so agreeing on the fallback matters more here than anywhere.
    """
    document = parse("A stack is LIFO. A queue is FIFO. Both are linear.")

    found = salience.build(document)

    assert len(found.ranked) == len(document.sentences)
    assert textrank.rank_sentences(document.sentences) == found.ranked


def test_is_salient_matches_a_lemmatised_term(parse):
    """
    A card front is a display string; a keyword key is lemmatised and lowercased.

    "A stack" has to find the keyword key "stack" for the term gate to mean
    anything, and this is the seam where it would quietly stop working.
    """
    found = salience.build(parse(PARAGRAPH))

    assert found.is_salient("A stack")
    assert found.is_salient("stack")
    assert found.key_of("A stack") == found.key_of("stack")


def test_summarise_uses_a_supplied_ranking(parse):
    """
    Passing the ranking in must not change the summary.

    This is the guard on the shared-ranking refactor: if `summarise` ranked the
    document itself and ignored `ranked`, this would still pass — so the second
    half of the test checks the plumbing by supplying a deliberately empty
    ranking and confirming it is *used*.
    """
    document = parse(PARAGRAPH)
    shared = salience.build(document)

    with_shared = summarise.summarise(document, 3, ranked=shared.ranked)
    without = summarise.summarise(document, 3)

    assert with_shared == without
    assert summarise.summarise(document, 3, ranked=[]) == []
