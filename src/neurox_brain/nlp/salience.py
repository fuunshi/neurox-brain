"""
What this document cares about: sentence centrality and term weight.

**Why this module exists.** The pipeline already computed both of these and used
neither. TextRank scored every sentence for the summary; TF-IDF scored every
candidate term for the keyword list. Card selection ignored both and took
definitions in extraction order, so a fact stated in a throwaway aside ranked
the same as the sentence the chapter is built around.

That is the whole reason a card can be *correct and worthless*: "A stack is a
linear data structure" and "A stack is a useful thing to know about" match the
same pattern, and only the document can say which one it is about.

**Computed once, deliberately.** `build` ranks the sentences and scores the
terms in a single pass and hands the result to everything downstream. Ranking
twice would double the pipeline's most expensive non-parse stage — the
similarity matrix is a Python double loop — and, worse, would let summarisation
and card selection disagree about salience in a way nothing would report.

**The honest limitation.** These are document-relative signals, so the same
sentence in a different chapter yields a different verdict. That is intended:
"worth remembering" is a question about a document, and a fact mentioned once in
passing genuinely is less worth remembering than one the chapter returns to. The
cost is that a correct-but-unremarkable sentence is dropped, and no threshold
makes that cost zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import keywords as keywords_module
from . import textrank


# Stripped from both sides of the term comparison.
#
# A card's term has already had its leading determiner removed — `_clean_term`
# does that, so the front reads "stack" and not "a stack" — while a keyword key
# keeps whatever the noun chunk started with, which for a sentence-initial
# mention is "a stack". Without normalising both, `is_salient("stack")` is false
# for every single-word term in the document and the gate never fires: a
# failure that looks exactly like "this document has no salient terms".
_LEADING_DETERMINERS = frozenset({"a", "an", "the"})


def _normalise_key(key: str) -> str:
    """A term key with any leading determiner removed."""
    words = key.split()

    while words and words[0] in _LEADING_DETERMINERS:
        words.pop(0)

    return " ".join(words)


@dataclass
class Salience:
    """One document's sentence scores and term weights."""

    #: Sentence index (into `doc.sentences`) -> TextRank score. The scores are
    #: the PageRank stationary distribution, so they sum to ~1 and are directly
    #: comparable within a document — but not between documents.
    sentences: dict[int, float]

    #: Keyword key -> TF-IDF score. Keys are `keywords._phrase_key` form:
    #: lemmatised, lowercased, punctuation stripped. Use `key_of` to build one.
    terms: dict[str, float]

    #: The ranked sentences themselves, for a caller that wants to pass them on
    #: rather than recompute them.
    ranked: list["textrank.RankedSentence"] = field(default_factory=list)

    #: term -> key, filled lazily by `is_salient`. Card terms arrive as display
    #: strings, so matching one against `terms` means lemmatising it, and the
    #: same term is asked about once per candidate rather than once per card.
    _keys: dict[str, str] = field(default_factory=dict, repr=False)

    def key_of(self, term: str) -> str:
        """
        The keyword-map key for a card's term.

        Re-parses the term, which is a spaCy call for a string that is usually
        two or three words — cheap next to the parse that produced the document,
        and it is what makes a term extracted as "Binary Search Trees" match the
        keyword key "binary search tree".
        """
        cached = self._keys.get(term)
        if cached is not None:
            return cached

        from .pipeline import get_nlp

        doc = get_nlp()(term)
        key = _normalise_key(
            " ".join(
                token.lemma_.lower()
                for token in doc
                if not token.is_punct and not token.is_space
            )
        )

        self._keys[term] = key
        return key

    def is_salient(self, term: str) -> bool:
        """Whether this document scored the term as one of its keywords."""
        return self.key_of(term) in self.terms

    def score_of_sentence(self, index: int) -> float:
        """A sentence's centrality, or 0.0 for an index the ranking does not hold."""
        return self.sentences.get(index, 0.0)

    def sentence_cut(self, fraction: float) -> float:
        """
        The score separating the top `fraction` of sentences from the rest.

        Returned as a threshold rather than a set because callers ask about
        sentences one at a time, while walking a document. Returns 0.0 when
        there is nothing to rank, which makes every comparison against it fail
        and the caller fall back to its base confidence — the honest outcome for
        a document with no rankable sentences.
        """
        if not self.sentences or fraction <= 0:
            return 0.0

        scores = sorted(self.sentences.values(), reverse=True)
        boundary = max(0, min(len(scores) - 1, int(len(scores) * fraction) - 1))

        return scores[boundary]


def build(doc) -> Salience:
    """
    Rank the document's sentences and score its terms, once.

    The ranked list is kept as well as the score map because `summarise` wants
    the sentences and card selection wants the scores, and both should come from
    one ranking rather than two.
    """
    ranked = textrank.rank_sentences(doc.sentences)
    scored = keywords_module.score_all(doc)

    return Salience(
        sentences={item.index: item.score for item in ranked},
        terms={_normalise_key(item.key): item.score for item in scored},
        ranked=ranked,
    )
