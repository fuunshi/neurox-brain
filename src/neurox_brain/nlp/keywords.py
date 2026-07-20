"""
Keyword and keyphrase extraction by TF-IDF.

**The formula, and what each half is doing.**

     score(t, d) = tf(t, d) × idf(t)

`tf` is how often the term occurs *in this document*: a term the author keeps
using is a term the document is about. `idf` is how rare the term is *across all
documents*: a term in every chapter carries no information about which chapter
you are reading, however often it appears.

Either half alone is useless, and the failure of each is worth stating because
it is what the combination is for:

- **TF alone** ranks "the", "is" and "system" at the top of a computer science
  text. Frequency without rarity measures English, not the subject.
- **IDF alone** ranks a typo, a product name mentioned once, and a stray proper
  noun at the top. Rarity without frequency measures accident.

**Sublinear term frequency.** Raw counts are replaced by `1 + log(count)`.
Someone who writes "stack" twenty times is not thinking about it twenty times as
much as someone who wrote it once, and without this one repeated word dominates
every score in the document. This is the standard dampening and it is the
difference between a keyword list and a list of whatever was repeated.

**Why phrases, not just words.** "Operating system" is a keyword; "operating"
and "system" separately are two words that happen to be adjacent. So noun chunks
from the parse are candidates alongside single tokens, and a longer phrase gets a
small bonus — bounded, because an unbounded one rewards the longest chunk the
parser produced rather than the most meaningful.

**What is deliberately excluded.** No stemming beyond lemmatisation, no
synonym merging, no embedding-based expansion. Each would find more candidates
and every one of them makes the output harder to explain to the person reading
it, which for a keyword list shown on a study page is the whole value.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .corpus import corpus

# Parts of speech that can be a keyword. Nouns and proper nouns name things;
# adjectives sometimes qualify a concept the text is about. Verbs, adverbs and
# function words do not name anything.
CONTENT_POS = {"NOUN", "PROPN", "ADJ"}

# A candidate must be at least this long to be a word rather than a symbol.
MIN_TERM_CHARS = 3

# Longest phrase considered. Beyond three words a "keyphrase" is a clause.
MAX_PHRASE_TOKENS = 3

# Filler that is frequent in academic prose, is not a stop word in the ordinary
# sense, and never belongs in a keyword list. Kept short on purpose: a long
# hand-written list is a list nobody maintains, and the corpus IDF learns most
# of this if the text is long enough.
# Stripped from the front of a keyword phrase, as it is from a card front.
LEADING_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}

DOMAIN_FILLER = {
    "example", "examples", "figure", "figures", "table", "tables", "chapter",
    "section", "page", "pages", "following", "given", "used", "using", "use",
    "way", "ways", "thing", "things", "case", "cases", "time", "times",
    "number", "numbers", "value", "values", "part", "parts", "type", "types",
    "kind", "kinds", "form", "forms", "term", "terms", "word", "words",
    "point", "points", "note", "notes", "order", "result", "results",
    "problem", "problems", "idea", "ideas", "fact", "facts", "lot", "lots",
}


@dataclass(frozen=True)
class Keyword:
    """
    A scored term.

    `key` is the lemmatised form used for matching — "operating system" as the
    parser sees it — while `term` is what a reader should see, "Operating
    System". Both are needed and they are not interchangeable: the cloze module
    looks terms up by their lemmatised form out of the parse, and a lookup keyed
    on the display form silently misses every time. That failure is invisible —
    every score comes back zero and the blanks are chosen by word length
    instead, which looks plausible and is arbitrary.
    """

    key: str
    term: str
    score: float
    count: int


def _phrase_key(span) -> str:
    """A stable key for a phrase: lemmatised, lowercased, sorted word order kept.

    Word order is kept — "system operating" is not "operating system" — but the
    *surface* form is not, so "Operating Systems" and "operating system" collapse
    into one keyword rather than two entries with half the count each.
    """
    return " ".join(token.lemma_.lower() for token in span if not token.is_punct)


def _is_candidate(span, text: str) -> bool:
    """Whether a noun chunk is worth scoring."""
    tokens = [t for t in span if not t.is_space and not t.is_punct]

    if not tokens or len(tokens) > MAX_PHRASE_TOKENS:
        return False

    if len(text) < MIN_TERM_CHARS:
        return False

    # Every token has to be a content word or a connector between content words.
    # This rejects "it is a", which the parser will happily produce as a chunk.
    content = [t for t in tokens if t.pos_ in CONTENT_POS]
    if not content:
        return False

    if len(content) / len(tokens) < 0.5:
        return False

    if any(t.is_stop for t in content):
        return False

    # A phrase made entirely of digits and punctuation is a figure reference.
    if not any(t.is_alpha for t in tokens):
        return False

    lowered = text.lower()
    if lowered in DOMAIN_FILLER:
        return False

    # Reject a phrase whose every word is filler ("type of form").
    words = [w for w in lowered.split() if w not in ("of", "and", "in", "for")]
    if words and all(w in DOMAIN_FILLER for w in words):
        return False

    return True


def _display_form(span) -> str:
    """How a keyword is shown to a reader.

    The first surface form encountered, not the lemma: "Operating System" reads
    as a term, "operating system" reads as a lemma dump. Lowercasing only what
    is not already capitalised keeps a proper noun proper.

    **Leading determiners are stripped**, which they were not in the first
    version — the keyword list came out as "A stack", "A queue", "An algorithm",
    "A way", "A computer". A keyword list is a list of terms; an article is not
    part of a term, and leaving it in also splits one concept across two entries
    for no reason ("A stack" and "The stack" score separately).
    """
    tokens = [t for t in span if not t.is_punct and not t.is_space]

    while tokens and tokens[0].lower_ in LEADING_DETERMINERS:
        tokens = tokens[1:]

    text = " ".join(t.text for t in tokens)
    text = " ".join(text.split())

    if not text:
        return text
    if text[0].isupper():
        return text
    return text[0].upper() + text[1:]


def extract(doc, limit: int = 20) -> list[Keyword]:
    """
    The document's keywords and keyphrases, best first.

    Counts are gathered across the whole document before scoring, because IDF is
    a property of the term and the corpus, not of any one occurrence of it.
    """
    counts: Counter[str] = Counter()
    displays: dict[str, str] = {}
    positions: dict[str, int] = {}

    token_count = max(len([t for t in doc.doc if not t.is_space and not t.is_punct]), 1)

    for span in doc.doc.noun_chunks:
        key = _phrase_key(span)
        text = " ".join(t.text for t in span if not t.is_punct and not t.is_space).strip()

        if not _is_candidate(span, text):
            continue

        counts[key] += 1
        displays.setdefault(key, _display_form(span))
        positions.setdefault(key, span.start)

    candidates = list(counts.items())
    if not candidates:
        return []

    # Longest raw count, for normalising TF so a long document does not simply
    # produce larger scores than a short one.
    max_count = max(counts.values())

    scored: list[Keyword] = []

    for key, count in candidates:
        # Sublinear TF, normalised. `1 + log(count)` is the standard dampening;
        # dividing by the document's largest count keeps the scale comparable
        # between a paragraph and a chapter.
        tf = (1 + math.log(count)) / (1 + math.log(max_count))
        idf = corpus.idf(key)

        # A phrase is more specific than its words in isolation, so it gets a
        # bounded bonus: 1.0 for one word, 1.1 for two, 1.15 for three. Not
        # proportional to length, which would rank the longest chunk first.
        words = len(key.split())
        phrase_bonus = {1: 1.0, 2: 1.1, 3: 1.15}.get(words, 1.15)

        # A mild early-position bonus. Terms introduced near the start of a
        # text are more often what it is about — a heuristic from keyphrase
        # extraction rather than a law, and deliberately small (at most 10%) so
        # it reorders near-ties without overturning a clear frequency signal.
        position = positions.get(key, token_count)
        position_bonus = 1.0 + 0.1 * (1.0 - min(position / token_count, 1.0))

        score = tf * idf * phrase_bonus * position_bonus

        scored.append(
            Keyword(
                key=key,
                term=displays[key],
                score=round(score, 4),
                count=count,
            )
        )

    scored.sort(key=lambda k: (-k.score, k.term.lower()))
    return scored[:limit]


def vocabulary(doc) -> set[str]:
    """
    The lemmatised term set a document contributes to the corpus.

    Separate from `extract` because the corpus wants *everything* the document
    contains — the whole point of IDF is knowing which terms are common, and
    common terms are precisely the ones `extract` throws away.
    """
    return {
        token.lemma_.lower()
        for token in doc.doc
        if token.is_alpha
        and not token.is_stop
        and len(token.text) >= MIN_TERM_CHARS
        and token.pos_ in CONTENT_POS
    }
