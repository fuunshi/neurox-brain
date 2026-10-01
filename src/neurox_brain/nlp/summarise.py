"""
Extractive summarisation.

**Extractive, not abstractive, and that is a deliberate ceiling.** An extractive
summary is made of sentences the author actually wrote, in their own words. An
abstractive one — the kind a language model produces — reads better and can say
something the source never said. For study material, that difference decides it:
a summary shown to a student revising for an exam must be traceable back to the
text, and every sentence here can be highlighted in the chapter it came from.

The cost is real and worth stating: the result is not as smooth as a written
summary, and the `score` on each sentence is what lets the interface say *why*
a sentence was chosen rather than presenting a selection as if it were authored.

**The method.** TextRank over the sentences — see `textrank.py` for the
algorithm and its reference. The pipeline here is selection *and* presentation:

1. Rank every sentence.
2. Take the top N.
3. Put them back into reading order.
4. Strip list markers and rejoin.

Step 3 is the one that is easy to get wrong and obvious when you do: a summary
printed in rank order reads as a shuffled document, and the reader has no way to
tell that the sentences are in the right set but the wrong sequence.
"""

from __future__ import annotations

from . import textrank
from .pipeline import collapse_whitespace, strip_list_marker



def summarise(doc, max_sentences: int = 5, ranked=None) -> list[dict]:
    """
    The N most representative sentences, in the order they appear in the text.

    Returns plain dicts matching `schemas.SummarySentence`; the route layer
    validates them. Building pydantic models here would make the algorithm
    package depend on the wire format, which is the wrong direction for a change
    to travel — a new response field should not be able to alter what gets
    summarised.

    `ranked` lets a caller that has already ranked the document (see
    `salience.build`) pass the result in rather than paying for it twice. It
    must be the output of `textrank.rank_sentences` for these same sentences —
    the point of taking it is to agree on one ranking, not to allow a different
    one.
    """
    if max_sentences <= 0 or not doc.sentences:
        return []

    if ranked is None:
        ranked = textrank.rank_sentences(doc.sentences)

    if not ranked:
        return []

    # A summary of everything is not a summary. When the text has fewer usable
    # sentences than were asked for, returning them all in reading order is
    # better than returning an empty list — and better than padding.
    selected = ranked[:max_sentences]

    # Back to document order. The scores travel with the sentences so a caller
    # can still see how each ranked.
    selected.sort(key=lambda sentence: sentence.index)

    return [
        {
            # Flattened: the sentence carries the source document's line breaks,
            # and a summary rendered with a newline in the middle of a clause
            # looks like a formatting bug rather than an extracted sentence.
            "text": collapse_whitespace(strip_list_marker(sentence.text)),
            "score": round(sentence.score, 6),
            "index": sentence.index,
        }
        for sentence in selected
    ]


def headings(doc, limit: int = 12) -> list[str]:
    """
    Section headings, for an outline rather than a summary.

    Headings are found by shape, not by markup, because the input is plain text
    from a PDF as often as it is markdown. A heading is a short line that does
    not end in a full stop, is not a question, and is not a list item — which is
    a heuristic with obvious failure cases, so it is offered as an outline hint
    and never used to cut the document up.

    Kept here rather than in a module of its own because it shares the property
    that makes it useful: both answers are "what is this text about", at
    different granularities.
    """
    found: list[str] = []

    for line in doc.text.split("\n"):
        candidate = line.strip()

        if not 3 < len(candidate) < 80:
            continue

        if candidate.endswith((".", "!", "?", ":", ";")):
            continue

        # Markdown headings are unambiguous, so accept them first and stop
        # second-guessing.
        if candidate.startswith("#"):
            found.append(candidate.lstrip("# ").strip())
            continue

        words = candidate.split()
        if not 1 < len(words) <= 10:
            continue

        # Title case or all caps, which is how most documents mark a heading.
        capitalised = sum(1 for word in words if word[:1].isupper())
        if capitalised / len(words) < 0.6:
            continue

        found.append(candidate)

    return found[:limit]
