"""
Cloze deletion: a sentence with its key term removed.

**Why cloze at all.** A definitional card asks "what is a stack?" and can be
answered vaguely and still feel right. A cloze asks the reader to produce the
exact word in a real sentence, which is a different and often harder kind of
recall — and it tests the term *in use* rather than in isolation. The two
complement each other, which is why both are generated.

**Choosing what to blank is the entire problem.** Blank the wrong word and the
card is either unanswerable or trivial:

- *"A _____ is a linear data structure"* — the definition gives it away. If the
  surrounding words only fit one term, the reader is not recalling, they are
  pattern-matching.
- *"A stack is a linear _____ structure"* — blanking "data" tests nothing; every
  reader supplies it without thinking about the subject.
- *"A stack is a _____ data structure"* — correct. The blank is the content
  word, the sentence still constrains the answer, and supplying it requires
  knowing what a stack is.

The rule that separates them: **blank the term the sentence exists to define.**

**This module used to do something else, and removing it was the single largest
quality decision in the package.** The first version also offered a generic
path — take the highest-scoring content word in *any* sentence and blank that —
so a document with few definitions would still produce practice material. That
path is precisely the one the literature warns against. Pino, Heilman &
Eskenazi measured it: among baseline cloze items chosen that way **34% admitted
several correct answers**, against 12% after switching to a constrained
selection strategy; Sumita et al. separately found ~6.5% of generated items were
unanswerable even by native speakers. A blank the reader cannot pin down is not
a hard question, it is a broken one, and it teaches the reader to distrust the
whole set.

So there is one path now, and it fires only when the text has already told us
what the sentence is about: the definiendum of a definitional sentence. A
document with no definitions produces no cloze cards. **Shipping fewer, sound
items is the intended behaviour rather than a shortfall** — the pipeline lands
generated cards as drafts for a reader to accept, so a small set of good
candidates is worth more than a large set of ambiguous ones.

**The honest limitation.** Judging whether a sentence uniquely determines a word
needs a model that can reason about it. Everything here is a proxy, and each is
listed in `_is_good_blank` with what it stands in for. The strongest cheap rule
available — Hill & Simha's, that a blanked term must not appear elsewhere in the
document so the reader cannot recover it from an earlier mention — is deliberately
*not* applied here, because for the definiendum of a definition the term
appearing repeatedly is the normal case rather than a leak.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline import collapse_whitespace

# The visible blank. Long enough to read as a gap rather than a typo.
BLANK = "_____"

# Blanking a word shorter than this tests nothing — it is a function word, or it
# is guessable from a single letter of context.
MIN_BLANK_CHARS = 4

# If the term is more than this fraction of the sentence, the sentence is
# essentially the term and the card has no context to constrain it.
MAX_BLANK_SHARE = 0.6


@dataclass(frozen=True)
class Cloze:
    """One cloze card."""

    front: str
    back: str
    evidence: str
    confidence: float


def _is_good_blank(span, sentence) -> bool:
    """
    Whether a span is worth blanking.

    Each check is a proxy for "the reader has to know the term to answer this",
    and each is here because of a concrete failure:

    - **Not a content word.** Blanking a determiner, preposition or auxiliary
      produces a card whose answer is grammar, not knowledge.
    - **Named entity.** Blanking a person's or product's name turns a study card
      into a trivia question, and the surrounding sentence rarely constrains it.
    - **Repeated in the sentence.** If the term appears twice, blanking one
      leaves the answer in plain sight.
    - **First token.** A sentence-initial blank loses the capital that would
      otherwise hint at a proper noun, but more importantly it is usually the
      subject of a sentence *about* something else, where the actual content is
      later.
    - **Too much of the sentence.** See `MAX_BLANK_SHARE`.
    """
    tokens = [t for t in span if not t.is_space and not t.is_punct]

    if not tokens:
        return False

    if not any(t.pos_ in ("NOUN", "PROPN", "ADJ") for t in tokens):
        return False

    if any(t.ent_type_ for t in tokens):
        return False

    # Measured *after* the article is stripped, which is what the reader has to
    # supply. The first version measured the span, so "a way" passed a
    # four-character minimum on the strength of its "a " and produced a card
    # whose answer was "way" — a word that carries no subject knowledge and is
    # guessable from the sentence around it.
    text = _strip_article(span.text.strip())
    if len(text) < MIN_BLANK_CHARS:
        return False

    lowered = text.lower()
    occurrences = sum(
        1
        for i in range(len(sentence) - len(tokens) + 1)
        if sentence[i : i + len(tokens)].text.lower() == lowered
    )
    if occurrences > 1:
        return False

    if span.start == sentence.start:
        return False

    content_tokens = [t for t in sentence if not t.is_space and not t.is_punct]
    if len(tokens) / max(len(content_tokens), 1) > MAX_BLANK_SHARE:
        return False

    return True


def _replace_first(sentence_text: str, target: str) -> str | None:
    """
    Replace the first occurrence of `target` with a blank.

    Done on the token sequence rather than with `str.replace`, so the match is
    the one the parser found and not an earlier substring that happens to look
    the same — "list" inside "listen" is the classic case, and replacing it
    produces a sentence that is no longer English.
    """
    import re

    pattern = re.compile(rf"(?<![\w-]){re.escape(target)}(?![\w-])", re.IGNORECASE)
    replaced, count = pattern.subn(BLANK, sentence_text, count=1)

    if count == 0:
        return None

    return replaced


def from_term(sentence, term_span, confidence: float = 0.7) -> Cloze | None:
    """
    A cloze for a known term — the definitional one, usually.

    Used when a definition has already been extracted, because the term in a
    definition is the thing the sentence exists to introduce.
    """
    if not _is_good_blank(term_span, sentence):
        return None

    # Flattened before blanking, not after: the sentence carries the source
    # document's line breaks, and replacing `term_span.text` in a string that
    # contains newlines can fail to match when the span straddles one.
    flattened = collapse_whitespace(sentence.text)
    front = _replace_first(flattened, collapse_whitespace(term_span.text))

    if front is None:
        return None

    return Cloze(
        front=front,
        # The article is stripped: "a stack" is not the answer, "stack" is.
        back=_strip_article(collapse_whitespace(term_span.text)),
        evidence=collapse_whitespace(sentence.text),
        confidence=confidence,
    )


def _strip_article(text: str) -> str:
    lowered = text.lower()
    for article in ("a ", "an ", "the "):
        if lowered.startswith(article):
            return text[len(article) :]
    return text
