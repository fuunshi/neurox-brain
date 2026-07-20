"""
The spaCy pipeline, loaded once, and the document wrapper everything else uses.

**Why one shared `nlp` object.** Loading a spaCy model costs about a second and
100–300MB of resident memory. Loading one per request would make the service
unusable; loading one per process and sharing it is the intended arrangement.
The object is read-only after construction, so sharing is safe.

**Why the parse is kept rather than discarded.** Every extractor in this package
needs a *different* view of the same analysis — the definitional extractor wants
dependency arcs, the cloze generator wants noun chunks and lemmas, the distractor
generator wants entities and POS tags, the summariser wants sentence vectors.
Running `nlp()` five times over the same text would parse it five times and
produce five documents that could disagree. So the document is parsed once, and
`Document` is the object passed between stages.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from ..config import settings

logger = logging.getLogger(__name__)

# A sentence shorter than this is almost always a heading, a fragment of a
# numbered list, or an artefact of a PDF's layout. None of them can carry a
# definition, and all of them produce embarrassing cards ("2.3.1. " -> "?").
MIN_SENTENCE_CHARS = 25

# Above this, a "sentence" is a parse failure — usually a PDF that lost its
# punctuation, so the whole page arrived as one span. Treating it as a sentence
# produces one enormous card.
MAX_SENTENCE_CHARS = 600

# A sentence must contain at least this many alphabetic tokens to be considered.
MIN_SENTENCE_WORDS = 5

_WHITESPACE = re.compile(r"\s+")
# Bullet and numbering artefacts that survive PDF extraction.
_LIST_MARKER = re.compile(r"^\s*(?:[-•*·]|\(?\d{1,3}[.)])\s*")


@dataclass(frozen=True)
class Document:
    """
    One parsed text, plus the derived views every extractor shares.

    Frozen because nothing downstream should mutate an analysis in place — a
    stage that edits the document it was handed makes every later stage depend
    on the order stages ran in.
    """

    text: str
    # The spaCy Doc. Typed loosely because spaCy's own types are generic over
    # the model's component set and pinning them here would mean importing
    # `spacy.tokens` in every module that touches a document.
    doc: object
    sentences: list[object]
    title: str | None

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_tokens(self) -> int:
        return sum(1 for token in self.doc if not token.is_space)  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def get_nlp():
    """
    The shared pipeline.

    `lru_cache` rather than a module global so the first caller pays for the
    load and every later one gets the same object without a lock — CPython's
    caches are thread-safe for this use, which a hand-rolled `if _nlp is None`
    check would not be.
    """
    import spacy

    started = time.perf_counter()

    try:
        nlp = spacy.load(settings.model)
    except OSError as exc:  # model not downloaded
        raise RuntimeError(
            f"spaCy model {settings.model!r} is not installed. "
            f"Run: python -m spacy download {settings.model}"
        ) from exc

    logger.info(
        "Loaded spaCy model %s in %.2fs (vectors: %s)",
        settings.model,
        time.perf_counter() - started,
        nlp.vocab.vectors.shape[0] > 0,
    )
    return nlp


def is_model_loaded() -> bool:
    """Whether the model is already in memory, without triggering a load.

    Used by `/health` so a liveness probe does not accidentally load a model —
    a probe that costs 300MB is not a probe.
    """
    return get_nlp.cache_info().currsize > 0


# Guards `get_nlp()` against two requests racing on first load. `lru_cache` is
# safe, but the *first* call is slow and both callers would otherwise sit inside
# `spacy.load` at once, doubling peak memory.
_LOAD_LOCK = threading.Lock()


def _nlp_locked():
    if is_model_loaded():
        return get_nlp()
    with _LOAD_LOCK:
        return get_nlp()


def normalise(text: str) -> str:
    """
    Tidy the whitespace a PDF leaves behind, and nothing else.

    Deliberately not lowercasing, not stripping punctuation, not removing stop
    words: spaCy's parser is trained on ordinary prose, and text that has been
    "cleaned" into a bag of words parses worse and yields worse sentences. The
    only transformations here are the ones that repair extraction damage.
    """
    # Non-breaking spaces and the various Unicode spaces a PDF emits.
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    # A soft hyphen is a line-break artefact, never a real character.
    text = text.replace("­", "")
    # Collapse runs of whitespace, but keep paragraph breaks — they are the
    # strongest signal that a topic changed.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_usable_sentence(span) -> bool:
    """
    Whether a span is worth extracting from.

    The filters are all false-positive controls, and each one is here because of
    something that actually appeared in a test run:

    - Too short: headings, list fragments, page numbers, and the "Figure 3."
      captions that survive extraction without their figure.
    - Too long: a PDF that lost its punctuation, where the "sentence" is a page.
    - Too few words: a line of table-of-contents dots.
    - Too few verbs: a run of nouns is a heading or a table row. A definition
      needs a verb to be a definition, so a verbless span cannot be one.
    """
    text = span.text.strip()

    if len(text) < MIN_SENTENCE_CHARS or len(text) > MAX_SENTENCE_CHARS:
        return False

    content_tokens = [t for t in span if not t.is_space and not t.is_punct]
    if len(content_tokens) < MIN_SENTENCE_WORDS:
        return False

    if not any(t.pos_ in ("VERB", "AUX") for t in content_tokens):
        return False

    # A sentence that is mostly digits is a data row, not prose.
    alpha = sum(1 for t in content_tokens if t.is_alpha)
    if alpha / max(len(content_tokens), 1) < 0.5:
        return False

    return True


def strip_list_marker(text: str) -> str:
    """Remove a leading bullet or number from a sentence.

    Done at display time rather than before parsing: removing it first changes
    the tokenisation and can move the parse, and the marker is not the problem —
    showing it in a flashcard is.
    """
    return _LIST_MARKER.sub("", text).strip()


def collapse_whitespace(text: str) -> str:
    """
    Flatten a fragment to a single line, for anything a reader will see.

    `normalise` deliberately preserves the line breaks a PDF left behind,
    because paragraph structure is real information while the text is being
    read. It stops being information the moment a fragment is lifted out of the
    document: a card front that reads

        "The push operation adds an element to the top of the stack, and _____
         removes the element at the top."

    has a newline in the middle of it because that is where the source PDF
    wrapped, and rendering that in a card is how a generated card announces
    that it is generated. Collapsing happens at the display boundary, not in
    `normalise`, so the parser still sees the structure.
    """
    return " ".join(text.split())


def parse(text: str, title: str | None = None) -> Document:
    """
    Parse once, and hand back everything the extractors need.

    Sentence segmentation comes from the dependency parser, which is why a
    model with a parser is required: splitting on `.` alone breaks on
    abbreviations, decimals and the "e.g." that academic prose is full of.
    """
    nlp = _nlp_locked()
    cleaned = normalise(text)

    started = time.perf_counter()
    doc = nlp(cleaned)
    elapsed = time.perf_counter() - started

    sentences = [span for span in doc.sents if is_usable_sentence(span)]

    logger.debug(
        "Parsed %d chars into %d sentences (%d usable) in %.2fs",
        len(cleaned),
        len(list(doc.sents)),
        len(sentences),
        elapsed,
    )

    return Document(text=cleaned, doc=doc, sentences=sentences, title=title)


def whitespace_of(text: str) -> float:
    """Fraction of a string that is whitespace.

    A crude guard against the binary garbage that occasionally survives a bad
    PDF extraction and parses into a document full of meaningless tokens.
    """
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isspace()) / len(text)
