"""
Finding definitions in prose, and turning them into flashcards.

This is the core of card generation, so it is worth being precise about the
approach and why it is not a regular expression.

**Why not regex.** The obvious implementation is `r"(\\w+) is (?:a|an|the)? (.+)"`.
It works on the two sentences you try it on and then produces:

    "This is a problem that arises when..."      -> term "This"
    "The algorithm is fast and uses O(n) space"  -> definition "fast and uses..."
    "A stack is a linear data structure"         -> correct, by luck

The failure is that "is" carries no information on its own. What distinguishes a
definition is *grammatical structure*: a subject, a copula, and a complement that
is a noun phrase describing that subject. That structure is exactly what a
dependency parse gives you, and it is why this module uses one.

**The four patterns.** Each is a different way English expresses "X means Y", and
each is a dependency shape rather than a word list:

1. **Copular** — "A stack *is* a linear data structure."
   `nsubj` + copula + `attr`. The most common and the most reliable.
2. **Cue phrase** — "A stack *is defined as* a linear data structure."
   An explicit signal, so it scores highest. Also covers "refers to",
   "is called", "denotes", "means", "is known as".
3. **Appositive** — "A stack, *a linear data structure*, stores items in order."
   `appos`, which is how academic prose defines a term mid-sentence without
   making it the subject.
4. **Hypernym** — "A stack *is a type of* linear data structure."
   A copular with a hypernym phrase; kept separate because the answer reads
   differently ("a type of X" is a worse card back than X itself) and gets
   rewritten.

**What this deliberately does not do.** It does not use a definition classifier
or a trained model. Those exist and are better; they are also a model file, a
training pipeline and a few hundred megabytes, and this service is meant to run
on a small VPS. The scoring below is the cheap version of the same judgement,
and the `confidence` it produces is what lets a reader sort the good drafts from
the bad ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pipeline import collapse_whitespace

# Cue phrases, as lemmatised token sequences. Matched against lemmas rather than
# raw text so "refers to", "referred to" and "referring to" are one pattern.
CUE_PHRASES: list[tuple[tuple[str, ...], str]] = [
    (("define", "as"), "is defined as"),
    (("refer", "to"), "refers to"),
    (("know", "as"), "is known as"),
    (("call",), "is called"),
    (("mean",), "means"),
    (("denote",), "denotes"),
    (("describe", "as"), "is described as"),
    (("term", "as"), "is termed"),
    (("represent",), "represents"),
]

# Copulas. `be` covers is/are/was/were/been; the others are the verbs that
# behave like it.
COPULAS = {"be", "become", "remain", "seem", "appear"}

# Phrases that signal the complement is a hypernym rather than a definition.
HYPERNYM_MARKERS = ("type of", "kind of", "form of", "class of", "subset of",
                    "example of", "instance of")

# Leading determiners, stripped from a noun phrase used as a card front.
LEADING_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}

# Words that make a "term" meaningless. A card asking "What is Another?" is
# worse than no card.
BAD_TERMS = {
    "this", "that", "these", "those", "it", "they", "there", "here", "which",
    "who", "what", "one", "ones", "such", "another", "other", "others", "he",
    "she", "we", "you", "i", "them", "him", "her", "us", "some", "many",
    "most", "all", "both", "each", "either", "neither", "few", "several",
}

# A term is a noun phrase, and noun phrases in these texts are short. Anything
# longer is a clause that the parser attached oddly.
MAX_TERM_TOKENS = 6
MAX_TERM_CHARS = 60

# Below this the "definition" is too thin to be worth a card. Above it, it is a
# paragraph and the card is unreadable.
MIN_DEFINITION_CHARS = 12
MAX_DEFINITION_CHARS = 320


@dataclass(frozen=True)
class Definition:
    """One extracted term/definition pair, before it becomes a card."""

    term: str
    definition: str
    evidence: str
    pattern: str
    confidence: float


# Dependency labels marking a token as belonging to a *different* role than the
# phrase being built. Each is excluded **together with everything beneath it**,
# because dropping a head while keeping its children strands them: excluding an
# appositive but not its determiner pulls the determiner back in, and the term
# comes out as "A binary search tree, a node-based".
#
#   mark  — a subordinating conjunction ("that", "which") introducing a clause
#           the reader does not need in a card back.
#   appos — a renaming. "My brother, a doctor, …" is a phrase about the
#           brother; the doctor is a second nominal, not a continuation of the
#           first. Before this was excluded, the appositive pattern produced a
#           *term* of "A binary search tree, a node-based tree" and the
#           plausibility filter then threw the whole card away.
#   cc    — the "and" itself. Left out of the kept set and recovered by the
#           enclosing range whenever the conjunct beside it is kept, which is
#           the behaviour that makes "vertices and edges" read correctly.
EXCLUDED_FROM_PHRASE = {"mark", "appos", "cc"}


def _conjunct_belongs_to_phrase(token) -> bool:
    """
    Whether a `conj` continues this phrase or opens a second clause.

    spaCy labels both with the same dependency, and the two are separable only
    by what the conjunct hangs off: "vertices and edges" coordinates two nouns
    *inside* the phrase, while "A stack is LIFO and a queue is FIFO" coordinates
    two clauses.

    In practice the parser attaches a conjoined clause to the verb, so it never
    lands in a noun phrase's subtree at all — verified on both sentences. This
    is the guard for the cases where that does not hold.
    """
    head = token.head

    # "a, b, and c": the outer conjunct has already been judged.
    if head.dep_ == "conj":
        return True

    return head.pos_ not in ("VERB", "AUX")


def _is_excluded(token, root) -> bool:
    """
    Whether a token, or anything it hangs from up to `root`, is excluded.

    Walking the ancestry rather than testing the token alone is what makes the
    exclusions work at depth. `root` is exempt because callers pass an
    appositive token as the root when they want *its* phrase — the root is the
    thing being expanded, whatever its own dependency happens to be.
    """
    # Compared by token index, not by object identity. spaCy does not promise
    # that two accesses to the same token return the same Python object, and
    # when they do not, `node is not root` is true forever and the walk never
    # terminates — which is exactly what the first version of this did.
    node = token
    while node.i != root.i:
        if node.dep_ in EXCLUDED_FROM_PHRASE:
            return True
        if node.dep_ == "conj" and not _conjunct_belongs_to_phrase(node):
            return True

        # The document root's `head` is itself. Without this the walk spins on
        # any token whose ancestry never reaches `root`, which is every token
        # whenever `root` is not one of its ancestors.
        parent = node.head
        if parent.i == node.i:
            return False
        node = parent

    return False


def _phrase_span(token):
    """
    The full noun phrase a token is the head of.

    **This is the fix for the bug that produced no cards at all.** The first
    version took `token.text` — the single head word. For "A stack is a linear
    data structure", the `attr` token is "structure", so the definition came out
    as the nine-character string "structure", which the minimum-length filter
    then correctly rejected as too thin to be a definition. Every copular
    definition in the document was extracted and then thrown away, and the only
    visible symptom was an empty card list.

    A definition is a *phrase*, not a word: "a linear data structure that
    follows the Last In First Out principle". The dependency subtree is exactly
    that phrase, and the exclusions above are what stop it being more.
    """
    tokens = [
        t
        for t in token.subtree
        if not _is_excluded(t, token) and not t.is_punct and not t.is_space
    ]

    if not tokens:
        return None

    indices = sorted(t.i for t in tokens)

    # The subtree can be non-contiguous when a token's children are separated by
    # something the exclusions removed. Taking the enclosing span and trimming
    # is the standard resolution, and trimming is what keeps "a linear data
    # structure" from becoming ", a linear data structure ,".
    #
    # **The right edge has to come from a token that was kept.** An exclusion
    # that falls *inside* the range is swallowed back in by the enclosing span,
    # which is why "vertices and edges" survives its excluded `and`. An
    # exclusion at the edge has nothing beyond it to be swallowed by, which is
    # how "…consisting of vertices and edges" became "…consisting of vertices":
    # `edges` is a `conj` and used to be dropped, `vertices` then became the
    # last kept token, and the span stopped there. `_conjunct_belongs_to_phrase`
    # is what keeps a nominal conjunct in the kept set.
    span = token.doc[indices[0] : indices[-1] + 1]

    while len(span) and (span[0].is_punct or span[0].is_space):
        span = span[1:]

    # Trailing punctuation is trimmed. Note this cannot repair a phrase whose
    # closing bracket was never in the subtree to begin with: spaCy attaches a
    # `)` to the sentence root rather than to the token it closes, so
    # "a time complexity of O(log n)" arrives here already missing it and the
    # card back reads "…of O(log n". Recorded in
    # `docs/algorithms/fact-extraction.md` as a known miss rather than guarded,
    # because a guard for it was written and never once fired.
    while len(span) and (span[-1].is_punct or span[-1].is_space):
        span = span[:-1]

    return span if len(span) else None


def _clean_term(span) -> str:
    """A noun phrase as a card front: no determiner, no trailing punctuation."""
    tokens = [t for t in span if not t.is_punct and not t.is_space]

    while tokens and tokens[0].lower_ in LEADING_DETERMINERS:
        tokens = tokens[1:]

    # A trailing preposition or conjunction means the span was cut badly.
    while tokens and tokens[-1].lower_ in {"and", "or", "of", "in", "with"}:
        tokens = tokens[:-1]

    return collapse_whitespace(" ".join(t.text for t in tokens))


def _clean_definition(span) -> str:
    """A definition as a card back: one line, sentence-cased, terminated."""
    text = collapse_whitespace(span.text)

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if text and not text.endswith((".", "!", "?")):
        text += "."

    return text


def _is_plausible_term(text: str) -> bool:
    """Whether a candidate front is worth asking about."""
    if not text or len(text) > MAX_TERM_CHARS:
        return False

    lowered = text.lower().strip()

    if lowered in BAD_TERMS:
        return False

    # A term made only of stop words, digits or punctuation is not a term.
    words = [w for w in re.split(r"[\s-]+", lowered) if w]
    if not words or all(w in BAD_TERMS for w in words):
        return False

    if len(words) > MAX_TERM_TOKENS:
        return False

    # Must contain at least one letter.
    return any(ch.isalpha() for ch in text)


def _confidence(
    pattern: str,
    term_span,
    definition: str,
    term_frequency: dict[str, int],
) -> float:
    """
    How much this looks like a real definition.

    A weighted sum rather than a learned model, and every weight is here for a
    reason that can be argued with:

    - **Pattern** is the largest term. An explicit cue phrase ("is defined as")
      is a stronger signal than a bare copula, and an appositive is weaker
      still because it is easy for the parser to attach to the wrong noun.
    - **Term shape** rewards a term that is a proper noun, a noun or a noun
      chunk — the things definitions are about. A term that is a pronoun or a
      bare verb is almost always a parse artefact.
    - **Document frequency** rewards terms that appear more than once. A term
      the author returns to is a term the document is about, and a card for it
      is more likely to be useful than one for a word mentioned in passing.
    - **Length ratio** penalises a "definition" shorter than its term, which is
      always wrong, and mildly rewards a definition that is a full phrase.

    The result is clamped to [0, 1] rather than normalised, because the absolute
    number is meaningful: 0.9 should mean "this is a definition" in every
    document, not "this is the best of a bad batch".
    """
    score = {"cue": 0.75, "copular": 0.6, "hypernym": 0.6, "appos": 0.45}.get(
        pattern, 0.4
    )

    root_pos = term_span.root.pos_
    if root_pos in ("PROPN", "NOUN"):
        score += 0.15
    elif root_pos in ("PRON", "DET"):
        score -= 0.25

    # A multi-word term that keeps its head noun is fine; one that lost it is
    # usually a clause.
    if len(term_span) == 1 and term_span.root.is_alpha:
        score += 0.05

    frequency = term_frequency.get(term_span.text.lower(), 1)
    if frequency >= 3:
        score += 0.1
    elif frequency == 2:
        score += 0.05

    term_len = len(term_span.text)
    if len(definition) > term_len:
        score += 0.05
    if len(definition) < term_len * 1.5:
        score -= 0.1

    return max(0.0, min(1.0, score))


# Modals make a copula non-committal. "A stack could be a data structure" is not
# a definition — the sentence is hedged, and a card asserting it would assert
# something the source did not.
MODALS = {"could", "would", "might", "may", "must", "should", "can", "shall"}

# Determiners marking the complement as *contrastive* rather than defining. "A
# stack is another data structure" points at a different thing; that sentence is
# comparing, not defining.
CONTRASTIVE = {"another", "other", "others", "different", "similar", "same"}

# Expletive subjects. "There are two kinds of stack" and "It is a linear
# structure" both parse with a subject that refers to nothing, or to something
# in an earlier sentence. Producing a card whose front is "There" or "It" is the
# visible symptom, and this package has no coreference resolution to fix it.
EXPLETIVE_SUBJECTS = {"there", "it", "this", "that", "these", "those", "here"}


def _passes_guards(head, subject, definition_span) -> bool:
    """
    The false-positive controls, each for a failure that was measured.

    Published work on definition extraction is unusually specific about *why*
    pattern-based systems fail, and every check here corresponds to a listed
    cause rather than to a hunch:

    - **Negation.** "Mediation is *not* a viable alternative to bankruptcy"
      matches the copular pattern exactly and asserts the opposite of a
      definition.
    - **Modals.** "A stack *could be* a data structure" is hedged; extracting it
      turns a possibility into a fact.
    - **Expletive subjects.** A subject that refers to nothing, or to something
      outside the sentence.
    - **Contrastive complements.** A comparison, not a definition, and the card
      would read as a definition of the wrong thing.
    - **Non-nominal heads.** A definition's complement is a noun phrase. When
      the parser attaches an adjective or a clause there instead, the result is
      a fragment.

    These do not close the gap between the pattern's raw precision — measured at
    roughly **0.16** on real course material, so about five extractions in six
    wrong — and something worth showing a reader. Nothing cheap does. They cut
    the worst of it; the pipeline's draft-and-review step handles the rest, and
    that is a product decision recorded in `docs/algorithms/README.md`.
    """
    # Negation anywhere in the copula's own subtree.
    if any(token.dep_ == "neg" for token in head.subtree):
        return False

    # A modal is normally an `aux` child of the copula rather than the head
    # itself: in "A stack could be a linear data structure" the head is `be`
    # and `could` hangs off it. Testing only the head's lemma — which is what
    # this did — missed the sentence the check above is named for, so hedged
    # definitions were extracted as facts.
    if head.lemma_.lower() in MODALS or any(
        child.dep_ in ("aux", "auxpass") and child.lemma_.lower() in MODALS
        for child in head.children
    ):
        return False

    # `subject` is the `nsubj` token itself, not a span — `_subject_of` returns
    # a single child of the verb.
    if subject.lower_ in EXPLETIVE_SUBJECTS and subject.pos_ == "PRON":
        return False

    first_token = next(
        (t for t in definition_span if not t.is_punct and not t.is_space), None
    )
    if first_token is not None and first_token.lower_ in CONTRASTIVE:
        return False

    return definition_span.root.pos_ in ("NOUN", "PROPN")


def _subject_of(head) -> object | None:
    """The `nsubj` child of a head token, if there is exactly one."""
    subjects = [t for t in head.children if t.dep_ in ("nsubj", "nsubjpass")]
    return subjects[0] if subjects else None


def _copular_patterns(sentence):
    """Patterns 1 and 4: `nsubj` + copula + `attr`/`acomp`, and hypernyms."""
    found = []

    for token in sentence:
        # The complement of a copula is tagged `attr` (noun phrase) or `acomp`.
        if token.dep_ not in ("attr", "acomp"):
            continue

        head = token.head
        if head.lemma_.lower() not in COPULAS:
            # `attr` hangs off a copula; anything else is a different structure.
            copulas = [c for c in head.children if c.dep_ == "cop"]
            if not copulas:
                continue

        subject = _subject_of(head)
        if subject is None:
            continue

        term_span = _phrase_span(subject)
        definition_span = _phrase_span(token)

        if term_span is None or definition_span is None:
            continue

        if not _passes_guards(head, subject, definition_span):
            continue

        pattern = "copular"
        if any(marker in definition_span.text.lower() for marker in HYPERNYM_MARKERS):
            pattern = "hypernym"

        found.append((term_span, definition_span, pattern))

    return found


def _cue_patterns(sentence):
    """Pattern 2: an explicit cue verb after the subject."""
    found = []

    for token in sentence:
        # The cue is a verb; its subject is the term.
        if token.pos_ not in ("VERB", "AUX"):
            continue

        if not any(_matches_lemmas(token, sequence) for sequence, _ in CUE_PHRASES):
            continue

        subject = _subject_of(token)
        if subject is None:
            continue

        # The definition is whatever the cue verb governs: an object, or the
        # object of the preposition that follows it ("refers *to* X").
        complement = _object_of(token)
        if complement is None:
            continue

        term_span = _phrase_span(subject)
        definition_span = _phrase_span(complement)

        if term_span is None or definition_span is None:
            continue

        if not _passes_guards(token, subject, definition_span):
            continue

        found.append((term_span, definition_span, "cue"))

    return found


def _matches_lemmas(token, sequence: tuple[str, ...]) -> bool:
    """Whether `token` begins `sequence` as consecutive lemmas in the sentence."""
    current = token
    for expected in sequence:
        if current is None or current.lemma_.lower() != expected:
            return False
        current = current.nbor(1) if current.i + 1 < len(current.doc) else None
    return True


def _object_of(verb):
    """The noun phrase a verb governs, following one preposition if needed."""
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "oprd"):
            return child

    for child in verb.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return grandchild

    return None


def _appositive_patterns(sentence):
    """Pattern 3: "X, a Y, ..." — an appositive."""
    found = []

    for token in sentence:
        if token.dep_ != "appos":
            continue

        head = token.head
        # An appositive defines the noun it hangs off.
        if head.pos_ not in ("NOUN", "PROPN"):
            continue

        term_span = _phrase_span(head)
        definition_span = _phrase_span(token)

        if term_span is None or definition_span is None:
            continue

        found.append((term_span, definition_span, "appos"))

    return found


def extract(doc, limit: int = 100) -> list[Definition]:
    """
    Pull definitions out of a parsed document, best first.

    Frequency is counted across the *whole document* before any single sentence
    is looked at, because the score for a term depends on how often the author
    used it — which is not knowable while processing sentence by sentence.
    """
    from collections import Counter

    term_frequency: dict[str, int] = Counter(
        chunk.text.lower() for chunk in doc.doc.noun_chunks
    )

    results: list[Definition] = []
    seen_terms: set[str] = set()

    for sentence in doc.sentences:
        candidates = [
            *_cue_patterns(sentence),
            *_copular_patterns(sentence),
            *_appositive_patterns(sentence),
        ]

        for term_span, definition_span, pattern in candidates:
            term = _clean_term(term_span)
            if not _is_plausible_term(term):
                continue

            definition = _clean_definition(definition_span)
            if not MIN_DEFINITION_CHARS <= len(definition) <= MAX_DEFINITION_CHARS:
                continue

            # One card per term. The first pattern to fire wins because the
            # candidate list is ordered cue -> copular -> appositive, which is
            # also confidence order.
            key = term.lower()
            if key in seen_terms:
                continue

            seen_terms.add(key)
            results.append(
                Definition(
                    term=term,
                    definition=definition,
                    evidence=collapse_whitespace(sentence.text),
                    pattern=pattern,
                    confidence=_confidence(pattern, term_span, definition, term_frequency),
                )
            )

    results.sort(key=lambda d: d.confidence, reverse=True)
    return results[:limit]
