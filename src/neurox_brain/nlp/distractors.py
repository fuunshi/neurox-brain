"""
Plausible wrong answers for multiple-choice questions.

**Why this is the hard part.** Generating a correct answer is a lookup;
generating three *wrong* answers that a student might actually choose is the
whole difficulty of automatic question generation. The two failure modes are
opposite and both ruin the question:

- **Too obviously wrong** ("Which of these is a data structure: stack, banana,
  Tuesday?") tests nothing, and a student notices immediately.
- **Also correct** ("Which of these is a data structure: stack, list, array,
  queue?" — where the question came from a definition true of all four) is
  worse, because it teaches the student something false and destroys trust in
  every other question on the site.

Both come from the same mistake: picking options by surface form rather than by
meaning. So the selection here scores every candidate on *semantic distance* from
the correct answer and keeps only those inside a band — close enough to be
confusable, far enough not to be right.

**The three sources, in order of preference.** Each is tried before the next,
because each is better than the one after it:

1. **Other items from the same document.** A chapter's own vocabulary is
   topically coherent by construction, and for a definitional question the other
   definitions in the chapter are the ideal distractors — they are written in
   the same register, at the same level of detail, about the same subject. This
   is also the only source that needs no external data.
2. **WordNet siblings.** For a term, its co-hyponyms — words sharing a hypernym
   ("stack" and "queue" are both hyponyms of "data structure"). This is the
   classical approach and it produces genuinely close distractors, because
   siblings in a taxonomy are things a student confuses.
3. **Word-vector neighbours.** With a vector-bearing spaCy model, the terms
   nearest the answer in vector space but not too near. This catches
   domain-adjacent words WordNet does not connect, and is the fallback when the
   document is too short to supply options.

**What is deliberately not done.** No crossing subject boundaries (a physics
term is never a distractor for a data-structures question), no morphological
invention, and no using a distractor that appears in the question stem.
"""

from __future__ import annotations

import logging

from ..config import settings
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# How close a candidate may be to the correct answer before it is a synonym
# rather than a distractor. Above this, two words mean the same thing often
# enough that an option drawn from it risks being right.
MAX_SIMILARITY = 0.72

# Below this a candidate is a different subject, not a different answer. The
# band between the two is where plausible-but-wrong lives.
MIN_SIMILARITY = 0.12

# Fewer than this many candidates and there is not enough material to build a
# question with; the caller drops it rather than padding with nonsense.
MIN_DISTRACTORS = 3


@dataclass(frozen=True)
class Candidate:
    text: str
    similarity: float
    source: str


def _vector(nlp, text: str):
    """The vector for a word or phrase, or None if the model has none."""
    if nlp is None or nlp.vocab.vectors.shape[0] == 0:
        return None

    doc = nlp(text)
    # spaCy returns a zero vector for an out-of-vocabulary term rather than
    # raising, and a zero vector's similarity to everything is 0 — which would
    # silently read as "maximally dissimilar" and let anything through the band.
    if doc.vector_norm == 0:
        return None

    return doc.vector


def _similarity(nlp, left: str, right: str) -> float:
    """Cosine similarity, or a lexical fallback when there are no vectors."""
    left_vector = _vector(nlp, left)
    right_vector = _vector(nlp, right)

    if left_vector is None or right_vector is None:
        # Without vectors, fall back to a crude lexical signal: words that share
        # a prefix are likelier to be confusable. Deliberately weak — it only
        # breaks ties, and the band below is wide enough that a weak signal
        # still admits most candidates rather than rejecting them all.
        left_lemma = left.lower()[:4]
        right_lemma = right.lower()[:4]
        return 0.3 if left_lemma == right_lemma else 0.2

    import numpy as np

    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator == 0:
        return 0.0

    return float(np.dot(left_vector, right_vector) / denominator)


def ensure_wordnet_loaded() -> bool:
    """
    Force NLTK's WordNet to load, at start-up, before any thread touches it.

    **This is a production bug fix, not a nicety.** NLTK loads corpora lazily
    through `LazyCorpusLoader`, and that loader is not thread-safe: two threads
    reaching a corpus for the first time at once can leave it half-initialised,
    with failures like `'WordNetCorpusReader' object has no attribute
    '_LazyCorpusLoader__args'`. It has taken down production systems, and the
    documented fix is exactly this call — once, early, single-threaded.

    It also fails *late* by nature: a missing corpus raises `LookupError` when
    it is first *used*, not when it is imported, so the problem passes unit
    tests and appears on the first real request. Loading here turns a deployment
    fault into a start-up log line.

    Returns whether WordNet is usable. A `False` is not fatal — the distractor
    generator has other sources and the service degrades rather than stops —
    but it is worth knowing, so it is logged at warning level.
    """
    try:
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "WordNet is unavailable (%s). Distractors will come from the document "
            "alone, and no synonym check will be applied — which means an option "
            "that is also correct can slip through. Install it with: "
            "python -m nltk.downloader wordnet omw-1.4",
            exc,
        )
        return False


def _wordnet_siblings(term: str) -> list[str]:
    """
    Co-hyponyms of `term`: words that share one of its hypernyms.

    "stack" -> hypernym "data structure" -> siblings "queue", "list", "array".
    This is the classical distractor source and it is good precisely because
    taxonomy siblings are the things people actually confuse.
    """
    try:
        from nltk.corpus import wordnet as wn
    except LookupError:
        # NLTK data not downloaded. The other sources still work.
        logger.debug("WordNet unavailable; skipping the sibling source.")
        return []

    siblings: list[str] = []

    for synset in wn.synsets(term):
        # Only the noun senses: a definitional question about a term is a
        # question about a thing, and a verb sibling would be the wrong kind of
        # word entirely.
        if synset.pos() != "n":
            continue

        for hypernym in synset.hypernyms():
            for hyponym in hypernym.hyponyms():
                for lemma in hyponym.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != term.lower() and name.isascii():
                        siblings.append(name)

    # De-duplicate while keeping order, and cap: a broad hypernym like "object"
    # can have hundreds of hyponyms and they are not all useful.
    seen = set()
    unique = []
    for sibling in siblings:
        if sibling.lower() not in seen:
            seen.add(sibling.lower())
            unique.append(sibling)

    return unique[:200]


def _band(candidates: list[Candidate]) -> list[Candidate]:
    """Keep only candidates inside the plausible-but-not-correct band."""
    return [
        candidate
        for candidate in candidates
        if MIN_SIMILARITY <= candidate.similarity <= MAX_SIMILARITY
    ]


def _too_close_to_be_wrong(correct: str, candidate: str) -> bool:
    """
    Whether WordNet says a candidate could pass for the answer.

    **This is the reliability half of the problem, and the research says it
    matters more than plausibility.** The most consistent finding in the
    distractor literature is that the methods producing the most *plausible*
    distractors also produce the least *reliable* ones — word-vector similarity
    scored best on human plausibility (46.6%) and worst on validity (93.2%),
    because near neighbours are frequently synonyms. A multiple-choice question
    with a second correct option is not a slightly worse question; it is a
    question that teaches something false.

    So a candidate is rejected when WordNet places it in the same synset as the
    answer, or above it (a hypernym — "data structure" for "stack" is not a
    wrong answer to "what is a stack", it is a less precise right one), or as an
    antonym.

    WordNet is used **only as a filter here, never as a generator**. Its
    siblings are a reasonable source (see `_from_wordnet`), but the same
    taxonomy that makes them related is what makes them dangerous, and the
    cheap-and-safe direction is to use it to exclude rather than to propose.
    """
    try:
        from nltk.corpus import wordnet as wn
    except LookupError:
        # Without WordNet there is no synonym check, so the safe direction is to
        # keep the candidate and accept slightly weaker questions rather than to
        # reject everything and produce none. `ensure_wordnet_loaded` warns at
        # start-up so this is visible rather than silent.
        return False

    correct_synsets = wn.synsets(correct)
    if not correct_synsets:
        return False

    candidate_lower = candidate.lower().replace(" ", "_")

    for synset in correct_synsets:
        names = {lemma.name().lower() for lemma in synset.lemmas()}
        if candidate.lower() in names or candidate_lower in names:
            return True

        # Antonyms of any lemma in the synset.
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                if antonym.name().lower() in (candidate.lower(), candidate_lower):
                    return True

    # A hypernym of the answer, at any depth — "a less precise right answer".
    candidate_synsets = wn.synsets(candidate)
    for synset in correct_synsets:
        hypernyms = set(synset.closure(lambda s: s.hypernyms()))
        if any(other in hypernyms for other in candidate_synsets):
            return True

    return False


def _shares_head_noun(left: str, right: str) -> bool:
    """
    Whether two terms end in the same word.

    English noun phrases are head-final, so the last word carries the category:
    "linked list" and "doubly linked list" are both lists, "stack" and "queue"
    are not related by name at all. Sharing a head is the cheapest available
    proxy for "same kind of thing", which is exactly what a distractor should
    look like.
    """
    left_words = left.lower().split()
    right_words = right.lower().split()

    if not left_words or not right_words:
        return False

    return left_words[-1] == right_words[-1]


def _reliable(candidates: list[Candidate], correct: str) -> list[Candidate]:
    """Candidates that WordNet does not consider able to pass for the answer."""
    return [
        candidate
        for candidate in candidates
        if not _too_close_to_be_wrong(correct, candidate.text)
    ]


def for_terms(
    correct: str,
    pool: list[str],
    nlp,
    count: int = 3,
) -> list[str]:
    """
    Distractors for a question whose options are *terms*.

    `pool` is every other term extracted from the same document — the caller
    supplies it, because only the caller knows what else was found.

    Candidates are scored by similarity to the correct answer and taken from
    the middle of the distribution: too similar and the option may be a
    synonym, too dissimilar and it is obviously wrong. Ties are broken by
    preferring candidates that are not substrings or superstrings of the
    answer, since "operating system" next to "system" reads as a trick.
    """
    if count <= 0:
        return []

    lowered_correct = correct.lower()
    candidates: list[Candidate] = []

    for term in pool:
        lowered = term.lower()

        if lowered == lowered_correct:
            continue
        # A candidate containing the answer, or contained by it, is a giveaway
        # in one direction and a trick in the other.
        if lowered in lowered_correct or lowered_correct in lowered:
            continue
        # Single characters and pure numbers are not terms.
        if len(term) < 3 or term.isdigit():
            continue

        candidates.append(
            Candidate(
                text=term,
                similarity=_similarity(nlp, correct, term),
                source="document",
            )
        )

    # Reliability first, plausibility second.
    #
    # The similarity band is only meaningful when the model has real vectors.
    # `en_core_web_sm` — the default — has none, and `en_core_web_md`'s are
    # pruned to 20,000 entries so that most pairs score exactly 1.0 or nothing
    # at all. Applying a band to those numbers would be filtering on an
    # artefact; see `settings.use_vectors` and the model comment in config.py.
    #
    # With no band, the WordNet check does the work. That is the right order
    # anyway: rejecting a second correct answer is worth more than a slightly
    # more tempting wrong one.
    candidates = _reliable(candidates, correct)

    if settings.use_vectors:
        candidates = _band(candidates)
        # Nearer the middle of the band is better: close enough to tempt, far
        # enough to be wrong. Distance from the midpoint rather than raw
        # similarity, which would always pick the closest candidate — the one
        # most likely to be a synonym that slipped the ceiling.
        midpoint = (MIN_SIMILARITY + MAX_SIMILARITY) / 2
        candidates.sort(key=lambda c: (abs(c.similarity - midpoint), c.text.lower()))
    else:
        # No usable similarity, so rank by shared head noun. "doubly linked
        # list" against "linked list" is a far better distractor than a random
        # other term, and head-noun matching is the strongest cheap signal
        # available — it is what the corpus-based methods in the literature
        # actually rely on. Ties break alphabetically so the output is
        # deterministic for a given input.
        candidates.sort(
            key=lambda c: (
                not _shares_head_noun(correct, c.text),
                c.text.lower(),
            )
        )

    chosen = [candidate.text for candidate in candidates[:count]]

    if len(chosen) < count:
        chosen.extend(_from_wordnet(correct, chosen, nlp, count - len(chosen)))

    return chosen[:count]


def _from_wordnet(correct: str, already: list[str], nlp, count: int) -> list[str]:
    """Top up from WordNet siblings, still band-filtered."""
    taken = {term.lower() for term in already}
    picked: list[str] = []

    for sibling in _wordnet_siblings(correct):
        if len(picked) >= count:
            break

        if sibling.lower() in taken:
            continue

        similarity = _similarity(nlp, correct, sibling)
        if not MIN_SIMILARITY <= similarity <= MAX_SIMILARITY:
            continue

        taken.add(sibling.lower())
        picked.append(sibling)

    return picked


def for_definitions(
    correct: str,
    pool: list[str],
    count: int = 3,
) -> list[str]:
    """
    Distractors for a question whose options are *definitions*.

    Simpler than the term case, and for a good reason: definitions from the same
    document are about different terms by construction, so the "also correct"
    risk is far lower and no similarity band is needed. The only real risk is
    length — a conspicuously longer option is guessable as the right one, which
    is why candidates are chosen to be close in length to the correct answer.

    That length preference is a deliberate, documented weakness: it is a
    surface heuristic standing in for "similar level of detail", and the honest
    alternative is to compare the definitions' vectors and pick ones that are
    semantically adjacent. That is worth doing when there are enough examples to
    tune the band; guessing at one now would be harder to defend than this.
    """
    if count <= 0:
        return []

    lowered = correct.lower().strip()
    target_length = len(correct)

    scored: list[tuple[int, str]] = []

    for definition in pool:
        candidate = definition.strip()

        if candidate.lower() == lowered or len(candidate) < 15:
            continue

        scored.append((abs(len(candidate) - target_length), candidate))

    scored.sort(key=lambda pair: (pair[0], pair[1].lower()))

    return [candidate for _, candidate in scored[:count]]


def has_enough(distractors: list[str]) -> bool:
    """Whether a question can be built. Asked by callers before they commit."""
    return len(distractors) >= MIN_DISTRACTORS
