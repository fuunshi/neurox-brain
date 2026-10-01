"""
Facts beyond definitions: properties, processes, comparisons, purpose.

**Why this module.** `definitions.py` answers one question — what *is* this
thing? — and a chapter is mostly other questions. What does it consist of, how
does it work, how is it different from the thing next to it, what is it for.
Those are what a student is actually examined on, and until now the service
produced none of them.

**One module, four families, and that is a deliberate shape.** The four need the
same span helpers, the same term-plausibility rules, the same salience gates and
the same confidence skeleton. Split into four modules they would either
duplicate roughly a hundred and fifty lines each or grow a shared base module —
and a base module plus four thin modules is one module with extra steps.

**Everything here is extractive.** No pattern invents text: a front is a source
span and a back is a source span, joined at most by a fixed connective. The
service has no language model and is not getting one, so the alternative to
"found in the sentence" is "made up", and a quiz answer that the source never
stated is worse than no card.

**The honest limitation, stated once.** These patterns are surface shapes, and
the published precision for hand-written extraction patterns on real course text
is around one in six. The salience gates in `selection.py` are what raise that
to something worth showing a reader, and they do it by dropping correct facts
that the document does not care about. That is a real cost, paid on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from .definitions import (
    _clean_definition,
    _clean_term,
    _phrase_span,
    _subject_of,
)

# ---------------------------------------------------------------- types --- #

#: Card families this module can produce. Mirrors the schema's `CardKind` minus
#: the two that `definitions.py` and `cloze.py` own.
PROPERTY = "PROPERTY"
PROCESS = "PROCESS"
COMPARISON = "COMPARISON"
PURPOSE = "PURPOSE"


@dataclass(frozen=True)
class Fact:
    """One extracted fact, before it becomes a card."""

    term: str
    back: str
    family: str
    pattern: str
    evidence: str
    sentence_index: int
    confidence: float

    #: The second participant, for a comparison. Unused by the other families.
    other: str | None = None


# ------------------------------------------------------------- vocabulary --- #

# Nouns whose presence as the object of "have"/"contain"/"include" marks a
# property statement rather than an incidental mention.
#
# **This list is the discipline of the whole family.** `have`, `contain` and
# `include` are among the most frequent verbs in expository prose, so firing on
# the shape alone would put a card on nearly every sentence in the document.
# Requiring the object to be one of these is what separates "A stack has a time
# complexity of O(1)" from "Algorithms have been studied for decades".
PROPERTY_NOUNS = frozenset(
    {
        "complexity",
        "size",
        "length",
        "capacity",
        "cost",
        "advantage",
        "disadvantage",
        "drawback",
        "benefit",
        "property",
        "characteristic",
        "feature",
        "requirement",
        "precondition",
        "invariant",
        "bound",
        "limit",
        "height",
        "depth",
        "degree",
        "weight",
        "value",
        "structure",
        "form",
        "representation",
        "purpose",
        "role",
        "behaviour",
        "behavior",
    }
)

#: Verbs that introduce what a thing is made of.
COMPOSITION_VERBS = frozenset({"consist", "comprise", "compose"})

#: Verbs that introduce something a thing has.
POSSESSION_VERBS = frozenset(
    {"have", "contain", "include", "hold", "offer", "provide", "require", "support"}
)

#: Verbs that introduce what a thing is for. Passive use only — see
#: `_purpose_patterns`.
PURPOSE_VERBS = frozenset({"use", "employ", "apply"})

#: Prepositions that carry a composition or a purpose.
OF = frozenset({"of", "in"})

#: Below this a fact is not shown to a reader at all, whatever its pattern.
#: Chosen so the closed-cue patterns clear it on their own and the open ones
#: clear it only when the document's own salience lifts them.
MIN_FACT_CONFIDENCE = 0.50


# ------------------------------------------------------------ extraction --- #


def _definition_ok(span) -> bool:
    """Whether a span is usable as a card back."""
    if span is None:
        return False

    text = span.text.strip()
    return 12 <= len(text) <= 320


def _property_patterns(sentence, index: int):
    """
    Facts about what a thing consists of, has, or is measured by.

    Four shapes, in descending order of how much they can be trusted:

    - `consist_of` — "The algorithm consists of three phases." A closed verb
      list and a closed preposition, and `consist` has almost no other use.
    - `property_of` — "The complexity of quicksort is O(n log n)." The subject
      is a property noun with an `of`, which makes the *inner* noun the term.
      Without this the copular pattern claims it and the card is fronted
      "Complexity of quicksort", which is a card nobody wants.
    - `be_of` — "The array is of fixed size." No `attr` or `acomp`, so the
      copular pattern does not see it at all.
    - `possess_*` — "A stack has a time complexity of O(1)." Split in two by
      whether the object is a property noun, because the shape alone is far too
      common to trust.
    """
    found = []

    for token in sentence:
        if token.pos_ not in ("VERB", "AUX"):
            continue

        subject = _subject_of(token)
        if subject is None:
            continue

        lemma = token.lemma_.lower()

        # --- consists of / comprises ------------------------------------- #
        if lemma in COMPOSITION_VERBS:
            complement = _prep_object(token, OF)
            if complement is not None and _definition_ok(_phrase_span(complement)):
                found.append((subject, complement, "consist_of", 0.65))
                continue

            # "is composed of" — the passive puts the preposition on the
            # participle's own head chain rather than on this verb.
            if any(child.dep_ in ("auxpass",) for child in token.children):
                complement = _prep_object(token, OF)
                if complement is not None:
                    found.append((subject, complement, "consist_of", 0.65))
                    continue

        # --- has / contains / includes ----------------------------------- #
        if lemma in POSSESSION_VERBS:
            obj = _direct_object(token)
            if obj is None:
                continue

            head_lemma = obj.lemma_.lower()
            if head_lemma in PROPERTY_NOUNS:
                found.append((subject, obj, "possess_property", 0.50))
            else:
                # Gated hard by the confidence floor: only a document that
                # keeps returning to this term lifts it over.
                found.append((subject, obj, "possess_salient", 0.35))

            continue

        # --- is of ---------------------------------------------------------- #
        if lemma == "be":
            complement = _prep_object(token, OF)
            if complement is not None:
                found.append((subject, complement, "be_of", 0.55))

    return found


def _direct_object(verb):
    """The verb's `dobj`, if it has one."""
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "oprd"):
            return child

    return None


def _prep_object(verb, prepositions):
    """The `pobj` of one of `prepositions`, hanging off `verb`."""
    for child in verb.children:
        if child.dep_ == "prep" and child.lemma_.lower() in prepositions:
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return grandchild

    return None


def extract(doc, salience_map=None) -> list[Fact]:
    """
    Every fact the document states, best first.

    `salience_map` is optional so the module can be exercised on its own in
    tests; when it is absent every fact scores on its pattern alone and the open
    patterns fall below the floor.
    """
    facts: list[Fact] = []

    for index, sentence in enumerate(doc.sentences):
        for subject, complement, pattern, base in _property_patterns(sentence, index):
            term_span = _phrase_span(subject)
            back_span = _phrase_span(complement)

            if term_span is None or back_span is None:
                continue

            term = _clean_term(term_span)
            back = _clean_definition(back_span)

            if not term or len(back) < 12:
                continue

            facts.append(
                Fact(
                    term=term,
                    back=back,
                    family=PROPERTY,
                    pattern=pattern,
                    evidence=sentence.text.strip(),
                    sentence_index=index,
                    confidence=_confidence(
                        base, term_span, back_span, sentence, index, salience_map
                    ),
                )
            )

    facts.sort(key=lambda f: (-f.confidence, f.term.lower()))
    return facts


def _confidence(base, term_span, back_span, sentence, index, salience_map) -> float:
    """
    The pattern's own weight, adjusted by how much this document cares.

    Deliberately the same skeleton as `definitions._confidence`, plus one term
    it does not have: sentence salience. Without that addition the open patterns
    — `possess_salient` in particular — could never reach the floor, because
    their base is below it by design.

    Absolute rather than document-normalised, for the same reason as the
    definition scorer: two documents' confidences should be comparable, and
    normalising within one makes a chapter of trivia score like a chapter of
    fundamentals.
    """
    score = base

    root = term_span.root
    if root.pos_ in ("NOUN", "PROPN"):
        score += 0.15
    elif root.pos_ in ("PRON", "DET"):
        score -= 0.25

    if len(back_span.text) > len(term_span.text):
        score += 0.05
    elif len(back_span.text) < 1.5 * len(term_span.text):
        score -= 0.10

    if salience_map is not None:
        # Top quartile is the strong signal; above the median is a weak one.
        if salience_map.score_of_sentence(index) >= salience_map.sentence_cut(0.25):
            score += 0.10
        elif salience_map.score_of_sentence(index) >= salience_map.sentence_cut(0.50):
            score += 0.05
        else:
            score -= 0.05

        if salience_map.is_salient(term_span.text):
            score += 0.05

    return max(0.0, min(1.0, score))
