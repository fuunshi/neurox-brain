"""
Choosing which of the extracted candidates become cards.

Everything upstream finds facts; this decides what a reader actually sees, and
most of the tuning lives here as named constants rather than as numbers sprinkled
through the extractors.

**Why a selection stage at all.** Four families running over one chapter produce
far more candidates than a deck should hold, and the surplus is not evenly
distributed: whatever the chapter discusses most produces the most, which is not
the same as what is most worth remembering. The caps below are what stop a rich
topic becoming the whole deck.

**Every front is a source span plus, at most, a fixed constant.** No phrasing is
generated. The service has no language model, so the alternative to "the term,
sometimes with a label" is an invented question, and a mis-fired pattern behind
an invented question asserts a frame the source never did.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import GeneratedCard
from .facts import COMPARISON, PROCESS, PROPERTY, PROPERTY_NOUNS, PURPOSE

DEFINITION = "DEFINITION"

#: The separator on a comparison card. A constant, never assembled from text.
VS = "vs."

#: Labels used only when a term produces more than one card, to keep the fronts
#: distinguishable. A bare term is right while there is one card for it; two
#: cards with the same prompt are indistinguishable to a reader, and the
#: scheduler treats them as independent items.
FAMILY_LABELS = {
    DEFINITION: "definition",
    PROPERTY: "properties",
    PROCESS: "how it works",
    COMPARISON: "compared",
    PURPOSE: "purpose",
}

#: At most this many cards for one term, so a term the chapter returns to does
#: not fill the deck. Raised by one when every card for the term is strong.
MAX_PER_TERM = 2
MAX_PER_TERM_STRONG = 3
STRONG_CONFIDENCE = 0.70

#: At most one card per (term, family). Two property cards about the same term
#: are almost always a parse artefact or a near-duplicate.
MAX_PER_TERM_FAMILY = 1

#: New families are capped as a share of the deck, so a property-rich chapter
#: cannot crowd out the definitions the service already produced.
FAMILY_BUDGET_FLOOR = 3
FAMILY_BUDGET_SHARE = 0.5


@dataclass
class _Candidate:
    term: str
    back: str
    family: str
    confidence: float
    other: str | None = None
    kind: str = DEFINITION


def build(definition_list, fact_list, options) -> list[GeneratedCard]:
    """
    The cards for one document: definitions first, then whatever else fits.

    Definitions keep the `max_cards` budget to themselves first, which preserves
    the existing behaviour of the service exactly — a document that produced
    only definition cards before produces the same ones now.
    """
    cards: list[GeneratedCard] = []

    for definition in definition_list:
        if len(cards) >= options.max_cards:
            break

        # "The advantage of a linked list is its dynamic size" is a *property
        # statement about the linked list*, and the copular pattern cannot tell
        # it from a definition. Left alone it produces a card fronted "advantage
        # of a linked list", which is a card nobody wants, and the real subject
        # — the linked list — never gets it.
        #
        # Suppressed at selection rather than in `definitions`, which cannot
        # import `PROPERTY_NOUNS` from `facts` without a cycle. The property
        # family does not yet produce the card this should have been; dropping a
        # bad card is the improvement available today.
        if _is_property_of(definition.term):
            continue

        cards.append(
            GeneratedCard(
                # The term alone rather than "What is X?" — the reader sees the
                # kind of card from its shape, and a question wrapper reads
                # oddly for terms that are not "what" questions at all, which
                # includes most of a computing syllabus ("Big-O notation").
                front=definition.term,
                back=definition.definition,
                hint=None,
                kind=DEFINITION,
                confidence=definition.confidence,
                evidence=definition.evidence,
            )
        )

    remaining = options.max_cards - len(cards)
    if remaining <= 0 or not fact_list:
        return cards

    budget = max(FAMILY_BUDGET_FLOOR, int(options.max_cards * FAMILY_BUDGET_SHARE))

    chosen: list[_Candidate] = []
    per_family: dict[str, int] = {}
    per_term_family: set[tuple[str, str]] = set()
    per_term: dict[str, list[_Candidate]] = {}
    seen_backs = {_normalise(card.back) for card in cards}

    def term_allows(candidate: _Candidate) -> bool:
        existing = per_term.get(candidate.term.lower(), [])
        if len(existing) < MAX_PER_TERM:
            return True
        if len(existing) < MAX_PER_TERM_STRONG:
            return all(c.confidence >= STRONG_CONFIDENCE for c in existing)
        return False

    for fact in fact_list:
        if len(chosen) >= min(remaining, budget):
            break

        term_key = fact.term.lower()

        if (term_key, fact.family) in per_term_family:
            continue

        back_key = _normalise(fact.back)
        if back_key in seen_backs:
            continue

        if not term_allows(_candidate_of(fact)):
            continue

        candidate = _candidate_of(fact)

        chosen.append(candidate)
        per_family[fact.family] = per_family.get(fact.family, 0) + 1
        per_term_family.add((term_key, fact.family))
        per_term.setdefault(term_key, []).append(candidate)
        seen_backs.add(back_key)

    # The definition cards already claim their terms' bare fronts, and the
    # label rule has to know that or a term with a definition *and* a fact
    # produces two cards with the same prompt — which is the one thing the
    # rule exists to prevent. Found by a test asserting exactly that.
    occupied = {card.front.lower() for card in cards}
    cards.extend(_to_cards(chosen, occupied))

    return cards


def _candidate_of(fact) -> _Candidate:
    return _Candidate(
        term=fact.term,
        back=fact.back,
        family=fact.family,
        confidence=fact.confidence,
        other=fact.other,
        kind=fact.family,
    )


def _to_cards(
    candidates: list[_Candidate], occupied: set[str] | None = None
) -> list[GeneratedCard]:
    """
    Turn chosen facts into cards, resolving fronts.

    A term with one card keeps the bare term, which is what a reader expects. A
    term with several gets each card labelled, because the alternative is
    duplicate prompts.

    `occupied` carries the terms the definition cards already used, so that a
    term with one definition and one fact is counted as two cards rather than
    one.
    """
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.term.lower()] = counts.get(candidate.term.lower(), 0) + 1

    for term in occupied or ():
        counts[term] = counts.get(term, 0) + 1

    cards: list[GeneratedCard] = []

    for candidate in candidates:
        if candidate.family == COMPARISON and candidate.other:
            # A bare term cannot express a two-party fact: the front would be a
            # lie about what is being asked.
            front = f"{candidate.term} {VS} {candidate.other}"
        elif counts[candidate.term.lower()] > 1:
            # Fixed frame, from a constant. An LLM would phrase this better and
            # this service does not have one, so it uses a plain label rather
            # than inventing a question.
            front = f"{candidate.term} — {FAMILY_LABELS.get(candidate.family, candidate.family)}"
        else:
            front = candidate.term

        cards.append(
            GeneratedCard(
                front=front,
                back=candidate.back,
                hint=None,
                kind=candidate.kind,
                confidence=candidate.confidence,
                evidence="",
            )
        )

    return cards


def _normalise(text: str) -> str:
    """A back, reduced to what makes two of them the same fact."""
    return " ".join(text.lower().split()).rstrip(".")


def _is_property_of(term: str) -> bool:
    """
    Whether a card front is a property *of* something rather than the thing.

    "advantage of a linked list", "complexity of quicksort", "size of the
    array" — the term the reader wants is the noun after the `of`, and a front
    built from the noun before it names a category rather than a subject.
    """
    words = term.lower().split()

    return len(words) > 2 and words[0] in PROPERTY_NOUNS and words[1] == "of"


# Re-exported for callers that classify without importing `facts`.
FAMILIES = (DEFINITION, PROPERTY, COMPARISON, PURPOSE, PROCESS)
