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

#: Nouns that name a comparison, for "the difference between A and B".
COMPARISON_NOUNS = frozenset({"difference", "distinction", "comparison", "contrast"})

#: Verbs that set one thing against another.
DIFFERS_VERBS = frozenset({"differ", "vary"})

#: Sentence-initial markers of a contrast.
UNLIKE_MARKERS = frozenset({"unlike", "contrary"})

#: Adjectives that introduce what a thing is for.
USEFUL_ADJECTIVES = frozenset(
    {"useful", "helpful", "needed", "required", "essential", "important", "necessary"}
)

#: Verbs that introduce what a thing enables.
ENABLING_VERBS = frozenset({"allow", "enable", "permit", "let"})

#: Verbs that open a procedure.
PROCEDURAL_VERBS = frozenset({"begin", "start", "proceed", "work"})

#: Adverbs that order a sequence. At least one is required before a chain of
#: conjoined verbs counts as a process: "X does A and B" is a conjunction, and
#: without this every multi-verb sentence in a document becomes a procedure.
ORDERING_ADVERBS = frozenset(
    {"first", "firstly", "then", "next", "finally", "lastly", "subsequently", "afterwards"}
)

#: Subordinators that make a clause temporal rather than incidental.
TEMPORAL_MARKS = frozenset({"when", "whenever", "after", "before", "once"})

#: Pronouns that mean a process card would need coreference to read. There is
#: none in this stack, so a back containing one is discarded rather than shown.
UNRESOLVED_PRONOUNS = frozenset(
    {"it", "its", "they", "them", "their", "this", "that", "these", "those", "he", "she"}
)

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

        # A negated clause asserts the opposite of the card it would
        # produce. See `_is_negated`.
        if _is_negated(token):
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
    """
    The complement of one of `prepositions`, hanging off `verb`.

    Accepts `pcomp` as well as `pobj`, which is not a detail: a gerund
    complement attaches as `pcomp`, so "works by mapping a key to a position"
    and "is useful for compressing images" were both silently missed by a
    `pobj`-only lookup. `definitions._object_of` has the same gap and is left
    alone there on purpose — see the note in the test suite.
    """
    for child in verb.children:
        if child.dep_ == "prep" and child.lemma_.lower() in prepositions:
            for grandchild in child.children:
                if grandchild.dep_ in ("pobj", "pcomp"):
                    return grandchild

    return None


# ---------------------------------------------------------- comparisons --- #


def _conjunct_of(token):
    """A token's `conj` child — the second half of "A and B"."""
    for child in token.children:
        if child.dep_ == "conj":
            return child

    return None


def _copular_complement(noun):
    """The complement of the copula whose subject this noun is."""
    head = noun.head

    if head is None or head.i == noun.i:
        return None

    is_copular = head.lemma_.lower() == "be" or any(
        child.dep_ == "cop" for child in head.children
    )
    if not is_copular:
        return None

    for child in head.children:
        if child.dep_ in ("attr", "acomp"):
            return child

    return None


def _than_object(adjective):
    """
    What an adjective compares against — "faster **than** a linked list".

    Two attachments, and both have to be accepted: `than` is a `prep` with a
    `pobj` in one parse and a `mark` heading an `advcl` in the other. Which one
    appears depends on the noun that follows, which is not something this can
    control.
    """
    for child in adjective.children:
        if child.dep_ == "prep" and child.lemma_.lower() == "than":
            return _first_pobj(child)

        # "faster than linked lists": `than` is not a child of the adjective at
        # all. It is a `mark` on an `advcl` that hangs off the adjective, and the
        # compared noun is that clause's object — so the clause is the
        # participant, and `_participant_span` trims it back to the noun.
        if child.dep_ == "advcl" and any(
            mark.dep_ == "mark" and mark.lemma_.lower() == "than"
            for mark in child.children
        ):
            return child

    return None


def _comparison_patterns(sentence, index):
    """
    How two things differ — the classic exam question, and the family a bare
    term cannot express.

    **This family needs its own guards, and the reason is concrete.**
    `definitions._passes_guards` rejects negation outright, which is right for a
    definition: "Mediation is not a viable alternative" asserts the opposite of
    a definition. But "Unlike arrays, linked lists **do not** require contiguous
    memory" is precisely a contrast, and the negation *is* the fact. Sharing one
    guard set with a flag to invert it would be a guard that one day silently
    stops guarding, so the two are written separately and this family simply
    does not apply the negation check.
    """
    found = []

    for token in sentence:
        lemma = token.lemma_.lower()

        # --- the difference between A and B is X --------------------------- #
        if lemma in COMPARISON_NOUNS:
            between = _prep_object(token, {"between"})
            other = _conjunct_of(between) if between is not None else None
            complement = _copular_complement(token)

            if between is not None and other is not None and complement is not None:
                found.append((between, complement, "difference_between", 0.65, other))

            continue

        # --- A differs from B (in X) --------------------------------------- #
        if lemma in DIFFERS_VERBS:
            subject = _subject_of(token)
            other = _prep_object(token, {"from"})

            if subject is not None and other is not None:
                found.append((subject, token, "differ_from", 0.65, other))

            continue

        # --- Unlike A, B ... ------------------------------------------------ #
        if token.dep_ == "prep" and lemma in UNLIKE_MARKERS:
            other = _first_pobj(token)
            main = token.head
            subject = _subject_of(main) if main is not None else None

            if other is not None and subject is not None:
                found.append((subject, main, "unlike", 0.65, other))

            continue

        # --- A is faster than B --------------------------------------------- #
        if token.dep_ == "acomp" and token.tag_ == "JJR":
            subject = _subject_of(token.head)
            other = _than_object(token)

            if subject is not None and other is not None:
                found.append((subject, token, "comparative_than", 0.50, other))

    return found


def _first_pobj(preposition):
    """The first `pobj` under a preposition."""
    for child in preposition.children:
        if child.dep_ == "pobj":
            return child

    return None


def _span_from(root, tokens):
    """A span covering `tokens`, trimmed of edge punctuation."""
    kept = [t for t in tokens if not t.is_punct and not t.is_space]
    if not kept:
        return None

    indices = sorted(t.i for t in kept)
    span = root.doc[indices[0] : indices[-1] + 1]

    while len(span) and (span[0].is_punct or span[0].is_space):
        span = span[1:]
    while len(span) and (span[-1].is_punct or span[-1].is_space):
        span = span[:-1]

    return span if len(span) else None


def _span_without_conjuncts(token):
    """
    A phrase with its coordinated second half removed.

    The phrase span deliberately *keeps* nominal coordination — that is what
    makes "vertices and edges" survive — but "the difference between a stack and
    a queue" wants the two halves apart, because each is one participant in a
    comparison. Same tokens, different question.
    """
    excluded: set[int] = set()

    for node in token.subtree:
        if node.head.i != token.i:
            continue
        if node.dep_ == "conj":
            excluded.update(t.i for t in node.subtree)
        elif node.dep_ == "cc":
            excluded.add(node.i)

    return _span_from(token, [t for t in token.subtree if t.i not in excluded])


def _is_negated(token) -> bool:
    """
    Whether a verb's own clause carries a negation.

    Applied by every family **except** comparison, and the exception is the
    point. The definition guards reject negation because "Mediation is not a
    viable alternative" asserts the opposite of a definition; the same check
    here stopped "linked lists do **not** require contiguous memory" being
    turned into the card "linked lists require contiguous memory", which is not
    a weak card but a false one. Comparison is the family where the negation is
    usually the fact, so it writes its own rule rather than sharing this one.
    """
    return any(node.dep_ == "neg" for node in token.subtree)


# --------------------------------------------------------------- purpose --- #


def _xcomp_of(verb):
    """A verb's open clausal complement — the "to …" of "used to …"."""
    for child in verb.children:
        if child.dep_ in ("xcomp", "pcomp", "advcl"):
            return child

    return None


def _purpose_patterns(sentence, index):
    """
    What a thing is for — the "why should I care" card.

    **No modal check here, and that is the point.** `definitions` rejects
    "could/might/may/can", correctly: a hedged definition asserts something the
    source did not. A purpose is the opposite case. "A stack **can** be used to
    reverse a string" is how course text states a capability, and rejecting the
    modal would throw away the family's most common sentence. This is the second
    concrete reason the guards are per-family rather than shared.

    **Passive and predicative use only.** "The algorithm uses a hash table" is a
    dependency, not a purpose, and `use` is one of the most frequent verbs in
    the corpus. Requiring `auxpass` or an adjectival predicate removes most of
    the false-positive mass without losing a single genuine purpose statement.
    """
    found = []

    for token in sentence:
        if token.pos_ not in ("VERB", "AUX", "ADJ"):
            continue

        if _is_negated(token):
            continue

        lemma = token.lemma_.lower()

        # --- is used to / is used for --------------------------------------- #
        if lemma in PURPOSE_VERBS and any(
            child.dep_ == "auxpass" for child in token.children
        ):
            complement = _xcomp_of(token) or _prep_object(token, {"for"})

            if complement is not None:
                subject = _subject_of(token)
                # A passive subject is tagged `nsubjpass`, which `_subject_of`
                # already accepts.
                if subject is not None:
                    pattern = "used_to_xcomp" if complement.dep_ != "pobj" else "used_for"
                    found.append((subject, complement, pattern, 0.60))

            continue

        # --- is useful for --------------------------------------------------- #
        if token.dep_ == "acomp" and lemma in USEFUL_ADJECTIVES:
            complement = _prep_object(token, {"for"})
            subject = _subject_of(token.head) if token.head is not None else None

            if complement is not None and subject is not None:
                found.append((subject, complement, "useful_for", 0.45))

            continue

        # --- allows / enables ------------------------------------------------- #
        if lemma in ENABLING_VERBS:
            subject = _subject_of(token)
            obj = _direct_object(token)

            if subject is not None and obj is not None:
                found.append((subject, obj, "enable", 0.40))

    return found


# -------------------------------------------------------------- process --- #


#: Dependency labels a predicate span never includes: they introduce a
#: different statement rather than completing this one.
_PREDICATE_DROP = frozenset({"mark", "cc", "conj", "advcl", "relcl", "acl", "csubj"})


def _predicate_span(
    verb, exclude=frozenset(), keep_conjunctions=False, keep_trailing_advcl=False
):
    """
    A verb and everything that completes it, as a span.

    Not `_phrase_span`: that expands a *noun*. This walks a verb's own
    arguments — the auxiliary, the negation, the object, the prepositional
    phrases that finish the thought — and deliberately leaves out subordinate
    clauses, which are a different statement rather than the completion of this
    one.

    `exclude` names prepositions to drop along with their objects, which is how
    a comparison removes the `from` that carries the other participant.
    """
    kept = []

    def dropped(node) -> bool:
        """
        Whether a token hangs off something this span leaves out.

        The walk is over the *ancestry*, not the token's own label, and that is
        the difference between cutting a subordinate clause and cutting only its
        head: "When a function is called, the return address is pushed" has
        `called` labelled `advcl` while `When`, `a` and `function` hang off it,
        so a check on the label alone leaves the whole clause in the card back.
        """
        current = node
        while current.i != verb.i:
            if current.dep_ in _PREDICATE_DROP:
                # A chain is its conjuncts. Every other caller wants a
                # single statement, where a conjoined clause is a second
                # one; `ordered_chain` is the exception because the second
                # clause is the second step.
                if current.dep_ == "conj" and keep_conjunctions:
                    break

                # A **trailing** temporal clause is part of the statement, not
                # framing around it. "Collisions occur when two keys map to the
                # same bucket" says almost nothing without the condition — the
                # card read "Collisions occur." — while "When a function is
                # called, the return address is pushed" is complete on its own
                # and the leading clause is context the front already implies.
                # Position against the verb is what separates the two.
                if (
                    current.dep_ == "advcl"
                    and keep_trailing_advcl
                    and current.i > verb.i
                ):
                    break

                return True
            if current.dep_ == "prep" and current.lemma_.lower() in exclude:
                return True
            parent = current.head
            if parent.i == current.i:
                return False
            current = parent

        return False

    for token in verb.subtree:
        if token.is_punct or token.is_space:
            continue

        # A dropped phrase takes its object with it, or the object's index keeps
        # the enclosing range open and the other term is pulled back in.
        if not dropped(token):
            kept.append(token)

    if not kept:
        return None

    indices = sorted(t.i for t in kept)
    span = verb.doc[indices[0] : indices[-1] + 1]

    while len(span) and (span[0].is_punct or span[0].is_space):
        span = span[1:]
    while len(span) and (span[-1].is_punct or span[-1].is_space):
        span = span[:-1]

    return span if len(span) else None


def _has_unresolved_pronoun(span) -> bool:
    """
    Whether a card back leans on a pronoun for its meaning.

    "The CPU fetches the instruction, decodes **it**, and executes **it**" is a
    real sentence and a useless card: there is no coreference resolution in this
    stack, so a reader meeting it out of context cannot tell what "it" is. The
    cost is real — that sentence is dropped — and that is the correct outcome.
    """
    return any(
        token.lower_ in UNRESOLVED_PRONOUNS and token.pos_ == "PRON" for token in span
    )


def _process_patterns(sentence, index):
    """
    How something works, or what happens when.

    **Single sentence, always.** A process is a sequence, and concatenating step
    one from one sentence with step two from another asserts an ordering the
    author never wrote. That is the line between extractive and inventive, and
    this service's whole claim is that it stays on the extractive side.

    Three shapes, and the third is the riskiest thing in the module:

    - `begin_by` — "The process begins by reading the input file." Closed verb
      list, closed preposition, and a `pcomp` complement.
    - `ordered_chain` — a verb with conjoined verbs *and* an ordering adverb.
      The adverb is required: "X does A and B" is a conjunction, and without
      the cue every multi-verb sentence in the document becomes a procedure.
    - `temporal_advcl` — "When a function is called, the return address is
      pushed onto the stack." `when` is everywhere in course prose, so this is
      the highest-frequency and lowest-precision shape here. It is left to the
      salience gate, which is strict enough that it only survives when the
      sentence is central *and* the term is a keyword.
    """
    found = []

    for token in sentence:
        if _is_negated(token):
            continue

        lemma = token.lemma_.lower()

        # --- begins by / works by -------------------------------------------- #
        if lemma in PROCEDURAL_VERBS:
            subject = _subject_of(token)
            complement = _prep_object(token, {"by"})

            if subject is not None and complement is not None:
                found.append((subject, complement, "begin_by", 0.60))

            continue

        # --- a chain with an ordering cue ------------------------------------ #
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            conjuncts = [c for c in token.children if c.dep_ == "conj" and c.pos_ == "VERB"]
            cue = any(
                child.dep_ == "advmod" and child.lower_ in ORDERING_ADVERBS
                for child in token.subtree
            )

            if conjuncts and cue:
                subject = _subject_of(token)
                if subject is not None:
                    found.append((subject, token, "ordered_chain", 0.50))

            continue

        # --- when X, Y -------------------------------------------------------- #
        if token.dep_ == "advcl" and token.children:
            # `advmod` as well as `mark`: spaCy tags "When" as a wh-adverb
            # modifying the clause verb rather than as a subordinating
            # conjunction, so a `mark`-only check missed every "When ..." —
            # which is the single most common shape in course prose.
            marks = [
                c for c in token.children if c.dep_ in ("mark", "advmod")
            ]

            if any(m.lower_ in TEMPORAL_MARKS for m in marks):
                main = token.head
                subject = _subject_of(main) if main is not None else None

                if subject is not None:
                    found.append((subject, main, "temporal_advcl", 0.40))

    return found


#: Which family each pattern belongs to. Kept as data rather than as a `family`
#: argument at each call site so that adding a pattern without classifying it
#: fails loudly here instead of silently producing a card of the wrong kind.
FAMILY_OF = {
    "consist_of": PROPERTY,
    "possess_property": PROPERTY,
    "possess_salient": PROPERTY,
    "be_of": PROPERTY,
    "difference_between": COMPARISON,
    "differ_from": COMPARISON,
    "unlike": COMPARISON,
    "comparative_than": COMPARISON,
    "used_to_xcomp": PURPOSE,
    "used_for": PURPOSE,
    "useful_for": PURPOSE,
    "enable": PURPOSE,
    "begin_by": PROCESS,
    "ordered_chain": PROCESS,
    "temporal_advcl": PROCESS,
}

#: Patterns whose back comes from a verb phrase rather than a noun phrase.
#: Everything a definition-shaped pattern produces is a noun phrase; a process
#: or a contrast is a clause, and `_phrase_span` would cut it to its subject.
_PREDICATE_PATTERNS = frozenset(
    {"differ_from", "unlike", "ordered_chain", "temporal_advcl"}
)


def _back_span(complement, pattern):
    """The span a card back should be taken from, given what the pattern names."""
    if pattern in _PREDICATE_PATTERNS:
        # `unlike` opens its sentence with the *other* participant, and both
        # participants are already on the card's front. Leaving it in produced a
        # back that read "Unlike arrays, linked lists do not require contiguous
        # memory" for the card "linked lists vs. arrays".
        #
        # `differ_from` deliberately keeps its `from` phrase. Excluding it looked
        # tidier and collapsed the back to "A stack differs." — the `in` phrase
        # that carries the actual content hangs off the object, not off the
        # verb, so dropping one dropped both.
        exclude = UNLIKE_MARKERS if pattern == "unlike" else frozenset()
        return _predicate_span(
            complement,
            exclude=exclude,
            keep_conjunctions=pattern == "ordered_chain",
            keep_trailing_advcl=pattern == "temporal_advcl",
        )

    return _phrase_span(complement)


def _participant_span(token):
    """
    One side of a comparison, without the rest of its clause.

    "a queue in the order of removal" is two ideas and the participant is the
    queue. Prepositional phrases, relative clauses and coordination all continue
    the sentence rather than naming the thing, so all of them are cut.
    """
    excluded: set[int] = set()

    for node in token.subtree:
        if node.i != token.i and node.dep_ in (
            "prep",
            "conj",
            "cc",
            "mark",
            "relcl",
            "acl",
            "advcl",
        ):
            excluded.update(t.i for t in node.subtree)

    return _span_from(token, [t for t in token.subtree if t.i not in excluded])


def _term_span(subject, pattern):
    """The span a card front should be taken from."""
    if pattern == "difference_between":
        return _participant_span(subject)

    return _phrase_span(subject)


def extract(doc, salience_map=None) -> list[Fact]:
    """
    Every fact the document states, best first.

    `salience_map` is optional so the module can be exercised on its own in
    tests; when it is absent every fact scores on its pattern alone and the open
    patterns fall below the floor.

    Deduplicated by `(term, back)`, because two patterns can legitimately find
    the same fact in the same sentence — an `unlike` construction is also a
    copular one — and the reader should see it once.
    """
    facts: list[Fact] = []
    seen: set[tuple[str, str]] = set()

    for index, sentence in enumerate(doc.sentences):
        candidates = (
            _property_patterns(sentence, index)
            + _comparison_patterns(sentence, index)
            + _purpose_patterns(sentence, index)
            + _process_patterns(sentence, index)
        )

        for candidate in candidates:
            subject, complement, pattern, base, *rest = candidate
            other_span = rest[0] if rest else None

            term_span = _term_span(subject, pattern)
            back_span = _back_span(complement, pattern)

            if term_span is None or back_span is None:
                continue

            term = _clean_term(term_span)
            back = _clean_definition(back_span)

            if not term or len(back) < 12:
                continue

            # A back that leans on a pronoun cannot be read out of context.
            if _has_unresolved_pronoun(back_span):
                continue

            confidence = _confidence(
                base, term_span, back_span, sentence, index, salience_map
            )

            # `temporal_advcl` is the riskiest shape in the module — "when" is
            # everywhere in course prose — so it is held to a harder gate than
            # the rest: it survives only when the document itself treats the
            # term as one of its keywords. A confidence bonus is not enough
            # here, because the term-POS bonus alone clears the floor for any
            # noun subject, which is how "number of elements" became a card
            # front.
            if pattern == "temporal_advcl":
                if salience_map is None or not salience_map.is_salient(term):
                    continue

            # The floor is applied here rather than in `selection` so that a
            # fact that cannot be shown is never constructed at all.
            if confidence < MIN_FACT_CONFIDENCE:
                continue

            key = (term.lower(), back.lower())
            if key in seen:
                continue
            seen.add(key)

            other = None
            if other_span is not None:
                other_text = _clean_term(_participant_span(other_span))
                other = other_text or None

            facts.append(
                Fact(
                    term=term,
                    back=back,
                    family=FAMILY_OF[pattern],
                    pattern=pattern,
                    evidence=sentence.text.strip(),
                    sentence_index=index,
                    confidence=round(confidence, 4),
                    other=other,
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
