"""
The pipeline: text in, study material out.

This module holds no algorithms. Its job is to run the stages in an order that
works, share the parse between them, and be honest in the statistics about what
happened. Every judgement about *how* to extract something lives in the module
named after it; every judgement about *what to do with the results* lives here.

**The order is not arbitrary.**

1. **Parse once.** Five stages need five different views of the same analysis.
   Parsing per stage would cost five times as much and let the stages disagree
   about sentence boundaries.
2. **Keywords before cloze.** Cloze needs to know which term in a sentence
   matters, and that comes from the document's own TF-IDF scores — so keyword
   extraction has to have run first. This is the one hard ordering constraint.
3. **Definitions before cards before quiz.** Each consumes the previous one's
   output. A quiz question with no correct answer is not a question.
4. **Corpus last.** The document is added to the IDF corpus only after every
   score has been computed. Doing it first would make the document influence its
   own IDF — subtly, and only on the second request for the same text, which is
   exactly the kind of bug that survives testing.
"""

from __future__ import annotations

import hashlib
import random
import time

from ..schemas import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisStats,
    GeneratedCard,
    GeneratedQuestion,
    Keyword as KeywordSchema,
    SummarySentence,
)
from . import cloze as cloze_module
from . import definitions as definitions_module
from . import distractors, keywords as keywords_module, summarise
from .corpus import corpus
from .pipeline import parse

from ..config import settings

# Below this, a definition is too weak to turn into a question. Cards are
# allowed lower — a card is a proposal the reader can discard, while a quiz
# question presents wrong answers as fact.
MIN_QUIZ_CONFIDENCE = 0.55

# Definitions per document that become quiz questions. The quiz is meant to be
# taken in one sitting, and a forty-question quiz is one nobody starts.
MAX_QUIZ_FROM_DEFINITIONS = 12


def _seeded_random(*parts: str) -> random.Random:
    """
    A generator seeded from the content it will shuffle.

    Determinism is the point: the same text must produce the same options in the
    same order, so that a cached result and a recomputed one agree and a bug
    report can be reproduced. Seeding from `hash()` would not work — Python
    randomises string hashing per process — so this uses a digest.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_cards(
    definition_list,
    doc,
    options: AnalysisOptions,
) -> list[GeneratedCard]:
    """Definitional cards, plus cloze cards when they were asked for."""
    cards: list[GeneratedCard] = []

    for definition in definition_list:
        cards.append(
            GeneratedCard(
                # The term alone rather than "What is X?" — the reader sees the
                # kind of card from its shape, and a question wrapper reads
                # oddly for terms that are not "what" questions at all, which
                # includes most of a computing syllabus ("Big-O notation").
                front=definition.term,
                back=definition.definition,
                hint=None,
                kind="DEFINITION",
                confidence=definition.confidence,
                evidence=definition.evidence,
            )
        )

        if len(cards) >= options.max_cards:
            break

    if not options.include_cloze or len(cards) >= options.max_cards:
        return cards[: options.max_cards]

    # Cloze comes from the definition sentences first — the term in a definition
    # is the one the sentence exists to introduce — and then from the highest
    # scoring remaining sentences, so a text with few definitions still yields
    # something to practise on.
    seen_fronts = {card.front for card in cards}

    for definition in definition_list:
        if len(cards) >= options.max_cards:
            break

        sentence = _sentence_containing(doc, definition.evidence)
        if sentence is None:
            continue

        term_span = _span_in(sentence, definition.term)
        if term_span is None:
            continue

        generated = cloze_module.from_term(sentence, term_span, definition.confidence)
        if generated is None or generated.front in seen_fronts:
            continue

        seen_fronts.add(generated.front)
        cards.append(
            GeneratedCard(
                front=generated.front,
                back=generated.back,
                kind="CLOZE",
                confidence=generated.confidence,
                evidence=generated.evidence,
            )
        )

    # And nothing else. The generic "blank a content word in some sentence" path
    # that used to run here is gone on purpose — see the module docstring in
    # `cloze.py`. It produced cards, and around a third of them admitted more
    # than one correct answer, which is worse than producing none: a reader who
    # cannot tell a hard card from a broken one stops trusting both.
    return cards[: options.max_cards]


def build_quiz(definition_list, options: AnalysisOptions) -> list[GeneratedQuestion]:
    """
    Multiple-choice questions whose options are definitions.

    Definitions rather than terms, because that is the question a student
    actually has to answer and because the distractor problem is tractable:
    other definitions in the same document are about other terms by
    construction, so the "also correct" risk is low.

    A question is only emitted when it is *complete* — a correct answer and
    enough plausible distractors. A question with two options is a coin flip,
    and one padded with nonsense teaches the reader to ignore the options.
    """
    questions: list[GeneratedQuestion] = []

    strong = [d for d in definition_list if d.confidence >= MIN_QUIZ_CONFIDENCE]
    definitions_pool = [d.definition for d in definition_list]

    for definition in strong[:MAX_QUIZ_FROM_DEFINITIONS]:
        if len(questions) >= options.max_quiz_questions:
            break

        wrong = distractors.for_definitions(
            definition.definition, definitions_pool, count=3
        )

        if not distractors.has_enough(wrong):
            continue

        options_text = [definition.definition, *wrong]

        # Shuffled deterministically, so the answer is not always first — which
        # is a real failure mode when questions are generated in a loop — while
        # the same input still yields the same paper.
        generator = _seeded_random(definition.term, definition.definition)
        generator.shuffle(options_text)

        questions.append(
            GeneratedQuestion(
                format="MULTIPLE_CHOICE",
                prompt=f"What is {definition.term}?",
                options=options_text,
                correct_index=options_text.index(definition.definition),
                explanation=definition.definition,
                evidence=definition.evidence,
            )
        )

    return questions


def analyse(
    text: str,
    title: str | None,
    options: AnalysisOptions,
) -> AnalysisResult:
    """The whole pipeline, for one document."""
    started = time.perf_counter()

    doc = parse(text, title)

    definition_list = definitions_module.extract(doc, limit=options.max_cards * 2)
    keyword_list = keywords_module.extract(doc, limit=options.max_keywords)

    cards = build_cards(definition_list, doc, options)
    quiz = build_quiz(definition_list, options)
    summary = summarise.summarise(doc, options.max_summary_sentences)

    # Only now. See the module docstring.
    corpus.observe(keywords_module.vocabulary(doc))

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return AnalysisResult(
        cards=cards,
        quiz=quiz,
        keywords=[
            KeywordSchema(term=k.term, score=k.score, count=k.count)
            for k in keyword_list
        ],
        summary=[SummarySentence(**s) for s in summary],
        stats=AnalysisStats(
            sentences=doc.n_sentences,
            tokens=doc.n_tokens,
            chunks=len(list(doc.doc.noun_chunks)),
            elapsed_ms=elapsed_ms,
            model=settings.model,
        ),
    )


# --------------------------------------------------------------------------- #
# Small lookups used by `build_cards`. They exist because the definition
# extractor returns strings — which is what the wire format needs — and putting
# a sentence back together from a string is a parse, not a lookup.
# --------------------------------------------------------------------------- #


def _sentence_containing(doc, text: str):
    """
    The parsed sentence whose text matches an evidence string.

    **Both sides are flattened before comparing, and that is not cosmetic.**
    `evidence` is stored collapsed — one line, no source line breaks — because
    that is what a reader should see. The parsed sentence keeps the document's
    own line breaks, because that is what the parser saw. Comparing them
    directly never matches on any sentence that spanned a newline in the source,
    which silently dropped every cloze card built from one, with no error and
    nothing in the output to say so.
    """
    from .pipeline import collapse_whitespace

    target = collapse_whitespace(text.strip())

    for sentence in doc.sentences:
        if collapse_whitespace(sentence.text.strip()) == target:
            return sentence

    return None


def _span_in(sentence, text: str):
    """The span of `text` within a sentence, or None."""
    lowered = text.lower().strip()
    for start in range(len(sentence)):
        for end in range(start + 1, min(start + 8, len(sentence)) + 1):
            if sentence[start:end].text.lower().strip() == lowered:
                return sentence[start:end]
    return None
