"""
The contract between `neurox-brain` and the NestJS API.

These types are the interface, so they are written before the algorithms and
changed reluctantly. The NestJS side has a mirror of them in
`src/integrations/neurox-brain/neurox-brain.types.ts`; the two are kept in step
by hand, deliberately, because generating one from the other would mean a build
step in a service whose whole point is that it can be reasoned about from its
source.

Everything here is plain JSON-able data. No dataclasses with behaviour, no
enums that serialise oddly — a response is a dict, and the shape of that dict is
this file.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class AnalysisOptions(BaseModel):
    """What to produce, and how much of it."""

    max_cards: int = Field(
        default=25,
        ge=1,
        le=200,
        description=(
            "Upper bound on flashcards. A cap, not a target — a short text "
            "yields fewer, and padding to reach a number is how a deck fills "
            "with questions nobody would ask."
        ),
    )
    max_quiz_questions: int = Field(default=10, ge=1, le=100)
    include_cloze: bool = Field(
        default=True,
        description=(
            "Also emit cloze cards. Off when only definitional cards are "
            "wanted, e.g. for a deck that will be exported to Anki as Basic."
        ),
    )
    max_keywords: int = Field(default=20, ge=1, le=100)
    max_summary_sentences: int = Field(default=5, ge=1, le=50)


class AnalyseRequest(BaseModel):
    """One document, and what to make of it."""

    text: str = Field(
        min_length=1,
        max_length=500_000,
        description=(
            "The full text. Chunking is the caller's business — the API already "
            "has a chunker, and having two would mean two answers to 'what is "
            "a chunk' that disagree."
        ),
    )
    title: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Used only as a fallback topic label. It is not prepended to the "
            "text: doing so would make every sentence look related to the "
            "title and flatten the term weights that pick keywords."
        ),
    )
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class TextRequest(BaseModel):
    """A request that wants one kind of output rather than all of them."""

    text: str = Field(min_length=1, max_length=500_000)
    title: str | None = Field(default=None, max_length=300)
    limit: int = Field(default=25, ge=1, le=200)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

CardKind = Literal["DEFINITION", "CLOZE", "RELATION"]


class GeneratedCard(BaseModel):
    """
    A flashcard.

    `front`/`back`/`hint` is the shape the API's `GeneratedCard` already uses,
    so a card produced here is indistinguishable downstream from one produced by
    the heuristic or the Gemini generator. That is the point: the generator is
    swappable because the output is not.

    `confidence` exists so the API can order drafts, and `evidence` so a reader
    can see the sentence a card came from when deciding whether to keep it. A
    generated card is a proposal, and a proposal is easier to judge with its
    source attached.
    """

    front: str
    back: str
    hint: str | None = None
    kind: CardKind = "DEFINITION"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(description="The sentence this was extracted from.")


QuizFormat = Literal["MULTIPLE_CHOICE", "CLOZE"]


class GeneratedQuestion(BaseModel):
    """
    One quiz question.

    The API's quiz service builds its questions from cards by a pure function;
    these are the same shape so they can be stored in the `questions` JSON
    column unchanged. `correct_index` is included here and stripped before the
    question reaches a reader — the API already withholds `correct` until an
    answer is submitted, and this is the same rule applied one layer earlier.
    """

    format: QuizFormat
    prompt: str
    options: list[str] = Field(min_length=2, max_length=8)
    correct_index: int = Field(ge=0)
    explanation: str | None = None
    evidence: str


class Keyword(BaseModel):
    term: str
    score: float = Field(ge=0.0)
    count: int = Field(ge=1)


class SummarySentence(BaseModel):
    """One sentence of an extractive summary, with where it came from."""

    text: str
    score: float = Field(ge=0.0)
    index: int = Field(ge=0, description="Position in the source, for ordering.")


class AnalysisStats(BaseModel):
    """What the pipeline saw and how long it took.

    Worth returning rather than logging: it is the only way a caller can tell
    'this text has nothing to extract' from 'something went wrong and the
    extractor returned empty', and those need different responses.
    """

    sentences: int
    tokens: int
    chunks: int
    elapsed_ms: int
    model: str


class AnalysisResult(BaseModel):
    cards: list[GeneratedCard] = Field(default_factory=list)
    quiz: list[GeneratedQuestion] = Field(default_factory=list)
    keywords: list[Keyword] = Field(default_factory=list)
    summary: list[SummarySentence] = Field(default_factory=list)
    stats: AnalysisStats


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "degraded"]
    model: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Mirrors the NestJS global exception filter's shape.

    Not because this service is a NestJS app, but because one error format
    across both services means the API's error normaliser does not need a
    second branch — and a second branch is where the two drift.
    """

    status: Literal[False] = False
    statusCode: int
    message: str | list[str]
    path: str


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


class JobEnvelope(BaseModel):
    """
    A unit of work on the message bus.

    `reply_to` is what makes the AMQP path asynchronous without being
    fire-and-forget: the worker publishes its result to the queue named here,
    and the API's consumer correlates it back to the `GenerationJob` row by
    `job_id`. A webhook is the same idea over HTTP — see `transport/webhook.py`.
    """

    job_id: str
    text: str
    title: str | None = None
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)
    reply_to: str | None = None
    correlation_id: str | None = None
