"""
The HTTP surface.

Seven routes — two about the service itself and five that extract — and the
reason there is more than one is that the API calls this service in two
different moods:

- **`POST /analyse`** — "here is a chapter, give me everything". One parse, all
  the output. This is what the API's generation worker uses.
- **The four part routes** — "I already have the material, I just want more of
  one thing". They run the same pipeline and return one section, because
  splitting the pipeline would mean a second code path that could disagree with
  the first about what a sentence is.

**On running the full pipeline for a part route.** It looks wasteful — asking
for keywords parses the document, extracts definitions, builds quiz questions
and throws them away. It is not, in practice: parsing is 90% of the cost and
every later stage is linear in sentences, so a second entry point that ran only
the cheap stages would save a few milliseconds and cost a second definition of
correctness. The shared path is the one that stays right.

**On error shape.** Failures are returned in the same shape as the NestJS
global exception filter produces, so the API's error normaliser needs no second
branch. See `schemas.ErrorResponse`.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .nlp import analyse as analyse_module
from .nlp.pipeline import is_model_loaded
from .nlp.corpus import corpus
from .schemas import (
    MAX_KEYWORDS,
    MAX_QUIZ_QUESTIONS,
    MAX_SUMMARY_SENTENCES,
    AnalyseRequest,
    AnalysisOptions,
    AnalysisResult,
    HealthResponse,
    Keyword,
    GeneratedCard,
    GeneratedQuestion,
    SummarySentence,
    TextRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_STARTED_AT = time.monotonic()


def _error(status: int, message: str, path: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "status": False,
            "statusCode": status,
            "message": message,
            "path": path,
        },
    )


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """
    Liveness, and what the service is currently capable of.

    `model_loaded` is reported without loading the model — see
    `pipeline.is_model_loaded`. A health probe that costs 300MB the first time it
    is called is not a probe, it is a surprise.

    `status` is `ok` once the model is resident, `loading` before that. Note that
    it is never `degraded` for a missing vector model: running `en_core_web_sm`
    is a supported configuration with a known weaker distractor stage, not a
    fault, and reporting it as a fault would train an operator to ignore the
    field.
    """
    return HealthResponse(
        status="ok" if is_model_loaded() else "loading",
        model=settings.model,
        model_loaded=is_model_loaded(),
        version=__version__,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 1),
    )


@router.get("/stats", tags=["meta"])
async def stats() -> dict:
    """
    What the corpus has learned.

    The IDF half of every keyword score depends on this, so it is worth being
    able to see: `documents` climbing means keyword weights are getting closer
    to real IDF and further from the cold-start prior.
    """
    return {
        "documents": corpus.documents,
        "model": settings.model,
        "transport": settings.transport,
    }


@router.post("/analyse", response_model=AnalysisResult, tags=["extract"])
async def analyse(payload: AnalyseRequest, request: Request):
    """
    Everything, in one pass: cards, quiz questions, keywords and a summary.

    This is the route the API uses. It is deliberately one request rather than
    four, because the four would each pay for a parse of the same text.
    """
    if len(payload.text) > settings.max_text_chars:
        return _error(
            413,
            f"Text is {len(payload.text)} characters; the limit is "
            f"{settings.max_text_chars}.",
            str(request.url.path),
        )

    try:
        return analyse_module.analyse(payload.text, payload.title, payload.options)
    except RuntimeError as exc:
        # Raised by the pipeline when the spaCy model is missing. That is a
        # deployment fault rather than a bad request, so it is a 503 — the call
        # will succeed once the service is fixed, and retrying is reasonable.
        logger.error("Model unavailable: %s", exc)
        return _error(503, str(exc), str(request.url.path))
    except Exception as exc:  # noqa: BLE001 — see below
        # Deliberately broad. An unhandled extractor failure must not take the
        # worker down or leave the API waiting on a request that will never
        # answer; the traceback goes to the log and the caller gets a retryable
        # error. Analysing arbitrary PDF text means meeting arbitrary text.
        logger.exception("Analysis failed")
        return _error(500, f"Analysis failed: {exc}", str(request.url.path))


def _clamp(value: int, maximum: int) -> int:
    """
    Bring a caller's `limit` inside what the target option allows.

    **This exists because not clamping produced a 500.** `TextRequest.limit`
    allows up to 200, but `max_quiz_questions`, `max_keywords` and
    `max_summary_sentences` each cap lower — 100, 100 and 50. Passing 150
    straight through meant constructing `AnalysisOptions` raised a pydantic
    `ValidationError` *inside the handler*, where it is neither a
    `RequestValidationError` nor caught by anything, so the caller got an
    unhandled plain-text 500 for what is really a bad request.

    Clamping rather than rejecting is the better contract: "give me 150
    keywords" and "give me as many as you have" mean the same thing to a
    caller, and 100 keywords is a more useful answer than an error. FastAPI
    still enforces the outer bound, so 500 still fails validation properly.
    """
    return max(1, min(value, maximum))


async def _run(payload: TextRequest, request: Request, options: AnalysisOptions):
    """
    Analyse, or raise an error in the shared envelope.

    Every part route goes through here so that all five extraction routes fail
    the same way. They did not: `/analyse` mapped a missing model to 503 and
    enforced the size cap, while the four part routes did neither — a missing
    model gave them a 500, an oversized body was not rejected at all, and the
    same deployment fault produced a "retryable" answer from one route and an
    "unexpected" one from another.

    Raises rather than returning an error object so the handler stays a plain
    `return`. The `HTTPException` is rendered by the handler in `main.py`, which
    is also what gives the shared envelope to 404s and 405s that FastAPI raises
    on its own.
    """
    if len(payload.text) > settings.max_text_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Text is {len(payload.text)} characters; the limit is "
                f"{settings.max_text_chars}."
            ),
        )

    try:
        return analyse_module.analyse(payload.text, payload.title, options)
    except RuntimeError as exc:
        # Raised by the pipeline when the spaCy model is missing: a deployment
        # fault, not a bad request, so 503 and retrying is reasonable.
        logger.error("Model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — see below
        # Deliberately broad. Analysing arbitrary PDF text means meeting
        # arbitrary text; an extractor failure must not leave the caller waiting
        # on a request that will never answer.
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {exc}"
        ) from exc


@router.post("/cards", response_model=list[GeneratedCard], tags=["extract"])
async def cards(payload: TextRequest, request: Request) -> list[GeneratedCard]:
    """
    Flashcards only.

    Definitional cards first, then cloze cards, ordered by confidence. The
    `limit` bounds the total, and the `evidence` field on each card carries the
    sentence it came from — a generated card is a proposal, and a proposal is
    easier to judge with its source attached.
    """
    # Cloze on explicitly. The option now defaults to off — a generated
    # deck is facts rather than fill-in-the-blanks — but this route's whole
    # purpose is "more of what /analyse gave me", and it documents that it
    # returns both kinds.
    options = AnalysisOptions(max_cards=payload.limit, include_cloze=True)
    return (await _run(payload, request, options)).cards


@router.post("/quiz", response_model=list[GeneratedQuestion], tags=["extract"])
async def quiz(payload: TextRequest, request: Request) -> list[GeneratedQuestion]:
    """
    Multiple-choice questions.

    Every returned question has a correct answer and at least three distractors,
    so a short document can legitimately yield none — see `build_quiz`.
    `correct_index` is included and must be stripped before the question reaches
    a reader.
    """
    options = AnalysisOptions(
        max_quiz_questions=_clamp(payload.limit, MAX_QUIZ_QUESTIONS)
    )
    return (await _run(payload, request, options)).quiz


@router.post("/keywords", response_model=list[Keyword], tags=["extract"])
async def keywords(payload: TextRequest, request: Request) -> list[Keyword]:
    """
    TF-IDF keywords and keyphrases, best first.

    Scores are comparable *within* one response, not across responses: they are
    a product of a document-dependent term frequency and a corpus-dependent IDF,
    so two documents' scores are on different scales by construction.
    """
    options = AnalysisOptions(max_keywords=_clamp(payload.limit, MAX_KEYWORDS))
    return (await _run(payload, request, options)).keywords


@router.post("/summary", response_model=list[SummarySentence], tags=["extract"])
async def summary(payload: TextRequest, request: Request) -> list[SummarySentence]:
    """
    An extractive summary, in reading order.

    Sentences are chosen by TextRank — or by frequency, below ten sentences —
    and returned in the order they appear in the source, because a summary in
    rank order reads as a shuffled document. `score` is included so an interface
    can say why each was chosen.
    """
    options = AnalysisOptions(
        max_summary_sentences=_clamp(payload.limit, MAX_SUMMARY_SENTENCES)
    )
    return (await _run(payload, request, options)).summary
