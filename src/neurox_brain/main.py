"""
Application entry point.

    uvicorn neurox_brain.main:app --host 0.0.0.0 --port 8000

**Lifespan rather than import-time work.** Loading the spaCy model inside the
lifespan handler means the process starts, binds its port and answers `/health`
*while* the model loads, rather than blocking startup for a second or more. A
container orchestrator reading `/health` therefore sees a live process that
reports `loading`, which it can distinguish from a process that never came up.

Contrast with the AMQP worker in `worker.py`, which has no port to bind and
loads the model eagerly — there is nothing to report to, so waiting is correct.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .nlp import distractors
from .nlp.corpus import corpus
from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start-up and shutdown.

    The corpus is loaded here rather than at import so that a failure to read it
    is logged and survived rather than preventing the module from being
    imported at all. See `corpus.CorpusStats.load`.
    """
    corpus.load()

    # WordNet has to be loaded once, single-threaded, before any request can
    # touch it from a worker thread. See `distractors.ensure_wordnet_loaded`.
    wordnet_ready = distractors.ensure_wordnet_loaded()

    logger.info(
        "neurox-brain %s ready — model=%s transport=%s corpus=%d documents wordnet=%s",
        __version__,
        settings.model,
        settings.transport,
        corpus.documents,
        "yes" if wordnet_ready else "NO",
    )

    yield

    # Save on the way out. The counts only ever grow, so a lost save costs the
    # documents processed since the last one and nothing else.
    corpus.save()
    logger.info("Corpus saved (%d documents).", corpus.documents)


app = FastAPI(
    title="neurox-brain",
    version=__version__,
    description=(
        "Turns expository text into flashcards, quiz questions, keywords and "
        "summaries. No language model: spaCy, TF-IDF and TextRank.\n\n"
        "See `docs/` for what each route does and how each algorithm works."
    ),
    lifespan=lifespan,
)

# The API is the only intended caller, and it calls server-to-server, so this
# does not need to admit a browser. It is left permissive for local development
# and should be pinned to the API's origin in production — an open CORS policy
# on a service that accepts 500KB of text is a free compute donation.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Validation failures in the same shape as everything else.

    FastAPI's default is `{"detail": [...]}`, which is a different envelope from
    the one `routes.py` returns and from the one the NestJS API produces. Left
    alone, the API's error normaliser would need a third branch that only ever
    fires when *this* service rejects a payload — the least-tested path, on the
    errors hardest to reproduce. Flattening the messages into the shared shape
    costs six lines and removes the branch.
    """
    messages = [
        f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={
            "status": False,
            "statusCode": 422,
            "message": messages or ["The request could not be validated."],
            "path": str(request.url.path),
        },
    )
