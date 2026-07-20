"""
Configuration, read from the environment once at import.

Deliberately not pydantic-settings. There are eight values, none of them
nested, and a settings library would add a dependency and a validation model to
do what `os.environ.get` already does — with the one thing that actually
matters here made explicit: every default is written down next to the name, so
running with no environment at all is a supported state rather than an
accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        # A typo in a number should not take the service down at import; the
        # default is a working value and the warning is visible in the log.
        import logging

        logging.getLogger(__name__).warning(
            "Ignoring %s=%r: not an integer. Using %d.", name, raw, default
        )
        return default


@dataclass(frozen=True)
class Settings:
    """Everything this service can be told, and nothing it cannot."""

    # --- HTTP ---------------------------------------------------------------
    host: str = os.environ.get("BRAIN_HOST", "0.0.0.0")
    port: int = _int("BRAIN_PORT", 8000)

    # --- spaCy --------------------------------------------------------------
    #
    # **`sm`, and the reason is not size.** The obvious upgrade is `md`, for its
    # word vectors — and its vectors are the problem. spaCy prunes `md`'s vector
    # table to 20,000 entries, so 26–34 vocabulary keys share each vector and
    # cosine similarity between most word pairs comes out as either exactly 1.0
    # or an artefact of which bucket the two words landed in. Graded similarity
    # is the entire point of using vectors for distractors, and `md` does not
    # provide it. A spaCy maintainer confirms the pruning and notes it differs
    # between versions, so the scores would not even be reproducible across an
    # upgrade.
    #
    # There is no parsing argument for the bigger model either: `md` and `lg`
    # beat `sm` by about a point on dependency accuracy (LAS 0.902 vs 0.920),
    # which is nothing next to a 40MB download and 250MB of resident memory. So
    # the vectors are unusable and the parse is not better, and `sm` is what is
    # left. `lg` — 560MB on disk, ~750MB–1GB resident — would give real vectors
    # and is a whole VPS for a stage that has a WordNet-based alternative.
    #
    # Anything needing true graded similarity should use a dedicated embedding
    # model loaded with `mmap`, not a spaCy pipeline's vectors.
    model: str = os.environ.get("BRAIN_SPACY_MODEL", "en_core_web_sm")

    # --- Limits -------------------------------------------------------------
    # Guards against a request that would occupy the process for minutes. The
    # API's own source cap is 500k characters, but a single request that large
    # is 30+ seconds of parsing, so this is deliberately lower.
    max_text_chars: int = _int("BRAIN_MAX_TEXT_CHARS", 500_000)

    # --- Concurrency --------------------------------------------------------
    # spaCy's `nlp` object is not thread-safe for concurrent `pipe` calls across
    # threads with different docs in flight in every configuration; the safe and
    # simple arrangement is one worker process per core, each single-threaded.
    # This is that count, and it is also what `uvicorn --workers` should match.
    workers: int = _int("BRAIN_WORKERS", 2)

    # --- Transport ----------------------------------------------------------
    # "http"  — the API calls this service directly and waits. Simple, and the
    #           only mode that needs no extra infrastructure.
    # "amqp"  — jobs arrive on a RabbitMQ queue and results are published back.
    #           Needed when a document is large enough that holding an HTTP
    #           request open is the wrong shape.
    transport: str = os.environ.get("BRAIN_TRANSPORT", "http")

    amqp_url: str = os.environ.get(
        "BRAIN_AMQP_URL", "amqp://neurox:neurox@localhost:5672"
    )
    amqp_job_queue: str = os.environ.get("BRAIN_AMQP_JOB_QUEUE", "brain.jobs")
    amqp_result_queue: str = os.environ.get(
        "BRAIN_AMQP_RESULT_QUEUE", "brain.results"
    )
    amqp_prefetch: int = _int("BRAIN_AMQP_PREFETCH", 1)

    @property
    def use_vectors(self) -> bool:
        """Whether the configured model supplies word vectors worth using.

        `md` is excluded even though it has vectors, for the reason in the model
        comment above: they are quantised and their cosine similarities are
        artefacts. Reporting `True` for it would let a caller believe it was
        getting graded similarity when it was getting a lookup table.
        """
        return self.model.endswith(("_lg", "_trf"))


settings = Settings()
