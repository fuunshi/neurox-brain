"""
A document-frequency store, so IDF means something.

**The problem this exists to solve.** TF-IDF needs a corpus. It is named after
two things and only one of them is available when you are handed a single
chapter: term frequency is right there in the text, but inverse *document*
frequency needs a set of documents to be inverse over.

Most implementations quietly resolve this by using one document as its own
corpus, at which point IDF becomes a constant and the "TF-IDF" is term frequency
wearing a hat. That is why keyword extraction so often returns "the" and "also".

**What this does instead.** It counts how many documents each term has appeared
in, across every document the service has processed, and persists that to disk.
The result is a real IDF that gets *better* over time: a term that shows up in
every chapter of a textbook converges on a low weight, and a term specific to one
topic keeps a high one. Cold — on the very first request — there is nothing to be
inverse over, so the weights fall back to a smoothed prior and the service says
so in the response stats.

The file is JSON rather than a database on purpose. It is one map from term to
document count, it is rebuilt by simply using the service, and a service that
cannot write it still works — degraded, not broken.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the counts live. Alongside the package rather than the working
# directory, so the service finds the same file whatever it was started from.
DEFAULT_PATH = Path(
    os.environ.get("BRAIN_CORPUS_PATH", Path(__file__).resolve().parents[3] / "data" / "corpus.json")
)

# Below this many documents the IDF is noise — one document containing a term
# says nothing about whether the term is common. Until the threshold is reached
# the prior below is used instead, blended in gradually.
MIN_DOCUMENTS_FOR_IDF = 5

# What a term's IDF is before there is a corpus. Chosen so that a term appearing
# in every document lands at roughly 0.4 and a term appearing in one lands at
# roughly 1.0 — the same range real IDF settles into, so scores do not jump
# discontinuously once the corpus warms up.
COLD_START_IDF = 0.7


class CorpusStats:
    """Document counts, in memory and on disk."""

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._documents = 0
        self._term_documents: dict[str, int] = {}
        self._loaded = False

    # -- persistence ------------------------------------------------------- #

    def load(self) -> None:
        """Read the counts, if there is a file. Failure is not fatal."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True

            try:
                raw = json.loads(self._path.read_text("utf-8"))
                self._documents = int(raw.get("documents", 0))
                self._term_documents = {
                    str(k): int(v) for k, v in raw.get("terms", {}).items()
                }
                logger.info(
                    "Corpus loaded: %d documents, %d terms",
                    self._documents,
                    len(self._term_documents),
                )
            except FileNotFoundError:
                logger.info("No corpus file yet at %s; using the cold-start prior.", self._path)
            except (OSError, ValueError, TypeError) as exc:
                # A corrupt file must not stop the service. Starting over costs
                # only the accumulated counts, which rebuild by being used.
                logger.warning("Could not read %s (%s). Starting a fresh corpus.", self._path, exc)

    def save(self) -> None:
        """Write the counts. Best effort — a read-only disk is survivable."""
        with self._lock:
            payload = {
                "documents": self._documents,
                "terms": self._term_documents,
            }

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Written to a temporary file and moved into place: a process killed
            # mid-write would otherwise leave a truncated JSON file that the
            # next start refuses to parse, losing the whole corpus.
            temporary = self._path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload), "utf-8")
            temporary.replace(self._path)
        except OSError as exc:
            logger.warning("Could not save the corpus to %s: %s", self._path, exc)

    # -- counting ---------------------------------------------------------- #

    def observe(self, terms: set[str]) -> None:
        """
        Record one document's vocabulary.

        Takes a *set*, not a list: IDF counts documents a term appears in, so a
        term used fifty times in one chapter still counts once. Callers that
        pass a list are silently deduplicated, which is the right behaviour but
        worth doing at this boundary rather than trusting every caller to.
        """
        if not terms:
            return

        with self._lock:
            self._documents += 1
            for term in terms:
                self._term_documents[term] = self._term_documents.get(term, 0) + 1

    # -- querying ---------------------------------------------------------- #

    @property
    def documents(self) -> int:
        with self._lock:
            return self._documents

    def idf(self, term: str) -> float:
        """
        Inverse document frequency, smoothed.

        The standard formula is `log(N / df)`, which is undefined for a term in
        no document and zero for a term in all of them. The `+1` smoothing used
        here (`log((N + 1) / (df + 1)) + 1`) is the scikit-learn form: it keeps
        every weight positive, so a term can never be *penalised* into
        irrelevance, only ranked low — and ranking low is what "this word is in
        every document" should mean.
        """
        with self._lock:
            documents = self._documents
            frequency = self._term_documents.get(term, 0)

        if documents < MIN_DOCUMENTS_FOR_IDF:
            # Blend the prior with whatever little evidence exists, so the
            # weights move smoothly as the corpus warms rather than jumping.
            evidence = documents / MIN_DOCUMENTS_FOR_IDF
            observed = _smoothed_idf(documents, frequency)
            return (1 - evidence) * COLD_START_IDF + evidence * observed

        return _smoothed_idf(documents, frequency)


def _smoothed_idf(documents: int, frequency: int) -> float:
    return math.log((documents + 1) / (frequency + 1)) + 1.0


# The process-wide store. Imported and used directly rather than injected,
# because there is exactly one corpus and pretending otherwise would be a
# configuration surface with one valid setting.
corpus = CorpusStats()
