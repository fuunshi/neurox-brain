"""
Shared fixtures for the neurox-brain suite.

Two things are set up here that every test depends on, and both exist because
of a failure that is silent rather than loud.
"""

from __future__ import annotations

import pytest

from neurox_brain.nlp import analyse as analyse_module
from neurox_brain.nlp import corpus as corpus_module
from neurox_brain.nlp import keywords as keywords_module
from neurox_brain.nlp.corpus import CorpusStats


@pytest.fixture(autouse=True)
def isolated_corpus(tmp_path, monkeypatch):
    """
    A fresh document-frequency corpus for every test.

    Without this, `analyse()` calls `corpus.observe(...)` on a **process-wide
    singleton**, and the counts accumulate across the whole session. Nothing
    crashes. What happens instead is that IDF drifts as the suite runs, so
    keyword scores — and, once card selection is gated on them, which cards are
    produced — depend on how many tests ran before. A test would pass alone and
    fail in a file, or the reverse, and the cause would be three files away.

    Patching three module attributes rather than one, and that is not
    belt-and-braces: `analyse.py` and `keywords.py` both do
    `from .corpus import corpus`, which binds the object into their own
    namespace at import. Rebinding `corpus_module.corpus` alone changes the
    module the singleton lives in and nothing that already holds a reference.
    """
    fresh = CorpusStats(path=tmp_path / "corpus.json")

    monkeypatch.setattr(corpus_module, "corpus", fresh)
    monkeypatch.setattr(keywords_module, "corpus", fresh)
    monkeypatch.setattr(analyse_module, "corpus", fresh)

    return fresh


@pytest.fixture(scope="session")
def nlp():
    """
    The raw spaCy pipeline, loaded once.

    Session-scoped because `spacy.load` costs about a second, and it would
    otherwise be paid per test. `get_nlp` is `lru_cache`d, so this reaches the
    same object every other caller gets.

    For parsing sentences to assert on dependencies. **Not** a substitute for
    `parse` — it skips normalisation and the sentence filters, so a test that
    reaches for this when it means `parse` is testing something the service
    never does.
    """
    from neurox_brain.nlp.pipeline import get_nlp

    return get_nlp()


@pytest.fixture
def parse():
    """
    The real `pipeline.parse` — normalise, parse, filter — returning a `Document`.

    Deliberately the production entry point rather than `nlp(text)`. The
    extractors walk `document.sentences`, which is the *filtered* list, so a
    test built on an unfiltered `Doc` would exercise code paths the service
    cannot reach.
    """
    from neurox_brain.nlp.pipeline import parse as _parse

    return _parse
