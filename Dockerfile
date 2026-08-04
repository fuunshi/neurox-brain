# syntax=docker/dockerfile:1
#
# neurox-brain — the NLP service, as a container.
#
# **The model and the corpus data are build steps, and that is the point.** Both
# are downloaded here rather than on first request, for two reasons the code
# itself relies on:
#
#   1. A container that fetches 40MB on its first request is a container that
#      looks healthy and then times out under a real one.
#   2. NLTK's lazy corpus loader raises when a corpus is first *used*, not when
#      it is imported — so a missing download passes every unit test and fails
#      in production, on whichever request happens to touch WordNet first.
#
# `NLTK_DATA` is set before the download so the data lands where the runtime
# will look for it, and before any import so the path is fixed for the process.

FROM python:3.12-slim AS base
WORKDIR /app

# Baked in before anything else, so it is cached and a code change does not
# re-download 40MB.
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV PYTHONUNBUFFERED=1
# No .pyc files in the image (they are written per-run and never reused) and no
# pip cache, which would otherwise be a few hundred megabytes of dead weight.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1


FROM base AS deps
COPY pyproject.toml README.md ./
# The package itself is installed after the source is copied — installing it
# here would need `src/` present and would invalidate the dependency layer on
# every source change.
RUN pip install --upgrade pip \
 && pip install "fastapi" "uvicorn[standard]" "spacy" "nltk" "numpy" "httpx" "aio-pika" \
 && python -m spacy download en_core_web_sm \
 && python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"


FROM deps AS runtime
COPY src ./src
COPY scripts ./scripts
# `pip install .` rather than an editable install: the image is immutable, so
# there is nothing for an editable install to point back at.
RUN pip install .

# The corpus is machine state, not source — see `.gitignore`. Declared as a
# volume so it survives a container replacement; without it, every restart
# starts from the cold-start IDF prior again.
VOLUME ["/app/data"]

# Not root. Nothing in this service writes outside the corpus volume.
RUN useradd --system --uid 1001 --create-home brain \
 && mkdir -p /app/data \
 && chown -R brain:brain /app
USER brain

EXPOSE 8000

# The model is loaded during the lifespan handler, so `/health` answers
# `loading` while it does — the process is up before it is ready, which is what
# lets an orchestrator tell "starting" from "dead".
#
# `${PORT:-8000}` rather than a literal, because a container host assigns the
# port and passes it in this variable: Render, for one, will route to whatever
# it chose and a service listening on a hardcoded 8000 would come up healthy and
# receive no traffic at all. The fallback keeps the compose file — which sets no
# PORT — working unchanged. `exec` form is lost to the shell here, so `sh` is
# PID 1 and forwards signals, which is enough for uvicorn's shutdown handling.
CMD ["sh", "-c", "exec uvicorn neurox_brain.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
