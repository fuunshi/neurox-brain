# Running it, configuring it, and what to do when it misbehaves

Practical notes for the person deploying or debugging this service. For what the
routes do, see `docs/routes.md`; for how it fits into the rest of the system, see
`docs/architecture.md`.

---

## Local setup from scratch

`pyproject.toml` requires Python ≥ 3.11. The venv in this tree is a 3.14 one, so
nothing depends on a specific minor version.

```bash
git clone <this repo> && cd neurox-brain

python3 -m venv .venv
source .venv/bin/activate

# Editable install. This also puts the `neurox_brain` package on the path —
# see the note below, because it is the step that is easy to skip.
pip install -e .
# or, for the test extras:  pip install -e ".[dev]"

# The spaCy model named by BRAIN_SPACY_MODEL (default en_core_web_sm, ~12MB).
python -m spacy download en_core_web_sm

# NLTK WordNet data, ~10MB. Both the corpus and omw-1.4 — the latter is what
# the multilingual lemmas resolve through, and pip's nltk does not ship either.
python -m nltk.downloader wordnet omw-1.4
```

Then run it:

```bash
uvicorn neurox_brain.main:app --host 0.0.0.0 --port 8000   # HTTP service
python -m neurox_brain.worker                              # queue worker
```

**The install step is not optional, and there is no error message that says so.**
This tree's `.venv` has the dependencies but not the package itself, and
`uvicorn neurox_brain.main:app` from the repository root fails with:

```
ModuleNotFoundError: No module named 'neurox_brain'
```

Either `pip install -e .` (preferred — it also gives you the console scripts) or
run with `PYTHONPATH=src`. `[tool.setuptools.packages.find] where = ["src"]` means
the package itself lives under `src/`, or it means an editable install that puts
it on `sys.path`. There is no `scripts/dev.sh` in this repository, though
`flash-cards-backend/.env.template` refers to one — the commands above are the
setup.

**Configuration is environment variables only.** `config.py` reads `os.environ`
at import; there is no config file, and no `python-dotenv` call anywhere in
`src/`, so `.env.example` is documentation rather than something the service
loads. Export the variables, or use a runner (`docker run --env-file`, compose's
`environment:`, systemd's `Environment=`) that exports them for you. A shell
loop is enough locally:

```bash
set -a; . ./.env; set +a
uvicorn neurox_brain.main:app
```

### Why the NLTK download has to happen at build time

It is tempting to treat `python -m nltk.downloader` as something to do on the
first request that complains. It must not be, for two reasons that
`distractors.ensure_wordnet_loaded` documents.

**It fails late by nature.** A missing corpus raises `LookupError` when it is
first *used*, not when it is imported: "the problem passes unit tests and appears
on the first real request." A build that does not download WordNet produces an
image that starts cleanly, passes its health check, answers `/health` with `ok`,
and then degrades — or, if the code path ever stops catching `LookupError`,
fails — on the first quiz question that needs a distractor. Downloading at build
time makes it a start-up log line instead of a production surprise:

```
neurox-brain 0.1.0 ready — model=en_core_web_sm transport=http corpus=0 documents wordnet=yes
```

**And loading it lazily from a worker thread is a real bug.** NLTK's
`LazyCorpusLoader` is not thread-safe; two threads reaching a corpus for the
first time simultaneously can leave it half-initialised with failures like
`'WordNetCorpusReader' object has no attribute '_LazyCorpusLoader__args'`. The
documented fix is exactly what the service does — load it once, early,
single-threaded — and that fix only helps if the data is present:

```python
# main.py, in the lifespan
wordnet_ready = distractors.ensure_wordnet_loaded()

# worker.py, before the consumer starts
distractors.ensure_wordnet_loaded()
```

If the download is skipped, `ensure_wordnet_loaded()` returns `False`, the
start-up log says `wordnet=NO`, and a warning explains what degrades. The service
still starts and still generates.

---

## Every environment variable

All of these are read once, at import. Changing one requires restarting the
process. Every value has a working default, so "no environment at all" is a
supported state (the point of `config.py`'s docstring).

### HTTP

| Variable | Default | What changing it does |
| --- | --- | --- |
| `BRAIN_HOST` | `0.0.0.0` | **Read by nothing.** It is a documented value for the deployment, not a control: `uvicorn` takes the bind address from its own `--host` flag, which is how `main.py`'s docstring starts the service. |
| `BRAIN_PORT` | `8000` | Same — `uvicorn --port` is what actually binds. |

### Model

| Variable | Default | What changing it does |
| --- | --- | --- |
| `BRAIN_SPACY_MODEL` | `en_core_web_sm` | The spaCy pipeline to load. Must be installed in the same interpreter (`python -m spacy download <name>`). Anything other than `_lg`/`_trf` disables the vector path in the distractor generator — see below. |

### Limits

| Variable | Default | What changing it does |
| --- | --- | --- |
| `BRAIN_MAX_TEXT_CHARS` | `500000` | The `/analyse` request cap. Lower it on a small machine; a 500k-character request is 30+ seconds of parsing. Only `/analyse` enforces it (see `docs/routes.md`). The schema's own 500,000-character cap on `text` applies everywhere and is not configurable, so setting this *above* 500,000 has no effect at all. |

### Concurrency

| Variable | Default | What changing it does |
| --- | --- | --- |
| `BRAIN_WORKERS` | `2` | **Read by nothing at runtime.** Nothing in the service starts or sizes processes from it. It is the number the deployment should use: "one worker process per core, each single-threaded… and it is also what `uvicorn --workers` should match." Set `uvicorn --workers` to match it; run the same number of queue-worker containers. |

### Transport

| Variable | Default | What changing it does |
| --- | --- | --- |
| `BRAIN_TRANSPORT` | `http` | Reported in `GET /stats` and the start-up log. **Nothing dispatches on it** — which transport runs is decided by which entry point you start. See `docs/architecture.md`. |
| `BRAIN_AMQP_URL` | `amqp://neurox:neurox@localhost:5672` | Broker URL for the queue worker. A wrong URL fails after a bounded 30-second connect rather than hanging. Must match the API's `NEUROX_BRAIN_AMQP_URL`. |
| `BRAIN_AMQP_JOB_QUEUE` | `brain.jobs` | Queue the worker consumes. Must match the API's `NEUROX_BRAIN_JOB_QUEUE`, because the API publishes to a queue by name and declares nothing on that side. |
| `BRAIN_AMQP_RESULT_QUEUE` | `brain.results` | Fallback target for results when the envelope carries no `reply_to`. Must match the API's `NEUROX_BRAIN_RESULT_QUEUE`, which is also the queue the API declares and consumes. |
| `BRAIN_AMQP_PREFETCH` | `1` | Unacknowledged jobs the worker will hold. Leave it at 1: the process parses single-threaded, so a second job only waits while the broker's unacknowledged-message timer runs. Scale by running more worker processes. |

### Corpus

| Variable | Default | What changing it does |
| --- | --- | --- |
| `BRAIN_CORPUS_PATH` | `<repo>/data/corpus.json` | Where the document-frequency counts are kept. See the section below. Note that `.env.example` sets it to `./data/corpus.json` — a *relative* path, resolved against the working directory, unlike the default which is resolved from the package location. If you copy that line into a container with a different working directory, you get a different file. |

Numbers are parsed by `config._int`, which is deliberately forgiving: a value
that is not an integer logs `Ignoring X='...': not an integer. Using N.` and falls
back to the default rather than taking the service down at import.

`settings.use_vectors` (a property, not a variable) is `True` only when the model
name ends in `_lg` or `_trf`. `true` enables the similarity band in the
distractor generator; `false` skips it and ranks candidates by shared head noun
instead.

### Variables the API reads that this service also has an opinion about

Kept here because the two sides have to agree, and the pairing is the usual
source of a half-configured deployment. All of these live in
`flash-cards-backend/src/infra/config/neurox-brain.config.ts`.

| API variable | Default | This service's counterpart |
| --- | --- | --- |
| `NEUROX_BRAIN_ENABLED` | `false` | — (turns the client on at all) |
| `NEUROX_BRAIN_URL` | `http://localhost:8000` | `BRAIN_HOST` / `BRAIN_PORT` |
| `NEUROX_BRAIN_TRANSPORT` | `http` | `BRAIN_TRANSPORT` (informational here) |
| `NEUROX_BRAIN_TIMEOUT_MS` | `30000` | — |
| `NEUROX_BRAIN_AMQP_URL` | `amqp://neurox:neurox@localhost:5672` | `BRAIN_AMQP_URL` |
| `NEUROX_BRAIN_JOB_QUEUE` | `brain.jobs` | `BRAIN_AMQP_JOB_QUEUE` |
| `NEUROX_BRAIN_RESULT_QUEUE` | `brain.results` | `BRAIN_AMQP_RESULT_QUEUE` |
| `NEUROX_BRAIN_WEBHOOK_SECRET` | `""` | **not read by this service** |

`NEUROX_BRAIN_WEBHOOK_SECRET` has a counterpart in `.env.example`
(`BRAIN_WEBHOOK_SECRET`) and a counterpart in the API's config, but no code in
`src/` reads either name: the webhook transport is documented and implemented as
`sign()`/`deliver()` in `transport/webhook.py` but is not called by any route or
worker, and there is no callback endpoint on the API side. Setting it today
changes nothing. See `docs/architecture.md`.

---

## Why the default model is `en_core_web_sm` and not `md`

The obvious upgrade is `md`, for its word vectors, and the vectors are the
problem. From the comment on `Settings.model`:

> spaCy prunes `md`'s vector table to 20,000 entries, so 26–34 vocabulary keys
> share each vector and cosine similarity between most word pairs comes out as
> either exactly 1.0 or an artefact of which bucket the two words landed in.
> Graded similarity is the entire point of using vectors for distractors, and
> `md` does not provide it. A spaCy maintainer confirms the pruning and notes it
> differs between versions, so the scores would not even be reproducible across
> an upgrade.

**Stated precisely: `en_core_web_md` is usable — it loads, parses, and produces
cards, quizzes and keywords — but its vectors are not trustworthy, so nothing
here is allowed to depend on them.** `settings.use_vectors` returns `False` for
it, which means the distractor generator skips the similarity band entirely and
falls back to shared-head-noun ranking. The band is not applied "to an artefact".

There is no parsing argument for the bigger model either: `md` and `lg` beat `sm`
by about a point on dependency accuracy (LAS 0.902 vs 0.920), which is nothing
next to a 40MB download and 250MB of resident memory. And `lg` — 560MB on disk,
roughly 750MB–1GB resident — would give real vectors, at the cost of "a whole VPS
for a stage that has a WordNet-based alternative".

The comment ends with the direction for anything that genuinely needs graded
similarity:

> Anything needing true graded similarity should use a dedicated embedding model
> loaded with `mmap`, not a spaCy pipeline's vectors.

So: keep `sm`. If you set `BRAIN_SPACY_MODEL=en_core_web_md`, everything still
works and nothing gets better — you pay 40MB and 250MB of RSS for vectors the
code refuses to use. `en_core_web_lg` will turn the vector path on, and
`en_core_web_trf` will too; both are a large step up in memory.

Note also that `.env.example` is out of step with the code here: its comment
describes `en_core_web_md` as "(default)" and sets `BRAIN_SPACY_MODEL=en_core_web_md`.
The actual default in `config.py` is `en_core_web_sm`. The example file is not
loaded by anything, so nothing breaks — but a deployment that copies it line by
line will run `md` for no benefit.

---

## The corpus file

The IDF half of every keyword score needs to know how common a term is across
documents, and a single chapter is not a corpus. `nlp/corpus.py` counts how many
documents each term has appeared in, across everything the service has processed,
and keeps the counts in one JSON file.

- **Where.** `BRAIN_CORPUS_PATH`, defaulting to `data/corpus.json` in the
  repository root — resolved from the package's own location
  (`Path(__file__).resolve().parents[3] / "data" / "corpus.json"`), "so the
  service finds the same file whatever it was started from."
- **What.** JSON, one document count and one term map:

  ```json
  {"documents": 3, "terms": {"data": 3, "queue": 3, "algorithm": 2, "recursion": 1}}
  ```

  `documents` counts *pipeline runs*, not files — every `/analyse`, `/cards`,
  `/quiz`, `/keywords` and `/summary` request adds one. `terms` counts the
  documents each term appeared in, so a term used fifty times in one chapter
  counts once.
- **Deleting it is safe.** It is the only state this service keeps, it rebuilds
  by being used, and a missing file is an informational log line ("No corpus file
  yet at …, using the cold-start prior"), not an error. A *corrupt* file is a
  warning and a fresh start: "A corrupt file must not stop the service. Starting
  over costs only the accumulated counts, which rebuild by being used."
- **It is written atomically.** To `<name>.tmp`, then `Path.replace()` — "a
  process killed mid-write would otherwise leave a truncated JSON file that the
  next start refuses to parse, losing the whole corpus."
- **It is written on shutdown only.** `corpus.save()` is called in exactly two
  places: `main.py`'s lifespan, after `yield`, and `worker.py`'s `finally`. There
  is no periodic flush and nothing writes it per request. A `SIGKILL` — or a
  container runtime's kill timeout expiring — loses every count since the process
  started. `SIGTERM` is handled properly: the worker logs "Shutting down; waiting
  for the current job to finish", lets the in-flight job complete and be
  acknowledged, and then saves. Give the container a stop grace period longer
  than your longest job.
- **Two processes means two counts, and last-writer-wins.** The HTTP service and
  the queue worker each hold their own in-memory counts and each write the whole
  file on exit, so whichever exits last defines the file and the other's
  increments since start are lost. This is a consequence of a whole-file save,
  not something the code tries to prevent. If keyword weights matter to you, run
  one of the two, or accept a smaller corpus than the number of requests
  suggests.

**What degrades on a cold start.** Below `MIN_DOCUMENTS_FOR_IDF` (5) documents,
`CorpusStats.idf` blends a fixed prior with whatever evidence exists, in
proportion to `documents / 5`:

```python
return (1 - evidence) * COLD_START_IDF + evidence * observed
```

`COLD_START_IDF` is 0.7, "chosen so that a term appearing in every document lands
at roughly 0.4 and a term appearing in one lands at roughly 1.0 — the same range
real IDF settles into, so scores do not jump discontinuously once the corpus
warms up."

Nothing fails and nothing is wrong. What you get is keyword scores that are
closer to weighted term frequency: a term used often will rank highly even if it
is common across the subject. The first ~5 requests are on the prior; `GET /stats`
tells you where you are, and it is the number to check before believing a keyword
list. Two documents scored on different corpus sizes are not comparable — which
is why the `/keywords` route docstring limits the claim to "within one response".

---

## Health and stats

| Endpoint | What to look at | Reading it |
| --- | --- | --- |
| `GET /health` | `status` | `loading` = the model is not resident yet. `ok` = it is, and requests will be parsed by it. |
| | `model_loaded` | Raw boolean behind `status`. Nothing loads the model to answer this. |
| | `model` | The configured name. Compare it against `python -m spacy info` if you suspect it is not installed. |
| | `uptime_seconds` | Since import. A number that resets in a loop is a crash loop. |
| `GET /stats` | `documents` | Pipeline runs this process has counted. Climbing means IDF is approaching real IDF. |
| | `transport` | What `BRAIN_TRANSPORT` says. Report only — see above. |

`status` never becomes `degraded`; see `docs/routes.md` for why the state is
deliberately unused rather than reused for a missing vector model.

**A liveness probe should read `/health` and nothing else.** It is cheap by
construction and takes no lock. Do not point a readiness probe at `/analyse`: it
is a full parse, and a probe that costs a core is worse than no probe.

One trap worth knowing: `status: "loading"` is also what you get from a
**misspelled model name**, because nothing tries to load the model until a
request arrives. If `status` is still `loading` after a minute, check
`model_loaded` and try one `/analyse` by hand — a 503 with the message `spaCy
model 'x' is not installed` is the real answer.

---

## Scaling

**Processes, not threads.**

> spaCy's `nlp` object is not thread-safe for concurrent `pipe` calls across
> threads with different docs in flight in every configuration; the safe and
> simple arrangement is one worker process per core, each single-threaded.

This is not a tuning choice, it is a correctness one, and the code is arranged
around it: one shared `nlp` per process (`pipeline.get_nlp`), parsing called with
no `n_process` argument, no thread pool around the pipeline, and `asyncio.to_thread`
used in the AMQP handler only to keep one blocking call off the event loop.
Adding threads buys nothing and risks a half-initialised pipeline.

**Memory.** Budget **100–300MB of resident memory per process** for the model,
plus the interpreter and the corpus. The figure is `pipeline.py`'s: "Loading a
spaCy model costs about a second and 100–300MB of resident memory." `sm` sits at
the low end, `lg` is quoted at 750MB–1GB, and `md` at roughly 250MB. A `_trf`
model is larger still and is not a configuration anyone has measured here.

**How many processes.** `BRAIN_WORKERS` (default 2) is the number to copy:
`uvicorn --workers` for the HTTP service, and the same count of queue-worker
containers. They are separate pools — an HTTP worker and a queue worker each hold
their own model, so the total is the sum. On a 2-core VPS, two HTTP workers and
zero queue workers is right if you use HTTP transport; with AMQP, give the queue
the cores and keep one HTTP worker for health checks.

**Parsing cost is what you are sizing.** Parsing is roughly 90% of a request's
cost and every later stage is linear in sentences (`routes.py`). A 2,000-character
chunk is around 1.5 seconds on the reference machine — the figure the API's
timeout is sized against. Throughput scales linearly with processes and not at
all with prefetch.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Start-up fails with `ModuleNotFoundError: No module named 'neurox_brain'` | The package is not installed in the interpreter that is running `uvicorn`. | `pip install -e .`, or run with `PYTHONPATH=src`. |
| `/health` says `loading` forever, `/analyse` returns 503 `spaCy model 'x' is not installed` | `BRAIN_SPACY_MODEL` names a model that is not downloaded in this interpreter. | `python -m spacy download <name>`. Note the download is per-interpreter, so a venv that is not the one running uvicorn will not help. |
| `/analyse` works but a part route returns `500 Internal Server Error` (plain text, no envelope) | Two known causes: `limit` is above that route's option maximum (`/quiz` and `/keywords` cap at 100, `/summary` at 50), or the model is missing. The part routes have no error handling. | Keep `limit` ≤ 50 on part routes, or use `/analyse` where the bound is validated and a missing model is a 503. See `docs/routes.md`. |
| Start-up log shows `wordnet=NO` and `WordNet is unavailable (…)` | NLTK data missing. | `python -m nltk.downloader wordnet omw-1.4`, then restart. Until then, distractors come from the document alone with **no synonym check** — "an option that is also correct can slip through". Quiz questions still generate; they are just more likely to have a second defensible answer. |
| Quiz questions are missing, or far fewer than `max_quiz_questions` | Working as designed. A question is only emitted if its definition clears `MIN_QUIZ_CONFIDENCE` (0.55) *and* three distractors were found; at most 12 definitions become questions per document. | Nothing. A document with few definitions cannot yield many questions, and padding them "teaches the reader to ignore the options". |
| Keywords look like ordinary frequent words | Cold corpus: below 5 documents the IDF is a prior, so scores behave like weighted term frequency. | Check `documents` in `/stats`. It improves by being used; there is no way to warm it except processing text. |
| Keyword scores differ between two identical requests | Expected — the corpus grew between them, and the IDF half changed. | Compare scores only within a response (`docs/routes.md`). |
| `data/corpus.json` is not being written | It is written on clean shutdown only. | Stop the process with `SIGTERM`/`SIGINT` (Ctrl-C), not `SIGKILL`, and allow the grace period. |
| Log: `Could not save the corpus to /…: …` | The directory is not writable — commonly a read-only container filesystem or a volume not mounted. | Mount a writable path and point `BRAIN_CORPUS_PATH` at it. The service keeps working with a cold corpus. |
| Log: `Could not read … (…). Starting a fresh corpus.` | The corpus file is corrupt. | Nothing required; it rebuilds. Investigate only if it happens repeatedly — a full disk is the usual cause. |
| API log: `neurox-brain answered 500: …` | The service is up but failing. | Read its log. A model-missing 503 or a part-route 500 will say so. |
| API log: `fetch failed` / `ECONNREFUSED` on generation | The service is not reachable from the API container. | Check `NEUROX_BRAIN_URL`, that the container is running, and that both are on the same network. `docker compose logs` on the brain container first. |
| API job fails with `neurox-brain did not answer within 30000ms` (AMQP) or an abort error (HTTP) | The analysis took longer than `NEUROX_BRAIN_TIMEOUT_MS`, or the model was still loading. | Check the first-request cost: a cold process pays the model load. Raise `NEUROX_BRAIN_TIMEOUT_MS`, or lower the chunk size on the API side. On the AMQP path the job keeps running and its result is discarded as late. |
| AMQP jobs are published and never answered | The worker is not running (the HTTP service does not consume the queue), or the queue names disagree. | Start `python -m neurox_brain.worker` and compare `BRAIN_AMQP_*_QUEUE` against `NEUROX_BRAIN_*_QUEUE`. |
| Queue jobs disappear without a result | The failure path ended in `nack(requeue=False)`, which drops the message when no dead-letter queue is configured. | Check the worker log for `Job … failed`; configure a DLQ on `brain.jobs` if you need the messages kept. |
| Generated cards look noisy or wrong about a third of the time | Expected. Hand-written definition patterns have a published precision around **0.16** on real course text. | Nothing at this layer — that is why cards land as `DRAFT`. See `docs/algorithms/README.md`. |

---

## Turning it off: `NEUROX_BRAIN_ENABLED=false`

`NEUROX_BRAIN_ENABLED` defaults to `false` on the API side, and that is the safe
default rather than an oversight. From `neurox-brain.config.ts`:

> It would be convenient to treat "a URL is configured" as "use it", but the URL
> has a working localhost default, so that rule would make every developer's
> machine depend on a Python service being up. An explicit switch means the
> default behaviour is the one that already works, and turning the brain on is a
> decision rather than an accident of what happens to be running.

With it off, generation still works and nothing else changes:

1. `NlpCardGenerator.isAvailable()` returns `false` — it reads `enabled`, and
   deliberately does not probe `/health`, so provider selection stays
   deterministic.
2. `GenerationService.defaultGenerator()` filters to the available generators and
   walks its preference list, `brain → gemini → heuristic`. The heuristic
   generator "needs no model, no network, no service. Always available, which is
   what makes it the floor the whole pipeline can rest on."
3. The job row records the provider that will actually run, so provenance is
   knowable even if the job dies before it starts.
4. If *no* generator is available (brain off, no Gemini key, heuristic somehow
   removed) `createJob` throws `ServiceUnavailableException: No card generator is
   available in this environment.` — a 503 on the request that asked for cards,
   not a broken job.

So with the brain off, `POST /generation/decks/:deckId` either runs on Gemini or
the heuristic, or is refused up front. Nothing about the source, the deck, the
job list or the review flow changes.

**The inverse is the case worth planning for: enabled but down.** Availability is
configuration, not reachability — "An unreachable brain fails loudly at generation
time instead, which is the right place for it." Turning the switch on without
starting the service produces jobs that fail with a connection error and are
retried once by BullMQ (`attempts: 2`), rather than jobs that quietly fall back
to the heuristic.
That is intentional: a silent downgrade would be invisible, and the whole point of
recording `provider` on the job is that the answer to "why do these cards look
different" is knowable.
