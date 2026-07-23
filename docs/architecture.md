# Where this service sits, and why it is a service

`neurox-brain` turns text into study material: flashcards, quiz questions,
keywords and an extractive summary. It uses spaCy, TF-IDF and TextRank — no
language model, no third-party API, no per-card cost.

This file covers the boundary: why it is a separate process, how a request
travels from the NestJS API to here and back, the three transports, where the
model lives, and how the pieces are deployed. The algorithms are in
`docs/algorithms/`.

---

## The shape of the system

```
  browser
     │
     ▼
  NestJS API  ──►  BullMQ (Redis) "generation" queue
     │                      │
     │                      ▼
     │           generation worker process
     │                      │
     │                      ▼
     │           GenerationService.processJob
     │                      │
     │                      ▼
     │           NlpCardGenerator.generate        ── chunk-by-chunk ──┐
     │                      │                                         │
     │                      ▼                                         │
     │           NeuroxBrainClient.analyse  ◄──────────────────────── │
     │                 │        ▲
     │        HTTP ────┘        │  ──── AMQP (brain.jobs / brain.results)
     │                 ▼        │
     │           ┌─────────────────────────┐
     └───────────┤      neurox-brain       │
      DRAFT rows │  spaCy · TF-IDF · TextRank │
                 └─────────────────────────┘
```

The API is the only intended caller. There is no authentication on this service
and no API-envelope wrapper on responses — the wire format is the pydantic models
in `src/neurox_brain/schemas.py`, mirrored by hand in
`flash-cards-backend/src/integrations/neurox-brain/neurox-brain.types.ts`.

---

## Why a separate process, not a library

The obvious question is why this is not a Python module the Nest app shells out
to, or a Python library imported by something. `src/neurox_brain/__init__.py`
answers it in three parts, and each is worth expanding.

**1. Different runtime, different failure mode.**

> A spaCy model is 300MB of resident memory and a second of load time. Putting
> that inside the API process means every API instance pays for it whether or not
> anyone is generating cards, and a malformed PDF that crashes the parser takes
> the HTTP server with it. Here, it crashes one worker and the API returns a
> 500.

The containment is real but partial, and the partiality is worth stating: the
crash stops at the process boundary, so a segfault in spaCy or a pathological
document kills one container and the API turns it into a failed generation job.
It does not mean every route is defended — `/analyse` catches and reports
exceptions, and the four part routes do not, so a model-missing fault is a 503 on
one route and a bare 500 on the others. See `docs/routes.md`.

Also relevant: this service holds a *stateful* asset the API should not hold —
the IDF corpus in `nlp/corpus.py`. That file is written by the process that owns
it, not shared with anything.

**2. Different scaling axis.**

> Text analysis is CPU-bound and bursty; serving pages is neither. Running them
> in one process means sizing the web tier for the NLP tier.

Cards are generated in bursts when somebody adds a source, and a single
generation job can occupy a core for tens of seconds. A web tier sized for that
is a web tier that is idle most of the time; a web tier sized for traffic
throttles generation under load. Two services means two replica counts and two
resource limits.

**3. It is replaceable.**

> The contract is four JSON endpoints. If the algorithms are later replaced by a
> fine-tuned model, or by an LLM, the API does not change — which is the same
> reasoning that makes the existing `CardGenerator` interface in the Nest app
> swappable.

(The docstring says four; there are five extract routes. The point stands either
way — the contract is the schemas, not the internals.)

This is not aspirational. The API already has three generators behind one
interface — `CardGenerator` in
`flash-cards-backend/src/application/generation/generators/card-generator.interface.ts`,
implemented by `heuristic.generator.ts`, `gemini.generator.ts` and
`nlp.generator.ts`. The brain is one of three implementations and the API's
generation pipeline does not know which one ran beyond a `provider` string on
the job row. Swapping this service for a different one is a change to
`NeuroxBrainClient`, and swapping to a different *provider* is a config change.

The reasoning also explains a design choice inside this service: the algorithms
never import the wire format. `summarise.summarise` returns plain dicts, and
`nlp/analyse.py` is the only place that builds pydantic models. A new response
field cannot change what gets summarised.

---

## The request flow, from the API's point of view

This is the path a generated card actually takes. Every step names the file it
happens in, so it can be checked.

| # | What happens | Where |
| --- | --- | --- |
| 1 | `POST /generation/decks/:deckId` validates the source, picks a generator, writes a `GenerationJob` row with `status = PENDING`, and enqueues. Returns immediately. | `flash-cards-backend/src/api/generation-api/generation.controller.ts` → `src/application/generation/generation.service.ts::createJob` |
| 2 | The queue job is `generate-cards` on the `generation` queue, with `attempts: 2` and exponential backoff from 2000ms. | `src/infra/queue/queue.constants.ts`, `generation.service.ts::createJob` |
| 3 | The worker consumes the job and calls the same service the HTTP side uses. | `src/worker/workers/generation/generation.worker.processor.ts` |
| 4 | `processJob` re-reads the job, marks it `RUNNING`, and calls `writeCards`. | `generation.service.ts` |
| 5 | `writeCards` soft-deletes any cards a previous attempt wrote, chunks `job.source.rawText`, and caps the chunk list. | `generation.service.ts::writeCards`, `src/application/source/chunking.service.ts` |
| 6 | The generator recorded on the job (`brain`) is resolved by provider. | `generation.service.ts::generatorFor` |
| 7 | `NlpCardGenerator.generate` loops the chunks **one at a time**, one `analyse()` call per chunk, until `maxCards` is reached. | `src/application/generation/generators/nlp.generator.ts` |
| 8 | `NeuroxBrainClient.analyse` maps camelCase options to the service's snake_case body and issues the call. | `src/integrations/neurox-brain/neurox-brain.client.ts` |
| 9 | **This service** runs the whole pipeline in one parse and returns an `AnalysisResult`. | `src/neurox_brain/routes.py` → `src/neurox_brain/nlp/analyse.py` |
| 10 | The client returns the payload **unmapped** — `BrainAnalysis` is snake_case on purpose. | `neurox-brain.types.ts` |
| 11 | Only `analysis.cards` is consumed; `card.confidence` and `card.kind` go into the job's usage metadata, ordered best-first. `evidence` is deliberately not used as a hint — it contains the answer. | `nlp.generator.ts` |
| 12 | `FlashCard` rows are created with `status = CARD_STATUS.DRAFT`. Never `ACTIVE`. | `generation.service.ts::writeCards` |
| 13 | The job status is re-read so a cancel that arrived mid-run wins, then set to `SUCCEEDED` with `cardsCreated`, and an activity is recorded. | `generation.service.ts::processJob` |
| 14 | A reader accepts or discards the drafts. | frontend |

A few of these steps carry reasoning worth knowing:

**Why `NlpCardGenerator` sends chunks one at a time.** Its docstring gives the
same three reasons as the Gemini generator: "one request carrying a whole
document produces cards concentrated on whatever the extractor considered the
main theme, a single bad chunk fails the whole job rather than itself, and
stopping early once the card cap is reached is impossible if everything was
already sent." The chunking itself is not done here — the API already has a
chunker and this service's schema explicitly refuses to have a second one.

**Why the generator asks for `maxKeywords: 1`, `maxSummarySentences: 1`,
`maxQuizQuestions: 1`.** Nothing downstream consumes keywords, a summary or quiz
questions from this generator. Asking for the minimum is the cheapest way to say
"none of this" — and because part-route and full-pipeline cost is the same (see
`docs/routes.md`), the minimum is genuinely the cheapest, not a rounding error.

**Why a chunk failure does not fail the job.** The catch is per chunk: "One chunk
failing must not discard the cards the others produced. A malformed chunk is
common — a PDF page of table rows parses into text with no sentences in it — and
failing the job for it would mean losing an entire document to one bad page." A
structured failure is raised only when *no* chunk answered at all.

**Why the service's `confidence` survives all the way to the job.** It is the
only signal separating "the parser found a definition" from "the parser found
something shaped like one", and the difference is large — the published precision
for hand-written definition patterns is around 0.16 on real course text. It is
carried into usage metadata because nothing downstream can reconstruct it.

**Idempotency.** `writeCards` soft-deletes the previous attempt's cards before
writing, and `processJob` returns early on a job already in a terminal state,
because BullMQ retries and "a retry that re-appended cards would silently double
every card the reader is about to review."

---

## The two transports

`NeuroxBrainClient.analyse(text, title, options)` has one signature and two
implementations behind it. That is the whole point: the generator above it does
not know or care which is in use, and the choice "is genuinely a deployment one:
a direct call is right when the service is next to the API and the text is small,
and a queue is right when it is not."

### HTTP — the default, on both sides

`NEUROX_BRAIN_TRANSPORT=http` (API) and `BRAIN_TRANSPORT=http` (service). The
client POSTs to `{baseUrl}/analyse` with `AbortSignal.timeout(timeoutMs)`,
default 30s. `AbortSignal.timeout` rather than a manual timer "because it aborts
the socket as well as rejecting the promise, so a slow request does not leave a
connection held until the server eventually gives up."

HTTP is not a placeholder. From the client's docstring: "The generator sends one
chunk at a time — the same shape the Gemini generator uses, and for the same
reasons — and a chunk analyses in roughly 1.5 seconds, so holding a request open
is entirely reasonable. The queue exists for the case where it stops being
reasonable, not because synchronous calls are embarrassing."

The routes are also always available in every configuration, which is what makes
the service testable by hand.

### AMQP — when a request/response shape stops fitting

The reason a queue exists at all, from `transport/amqp.py`:

> Because the API already has a queue for generation — BullMQ on Redis — and a
> large document does not fit the request/response shape. Holding an HTTP
> connection open for thirty seconds of parsing invites a proxy timeout, and a
> timeout leaves both sides unsure whether the work happened. A queue makes the
> handoff explicit: the job is durable, the worker acknowledges when it is done,
> and a worker that dies mid-run has its message redelivered rather than lost.

And why RabbitMQ rather than reusing BullMQ: "BullMQ is a Node library; its wire
format is Redis data structures that a Python process would have to reimplement
and keep in step with a moving upstream. RabbitMQ speaks AMQP, which is a
protocol with clients in both languages, and the backend already provisions a
RabbitMQ service in every compose file." (That same infrastructure note is in
`flash-cards-backend/.env.template`, where the RabbitMQ service is described as
currently unused and kept for this purpose.)

#### Topology

```
  API  ──publish──▶  brain.jobs     ──consume──▶  neurox-brain worker
                                                         │
  API  ◀──consume──  brain.results  ◀──publish───────────┘
```

| Element | Value | Declared by | Consumed by |
| --- | --- | --- | --- |
| Job queue | `brain.jobs` (`BRAIN_AMQP_JOB_QUEUE` / `NEUROX_BRAIN_JOB_QUEUE`), durable | the worker, on start | the worker |
| Result queue | `brain.results` (`BRAIN_AMQP_RESULT_QUEUE` / `NEUROX_BRAIN_RESULT_QUEUE`), durable | the worker **and** the API, on start | the API |

Both queues are durable, both are declared by the worker, and the result queue is
additionally asserted by the API's channel setup. Each side says why:

- Worker: "Declared as well as published to, so a worker that starts before the
  API has ever sent anything does not fail to publish its first result."
- API: "Declared here as well as by the worker, so the API can start first and
  still receive results. Declaring is idempotent as long as the arguments match,
  which is why both sides use the same durable flag."

There is no exchange and no routing key beyond the queue name: both sides publish
to the default exchange with the queue name as the routing key. `amqp.py`:
results go to a queue "rather than being published to an exchange with routing
keys, because there is exactly one consumer and one kind of message; an exchange
would add a routing concept that never varies."

#### Message shapes

Job, published by the API (`JobEnvelope` in `schemas.py`):

```json
{
  "job_id": "2f1c…",
  "correlation_id": "2f1c…",
  "reply_to": "brain.results",
  "text": "…",
  "title": "Data Structures",
  "options": {"max_cards": 12, "include_cloze": true, "max_keywords": 1,
              "max_summary_sentences": 1, "max_quiz_questions": 1}
}
```

The API sends `deliveryMode: 2` (persistent), so the broker writes it to disk
before acknowledging.

Result, published by the worker:

```json
{"jobId": "2f1c…", "correlationId": "2f1c…", "ok": true, "error": null, "result": {…}}
```

`result` is the full `AnalysisResult` (`result.model_dump(mode="json")`), or
`null` with `ok: false` and an `error` string when the analysis raised.

#### How correlation works

The API generates one `randomUUID()`, uses it for both `job_id` and
`correlation_id` in the envelope, sets it as the AMQP `correlationId` property,
and stores its `resolve`/`reject` pair in a `pending` map keyed by that UUID.

The worker reads the reply target from the envelope rather than from its own
configuration — `envelope.reply_to or settings.amqp_result_queue` — "so that one
worker can serve several callers with different result queues, and so a caller
can use a temporary queue of its own without arranging anything here." It sets
the AMQP `correlation_id` property to `envelope.correlation_id or
envelope.job_id` as well as putting both in the body, "so a consumer can filter
without parsing — useful for a consumer that only wants its own results off a
shared queue."

The API resolves on `parsed.correlationId ?? parsed.jobId`. A result with no
waiting entry is normal — the job timed out and the answer arrived late — and is
discarded at debug level rather than logged loudly.

#### Why the worker acknowledges only after publishing

This is the ordering rule that makes the transport safe, and `amqp.py` states it
directly:

> A job is acknowledged only after its result has been published *and confirmed*
> by the broker. Acknowledging first would mean a worker crash between the two
> loses the result silently — the job would be marked done and the API would wait
> forever.

The `await channel.default_exchange.publish(...)` completes before
`await message.ack()`, so a broker that never confirmed leaves the message
unacknowledged and the job is redelivered.

The other two paths are handled differently, on purpose:

| Situation | What the worker does | Why |
| --- | --- | --- |
| Message does not validate as `JobEnvelope` | logs and `ack()`s | "A malformed message can never succeed on redelivery… rejecting with requeue would put it straight back at the head of the queue, forever." |
| The analysis raises | publishes `ok: false` with the error, then `nack(requeue=False)` | Requeueing a deterministic failure is "an infinite loop that also blocks every job behind it". `requeue=False` dead-letters it if a DLQ is configured and drops it otherwise. |
| The failure result itself cannot be published | logs the exception and still `nack(requeue=False)` | So the broker redelivers rather than dropping it silently. |

#### Running the analysis off the event loop

The handler calls `asyncio.to_thread(analyse_module.analyse, ...)` rather than
`analyse(...)` directly: "The analysis is CPU-bound and synchronous, and running
it on the event loop would block the AMQP heartbeats — the broker would decide
the worker was dead and redeliver the job it is in the middle of doing." Note
that this moves one blocking call off the loop; it does not parallelise parsing
within the process.

#### Connection and concurrency

- `aio_pika.connect_robust`, wrapped in `asyncio.wait_for` with a 30-second
  bound. Robust connections reconnect indefinitely; the bound exists "where a
  wrong URL should fail loudly rather than hanging a container forever."
- Worker prefetch is `BRAIN_AMQP_PREFETCH`, default 1: "Analysis is CPU-bound and
  the process is single-threaded for parsing; pulling a second job only means it
  waits while the broker's unacknowledged-message timer runs. Scale by running
  more processes, not by raising this."
- The API's result consumer uses `prefetch(1)` too, "because results are small
  and this process is not the bottleneck".
- The API's `timeoutMs` (default 30s) applies to the queue path as well. A job
  that outlives it is abandoned client-side and its eventual result is discarded;
  the brain does not know and does not stop.

#### A note on `BRAIN_TRANSPORT`

On the Python side, `settings.transport` is **reported, not dispatched**: it
appears in the start-up log line (`main.py`) and in `GET /stats`, and nothing
else reads it. Which transport actually runs is decided by which entry point you
start — `uvicorn neurox_brain.main:app` for the HTTP routes, `python -m
neurox_brain.worker` for the queue consumer. `worker.py` imports
`transport.amqp` unconditionally, so running the worker requires `aio-pika` and a
reachable broker regardless of what `BRAIN_TRANSPORT` says.

On the API side, `NEUROX_BRAIN_TRANSPORT` *does* dispatch, and the AMQP client is
only set up when the brain is both enabled and configured for AMQP: `onModuleInit`
returns early if `!this.enabled || this.transport !== "amqp"`. The `amqp` stack
is imported lazily "so an HTTP deployment never loads the AMQP stack, and so a
missing optional dependency cannot break start-up for everyone."

### Webhook — the third option, and the weakest

`transport/webhook.py` describes the callback shape: the API sends work and
forgets it, and this service POSTs the result to a URL when it is done. It is
described as "the fallback the plan asked for: it needs no message broker, only
an HTTP endpoint on the API side."

**What this is not**, in the module's own words:

> It is not a substitute for the AMQP path when RabbitMQ is available. A webhook
> has no delivery guarantee — if the API is restarting when the post arrives, the
> result is gone — whereas a durable queue retries until the consumer
> acknowledges. The webhook sender retries a few times, which covers a blip and
> not an outage, and that difference is the reason to prefer the queue when there
> is one.

The retry policy is honest about its own size: `MAX_ATTEMPTS = 3` with
`BACKOFF_SECONDS = (0.5, 2.0, 5.0)` — "Three attempts over roughly seven seconds
rides out a restart; it is not meant to survive an outage." Each attempt has a
15-second request timeout. A 2xx is success; a non-429 4xx is treated as the
receiver rejecting the payload and is *not* retried ("retrying sends the same
rejected bytes three times and delays the real error"); 5xx and 429 are retried.
`deliver()` returns a bool rather than raising, "because a failed delivery is a
fact the caller records against the job, not an exception it can do anything
about."

**The signature.** Every post carries an HMAC-SHA256 of the body in
`X-Brain-Signature`, computed over the raw bytes:

```python
body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
```

Signing the raw bytes and not a re-serialised dict is deliberate: "the receiving
side verifies against exactly what was sent, and JSON key order is not guaranteed
to survive a parse-and-dump round trip. Signing a re-serialisation is how
signature checks fail intermittently and for no visible reason."

The API's side of that is `NeuroxBrainClient.verifySignature`, which recomputes
the digest and compares with `timingSafeEqual` (length checked first, because
`timingSafeEqual` throws on a length mismatch) — "string comparison exits at the
first differing byte and the time it takes therefore leaks how much of a guessed
signature was right."

Why it matters at all is stated in both codebases, because it is the reason the
secret is required rather than nice: "any host that can reach the API's callback
endpoint can inject a job result — and a job result is what writes cards into
somebody's deck."

**Status of this transport in the current tree: not wired up.** `sign()` and
`deliver()` are defined and documented, but nothing calls them. In this
repository the only references to `transport/webhook.py` outside the module
itself are the transport package docstring and a pointer in `schemas.py`; neither
a route nor `worker.py` accepts a callback URL. On the API side `verifySignature`
is defined and exported but no controller calls it — there is no callback
endpoint. So the interface is specified and the code exists, but the path is not
present on either end yet; what runs today is HTTP or AMQP. If it is wired up,
the pieces above are the contract to build against.

---

## Where the model fits

**One shared spaCy pipeline per process.** `nlp/pipeline.py::get_nlp` is
`@lru_cache(maxsize=1)`: "Loading a spaCy model costs about a second and 100–300MB
of resident memory. Loading one per request would make the service unusable;
loading one per process and sharing it is the intended arrangement. The object is
read-only after construction, so sharing is safe." `lru_cache` rather than a
module global with an `if _nlp is None` check, because CPython's cache is
thread-safe for this and a hand-rolled check is not.

A `threading.Lock` (`_LOAD_LOCK`) still wraps the first load: "`lru_cache` is
safe, but the *first* call is slow and both callers would otherwise sit inside
`spacy.load` at once, doubling peak memory."

**One parse per document, shared by every stage.** `parse()` runs the model once
and returns a frozen `Document` carrying the spaCy `Doc`, the filtered sentence
list and the title. The docstring: "Running `nlp()` five times over the same text
would parse it five times and produce five documents that could disagree." Both
the sentences and the "usable sentence" filter (length bounds, minimum words,
must contain a verb, must be mostly alphabetic) come from the parser, which is
why a model with a dependency parser is required — splitting on `.` breaks on the
abbreviations academic prose is full of.

**HTTP: loaded in the lifespan, on purpose not eagerly.**

> Loading the spaCy model inside the lifespan handler means the process starts,
> binds its port and answers `/health` *while* the model loads, rather than
> blocking startup for a second or more. A container orchestrator reading
> `/health` therefore sees a live process that reports `loading`, which it can
> distinguish from a process that never came up.

`GET /health` reports `model_loaded` from `get_nlp.cache_info().currsize` without
triggering a load, so a probe never pays the 300MB. Note the consequence, stated
plainly in `docs/routes.md`: a *misconfigured* model also reports `loading`
forever, because nothing tries to load it until a request does.

**The worker: loaded eagerly.** From `worker.py`'s docstring: "Unlike `main.py`,
this loads the model eagerly. There is no port to bind and nothing to report
progress to, so a worker that is not ready to work should not claim to be
running." A worker that started consuming before the model was loaded would take
jobs it could not answer; it loads first, then connects.

**WordNet is loaded at start-up too**, in both entry points, before anything can
touch it from a thread. `distractors.ensure_wordnet_loaded` explains why that is
"a production bug fix, not a nicety": NLTK loads corpora lazily through
`LazyCorpusLoader`, which is not thread-safe — two threads reaching a corpus for
the first time at once can leave it half-initialised with failures like
`'WordNetCorpusReader' object has no attribute '_LazyCorpusLoader__args'`. It
also fails *late* by nature: a missing corpus raises `LookupError` when first
*used*, not at import, "so the problem passes unit tests and appears on the first
real request."

What a missing model looks like from outside is in `docs/routes.md`: 503 on
`/analyse`, a bare 500 on the part routes, `loading` on `/health`.

---

## Deployment shape

**Two processes, and they should be two containers.** `worker.py`'s docstring:

> A separate process from the HTTP service, and it should be a separate
> *container* too. The reasoning is the one in `transport/amqp.py`: analysis is
> CPU-bound and blocking, so a worker busy on a large document cannot answer a
> health probe or serve a synchronous request. Running the two together means the
> API's latency is set by whatever the worker happens to be doing.

| Process | Started with | Serves |
| --- | --- | --- |
| HTTP service | `uvicorn neurox_brain.main:app --host 0.0.0.0 --port 8000` | `/health`, `/stats`, and the five extract routes |
| Queue worker | `python -m neurox_brain.worker` | consumes `brain.jobs`, publishes `brain.results` |

`pyproject.toml` declares two console scripts:

```toml
[project.scripts]
neurox-brain = "neurox_brain.main:app"
neurox-brain-worker = "neurox_brain.worker:main"
```

Only the second is a runnable command: it points at the `main()` function.
`neurox-brain` points at the ASGI application object, which is not a zero-argument
callable — invoking it raises `TypeError: FastAPI.__call__() missing 3 required
positional arguments: 'scope', 'receive', and 'send'`. Use the `uvicorn`
invocation shown in `main.py`'s own docstring for the HTTP process.

**Scale with processes, not threads.**

> spaCy's `nlp` object is not thread-safe for concurrent `pipe` calls across
> threads with different docs in flight in every configuration; the safe and
> simple arrangement is one worker process per core, each single-threaded. This
> is that count, and it is also what `uvicorn --workers` should match.

That is the comment on `BRAIN_WORKERS` (default 2). Two things to know about it:

- `settings.workers` is currently read nowhere outside `config.py`. Nothing in
  the service starts, sizes or limits anything from it. The number of HTTP
  processes is whatever `uvicorn --workers` is given, and the number of queue
  workers is however many worker containers are run. `BRAIN_WORKERS` is the
  documented value those should match, not a control.
- `n_process` is never passed to the pipeline — `parse()` calls `nlp(cleaned)`
  with no parallelism arguments, so each process parses on one thread. The only
  thread use in the service is `asyncio.to_thread` in the AMQP handler, which
  moves one already-blocking call off the event loop rather than splitting work
  up.

The practical rule: **one model per process, and one process per core you are
willing to spend.** A given deployment's model memory is `(HTTP workers + queue
workers) × 100–300MB` for spaCy plus the process itself, so a two-worker HTTP
service alongside two queue workers is four models resident.

**Configuration** is entirely environment variables read once at import
(`config.py`); there is no config file and no `.env` loading — see
`docs/operations.md` for every variable and its default.

**Nothing here is required for the API to work.** `NEUROX_BRAIN_ENABLED` defaults
to `false`, and with it off the generator chain falls through to Gemini or the
heuristic. Running this service is opt-in, and its absence is a supported state
rather than an outage — see `docs/operations.md`.
