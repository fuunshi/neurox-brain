# The HTTP surface

Seven routes, registered by `src/neurox_brain/routes.py` on the FastAPI app built
in `src/neurox_brain/main.py`. There is no authentication: the API is the only
intended caller and it calls server-to-server. Start it with

```
uvicorn neurox_brain.main:app --host 0.0.0.0 --port 8000
```

Everything below was run against a live instance; the response bodies are copied
from real output, not written by hand. `POST` bodies are JSON and every response
is JSON.

> `routes.py`'s module docstring says "six routes". Seven are registered — the
> docstring is counting the five extract routes (one `analyse` plus the four part
> routes) and not the two `meta` ones. The OpenAPI document at `/openapi.json` is
> the authority.

## The error envelope

Every failure that the route handlers and the validation handler produce has this
shape:

| Field | Type | Notes |
| --- | --- | --- |
| `status` | `false` | Always `false`. `schemas.ErrorResponse` types it `Literal[False]`. |
| `statusCode` | `int` | Duplicated from the HTTP status. |
| `message` | `str \| list[str]` | A single sentence for a handled failure; a list of per-field lines for a validation failure. |
| `path` | `str` | `request.url.path` — the path only, no query string. |

```json
{"status":false,"statusCode":413,"message":"Text is 140 characters; the limit is 100.","path":"/analyse"}
```

It is this shape, rather than FastAPI's `{"detail": ...}`, because it is the same
shape the NestJS global exception filter produces. From `schemas.ErrorResponse`:
"one error format across both services means the API's error normaliser does not
need a second branch — and a second branch is where the two drift."

The route-level failures go through `_error()` in `routes.py`, a nine-line
helper. Validation failures do not, because FastAPI raises them before the
handler runs; `main.py` installs a handler for `RequestValidationError` that
flattens FastAPI's `exc.errors()` into the same envelope:

```python
messages = [
    f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
    for error in exc.errors()
]
```

Note the `[1:]`: the first element of `loc` is the request part (`body`) and is
dropped, so an error on `text` reads `text: ...` and one on `options.max_cards`
reads `options.max_cards: ...`. The handler's own docstring gives the reason it
exists at all: without it, the API's error normaliser "would need a third branch
that only ever fires when *this* service rejects a payload — the least-tested
path, on the errors hardest to reproduce."

Real validation responses, all `422`:

```json
{"status":false,"statusCode":422,"message":["text: Field required"],"path":"/analyse"}
{"status":false,"statusCode":422,"message":["text: String should have at least 1 character"],"path":"/analyse"}
{"status":false,"statusCode":422,"message":["text: Input should be a valid string","options.max_cards: Input should be greater than or equal to 1"],"path":"/analyse"}
{"status":false,"statusCode":422,"message":["1: JSON decode error"],"path":"/analyse"}
```

The last one is an unparseable body; `loc` is `('body', 1)`, so the flattened
prefix is just the character index. If `exc.errors()` were ever empty the handler
falls back to `["The request could not be validated."]`.

## Route index

| Method | Path | `response_model` | Tag | What it is for |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | `HealthResponse` | `meta` | Liveness, and what the process can currently do. |
| `GET` | `/stats` | *(none)* | `meta` | What the IDF corpus has learned. |
| `POST` | `/analyse` | `AnalysisResult` | `extract` | Everything, in one parse. What the API calls. |
| `POST` | `/cards` | `list[GeneratedCard]` | `extract` | Flashcards only. |
| `POST` | `/quiz` | `list[GeneratedQuestion]` | `extract` | Multiple-choice questions only. |
| `POST` | `/keywords` | `list[Keyword]` | `extract` | TF-IDF keywords and keyphrases only. |
| `POST` | `/summary` | `list[SummarySentence]` | `extract` | Extractive summary only. |

The two `meta` routes have no request body. `response_model` on the rest is not
decoration: it is the filter that decides what leaves the process, so a field
added to an internal dataclass does not silently become part of the wire
contract.

---

## `GET /health`

Liveness, and what the service is currently capable of.

It exists separately from everything else because a container runtime needs to
know whether the process is up without paying for the thing the process does.
`model_loaded` is computed by `pipeline.is_model_loaded()`, which reads
`get_nlp.cache_info().currsize` — it never triggers the load. From the route
docstring: "A health probe that costs 300MB the first time it is called is not a
probe, it is a surprise."

| Field | Type | Meaning |
| --- | --- | --- |
| `status` | `"ok" \| "loading" \| "degraded"` | `ok` once the model is resident, `loading` before that. |
| `model` | `str` | `BRAIN_SPACY_MODEL` — the configured name, whether or not it is installed. |
| `model_loaded` | `bool` | Whether `spacy.load` has run in this process. |
| `version` | `str` | `neurox_brain.__version__`, currently `0.1.0`. |
| `uptime_seconds` | `float` | Seconds since the module was imported, one decimal, measured on `time.monotonic()`. |

No request body. No documented error response.

Before anything has been analysed:

```bash
curl -s http://localhost:8000/health
```

```json
{"status":"loading","model":"en_core_web_sm","model_loaded":false,"version":"0.1.0","uptime_seconds":11.1}
```

After the first analysis:

```json
{"status":"ok","model":"en_core_web_sm","model_loaded":true,"version":"0.1.0","uptime_seconds":74.5}
```

**`status` is never `degraded`, and `"degraded"` in the schema is unreachable.**
The route docstring is explicit about why the field is not used for a missing
vector model: running `en_core_web_sm` "is a supported configuration with a known
weaker distractor stage, not a fault, and reporting it as a fault would train an
operator to ignore the field." The literal is in `HealthResponse` because the
API's mirror type declares it; nothing sets it.

Worth knowing when reading `status`: a **misconfigured model name also reports
`loading`**, forever. Verified by starting the service with
`BRAIN_SPACY_MODEL=en_core_web_does_not_exist`:

```json
{"status":"loading","model":"en_core_web_does_not_exist","model_loaded":false,"version":"0.1.0","uptime_seconds":13.2}
```

`model_loaded: false` plus a `status` that never becomes `ok` is the signal. The
failure surfaces properly on `/analyse`, as a 503.

---

## `GET /stats`

What the corpus has learned. There is no `response_model`, so the dict is the
contract:

| Field | Type | Meaning |
| --- | --- | --- |
| `documents` | `int` | Documents observed by this process since it started, loaded from the corpus file at start-up. |
| `model` | `str` | `BRAIN_SPACY_MODEL`. |
| `transport` | `str` | `BRAIN_TRANSPORT`, reported as configured. |

```bash
curl -s http://localhost:8000/stats
```

```json
{"documents":5,"model":"en_core_web_sm","transport":"http"}
```

It exists because half of every keyword score depends on it. The docstring: "The
IDF half of every keyword score depends on this, so it is worth being able to
see: `documents` climbing means keyword weights are getting closer to real IDF
and further from the cold-start prior." Below `MIN_DOCUMENTS_FOR_IDF` (5) the
weights are a blend of the cold-start prior and whatever evidence exists — see
`docs/operations.md`.

`documents` is incremented once per completed pipeline run, so it moves on every
`/analyse`, `/cards`, `/quiz`, `/keywords` and `/summary` request — including
requests that returned empty lists. It is not a count of useful documents. It is
persisted on shutdown only; see `docs/operations.md`.

---

## `POST /analyse`

Everything, in one pass: cards, quiz questions, keywords and a summary.

**Why it exists separately.** It is the route the API's generation worker uses,
and it is deliberately one request rather than four, "because the four would each
pay for a parse of the same text." Parsing is the expensive stage and every
extractor needs a different view of the same parse, so the whole pipeline is run
once and all four sections are returned together.

### Request body — `AnalyseRequest`

| Field | Type | Required | Default | Constraints |
| --- | --- | --- | --- | --- |
| `text` | `str` | yes | — | 1–500,000 characters. |
| `title` | `str \| null` | no | `null` | ≤ 300 characters. |
| `options` | `AnalysisOptions` | no | `{}` → all defaults | see below |

`title` is "used only as a fallback topic label. It is not prepended to the text:
doing so would make every sentence look related to the title and flatten the term
weights that pick keywords."

`text` chunking is the caller's business, from the field description: "the API
already has a chunker, and having two would mean two answers to 'what is a chunk'
that disagree." The API's chunker is `ChunkingService` in
`neurox-backend/src/application/source/chunking.service.ts`.

### `options` — `AnalysisOptions`

| Field | Type | Default | Constraints | Effect |
| --- | --- | --- | --- | --- |
| `max_cards` | `int` | `25` | 1–200 | Upper bound on flashcards, definitional plus cloze. A cap, not a target. |
| `max_quiz_questions` | `int` | `10` | 1–100 | Upper bound on returned questions. |
| `include_cloze` | `bool` | `true` | — | Whether cloze cards are emitted as well as definitional ones. |
| `max_keywords` | `int` | `20` | 1–100 | Upper bound on keywords and keyphrases. |
| `max_summary_sentences` | `int` | `5` | 1–50 | Upper bound on summary sentences. |

`max_cards` carries the reasoning for the whole group: "A cap, not a target — a
short text yields fewer, and padding to reach a number is how a deck fills with
questions nobody would ask." `include_cloze` is off "when only definitional cards
are wanted, e.g. for a deck that will be exported to Anki as Basic."

Note that `max_cards` also drives definition extraction: the pipeline calls
`definitions.extract(doc, limit=max_cards * 2)` and then filters down, so a
larger `max_cards` costs a little more work in that stage too.

### Response — `AnalysisResult`

| Field | Type | Notes |
| --- | --- | --- |
| `cards` | `list[GeneratedCard]` | May be empty. |
| `quiz` | `list[GeneratedQuestion]` | May be empty. |
| `keywords` | `list[Keyword]` | May be empty. |
| `summary` | `list[SummarySentence]` | May be empty. |
| `stats` | `AnalysisStats` | Always present. |

`GeneratedCard`

| Field | Type | Notes |
| --- | --- | --- |
| `front` | `str` | The term, or a sentence with a `_____` blank for a `CLOZE` card. |
| `back` | `str` | The definition, or the blanked term with its article stripped. |
| `hint` | `str \| null` | Always `null` from this service. Nothing sets it. |
| `kind` | `"DEFINITION" \| "CLOZE" \| "RELATION"` | `"RELATION"` is declared in the schema and never produced. |
| `confidence` | `float` | 0.0–1.0. See `definitions._confidence`. |
| `evidence` | `str` | The source sentence, whitespace-collapsed to one line. |

`GeneratedQuestion`

| Field | Type | Notes |
| --- | --- | --- |
| `format` | `"MULTIPLE_CHOICE" \| "CLOZE"` | `build_quiz` only ever emits `MULTIPLE_CHOICE`; `"CLOZE"` is declared for the API's benefit. |
| `prompt` | `str` | Always `f"What is {term}?"`. |
| `options` | `list[str]` | Schema allows 2–8. Always 4 from `build_quiz`: the correct definition plus three distractors. |
| `correct_index` | `int` | ≥ 0. **Must be stripped before a question reaches a reader** — the route docstring and `schemas.GeneratedQuestion` both say so. |
| `explanation` | `str \| null` | The correct definition, repeated. |
| `evidence` | `str` | The definitional sentence. |

`Keyword`: `term` (`str`), `score` (`float`, ≥ 0), `count` (`int`, ≥ 1).
`SummarySentence`: `text` (`str`), `score` (`float`, ≥ 0), `index` (`int`, ≥ 0).

`AnalysisStats`

| Field | Type | Meaning |
| --- | --- | --- |
| `sentences` | `int` | Usable sentences after filtering (`pipeline.is_usable_sentence`), not raw sentences. |
| `tokens` | `int` | Non-whitespace tokens in the parsed document. |
| `chunks` | `int` | Noun chunks the parser produced. |
| `elapsed_ms` | `int` | The whole pipeline, wall clock. |
| `model` | `str` | The model that actually ran. |

`stats` exists so a caller can distinguish bad input from a broken service: it is
"the only way a caller can tell 'this text has nothing to extract' from
'something went wrong and the extractor returned empty', and those need different
responses."

### Errors

| Status | When | Body |
| --- | --- | --- |
| `413` | `len(text) > settings.max_text_chars`. | Envelope, message `Text is N characters; the limit is M.` |
| `422` | Schema validation, or unparseable JSON. | Envelope with a list of messages. |
| `500` | Any exception from the pipeline that is not the model-missing `RuntimeError`. | Envelope, message `Analysis failed: ...`. Traceback goes to the log. |
| `503` | The pipeline raised `RuntimeError` — in practice, the spaCy model is not installed. | Envelope, message from `pipeline.get_nlp`. |

The 413 is a deliberate extra check on top of the schema. `AnalyseRequest.text`
caps at 500,000 characters, and the default `BRAIN_MAX_TEXT_CHARS` is also
500,000, so **the 413 branch is only reachable when an operator lowers
`BRAIN_MAX_TEXT_CHARS`** — otherwise the schema's 422 fires first, with
`text: String should have at most 500000 characters`. The reason for the second
limit is in `config.py`: "The API's own source cap is 500k characters, but a
single request that large is 30+ seconds of parsing, so this is deliberately
lower."

The 503 rather than a 5xx-without-meaning, from the handler's comment: "That is a
deployment fault rather than a bad request, so it is a 503 — the call will
succeed once the service is fixed, and retrying is reasonable."

The broad `except Exception` is deliberate. From the comment: "An unhandled
extractor failure must not take the worker down or leave the API waiting on a
request that will never answer; the traceback goes to the log and the caller gets
a retryable error. Analysing arbitrary PDF text means meeting arbitrary text."

### Example

```bash
curl -s -X POST http://localhost:8000/analyse \
  -H 'content-type: application/json' \
  -d '{
    "text": "A stack is a linear data structure that follows the Last In First Out principle. The push operation adds an element to the top of the stack, and the pop operation removes the element at the top. A queue is a linear data structure that follows the First In First Out principle. Unlike a stack, a queue is open at both its ends.",
    "title": "Data Structures",
    "options": {"max_cards": 3, "max_quiz_questions": 2, "max_keywords": 5, "max_summary_sentences": 2}
  }'
```

Real response, from the first request against an empty corpus — 4 usable
sentences, so the summariser used its frequency fallback rather than TextRank
(see `/summary` below):

```json
{
  "cards": [
    {
      "front": "stack",
      "back": "A linear data structure that follows the Last In First Out principle.",
      "hint": null,
      "kind": "DEFINITION",
      "confidence": 0.8500000000000001,
      "evidence": "A stack is a linear data structure that follows the Last In First Out principle."
    },
    {
      "front": "queue",
      "back": "A linear data structure that follows the First In First Out principle.",
      "hint": null,
      "kind": "DEFINITION",
      "confidence": 0.8500000000000001,
      "evidence": "A queue is a linear data structure that follows the First In First Out principle."
    },
    {
      "front": "A _____ is a linear data structure that follows the Last In First Out principle.",
      "back": "stack",
      "hint": null,
      "kind": "CLOZE",
      "confidence": 0.8500000000000001,
      "evidence": "A stack is a linear data structure that follows the Last In First Out principle."
    }
  ],
  "quiz": [],
  "keywords": [
    {"term": "Stack", "score": 0.847, "count": 2},
    {"term": "Queue", "score": 0.7981, "count": 2},
    {"term": "Push operation", "score": 0.5109, "count": 1},
    {"term": "Pop operation", "score": 0.5004, "count": 1},
    {"term": "Element", "score": 0.4858, "count": 1}
  ],
  "summary": [
    {
      "text": "A stack is a linear data structure that follows the Last In First Out principle.",
      "score": 0.158889,
      "index": 0
    },
    {
      "text": "The push operation adds an element to the top of the stack, and the pop operation removes the element at the top.",
      "score": 0.179167,
      "index": 1
    }
  ],
  "stats": {"sentences": 4, "tokens": 69, "chunks": 19, "elapsed_ms": 617, "model": "en_core_web_sm"}
}
```

Two things to notice in that output:

- `confidence` is `0.8500000000000001`. The scores are a weighted sum of floats
  and are not rounded (`definitions._confidence` clamps to `[0, 1]` and returns
  as-is), unlike `Keyword.score`, which is rounded to four places at
  construction. If a caller displays the number, format it.
- `quiz` is empty, and that is the distractor rule working rather than a
  failure. Every option on a definition question is another *definition from the
  same document* (`distractors.for_definitions`, whose preferred source is "other
  items from the same document"), and this text defines two terms, so there is
  one candidate distractor for each. `distractors.has_enough` wants three, and
  `build_quiz` drops a question it cannot complete rather than padding it. The
  `/quiz` example below, on a text with four definitions, returns two questions.

`cards` shows the ordering the pipeline produces: definitional cards first, then
cloze, both by descending confidence. Three cards came back for `max_cards: 3`
and none was padded — the text simply had that many.

---

## The part routes

`/cards`, `/quiz`, `/keywords` and `/summary` all take the same body and all run
the same pipeline. They are the "I already have the material, I just want more of
one thing" mood described in the `routes.py` docstring, as opposed to `/analyse`'s
"here is a chapter, give me everything".

**Why they are not a separate code path.** "They run the same pipeline and return
one section, because splitting the pipeline would mean a second code path that
could disagree with the first about what a sentence is."

### Request body — `TextRequest`

| Field | Type | Required | Default | Constraints |
| --- | --- | --- | --- | --- |
| `text` | `str` | yes | — | 1–500,000 characters. |
| `title` | `str \| null` | no | `null` | ≤ 300 characters. |
| `limit` | `int` | no | `25` | 1–200. |

`limit` is a single field standing in for a different option per route, which is
the one sharp edge on this surface:

| Route | `limit` becomes | That option's own maximum |
| --- | --- | --- |
| `/cards` | `AnalysisOptions(max_cards=limit)` | 200 |
| `/quiz` | `AnalysisOptions(max_quiz_questions=limit)` | **100** |
| `/keywords` | `AnalysisOptions(max_keywords=limit)` | **100** |
| `/summary` | `AnalysisOptions(max_summary_sentences=limit)` | **50** |

### `POST /cards`

Flashcards only. Definitional cards first, then cloze cards, ordered by the
confidence the definition extractor assigned. `limit` bounds the total.

Cloze cards come from the definition sentences, one per definition, and only
where `cloze.from_term` accepts the blank; they inherit the definition's
confidence. `hint` is always null. `evidence` carries the source sentence: "a
generated card is a proposal, and a proposal is easier to judge with its source
attached."

```bash
curl -s -X POST http://localhost:8000/cards \
  -H 'content-type: application/json' \
  -d '{"text":"A stack is a linear data structure that follows the Last In First Out principle. The push operation adds an element to the top of the stack.","limit":2}'
```

```json
[
  {
    "front": "stack",
    "back": "A linear data structure that follows the Last In First Out principle.",
    "hint": null,
    "kind": "DEFINITION",
    "confidence": 0.8,
    "evidence": "A stack is a linear data structure that follows the Last In First Out principle."
  },
  {
    "front": "A _____ is a linear data structure that follows the Last In First Out principle.",
    "back": "stack",
    "hint": null,
    "kind": "CLOZE",
    "confidence": 0.8,
    "evidence": "A stack is a linear data structure that follows the Last In First Out principle."
  }
]
```

Errors: `422` for a bad body or `limit` outside 1–200. **No 413 and no 500
handling** — see "Where the envelope does not apply" below.

### `POST /quiz`

Multiple-choice questions whose options are definitions. Every returned question
has a correct answer and at least three distractors; questions that could not be
completed are dropped rather than padded.

The quote from the route docstring: "`correct_index` is included and must be
stripped before the question reaches a reader." The API mirrors that rule — the
`NlpCardGenerator` in the NestJS app consumes only `analysis.cards` and discards
the quiz entirely.

A question is emitted only when the definition cleared
`MIN_QUIZ_CONFIDENCE` (0.55) and three distractors were found
(`distractors.has_enough` requires ≥ 3), and at most
`MAX_QUIZ_FROM_DEFINITIONS` (12) definitions are turned into questions per
document. Those constants are in `nlp/analyse.py` and are not configurable.

```bash
curl -s -X POST http://localhost:8000/quiz \
  -H 'content-type: application/json' \
  -d '{"text":"A stack is a linear data structure that follows the Last In First Out principle. A queue is a linear data structure that follows the First In First Out principle. A binary search tree is a data structure in which each node has at most two children. Recursion is a technique in which a function calls itself directly or indirectly.","limit":2}'
```

```json
[
  {
    "format": "MULTIPLE_CHOICE",
    "prompt": "What is Recursion?",
    "options": [
      "A data structure in which each node has at most two children.",
      "A linear data structure that follows the First In First Out principle.",
      "A technique in which a function calls itself directly.",
      "A linear data structure that follows the Last In First Out principle."
    ],
    "correct_index": 2,
    "explanation": "A technique in which a function calls itself directly.",
    "evidence": "Recursion is a technique in which a function calls itself directly or indirectly."
  },
  {
    "format": "MULTIPLE_CHOICE",
    "prompt": "What is stack?",
    "options": [
      "A linear data structure that follows the First In First Out principle.",
      "A technique in which a function calls itself directly.",
      "A data structure in which each node has at most two children.",
      "A linear data structure that follows the Last In First Out principle."
    ],
    "correct_index": 3,
    "explanation": "A linear data structure that follows the Last In First Out principle.",
    "evidence": "A stack is a linear data structure that follows the Last In First Out principle."
  }
]
```

Option order is deterministic: `build_quiz` shuffles with a `random.Random`
seeded from a SHA-256 digest of the term and definition, so the same input yields
the same paper while the answer is not always first.

Errors: `422` for a bad body. **`limit` in 101–200 passes request validation and
then fails inside the handler** — see below.

### `POST /keywords`

TF-IDF keywords and keyphrases, best first. Scores are the product of a
sublinear term frequency, the corpus IDF, a bounded phrase bonus and a small
early-position bonus (`nlp/keywords.py::extract`).

Scores are comparable *within* one response, not across responses: "they are a
product of a document-dependent term frequency and a corpus-dependent IDF, so
two documents' scores are on different scales by construction." A term's `count`
is its number of occurrences in this document. `term` is the display form (first
surface form, leading determiner stripped, first letter capitalised); the internal
lemmatised `key` is not returned.

```bash
curl -s -X POST http://localhost:8000/keywords \
  -H 'content-type: application/json' \
  -d '{"text":"A stack is a linear data structure that follows the Last In First Out principle. The push operation adds an element to the top of the stack.","limit":3}'
```

```json
[
  {"term": "Stack", "score": 2.0712, "count": 1},
  {"term": "Push operation", "score": 2.0487, "count": 1},
  {"term": "Element", "score": 1.9318, "count": 1}
]
```

Those numbers are two to four times the `/analyse` example's for the same
vocabulary (`Stack` 0.847 → 2.0712, `Push operation` 0.5109 → 2.0487,
`Element` 0.4858 → 1.9318), and the difference is not a bug or a rounding
artefact: the `/analyse` example was the first request against an empty corpus,
and this one ran after three documents had been counted. The IDF half of the
score moved. It is the clearest
demonstration of why the route docstring says the scores are comparable "within
one response, not across responses" — and of what `/stats` is reporting.

Errors: `422` for a bad body. **`limit` in 101–200 fails inside the handler.**

### `POST /summary`

An extractive summary, in reading order. Sentences are chosen by TextRank and
returned in the order they appear in the source, "because a summary in rank order
reads as a shuffled document."

The method is chosen by input size: TextRank when the document has
`GRAPH_MIN_SENTENCES` (10) or more usable sentences, and a frequency-based
ranking otherwise. Below ~10 sentences the similarity graph "has no meaningful
stationary distribution" and ranking becomes arbitrary, so `summarise.py` swaps
in the frequency scorer. Both are described in `docs/algorithms/`.

`score` means different things depending on which ran, and the interface should
not compare scores across documents:

- TextRank: the sentence's share of the document's rank mass. All sentences sum
  to 1 before the top N are taken, so a returned subset does not sum to 1.
- Frequency fallback: the weighted score from `summarise._frequency_rank` — mean
  normalised content-word frequency, times a length factor, times a position
  factor. Not normalised, and on a different scale entirely.

`index` is the sentence's position among the *usable* sentences, which is what
makes reading-order sorting possible downstream.

```bash
curl -s -X POST http://localhost:8000/summary \
  -H 'content-type: application/json' \
  -d '{"text":"A stack is a linear data structure that follows the Last In First Out principle. The push operation adds an element to the top of the stack.","limit":1}'
```

```json
[
  {
    "text": "A stack is a linear data structure that follows the Last In First Out principle.",
    "score": 0.128333,
    "index": 0
  }
]
```

Errors: `422` for a bad body. **`limit` in 51–200 fails inside the handler.**

---

## What the part routes cost

Every part route runs the whole pipeline. `POST /keywords` parses the document,
extracts definitions, builds cards and builds quiz questions, then returns the
keywords and discards the rest. **A part route costs the same as `/analyse`.**

From the `routes.py` docstring, which states the trade rather than hiding it:

> It looks wasteful — asking for keywords parses the document, extracts
> definitions, builds quiz questions and throws them away. It is not, in
> practice: parsing is 90% of the cost and every later stage is linear in
> sentences, so a second entry point that ran only the cheap stages would save a
> few milliseconds and cost a second definition of correctness. The shared path
> is the one that stays right.

The practical consequences, in order of how likely they are to matter:

- Choosing `/keywords` over `/analyse` saves nothing. If a caller wants two
  sections, ask for them in one `/analyse`.
- Every part route call also does the full `.cards`/`.quiz` work, so a caller
  asking for a summary pays for distractor generation.
- Every part route call counts as a document in the corpus (`corpus.observe`
  runs unconditionally at the end of `analyse`).

The API's own generator is built around this fact: it calls `/analyse` and asks
for `maxKeywords: 1, maxSummarySentences: 1, maxQuizQuestions: 1` because it only
wants cards, and the minimum is the cheapest way to say "none of this".

---

## Where the envelope does not apply

The shared envelope covers the routes' own failures and payload validation. It
does not cover everything, and the gaps are worth knowing because they are the
ones a caller will hit while integrating.

| Situation | Status | Body | Envelope? |
| --- | --- | --- | --- |
| Payload validation (`RequestValidationError`) | 422 | `{"status":false,"statusCode":422,"message":[...],"path":"/analyse"}` | yes |
| `/analyse` text too long, run failure, model missing | 413 / 500 / 503 | `{"status":false,...}` | yes |
| `limit` above the target option's own maximum on a part route | **500** | `Internal Server Error` (plain text) | **no** |
| Model missing on a part route | **500** | `Internal Server Error` (plain text) | **no** |
| Unknown path | 404 | `{"detail":"Not Found"}` | **no** |
| Wrong method on a known path | 405 | `{"detail":"Method Not Allowed"}` | **no** |

All five extraction routes now fail the same way. They did not, and the three
differences are worth recording because each produced a wrong answer rather
than a missing one:

- **An over-large `limit` returned a bare 500** from `/quiz`, `/keywords` and
  `/summary`. `TextRequest.limit` allows 200, but `max_quiz_questions` and
  `max_keywords` stop at 100 and `max_summary_sentences` at 50, and the
  `pydantic.ValidationError` raised while building `AnalysisOptions` *inside the
  handler* is not a `RequestValidationError` — so nothing caught it and Starlette
  returned its default plain-text 500. `/cards` was unaffected only because its
  two limits happen to be the same number.

  **Fixed** by clamping: `_clamp(payload.limit, MAX_…)` in `routes.py`, with the
  ceilings named in `schemas.py` so the clamp cannot drift from the bound it
  clamps to. `limit: 150` now returns 100 keywords rather than an error, which
  is the better contract — "give me 150" and "give me as many as you have" mean
  the same thing to a caller.

- **A missing model returned 500 from a part route** where `/analyse` returned
  503. Same cause: only `/analyse` had a `try`/`except`. **Fixed** — all five go
  through `_run()`, which maps a missing model to 503 and a failed extraction to
  500, and logs the traceback for the latter.

- **The size cap applied to `/analyse` only.** **Fixed** — `_run()` checks
  `settings.max_text_chars` for every route. Note that this check is normally
  unreachable: the schema already caps `text` at 500,000 characters, the same
  number as the default `BRAIN_MAX_TEXT_CHARS`, so pydantic answers 422 first.
  It becomes live only when a deployment *lowers* `BRAIN_MAX_TEXT_CHARS`, which
  is the case it exists for.

Every error now leaves through one handler in `main.py`, registered on
**Starlette's** `HTTPException` rather than FastAPI's — the latter is a
subclass, and a 404 or 405 comes from the router raising the base class, so the
obvious registration silently did nothing for exactly the cases it was added
for. The envelope is identical across 404, 405, 413, 422, 500 and 503.

---

## CORS

`main.py` installs `CORSMiddleware` with `allow_origins=["*"]` and methods
`GET`/`POST`. The comment states the intent and the caveat:

> The API is the only intended caller, and it calls server-to-server, so this
> does not need to admit a browser. It is left permissive for local development
> and should be pinned to the API's origin in production — an open CORS policy on
> a service that accepts 500KB of text is a free compute donation.

There is no authentication, no rate limiting and no request-size limit at the
proxy level in this service. If it is ever reachable from outside the compose
network, that is the first thing to fix.
