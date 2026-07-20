# neurox-brain

Turns expository English text into study material: flashcards, multiple-choice
questions, cloze deletions, keywords and summaries.

**No language model.** spaCy for parsing, TF-IDF for keywords, TextRank for
summarisation, WordNet for the synonym checks, and a set of dependency patterns
for finding definitions. It runs on a small VPS, costs nothing per call, and
works with no network access.

## What it is for

It is the generator behind `neurox`'s card generation, sitting between two
existing options:

| Generator | Quality | Cost | Needs |
| --- | --- | --- | --- |
| Heuristic | Lowest — string patterns only | Free | Nothing |
| **neurox-brain** | **Middle — reads a definition across a sentence** | **Free** | This service running |
| Gemini | Highest — parses and paraphrases | Per card | An API key |

The API prefers this one when `NEUROX_BRAIN_ENABLED=true`. See
`../flash-cards-backend/src/application/generation/generation.service.ts`.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e .

# The model and the corpus data are build steps, not first-request steps.
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

./scripts/dev.sh          # http://localhost:8000
```

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/analyse \
  -H 'content-type: application/json' \
  -d '{"text":"A stack is a linear data structure that follows the Last In First Out principle. A queue is a linear data structure that follows the First In First Out principle.","title":"Data Structures"}'
```

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | How it fits with the API, the transports, the message flow |
| [docs/routes.md](docs/routes.md) | Every HTTP route, with examples |
| [docs/operations.md](docs/operations.md) | Running, configuring, troubleshooting, scaling |
| [docs/algorithms/](docs/algorithms/README.md) | Every algorithm, and **the honest numbers** |

**Start with `docs/algorithms/README.md`.** It states plainly what this class of
method achieves — definition-pattern precision around **0.16** on real course
text, distractors around **46% "plausible"** against a human ceiling of 53% —
and why the answer is a review step rather than a cleverer algorithm. Anything
that reads as a stronger claim than that is a claim about marketing.

## The short version of the design

- **One spaCy parse per document**, shared by every extractor, because five
  stages need five views of the same analysis and parsing once is what stops
  them disagreeing about what a sentence is.
- **Definition extraction reads dependency structure**, not regular expressions,
  plus a guard set for the five documented failure modes (negation, modals,
  expletive subjects, contrastive complements, non-nominal heads).
- **Cloze blanks the definiendum in its own definitional sentence** — and
  nothing else. The generic "blank a content word" path was removed because
  around a third of its output admitted more than one correct answer.
- **Distractors prioritise reliability over plausibility**, because a question
  with two correct answers is broken rather than merely worse.
- **Keywords use a real IDF**, from a corpus this service maintains itself,
  because a single document has no document frequency and "TF-IDF" on one
  document is just TF.
- **Graph ranking is gated on input length**, because below about ten sentences
  a similarity graph has no meaningful stationary distribution.
- **Every HTTP route is safe to call synchronously**, and the queue transport
  exists for when holding a request open is the wrong shape.
