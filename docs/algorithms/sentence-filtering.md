# Sentence segmentation and filtering

**Source:** `src/neurox_brain/nlp/pipeline.py`

Everything downstream works on sentences, so this is where the whole pipeline's
input quality is decided. It runs once per document, and every extractor shares
the result.

## Normalising the text

`normalise()` repairs the damage document extraction leaves behind, and does
**nothing else**. Deliberately no lowercasing, no punctuation stripping, no stop
word removal: spaCy's parser is trained on ordinary prose, and text that has been
"cleaned" into a bag of words parses worse and yields worse sentences.

| Transformation | Why |
| --- | --- |
| Non-breaking and Unicode spaces → plain space | PDFs emit several kinds, and they break tokenisation |
| Soft hyphens removed | A line-break artefact, never a real character |
| Runs of spaces and tabs collapsed | |
| Three or more newlines → two | **Paragraph breaks are kept** — they are the strongest signal that a topic changed, and later stages use them |

## Segmenting

Sentence boundaries come from **the dependency parser**, not from splitting on
full stops. Splitting on `.` alone breaks on abbreviations, decimals and the
"e.g." that academic prose is full of — a single `.` after "e.g" ends a
"sentence" three words long.

This is why a spaCy model with a parser is required rather than optional.

## Discarding what cannot be useful

`is_usable_sentence` — every rule is a false-positive control, and each exists
because of something seen in a real run:

| Rule | Threshold | What it rejects |
| --- | --- | --- |
| Minimum length | 25 chars | Headings, list fragments, page numbers, "Figure 3." captions that survived extraction without their figure |
| Maximum length | 600 chars | A PDF that lost its punctuation, where the "sentence" is a page. Treating it as one sentence produces one enormous card |
| Minimum words | 5 content tokens | A line of table-of-contents dots |
| Must contain a verb or auxiliary | | A run of nouns is a heading or a table row. **A definition needs a verb to be a definition**, so a verbless span cannot be one |
| At least 50% alphabetic tokens | | A data row. A sentence that is mostly digits is not prose |

The verb check is the one that earns its place: it removes headings cheaply and
correctly, because the thing being extracted is a relationship between a term
and a description, and that relationship is expressed by a verb.

## Why the parse is kept rather than discarded

Every extractor needs a *different* view of the same analysis:

| Stage | What it needs |
| --- | --- |
| Definition extraction | Dependency arcs, POS, noun chunks |
| Cloze | Noun chunks, POS, named entities, spans |
| Distractors | POS, named entities, head nouns |
| Keywords | Noun chunks, lemmas |
| Summarisation | Lemmas, sentence vectors |

Running `nlp()` five times would parse the text five times and produce five
documents that could disagree about where a sentence ends. So it is parsed once
and the `Document` object is passed between stages — frozen, so no stage can
mutate the analysis another stage depends on.

## The shared pipeline

One spaCy model per process, loaded once via `lru_cache` and guarded by a lock
on first load so two concurrent requests do not both call `spacy.load` and
double peak memory.

Loading happens in the **lifespan handler**, not at import, so the process binds
its port and answers `/health` *while* the model loads. A probe therefore sees a
live process reporting `loading`, which it can distinguish from one that never
came up.

## Known limitations

- **The thresholds are tuned by inspection, not measured.** 25 characters, 5
  words, 50% alphabetic — these are reasonable values that reject the cases
  encountered, not optima derived from a labelled set. A document in a very
  different register may need different ones.
- **No paragraph structure is used downstream.** Paragraph breaks survive
  `normalise` but nothing currently uses them; per-paragraph scoring would be a
  natural improvement for keyword extraction and summarisation on long
  documents.
- **One model, one language.** English only. `en_core_web_sm` is a general model
  and will parse a computing textbook less well than a model adapted to it.
