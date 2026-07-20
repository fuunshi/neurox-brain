# Keyword extraction — TF-IDF

**Sources:** `src/neurox_brain/nlp/keywords.py`, `src/neurox_brain/nlp/corpus.py`

## The formula, and what each half does

```
score(t, d) = tf(t, d) × idf(t)
```

`tf` is how often the term occurs **in this document**: a term the author keeps
using is a term the document is about. `idf` is how rare the term is **across
all documents**: a term in every chapter carries no information about which
chapter you are reading, however often it appears.

Either half alone is useless, and the failure of each is what the combination
exists to prevent:

- **TF alone** ranks "the", "is" and "system" at the top of a computer science
  text. Frequency without rarity measures English, not the subject.
- **IDF alone** ranks a typo, a product name mentioned once, and a stray proper
  noun at the top. Rarity without frequency measures accident.

## The problem: a single document has no IDF

IDF is a corpus statistic. With one document, every document frequency is 1, so
IDF is a constant factor and **TF-IDF degenerates to TF up to a monotone
transform.** This is uncontroversial in information retrieval, and it is why so
many implementations return "the" and "also" while calling themselves TF-IDF.

### The solution: keep a corpus

`corpus.py` maintains a document-frequency table across **every document the
service has processed**, persisted to a JSON file. It is a real IDF that gets
better with use:

- A term appearing in every chapter of a textbook converges on a low weight.
- A term specific to one topic keeps a high one.

**Cold start.** With fewer than five documents there is nothing to be inverse
over, so weights fall back to a smoothed prior of `0.7` — chosen so a term in
every document lands near `0.4` and a term in one lands near `1.0`, the range
real IDF settles into. The prior is *blended in* proportionally rather than
switched out, so scores move smoothly as the corpus warms instead of jumping.

**Where it is stored.** A JSON file, not a database. It is one map from term to
count; it rebuilds simply by using the service; and a service that cannot write
it still works. Deleting it is safe — see `operations.md`.

**What is counted.** `vocabulary()` returns the lemmatised content-word set of a
document — deliberately *everything* the document contains, not just the terms
`extract` kept. The whole point of IDF is knowing which terms are common, and
common terms are exactly the ones extraction throws away.

**Ordering matters.** The document is added to the corpus *after* every score is
computed. Adding it first would let a document influence its own IDF — subtly,
and only on the second request for the same text, which is the kind of bug that
survives testing.

## Smoothing

The textbook formula `log(N / df)` is undefined for a term in no document and
zero for a term in all of them. This service uses the smoothed form:

```python
idf(t) = log((N + 1) / (df + 1)) + 1
```

Every weight stays positive, so a term can never be *penalised* into
irrelevance — only ranked low. Ranking low is what "this word is in every
document" should mean.

## The full scoring pipeline

For each candidate phrase from the parse:

**1. Sublinear term frequency.** Raw counts are replaced by `1 + log(count)`.
Someone who writes "stack" twenty times is not thinking about it twenty times as
much as someone who wrote it once, and without this one repeated word dominates
every score in the document. Then normalised by the document's largest count, so
a long document does not simply produce larger scores than a short one:

```
tf = (1 + log(count)) / (1 + log(max_count))
```

**2. IDF** from the corpus, as above.

**3. Phrase bonus.** A phrase is more specific than its words in isolation, so a
multi-word candidate gets a **bounded** bonus — `1.0` for one word, `1.1` for
two, `1.15` for three. Bounded, not proportional: an unbounded bonus rewards the
longest chunk the parser produced rather than the most meaningful.

This is well supported — reader-assigned keyphrase sets contain roughly twice as
many multi-word phrases as author-assigned ones, and one study found F rising
from 0.30 to 0.40 simply by filtering out single-word candidates.

**4. Position bonus.** At most 10%, decaying from the start of the document.
Terms introduced early are more often what a text is about. This is a heuristic
from keyphrase extraction rather than a law, and it is deliberately small so it
reorders near-ties without overturning a clear frequency signal.

## Candidate selection

Candidates are **noun chunks** from the parse, plus their individual content
tokens. Constrained by:

| Filter | Reason |
| --- | --- |
| ≤ 3 tokens | Beyond that a "keyphrase" is a clause |
| ≥ 50% content POS (`NOUN`/`PROPN`/`ADJ`) | Rejects "it is a", which the parser will happily produce |
| No stop words in the content tokens | |
| Must contain a letter | Rejects figure references |
| Not in `DOMAIN_FILLER` | `example`, `figure`, `value`, `type`… frequent in academic prose, never a keyword. Kept short on purpose — a long hand-written list is a list nobody maintains, and the corpus IDF learns most of it |
| Leading determiners stripped | See below |

**The determiner fix.** The first version produced `A stack`, `A queue`,
`An algorithm`, `A way`, `A computer` — a keyword list is a list of terms, and
an article is not part of a term. Leaving it in also splits one concept across
two entries ("A stack" and "The stack" scoring separately).

**Display versus key.** `Keyword` carries both: `key` is the lemmatised form
used for matching, `term` is what a reader sees. They are not interchangeable,
and conflating them was a real bug — see the note in `cloze.py::keyword_scores_from`.

## What this is used for

A navigation and index surface, **not a graded answer**. The best unsupervised
methods reach F1@5 of about 0.15–0.37 depending on the dataset, and "correct" is
contested between annotators. A keyword list is a set of doors into a document,
so a wrong keyword is cheap; nothing downstream is gated on the list being
clean.

## Known limitations

- **The corpus is per-instance.** Two processes running against different corpus
  files produce different weights for the same document. Fine for one
  deployment; worth knowing before scaling horizontally.
- **No stemming beyond lemmatisation, no synonym merging, no embedding
  expansion.** Each would find more candidates and make the output harder to
  explain to the person reading it — which, for a keyword list shown on a study
  page, is the whole value.
- **No document-title weighting.** Adding the title to the scored text improves
  F1 consistently in the literature and is cheap. The API passes a title, and
  this version does not use it for keywords — the obvious next improvement.
- **Single-word candidates are kept.** Filtering them out entirely matched full
  ranking in one study, so it is not obviously wrong to keep them — but the
  list will contain some.
