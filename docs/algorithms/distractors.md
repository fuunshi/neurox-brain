# Distractor generation

**Source:** `src/neurox_brain/nlp/distractors.py`

## Why this is the hard part

Generating a correct answer is a lookup. Generating three *wrong* answers a
student might actually choose is the whole difficulty of automatic question
generation. The two failure modes are opposite and both ruin the question:

- **Too obviously wrong** — "Which of these is a data structure: stack, banana,
  Tuesday?" tests nothing, and a student notices immediately.
- **Also correct** — where the question came from a definition true of several
  options. This is worse, because it teaches something false and destroys trust
  in every other question on the site.

Both come from picking options by surface form rather than by meaning.

## The trade-off this service resolves deliberately

The single most consistent finding in the distractor literature is that the
methods producing the most *plausible* distractors also produce the least
*reliable* ones:

| Method | Plausible | Reliable (not also correct) |
| --- | --- | --- |
| Word-vector similarity — the "clever" option | **46.6%** | **93.2%** |
| Spelling similarity | lower | 93.2% |
| POS + frequency — the "dumb" option | ~5% | **100%** |

Near neighbours in vector space are frequently synonyms. **No classical method
gets both**, and the literature's resolution is a product decision rather than
an algorithm: prioritise reliability, ship fewer items, and let review carry the
rest.

**So this service prioritises reliability.** A multiple-choice question with two
correct answers is not a slightly worse question — it is a broken one. Where the
choice arises, it is made in that direction.

## Source 1 — definitions from the document (the default for MCQs)

`for_definitions` builds the questions the API actually generates. Given a
correct definition, it draws distractors from the *other definitions in the same
document*.

This is the strongest source available and it needs no external data. The reason
is structural: definitions in one document are about different terms **by
construction**, so the "also correct" risk is far lower, and they arrive in the
same register, at the same level of detail, about the same subject.

The only real risk is length — a conspicuously longer option is guessable as the
right one — so candidates are chosen by how close their length is to the
correct answer's. That is a **documented weakness**: length is a surface
heuristic standing in for "similar level of detail". The honest alternative is
to compare the definitions' vectors and pick semantically adjacent ones, which
needs an embedding model.

**Worked example.** From data-structures prose:

```
What is stack?
   A linear data structure that follows the First In First Out principle.
   A way of organising data in a computer so that it can be used effectively.
   A linear data structure where each element points to the next element.
 * A linear data structure that follows the Last In First Out principle.
```

The FIFO/LIFO confusion is precisely the distractor a student needs to
discriminate against.

## Source 2 — terms, for term-option questions

`for_terms` is used when the options are *terms* rather than definitions. The
pipeline is:

**1. Reject candidates that WordNet says could pass for the answer**
(`_too_close_to_be_wrong`). A candidate is dropped if it shares a synset with
the correct answer, is an **antonym**, or is a **hypernym at any depth** — "data
structure" for "stack" is not a wrong answer, it is a less precise right one.

**WordNet is used here only as a filter, never as a generator.** The same
taxonomy that makes siblings related is what makes them dangerous; the
cheap-and-safe direction is to use it to exclude rather than to propose.
(`_from_wordnet` does use siblings to *top up* when the document is short, which
is the one place it generates — and those candidates go through the same filter.)

**2. Rank what survives.** How depends on whether the model has usable vectors:

- **With real vectors** (`en_core_web_lg`): keep only candidates inside a
  similarity band, and take them from the **middle** of it — close enough to
  tempt, far enough to be wrong. Sorting by distance from the midpoint rather
  than by raw similarity avoids always picking the single closest candidate,
  which is the one most likely to be a synonym that slipped the ceiling.
- **Without vectors** — which is the default, since `en_core_web_sm` has none
  and `en_core_web_md`'s are pruned to 20,000 entries so that most cosine
  similarities are bucket artefacts — rank by **shared head noun**.

Head-noun matching is the strongest cheap signal available: English noun phrases
are head-final, so "linked list" and "doubly linked list" are both lists while
"stack" and "queue" are unrelated by name. Ties break alphabetically, so the
output is deterministic for a given input.

**3. Reject structural giveaways.** A candidate containing the answer, or
contained by it, is dropped — "operating system" beside "system" reads as a
trick in one direction and a giveaway in the other. Single characters, pure
numbers, and anything under three characters go too.

## What is deliberately not done

| Not done | Why |
| --- | --- |
| Neural distractor generation | Beyond the constraint, and its evaluation metrics require transformers so it could not even be scored here |
| WordNet hypernym/hyponym expansion as the primary source | For CS terms it yields "list" for "linked list", or nothing |
| Cosine-to-key as the headline quality score | High similarity means plausible *and often secretly correct* — it is the failure this module exists to avoid |
| Similarity bands as a *filter* when vectors are quantised | Filtering on an artefact is worse than not filtering |
| "Confusion rate" as a metric | No paper defines it; the field uses distractor selection rate, plausibility, GDR/NDR |
| Four options always | Contradicted by the psychometrics; the supported rule is "as many *functional* distractors as possible" |

## A question is only emitted when it is complete

`has_enough` requires three distractors before a question is built. A question
with two options is a coin flip, and one padded with nonsense teaches the reader
to ignore the options.

**The consequence, visible in real output:** a short document with only three
definitions yields **zero** quiz questions, because after removing the correct
answer there are not three left to draw from. That is the intended trade.
`/analyse` returns `quiz: []` and the caller links it to the document, not the
code, being short.

## Known limitations

- **Reliability is not verified at runtime.** `_too_close_to_be_wrong` needs
  WordNet; without it the check returns `false` for everything, so no synonym
  filter is applied. The service logs `wordnet=no` at start-up for this reason —
  see `ensure_wordnet_loaded`, and `operations.md`.
- **No carrier-sentence validity test.** The strongest available check is
  Goto/Sumita's: reject a distractor `D` if the question stem with the blank
  filled by `D` is attested in the corpus, or scores comparably to the key under
  an n-gram language model. That needs a model trained on the domain corpus.
- **No set-level checks.** No two distractors should be mutually synonymous
  (students eliminate them in pairs), and none should be a hypernym of the key.
  The single-candidate filter catches the second case per candidate but not as a
  set.
- **The ceiling is real and low.** Published classical distractors reach ~46%
  "plausible" against a human ceiling of 53%. Expect roughly half the generated
  distractors to be mediocre, and design the review step accordingly.
