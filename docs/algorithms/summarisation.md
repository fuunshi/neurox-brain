# Summarisation

**Source:** `src/neurox_brain/nlp/summarise.py`

## Extractive, not abstractive — and that is a deliberate ceiling

An extractive summary is made of sentences the author actually wrote. An
abstractive one reads better and can say something the source never said. For
study material that difference decides it: a summary shown to a student revising
for an exam must be traceable back to the text, and every sentence here can be
highlighted in the chapter it came from.

The cost is real: the result is not as smooth as a written summary. The `score`
on each sentence is what lets an interface say *why* a sentence was chosen rather
than presenting a selection as if it were authored.

## Choosing the algorithm by input length

Two rankers, and the switch between them is the most important decision in this
module.

```python
GRAPH_MIN_SENTENCES = 10
```

**At 10 sentences or more:** [TextRank](textrank.md) over the sentences.

**Below 10:** frequency scoring. TextRank's scores come from the stationary
distribution of a random walk over a similarity graph, and that distribution
only means something when the graph is connected enough to have one. With five
sentences, each sharing vocabulary with one or two others, the walk has nowhere
to go — scores collapse toward uniform, or the graph splits and every sentence
in a component ties with its neighbours. **Both outcomes rank arbitrarily.**

This is not a theoretical concern. gensim's maintainers removed their
summariser partly because on a two-sentence input it returns *nothing*. And the
failure is silent: it returns five sentences in a plausible order, and nothing
indicates the order was noise.

It matters here specifically because this service is called with ~2,000-character
chunks — often five to fifteen sentences, right at the boundary.

## The frequency ranker

Three adjustments, each for a known problem:

**1. Mean normalised frequency.** For each sentence, the mean of its content
words' frequencies divided by the peak frequency in the document. A **mean**, not
a sum: a sum rewards length twice over, once in the numerator and again through
the normalisation meant to correct for it.

**2. A length factor.** `min(words, 30) / 30`. Without it the longest sentence
wins always, because it contains the most words. The cap at 30 means a very long
sentence gains nothing further.

**3. A positional prior.** At most 10%, decaying from the start. Deliberately
weak, because the lead-bias literature is clear that "earlier is better" is
largely a **newspaper artefact**: Lead-3 beats a neural summariser on
CNN/DailyMail and collapses on XSum. Academic prose and lecture notes are much
closer to the non-newspaper setting, where lead bias is weak or absent.

Frequency scoring is not a consolation prize. SumBasic — frequency plus a
context adjustment — came within about a percentage point of DUC-2004's best
system, which is the most robust result in the extractive summarisation
literature: human model summaries contained **94.66%** of the top-5 frequency
words, while a state-of-the-art summariser contained 84%.

## Presentation: back into reading order

Sentences are ranked best-first, then the chosen few are **re-sorted into
document order**. It is easy to get wrong and obvious when you do — a summary
printed in rank order reads as a shuffled document, and the reader has no way to
tell that the sentences are the right set in the wrong sequence.

List markers are stripped and whitespace is flattened at the same point, because
the sentence carries the source document's line breaks and a break in the middle
of a clause looks like a formatting bug rather than an extracted sentence.

## An honest caveat about summaries as a study technique

Dunlosky et al. (2013) rated ten study techniques and placed **practice testing
and distributed practice at high utility**, with **summarisation at low
utility** — alongside highlighting and rereading. The generation effect accrues
to the person doing the generating, and a machine-written summary handed to a
student is closer to rereading than to retrieval practice.

This is not an argument against producing summaries. It is an argument against
**making them the flagship artifact**, and for weighting effort toward the
cloze, flashcard and quiz output instead. Position summaries as a navigation aid
— "what is in this chapter" — rather than as the thing the student learns from.

## Known limitations

- **No redundancy control beyond TextRank's implicit penalty.** SumBasic's
  context adjustment — dividing a word's probability by how often it has already
  been used — is proven sufficient for duplication removal and costs nothing.
  It is not implemented here; the frequency ranker takes the top-scoring
  sentences by independent score, so two near-identical sentences can both be
  selected.
- **No evaluation against a gold summary**, because none exists for this
  material. The sensible proxies are computable against the source: numeral and
  named-entity preservation, compression ratio, and keyphrase coverage. None is
  implemented yet.
- **`headings()` is a heuristic.** It guesses section headings from shape —
  short lines without terminal punctuation. It has obvious failure cases and is
  offered as an outline hint, never used to cut a document up.
- **No ROUGE.** Not because it would be hard, but because it correlates poorly
  with human judgement (0.22–0.54 depending on the dimension), rewards
  first-sentence copying — precisely the wrong behaviour for dense lecture notes
  — and is gameable. If it is added, it should be as a regression guard against
  a previous version of this service, never as a quality claim.
