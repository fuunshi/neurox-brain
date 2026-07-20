# TextRank

**Source:** `src/neurox_brain/nlp/textrank.py`
**Reference:** Mihalcea & Tarau, *TextRank: Bringing Order into Texts* (EMNLP 2004)

## The idea

PageRank ranks web pages by the links between them: a page linked to by
important pages is itself important. TextRank applies the same recursion to
sentences, with **similarity standing in for a link**. A sentence sharing a lot
of vocabulary with many others is representative of the text; a sentence sharing
nothing with anything is an aside.

The recursion is circular — a sentence is important if it resembles important
sentences — and it is resolved the way PageRank resolves it: **iterate until the
scores stop moving.** That is not a trick, it is the definition. The damping
factor is what makes it converge rather than oscillate.

## Why not just score sentences by term frequency

Because frequency alone picks whichever sentence is longest and most repetitive.
TextRank measures *centrality*: a sentence can use a common word once and still
rank highly if that word is what the other sentences are about. It also
penalises redundancy naturally — two near-identical sentences split one share of
the rank instead of each taking a full one, which is exactly what a summary
wants.

## Step 1 — Reduce each sentence to content words

For every sentence, collect the lemmas of its alphabetic, non-stop-word tokens
longer than two characters. Lemmatised so "stack" and "stacks" count as the
same word — otherwise the plural looks unrelated to the singular, which is
precisely the pair that should look most related.

```
"A stack is a linear data structure."  ->  {stack, linear, data, structure}
"A queue is also a linear data structure." -> {queue, also, linear, data, structure}
```

## Step 2 — Score every pair of sentences

The similarity function is the paper's:

```
                    |words(i) ∩ words(j)|
sim(i, j) = ──────────────────────────────────────
             log(|words(i)| + 1) + log(|words(j)| + 1)
```

Two things about this form matter:

- **The numerator is the count of shared words.** Working through the example
  above: `{linear, data, structure}` — three shared words.
- **The denominator grows with the *logarithm* of each length.** This is the
  important part. A plain ratio would make short sentences look dissimilar to
  everything, so a summary built from it drifts toward the longest sentences.
  The log keeps length in the calculation without letting it dominate.

Worked: `3 / (log(4+1) + log(5+1)) = 3 / (1.609 + 1.792) = 3 / 3.401 = 0.882`.

The `+ 1` inside each logarithm is not in the paper and is necessary here:
`log(1)` is zero, so a one-word sentence would divide by zero. It shifts every
value slightly and changes no ordering.

## Step 3 — Build the graph

A symmetric matrix `W` where `W[i][j] = sim(i, j)`. The diagonal is zero — a
sentence is not similar to itself for this purpose.

## Step 4 — PageRank by power iteration

This is the part worth being able to explain in full.

**1. Start uniform.** Every sentence holds an equal share, `1/n`.

**2. Normalise each row** so its outgoing weights sum to 1. A sentence's
similarity to all others becomes a probability distribution over where its vote
goes:

```
P[i][j] = W[i][j] / Σₖ W[i][k]
```

**3. Handle dangling nodes.** A sentence similar to nothing has a row of zeros.
Left alone, its share would leak out of the system every round and all scores
would drift toward zero — subtly, and only on texts containing an unrelated
sentence, which is most of them. So any all-zero row is replaced with a uniform
distribution over every sentence, keeping its mass in the system.

**4. Iterate.**

```
score' = (1 − d)/n  +  d · (Pᵀ · score)
```

Read it as: with probability `1 − d`, jump to a random sentence; otherwise
follow the graph. The transpose is because `P[i][j]` is the weight *from* i *to*
j, so what arrives at j is the column.

The damping factor `d = 0.85` is the PageRank value — high enough to trust the
graph, low enough to converge.

**5. Stop** when no score moves by more than `1e-4`, or after 100 iterations.
The cap exists so a pathological graph cannot hang a request; convergence
normally takes 10–20 rounds.

**6. Normalise** the final scores to sum to 1, so a score reads as a share of
the document's attention rather than an arbitrary number.

## Step 5 — Rank, then restore order

Sentences come back best-first. `summarise.py` re-sorts the chosen few back into
document order, because a summary printed in rank order reads as a shuffled
document.

## What it is used for, and what it is not

Used for: summarisation, on inputs of **10 sentences or more**. Below that the
gate in `summarise.py` switches to frequency scoring — see
[summarisation.md](summarisation.md) for why a graph is the wrong tool for a
short input.

Not used for: keyword extraction. Graph-based keyphrase methods exist
(TextRank-for-keywords, SingleRank, PositionRank) and the survey numbers put
them roughly level with TF-IDF on short texts and worse on long ones, with a
much larger implementation surface. The service uses TF-IDF with its own corpus
instead — see [keywords-tfidf.md](keywords-tfidf.md).

## Known limitations

- **No controlled head-to-head** of TextRank against LexRank, TF-IDF or centroid
  methods with significance testing exists in the literature. Published
  comparisons vary dataset, preprocessing and *k*, and split roughly evenly
  between TextRank winning and losing.
- **Sparse graphs degrade quietly.** With few sentences, or with a high
  similarity threshold, scores collapse toward uniform and the ranking becomes
  arbitrary. The length gate is the mitigation, not a fix.
- **Similarity is lexical.** Two sentences making the same point in different
  words score zero against each other. Embedding-based similarity would fix that
  and would need a model this service deliberately does not load.
