"""
TextRank: ranking sentences by how much the rest of the text agrees with them.

**The idea.** PageRank ranks web pages by the links between them, on the premise
that a page linked to by important pages is itself important. TextRank applies
the same recursion to sentences, with *similarity* standing in for a link: a
sentence that shares a lot of vocabulary with many other sentences is
representative of the text, and a sentence that shares nothing with anything is
an aside.

The recursion is circular — a sentence is important if it resembles important
sentences — and it is resolved the way PageRank resolves it: by iterating until
the scores stop moving. That is not a trick, it is the definition. The damping
factor is what makes it converge rather than oscillate.

**Why not just score sentences by term frequency.** Because frequency alone
picks whichever sentence is longest and most repetitive. TextRank's virtue is
that it measures *centrality*: a sentence can use a common word once and still
rank highly if that word is what the other sentences are about. It also has a
natural redundancy penalty — two near-identical sentences split one share of the
rank rather than each taking a full one, which is exactly what you want from a
summary.

**Reference.** Mihalcea & Tarau, *TextRank: Bringing Order into Texts* (EMNLP
2004). The similarity function below is theirs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Probability of jumping to a random sentence rather than following a
# similarity edge. 0.85 is the value from the original paper and from PageRank
# before it; it is high enough to trust the graph and low enough to converge.
DAMPING = 0.85

# Iteration stops when no score moves more than this between rounds. The scores
# are normalised to sum to 1, so this is "no sentence's share changed by more
# than one part in ten thousand".
TOLERANCE = 1e-4

MAX_ITERATIONS = 100

# Words shorter than this are ignored when comparing sentences — they are
# overwhelmingly function words, and including them makes every sentence look
# similar to every other one.
MIN_WORD_CHARS = 2


@dataclass(frozen=True)
class RankedSentence:
    index: int
    text: str
    score: float


def _content_words(sentence) -> set[str]:
    """
    The comparable vocabulary of a sentence.

    Lemmas, lowercased, stop words and punctuation removed. Lemmatised so that
    "stack" and "stacks" count as the same word — otherwise a sentence using the
    plural looks unrelated to one using the singular, which is exactly the pair
    that should look most related.
    """
    return {
        token.lemma_.lower()
        for token in sentence
        if token.is_alpha
        and not token.is_stop
        and len(token.text) > MIN_WORD_CHARS
    }


def similarity(left: set[str], right: set[str]) -> float:
    """
    TextRank's sentence similarity, from the paper:

        sim(i, j) = |{w in i and w in j}| / (log(|i|) + log(|j|))

    Two things about this form are worth noticing.

    The numerator is the count of shared words. The denominator grows with the
    *logarithm* of each sentence's length, which is the important part: a plain
    ratio would make short sentences look dissimilar to everything, so a summary
    built from it drifts toward the longest sentences. The log keeps length in
    the calculation without letting it dominate.

    The `+ 1` inside each logarithm is not in the paper but is necessary here:
    `log(1)` is zero, so a one-word sentence would divide by zero. It shifts
    every value slightly and changes no ordering.
    """
    if not left or not right:
        return 0.0

    shared = len(left & right)
    if shared == 0:
        return 0.0

    return shared / (math.log(len(left) + 1) + math.log(len(right) + 1))


def build_similarity_matrix(words: list[set[str]]) -> np.ndarray:
    """The sentence graph, as a symmetric matrix of similarity weights."""
    count = len(words)
    matrix = np.zeros((count, count), dtype=np.float64)

    for i in range(count):
        for j in range(i + 1, count):
            weight = similarity(words[i], words[j])
            matrix[i, j] = weight
            matrix[j, i] = weight

    return matrix


def pagerank(matrix: np.ndarray) -> np.ndarray:
    """
    PageRank over the sentence graph, by power iteration.

    In full, because this is the part worth being able to explain:

    1. Start with every sentence holding an equal share, `1 / n`.
    2. Normalise each row so its outgoing weights sum to 1. A sentence's
       similarity to all others becomes a probability distribution over where
       its vote goes. A sentence similar to nothing has a row of zeros — a
       "dangling" node, whose vote would otherwise vanish from the system and
       whose mass is redistributed uniformly below.
    3. Compute the next scores as

           score' = (1 - d) / n  +  d * (Pᵀ · score)

       which reads as: with probability `1 - d` jump to a random sentence, and
       otherwise follow the graph. The transpose is because `P[i][j]` is the
       weight *from* i *to* j, so to find what arrives at j you need the column.
    4. Repeat until nothing moves by more than `TOLERANCE`, or until
       `MAX_ITERATIONS` — the cap exists so a pathological graph cannot hang a
       request, and it is generous because convergence normally takes 10–20
       rounds.
    5. Normalise the final scores to sum to 1, so a score is interpretable as a
       share of the document's attention rather than an arbitrary scale.

    Dangling nodes are handled by distributing the mass of any all-zero row
    uniformly. Skipping that step leaks probability out of the system every
    round, and the scores drift toward zero — subtly, and only on texts that
    contain an unrelated sentence, which is most of them.
    """
    count = matrix.shape[0]

    if count == 0:
        return np.zeros(0, dtype=np.float64)

    if count == 1:
        return np.ones(1, dtype=np.float64)

    row_sums = matrix.sum(axis=1)
    normalised = np.zeros_like(matrix)

    nonzero = row_sums > 0
    normalised[nonzero] = matrix[nonzero] / row_sums[nonzero, None]

    # Any row that summed to zero has no outgoing weight; treat it as linking to
    # everything equally so its share stays in the system.
    dangling = ~nonzero
    if dangling.any():
        normalised[dangling] = 1.0 / count

    scores = np.full(count, 1.0 / count, dtype=np.float64)
    teleport = (1.0 - DAMPING) / count

    for _ in range(MAX_ITERATIONS):
        updated = teleport + DAMPING * (normalised.T @ scores)

        if np.abs(updated - scores).max() < TOLERANCE:
            scores = updated
            break

        scores = updated

    total = scores.sum()
    if total > 0:
        scores = scores / total

    return scores


def rank(sentences: list) -> list[RankedSentence]:
    """
    Every sentence, with its TextRank score, best first.

    Returned in rank order rather than document order because the caller
    usually wants the top few; it re-sorts by index itself when it needs to
    present them in reading order.
    """
    if not sentences:
        return []

    if len(sentences) == 1:
        return [RankedSentence(0, sentences[0].text.strip(), 1.0)]

    words = [_content_words(sentence) for sentence in sentences]
    matrix = build_similarity_matrix(words)
    scores = pagerank(matrix)

    ranked = [
        RankedSentence(index=i, text=sentences[i].text.strip(), score=float(scores[i]))
        for i in range(len(sentences))
    ]

    ranked.sort(key=lambda item: (-item.score, item.index))
    return ranked
