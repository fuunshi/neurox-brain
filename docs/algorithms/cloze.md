# Cloze deletion

**Source:** `src/neurox_brain/nlp/cloze.py`

## What it is, and why it earns its place

A definitional card asks "what is a stack?" and can be answered vaguely and still
feel right. A cloze asks the reader to produce the exact word in a real
sentence — a different and often harder kind of recall, and it tests the term
*in use* rather than in isolation.

## The rule: blank the term the sentence exists to define

```
A _____ is a linear data structure that follows the Last In First Out principle.
        answer: stack
```

There is exactly one blank-selection rule in this service, and it is that one.

### What it used to do, and why that was removed

The first version had a second path: take the highest-scoring `TF-IDF`
content word in *any* sentence and blank it, so a document with few definitions
would still produce practice material. That path was removed after reading the
evidence, and it is the largest single quality decision in the package.

Pino, Heilman & Eskenazi measured exactly that strategy: among baseline cloze
items chosen that way, **34% admitted several correct answers**, against 12%
after switching to a constrained selection. Sumita et al. separately found ~6.5%
of generated items were unanswerable even by native speakers.

It is worth being concrete about what that looked like in this service's own
output. From a real run over data-structures prose, the generic path produced:

```
"A data structure is a way of organising data in _____ so that it can be used
 effectively."                                          answer: computer

"Unlike _____, a queue removes elements from the front." answer: stack
```

The first is guessable without understanding anything. The second is fine — and
the reason it is fine is that it happens to be definitional. A blank the reader
cannot pin down is not a hard question, it is a broken one, and it teaches the
reader to distrust the whole set.

**A document with no definitions now produces no cloze cards.** That is the
intended behaviour rather than a shortfall: cards land as drafts for a reader to
accept, so a small set of sound candidates beats a large set of ambiguous ones.

## The alignment step

The definiendum has already been located by the definition extractor, so the
blank is not *chosen* — it is looked up. `from_term` takes the sentence and the
term's span and produces the card.

Two details do real work:

1. **Both sides are flattened before the replacement.** The parsed sentence
   carries the source document's line breaks; `term_span.text` may straddle one.
   Replacing a term in a string that contains a newline inside the match fails
   silently — `str.replace` simply finds nothing — and the card is dropped.
   Flattening both first is why this works.

2. **The replacement is boundary-aware.** A compiled pattern with `(?<![\w-])`
   and `(?![\w-])` guards, not `str.replace`. Without them, blanking "list"
   inside "listen" produces a sentence that is no longer English. This happened
   with a naive `replace` and produces output that looks like a typo rather than
   a bug.

## The guard set

`_is_good_blank` — each check for a concrete failure:

| Check | Rejects | Why |
| --- | --- | --- |
| Content POS | Not `NOUN`/`PROPN`/`ADJ` | Blanking a determiner or auxiliary makes the answer grammar, not knowledge |
| No named entity | Any token with `ent_type_` | A blanked person or product name is trivia, and the sentence rarely constrains it |
| Occurs once in the sentence | The term appearing twice | Blanking one leaves the answer in plain sight |
| Not sentence-initial | The first token | Usually the subject of a sentence about something else, where the content is later |
| ≤ 60% of the sentence | A term that is most of the sentence | No context remains to constrain the answer |
| ≥ 4 characters **after stripping the article** | `a way`, `the use` | Measured on the span, "a way" passed a four-character floor on the strength of its "a " |

That last check is a fix from a real run: blanking `way` produced a card whose
answer carried no subject knowledge and was guessable from the sentence around
it. The article has to be removed *before* measuring, because the answer is what
the reader supplies, and the reader does not supply the article.

## Known limitations

- **No exclusivity check.** The strongest available signal is the gap score from
  Matsumori et al. (2023): build a distribution over candidate fillers from an
  n-gram language model, and if the top candidate does not dominate the rest,
  present the item as multiple choice instead of open cloze. That is a real
  improvement and it needs a language model trained on a corpus. The fallback is
  the review step — a reader who sees two plausible answers discards the card.
- **No document-level occurrence filter.** Hill & Simha's rule — drop any
  candidate whose stemmed form appears more than once in the *document*, so the
  reader cannot recover it from an earlier mention — is deliberately not applied
  here. For the definiendum of a definition, the term recurring throughout the
  document is the normal case rather than a leak; applying the rule would reject
  almost every good blank.
- **Cloze is the hardest format to evaluate.** Judge–evaluator agreement
  degrades faster on cloze than on multiple choice, which means the quality
  numbers available for this format are the least trustworthy of any generated
  here. Treat the output with corresponding suspicion.
- **One blank per sentence, one card per term.** Two blanks give each other away.
