# Fact extraction

Finding what a text says beyond "what is this thing". Source:
`src/neurox_brain/nlp/facts.py`; selection into cards:
`src/neurox_brain/nlp/selection.py`.

`definition-extraction.md` covers the copular and cue patterns that produce
*definitions*. This covers the four families added alongside them.

---

## Why

Definitions answer one question. A chapter is mostly other questions: what does
this consist of, how does it work, how is it different from the thing next to
it, what is it for. Those are what an exam asks about, and the service produced
none of them.

**Everything here is extractive.** No pattern invents text. A front is a source
span and a back is a source span, joined at most by a fixed connective. There is
no language model in this service and there is not going to be one, so the
alternative to "found in the sentence" is "made up" — and a card asserting
something the author never wrote is worse than no card.

---

## The families

### Properties — what a thing consists of, has, or is measured by

| Pattern | Shape | Base |
| --- | --- | --- |
| `consist_of` | verb ∈ {consist, comprise} + `prep(of\|in)` → `pobj` | 0.65 |
| `possess_property` | verb ∈ {have, contain, include, …} + `dobj` whose head ∈ `PROPERTY_NOUNS` | 0.50 |
| `possess_salient` | the same shape, object outside that list | 0.35, gated |
| `be_of` | copula + `prep(of)` → `pobj` | 0.55 |

**`PROPERTY_NOUNS` is the discipline of the family.** `have`, `contain` and
`include` are among the most frequent verbs in expository prose. Firing on the
shape alone puts a card on nearly every sentence in a document. Requiring the
object to be a property noun is what separates

- *"A binary search tree has a time complexity of O(log n)"* — a card

from

- *"Algorithms have been studied for decades"* — not a card, and there is
  nothing in the syntax to tell them apart except the object.

`possess_salient` keeps the second kind but scores it **below the shipping
floor** (0.35 against `MIN_FACT_CONFIDENCE = 0.50`), so it can only reach a deck
when the document's own salience lifts it. That is the mechanism, and it is
deliberate: a fact the chapter returns to is worth a card even when its shape is
weak, and the same shape in passing is not.

### Comparisons — how two things differ

| Pattern | Shape | Base |
| --- | --- | --- |
| `difference_between` | nsubj ∈ {difference, distinction, comparison, contrast} + `prep(between)` → `pobj(A)` + `conj(B)` | 0.65 |
| `differ_from` | nsubj(A) + verb ∈ {differ, vary} + `prep(from)` → `pobj(B)` | 0.65 |
| `unlike` | leading `prep` ∈ {unlike, contrary}, main nsubj is A, `pobj` is B | 0.65 |
| `comparative_than` | copula + `acomp` tagged `JJR` with a `than` child | 0.50 |

**This family needs its own guard set, and the reason is concrete.**
`definitions._passes_guards` rejects negation outright — right for a definition,
because *"Mediation is not a viable alternative"* asserts the opposite of a
definition. But

> *"Unlike arrays, linked lists **do not** require contiguous memory."*

is precisely a contrast, and the negation *is* the fact. Sharing one guard set
with a flag to invert it would be a check that one day silently stops checking,
so the comparison family simply does not apply the negation test.

**A comparison is about a pair, so its phrase has to be split.** The phrase span
deliberately *keeps* nominal coordination — that is what makes "vertices and
edges" survive in a definition — but in "the difference between a stack and a
queue" each half is one participant. `_span_without_conjuncts` and
`_participant_span` cut the coordination and the trailing clauses, so the
participant is "queue" and not "queue in the order of removal".

### Purpose — what a thing is for

| Pattern | Shape | Base |
| --- | --- | --- |
| `used_to_xcomp` | passive verb ∈ {use, employ, apply} + `xcomp` | 0.60 |
| `used_for` | the same, + `prep(for)` → `pobj` | 0.60 |
| `useful_for` | copula + `acomp` ∈ {useful, helpful, needed, …} + `prep(for)` | 0.45 |
| `enable` | nsubj + verb ∈ {allow, enable, permit, let} + `dobj` | 0.40 |

**No modal check, and that is the second concrete reason the guards are
per-family.** `definitions` rejects "could / might / may / can": a hedged
*definition* asserts something the source did not. A purpose is the opposite
case — *"A stack **can** be used to reverse a string"* is how course text states
a capability, and the modal check would delete the family's most common
sentence.

**Passive and predicative only.** *"The algorithm uses a hash table"* is a
dependency, not a purpose, and `use` is one of the most frequent verbs in the
corpus. Requiring `auxpass` removes most of the false-positive mass without
losing a genuine purpose statement.

### Processes — how something works, or what happens when

| Pattern | Shape | Base |
| --- | --- | --- |
| `begin_by` | nsubj + verb ∈ {begin, start, proceed, work} + `prep(by)` → `pcomp` | 0.60 |
| `ordered_chain` | `ROOT` verb with `conj` verb children **and** an ordering adverb | 0.50 |
| `temporal_advcl` | main verb + `advcl` marked by {when, whenever, after, before, once} | 0.40, gated hardest |

**Single sentence, always.** A process is a sequence, and stitching step one
from one sentence to step two from another asserts an ordering the author never
wrote. That is the line between extractive and inventive.

**The ordering adverb is required.** *"X does A and B"* is a conjunction, not a
procedure; without the cue every multi-verb sentence in a document becomes one.

**A back containing an unresolved pronoun is discarded.** There is no coreference
resolution in this stack, so a reader meeting *"The CPU fetches the instruction,
decodes **it**, and executes **it**"* out of context cannot tell what "it" is.
That sentence is dropped, which is the correct outcome and a real cost.

**`temporal_advcl` is the highest-frequency, lowest-precision shape here.**
"When" is everywhere in course prose. It is left to the salience gate, which is
strict enough that it survives only when the sentence is central *and* the term
is a document keyword. If it underperforms on real material, cut it — the other
two carry the family.

---

## Confidence

The same skeleton as `definitions._confidence`, plus one term that scorer does
not have:

```
base(pattern)
+ 0.15   term root is NOUN or PROPN        (-0.25 for PRON or DET)
+ 0.05   back longer than term             (-0.10 if shorter than 1.5x)
+ 0.10   sentence in the top TextRank quartile
+ 0.05   sentence above the median         (-0.05 below)
+ 0.05   term is one of the document's keywords
clamp [0, 1]
```

Absolute, not document-normalised, for the same reason as the definition
scorer: two documents' confidences should be comparable.

**The floor is `MIN_FACT_CONFIDENCE = 0.50`, and it applies here and not to
definitions.** That inconsistency is deliberate. Definitions are the incumbent;
moving their threshold in the same release would make the new families' effect
unmeasurable. If the families prove out, revisit the definition floor
separately.

The floor is what makes the salience term load-bearing rather than decorative:
the closed-cue patterns clear it on their own, and `possess_salient`,
`temporal_advcl`, `comparative_than` and `enable` clear it *only* when the
document's own signals lift them.

---

## Selection into cards

`selection.py`, and most of the tuning lives there as named constants.

- **Fronts.** A comparison is fronted `"A vs. B"` — a bare term cannot express a
  two-party fact. Everything else keeps the bare term. Where a term has more
  than one card, the extras get a label from a constant (`"stack — purpose"`),
  never generated phrasing.
- **Caps.** One card per `(term, family)`; two per term, three when every card
  for that term clears 0.70; new families capped at `max(3, max_cards // 2)`.
- **Definitions claim the budget first**, preserving the existing behaviour for a
  document that only produced definitions.
- **Deduplicated by normalised back**, because two patterns can find the same
  fact in the same sentence.

---

## Known misses

Recorded rather than guarded, because a guard for each was written and never
once fired.

- **`O(n log n)` is not extractable with this tokenizer.** It fragments into
  `O(n` / `log` / `n` / `)`, with `log` as the copular complement. It happens not
  to produce a card — the span is five characters and the 12-character minimum
  rejects it — but that is luck rather than design, and complexity cards are out
  of scope until the tokenizer is dealt with.
- **A parenthesised phrase loses its closing bracket.** spaCy attaches `)` to
  the sentence root rather than to the token it closes, so
  `"a time complexity of O(log n)"` arrives already missing it and the back
  reads "…of O(log n".
- **Hearst patterns are not used as a card source**, per
  `docs/algorithms/README.md`: their standalone F is about 0.15 and the output is
  hyponymy, which is list-shaped.

---

## What would improve this

The published route from ~0.16 precision to ~0.78 is **supervision** — a
classifier trained on annotated corpora — and that is the obvious next step if
quality becomes the bottleneck again. Nothing cheap closes the gap; the salience
gates cut the worst of it and the pipeline's draft-and-review step handles the
rest.
