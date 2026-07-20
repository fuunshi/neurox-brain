# Definition extraction

**Source:** `src/neurox_brain/nlp/definitions.py`

## The problem

Find every place the text says what something *is*, and turn each into a
term/definition pair.

## Why not a regular expression

The obvious implementation is `(\w+) is (?:a|an|the)? (.+)`. It works on the two
sentences you try it on, then produces:

```
"This is a problem that arises when..."       -> term "This"
"The algorithm is fast and uses O(n) space"   -> definition "fast and uses…"
"A stack is a linear data structure"          -> correct, by luck
```

The failure is that **"is" carries no information by itself.** What marks a
definition is *grammatical structure*: a subject, a copula, and a complement
that is a noun phrase describing that subject. A dependency parse gives exactly
that, which is why this module uses one.

Measured, this matters: hand-written definition patterns reach precision
**0.16** on real course material — roughly five extractions in six are wrong.
A regex would be worse, and the guard set below is what makes the difference
between "mostly noise" and "worth showing a reader".

## The four patterns

Each is a dependency shape, not a word list. Every one is found by walking the
parse, never by matching strings.

### 1. Copular — the most common and most reliable

> A stack **is** a linear data structure.

```
stack     nsubj ─┐
is        ROOT   ├─→ head of the complement
a         det  ──┤
linear    amod ──┤
data      compound┤
structure attr ──┘
```

**Rule:** find a token whose `dep_` is `attr` or `acomp`. Its head must be a
copula (`be`, `become`, `remain`, `seem`, `appear`) or have a `cop` child. The
subject is the head's `nsubj`. The term is the subject's phrase; the definition
is the complement's phrase.

### 2. Cue phrase — the strongest signal

> A stack **is defined as** a linear data structure.

A verb from a fixed lexicon, matched on **lemmas** so that "refers to",
"referred to" and "referring to" are one pattern:

```
is defined as · refers to · is known as · is called · means ·
denotes · is described as · is termed · represents
```

**Rule:** find a verb whose lemma sequence matches a cue. The term is its
`nsubj`; the definition is whatever it governs — a `dobj`, or the `pobj` of a
following preposition (which is how "refers **to** X" resolves).

### 3. Appositive — definition mid-sentence

> A stack, **a linear data structure**, stores items in order.

**Rule:** a token with `dep_ == "appos"` defines its head, provided the head is
a noun. Scored lower than the other two because the parser attaches appositives
loosely and it misfires more often.

### 4. Hypernym — a copular with a category marker

> A stack **is a type of** linear data structure.

Found by the copular rule, then classified separately when the complement
contains `type of`, `kind of`, `form of`, `class of`, `subset of`, `example of`
or `instance of`. It is its own pattern because the card reads differently:
"a type of X" is a worse card back than X itself.

## Expanding a token into a phrase

**This was the bug that produced no cards at all**, and it is worth recording.

The first version took `token.text` — the single head word. For "A stack is a
linear data structure", the `attr` token is `structure`, so the extracted
definition was the nine-character string `"structure"`, which the
minimum-length filter then correctly rejected as too thin. Every copular
definition in the document was found and thrown away, and the only symptom was
an empty card list.

A definition is a phrase. `_phrase_span` takes the token's **dependency
subtree** and excludes the labels that belong to a different clause:

| Excluded | Why |
| --- | --- |
| `conj` | "A stack is LIFO **and a queue is FIFO**" — the second half is a sibling, not part of the first |
| `cc` | the "and" itself |
| `mark` | subordinating conjunctions introducing a clause the card does not need |
| `punct` | trailing commas and full stops |

The subtree can be non-contiguous once exclusions are applied, so the enclosing
span is taken and trimmed of leading and trailing punctuation.

## The guard set

Five checks, each corresponding to a failure mode documented in the literature
rather than to a hunch. They run before a candidate becomes a `Definition`.

| Guard | Rejects | Example |
| --- | --- | --- |
| **Negation** | A `neg` anywhere in the copula's subtree | "Mediation is **not** a viable alternative" — matches the pattern exactly and asserts the opposite |
| **Modals** | `could`, `would`, `might`, `may`, `must`, `should`, `can`, `shall` as the copula lemma | "A stack **could be** a data structure" — hedged; extracting it turns a possibility into a fact |
| **Expletive subject** | A pronominal subject in `there, it, this, that, these, those, here` | "**There** are two kinds of stack" — the subject refers to nothing, and there is no coreference resolution here to find the second case |
| **Contrastive complement** | Complement beginning `another, other, others, different, similar, same` | "A stack is **another** data structure" — a comparison, not a definition |
| **Non-nominal head** | Complement whose root is not `NOUN` or `PROPN` | A definition's complement is a noun phrase; anything else is a fragment |

Plus the structural filters: term length, term token count, the `BAD_TERMS`
stoplist (pronouns and determiners that make a card front meaningless), a
12–320 character band on the definition, and one card per term, first pattern
wins in the order cue → copular → appositive.

## Scoring

`confidence` is a weighted sum, clamped to `[0, 1]`, and deliberately **not**
normalised across the document — 0.9 should mean "this is a definition" in
every document, not "this is the best of a bad batch".

| Term | Effect | Reasoning |
| --- | --- | --- |
| Pattern | cue **0.75**, copular/hypernym **0.6**, appositive **0.45** | An explicit signal beats a bare copula; an appositive is the easiest for the parser to misattach |
| Term POS | `NOUN`/`PROPN` **+0.15**, `PRON`/`DET` **−0.25** | Definitions are about things |
| Document frequency | ≥3 occurrences **+0.1**, 2 **+0.05** | A term the author returns to is what the document is about |
| Length ratio | definition longer than term **+0.05**; shorter than 1.5× **−0.1** | A "definition" shorter than its term is always wrong |

## Worked example

Input: *"A stack is a linear data structure that follows the Last In First Out
principle."*

1. Parse. Root is `is` (AUX). `structure` has `dep_ = attr`, head `is`; `is` has
   no `neg` child and its lemma `be` is a copula.
2. `_subject_of(is)` → `stack` (`nsubj`).
3. `_phrase_span(stack)` → `stack`; `_phrase_span(structure)` → the subtree
   `a linear data structure that follows the Last In First Out principle`
   (`relcl` is *not* excluded — it is part of the definition, not a sibling
   clause).
4. Guards: no negation, not modal, subject is a NOUN not a pronoun, complement
   does not open with a contrastive determiner, complement root `structure` is a
   NOUN. Passes.
5. `_clean_term` strips the determiner → `stack`. `_clean_definition` flattens,
   capitalises and terminates →
   `A linear data structure that follows the Last In First Out principle.`
6. Confidence: `0.6` (copular) `+ 0.15` (NOUN) `+ 0.05` (single alpha token)
   `+ 0.1` (`stack` occurs ≥3 times) `+ 0.05` (definition longer) = **0.95**,
   clamped.

Output: `{term: "stack", confidence: 0.95, pattern: "copular"}`.

## Known limitations

- **One sentence only.** Roughly half of the term/definition pairs in a
  reference corpus span sentence boundaries or need deduction. There is no
  coreference resolution in this stack, so those are simply missed. This is the
  dominant recall hole and it is not fixable with a parser alone.
- **`this`-anaphora.** Definitions introduced by "This means…" are lost, because
  `this` is rejected as an expletive subject. In one published pipeline those
  accounted for 23% of useful extractions — a real cost, paid deliberately
  because the alternative is a card whose front is the word "This".
- **No domain adaptation.** The cue lexicon is general English. A subject that
  defines terms in its own idiom will be under-served.
- **The 0.16 ceiling stands.** These guards cut the worst of the false
  positives; they do not turn a pattern extractor into a trained classifier. The
  published way to get from ~0.16 to ~0.78 is supervision — a classifier trained
  on annotated definition corpora — and that is the obvious next step if quality
  becomes the bottleneck.
