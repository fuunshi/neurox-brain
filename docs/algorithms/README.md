# The algorithms, and what to expect from them

This directory documents every algorithm the service runs, with the reasoning
behind each choice and a step-by-step breakdown of the ones that are not
obvious. The per-module docstrings in `src/neurox_brain/nlp/` say the same
things more briefly; these files are where the worked examples and the
literature live.

---

## Read this part first: the honest numbers

Everything below is a classical, non-neural method. That is a deliberate
constraint, and it has a measurable cost. These are the figures the published
work reports for systems of this kind, on real course material rather than on
the clean sentences one picks when demonstrating:

| What | Realistic value | Source |
| --- | --- | --- |
| Handcrafted definition-pattern precision, real course text | **0.16** (Dutch LT4eL, all types); **0.07** (punctuation/appositive patterns); **0.14** (Portuguese, CS for non-experts) | Borg thesis; Del Gaudio & Branco |
| Hearst patterns standalone | **F ≈ 0.15** | Snow, Jurafsky & Ng 2004 |
| A *tuned* definition system (classifier + index match) | **P 0.78 / R 0.78**; 83% understandable, 89% answerable | Steuer et al. 2021 |
| Distractors rated "plausible or somewhat" | **46.6%** classical best vs **53.4%** human | Jiang & Lee 2017 |
| Keyphrase F1@5, best unsupervised | **0.15–0.37**, dataset-dependent | Xie et al. 2023 |
| Auto-generated item acceptability across the question-generation field | median **72%**, field high 79% | Kurdi et al. 2020 |
| Inter-annotator agreement on "is this item good" | **Krippendorff α ≈ 0.2–0.5** | compiled, seven studies |

**What this means for the product, stated plainly:**

1. **Generated material is a draft, always.** The pipeline already lands
   generated cards as `DRAFT` rows that a reader accepts or discards, and this
   is why. It is not caution; it is the documented ceiling of the toolchain.
2. **The headline precision for definition extraction is around one in six.**
   The guards described in `definition-extraction.md` cut the worst of it, and
   they do not close that gap. Nothing cheap does.
3. **Nobody should claim better than ~70% acceptability** for automatically
   generated study items from this class of method. No published classical
   system has reached it, and a claim in that territory is a claim about the
   marketing rather than about the software.
4. **Even humans barely agree** on whether a generated question is good —
   α ≈ 0.2–0.5 between trained annotators. Any quality metric here is a
   tripwire against regression, not a grade.

---

## The single biggest risk, and the product decision it forces

The most consistent finding across the whole distractor literature is a
trade-off with no classical escape:

| Method | Plausible | Reliable (not also correct) |
| --- | --- | --- |
| Word-vector similarity | **46.6%** | **93.2%** |
| Spelling similarity | lower | 93.2% |
| POS + frequency | ~5% | **100%** |

Semantically clever distractors are the most tempting *and* the most likely to
be secretly correct. There is **no classical method that gets both.**

The resolution in the literature is unanimous, and it is a product decision
rather than an algorithm: **build the validity pipeline and the review UI, ship
fewer and better items, and let the scheduler and the testing effect carry the
learning.** Every system reporting high quality got there with a human in the
loop or with real learner data — not with a cleverer generator.

That is why this service prioritises *reliability*: a distractor that is also
correct teaches something false, and a multiple-choice question with two right
answers is not a slightly worse question, it is a broken one. `distractors.md`
describes how that priority is implemented.

---

## What this service deliberately does not do

Each of these was considered and rejected, with the reason:

| Not built | Why |
| --- | --- |
| An LLM or any transformer | Out of scope by constraint; and the evaluation metrics for the neural state of the art would not even run here |
| `en_core_web_lg` | 560MB on disk, ~750MB–1GB resident, for a parse that `sm` does within a point |
| `en_core_web_md`'s vectors | Pruned to 20,000 entries, so cosine similarity is a bucket artefact — see `operations.md` |
| A neural distractor generator | Beyond the constraint, and its evaluation metrics require transformers |
| Hearst patterns as a card source | F ≈ 0.15 standalone, and the output is hyponymy — list-shaped, which the card-design heuristics explicitly warn against |
| Mid-sentence content-word cloze | Where the 34% "several answers possible" failure lives — see `cloze.md` |
| Graph summarisation on short inputs | Below ~10 sentences the similarity graph has no meaningful stationary distribution — see `summarisation.md` |
| BLEU / ROUGE / METEOR as quality claims | Documented to correlate poorly with human judgement, and for distractors to correlate *negatively* with the learned metric (DISTO reports −0.69 against BLEURT) |
| `keybert` | Its core dependency is `sentence-transformers`, which pulls torch and transformers |
| `pyate`, `spacy-wordnet`, `rake-nltk` | Archived, dead, or unmaintained respectively |

---

## Do not repeat these as fact

A folklore list, because these are widely repeated and none of them survived
checking:

- "Hearst 1998 `or other` = 63% precision on NYT" — not verifiable from any primary source.
- "Always use four options" — contradicted by the psychometrics; the supported rule is "as many *functional* distractors as possible".
- "Nearest neighbours make good distractors" — they are frequently synonyms, which is the reliability failure above.
- The patent's similarity-band constants (0.6–0.85 usable, >0.93 synonymous). The *shape* is right; the numbers are asserted and model-specific.
- "TF-IDF is the industry standard for keyword extraction", and keyword-density stopping rules.
- "TextRank is the best summariser for short documents" — traces to one practitioner blog post.
- Cloze-versus-basic-card retention — **no controlled experiment exists**; it is practitioner doctrine.
- Wozniak's Twenty Rules as empirical findings — they are design heuristics derived from the SuperMemo model.
- "The spaCy docs recommend `lg`" — the docs contain a code comment; the recommendation is a maintainer statement in a GitHub discussion.
- Self-BLEU / Distinct-n "good" thresholds — unattributable.

Two things that *are* solid, and worth keeping in view when the numbers above
feel discouraging:

- **The testing effect is real and large.** Roediger & Karpicke 2006: at two
  days, tested 68% vs restudied 54% (d = 0.95). Rowland 2014's meta-analysis of
  61 studies: g = 0.50. Adesope et al. 2017 (272 effect sizes): practice tests
  beat restudy by +0.51, and **multiple-choice practice +0.70** — the format
  this service generates most of.
- **Mechanically simple items still measure something.** Project LISTEN's
  Reading Tutor found generated cloze questions predicted a standard
  comprehension measure at r = 0.85 *even though* many of their distractors
  violated syntactic constraints. A mediocre card that gets answered beats a
  perfect card that is never opened.

---

## The algorithms

| File | Question it answers | Method |
| --- | --- | --- |
| [sentence-filtering.md](sentence-filtering.md) | What is a sentence, and which ones are worth looking at? | spaCy segmentation + discard rules |
| [definition-extraction.md](definition-extraction.md) | What does this text define? | Dependency patterns over four constructions, plus a guard set |
| [cloze.md](cloze.md) | What should be blanked? | The definiendum, in its own definitional sentence |
| [distractors.md](distractors.md) | What wrong answers would a student consider? | Document terms, WordNet as a *filter*, head-noun matching |
| [keywords-tfidf.md](keywords-tfidf.md) | What is this text about? | TF-IDF over a corpus the service maintains itself |
| [textrank.md](textrank.md) | Which sentences represent the whole? | PageRank over a sentence-similarity graph |
| [summarisation.md](summarisation.md) | Which few of those, in what order? | TextRank or frequency, chosen by input length |
