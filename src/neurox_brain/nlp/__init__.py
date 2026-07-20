"""
The extraction algorithms.

One module per question, and the module name is the question it answers:

- `pipeline`   — what is a sentence, and what is worth looking at?
- `definitions`— what does this text define?
- `keywords`   — what is this text about?
- `corpus`     — how common is a term across everything we have seen?
- `textrank`   — which sentences represent the whole?
- `summarise`  — pick the top few of those, in reading order.
- `cloze`      — what should be blanked, and why that word?
- `distractors`— what wrong answers would a student actually consider?
- `analyse`    — in what order do those run?

Each module's docstring explains the algorithm it implements and, where a
simpler approach exists, why that one was not taken. `docs/algorithms/` covers
the same ground at more length, with worked examples.
"""
