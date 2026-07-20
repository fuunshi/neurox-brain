"""
neurox-brain: turns text into study material.

A separate service rather than a library inside the API, for three reasons that
are worth stating because "why is this not just a Python module the Nest app
shells out to" is the obvious question:

1. **It is a different runtime with a different failure mode.** A spaCy model is
   300MB of resident memory and a second of load time. Putting that inside the
   API process means every API instance pays for it whether or not anyone is
   generating cards, and a malformed PDF that crashes the parser takes the HTTP
   server with it. Here, it crashes one worker and the API returns a 500.
2. **It scales on a different axis.** Text analysis is CPU-bound and bursty;
   serving pages is neither. Running them in one process means sizing the web
   tier for the NLP tier.
3. **It is replaceable.** The contract is four JSON endpoints. If the algorithms
   are later replaced by a fine-tuned model, or by an LLM, the API does not
   change — which is the same reasoning that makes the existing `CardGenerator`
   interface in the Nest app swappable.
"""

__version__ = "0.1.0"
