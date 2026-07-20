"""
The queue worker entry point.

    python -m neurox_brain.worker

A separate process from the HTTP service, and it should be a separate
*container* too. The reasoning is the one in `transport/amqp.py`: analysis is
CPU-bound and blocking, so a worker busy on a large document cannot answer a
health probe or serve a synchronous request. Running the two together means the
API's latency is set by whatever the worker happens to be doing.

Unlike `main.py`, this loads the model eagerly. There is no port to bind and
nothing to report progress to, so a worker that is not ready to work should not
claim to be running.

**Graceful shutdown.** SIGTERM cancels the consume loop. `aio_pika`'s iterator
waits for the in-flight handler before unwinding, so a job in progress finishes
and is acknowledged rather than being redelivered from the start — which matters
because a long job redelivered on every deploy never completes.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from .config import settings
from .nlp import distractors
from .nlp.corpus import corpus
from .nlp.pipeline import get_nlp
from .transport.amqp import consume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading spaCy model %s…", settings.model)
    get_nlp()
    corpus.load()

    # Before the consumer starts, and so before any thread can reach it. See
    # `distractors.ensure_wordnet_loaded` for why this is not optional.
    distractors.ensure_wordnet_loaded()

    loop = asyncio.new_event_loop()

    async def run() -> None:
        stop = asyncio.Event()

        def request_stop(*_: object) -> None:
            if not stop.is_set():
                logger.info("Shutting down; waiting for the current job to finish.")
                stop.set()

        for signal_name in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signal_name, request_stop)

        consumer = asyncio.create_task(consume())
        stopper = asyncio.create_task(stop.wait())

        done, pending = await asyncio.wait(
            {consumer, stopper}, return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

        # Surface a crash rather than swallowing it: a consumer that died is the
        # one thing an operator must not have to infer from silence.
        for task in done:
            if task is consumer and (error := task.exception()) is not None:
                raise error

    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run())
    finally:
        corpus.save()
        logger.info("Stopped. Corpus saved (%d documents).", corpus.documents)
        loop.close()


if __name__ == "__main__":
    main()
