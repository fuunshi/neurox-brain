"""
RabbitMQ transport.

**Why a queue at all, when there is already an HTTP route.** Because the API
already has a queue for generation — BullMQ on Redis — and a large document does
not fit the request/response shape. Holding an HTTP connection open for thirty
seconds of parsing invites a proxy timeout, and a timeout leaves both sides
unsure whether the work happened. A queue makes the handoff explicit: the job is
durable, the worker acknowledges when it is done, and a worker that dies mid-run
has its message redelivered rather than lost.

**Why RabbitMQ rather than reusing BullMQ.** BullMQ is a Node library; its wire
format is Redis data structures that a Python process would have to reimplement
and keep in step with a moving upstream. RabbitMQ speaks AMQP, which is a
protocol with clients in both languages, and the backend already provisions a
RabbitMQ service in every compose file — it is currently unused, kept for exactly
this kind of purpose.

**Topology.**

    API  ──publish──▶  brain.jobs   ──consume──▶  neurox-brain worker
                                                       │
    API  ◀──consume──  brain.results  ◀──publish───────┘

Two queues, both durable. Results go to a *queue* rather than being published to
an exchange with routing keys, because there is exactly one consumer and one
kind of message; an exchange would add a routing concept that never varies.

**Acknowledgement.** A job is acknowledged only after its result has been
published *and confirmed* by the broker. Acknowledging first would mean a worker
crash between the two loses the result silently — the job would be marked done
and the API would wait forever.
"""

from __future__ import annotations

import asyncio
import json
import logging

import aio_pika

from ..config import settings
from ..nlp import analyse as analyse_module
from ..schemas import AnalysisResult, JobEnvelope

logger = logging.getLogger(__name__)

# How long to keep retrying a lost connection. Robust connections reconnect
# indefinitely; this bounds the *initial* connect, where a wrong URL should fail
# loudly rather than hanging a container forever.
CONNECT_TIMEOUT_SECONDS = 30.0


async def publish_result(
    channel: aio_pika.abc.AbstractChannel,
    result: AnalysisResult,
    envelope: JobEnvelope,
    error: str | None = None,
) -> None:
    """
    Send a finished job back, on the queue the sender named.

    The reply queue comes from the envelope rather than from configuration so
    that one worker can serve several callers with different result queues, and
    so a caller can use a temporary queue of its own without arranging anything
    here.
    """
    target = envelope.reply_to or settings.amqp_result_queue

    payload = {
        "jobId": envelope.job_id,
        "correlationId": envelope.correlation_id,
        "ok": error is None,
        "error": error,
        "result": result.model_dump(mode="json") if result is not None else None,
    }

    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            # The caller correlates on this. RabbitMQ's own `correlation_id`
            # property is set as well as the body field, so a consumer can
            # filter without parsing — useful for a consumer that only wants
            # its own results off a shared queue.
            correlation_id=envelope.correlation_id or envelope.job_id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=target,
    )


async def _handle(
    message: aio_pika.abc.AbstractIncomingMessage,
    channel: aio_pika.abc.AbstractChannel,
) -> None:
    """Analyse one job and publish the result."""
    try:
        envelope = JobEnvelope.model_validate_json(message.body)
    except Exception as exc:  # noqa: BLE001
        # A malformed message can never succeed on redelivery. Acknowledging it
        # and logging is the only way to stop it cycling; rejecting with requeue
        # would put it straight back at the head of the queue, forever.
        logger.error("Discarding malformed job: %s", exc)
        await message.ack()
        return

    logger.info("Job %s: %d characters", envelope.job_id, len(envelope.text))

    try:
        # `to_thread`, not a direct call. The analysis is CPU-bound and
        # synchronous, and running it on the event loop would block the AMQP
        # heartbeats — the broker would decide the worker was dead and
        # redeliver the job it is in the middle of doing.
        result = await asyncio.to_thread(
            analyse_module.analyse, envelope.text, envelope.title, envelope.options
        )

        await publish_result(channel, result, envelope)

        # Acknowledged only now, after the result is published. See the module
        # docstring: ack-then-publish loses the result if the worker dies in
        # between, and nothing would ever notice.
        await message.ack()
        logger.info(
            "Job %s done: %d cards, %d questions, %dms",
            envelope.job_id,
            len(result.cards),
            len(result.quiz),
            result.stats.elapsed_ms,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", envelope.job_id)

        try:
            await publish_result(channel, None, envelope, error=str(exc))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            # Publishing the failure failed too. The message is nacked below so
            # the broker redelivers it rather than dropping it.
            logger.exception("Could not publish the failure for %s", envelope.job_id)

        # `requeue=False` sends it to the dead-letter queue if one is
        # configured, and drops it otherwise. Requeueing a job that fails
        # deterministically — a malformed document — is an infinite loop that
        # also blocks every job behind it.
        await message.nack(requeue=False)


async def consume() -> None:
    """
    Run the worker loop until cancelled.

    Prefetch is one by default. Analysis is CPU-bound and the process is
    single-threaded for parsing; pulling a second job only means it waits while
    the broker's unacknowledged-message timer runs. Scale by running more
    processes, not by raising this.
    """
    connection = await asyncio.wait_for(
        aio_pika.connect_robust(settings.amqp_url),
        timeout=CONNECT_TIMEOUT_SECONDS,
    )

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=settings.amqp_prefetch)

        job_queue = await channel.declare_queue(settings.amqp_job_queue, durable=True)
        # Declared as well as published to, so a worker that starts before the
        # API has ever sent anything does not fail to publish its first result.
        await channel.declare_queue(settings.amqp_result_queue, durable=True)

        logger.info(
            "Consuming %s (prefetch %d); results to %s",
            settings.amqp_job_queue,
            settings.amqp_prefetch,
            settings.amqp_result_queue,
        )

        async with job_queue.iterator() as jobs:
            async for message in jobs:
                await _handle(message, channel)
