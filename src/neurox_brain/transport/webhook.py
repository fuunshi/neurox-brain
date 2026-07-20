"""
Posting a result back to a caller that did not wait.

**When this is the right transport.** The API has two ways to use this service
and they have different shapes:

- **Synchronous** — the API calls `POST /analyse` and holds the request open.
  Simple, needs no extra infrastructure, and correct for the sizes the API
  actually sends: a 2,000-character chunk analyses in well under a second.
- **Callback** — the API sends work and forgets, and this service posts the
  answer to a URL when it is done. This is what webhooks are for, and it is the
  fallback the plan asked for: it needs no message broker, only an HTTP endpoint
  on the API side.

**What this is not.** It is not a substitute for the AMQP path when RabbitMQ is
available. A webhook has no delivery guarantee — if the API is restarting when
the post arrives, the result is gone — whereas a durable queue retries until the
consumer acknowledges. The webhook sender retries a few times, which covers a
blip and not an outage, and that difference is the reason to prefer the queue
when there is one.

**Signature.** Every post carries an HMAC-SHA256 of the body in
`X-Brain-Signature`, computed over the raw bytes with a shared secret. The API
verifies it before trusting the payload. Without it, any host that can reach the
API's callback endpoint can inject a job result — and a job result is what
writes cards into somebody's deck.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging

import httpx

logger = logging.getLogger(__name__)

# Attempts, and the gap between them. Three attempts over roughly seven seconds
# rides out a restart; it is not meant to survive an outage.
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (0.5, 2.0, 5.0)

REQUEST_TIMEOUT_SECONDS = 15.0


def sign(body: bytes, secret: str) -> str:
    """
    The signature for a payload.

    Over the raw bytes rather than a re-serialised dict: the receiving side
    verifies against exactly what was sent, and JSON key order is not guaranteed
    to survive a parse-and-dump round trip. Signing a re-serialisation is how
    signature checks fail intermittently and for no visible reason.
    """
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


async def deliver(
    url: str,
    payload: dict,
    secret: str,
    *,
    headers: dict[str, str] | None = None,
) -> bool:
    """
    Post `payload` to `url`, retrying briefly. Returns whether it landed.

    Returns a bool rather than raising: a failed delivery is a fact the caller
    records against the job, not an exception it can do anything about. Raising
    would make every caller write the same try/except around a network call it
    cannot fix.
    """
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    request_headers = {
        "content-type": "application/json",
        "x-brain-signature": sign(body, secret),
        **(headers or {}),
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for attempt, delay in enumerate(BACKOFF_SECONDS[:MAX_ATTEMPTS], start=1):
            try:
                response = await client.post(url, content=body, headers=request_headers)

                # 4xx other than 429 is the caller rejecting the payload; retrying
                # sends the same rejected bytes three times and delays the real
                # error. 5xx and 429 are worth another go.
                if 200 <= response.status_code < 300:
                    logger.info("Delivered %s (%d) on attempt %d", url, response.status_code, attempt)
                    return True

                if 400 <= response.status_code < 500 and response.status_code != 429:
                    logger.error(
                        "Delivery to %s rejected with %d; not retrying: %s",
                        url,
                        response.status_code,
                        response.text[:200],
                    )
                    return False

                logger.warning(
                    "Delivery to %s got %d on attempt %d",
                    url,
                    response.status_code,
                    attempt,
                )
            except httpx.HTTPError as exc:
                logger.warning("Delivery to %s failed on attempt %d: %s", url, attempt, exc)

            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(delay)

    logger.error("Gave up delivering to %s after %d attempts.", url, MAX_ATTEMPTS)
    return False
