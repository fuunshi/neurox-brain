"""
How work arrives and how results get back.

Two transports, and the choice between them is a deployment decision rather than
a code one:

- `webhook` — results are POSTed to a URL the caller supplied. Needs nothing but
  HTTP; loses a result if the receiver is down when it arrives.
- `amqp` — jobs and results travel over RabbitMQ. Durable, retried by the
  broker, and the right answer when a document is large enough that the caller
  should not be holding a connection open.

Which one runs is `BRAIN_TRANSPORT`. The HTTP routes are always available
regardless — they are how the API does a synchronous analysis, and they are what
makes the service testable by hand.
"""
