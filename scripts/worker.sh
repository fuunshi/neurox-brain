#!/usr/bin/env bash
# Runs the RabbitMQ worker. Requires BRAIN_TRANSPORT=amqp and a broker.
set -euo pipefail
cd "$(dirname "$0")/.."
exec env PYTHONPATH=src .venv/bin/python -m neurox_brain.worker
