from __future__ import annotations

import logging
import os
import signal
import threading

import redis
from openai import APIConnectionError
from sqlalchemy.exc import DBAPIError

from .config import settings
from .db import Job, transaction
from .jobs import dispatch
from .queue import (
    broker,
    claim,
    ensure_group,
    fail,
    finish,
    flush_outbox,
    heartbeat,
    reconcile_delivery,
)

log = logging.getLogger("retail.worker")
stopping = threading.Event()


def retryable_error(exc):
    """Only transient failures retry; code errors and invalid credentials fail clearly."""
    status = getattr(exc, "status_code", None)
    return isinstance(
        exc, (TimeoutError, ConnectionError, APIConnectionError, redis.RedisError, DBAPIError)
    ) or (isinstance(status, int) and (status == 429 or status >= 500))


def process(client, message_id, fields, consumer):
    claimed = claim(fields["job_id"])
    if claimed is None:
        with transaction() as session:
            row = session.get(Job, fields["job_id"])
            terminal = not row or row.status in {"succeeded", "failed", "cancelled", "invalidated"}
            delayed = row and row.status == "queued"
            dead = row and row.status == "failed"
            conv_id = row.payload.get("conversation_id") if row and row.kind == "chat" else None
            if dead and conv_id:
                from .db import Conversation

                conv = session.get(Conversation, conv_id, with_for_update=True)
                if conv and conv.active_job == row.id:
                    conv.active_job = None
        if dead:
            client.xadd(
                settings.dead_stream, {"job_id": fields["job_id"], "error": "attempt limit"}
            )
        if terminal or delayed:
            client.xack(settings.stream, settings.group, message_id)
        return
    job, token = claimed
    completed = threading.Event()

    def pulse():
        while not completed.wait(max(2, settings.lease_seconds // 3)):
            try:
                if not heartbeat(job["id"], token, client, message_id, consumer):
                    return
            except Exception:
                log.warning("heartbeat temporarily unavailable for %s", job["id"])

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        result = dispatch(job, token)
        finish(job["id"], token, result)
    except Exception as exc:
        error = str(exc)
        if settings.api_key:
            error = error.replace(settings.api_key, "[redacted]")
        error = error[:3000]
        action = fail(job["id"], token, error, retryable=retryable_error(exc))
        log.error("job %s: %s: %s", job["id"], type(exc).__name__, error)
        if action == "dead":
            client.xadd(settings.dead_stream, {"job_id": job["id"], "error": error[:300]})
    finally:
        completed.set()
        thread.join(timeout=2)
    # A stale worker must not acknowledge a message now leased to its successor.
    with transaction() as session:
        live = session.get(Job, job["id"])
        acknowledge = live.status in {"succeeded", "failed", "cancelled", "invalidated"} or (
            live.status == "queued" and live.lease_token is None
        )
    if acknowledge:
        client.xack(settings.stream, settings.group, message_id)


def main():
    logging.basicConfig(level=logging.INFO)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    client = broker()
    consumer = f"worker-{os.getpid()}"
    ensure_group(client)
    while not stopping.is_set():
        try:
            reconcile_delivery()
            flush_outbox(client)
            recovered = client.xautoclaim(
                settings.stream,
                settings.group,
                consumer,
                min_idle_time=settings.lease_seconds * 1000,
                start_id="0-0",
                count=1,
            )
            entries = recovered[1]
            if not entries:
                items = client.xreadgroup(
                    settings.group, consumer, {settings.stream: ">"}, count=1, block=1000
                )
                entries = items[0][1] if items else []
            for message_id, fields in entries:
                process(client, message_id, fields, consumer)
        except Exception as exc:
            log.warning("queue unavailable: %s", type(exc).__name__)
            stopping.wait(2)


if __name__ == "__main__":
    main()
