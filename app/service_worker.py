"""Dedicated product queue keeps interactive support separate from long experiments.
ServiceTurn is both the durable request and transactional outbox record.
"""

import os
import signal
import threading
import time
from datetime import timedelta

from sqlalchemy import or_, select

from .config import settings
from .db import identifier, now, transaction
from .queue import broker
from .service import audit, initialize_service
from .service_agent import run_turn
from .service_models import ServiceTurn, Ticket
from .support_flow import (
    ROUTED,
    RoutingPending,
    invalidate_proposals,
    lock_conversation_for_ticket,
    pending_input,
)

STREAM = "retail:service"
GROUP = "service-workers"
stopping = threading.Event()


def flush(client):
    with transaction() as s:
        rows = s.scalars(
            select(ServiceTurn)
            .where(ServiceTurn.status == "queued", ServiceTurn.published == 0)
            .with_for_update(skip_locked=True)
            .limit(20)
        ).all()
        for r in rows:
            client.xadd(STREAM, {"turn_id": r.id})
            r.published = 1
        # Redis loss or expired process lease: republish from authoritative DB.
        for r in s.scalars(
            select(ServiceTurn)
            .where(
                ServiceTurn.published != ROUTED,
                or_(
                    (ServiceTurn.status == "running") & (ServiceTurn.lease_until < now()),
                    (ServiceTurn.status == "queued")
                    & (ServiceTurn.created_at < now() - timedelta(minutes=5)),
                ),
            )
            .with_for_update(skip_locked=True)
            .limit(20)
        ):
            client.xadd(STREAM, {"turn_id": r.id})


def process_turn(tid, *, routed=False):
    token = identifier()
    with transaction() as s:
        if routed:
            ticket_id = s.scalar(select(ServiceTurn.ticket_id).where(ServiceTurn.id == tid))
            if not ticket_id:
                return True
            chat = lock_conversation_for_ticket(s, ticket_id)
            ticket = s.get(Ticket, ticket_id, with_for_update=True)
            if (
                not ticket
                or ticket.active_turn != tid
                or not chat
                or pending_input(s, chat.id, lock=True)
            ):
                return False
        r = s.get(ServiceTurn, tid, with_for_update=True, populate_existing=True)
        if not r or r.status not in {"queued", "running"} or (r.published == ROUTED) != routed:
            return True
        if routed and r.status != "queued":
            return False
        if r.status == "running" and r.lease_until and r.lease_until > now():
            return False
        r.status = "running"
        r.lease = token
        r.lease_until = now() + timedelta(seconds=180)
    done = threading.Event()
    deadline = time.monotonic() + 600

    def pulse():
        while not done.wait(30):
            if routed and time.monotonic() >= deadline:
                return
            try:
                with transaction() as s:
                    r = s.get(ServiceTurn, tid, with_for_update=True)
                    if (
                        not r
                        or r.lease != token
                        or r.status != "running"
                        or (routed and r.lease_until <= now())
                    ):
                        return
                    r.lease_until = now() + timedelta(seconds=180)
            except Exception:
                pass

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        run_turn(tid, token)
    except RoutingPending:
        with transaction() as s:
            r = s.get(ServiceTurn, tid, with_for_update=True)
            if r and r.lease == token and r.status == "running":
                r.status, r.stage = "queued", "等待新消息分派"
                r.lease = r.lease_until = None
    except Exception as exc:
        with transaction() as s:
            r = s.get(ServiceTurn, tid)
            if routed:
                lock_conversation_for_ticket(s, r.ticket_id)
            t = s.get(Ticket, r.ticket_id, with_for_update=True)
            r = s.get(ServiceTurn, tid, with_for_update=True, populate_existing=True)
            if r.lease == token and r.status == "running" and (not routed or r.lease_until > now()):
                r.status = "failed"
                error = str(exc)
                r.error = (
                    error.replace(settings.api_key, "[redacted]") if settings.api_key else error
                )[:1000]
                r.stage = "调用失败，可重新发送或转人工"
                r.lease = None
                r.lease_until = None
                if t.active_turn == tid:
                    if routed:
                        invalidate_proposals(s, t)
                    t.active_turn = None
                    t.revision += 1
                audit(s, t.id, "agent_failed", {"turn_id": tid, "error": r.error}, "system")
    finally:
        done.set()
        thread.join(timeout=1)
    return True


def process(client, mid, tid):
    if process_turn(tid):
        client.xack(STREAM, GROUP, mid)


def main():
    initialize_service()
    from .support_runner import run_scheduler

    scheduler = threading.Thread(target=run_scheduler, args=(stopping, process_turn), daemon=True)
    scheduler.start()
    client = broker()
    try:
        client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    consumer = "service-" + str(os.getpid())
    while not stopping.is_set():
        try:
            flush(client)
            reclaimed = client.xautoclaim(STREAM, GROUP, consumer, 180000, "0-0", count=1)[1]
            data = (
                []
                if reclaimed
                else client.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=1, block=1000)
            )
            for mid, fields in reclaimed or (data[0][1] if data else []):
                process(client, mid, fields["turn_id"])
        except Exception as e:
            print(
                type(e).__name__, str(e).replace(settings.api_key, "[redacted]")[:500], flush=True
            )
            stopping.wait(2)


if __name__ == "__main__":
    main()
