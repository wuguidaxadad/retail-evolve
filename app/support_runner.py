"""Poll durable inputs and start bounded task runs only when needed."""

import logging
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select
from sqlalchemy.orm import aliased

from .db import now, transaction
from .service import audit
from .service_models import (
    ServiceTurn,
    SupportConversation,
    SupportDispatch,
    SupportMessage,
    SupportTask,
    Ticket,
)
from .support_flow import (
    MAX_RECOVERIES,
    ROUTED,
    UNROUTED,
    invalidate_proposals,
    lock_conversation_for_ticket,
    process_message,
)

log = logging.getLogger(__name__)


def ready_inputs(limit=20):
    prior = aliased(SupportMessage)
    blocked = (
        select(prior.id)
        .where(
            prior.conversation_id == SupportMessage.conversation_id,
            prior.seq < SupportMessage.seq,
            prior.status.in_(UNROUTED),
        )
        .exists()
    )
    with transaction() as session:
        return session.execute(
            select(SupportMessage.conversation_id, SupportMessage.id)
            .where(SupportMessage.status == "queued", ~blocked)
            .order_by(SupportMessage.id)
            .limit(limit)
        ).all()


def ready_turns(limit=20):
    pending = (
        select(SupportMessage.id)
        .where(
            SupportMessage.conversation_id == SupportTask.conversation_id,
            SupportMessage.status.in_(UNROUTED),
        )
        .exists()
    )
    with transaction() as session:
        return session.execute(
            select(Ticket.id, ServiceTurn.id)
            .join(ServiceTurn, Ticket.active_turn == ServiceTurn.id)
            .join(SupportTask, SupportTask.ticket_id == Ticket.id)
            .where(ServiceTurn.status == "queued", ServiceTurn.published == ROUTED, ~pending)
            .order_by(ServiceTurn.created_at, ServiceTurn.id)
            .limit(limit)
        ).all()


def recover_expired():
    stamp = now()
    with transaction() as session:
        messages = session.execute(
            select(SupportMessage.conversation_id, SupportMessage.id)
            .where(SupportMessage.status == "running", SupportMessage.lease_until <= stamp)
            .limit(20)
        ).all()
        turns = session.execute(
            select(ServiceTurn.ticket_id, ServiceTurn.id)
            .where(
                ServiceTurn.status == "running",
                ServiceTurn.published == ROUTED,
                ServiceTurn.lease_until <= stamp,
            )
            .limit(20)
        ).all()
    for cid, mid in messages:
        with transaction() as session:
            session.get(SupportConversation, cid, with_for_update=True)
            row = session.get(SupportMessage, mid, with_for_update=True)
            if not row or row.status != "running" or not row.lease_until or row.lease_until > now():
                continue
            row.recoveries += 1
            row.lease = row.lease_until = None
            row.status = "queued" if row.recoveries <= MAX_RECOVERIES else "failed"
            if row.status == "failed":
                row.reply = "分派多次中断，请重试此消息。"
    for tid, rid in turns:
        with transaction() as session:
            lock_conversation_for_ticket(session, tid)
            ticket = session.get(Ticket, tid, with_for_update=True)
            turn = session.get(ServiceTurn, rid, with_for_update=True)
            if (
                not ticket
                or not turn
                or ticket.active_turn != rid
                or turn.status != "running"
                or not turn.lease_until
                or turn.lease_until > now()
            ):
                continue
            dispatch = session.get(SupportDispatch, rid, with_for_update=True)
            dispatch.recoveries += 1
            turn.lease = turn.lease_until = None
            if ticket.status in {"human", "closed", "resolved"}:
                turn.status, turn.stage = "succeeded", "工单状态已保存"
                ticket.active_turn = None
                ticket.revision += 1
            elif dispatch.recoveries <= MAX_RECOVERIES:
                turn.status, turn.stage = "queued", "执行中断，等待恢复"
            else:
                turn.status, turn.stage = "failed", "多次中断，请重新发送或转人工"
                ticket.active_turn = None
                ticket.revision += 1
                invalidate_proposals(session, ticket)
            audit(
                session,
                tid,
                "support_watchdog",
                {"turn_id": rid, "attempt": dispatch.recoveries, "outcome": turn.status},
                "system",
            )


def run_scheduler(stopping, process_turn):
    with (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="support-router") as routers,
        ThreadPoolExecutor(max_workers=4, thread_name_prefix="support-task") as workers,
    ):
        routing, running = {}, {}
        while not stopping.is_set():
            for active in (routing, running):
                for key, future in list(active.items()):
                    if future.done():
                        del active[key]
                        if future.exception():
                            log.error(
                                "Support execution interrupted; watchdog will check the lease"
                            )
            try:
                recover_expired()
                for cid, mid in ready_inputs():
                    if len(routing) >= 2:
                        break
                    if cid not in routing:
                        routing[cid] = routers.submit(process_message, mid)
                for tid, rid in ready_turns():
                    if len(running) >= 4:
                        break
                    if tid not in running:
                        running[tid] = workers.submit(process_turn, rid, routed=True)
            except Exception:
                log.error("Support scheduler tick failed; will retry")
            stopping.wait(0.5)
