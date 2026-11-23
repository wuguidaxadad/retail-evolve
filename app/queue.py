"""Transactional outbox + Redis Streams, with DB leases and at-least-once delivery."""

from __future__ import annotations

import json
from datetime import timedelta

import redis
from sqlalchemy import select, update

from .config import settings
from .db import Job, Outbox, identifier, now, transaction


def broker() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=5)


def ensure_group(client=None):
    client = client or broker()
    try:
        client.xgroup_create(settings.stream, settings.group, id="0", mkstream=True)
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def submit_in_transaction(session, kind, payload, key):
    existing = session.scalar(select(Job).where(Job.idempotency_key == key))
    if existing:
        if existing.kind != kind or existing.payload != payload:
            raise ValueError("相同幂等键对应了不同请求")
        return existing, False
    job = Job(id=identifier(), kind=kind, payload=payload, idempotency_key=key)
    session.add(job)
    session.flush()
    session.add(Outbox(job_id=job.id, attempt=0))
    return job, True


def flush_outbox(client=None, limit=20) -> int:
    """DB commit after XADD. A crash in between can duplicate, never lose, a job."""
    client = client or broker()
    count = 0
    for _ in range(limit):
        with transaction() as session:
            row = session.scalar(
                select(Outbox)
                .where(Outbox.published.is_(False), Outbox.available_at <= now())
                .order_by(Outbox.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if not row:
                break
            client.xadd(settings.stream, {"job_id": row.job_id, "attempt": row.attempt})
            row.published = True
            count += 1
    return count


def reconcile_delivery(limit=20, stale_seconds=300):
    """Replay stale DB jobs; duplicate messages are fenced by database leases."""
    from sqlalchemy import and_, or_

    replayed = 0
    with transaction() as session:
        rows = session.scalars(
            select(Job)
            .where(
                Job.available_at <= now(),
                or_(
                    and_(
                        Job.status == "queued",
                        Job.updated_at < now() - timedelta(seconds=stale_seconds),
                    ),
                    and_(Job.status == "running", Job.lease_until < now()),
                ),
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
        for job in rows:
            event = session.scalar(
                select(Outbox)
                .where(Outbox.job_id == job.id)
                .order_by(Outbox.attempt.desc())
                .limit(1)
                .with_for_update()
            )
            if event and event.published:
                event.published = False
                event.available_at = now()
                job.updated_at = now()
                job.stage = "等待恢复投递"
                replayed += 1
    return replayed


def claim(job_id: str) -> tuple[dict, str] | None:
    with transaction() as session:
        job = session.get(Job, job_id, with_for_update=True)
        if not job or job.status in {"succeeded", "failed", "cancelled", "invalidated"}:
            return None
        if job.available_at > now():
            return None
        if job.status == "running" and job.lease_until and job.lease_until > now():
            return None
        if job.attempts >= settings.max_attempts:
            job.status = "failed"
            job.error = "执行次数达到上限"
            return None
        token = identifier()
        job.status = "running"
        job.attempts += 1
        job.lease_token = token
        job.lease_until = now() + timedelta(seconds=settings.lease_seconds)
        job.stage = "开始执行"
        return {"id": job.id, "kind": job.kind, "payload": job.payload}, token


def heartbeat(job_id, token, client, message_id, consumer="heartbeat"):
    with transaction() as session:
        count = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.lease_token == token, Job.status == "running")
            .values(lease_until=now() + timedelta(seconds=settings.lease_seconds))
        ).rowcount
    if count:
        client.xclaim(
            settings.stream,
            settings.group,
            consumer,
            min_idle_time=0,
            message_ids=[message_id],
            justid=True,
        )
    return bool(count)


def progress(job_id, token, value, stage):
    with transaction() as session:
        count = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.lease_token == token, Job.status == "running")
            .values(progress=value, stage=stage)
        ).rowcount
        if not count:
            raise RuntimeError("任务已取消或执行租约已失效")


def finish(job_id, token, result):
    with transaction() as session:
        changed = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.lease_token == token, Job.status == "running")
            .values(
                status="succeeded",
                result=result,
                progress=1,
                stage="完成",
                lease_token=None,
                lease_until=None,
            )
        ).rowcount
    return bool(changed)


def fail(job_id, token, error, retryable=True):
    with transaction() as session:
        job = session.get(Job, job_id, with_for_update=True)
        if not job or job.lease_token != token or job.status != "running":
            return "ignored"
        job.error = error
        job.lease_token = None
        job.lease_until = None
        if retryable and job.attempts < settings.max_attempts:
            job.status = "queued"
            job.stage = "稍后重试"
            job.available_at = now() + timedelta(seconds=min(60, 2**job.attempts))
            session.add(
                Outbox(
                    job_id=job.id,
                    attempt=job.attempts,
                    available_at=job.available_at,
                )
            )
            return "retry"
        job.status = "failed"
        job.stage = "执行失败"
        if job.kind == "chat":
            from .db import Conversation

            conv = session.get(Conversation, job.payload["conversation_id"])
            if conv and conv.active_job == job.id:
                conv.active_job = None
        return "dead"


def cache_job(job_id, data):
    try:
        broker().setex(f"retail:job:{job_id}", 2, json.dumps(data, ensure_ascii=False))
    except redis.RedisError:
        pass  # MySQL remains authoritative; cache failures do not lose tasks.
