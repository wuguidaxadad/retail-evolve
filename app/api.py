from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from .bootstrap import initialize
from .config import ROOT, model_signature, settings
from .dataset import choose_tasks, raw_database, raw_tasks, splits, task_summary
from .db import (
    Conversation,
    Deployment,
    Job,
    Outbox,
    ReleaseEvent,
    RetailRecord,
    TaskRun,
    Version,
    identifier,
    transaction,
)
from .jobs import source_runs, version_dict
from .queue import broker, cache_job, ensure_group, submit_in_transaction
from .support_api import router as support_router


@asynccontextmanager
async def lifespan(app):
    if os.getenv("INITIALIZE_DATABASE", "1") == "1":
        initialize()
    ensure_group()
    from .service import initialize_service
    initialize_service()
    yield


app = FastAPI(title="Retail Evolve", version="1.0.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")


def job_dict(row):
    return {
        "id": row.id,
        "kind": row.kind,
        "status": row.status,
        "payload": row.payload,
        "result": row.result,
        "error": row.error,
        "attempts": row.attempts,
        "progress": row.progress,
        "stage": row.stage,
        "created_at": row.created_at.isoformat(),
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/health")
def health():
    checks = {}
    try:
        with transaction() as s:
            s.execute(text("SELECT 1"))
        checks["mysql"] = True
    except Exception:
        checks["mysql"] = False
    try:
        checks["redis"] = bool(broker().ping())
    except Exception:
        checks["redis"] = False
    return {
        "services": checks,
        "model": settings.agent_model,
        "provider": urlparse(settings.base_url).hostname,
        "model_configured": bool(settings.api_key),
    }


@app.get("/api/live")
def live():
    return {"alive": True}


@app.get("/api/ready")
def ready():
    report = health()
    if not all(report["services"].values()):
        raise HTTPException(503, "Storage dependencies unavailable")
    return {"ready": True}


@app.get("/api/overview")
def overview():
    with transaction() as s:
        deployment = s.get(Deployment, 1)
        counts = dict(
            s.execute(select(RetailRecord.kind, func.count()).group_by(RetailRecord.kind)).all()
        )
        states = dict(s.execute(select(Job.status, func.count()).group_by(Job.status)).all())
        return {
            "records": counts,
            "tasks": {k: len(v) for k, v in splits().items()},
            "deployment": {"version_id": deployment.version_id, "revision": deployment.revision},
            "jobs": states,
        }


@app.get("/api/tasks")
def tasks(split: Literal["learn", "validation", "final"] = "learn"):
    return [
        {"id": str(t["id"]), "summary": task_summary(t)}
        for t in raw_tasks()
        if str(t["id"]) in splits()[split]
    ]


@app.get("/api/records")
def records(kind: Literal["users", "orders", "products"] = "orders", limit: int = 10):
    with transaction() as s:
        rows = s.scalars(
            select(RetailRecord).where(RetailRecord.kind == kind).limit(max(1, min(limit, 50)))
        ).all()
        result = [{"id": r.id, "data": r.data} for r in rows]
        if kind == "orders":
            user_ids = {r.data["user_id"] for r in rows}
            customers = {
                r.id: r.data
                for r in s.scalars(
                    select(RetailRecord).where(
                        RetailRecord.kind == "users", RetailRecord.id.in_(user_ids)
                    )
                )
            }
            for row in result:
                user = customers.get(row["data"]["user_id"])
                if user:
                    row["customer"] = {"name": user["name"], "zip": user["address"]["zip"]}
        return result


@app.get("/api/versions")
def versions():
    with transaction() as s:
        # Sort narrow IDs, not multi-megabyte learning traces in JSON columns.
        ids = list(s.scalars(select(Version.id).order_by(Version.created_at.desc()).limit(60)))
        ids.append(s.get(Deployment, 1).version_id)
        manifest = ROOT / "docs/aligned-customer-experiment.json"
        if manifest.exists():
            ids.extend(json.loads(manifest.read_text()).get("versions", {}).values())
        result = []
        for vid in dict.fromkeys(ids):
            row = s.get(Version, vid)
            if row is None:
                continue
            value = version_dict(row)
            metadata = dict(value["metadata"])
            audit = metadata.pop("audit", [])
            if audit:
                metadata["learning_steps"] = len(audit)
                metadata["audit_note"] = "完整学习审计保留在数据库；列表不加载大型轨迹"
            value["metadata"] = metadata
            result.append(value)
        return result


@app.get("/api/jobs")
def jobs():
    with transaction() as s:
        return [
            job_dict(r) for r in s.scalars(select(Job).order_by(Job.created_at.desc()).limit(100))
        ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        cached = broker().get(f"retail:job:{job_id}")
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    with transaction() as s:
        row = s.get(Job, job_id)
        if not row:
            raise HTTPException(404, "任务不存在")
        result = job_dict(row)
    cache_job(job_id, result)
    return result


@app.get("/api/jobs/{job_id}/traces")
def traces(job_id: str):
    with transaction() as s:
        return [
            {
                "task_id": r.task_id,
                "trial": r.trial,
                "label": r.label,
                "reward": r.reward,
                "data": r.data,
            }
            for r in s.scalars(select(TaskRun).where(TaskRun.job_id == job_id))
        ]


class JobRequest(BaseModel):
    kind: Literal["evaluate", "learn", "evolve", "gate", "rsi"]
    version_id: str
    split: Literal["learn", "validation", "final", "upstream_test"] = "learn"
    mode: Literal["baseline", "skills", "harness", "combined"] = "combined"
    limit: int = Field(default=1, ge=1, le=settings.max_tasks)
    trials: int = Field(default=1, ge=1, le=4)
    task_ids: list[str] | None = None
    source_job_id: str | None = None
    train_ids: list[str] | None = None
    validation_ids: list[str] | None = None
    resume_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    optimizer_model: Literal["gpt-5.4", "gpt-6-luna", "hy3", "claude-opus-4-7", "deepseek-v4.1-flash"] | None = None


def invalidate_job(job_id):
    try:
        broker().delete(f"retail:job:{job_id}")
    except Exception:
        pass


@app.post("/api/jobs", status_code=202)
def create_job(request: JobRequest, idempotency_key: str | None = Header(default=None)):
    key = idempotency_key or identifier()
    if len(key) > 128:
        raise HTTPException(422, "幂等键过长")
    p = request.model_dump(exclude_none=True)
    kind = p.pop("kind")
    if kind == "evolve":
        raise HTTPException(422, "旧受限优化已停用，请使用完整 RSI")
    try:
        if kind == "evaluate":
            ids = choose_tasks(p["split"], p.get("task_ids"), p["limit"])
            if len(ids) > settings.max_tasks:
                raise ValueError("任务数超出单次限制")
            p["task_ids"] = ids
        elif kind in {"learn", "evolve"}:
            source_runs(p.get("source_job_id", ""), p["version_id"])
        elif kind == "gate":
            p["train_ids"] = choose_tasks("learn", p.get("train_ids"), p["limit"])
            p["validation_ids"] = choose_tasks("validation", p.get("validation_ids"), p["limit"])
            if max(len(p["train_ids"]), len(p["validation_ids"])) > settings.max_tasks:
                raise ValueError("任务数超出单次限制")
        with transaction() as s:
            existing = s.scalar(select(Job).where(Job.idempotency_key == key))
            if existing:
                stored = {
                    k: v
                    for k, v in existing.payload.items()
                    if k not in {"incumbent_version_id", "model_config"}
                }
                if existing.kind != kind or stored != p:
                    raise ValueError("相同幂等键对应了不同请求")
                return job_dict(existing)
            version = s.get(Version, p["version_id"], with_for_update=True)
            if not version:
                raise ValueError("版本不存在")
            if version.status in {"invalidated", "superseded"}:
                raise ValueError("该版本已作废，请选择有效版本")
            if (
                version.metadata_json.get("runtime_kind") == "auto_harness_full"
                and version.metadata_json.get("model_config") != model_signature()
            ):
                raise ValueError("此版本属于另一套模型配置，请选择当前模型的基线或候选")
            if kind == "rsi":
                if version.metadata_json.get("runtime_kind") != "auto_harness_full":
                    raise ValueError("完整 RSI 必须使用原生 Agent 基线，旧受限版本仅供历史查看")
                active_rsi = s.scalar(
                    select(Job).where(Job.kind == "rsi", Job.status.in_(["queued", "running"]))
                )
                if active_rsi:
                    raise HTTPException(409, "已有完整 RSI 正在执行，请查看该任务进度")
                if p.get("resume_run_id"):
                    previous = s.get(Job, p["resume_run_id"])
                    if not previous or previous.kind != "rsi" or previous.status != "failed":
                        raise ValueError("只能恢复已失败的完整 RSI 运行")
                    if (
                        previous.payload.get("version_id") != p["version_id"]
                        or previous.payload.get("model_config") != model_signature()
                    ):
                        raise ValueError("恢复实验必须保持相同基线和模型配置")
            if kind == "gate" and not version.parent_id:
                raise ValueError("请选择候选版本")
            if kind == "gate" and version.status == "validated":
                raise ValueError("已验证版本被冻结，不能重新改写检查状态")
            p["model_config"] = model_signature()
            job, _ = submit_in_transaction(s, kind, p, key)
            result = job_dict(job)
        return result
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except IntegrityError:
        with transaction() as s:
            job = s.scalar(select(Job).where(Job.idempotency_key == key))
            if not job or job.kind != kind or job.payload != p:
                raise HTTPException(409, "并发请求幂等键冲突")
            return job_dict(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with transaction() as s:
        job = s.get(Job, job_id, with_for_update=True)
        if not job:
            raise HTTPException(404)
        if job.status not in {"succeeded", "failed", "cancelled", "invalidated"}:
            job.status, job.stage = "cancelled", "已取消"
            job.lease_token, job.lease_until = None, None
            if job.kind == "chat":
                conv = s.get(Conversation, job.payload["conversation_id"], with_for_update=True)
                if conv.active_job == job.id:
                    conv.active_job = None
    invalidate_job(job_id)
    return {"status": job.status}


@app.post("/api/conversations", status_code=201)
def create_conversation():
    with transaction() as s:
        deployment = s.get(Deployment, 1)
        conv = Conversation(version_id=deployment.version_id, state=raw_database(), messages=[])
        s.add(conv)
        s.flush()
        return {"id": conv.id, "version_id": conv.version_id, "messages": []}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    with transaction() as s:
        c = s.get(Conversation, conversation_id)
        if not c:
            raise HTTPException(404)
        return {
            "id": c.id,
            "version_id": c.version_id,
            "messages": c.messages,
            "revision": c.revision,
            "active_job": c.active_job,
        }


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.post("/api/conversations/{conversation_id}/messages", status_code=202)
def send_message(
    conversation_id: str,
    request: MessageRequest,
    idempotency_key: str | None = Header(default=None),
):
    key = idempotency_key or identifier()
    if len(key) > 128:
        raise HTTPException(422, "幂等键过长")
    p = {"conversation_id": conversation_id, "message": request.message}
    try:
        with transaction() as s:
            c = s.get(Conversation, conversation_id, with_for_update=True)
            if not c:
                raise HTTPException(404)
            old = s.scalar(select(Job).where(Job.idempotency_key == key))
            if old:
                stored = {k: v for k, v in old.payload.items() if k != "model_config"}
                if old.kind != "chat" or stored != p:
                    raise HTTPException(409, "幂等键冲突")
                return job_dict(old)
            if c.active_job:
                raise HTTPException(409, "请等待本轮客服处理完成")
            p["model_config"] = model_signature()
            job, _ = submit_in_transaction(s, "chat", p, key)
            c.active_job = job.id
            return job_dict(job)
    except IntegrityError:
        raise HTTPException(409, "并发请求冲突，请使用原幂等键重试")


class ReleaseRequest(BaseModel):
    version_id: str
    expected_revision: int


def release_version(request, action):
    with transaction() as s:
        d = s.get(Deployment, 1, with_for_update=True)
        v = s.get(Version, request.version_id, with_for_update=True)
        if d.revision != request.expected_revision:
            raise HTTPException(409, "发布版本已变化，请刷新后重试")
        if not v or v.status != "validated":
            raise HTTPException(409, "版本需要先通过检查")
        historical = s.scalar(
            select(ReleaseEvent.id).where(ReleaseEvent.version_id == v.id).limit(1)
        )
        if action == "release" and v.id != "baseline" and not historical:
            report = v.metadata_json.get("gate", {})
            if (
                not report.get("passed")
                or report.get("incumbent_version_id") != d.version_id
                or report.get("model_config") != model_signature()
            ):
                raise HTTPException(409, "检查所对比的服务版本已变化，需要重新检查新候选")
        if action == "rollback" and v.id != "baseline":
            if not historical:
                raise HTTPException(409, "只能回滚至曾发布的版本")
        old = d.version_id
        d.version_id, d.revision = v.id, d.revision + 1
        s.add(
            ReleaseEvent(
                version_id=v.id, previous_id=old, action=action, details={"revision": d.revision}
            )
        )
        return {"version_id": v.id, "revision": d.revision}


@app.post("/api/releases")
def release(request: ReleaseRequest):
    return release_version(request, "release")


@app.post("/api/rollback")
def rollback(request: ReleaseRequest):
    return release_version(request, "rollback")


@app.get("/api/operations")
def operations():
    with transaction() as s:
        unpublished = s.scalar(
            select(func.count()).select_from(Outbox).where(Outbox.published.is_(False))
        )
        releases = [
            {
                "version_id": r.version_id,
                "previous_id": r.previous_id,
                "action": r.action,
                "created_at": r.created_at.isoformat(),
            }
            for r in s.scalars(select(ReleaseEvent).order_by(ReleaseEvent.id.desc()).limit(20))
        ]
    client = broker()
    return {
        "outbox_pending": unpublished,
        "queue_length": client.xlen(settings.stream),
        "pending": client.xpending(settings.stream, settings.group),
        "dead_letters": client.xlen(settings.dead_stream),
        "releases": releases,
    }


@app.get("/api/project")
def project():
    from .product import project_status
    return project_status()


@app.get("/api/demo-cases")
def business_cases():
    from .product import demo_cases
    return demo_cases()


@app.get("/api/conversations/{conversation_id}/orders/{order_id}")
def conversation_order(conversation_id: str, order_id: str):
    """Only read the selected order from the isolated conversation business state."""
    with transaction() as s:
        c = s.get(Conversation, conversation_id)
        if not c:
            raise HTTPException(404, "会话不存在")
        order = c.state.get('orders', {}).get(order_id)
        if not order:
            raise HTTPException(404, "订单不存在")
        original = raw_database()['orders'].get(order_id, {})
        fields = ('status', 'items', 'payment_history', 'return_items', 'exchange_items', 'address')
        changes = {key: {'before': original.get(key), 'after': order.get(key)}
                   for key in fields if original.get(key) != order.get(key)}
        return {'order': order, 'changes': changes, 'revision': c.revision}


from .service_api import router as service_router  # noqa: E402

app.include_router(service_router)
app.include_router(support_router)
