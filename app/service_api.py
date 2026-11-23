from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select

from .config import model_signature
from .db import Deployment, Version, identifier, transaction
from .service import approve, audit, get_record, propose, ticket_dict
from .service_models import (
    Knowledge,
    KnowledgeScope,
    Proposal,
    ServiceAudit,
    ServiceRecord,
    ServiceTurn,
    SpecialistRun,
    Ticket,
    TicketMemory,
)
from .support_flow import task_scope

router = APIRouter(prefix="/api/service", tags=["售后业务"])


def conflict(fn):
    try:
        return fn()
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/overview")
def overview():
    with transaction() as s:
        counts = dict(s.execute(select(Ticket.status, func.count()).group_by(Ticket.status)).all())
        return {
            "tickets": counts,
            "pending_approvals": s.scalar(
                select(func.count()).select_from(Proposal).where(Proposal.status == "pending")
            ),
            "orders": s.scalar(
                select(func.count())
                .select_from(ServiceRecord)
                .where(ServiceRecord.kind == "orders")
            ),
            "knowledge": s.scalar(
                select(func.count()).select_from(Knowledge).where(Knowledge.status == "published")
            ),
        }


@router.get("/records")
def records(
    kind: Literal["orders", "users", "products"] = "orders",
    q: str = "",
    status: str = "",
    offset: int = 0,
):
    with transaction() as s:
        query = select(ServiceRecord).where(ServiceRecord.kind == kind)
        if q:
            query = query.where(
                or_(
                    ServiceRecord.id.contains(q),
                    func.cast(ServiceRecord.data, __import__("sqlalchemy").String).contains(q),
                )
            )
        if status and kind == "orders":
            query = query.where(ServiceRecord.data["status"].as_string() == status)
        total = s.scalar(select(func.count()).select_from(query.subquery()))
        rows = s.scalars(query.order_by(ServiceRecord.id).offset(max(0, offset)).limit(20)).all()
        return {
            "total": total,
            "rows": [{"id": r.id, "data": r.data, "revision": r.revision} for r in rows],
        }


class NewTicket(BaseModel):
    order_id: str
    title: str = Field(min_length=1, max_length=160)
    priority: Literal["normal", "urgent"] = "normal"
    version_id: str | None = None
    mode: Literal["baseline", "harness", "skills", "combined"] = "combined"


@router.post("/tickets", status_code=201)
def create_ticket(body: NewTicket):
    def create():
        with transaction() as s:
            order = get_record(s, "orders", body.order_id)
            vid = body.version_id or s.get(Deployment, 1).version_id
            v = s.get(Version, vid)
            if not v or v.status in {"invalidated", "superseded", "rejected"}:
                raise ValueError("版本不可用")
            if v.metadata_json.get("model_config") != model_signature():
                raise ValueError("版本模型配置不一致")
            t = Ticket(
                order_id=order.id,
                customer_id=order.data["user_id"],
                title=body.title,
                priority=body.priority,
                version_id=vid,
                mode=body.mode,
            )
            s.add(t)
            s.flush()
            audit(
                s,
                t.id,
                "ticket_created",
                {"order_id": order.id, "version_id": vid, "mode": body.mode},
            )
            return ticket_dict(t)

    return conflict(create)


@router.get("/tickets")
def tickets(status: str = ""):
    with transaction() as s:
        q = select(Ticket)
        if status:
            q = q.where(Ticket.status == status)
        return [ticket_dict(t) for t in s.scalars(q.order_by(Ticket.updated_at.desc()).limit(100))]


@router.get("/tickets/{tid}")
def ticket(tid: str):
    with transaction() as s:
        t = s.get(Ticket, tid)
        if not t:
            raise HTTPException(404, "工单不存在")
        order = get_record(s, "orders", t.order_id)
        customer = get_record(s, "users", t.customer_id)
        proposals = s.scalars(
            select(Proposal).where(Proposal.ticket_id == tid).order_by(Proposal.created_at.desc())
        ).all()
        events = s.scalars(
            select(ServiceAudit)
            .where(ServiceAudit.ticket_id == tid)
            .order_by(ServiceAudit.id.desc())
            .limit(40)
        ).all()
        turns = s.scalars(
            select(ServiceTurn)
            .where(ServiceTurn.ticket_id == tid)
            .order_by(ServiceTurn.created_at.desc())
            .limit(10)
        ).all()
        specialists = s.scalars(
            select(SpecialistRun)
            .where(SpecialistRun.ticket_id == tid)
            .order_by(SpecialistRun.created_at.desc())
            .limit(30)
        ).all()
        memory = s.get(TicketMemory, tid)
        return {
            "memory": memory.data if memory else None,
            "specialists": [
                {
                    "id": r.id,
                    "role": r.role,
                    "model": r.model,
                    "status": r.status,
                    "output": r.output,
                    "elapsed_ms": r.elapsed_ms,
                    "evidence": r.evidence,
                }
                for r in specialists
            ],
            **ticket_dict(t),
            "routed": task_scope(s, tid) is not None,
            "messages": t.messages,
            "order": order.data,
            "customer": customer.data,
            "proposals": [
                {
                    "id": p.id,
                    "action": p.action,
                    "parameters": p.parameters,
                    "summary": p.summary,
                    "status": p.status,
                }
                for p in proposals
            ],
            "audit": [
                {
                    "id": e.id,
                    "action": e.action,
                    "actor": e.actor,
                    "details": e.details,
                    "at": e.created_at.isoformat(),
                }
                for e in events
            ],
            "turns": [
                {
                    "id": r.id,
                    "status": r.status,
                    "stage": r.stage,
                    "error": r.error,
                    "trace": r.trace,
                }
                for r in turns
            ],
        }


class UpdateTicket(BaseModel):
    revision: int
    status: Literal["open", "human", "resolved", "closed"] | None = None
    owner: str | None = Field(default=None, max_length=80)
    priority: Literal["normal", "urgent"] | None = None
    note: str = Field(default="", max_length=4000)


@router.patch("/tickets/{tid}")
def update_ticket(tid: str, body: UpdateTicket):
    def update():
        with transaction() as s:
            t = s.get(Ticket, tid, with_for_update=True)
            if not t:
                raise ValueError("工单不存在")
            if t.revision != body.revision:
                raise ValueError("工单已变化，请刷新")
            if t.active_turn:
                raise ValueError("请等待当前 Agent 回复完成")
            if body.status and body.status != t.status:
                allowed = {
                    "open": {"human", "resolved"},
                    "human": {"open", "resolved"},
                    "waiting_confirmation": {"human"},
                    "resolved": {"closed", "open"},
                    "closed": {"open"},
                }
                if body.status not in allowed.get(t.status, set()):
                    raise ValueError("不允许该状态流转")
                if body.status == "human":
                    for p in s.scalars(
                        select(Proposal).where(
                            Proposal.ticket_id == tid, Proposal.status == "pending"
                        )
                    ):
                        p.status = "rejected"
                t.status = body.status
            if body.owner is not None:
                t.owner = body.owner
            if body.priority:
                t.priority = body.priority
            t.revision += 1
            audit(s, t.id, "ticket_updated", body.model_dump(exclude_none=True))
            return ticket_dict(t)

    return conflict(update)


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@router.post("/tickets/{tid}/messages", status_code=202)
def message(tid: str, body: Message, idempotency_key: str | None = Header(default=None)):
    key = idempotency_key or identifier()
    if len(key) > 128:
        raise HTTPException(422, "幂等键过长")

    def create():
        with transaction() as s:
            t = s.get(Ticket, tid, with_for_update=True)
            if not t:
                raise ValueError("工单不存在")
            old = s.scalar(select(ServiceTurn).where(ServiceTurn.key == key))
            if old:
                if old.ticket_id != tid or old.message != body.message:
                    raise ValueError("幂等键冲突")
                return {"id": old.id, "status": old.status}
            if task_scope(s, tid):
                raise ValueError("此工单来自统一客服，请在原会话继续发送消息")
            if t.active_turn:
                raise ValueError("请等待当前回复")
            if t.status not in {"open", "waiting_confirmation"}:
                raise ValueError("当前工单已转人工或已结束，重新打开后可使用 Agent")
            turn = ServiceTurn(ticket_id=tid, key=key, message=body.message)
            s.add(turn)
            s.flush()
            t.active_turn = turn.id
            t.revision += 1
            audit(s, tid, "message_submitted", {"turn_id": turn.id})
            return {"id": turn.id, "status": turn.status}

    return conflict(create)


class Action(BaseModel):
    action: Literal["cancel", "return", "exchange", "address"]
    parameters: dict
    summary: str = Field(min_length=1, max_length=1000)


@router.post("/tickets/{tid}/proposals")
def create_proposal(tid: str, body: Action):
    with transaction() as s:
        if task_scope(s, tid):
            raise HTTPException(409, "统一客服工单的方案由原会话的 Worker 生成")
    return conflict(lambda: propose(tid, body.action, body.parameters, body.summary))


class Approval(BaseModel):
    approved: bool
    revision: int


@router.post("/proposals/{pid}/decision")
def decision(pid: str, body: Approval):
    try:
        return conflict(lambda: approve(pid, body.approved, body.revision))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/knowledge")
def knowledge():
    with transaction() as s:
        return [
            {
                "id": d.id,
                "title": d.title,
                "content": d.content,
                "tags": d.tags,
                "status": d.status,
                "revision": d.revision,
                "order_states": (
                    s.get(KnowledgeScope, d.id).order_states if s.get(KnowledgeScope, d.id) else []
                ),
                "product_ids": (
                    s.get(KnowledgeScope, d.id).product_ids if s.get(KnowledgeScope, d.id) else []
                ),
            }
            for d in s.scalars(select(Knowledge).order_by(Knowledge.updated_at.desc()))
        ]


class Article(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=6000)
    tags: str = Field(default="", max_length=240)
    status: Literal["draft", "published", "archived"] = "draft"
    revision: int = 0
    order_states: list[
        Literal[
            "pending",
            "processed",
            "delivered",
            "cancelled",
            "return requested",
            "exchange requested",
        ]
    ] = Field(default_factory=list, max_length=6)
    product_ids: list[str] = Field(default_factory=list, max_length=50)


@router.post("/knowledge", status_code=201)
def add_article(body: Article):
    with transaction() as s:
        d = Knowledge(**body.model_dump(exclude={"order_states", "product_ids"}))
        s.add(d)
        s.flush()
        s.add(
            KnowledgeScope(
                knowledge_id=d.id, order_states=body.order_states, product_ids=body.product_ids
            )
        )
        audit(s, None, "knowledge_created", {"id": d.id, "status": d.status})
        return {"id": d.id}


@router.put("/knowledge/{kid}")
def edit_article(kid: str, body: Article):
    with transaction() as s:
        d = s.get(Knowledge, kid, with_for_update=True)
        if not d:
            raise HTTPException(404, "知识不存在")
        if d.revision != body.revision:
            raise HTTPException(409, "知识已变化，请刷新")
        before = {
            "title": d.title,
            "content": d.content,
            "status": d.status,
            "revision": d.revision,
        }
        for k, v in body.model_dump(exclude={"revision", "order_states", "product_ids"}).items():
            setattr(d, k, v)
        scope = s.get(KnowledgeScope, kid)
        if not scope:
            scope = KnowledgeScope(knowledge_id=kid)
            s.add(scope)
        scope.order_states, scope.product_ids = body.order_states, body.product_ids
        d.revision += 1
        audit(
            s, None, "knowledge_updated", {"id": d.id, "before": before, "after": body.model_dump()}
        )
        return {"id": d.id, "revision": d.revision}


class KnowledgeQuery(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    order_id: str | None = None


@router.post("/knowledge/search")
def knowledge_search(body: KnowledgeQuery):
    from .service import search_knowledge

    order = None
    if body.order_id:
        with transaction() as s:
            order = conflict(lambda: get_record(s, "orders", body.order_id)).data
    try:
        return {"sources": search_knowledge(body.query, order)}
    except Exception as exc:
        raise HTTPException(
            502, "政策检索暂未完成，请稍后重试；未将无依据结果当作有效政策"
        ) from exc
