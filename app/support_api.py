"""The thin customer conversation entry for the local demonstration."""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from .db import transaction
from .service import approve
from .service_models import Proposal, ServiceTurn, SupportMessage, SupportTask, Ticket
from .support_flow import UNCONFIRMABLE, authorize, create_conversation, submit_message

router = APIRouter(prefix="/api/support", tags=["统一售后客服"])


class StartConversation(BaseModel):
    customer_id: str = Field(min_length=1, max_length=80)


class CustomerMessage(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    key: str = Field(min_length=1, max_length=128)
    clarifies_id: str | None = Field(default=None, max_length=32)


class Decision(BaseModel):
    approved: bool
    revision: int


def support_call(function):
    try:
        return function()
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/conversations", status_code=201)
def start(body: StartConversation):
    return support_call(lambda: create_conversation(body.customer_id.strip()))


@router.post("/conversations/{cid}/messages", status_code=202)
def send(cid: str, body: CustomerMessage, support_token: str = Header()):
    def save():
        if not body.message.strip():
            raise ValueError("请输入售后诉求")
        return submit_message(cid, support_token, body.key, body.message, body.clarifies_id)

    return support_call(save)


@router.post("/conversations/{cid}/messages/{mid}/retry", status_code=202)
def retry(cid: str, mid: str, support_token: str = Header()):
    def reset():
        with transaction() as session:
            authorize(session, cid, support_token, lock=True)
            row = session.get(SupportMessage, mid, with_for_update=True)
            if not row or row.conversation_id != cid:
                raise LookupError("消息不属于当前会话")
            if row.status != "failed":
                raise ValueError("只可重试未完成分派的消息")
            row.status, row.reply, row.recoveries = "queued", "", 0
            return {"id": mid, "status": row.status}

    return support_call(reset)


@router.post("/conversations/{cid}/messages/{mid}/discard")
def discard(cid: str, mid: str, support_token: str = Header()):
    def close():
        with transaction() as session:
            authorize(session, cid, support_token, lock=True)
            row = session.get(SupportMessage, mid, with_for_update=True)
            if not row or row.conversation_id != cid:
                raise LookupError("消息不属于当前会话")
            if row.status == "discarded":
                return {"id": mid, "status": row.status}
            if row.status not in {"failed", "clarification"}:
                raise ValueError("只可忽略分派失败或仍待补充的内容")
            row.status = "discarded"
            row.reply = "未明确的内容已忽略；已经分派的任务仍会继续。"
            return {"id": mid, "status": row.status}

    return support_call(close)


@router.get("/conversations/{cid}")
def conversation(cid: str, support_token: str = Header()):
    def read():
        with transaction() as session:
            chat = authorize(session, cid, support_token)
            events = session.scalars(
                select(SupportMessage)
                .where(SupportMessage.conversation_id == cid)
                .order_by(SupportMessage.seq.desc())
                .limit(100)
            ).all()
            blocked = (
                session.scalar(
                    select(SupportMessage.id)
                    .where(
                        SupportMessage.conversation_id == cid,
                        SupportMessage.status.in_(UNCONFIRMABLE),
                    )
                    .limit(1)
                )
                is not None
            )
            pairs = session.execute(
                select(SupportTask, Ticket)
                .join(Ticket, Ticket.id == SupportTask.ticket_id)
                .where(SupportTask.conversation_id == cid)
                .order_by(Ticket.updated_at.desc())
            ).all()
            tasks = []
            for scope, ticket in pairs:
                pending = session.scalar(
                    select(Proposal).where(
                        Proposal.ticket_id == ticket.id, Proposal.status == "pending"
                    )
                )
                latest = session.scalar(
                    select(ServiceTurn)
                    .where(ServiceTurn.ticket_id == ticket.id)
                    .order_by(ServiceTurn.created_at.desc(), ServiceTurn.id.desc())
                    .limit(1)
                )
                tasks.append(
                    {
                        "id": ticket.id,
                        "title": ticket.title,
                        "status": ticket.status,
                        "kind": scope.kind,
                        "revision": ticket.revision,
                        "working": ticket.active_turn is not None,
                        "error": latest.stage if latest and latest.status == "failed" else None,
                        "replies": [
                            entry["content"]
                            for entry in ticket.messages
                            if entry.get("role") == "assistant" and entry.get("content")
                        ][-8:],
                        "proposal": (
                            {
                                "id": pending.id,
                                "summary": pending.summary,
                                "parameters": pending.parameters,
                            }
                            if pending and not blocked and not ticket.active_turn
                            else None
                        ),
                    }
                )
            return {
                "id": cid,
                "customer_id": chat.customer_id,
                "tasks": tasks,
                "messages": [
                    {"id": row.id, "text": row.text, "status": row.status, "reply": row.reply}
                    for row in reversed(events)
                ],
            }

    return support_call(read)


@router.post("/conversations/{cid}/proposals/{pid}/decision")
def decide(cid: str, pid: str, body: Decision, support_token: str = Header()):
    def confirm():
        with transaction() as session:
            authorize(session, cid, support_token)
        return approve(pid, body.approved, body.revision, conversation_id=cid)

    return support_call(confirm)
