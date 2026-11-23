"""Conversation Router, durable dispatch, and task write fences."""

from datetime import timedelta
from hashlib import sha256
from hmac import compare_digest
from re import findall
from secrets import token_urlsafe
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .config import model_signature
from .db import Deployment, Version, identifier, now, transaction
from .service_llm import structured_report
from .service_models import (
    Proposal,
    ServiceRecord,
    ServiceTurn,
    SupportClarification,
    SupportConversation,
    SupportDispatch,
    SupportMessage,
    SupportTask,
    Ticket,
)

ROUTED = -1
LEASE_SECONDS = 180
MAX_RECOVERIES = 2
TITLES = {
    "cancel": "取消订单",
    "return": "退货",
    "exchange": "换货",
    "address": "改地址",
    "query": "订单查询",
    "human": "人工协助",
}
LIVE = ("open", "waiting_confirmation")
UNROUTED = ("queued", "running", "failed")
UNCONFIRMABLE = (*UNROUTED, "clarification")


class RoutingPending(ValueError):
    pass


class RouteItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["cancel", "return", "exchange", "address", "query", "human"]
    action: Literal["start", "follow_up", "revise", "withdraw"]
    ticket_id: str | None
    order_id: str | None
    source_quote: str = Field(min_length=1, max_length=4000)


class Routing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dispatches: list[RouteItem] = Field(max_length=8)
    reply: str = Field(max_length=1000)
    needs_clarification: bool = False


ROUTER_INSTRUCTION = """你是本地售后演示的消息 Router，只分派，不办理。
只处理消息中的明确诉求，把一句话里的不同事项分别分派。返回客户原文中的连续片段。
优先关联已有工单：follow_up 为补充/追问，revise 为明确改要求，withdraw 为停止尚未受理的办理。
同一工单或同一订单同类事项最多分派一次。已有工单填 ticket_id 和对应 kind，
新任务用 start 并填可唯一确定的 order_id。旧工单已受理的撤回请求不要当成新申请。
订单或工单有歧义时，请客户明确商品或订单号；不要按最近购买、排列位置来猜。
可以同时分派明确事项并询问不明确事项。reply 只用于澄清或回答简单寒暄，
还有未能明确分派的业务诉求时，needs_clarification 必须为 true。
不承诺业务处理结果。客户、工单和订单只能选自输入，不能扩大权限。"""


def token_digest(token):
    return sha256(token.encode()).hexdigest()


def create_conversation(customer_id):
    token = token_urlsafe(32)
    with transaction() as session:
        if not session.get(ServiceRecord, ("users", customer_id)):
            raise ValueError("演示客户不存在")
        deployment = session.get(Deployment, 1)
        version = session.get(Version, deployment.version_id) if deployment else None
        if (
            not version
            or version.status in {"invalidated", "superseded", "rejected"}
            or version.metadata_json.get("model_config") != model_signature()
        ):
            raise ValueError("当前部署版本不可用或模型配置不匹配")
        chat = SupportConversation(
            customer_id=customer_id, version_id=version.id, token_hash=token_digest(token)
        )
        session.add(chat)
        session.flush()
        return {"id": chat.id, "token": token}


def authorize(session, cid, token, lock=False):
    chat = session.get(SupportConversation, cid, with_for_update=lock)
    if not chat or not compare_digest(chat.token_hash, token_digest(token)):
        raise LookupError("会话不存在或访问凭证无效")
    return chat


def submit_message(cid, token, key, content, clarifies_id=None):
    with transaction() as session:
        chat = authorize(session, cid, token, lock=True)
        old = session.scalar(
            select(SupportMessage).where(
                SupportMessage.conversation_id == cid, SupportMessage.key == key
            )
        )
        if old:
            link = session.get(SupportClarification, old.id)
            if old.text != content or (link.clarifies_id if link else None) != clarifies_id:
                raise ValueError("幂等键已用于另一条消息")
            return {"id": old.id, "seq": old.seq}
        if clarifies_id:
            original = session.get(SupportMessage, clarifies_id, with_for_update=True)
            if (
                not original
                or original.conversation_id != cid
                or original.status != "clarification"
            ):
                raise ValueError("这条消息已无需补充，或不属于当前会话")
        row = SupportMessage(conversation_id=cid, seq=chat.next_seq, key=key, text=content)
        chat.next_seq += 1
        session.add(row)
        session.flush()
        if clarifies_id:
            session.add(SupportClarification(message_id=row.id, clarifies_id=clarifies_id))
        return {"id": row.id, "seq": row.seq}


def task_scope(session, tid):
    return session.get(SupportTask, tid)


def lock_conversation_for_ticket(session, tid):
    task = task_scope(session, tid)
    if task:
        return session.get(SupportConversation, task.conversation_id, with_for_update=True)
    return None


def pending_input(session, cid, *, lock=False):
    query = (
        select(SupportMessage)
        .where(SupportMessage.conversation_id == cid, SupportMessage.status.in_(UNROUTED))
        .order_by(SupportMessage.seq)
        .limit(1)
    )
    return session.scalar(query.with_for_update() if lock else query)


def require_routed(session, chat):
    if chat and pending_input(session, chat.id, lock=True):
        raise RoutingPending("会话有尚未分派的消息，请等待分派或重试失败消息")


def require_confirmable(session, chat):
    require_routed(session, chat)
    if chat and session.scalar(
        select(SupportMessage.id)
        .where(SupportMessage.conversation_id == chat.id, SupportMessage.status == "clarification")
        .limit(1)
        .with_for_update()
    ):
        raise ValueError("会话有尚未明确归属的诉求，请先处理或忽略该消息")


def router_candidates(session, chat):
    orders = session.scalars(
        select(ServiceRecord)
        .where(
            ServiceRecord.kind == "orders",
            ServiceRecord.data["user_id"].as_string() == chat.customer_id,
        )
        .order_by(ServiceRecord.id)
        .limit(50)
    ).all()
    tasks = session.execute(
        select(SupportTask, Ticket, SupportDispatch.quote)
        .join(Ticket, Ticket.id == SupportTask.ticket_id)
        .outerjoin(SupportDispatch, SupportDispatch.turn_id == Ticket.active_turn)
        .where(SupportTask.conversation_id == chat.id)
        .order_by(Ticket.updated_at.desc())
        .limit(50)
    )
    return {
        "orders": [
            {
                "id": order.id,
                "status": order.data["status"],
                "items": [item["name"] for item in order.data["items"]],
            }
            for order in orders
        ],
        "tickets": [
            {
                "id": ticket.id,
                "kind": task.kind,
                "order_id": ticket.order_id,
                "status": ticket.status,
                "pending_customer_input": pending_quote,
                "title": ticket.title,
                "recent": [
                    {"role": entry["role"], "text": entry["content"][:800]}
                    for entry in ticket.messages
                    if entry.get("role") in ("user", "assistant")
                    and isinstance(entry.get("content"), str)
                ][-2:],
            }
            for task, ticket, pending_quote in tasks
        ],
    }


def interpret(evidence):
    return Routing.model_validate(
        structured_report(ROUTER_INSTRUCTION, evidence, Routing.model_json_schema())
    )


def claim_message(mid):
    with transaction() as session:
        event = session.get(SupportMessage, mid)
        if not event:
            return None
        chat = session.get(SupportConversation, event.conversation_id, with_for_update=True)
        first = pending_input(session, chat.id, lock=True)
        if not first or first.id != mid or first.status != "queued":
            return None
        first.status, first.lease = "running", identifier()
        first.lease_until = now() + timedelta(seconds=LEASE_SECONDS)
        history = session.scalars(
            select(SupportMessage)
            .where(SupportMessage.conversation_id == chat.id, SupportMessage.seq < first.seq)
            .order_by(SupportMessage.seq.desc())
            .limit(6)
        ).all()
        link = session.get(SupportClarification, first.id)
        original = session.get(SupportMessage, link.clarifies_id) if link else None
        return first.lease, {
            "message": first.text,
            "message_id": first.id,
            "clarifies": {"text": original.text, "question": original.reply} if original else None,
            "recent_messages": [
                {"text": row.text, "router_reply": row.reply, "status": row.status}
                for row in reversed(history)
                if row.status != "discarded"
            ],
            **router_candidates(session, chat),
        }


def resolve_ticket(session, chat, item):
    if item.ticket_id:
        scope = session.get(SupportTask, item.ticket_id)
        if not scope or scope.conversation_id != chat.id or scope.kind != item.kind:
            raise ValueError("分派引用了错误的工单")
        ticket = session.get(Ticket, item.ticket_id, with_for_update=True)
        if (
            not ticket
            or ticket.customer_id != chat.customer_id
            or (item.order_id and item.order_id != ticket.order_id)
            or item.action == "start"
        ):
            raise ValueError("分派与工单不匹配")
        return ticket
    if item.action != "start" or not item.order_id:
        raise ValueError("需要已核实的订单才能建立任务")
    order = session.get(ServiceRecord, ("orders", item.order_id))
    if not order or order.data["user_id"] != chat.customer_id:
        raise ValueError("分派引用了不属于本客户的订单")
    ticket = session.scalar(
        select(Ticket)
        .join(SupportTask, SupportTask.ticket_id == Ticket.id)
        .where(
            SupportTask.conversation_id == chat.id,
            SupportTask.kind == item.kind,
            Ticket.order_id == order.id,
            Ticket.status.in_(LIVE),
        )
        .limit(1)
        .with_for_update()
    )
    if ticket:
        return ticket
    ticket = Ticket(
        title=f"{TITLES[item.kind]} · {order.id}",
        order_id=order.id,
        customer_id=chat.customer_id,
        version_id=chat.version_id,
        mode="combined",
    )
    session.add(ticket)
    session.flush()
    session.add(SupportTask(ticket_id=ticket.id, conversation_id=chat.id, kind=item.kind))
    return ticket


def dispatch(session, ticket, event, item, previous_request=None):
    if ticket.status not in LIVE:
        return False
    previous = (
        session.get(ServiceTurn, ticket.active_turn, with_for_update=True)
        if ticket.active_turn
        else None
    )
    text = item.source_quote
    if previous_request:
        text = f"此前待澄清的客户原话：{previous_request}\n本次补充：{text}"
    if previous and previous.status in ("queued", "running"):
        previous.status, previous.stage = "superseded", "已有更新的客户消息"
        previous.lease = previous.lease_until = None
        text = previous.message + "\n" + text
    invalidate_proposals(session, ticket)
    turn = ServiceTurn(
        ticket_id=ticket.id, key=f"support:{event.id}:{ticket.id}", message=text, published=ROUTED
    )
    session.add(turn)
    session.flush()
    session.add(
        SupportDispatch(
            turn_id=turn.id,
            message_id=event.id,
            ticket_id=ticket.id,
            quote=item.source_quote,
            action=item.action,
        )
    )
    ticket.active_turn = turn.id
    ticket.revision += 1
    from .service import audit

    audit(
        session,
        ticket.id,
        "support_dispatched",
        {"message_id": event.id, "turn_id": turn.id, "action": item.action},
        "router",
    )
    return True


def invalidate_proposals(session, ticket):
    for proposal in session.scalars(
        select(Proposal)
        .where(Proposal.ticket_id == ticket.id, Proposal.status == "pending")
        .with_for_update()
    ):
        proposal.status = "superseded"
    if ticket.status == "waiting_confirmation":
        ticket.status = "open"


def finish_routing(mid, lease, route, evidence):
    with transaction() as session:
        event = session.get(SupportMessage, mid)
        chat = session.get(SupportConversation, event.conversation_id, with_for_update=True)
        event = session.get(SupportMessage, mid, with_for_update=True, populate_existing=True)
        if event.status != "running" or event.lease != lease or event.lease_until <= now():
            return
        seen = set()
        received = False
        skipped = False
        clarification = route.needs_clarification or not route.dispatches
        link = session.get(SupportClarification, mid)
        original = (
            session.get(SupportMessage, link.clarifies_id, with_for_update=True) if link else None
        )
        previous_request = None
        if original and original.status == "clarification" and len(route.dispatches) == 1:
            already_dispatched = session.scalar(
                select(SupportDispatch.turn_id)
                .where(SupportDispatch.message_id == original.id)
                .limit(1)
            )
            if already_dispatched is None:
                previous_request = original.text
        for item in route.dispatches:
            if item.source_quote not in event.text:
                raise ValueError("分派引用并非客户原话")
            if item.ticket_id and item.ticket_id not in {row["id"] for row in evidence["tickets"]}:
                raise ValueError("工单不在 Router 本次的候选范围内")
            if item.order_id and item.order_id not in {row["id"] for row in evidence["orders"]}:
                raise ValueError("订单不在 Router 本次的候选范围内")
            ticket = resolve_ticket(session, chat, item)
            explicit_orders = set(findall(r"(?i)#W[0-9]+\b", item.source_quote))
            orders_in_message = set(findall(r"(?i)#W[0-9]+\b", event.text))
            if not explicit_orders and len(orders_in_message) == 1 and len(route.dispatches) == 1:
                explicit_orders = orders_in_message
            if explicit_orders and {value.upper() for value in explicit_orders} != {
                ticket.order_id.upper()
            }:
                raise ValueError("分派订单与客户明确给出的订单号不一致")
            if ticket.id in seen:
                raise ValueError("同一工单被重复分派")
            seen.add(ticket.id)
            dispatched = dispatch(session, ticket, event, item, previous_request)
            received |= dispatched
            skipped |= not dispatched
        clarification |= skipped
        event.status = "clarification" if clarification else "routed"
        event.lease = event.lease_until = None
        if not clarification and original and original.status == "clarification":
            original.status = "clarified"
        event.reply = route.reply or (
            "请补充具体订单或商品，以及你需要办理的事项。" if clarification and not skipped else ""
        )
        if clarification:
            event.reply += "\n有诉求尚未明确。可以继续补充；确认方案前请处理这条未明确消息。"
        if received and clarification:
            event.reply += "\n能够明确的事项已单独分派。"
        if skipped:
            event.reply += "\n已有事项已受理、关闭或转人工，请打开对应工单查看进度。"


def process_message(mid):
    claimed = claim_message(mid)
    if not claimed:
        return
    lease, evidence = claimed
    try:
        finish_routing(mid, lease, interpret(evidence), evidence)
    except Exception:
        with transaction() as session:
            event = session.get(SupportMessage, mid, with_for_update=True)
            if event and event.status == "running" and event.lease == lease:
                event.status, event.lease, event.lease_until = "failed", None, None
                event.reply = "本条消息未完成分派，请重试。"
