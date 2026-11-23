"""Order service and ticket actions. Approval is enforced in code, never by prompts alone."""

from copy import deepcopy

from sqlalchemy import select
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment

from .dataset import raw_database
from .db import Base, engine, transaction
from .service_models import Knowledge, Proposal, ServiceAudit, ServiceRecord, ServiceTurn, Ticket

SEEDS = [
    (
        "取消未发货订单",
        "只有 pending 状态订单可以取消。原因限于 no longer needed 或 ordered by mistake。先展示取消方案，用户明确点击确认后执行。原支付方式退款，礼品卡余额立即回补，其他退款通常需要 5–7 个工作日。",
        "cancel 取消 pending 退款",
    ),
    (
        "收货后退货",
        "只有 delivered 状态可以申请退货。确认商品与数量，退款到原支付方式或客户已有礼品卡。申请成功表示 return requested，不代表退款已到账。",
        "return 退货 delivered 退款",
    ),
    (
        "换货与库存",
        "已送达订单换货需匹配同一商品的其他规格，先查询库存和差价。提交换货方案时要提供原商品、新规格、库存和差价依据，人工确认后才执行。",
        "exchange 换货 库存",
    ),
    (
        "人工接管",
        "遇到政策例外、物流争议、支付异常、客户明确要求人工时，记录原因并转人工。停止 Agent 自动操作，保留对话和工具记录供接手。",
        "human 人工 物流 支付",
    ),
    (
        "发货前修改配送地址",
        "只允许修改 pending 订单的配送地址。必须提供 address1、address2（可为空）、city、state、country、zip。先生成完整地址方案，操作员确认后执行。已发货或已送达订单不得自动改地址，转人工。",
        "address 地址 配送 pending",
    ),
]


def initialize_service():
    Base.metadata.create_all(engine)
    with transaction() as s:
        if not s.scalar(select(ServiceRecord.id).limit(1)):
            for kind, rows in raw_database().items():
                s.add_all(ServiceRecord(kind=kind, id=k, data=v) for k, v in rows.items())
        for i, (title, content, tags) in enumerate(SEEDS):
            if not s.get(Knowledge, f"policy-{i}"):
                s.add(
                    Knowledge(
                        id=f"policy-{i}",
                        title=title,
                        content=content,
                        tags=tags,
                        status="published",
                    )
                )


def audit(s, ticket_id, action, details, actor="operator"):
    s.add(ServiceAudit(ticket_id=ticket_id, action=action, actor=actor, details=details))


def ticket_dict(t):
    return {
        k: getattr(t, k)
        for k in (
            "id",
            "title",
            "order_id",
            "customer_id",
            "status",
            "priority",
            "owner",
            "version_id",
            "mode",
            "revision",
            "active_turn",
        )
    }


def get_record(s, kind, rid, lock=False):
    r = s.get(ServiceRecord, (kind, rid), with_for_update=lock)
    if not r:
        raise ValueError("业务数据不存在")
    return r


def search_knowledge(query, order=None):
    from .service_retrieval import search_policies

    return search_policies(query, order)


def execute_action(db, order_id, action, parameters):
    env = get_environment(db=RetailDB.model_validate(deepcopy(db)))
    if action == "cancel":
        env.make_tool_call(
            "cancel_pending_order", order_id=order_id, reason=parameters.get("reason", "")
        )
    elif action == "return":
        if not parameters.get("item_ids"):
            raise ValueError("请先选择要退货的商品")
        env.make_tool_call(
            "return_delivered_order_items",
            order_id=order_id,
            item_ids=parameters["item_ids"],
            payment_method_id=parameters.get("payment_method_id", ""),
        )
    elif action == "exchange":
        if not parameters.get("item_ids") or len(parameters["item_ids"]) != len(
            parameters.get("new_item_ids", [])
        ):
            raise ValueError("必须提供一一对应的原商品与新规格")
        if any(a == b for a, b in zip(parameters["item_ids"], parameters["new_item_ids"])):
            raise ValueError("换货必须选择不同规格，不能消耗一次性操作资格做无效换货")
        env.make_tool_call(
            "exchange_delivered_order_items",
            order_id=order_id,
            item_ids=parameters["item_ids"],
            new_item_ids=parameters["new_item_ids"],
            payment_method_id=parameters.get("payment_method_id", ""),
        )
    elif action == "address":
        keys = ("address1", "address2", "city", "state", "country", "zip")
        if any(not isinstance(parameters.get(k), str) for k in keys):
            raise ValueError("新地址字段不完整")
        if any(not parameters[k].strip() for k in keys if k != "address2"):
            raise ValueError("新地址必填字段不能为空")
        env.make_tool_call(
            "modify_pending_order_address", order_id=order_id, **{k: parameters[k] for k in keys}
        )
    else:
        raise ValueError("不支持的订单操作，请转人工")
    return env.tools.db.model_dump(mode="json")


def action_snapshot(s, ticket, lock=False):
    order = get_record(s, "orders", ticket.order_id, lock)
    user = get_record(s, "users", ticket.customer_id, lock)
    if order.data["user_id"] != ticket.customer_id:
        raise ValueError("客户与订单不匹配")
    # Only this customer's selected order is writable in this product workflow.
    data = raw_database()
    data["orders"] = {ticket.order_id: order.data}
    data["users"] = {ticket.customer_id: user.data}
    return data, order, user


def propose(
    ticket_id, action, parameters, summary, turn_id=None, lease=None, expected_order_revision=None
):
    from .support_flow import lock_conversation_for_ticket, require_routed, task_scope

    with transaction() as s:
        chat = lock_conversation_for_ticket(s, ticket_id)
        t = s.get(Ticket, ticket_id, with_for_update=True)
        if not t or t.status not in {"open", "waiting_confirmation"}:
            raise ValueError("工单当前不可提出自动处理方案")
        if turn_id:
            check_turn(s, t, turn_id, lease)
        if chat:
            require_routed(s, chat)
            if action != task_scope(s, ticket_id).kind:
                raise ValueError("方案超出这张工单的业务范围")
        existing = s.scalar(
            select(Proposal).where(Proposal.ticket_id == ticket_id, Proposal.status == "pending")
        )
        if existing:
            if existing.action != action or existing.parameters != parameters:
                raise ValueError("已有待确认方案，请先确认或驳回")
            return {"proposal_id": existing.id, "status": "waiting_confirmation"}
        db, order, _ = action_snapshot(s, t, lock=True)
        if expected_order_revision is not None and order.revision != expected_order_revision:
            raise ValueError("订单已变化，请重新生成方案")
        execute_action(db, t.order_id, action, parameters)  # Validate without writing.
        p = Proposal(
            ticket_id=t.id,
            action=action,
            parameters=parameters,
            order_revision=order.revision,
            summary=summary,
        )
        s.add(p)
        s.flush()
        t.status = "waiting_confirmation"
        t.revision += 1
        audit(
            s,
            t.id,
            "proposal_created",
            {"proposal_id": p.id, "action": action, "parameters": parameters},
            "agent" if turn_id else "operator",
        )
        return {
            "proposal_id": p.id,
            "status": "waiting_confirmation",
            "message": "方案未执行，必须在界面点击确认",
        }


def approve(proposal_id, approved, expected_revision, *, conversation_id=None):
    from .support_flow import lock_conversation_for_ticket, require_confirmable

    with transaction() as s:
        initial = s.get(Proposal, proposal_id)
        if not initial:
            raise ValueError("方案不存在")
        chat = lock_conversation_for_ticket(s, initial.ticket_id)
        if (chat.id if chat else None) != conversation_id:
            raise LookupError("方案不属于当前会话")
        t = s.get(Ticket, initial.ticket_id, with_for_update=True)
        p = s.get(Proposal, proposal_id, with_for_update=True, populate_existing=True)
        if p.status != "pending":
            return {"status": p.status, "ticket_id": t.id}
        require_confirmable(s, chat)
        if t.active_turn:
            raise ValueError("请等待 Agent 本轮结束后再确认")
        if t.revision != expected_revision:
            raise ValueError("工单已变化，请刷新后重新确认")
        if t.status != "waiting_confirmation":
            raise ValueError("工单状态不允许审批")
        if not approved:
            p.status = "rejected"
            t.status = "open"
            t.revision += 1
            audit(s, t.id, "proposal_rejected", {"proposal_id": p.id})
            return {"status": p.status, "ticket_id": t.id}
        db, order, user = action_snapshot(s, t, True)
        if order.revision != p.order_revision:
            raise ValueError("订单已被其他工单修改，请驳回并重新生成方案")
        after = execute_action(db, t.order_id, p.action, p.parameters)
        before = deepcopy(order.data)
        order.data = after["orders"][t.order_id]
        order.revision += 1
        user.data = after["users"][t.customer_id]
        user.revision += 1
        p.status = "approved"
        t.status = "resolved"
        t.revision += 1
        audit(
            s, t.id, "action_executed", {"proposal_id": p.id, "before": before, "after": order.data}
        )
        t.messages = [
            *t.messages,
            {
                "role": "assistant",
                "content": "已收到人工确认并执行方案。订单当前状态：" + order.data["status"],
            },
        ]
        return {"status": "approved", "ticket_id": t.id, "order_status": order.data["status"]}


def check_turn(s, t, turn_id, lease):
    from .db import now
    from .support_flow import ROUTED

    turn = s.get(ServiceTurn, turn_id, with_for_update=True, populate_existing=True)
    if not turn or turn.status != "running" or turn.lease != lease or t.active_turn != turn_id:
        raise ValueError("Agent 执行租约已失效")
    if turn.published == ROUTED and (not turn.lease_until or turn.lease_until <= now()):
        raise ValueError("Agent 执行租约已过期")
    return turn
