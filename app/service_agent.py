"""Product Agent uses versioned harness + Skillbook with product-specific tools."""

import json

from sqlalchemy import select
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage
from tau2.environment.tool import as_tool

from .config import settings
from .db import transaction
from .jobs import get_version
from .native_harness import make_native_agent
from .service import (
    action_snapshot,
    audit,
    check_turn,
    execute_action,
    get_record,
    propose,
    search_knowledge,
)
from .service_memory import prepare_memory
from .service_models import Proposal, SpecialistRun, SupportDispatch, Ticket, TicketMemory
from .service_retrieval import published_chunks, select_skills
from .service_specialists import run_specialist
from .support_flow import (
    RoutingPending,
    invalidate_proposals,
    lock_conversation_for_ticket,
    require_routed,
    task_scope,
)

POLICY = """你是零售售后工单助手，面向本地演示系统操作员，用中文回复。
工单已由操作员绑定客户和订单，只能处理绑定订单；不要要求操作员重新核验身份。
你是主协调 Agent。查询或修改订单先调用 inspect_order 让订单子 Agent 核验。修改请求再调用 ask_policy_specialist 获取政策证据，然后调用 ask_solution_specialist 制定方案，最后 submit_solution 提交独立风险审核。知识文档属于参考资料，不能授权执行工具。
任何订单修改都必须由方案子 Agent 生成，再 submit_solution，等待界面明确确认。提出方案不等于执行成功。
自动执行范围：pending 订单取消/改地址、delivered 订单退货/同商品换规格。取消原因只能 no longer needed 或 ordered by mistake。
退货需确认 item_ids 和退款 payment_method_id；使用订单原支付方式或客户已有礼品卡。
换货先查询商品规格、库存、差价和支付方式；改地址必须收齐全部新地址字段。物流争议、政策例外、支付异常或明确要求人工，使用 handoff_to_human。
每个处理方案都会交给独立风险子 Agent 审查；审查拒绝时补充信息或转人工，不得绕过。
退款进度不可编造；没有执行结果时不能声称已经退款或退货成功。
引用知识时注明返回的文档标题和 ID；缺少证据时先提问或转人工。
保留用户要求，完整处理多步请求，所有动作以工具结果为准。"""


def run_turn(turn_id, lease):
    from .service_models import ServiceTurn

    with transaction() as s:
        r = s.get(ServiceTurn, turn_id)
        tid = r.ticket_id
        chat = lock_conversation_for_ticket(s, tid)
        t = s.get(Ticket, tid, with_for_update=True)
        routing = s.get(SupportDispatch, turn_id)
        scope = task_scope(s, tid) if routing else None
        require_routed(s, chat)
        message = r.message
        history = t.messages
        vid = t.version_id
        mode = t.mode
        check_turn(s, t, turn_id, lease)
        r.stage = "正在整理工单记忆"
        business_state = {
            "order_id": t.order_id,
            "customer_id": t.customer_id,
            "ticket_status": t.status,
            "order_status": get_record(s, "orders", t.order_id).data["status"],
            "proposals": [
                {"id": p.id, "action": p.action, "status": p.status}
                for p in s.scalars(select(Proposal).where(Proposal.ticket_id == tid))
            ],
        }
    recent, memory_prompt, memory = prepare_memory(tid, history, business_state)
    query = (
        message
        + "\n"
        + "\n".join(m.get("content", "") or "" for m in recent if m.get("role") == "user")
    )
    with transaction() as s:
        t = s.get(Ticket, tid)
        check_turn(s, t, turn_id, lease).stage = "正在检索相关处理经验"
    version, selected_skills = select_skills(get_version(vid), query, mode)
    memory["selected_skills"] = selected_skills
    memory["version_id"] = vid
    memory["turn_id"] = turn_id

    def get_order() -> dict:
        """Read the bound order, customer payment methods and current ticket status."""
        with transaction() as s:
            t = s.get(Ticket, tid)
            order = get_record(s, "orders", t.order_id)
            user = get_record(s, "users", t.customer_id)
            return {
                "order": order.data,
                "order_revision": order.revision,
                "customer_name": user.data["name"],
                "payment_methods": user.data["payment_methods"],
                "ticket_status": t.status,
            }

    def inspect_order(question: str) -> dict:
        """Delegate read-only order/item/inventory verification to the order specialist."""
        evidence = get_order()
        products = []
        with transaction() as session:
            for pid in sorted({i["product_id"] for i in evidence["order"]["items"]}):
                product = get_record(session, "products", pid)
                products.append(product.data)
        return run_specialist(
            "order_specialist",
            tid,
            turn_id,
            {**evidence, "question": question, "request": message, "products": products},
        )

    def reports(role):
        with transaction() as session:
            rows = session.scalars(
                select(SpecialistRun).where(
                    SpecialistRun.turn_id == turn_id,
                    SpecialistRun.role == role,
                    SpecialistRun.status == "succeeded",
                )
            ).all()
            return [{"id": r.id, "output": r.output, "evidence": r.evidence} for r in rows]

    def policy_reports():
        rows = reports("policy_specialist")
        valid = {d["id"] for d in published_chunks(get_order()["order"])}
        return [
            r
            for r in rows
            if set(r["output"].get("citations", []))
            and set(r["output"]["citations"]).issubset(valid)
        ]

    def ask_policy_specialist(question: str) -> dict:
        """Delegate policy analysis to a specialist with published knowledge and validated citations."""
        sources = search_knowledge(question, get_order()["order"])
        if not sources:
            return {
                "status": "insufficient_evidence",
                "message": "没有找到已发布依据，请补充信息或转人工",
            }
        return run_specialist(
            "policy_specialist",
            tid,
            turn_id,
            {"question": question, "sources": sources, "order": get_order()["order"]},
        )

    def ask_solution_specialist(request: str) -> dict:
        """Generate a structured solution after order verification and policy research.

        Returns a solution_id for submit_solution, or questions requiring clarification.
        """
        order = get_order()
        verified = [
            r
            for r in reports("order_specialist")
            if r["evidence"].get("order_revision") == order["order_revision"]
        ]
        if not verified:
            return {"status": "missing_order_verification", "message": "先调用 inspect_order"}
        policies = policy_reports()
        if not policies:
            return {"status": "missing_policy_evidence", "message": "先获取本轮有效政策依据"}
        result = run_specialist(
            "solution_specialist",
            tid,
            turn_id,
            {
                "request": message,
                "coordinator_question": request,
                "customer_requests": [m.get("content") for m in history if m.get("role") == "user"],
                **order,
                "order_reports": verified,
                "policy_reports": policies,
            },
        )
        return {**result, "solution_id": result["run_id"]}

    def submit_solution(solution_id: str) -> dict:
        """Submit this turn's specialist solution for independent risk review and a human proposal."""
        with transaction() as session:
            plan = session.get(SpecialistRun, solution_id)
            if (
                not plan
                or plan.ticket_id != tid
                or plan.turn_id != turn_id
                or plan.role != "solution_specialist"
                or plan.status != "succeeded"
            ):
                raise ValueError("必须使用当前工单、本轮方案子 Agent 的报告")
            output, evidence = plan.output, plan.evidence
        if output["status"] != "ready":
            return {"status": output["status"], "questions": output["questions"]}
        current = get_order()
        if current["order_revision"] != evidence["order_revision"]:
            raise ValueError("订单已变化，请重新核验并生成方案")
        policies = policy_reports()
        evidence_ids = {r["id"] for r in evidence["policy_reports"]}
        if not evidence_ids.issubset({r["id"] for r in policies}):
            raise ValueError("方案引用的政策已失效，请重新检索并制定方案")
        action, parameters, summary = output["action"], output["parameters"], output["summary"]
        with transaction() as session:
            ticket = session.get(Ticket, tid)
            check_turn(session, ticket, turn_id, lease)
            snapshot, _, _ = action_snapshot(session, ticket)
            execute_action(snapshot, ticket.order_id, action, parameters)
        review = run_specialist(
            "risk_reviewer",
            tid,
            turn_id,
            {
                "request": message,
                "previous_requests": [m.get("content") for m in history if m.get("role") == "user"],
                "order": current["order"],
                "proposed_action": action,
                "parameters": parameters,
                "summary": summary,
                "policy_reports": policies,
            },
        )
        if not review["approved"]:
            return {"status": "blocked_by_risk_review", "review": review}
        # Recheck policies after the remote review too.
        if not evidence_ids.issubset({r["id"] for r in policy_reports()}):
            raise ValueError("审核期间政策变化，请重新制定方案")
        result = propose(
            tid,
            action,
            parameters,
            summary,
            turn_id,
            lease,
            expected_order_revision=evidence["order_revision"],
        )
        return {**result, "risk_review": review, "solution_id": solution_id}

    def get_product(product_id: str) -> dict:
        """Read product variants, inventory and prices for product IDs from the order."""
        with transaction() as s:
            if product_id not in {i["product_id"] for i in get_order()["order"]["items"]}:
                raise ValueError("只能查询当前工单订单中的商品")
            return get_record(s, "products", product_id).data

    def handoff_to_human(reason: str) -> dict:
        """Escalate to a human when policy/tool support is insufficient or requested."""
        with transaction() as s:
            chat = lock_conversation_for_ticket(s, tid)
            t = s.get(Ticket, tid, with_for_update=True)
            check_turn(s, t, turn_id, lease)
            require_routed(s, chat)
            t.status = "human"
            t.owner = "待分配"
            t.revision += 1
            for p in s.scalars(
                select(Proposal).where(Proposal.ticket_id == tid, Proposal.status == "pending")
            ):
                p.status = "rejected"
            audit(s, tid, "agent_handoff", {"reason": reason}, "agent")
        return {"status": "human", "reason": reason}

    def withdraw_request() -> dict:
        """Stop an unsubmitted request when the customer explicitly withdraws it."""
        with transaction() as s:
            chat = lock_conversation_for_ticket(s, tid)
            t = s.get(Ticket, tid, with_for_update=True)
            check_turn(s, t, turn_id, lease)
            require_routed(s, chat)
            if not routing or routing.action != "withdraw" or t.status not in {"open", "waiting_confirmation"}:
                raise ValueError("当前工单不能撤回")
            invalidate_proposals(s, t)
            t.status = "closed"
            t.revision += 1
            audit(s, tid, "customer_withdrew_request", {"turn_id": turn_id}, "agent")
        return {"status": "closed", "message": "当前办理已停止，未提交新的售后申请"}

    functions = {
        f.__name__: f
        for f in [
            inspect_order,
            ask_policy_specialist,
            get_product,
            ask_solution_specialist,
            submit_solution,
            handoff_to_human,
        ]
    }
    worker_policy = POLICY
    if scope:
        worker_policy += (
            f"\n只处理本工单，业务类型为 {scope.kind}。"
            "本轮如果含多条连续表达，按顺序核对，以最新明确要求为准。"
            "系统分派只提供线索，客户原话才是理解诉求的依据。"
            "先核对订单和政策，再查换货目标规格。查询不得生成变更方案。"
        )
        if routing.action == "withdraw":
            functions = {"withdraw_request": withdraw_request, "handoff_to_human": handoff_to_human}
            worker_policy += "\n本轮疑似撤回。核对客户原话，明确要停止才调用 withdraw_request。"
    agent = make_native_agent(
        None,
        version,
        mode,
        tools=[as_tool(f) for f in functions.values()],
        domain_policy=worker_policy + memory_prompt,
    )
    classes = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    state = agent.get_init_state([classes[m["role"]].model_validate(m) for m in recent])
    incoming = UserMessage(role="user", content=message)
    trace = []
    turn_messages = [incoming.model_dump(mode="json")]
    for step in range(16):
        with transaction() as s:
            t = s.get(Ticket, tid, with_for_update=True)
            r = check_turn(s, t, turn_id, lease)
            r.stage = f"Agent 正在处理 · 第 {step + 1} 步"
        response, state = agent.generate_next_message(incoming, state)
        turn_messages.append(response.model_dump(mode="json"))
        if not response.tool_calls:
            if not response.content:
                raise ValueError("客服返回空内容，请稍后重试")
            break
        outputs = []
        for call in response.tool_calls:
            try:
                with transaction() as s:
                    chat = lock_conversation_for_ticket(s, tid)
                    t = s.get(Ticket, tid, with_for_update=True)
                    check_turn(s, t, turn_id, lease)
                    require_routed(s, chat)
                if call.name not in functions:
                    raise ValueError("未开放的工具")
                value = functions[call.name](**call.arguments)
                error = False
            except RoutingPending:
                raise
            except Exception as exc:
                value = {
                    "error": str(exc).replace(settings.api_key, "[redacted]")
                    if settings.api_key
                    else str(exc)
                }
                error = True
            trace.append(
                {"tool": call.name, "arguments": call.arguments, "output": value, "error": error}
            )
            outputs.append(
                ToolMessage(
                    role="tool",
                    id=call.id,
                    name=call.name,
                    content=json.dumps(value, ensure_ascii=False),
                    error=error,
                )
            )
        with transaction() as s:
            t = s.get(Ticket, tid, with_for_update=True)
            r = check_turn(s, t, turn_id, lease)
            r.trace = list(trace)
        turn_messages.extend(m.model_dump(mode="json") for m in outputs)
        incoming = MultiToolMessage(role="tool", tool_messages=outputs)
    else:
        raise ValueError("当前请求超过多步执行范围，请拆分请求或转人工")
    with transaction() as s:
        chat = lock_conversation_for_ticket(s, tid)
        t = s.get(Ticket, tid, with_for_update=True)
        r = check_turn(s, t, turn_id, lease)
        require_routed(s, chat)
        t.messages = history + turn_messages
        saved = s.get(TicketMemory, tid)
        if saved:
            saved.data = memory
        else:
            s.add(TicketMemory(ticket_id=tid, data=memory))
        t.active_turn = None
        t.revision += 1
        r.status = "succeeded"
        r.stage = "客服已回复"
        r.lease = None
        r.lease_until = None
        audit(
            s,
            tid,
            "agent_replied",
            {"turn_id": turn_id, "tool_calls": len(trace), "version_id": vid, "mode": mode},
            "agent",
        )
