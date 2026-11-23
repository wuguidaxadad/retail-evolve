# ruff: noqa: F811
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from ace import Skillbook
from sqlalchemy import select
from test_invariants import isolated  # noqa: F401
from test_service_workflow import ticket  # noqa: F401

from app import service_agent, service_memory, service_retrieval, service_specialists
from app.db import now, transaction
from app.service import approve
from app.service_models import (
    Knowledge,
    KnowledgeScope,
    Proposal,
    ServiceTurn,
    SpecialistRun,
    Ticket,
    TicketMemory,
)


def test_semantic_branch_recalls_without_lexical_overlap(monkeypatch):
    monkeypatch.setattr(
        service_retrieval, "embed_texts", lambda texts: [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    docs = [{"id": "refund", "text": "退回原支付渠道"}, {"id": "shipping", "text": "快递运输"}]
    candidates = service_retrieval.hybrid_candidates("money back", docs)
    assert candidates[0]["id"] == "refund"
    assert candidates[0]["retrieval"]["bm25"] == 0
    assert candidates[0]["retrieval"]["cosine"] == 1


def test_reranker_excludes_irrelevance_and_rejects_fabricated_ids(monkeypatch):
    candidates = [{"id": x, "text": x, "retrieval": {"rrf": 0.03}} for x in ["a", "b"]]
    monkeypatch.setattr(
        service_retrieval,
        "structured_report",
        lambda *args: {"scores": [{"id": "a", "score": 0}, {"id": "b", "score": 3}]},
    )
    assert [x["id"] for x in service_retrieval.rerank("question", candidates)] == ["b"]
    monkeypatch.setattr(
        service_retrieval,
        "structured_report",
        lambda *args: {"scores": [{"id": "fake", "score": 3}]},
    )
    with pytest.raises(ValueError, match="来源"):
        service_retrieval.rerank("question", candidates)


def test_policy_revision_archive_and_scope_invalidate_chunks(ticket):
    _, order = ticket
    with transaction() as s:
        s.add(Knowledge(id="k", title="规则", content="取消规则。", tags="", status="published"))
        s.add(KnowledgeScope(knowledge_id="k", order_states=["pending"], product_ids=[]))
    assert not service_retrieval.published_chunks()
    first = service_retrieval.published_chunks(order)
    assert first[0]["id"] == "k:r0:c0"
    assert not service_retrieval.published_chunks({**order, "status": "delivered"})
    with transaction() as s:
        d = s.get(Knowledge, "k")
        d.revision += 1
        d.content = "新版取消规则。"
    assert service_retrieval.published_chunks(order)[0]["id"] == "k:r1:c0"
    with transaction() as s:
        s.get(Knowledge, "k").status = "archived"
    assert not service_retrieval.published_chunks(order)


def test_policy_archived_during_rerank_is_not_returned(ticket, monkeypatch):
    with transaction() as s:
        s.add(Knowledge(id="k", title="规则", content="取消", tags="", status="published"))
    monkeypatch.setattr(service_retrieval, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])

    def rerank(*args):
        with transaction() as s:
            s.get(Knowledge, "k").status = "archived"
        return args[1]

    monkeypatch.setattr(service_retrieval, "rerank", rerank)
    assert not service_retrieval.search_policies("取消")


def test_memory_keeps_constraints_tool_pairs_and_original_transcript(ticket, monkeypatch):
    tid, _ = ticket
    history = []
    for i in range(6):
        history.extend(
            [
                {"role": "user", "content": "不允许退款" if i == 0 else f"继续{i}"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": str(i)}]},
                {"role": "tool", "content": "large tool result " * 500},
                {"role": "assistant", "content": "已查询"},
            ]
        )
    original = deepcopy(history)
    # Even an incomplete summary cannot erase the raw customer restriction.
    monkeypatch.setattr(
        service_memory,
        "structured_report",
        lambda *a: {
            k: [] for k in ["customer_goals", "constraints", "confirmed_facts", "open_questions"]
        },
    )
    recent, prompt, data = service_memory.prepare_memory(tid, history, {})
    assert recent[0]["role"] == "user"
    assert "不允许退款" in prompt
    assert data["covered_messages"] == 8
    assert data["context_tokens_estimate"] < data["full_history_tokens_estimate"]
    assert history == original
    with transaction() as s:
        s.add(TicketMemory(ticket_id=tid, data=data))
    monkeypatch.setattr(
        service_memory,
        "structured_report",
        lambda *a: pytest.fail("Should reuse unchanged summary"),
    )
    assert (
        service_memory.prepare_memory(tid, history, {})[2]["prefix_sha256"] == data["prefix_sha256"]
    )


def test_summary_rejects_invented_customer_quote(monkeypatch):
    monkeypatch.setattr(
        service_memory,
        "structured_report",
        lambda *a: {"constraints": [{"text": "允许退款", "source": 0, "quote": "允许退款"}]},
    )
    with pytest.raises(ValueError, match="无来源"):
        service_memory.summarize([{"role": "user", "content": "仅查询订单"}])


def test_skill_selection_filters_inactive_and_preserves_version(monkeypatch):
    book = Skillbook()
    a = book.add_skill(section="context", issue="换货", insight="先查库存")
    b = book.add_skill(section="context", issue="退货", insight="确认退款方式")
    c = book.add_skill(section="context", issue="旧经验", insight="不再使用")
    book.remove_skill(c.id)
    version = {"skillbook": book.to_dict(), "id": "v"}
    original = deepcopy(version)

    def candidates(q, docs):
        assert c.id not in {d["id"] for d in docs}
        return [d for d in docs if d["id"] == a.id]

    monkeypatch.setattr(service_retrieval, "hybrid_candidates", candidates)
    monkeypatch.setattr(service_retrieval, "rerank", lambda q, docs, limit: docs)
    selected, evidence = service_retrieval.select_skills(version, "换货", "skills")
    assert [x.id for x in Skillbook.from_dict(selected["skillbook"]).skills()] == [a.id]
    assert b.id not in [x["id"] for x in evidence]
    assert version == original
    assert not service_retrieval.select_skills(version, "换货", "baseline")[1]


def prepare_turn(tid):
    with transaction() as s:
        r = ServiceTurn(
            ticket_id=tid,
            key="pipeline",
            message="买错了，请取消订单",
            status="running",
            lease="token",
            lease_until=now(),
        )
        s.add(r)
        s.flush()
        s.get(Ticket, tid).active_turn = r.id
        s.add(
            Knowledge(
                id="p", title="取消规则", content="pending可取消", tags="", status="published"
            )
        )
        return r.id


@pytest.mark.parametrize(
    "risk_ok,fault",
    [(True, None), (False, None), (True, "policy"), (True, "order"), (True, "foreign_turn")],
)
def test_five_agent_flow_and_approval_boundary(ticket, monkeypatch, risk_ok, fault):
    from tau2.data_model.message import AssistantMessage, ToolCall

    from app.dataset import raw_database
    from app.service_models import ServiceRecord

    tid, original = ticket
    with transaction() as s:
        for pid in {i["product_id"] for i in original["items"]}:
            s.add(ServiceRecord(kind="products", id=pid, data=raw_database()["products"][pid]))
    rid = prepare_turn(tid)
    roles = []

    def report(instruction, evidence, schema):
        if "item_ids" in schema["properties"] and "answer" in schema["properties"]:
            roles.append("order")
            return {"answer": "订单pending", "item_ids": [], "missing_information": []}
        if "citations" in schema["properties"]:
            roles.append("policy")
            return {"answer": "允许取消", "citations": ["p:r0:c0"]}
        if "questions" in schema["properties"]:
            roles.append("solution")
            return {
                "status": "ready",
                "action": "cancel",
                "parameters": {"reason": "ordered by mistake"},
                "summary": "取消误购订单",
                "questions": [],
            }
        roles.append("risk")
        return {"approved": risk_ok, "reason": "核对客户范围"}

    monkeypatch.setattr(service_specialists, "structured_report", report)
    monkeypatch.setattr(
        service_agent, "search_knowledge", lambda *a: service_retrieval.published_chunks(original)
    )
    seen = {}

    class Agent:
        def get_init_state(self, history):
            return SimpleNamespace(messages=list(history))

        def generate_next_message(self, incoming, state):
            n = len([m for m in state.messages if m.role == "assistant"])
            if hasattr(incoming, "tool_messages"):
                state.messages.extend(incoming.tool_messages)
                output = json.loads(incoming.tool_messages[0].content)
                seen[n] = output
            else:
                state.messages.append(incoming)
            if n == 3 and fault:
                with transaction() as session:
                    if fault == "policy":
                        session.get(Knowledge, "p").revision += 1
                    elif fault == "order":
                        session.get(ServiceRecord, ("orders", original["order_id"])).revision += 1
                    else:
                        session.get(SpecialistRun, seen[3]["solution_id"]).turn_id = "another-turn"
            actions = [
                ("inspect_order", {"question": "查订单"}),
                ("ask_policy_specialist", {"question": "取消政策"}),
                ("ask_solution_specialist", {"request": "取消误购订单"}),
                ("submit_solution", {"solution_id": seen.get(3, {}).get("solution_id", "")}),
            ]
            reply = (
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id=str(n), name=actions[n][0], arguments=actions[n][1])],
                )
                if n < 4
                else AssistantMessage(
                    role="assistant", content="请核对方案" if risk_ok else "方案被拒绝"
                )
            )
            state.messages.append(reply)
            return reply, state

    monkeypatch.setattr(service_agent, "make_native_agent", lambda *a, **kw: Agent())
    service_agent.run_turn(rid, "token")
    assert roles == (
        ["order", "policy", "solution"] if fault else ["order", "policy", "solution", "risk"]
    )
    with transaction() as s:
        assert s.get(ServiceRecord, ("orders", original["order_id"])).data == original
        plans = s.scalars(select(Proposal).where(Proposal.ticket_id == tid)).all()
        assert len(plans) == int(risk_ok and not fault)
        assert s.get(TicketMemory, tid)
        assert s.get(Ticket, tid).messages[0]["content"] == "买错了，请取消订单"
        if risk_ok and not fault:
            pid, revision = plans[0].id, s.get(Ticket, tid).revision
    if risk_ok and not fault:
        assert approve(pid, True, revision)["order_status"] == "cancelled"


def test_order_specialist_rejects_foreign_item_ids(ticket, monkeypatch):
    tid, order = ticket
    monkeypatch.setattr(
        service_specialists,
        "structured_report",
        lambda *a: {"answer": "x", "item_ids": ["foreign"], "missing_information": []},
    )
    with pytest.raises(ValueError, match="不属于"):
        service_specialists.run_specialist("order_specialist", tid, "t", {"order": order})
    with transaction() as s:
        assert (
            s.scalar(select(SpecialistRun.status).where(SpecialistRun.ticket_id == tid)) == "failed"
        )


def test_embedding_cache_reuses_content_and_indexes_new_revision(ticket, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs["input"])
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[1.0, float(i + 1)])
                for i, _ in enumerate(kwargs["input"])
            ]
        )

    monkeypatch.setattr(
        service_retrieval,
        "OpenAI",
        lambda **kw: SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )
    a = service_retrieval.embed_texts(["policy v1", "policy v1"])
    assert a[0] == a[1]
    service_retrieval.embed_texts(["policy v1"])
    service_retrieval.embed_texts(["policy v2"])
    assert calls == [["policy v1"], ["policy v2"]]


def test_knowledge_scope_roundtrips_through_api(ticket):
    from fastapi.testclient import TestClient

    from app.api import app

    _, order = ticket
    client = TestClient(app)
    body = {
        "title": "限定规则",
        "content": "只适用此商品",
        "status": "published",
        "order_states": ["pending"],
        "product_ids": [order["items"][0]["product_id"]],
    }
    created = client.post("/api/service/knowledge", json=body)
    assert created.status_code == 201
    kid = created.json()["id"]
    docs = client.get("/api/service/knowledge").json()
    doc = next(x for x in docs if x["id"] == kid)
    assert doc["order_states"] == ["pending"] and doc["product_ids"] == body["product_ids"]
    assert (
        client.put(
            "/api/service/knowledge/" + kid,
            json={**body, "revision": 0, "order_states": ["delivered"]},
        ).status_code
        == 200
    )
    assert not service_retrieval.published_chunks(order)


def test_solution_schema_rejects_extra_or_mismatched_action_parameters(ticket, monkeypatch):
    from jsonschema import ValidationError, validate

    tid, order = ticket
    observed = {}

    def report(instruction, evidence, schema):
        observed["schema"] = schema
        return {
            "status": "needs_information",
            "action": "none",
            "parameters": {},
            "summary": "请确认商品",
            "questions": ["哪件商品？"],
        }

    monkeypatch.setattr(service_specialists, "structured_report", report)
    service_specialists.run_specialist("solution_specialist", tid, "t", {"order": order})
    schema = observed["schema"]
    payload = {
        "status": "ready",
        "action": "cancel",
        "parameters": {"reason": "ordered by mistake"},
        "summary": "取消",
        "questions": [],
    }
    validate(payload, schema)
    with pytest.raises(ValidationError):
        validate(
            {**payload, "parameters": {**payload["parameters"], "order_id": "foreign"}}, schema
        )
    with pytest.raises(ValidationError):
        validate({**payload, "action": "return"}, schema)
    with pytest.raises(ValidationError):
        validate({**payload, "action": "none"}, schema)


def test_structured_report_retries_invalid_format_once(monkeypatch):
    from app import service_llm

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        args = '{"approved":"true"}' if len(calls) == 1 else '{"approved":true}'
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(name="submit_report", arguments=args)
                            )
                        ]
                    )
                )
            ]
        )

    monkeypatch.setattr(
        service_llm,
        "OpenAI",
        lambda **kw: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    result = service_llm.structured_report(
        "审核",
        {},
        {
            "type": "object",
            "properties": {"approved": {"type": "boolean"}},
            "required": ["approved"],
        },
    )
    assert result == {"approved": True} and len(calls) == 2


def test_full_history_survives_agent_state_compaction(ticket, monkeypatch):
    from tau2.data_model.message import AssistantMessage

    tid, _ = ticket
    history = [{"role": "user", "content": "禁止退款"}, {"role": "assistant", "content": "已记录"}]
    with transaction() as s:
        s.get(Ticket, tid).messages = history
    rid = prepare_turn(tid)

    class Agent:
        def get_init_state(self, history):
            return SimpleNamespace(messages=history)

        def generate_next_message(self, incoming, state):
            state.messages = []  # An evolved implementation may compact its own state.
            return AssistantMessage(role="assistant", content="等待人工确认"), state

    monkeypatch.setattr(service_agent, "make_native_agent", lambda *a, **kw: Agent())
    service_agent.run_turn(rid, "token")
    with transaction() as s:
        saved = s.get(Ticket, tid).messages
        assert saved[:2] == history
        assert saved[2]["role"] == "user" and saved[3]["content"] == "等待人工确认"
