from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_invariants import isolated  # noqa: F401

from app.api import app
from app.config import model_signature
from app.dataset import raw_database
from app.db import Version, transaction
from app.service import approve, propose, search_knowledge
from app.service_api import Message, message
from app.service_models import Knowledge, Proposal, ServiceRecord, Ticket


@pytest.fixture
def ticket(isolated):  # noqa: F811
    db = raw_database()
    order = next(o for o in db["orders"].values() if o["status"] == "pending")
    user = db["users"][order["user_id"]]
    with transaction() as s:
        baseline = s.get(Version, "baseline")
        baseline.metadata_json = {"model_config": model_signature()}
        s.add(ServiceRecord(kind="orders", id=order["order_id"], data=deepcopy(order)))
        s.add(ServiceRecord(kind="users", id=user["user_id"], data=deepcopy(user)))
        t = Ticket(
            title="Cancellation",
            order_id=order["order_id"],
            customer_id=user["user_id"],
            version_id="baseline",
        )
        s.add(t)
        s.flush()
        tid = t.id
    return tid, order


def test_proposal_does_not_write_until_approval_and_approval_is_idempotent(ticket):
    tid, original = ticket
    p = propose(tid, "cancel", {"reason": "ordered by mistake"}, "取消订单")
    with transaction() as s:
        assert s.get(ServiceRecord, ("orders", original["order_id"])).data == original
        revision = s.get(Ticket, tid).revision
    result = approve(p["proposal_id"], True, revision)
    assert result["order_status"] == "cancelled"
    approve(p["proposal_id"], True, revision)
    with transaction() as s:
        order = s.get(ServiceRecord, ("orders", original["order_id"]))
        assert order.revision == 1
        assert len(order.data["payment_history"]) == 2 * len(original["payment_history"])
    assert raw_database()["orders"][original["order_id"]] == original


def test_second_ticket_cannot_apply_stale_order_proposal(ticket):
    tid, original = ticket
    with transaction() as s:
        t = Ticket(
            title="Second",
            order_id=original["order_id"],
            customer_id=original["user_id"],
            version_id="baseline",
        )
        s.add(t)
        s.flush()
        other = t.id
    a = propose(tid, "cancel", {"reason": "no longer needed"}, "取消")
    b = propose(other, "cancel", {"reason": "no longer needed"}, "取消")
    approve(a["proposal_id"], True, 1)
    with pytest.raises(ValueError, match="其他工单"):
        approve(b["proposal_id"], True, 1)


def test_rejection_keeps_order_and_handoff_blocks_agent(ticket):
    tid, original = ticket
    p = propose(tid, "cancel", {"reason": "no longer needed"}, "取消")
    approve(p["proposal_id"], False, 1)
    with transaction() as s:
        assert s.get(ServiceRecord, ("orders", original["order_id"])).data == original
    client = TestClient(app)
    response = client.patch(
        "/api/service/tickets/" + tid, json={"revision": 2, "status": "human", "note": "人工处理"}
    )
    assert response.status_code == 200
    assert (
        client.post(
            "/api/service/tickets/" + tid + "/messages", json={"message": "继续"}
        ).status_code
        == 409
    )


def test_turn_submission_is_idempotent_and_blocks_overlap(ticket):
    tid, _ = ticket
    first = message(tid, Message(message="查询"), idempotency_key="same")
    assert message(tid, Message(message="查询"), idempotency_key="same") == first
    client = TestClient(app)
    assert (
        client.post(
            "/api/service/tickets/" + tid + "/messages", json={"message": "另一个"}
        ).status_code
        == 409
    )


def test_knowledge_search_excludes_drafts_and_archives(ticket, monkeypatch):
    from app import service_retrieval

    monkeypatch.setattr(service_retrieval, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(service_retrieval, "rerank", lambda q, candidates: candidates)
    with transaction() as s:
        for status in ["draft", "published", "archived"]:
            s.add(Knowledge(title="退货", content="退货说明", tags="return", status=status))
    assert len(search_knowledge("退货")) == 1


def test_invalid_policy_or_stale_lease_cannot_create_proposal(ticket):
    tid, _ = ticket
    with pytest.raises(ValueError):
        propose(tid, "return", {"item_ids": ["missing"], "payment_method_id": "bad"}, "退货")
    with pytest.raises(ValueError, match="租约"):
        propose(
            tid, "cancel", {"reason": "no longer needed"}, "取消", turn_id="stale", lease="stale"
        )
    with transaction() as s:
        assert not s.scalar(select(Proposal).where(Proposal.ticket_id == tid))


def test_address_change_requires_approval_and_complete_fields(ticket):
    tid, original = ticket
    address = dict(original["address"], address1="123 Demo Avenue")
    p = propose(tid, "address", address, "更改完整地址")
    with transaction() as s:
        assert s.get(ServiceRecord, ("orders", original["order_id"])).data == original
    approve(p["proposal_id"], True, 1)
    with transaction() as s:
        assert (
            s.get(ServiceRecord, ("orders", original["order_id"])).data["address"]["address1"]
            == "123 Demo Avenue"
        )


def test_exchange_rejects_noop_variant(ticket):
    from app.service import execute_action

    _, original = ticket
    item = original["items"][0]["item_id"]
    with pytest.raises(ValueError, match="不同规格"):
        execute_action(
            raw_database(),
            original["order_id"],
            "exchange",
            {"item_ids": [item], "new_item_ids": [item], "payment_method_id": "none"},
        )


def test_specialists_cannot_fabricate_citations_or_string_approval():
    from app.service_specialists import validate_report

    with pytest.raises(ValueError, match="不存在"):
        validate_report(
            "policy_specialist", {"answer": "规则", "citations": ["fake"]}, [{"id": "real"}]
        )
    with pytest.raises(ValueError, match="明确结论"):
        validate_report("risk_reviewer", {"approved": "true", "reason": "ok"}, [])


def test_exchange_updates_delivered_order_with_available_same_product_variant():
    from app.service import execute_action

    data = raw_database()
    for order in data["orders"].values():
        if order["status"] != "delivered":
            continue
        for item in order["items"]:
            product = data["products"][item["product_id"]]
            variant = next(
                (
                    v
                    for v in product["variants"].values()
                    if v["available"] and v["item_id"] != item["item_id"]
                ),
                None,
            )
            if variant:
                updated = execute_action(
                    data,
                    order["order_id"],
                    "exchange",
                    {
                        "item_ids": [item["item_id"]],
                        "new_item_ids": [variant["item_id"]],
                        "payment_method_id": order["payment_history"][0]["payment_method_id"],
                    },
                )
                assert updated["orders"][order["order_id"]]["status"] == "exchange requested"
                assert data["orders"][order["order_id"]]["status"] == "delivered"
                return
    pytest.fail("No public exchange fixture found")
