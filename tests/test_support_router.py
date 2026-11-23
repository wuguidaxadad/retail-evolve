import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from tau2.data_model.message import AssistantMessage, ToolCall
from test_invariants import isolated  # noqa: F401

from app import db, service_agent, service_worker, support_flow
from app.api import app
from app.config import model_signature
from app.dataset import raw_database
from app.service import check_turn, propose
from app.service_models import (
    Proposal,
    ServiceRecord,
    ServiceTurn,
    SupportDispatch,
    SupportMessage,
    SupportTask,
    Ticket,
)
from app.support_flow import ROUTED, Routing, RoutingPending
from app.support_runner import ready_inputs, ready_turns, recover_expired


@pytest.fixture
def support_data(tmp_path, monkeypatch, request):
    engine = None
    if os.getenv("SUPPORT_REAL_MYSQL") == "1":
        request.getfixturevalue("isolated")
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'support.db'}")
        db.Base.metadata.create_all(engine)
        monkeypatch.setattr(db, "Session", sessionmaker(engine, expire_on_commit=False))
    records = raw_database()
    delivered_by_customer = {
        row["user_id"]: row for row in records["orders"].values() if row["status"] == "delivered"
    }
    pending = next(
        row
        for row in records["orders"].values()
        if row["status"] == "pending" and row["user_id"] in delivered_by_customer
    )
    customer = pending["user_id"]
    delivered = delivered_by_customer[customer]
    with db.transaction() as session:
        version = session.get(db.Version, "baseline")
        if version is None:
            version = db.Version(
                id="baseline",
                name="test",
                origin="baseline",
                status="validated",
                metadata_json={"model_config": model_signature()},
                harness_code="",
                skillbook={},
            )
            session.add(version)
            session.flush()
            session.add(db.Deployment(id=1, version_id="baseline", revision=0))
        version.metadata_json = {"model_config": model_signature()}
        session.add(
            ServiceRecord(kind="users", id=customer, data=deepcopy(records["users"][customer]))
        )
        for order in (delivered, pending):
            session.add(ServiceRecord(kind="orders", id=order["order_id"], data=deepcopy(order)))
    yield customer, delivered["order_id"], pending["order_id"]
    if engine is not None:
        engine.dispose()


def start(client, customer):
    response = client.post("/api/support/conversations", json={"customer_id": customer})
    assert response.status_code == 201, response.json()
    chat = response.json()
    return f"/api/support/conversations/{chat['id']}", {"Support-Token": chat["token"]}


def send(client, path, headers, message, key):
    response = client.post(
        path + "/messages", headers=headers, json={"message": message, "key": key}
    )
    assert response.status_code == 202, response.json()
    return response.json()["id"]


def route(monkeypatch, mid, *items):
    monkeypatch.setattr(
        support_flow,
        "interpret",
        lambda evidence: Routing.model_validate({"dispatches": list(items), "reply": ""}),
    )
    support_flow.process_message(mid)


def target(kind, quote, order_id=None, ticket_id=None, action="start"):
    return {
        "kind": kind,
        "action": action,
        "source_quote": quote,
        "order_id": order_id,
        "ticket_id": ticket_id,
    }


def task(kind):
    with db.transaction() as session:
        key = session.scalar(select(SupportTask.ticket_id).where(SupportTask.kind == kind))
        return session.get(Ticket, key)


def test_customer_revisions_are_routed_in_order_and_block_stale_execution(
    support_data, monkeypatch
):
    customer, delivered, pending = support_data
    client = TestClient(app)
    path, headers = start(client, customer)
    assert client.get(path, headers={"Support-Token": "wrong"}).status_code == 404
    first = send(client, path, headers, "鞋子换42；看看外套发货没", "first")
    assert send(client, path, headers, "鞋子换42；看看外套发货没", "first") == first
    route(
        monkeypatch,
        first,
        target("exchange", "鞋子换42", delivered),
        target("query", "看看外套发货没", pending),
    )
    shoe, query = task("exchange"), task("query")
    assert (shoe.id, shoe.active_turn) in ready_turns()
    assert (query.id, query.active_turn) in ready_turns()
    old_turn = shoe.active_turn
    observed = {}

    def old_worker(turn_id, lease):
        next_msg = send(client, path, headers, "鞋子改43", "second")
        final_msg = send(client, path, headers, "还是42", "third")
        assert support_flow.claim_message(final_msg) is None
        assert [mid for _, mid in ready_inputs()] == [next_msg]
        with pytest.raises(RoutingPending):
            propose(shoe.id, "exchange", {}, "过期方案", turn_id=turn_id, lease=lease)
        route(
            monkeypatch,
            next_msg,
            target("exchange", "鞋子改43", ticket_id=shoe.id, action="revise"),
        )
        route(
            monkeypatch, final_msg, target("exchange", "还是42", ticket_id=shoe.id, action="revise")
        )
        observed["old_lease"] = lease
        with db.transaction() as session:
            with pytest.raises(ValueError, match="租约"):
                check_turn(session, session.get(Ticket, shoe.id), turn_id, lease)

    monkeypatch.setattr(service_worker, "run_turn", old_worker)
    service_worker.process_turn(old_turn, routed=True)
    assert observed
    with db.transaction() as session:
        active = session.get(Ticket, shoe.id).active_turn
        assert session.get(ServiceTurn, old_turn).status == "superseded"
        assert session.get(ServiceTurn, active).message == "鞋子换42\n鞋子改43\n还是42"
        assert session.get(Ticket, query.id).active_turn == query.active_turn
        assert session.scalar(select(Proposal)) is None


def test_pending_and_superseded_proposal_never_executes(support_data, monkeypatch):
    customer, _, pending = support_data
    client = TestClient(app)
    path, headers = start(client, customer)
    first = send(client, path, headers, "取消订单，买错了", "first")
    route(monkeypatch, first, target("cancel", "取消订单，买错了", pending))
    ticket = task("cancel")
    assert (
        client.post(
            f"/api/service/tickets/{ticket.id}/proposals",
            json={
                "action": "cancel",
                "parameters": {"reason": "ordered by mistake"},
                "summary": "取消",
            },
        ).status_code
        == 409
    )
    proposal = propose(ticket.id, "cancel", {"reason": "ordered by mistake"}, "取消订单")
    with db.transaction() as session:
        turn = session.get(ServiceTurn, ticket.active_turn)
        turn.status = "succeeded"
        session.get(Ticket, ticket.id).active_turn = None
    revision = task("cancel").revision
    follow = send(client, path, headers, "先不取消了", "second")
    endpoint = path + "/proposals/" + proposal["proposal_id"] + "/decision"
    body = {"approved": True, "revision": revision}
    assert (
        client.post(
            "/api/service/proposals/" + proposal["proposal_id"] + "/decision", json=body
        ).status_code
        == 404
    )
    other_path, other_headers = start(client, customer)
    assert (
        client.post(
            other_path + "/proposals/" + proposal["proposal_id"] + "/decision",
            headers=other_headers,
            json=body,
        ).status_code
        == 404
    )
    assert client.post(endpoint, headers=headers, json=body).status_code == 409
    route(
        monkeypatch, follow, target("cancel", "先不取消了", ticket_id=ticket.id, action="follow_up")
    )
    assert client.post(endpoint, headers=headers, json=body).json()["status"] == "superseded"
    with db.transaction() as session:
        assert session.get(ServiceRecord, ("orders", pending)).data["status"] == "pending"


def test_watchdog_recovers_only_lost_work_and_quarantines_invalid_route(support_data, monkeypatch):
    customer, _, pending = support_data
    client = TestClient(app)
    path, headers = start(client, customer)
    first = send(client, path, headers, "取消订单", "first")
    route(monkeypatch, first, target("cancel", "取消订单", pending))
    ticket = task("cancel")
    with db.transaction() as session:
        turn = session.get(ServiceTurn, ticket.active_turn)
        turn.status, turn.lease, turn.lease_until = "running", "stale", db.now()
    recover_expired()
    with db.transaction() as session:
        turn = session.get(ServiceTurn, ticket.active_turn)
        assert turn.published == ROUTED and turn.status == "queued" and not turn.lease
        assert session.get(SupportDispatch, turn.id).recoveries == 1
        turn.status = "succeeded"
        current = session.get(Ticket, ticket.id)
        current.active_turn, current.status = None, "waiting_confirmation"
    recover_expired()
    assert task("cancel").status == "waiting_confirmation"
    assert not ready_turns()
    second = send(client, path, headers, "取消另一单", "second")
    route(monkeypatch, second, target("cancel", "取消另一单", "not-owned"))
    with db.transaction() as session:
        assert session.get(SupportMessage, second).status == "failed"
        assert len(session.scalars(select(SupportTask)).all()) == 1
    assert client.post(path + f"/messages/{second}/retry", headers=headers).status_code == 202
    assert [mid for _, mid in ready_inputs()] == [second]


def test_unclear_revision_keeps_existing_proposal_blocked_until_clarified(
    support_data, monkeypatch
):
    customer, _, pending = support_data
    client = TestClient(app)
    path, headers = start(client, customer)
    first = send(client, path, headers, "取消订单，买错了", "first")
    route(monkeypatch, first, target("cancel", "取消订单，买错了", pending))
    ticket = task("cancel")
    proposal = propose(ticket.id, "cancel", {"reason": "ordered by mistake"}, "取消订单")
    with db.transaction() as session:
        session.get(ServiceTurn, ticket.active_turn).status = "succeeded"
        session.get(Ticket, ticket.id).active_turn = None
    revision = task("cancel").revision
    unclear = send(client, path, headers, "那个先别办", "unclear")
    route(monkeypatch, unclear)
    endpoint = path + "/proposals/" + proposal["proposal_id"] + "/decision"
    assert (
        client.post(
            endpoint, headers=headers, json={"approved": True, "revision": revision}
        ).status_code
        == 409
    )
    assert client.get(path, headers=headers).json()["tasks"][0]["proposal"] is None
    reply = client.post(
        path + "/messages",
        headers=headers,
        json={"message": "取消订单的那张不办了", "key": "clarify", "clarifies_id": unclear},
    )
    assert reply.status_code == 202
    route(
        monkeypatch,
        reply.json()["id"],
        target("cancel", "取消订单的那张不办了", ticket_id=ticket.id, action="withdraw"),
    )
    with db.transaction() as session:
        assert session.get(SupportMessage, unclear).status == "clarified"
        assert session.get(Proposal, proposal["proposal_id"]).status == "superseded"


def test_explicit_order_mismatch_cannot_dispatch_and_can_be_discarded(support_data, monkeypatch):
    customer, delivered, pending = support_data
    client = TestClient(app)
    path, headers = start(client, customer)
    first = send(client, path, headers, f"查订单 {delivered}", "first")
    second = send(client, path, headers, "查另一个订单", "second")
    route(monkeypatch, first, target("query", "查订单", pending))
    assert not ready_inputs()
    with db.transaction() as session:
        assert session.get(SupportMessage, first).status == "failed"
        assert session.scalar(select(SupportTask)) is None
    assert client.post(path + f"/messages/{first}/discard", headers=headers).status_code == 200
    assert [mid for _, mid in ready_inputs()] == [second]


def test_worker_consumes_customer_withdrawal_and_leaves_order_unchanged(support_data, monkeypatch):
    customer, _, pending = support_data
    client = TestClient(app)
    path, headers = start(client, customer)
    first = send(client, path, headers, "帮我取消订单", "first")
    route(monkeypatch, first, target("cancel", "帮我取消订单", pending))
    ticket = task("cancel")
    second = send(client, path, headers, "等等，先不取消了", "second")
    route(
        monkeypatch,
        second,
        target("cancel", "等等，先不取消了", ticket_id=ticket.id, action="withdraw"),
    )

    class ScriptedWorker:
        def get_init_state(self, history):
            return SimpleNamespace()

        def generate_next_message(self, incoming, state):
            if incoming.role == "user":
                assert incoming.content == "帮我取消订单\n等等，先不取消了"
                return AssistantMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="stop", name="withdraw_request", arguments={}),
                    ],
                ), state
            assert not incoming.tool_messages[0].error
            return AssistantMessage(role="assistant", content="已停止办理，未取消订单。"), state

    monkeypatch.setattr(service_agent, "select_skills", lambda version, query, mode: (version, []))
    monkeypatch.setattr(
        service_agent, "make_native_agent", lambda *args, **kwargs: ScriptedWorker()
    )
    turn = task("cancel").active_turn
    service_worker.process_turn(turn, routed=True)
    result = client.get(path, headers=headers).json()["tasks"][0]
    with db.transaction() as session:
        assert result["status"] == "closed", session.get(ServiceTurn, turn).error
    assert result["replies"] == ["已停止办理，未取消订单。"]
    with db.transaction() as session:
        assert session.get(ServiceRecord, ("orders", pending)).data["status"] == "pending"
