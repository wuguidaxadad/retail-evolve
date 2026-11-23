from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import pymysql
import pytest
from ace import Skillbook
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app import db, queue
from app.api import app
from app.config import model_signature, settings
from app.dataset import choose_tasks, splits
from app.gate import run_gate
from app.harness import BASE_CODE, validate_code
from app.jobs import source_runs


@pytest.mark.parametrize(
    "malicious",
    [
        BASE_CODE + "\nimport os",
        BASE_CODE.replace("return base_instruction", "return open('/tmp/x')"),
        BASE_CODE.replace("return base_instruction", "return base_instruction.upper()"),
        BASE_CODE.replace("MAX_STEPS = 60", "MAX_STEPS = 600"),
        BASE_CODE.replace("REQUIRE_CONFIRMATION = False", "REQUIRE_CONFIRMATION = 1"),
        BASE_CODE.replace(
            'return base_instruction + "\\n\\n" + skill_context', 'return "discard skills"'
        ),
    ],
)
def test_generated_code_is_bounded(malicious):
    with pytest.raises((ValueError, SyntaxError)):
        validate_code(malicious)


def test_frozen_partitions_and_holdout():
    parts = splits()
    assert list(map(len, parts.values())) == [74, 20, 20]
    assert len(set.union(*(set(ids) for ids in parts.values()))) == 114
    with pytest.raises(ValueError):
        choose_tasks("learn", [parts["final"][0]])


def test_gate_rejects_regression_even_when_validation_improves(tmp_path):
    report = run_gate(tmp_path, {"0": 1}, {"0": 0}, {"1": 0}, {"1": 1})
    assert not report["passed"]
    assert "FAILED" in report["log"]


def test_gate_accepts_tie_and_missing_result_fails(tmp_path):
    assert run_gate(tmp_path / "tie", {"0": 1}, {"0": 1}, {"1": 1}, {"1": 1})["passed"]
    assert not run_gate(tmp_path / "missing", {"0": 1}, {}, {"1": 1}, {"1": 1})["passed"]


def test_gate_rejects_validation_decline(tmp_path):
    assert not run_gate(tmp_path, {"0": 0}, {"0": 1}, {"1": 1}, {"1": 0})["passed"]


def test_simulated_customer_receives_the_public_business_scenario():
    from app.simulation import get_tasks, simulation_user

    task = next(t for t in get_tasks(None) if str(t.id) == "0")
    user = simulation_user(task)
    assert "Yusuf" in user.system_prompt
    assert "Rossi" in user.system_prompt
    assert "19122" in user.system_prompt
    assert str(task.user_scenario) in user.system_prompt


@pytest.fixture
def isolated(monkeypatch):
    name = "retail_test_" + uuid.uuid4().hex[:10]
    admin = pymysql.connect(host="127.0.0.1", port=3307, user="root", autocommit=True)
    with admin.cursor() as c:
        c.execute("CREATE DATABASE " + name + " CHARACTER SET utf8mb4")
    engine = create_engine("mysql+pymysql://root@127.0.0.1:3307/" + name)
    db.Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "Session", sessionmaker(engine, expire_on_commit=False))
    values = {
        **settings.__dict__,
        "stream": "retail:test:" + name,
        "group": "test-workers",
        "dead_stream": "retail:test:dead:" + name,
    }
    monkeypatch.setattr(queue, "settings", SimpleNamespace(**values))
    client = queue.broker()
    queue.ensure_group(client)
    with db.transaction() as s:
        s.add(
            db.Version(
                id="baseline",
                name="test",
                origin="baseline",
                status="validated",
                metadata_json={"gate": {"passed": True, "incumbent_version_id": "baseline"}},
                harness_code=BASE_CODE,
                skillbook=Skillbook().to_dict(),
            )
        )
        s.flush()
        s.add(db.Deployment(id=1, version_id="baseline", revision=0))
    yield client
    client.delete(values["stream"], values["dead_stream"])
    engine.dispose()
    with admin.cursor() as c:
        c.execute("DROP DATABASE " + name)
    admin.close()


def test_atomic_outbox_idempotency_and_duplicate_delivery(isolated):
    with pytest.raises(RuntimeError):
        with db.transaction() as s:
            queue.submit_in_transaction(s, "evaluate", {}, "rolled-back")
            raise RuntimeError("simulate transaction failure")
    with db.transaction() as s:
        assert s.scalar(select(func.count()).select_from(db.Job)) == 0
        assert s.scalar(select(func.count()).select_from(db.Outbox)) == 0
        job, created = queue.submit_in_transaction(s, "evaluate", {}, "same")
        job_id = job.id
        assert created
    with db.transaction() as s:
        duplicate, created = queue.submit_in_transaction(s, "evaluate", {}, "same")
        assert not created and duplicate.id == job_id
        with pytest.raises(ValueError):
            queue.submit_in_transaction(s, "evaluate", {"different": True}, "same")
    assert queue.flush_outbox(isolated) == 1
    assert queue.flush_outbox(isolated) == 0
    # Simulate duplicate XADD after publish succeeded but DB acknowledgement was lost.
    isolated.xadd(queue.settings.stream, {"job_id": job_id, "attempt": "0"})
    claims = list(ThreadPoolExecutor(2).map(lambda _: queue.claim(job_id), range(2)))
    assert sum(c is not None for c in claims) == 1
    token = next(c[1] for c in claims if c)
    assert queue.finish(job_id, token, {"ok": True})
    assert queue.claim(job_id) is None
    assert not queue.finish(job_id, token, {"overwrite": True})


def test_expired_lease_fences_old_worker_and_retries(isolated):
    with db.transaction() as s:
        job, _ = queue.submit_in_transaction(s, "evaluate", {}, "recovery")
        job_id = job.id
    _, old = queue.claim(job_id)
    with db.transaction() as s:
        s.get(db.Job, job_id).lease_until = db.now() - timedelta(seconds=1)
    _, new = queue.claim(job_id)
    assert old != new
    assert not queue.finish(job_id, old, {"wrong": True})
    assert queue.fail(job_id, new, "transient", True) == "retry"
    with db.transaction() as s:
        row = s.get(db.Job, job_id)
        assert row.status == "queued"
        row.available_at = db.now()
        assert s.scalar(select(func.count()).select_from(db.Outbox)) == 2
    _, last = queue.claim(job_id)
    assert queue.fail(job_id, last, "exhausted", True) == "dead"


def test_no_holdout_learning_and_release_compare_and_swap(isolated):
    with db.transaction() as s:
        s.add(
            db.Job(
                id="holdout",
                kind="evaluate",
                status="succeeded",
                idempotency_key="holdout",
                payload={"split": "final", "version_id": "baseline"},
            )
        )
        s.add(
            db.Version(
                id="untested",
                parent_id="baseline",
                name="candidate",
                origin="ace",
                harness_code=BASE_CODE,
                skillbook=Skillbook().to_dict(),
            )
        )
        s.add(
            db.Version(
                id="validated",
                parent_id="baseline",
                name="validated",
                origin="ace",
                status="validated",
                metadata_json={
                    "gate": {
                        "passed": True,
                        "incumbent_version_id": "baseline",
                        "model_config": model_signature(),
                    }
                },
                harness_code=BASE_CODE,
                skillbook=Skillbook().to_dict(),
            )
        )
    with pytest.raises(ValueError):
        source_runs("holdout", "baseline")
    client = TestClient(app)  # No lifespan: use the explicit isolated test schema.
    assert (
        client.post(
            "/api/releases", json={"version_id": "untested", "expected_revision": 0}
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/releases", json={"version_id": "validated", "expected_revision": 0}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/releases", json={"version_id": "baseline", "expected_revision": 0}
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/rollback", json={"version_id": "baseline", "expected_revision": 1}
        ).status_code
        == 200
    )


def test_conversation_serialization_and_cancel(isolated):
    client = TestClient(app)
    conversation = client.post("/api/conversations").json()["id"]
    route = "/api/conversations/" + conversation + "/messages"
    headers = {"Idempotency-Key": "same-message"}
    first = client.post(route, json={"message": "query order"}, headers=headers)
    duplicate = client.post(route, json={"message": "query order"}, headers=headers)
    assert first.status_code == duplicate.status_code == 202
    assert first.json()["id"] == duplicate.json()["id"]
    assert client.post(route, json={"message": "second"}).status_code == 409
    job_id = first.json()["id"]
    _, token = queue.claim(job_id)
    assert client.post("/api/jobs/" + job_id + "/cancel").status_code == 200
    assert not queue.finish(job_id, token, {"stale": True})
    assert client.post(route, json={"message": "new after cancellation"}).status_code == 202


def test_idempotent_replay_survives_a_model_configuration_change(isolated, monkeypatch):
    from app import api as routes

    client = TestClient(app)
    body = {"kind": "evaluate", "version_id": "baseline"}
    headers = {"Idempotency-Key": "model-switch-replay"}
    first = client.post("/api/jobs", json=body, headers=headers)
    assert first.status_code == 202
    monkeypatch.setattr(routes, "model_signature", lambda: {"agent": "another-model"})
    replay = client.post("/api/jobs", json=body, headers=headers)
    assert replay.status_code == 202 and first.json()["id"] == replay.json()["id"]


def test_worker_refuses_to_mix_models_inside_a_queued_experiment():
    from app.jobs import dispatch

    with pytest.raises(ValueError, match="模型配置已变化"):
        dispatch(
            {"payload": {"model_config": {"agent": "another-model"}}, "kind": "evaluate"}, "token"
        )


def test_database_reconciliation_replays_a_lost_queue_message(isolated):
    with db.transaction() as s:
        job, _ = queue.submit_in_transaction(s, "evaluate", {}, "lost-message")
        job_id = job.id
    assert queue.flush_outbox(isolated) == 1
    messages = isolated.xrange(queue.settings.stream)
    isolated.xdel(queue.settings.stream, messages[0][0])
    with db.transaction() as s:
        s.get(db.Job, job_id).updated_at = db.now() - timedelta(seconds=301)
    assert queue.reconcile_delivery() == 1
    assert queue.flush_outbox(isolated) == 1
    assert isolated.xrange(queue.settings.stream)[0][1]["job_id"] == job_id
    _, token = queue.claim(job_id)
    assert queue.finish(job_id, token, {"recovered": True})
    assert queue.reconcile_delivery() == 0
