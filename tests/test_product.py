from types import SimpleNamespace

import pytest

from app.dataset import raw_database
from app.product import comparison_summary, demo_cases


def test_cases_use_real_orders_and_explicit_authorization():
    data = raw_database()
    cases = demo_cases()
    assert {c["id"] for c in cases} == {"lookup", "cancel", "return"}
    for case in cases:
        order = data["orders"][case["order_id"]]
        assert order["items"] == case["items"]
        assert case["order_id"] in case["message"]
        if case["id"] in {"cancel", "return"}:
            assert "确认后" in case["message"]
    assert next(c for c in cases if c["id"] == "return")["status"] == "delivered"


def examples():
    entries = [
        dict(job_id=m, mode=m, task_id="t", version_id="v")
        for m in ("baseline", "harness", "skills")
    ]
    rows = [
        SimpleNamespace(
            job_id=m,
            task_id="t",
            trial=0,
            version_id="v",
            reward=0,
            data={"termination_reason": "user_stop"},
        )
        for m in ("baseline", "harness", "skills")
    ]
    return entries, rows


def test_dashboard_cannot_cherry_pick_successful_repeat():
    entries, rows = examples()
    rows += [
        SimpleNamespace(job_id="baseline", task_id="t", trial=1, version_id="v", reward=1, data={})
    ]
    out = comparison_summary(entries, {}, rows)
    assert out["complete"]
    assert out["arms"]["baseline"]["success_rate"] == 0
    assert out["arms"]["baseline"]["completed"] == 1


def test_partial_or_infrastructure_result_cannot_be_leaderboard():
    entries, rows = examples()
    rows[0].data = {"termination_reason": "infrastructure_error"}
    out = comparison_summary(entries, {}, rows)
    assert not out["complete"]
    assert out["arms"]["baseline"]["invalid"] == 1
    assert all(b["success_rate"] is None for b in out["arms"].values())


def test_dashboard_rejects_cross_version_results():
    entries, rows = examples()
    rows[0].version_id = "other"
    with pytest.raises(ValueError, match="identity"):
        comparison_summary(entries, {}, rows)
