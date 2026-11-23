"""Small real-provider acceptance run. Creates one unexecuted demo proposal."""

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.config import ROOT, model_signature
from app.db import Version, transaction
from app.service_memory import prepare_memory
from app.service_models import ServiceRecord
from app.service_retrieval import search_policies, select_skills


def api(path, body=None):
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/service" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def main():
    plan = json.loads((ROOT / "docs/aligned-customer-experiment.json").read_text())
    with transaction() as s:
        v = s.get(Version, plan["baseline_version_id"])
        assert v.metadata_json["model_config"] == model_signature()
        orders = s.scalars(select(ServiceRecord).where(ServiceRecord.kind == "orders")).all()
        order = next(r for r in orders if r.data["status"] == "pending")
        order_id, original = order.id, order.data
    result = {"checks": {}}
    policies = search_policies("买错了，尚未发货的订单可以取消吗？", original)
    assert policies and any(d["document_id"] == "policy-0" for d in policies)
    result["checks"]["rag"] = [{"id": d["id"], "retrieval": d["retrieval"]} for d in policies]
    print("Real hybrid retrieval + reranking passed", flush=True)
    ticket = api(
        "/tickets",
        {
            "order_id": order_id,
            "title": "增强验收：五 Agent 取消方案",
            "version_id": v.id,
            "mode": "baseline",
        },
    )
    result["ticket_id"] = ticket["id"]
    turn = api(
        "/tickets/" + ticket["id"] + "/messages",
        {
            "message": "客户买错了，请核验这个订单和取消政策。原因是 ordered by mistake，客户希望取消整笔订单并退回原支付方式。请生成待人工确认方案，暂时不要执行。"
        },
    )
    print("Live five-Agent ticket:", ticket["id"], flush=True)
    deadline = time.monotonic() + 480
    last_stage = None
    while time.monotonic() < deadline:
        detail = api("/tickets/" + ticket["id"])
        current = next((t for t in detail["turns"] if t["id"] == turn["id"]), None)
        if current and current["stage"] != last_stage:
            print(current["stage"], flush=True)
            last_stage = current["stage"]
        if current and current["status"] in {"succeeded", "failed"}:
            result["turn"] = current
            result["roles"] = [
                {"role": x["role"], "status": x["status"]} for x in detail["specialists"]
            ]
            result["memory"] = detail["memory"]
            result["proposal_ids"] = [p["id"] for p in detail["proposals"]]
            (ROOT / ".runtime/enhancements-smoke.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            assert current["status"] == "succeeded", current["error"]
            assert all(
                any(r["role"] == role and r["status"] == "succeeded" for r in detail["specialists"])
                for role in [
                    "order_specialist",
                    "policy_specialist",
                    "solution_specialist",
                    "risk_reviewer",
                ]
            )
            assert any(p["status"] == "pending" for p in detail["proposals"])
            assert detail["order"] == original
            break
        time.sleep(3)
    else:
        raise RuntimeError("Live acceptance timed out; inspect ticket before retrying")
    history = []
    for i in range(6):
        history.extend(
            [
                {
                    "role": "user",
                    "content": "只允许换货，禁止直接退款" if i == 0 else f"继续核对第{i}项",
                },
                {"role": "assistant", "content": "已核对订单信息。" * 120},
            ]
        )
    _, prompt, memory = prepare_memory(
        ticket["id"], history, {"order_id": order_id, "status": "pending"}
    )
    assert "禁止直接退款" in prompt
    result["checks"]["memory"] = memory
    from app.jobs import get_version

    ace_id = plan.get("versions", {}).get("skills")
    if ace_id:
        _, skills = select_skills(
            get_version(ace_id), "取消尚未发货的误购订单，退回原支付方式", "skills"
        )
        result["checks"]["skill_retrieval"] = {"version": ace_id, "selected": skills}
    result["passed"] = True
    (ROOT / ".runtime/enhancements-smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print("Live acceptance passed; proposal remains pending human confirmation.", flush=True)


if __name__ == "__main__":
    main()
