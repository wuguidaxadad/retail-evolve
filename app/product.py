"""Read models for the retail demo. Never train on scorecard or mutate experiments."""

import json
from collections import Counter

from sqlalchemy import select

from .config import ROOT, settings
from .dataset import raw_database, splits
from .db import Job, TaskRun, Version, transaction
from .evaluation_validity import require_scoreable


def demo_cases():
    data = raw_database()
    cases = []
    for kind, status, title, goal in [
        ("lookup", "pending", "发货前 · 查询订单", "核验身份，查询订单状态和商品"),
        ("cancel", "pending", "计划变化 · 取消订单", "确认取消原因和用户授权，再取消未发货订单"),
        (
            "return",
            "delivered",
            "收货后 · 申请退货",
            "识别商品、确认退款方式和用户授权，再提交退货",
        ),
    ]:
        order = next(o for o in data["orders"].values() if o["status"] == status and o["items"])
        user = data["users"][order["user_id"]]
        name = user["name"]["first_name"] + " " + user["name"]["last_name"]
        identity = f"我是 {name}，邮编是 {user['address']['zip']}。"
        request = {
            "lookup": "请帮我查一下这个订单的状态和商品。",
            "cancel": "我想取消这个订单，原因是买错了。请先告诉我能否取消，等我确认后再操作。",
            "return": f"我想退回订单里的 {order['items'][0]['name']}。请先确认可退商品和退款方式，等我确认后再提交。",
        }[kind]
        cases.append(
            {
                "id": kind,
                "title": title,
                "goal": goal,
                "order_id": order["order_id"],
                "customer": name,
                "status": status,
                "items": order["items"],
                "message": identity + f"订单号是 {order['order_id']}。" + request,
                "checkpoints": ["身份核验", "查询业务数据", "用户确认后执行", "核对订单变化"],
            }
        )
    return cases


def comparison_summary(entries, jobs, rows):
    """Count only trial 0. Failed requests never become business failures."""
    mapping = {e["job_id"]: e for e in entries}
    output = {
        mode: {"planned": 0, "completed": 0, "successes": 0, "invalid": 0, "jobs": Counter()}
        for mode in ("baseline", "harness", "skills")
    }
    for e in entries:
        output[e["mode"]]["planned"] += 1
        output[e["mode"]]["jobs"][jobs.get(e["job_id"], "missing")] += 1
    seen = set()
    for r in rows:
        if r.trial != 0 or r.job_id not in mapping:
            continue
        spec = mapping[r.job_id]
        key = (spec["mode"], r.task_id)
        if key in seen or r.task_id != spec["task_id"] or r.version_id != spec["version_id"]:
            raise ValueError("Comparison trace identity mismatch")
        seen.add(key)
        bucket = output[spec["mode"]]
        try:
            require_scoreable(r.data)
            if r.data.get("termination_reason") in {"timeout", "infrastructure_error"}:
                raise ValueError("Incomplete episode")
        except ValueError:
            bucket["invalid"] += 1
            continue
        bucket["completed"] += 1
        bucket["successes"] += int(r.reward >= 1 - 1e-9)
    complete = bool(entries) and all(
        b["completed"] == b["planned"] and not b["invalid"] for b in output.values()
    )
    for b in output.values():
        b["jobs"] = dict(b["jobs"])
        # Partial cohorts differ; don't present misleading live leaderboards.
        b["success_rate"] = b["successes"] / b["completed"] if complete and b["completed"] else None
    return {"complete": complete, "arms": output, "trials": 1}


def project_status():
    path = ROOT / "docs/aligned-customer-experiment.json"
    plan = json.loads(path.read_text()) if path.exists() else {}
    entries = plan.get("evaluation_jobs", [])
    with transaction() as s:
        jobs = (
            s.scalars(select(Job).where(Job.id.in_([e["job_id"] for e in entries]))).all()
            if entries
            else []
        )
        # Load each immutable observation by primary key; keep bulk lists narrow.
        row_ids = (
            list(
                s.scalars(
                    select(TaskRun.id).where(
                        TaskRun.job_id.in_([e["job_id"] for e in entries]), TaskRun.trial == 0
                    )
                )
            )
            if entries
            else []
        )
        rows = [s.get(TaskRun, rid) for rid in row_ids]
        comparison = comparison_summary(entries, {j.id: j.status for j in jobs}, rows)
        roles = {}
        for label, key in [("rsi", "rsi_job_id"), ("ace", "skill_job_id")]:
            j = s.get(Job, plan[key]) if plan.get(key) else None
            roles[label] = (
                {"status": j.status, "stage": j.stage} if j else {"status": "not_started"}
            )
        versions = plan.get("versions", {})
        ace = s.get(Version, versions.get("skills", ""))
        skills = (ace.skillbook or {}).get("skills", {}) if ace else {}
        skills = list(skills.values()) if isinstance(skills, dict) else skills
        skill_examples = [
            {"issue": x.get("issue"), "insight": x.get("insight"), "section": x.get("section")}
            for x in skills
            if x.get("insight")
        ][:4]
    root = ROOT / ".runtime/full-rsi/66ecd477f6aa4772896c451a16ddf917/repo/workspace"
    f = root / "iterations.jsonl"
    iterations = (
        [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if f.exists()
        else []
    )
    return {
        "name": "Retail Evolve",
        "business": "零售订单与售后服务",
        "models": {
            "customer": settings.agent_model,
            "simulator": settings.user_model,
            "optimizer": settings.optimizer_model,
            "skills": settings.learning_model,
        },
        "splits": {k: len(v) for k, v in splits().items()},
        "comparison": comparison,
        "learning": roles,
        "iterations": iterations,
        "skill_examples": skill_examples,
        "skill_count": len(skills),
        "versions": versions,
        "early_stop": bool(plan.get("user_early_stop")),
        "limits": [
            "公开模拟业务数据",
            "组合实现，非单篇论文严格复现",
            "历史重复会话保留，仅 trial 0 计分",
            "Hy3 渠道切换后补跑，渠道差异需在报告披露",
        ],
    }
