"""Resumeable, frozen three-arm experiment through the real MySQL/Redis queue."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from app.config import ROOT, model_signature
from app.dataset import splits
from app.db import Deployment, Job, Session, TaskRun, Version
from app.experiment_metrics import MODES, summarize_runs

MANIFEST = ROOT / "docs/customer-ablation.json"
REPORT = ROOT / "docs/CUSTOMER_ABLATION.md"
LOCK = ROOT / ".runtime/customer-ablation.lock"
STOP = False


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(version):
    value = json.dumps(
        {"harness_code": version.harness_code, "skillbook": version.skillbook},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(value.encode()).hexdigest()


def save(plan):
    plan["updated_at"] = timestamp()
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(MANIFEST)
    lines = [
        "# 零售客服：RSI 与 Skill 三组对照",
        "",
        f"状态：{plan['status']}；已完成 {plan.get('completed_runs', 0)}/{plan['planned_runs']} 次会话。",
        "",
        "固定 a6 GPT-4o-mini 和同一候选；基础版、仅 RSI 策略、仅 ACE Skill。",
        f"20 道最终保留题，每题每组 {plan['trials']} 次；本轮种子 300。测试轨迹不参与学习或优化。",
        "",
        f"候选版本：{plan['version_id']}；当前候选由 1 道学习题提炼的 2 条经验及一次策略生成构成。",
        "",
        "RSI 在本项目指受限执行代码和提示的改进，不涉及模型权重训练。",
        "",
        "这是关闭组件的消融实验；策略是在已有经验版本上生成，不能解读成两条独立训练路线。",
        "",
    ]
    if plan.get("metrics"):
        lines += [
            "| 分组 | 成功率 | 比基础版变化 | 95% 区间（百分点） | 平均耗时 | 客服及模拟客户费用估算 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for mode, name in [("baseline", "基础版"), ("harness", "仅 RSI"), ("skills", "仅 Skill")]:
            m = plan["metrics"][mode]
            delta = m.get("success_delta_percentage_points")
            interval = m.get("delta_95ci_percentage_points")
            lines.append(
                f"| {name} | {m['successes']}/{m['runs']} = {m['success_rate']:.1%} | "
                + (f"{delta:+.1f} 个百分点" if delta is not None else "—")
                + " | "
                + (f"[{interval[0]:+.1f}, {interval[1]:+.1f}]" if interval else "—")
                + f" | {m['mean_seconds']:.1f} 秒 | ${m['estimated_agent_user_usd']:.4f} |"
            )
        lines += [
            "",
            "成功指官方 reward 完整达到 1；分母包括全部任务及重复次数。",
            "95% 区间用配对任务级 bootstrap，保留同题重复执行的相关性，10000 次重采样。",
            "区间包含 0 时，这一规模的测试不足以确认提升；下降按原样报告。",
            "本轮每题一次，不估算多次执行稳定性，也不与上游 pass^4 直接比较。",
            "",
            "费用按运行前核对的主要渠道价估算：输入 $0.0576、缓存 $0.0288、输出 $0.2304／百万 token。",
            "费用包含客服与模拟客户的已记录 token，不包含未记录 token 的自然语言评判，且不是 a6 最终账单。",
        ]
    if plan.get("errors"):
        lines += [
            "",
            "存在未完成或执行失败的任务，不报告完整提升率。",
            json.dumps(plan["errors"], ensure_ascii=False),
        ]
    if plan.get("repeat_reduction"):
        lines += [
            "",
            "用户在运行中将重复次数从 3 改为 1：保留已完成的 trial=0，取消原多次任务，未完成的任务重新按一次执行。",
            "已发生的额外重复保留在审计文件，但不参与此次统计；没有按结果好坏选择轨迹。",
        ]
    lines += [
        "",
        "完整计划、版本指纹、任务、任务 ID 和逐次结果见 customer-ablation.json；原始对话在网页轨迹及 MySQL TaskRun。",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def observations(job_specs, trials):
    mapping = {j["job_id"]: j for j in job_specs}
    if not mapping:
        return [], []
    output = []
    with Session() as session:
        states = [
            {"job_id": j.id, "status": j.status, "stage": j.stage, "error": j.error}
            for j in session.scalars(select(Job).where(Job.id.in_(mapping)))
        ]
        for row in session.scalars(select(TaskRun).where(TaskRun.job_id.in_(mapping))):
            if row.trial >= trials:
                continue
            spec = mapping[row.job_id]
            data = row.data
            if data.get("mode") != spec["mode"] or data.get("adapter_version") != 3:
                raise ValueError("Mixed runtime or mode in experiment")
            expected_skills = bool(data.get("injected_skill_ids"))
            if expected_skills != (spec["mode"] == "skills"):
                raise ValueError("Skill component isolation failed")
            inp = out = cached = 0
            for message in data.get("messages", []):
                usage = message.get("usage") or {}
                inp += usage.get("prompt_tokens", 0) or 0
                out += usage.get("completion_tokens", 0) or 0
                raw = (message.get("raw_data") or {}).get("usage") or {}
                cached += (raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            output.append(
                {
                    "run_id": row.id,
                    "job_id": row.job_id,
                    "task_id": row.task_id,
                    "trial": row.trial,
                    "mode": spec["mode"],
                    "reward": row.reward,
                    "seconds": data["elapsed_seconds"],
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cached_input_tokens": cached,
                    "estimated_agent_user_usd": (
                        (inp - cached) * 0.0576 + cached * 0.0288 + out * 0.2304
                    )
                    / 1e6,
                }
            )
    covered = {row["job_id"] for row in output}
    for state in states:
        state["reused_complete_observation"] = (
            mapping[state["job_id"]].get("reused_first_trial", False) and state["job_id"] in covered
        )
    return output, states


def main():
    global STOP
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: globals().__setitem__("STOP", True))
    LOCK.parent.mkdir(exist_ok=True)
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=30)
        if MANIFEST.exists():
            plan = json.loads(MANIFEST.read_text())
            if plan["status"] == "completed":
                print("Experiment already completed", flush=True)
                return
        else:
            with Session() as s:
                version = s.get(Version, s.get(Deployment, 1).version_id)
                task_ids = splits()["final"]
                if s.scalar(select(TaskRun.id).where(TaskRun.task_id.in_(task_ids)).limit(1)):
                    raise ValueError("Final holdout already used; do not silently call it unseen")
                plan = {
                    "experiment_id": uuid.uuid4().hex,
                    "created_at": timestamp(),
                    "status": "preparing",
                    "version_id": version.id,
                    "version_fingerprint": fingerprint(version),
                    "frozen_code": version.harness_code,
                    "frozen_skillbook": version.skillbook,
                    "model_config": model_signature(),
                    "task_ids": task_ids,
                    "split": "final",
                    "trials": 1,
                    "modes": list(MODES),
                    "planned_runs": len(task_ids) * len(MODES),
                    "jobs": [],
                    "completed_runs": 0,
                    "workers": 6,
                    "source_training_task_ids": ["0"],
                    "learning_frozen_before_test": True,
                }
            plan["runtime_fingerprints"] = {
                name: hashlib.sha256((ROOT / "app" / name).read_bytes()).hexdigest()
                for name in ["agent.py", "simulation.py", "harness.py", "learning.py"]
            }
            save(plan)
        with Session() as s:
            if fingerprint(s.get(Version, plan["version_id"])) != plan["version_fingerprint"]:
                raise ValueError("Frozen candidate was modified")
        if plan["model_config"] != model_signature():
            raise ValueError("Model configuration changed during experiment")
        for name, value in plan["runtime_fingerprints"].items():
            if hashlib.sha256((ROOT / "app" / name).read_bytes()).hexdigest() != value:
                raise ValueError("Frozen runtime was modified")
        schedule = [(task, mode) for task in plan["task_ids"] for mode in MODES]
        random.Random(20261013).shuffle(schedule)
        known = {(j["task_id"], j["mode"]) for j in plan["jobs"]}
        for task, mode in schedule:
            if (task, mode) in known:
                continue
            response = client.post(
                "/api/jobs",
                headers={
                    "Idempotency-Key": f"ablation-{plan['experiment_id']}-{task}-{mode}-r{plan['trials']}"
                },
                json={
                    "kind": "evaluate",
                    "version_id": plan["version_id"],
                    "split": "final",
                    "mode": mode,
                    "task_ids": [task],
                    "limit": 1,
                    "trials": plan["trials"],
                },
            )
            response.raise_for_status()
            plan["jobs"].append({"task_id": task, "mode": mode, "job_id": response.json()["id"]})
            save(plan)
        children = []
        try:
            for index in range(5):
                logfile = (ROOT / f".runtime/logs/ablation-worker-{index}.log").open(
                    "ab", buffering=0
                )
                child = subprocess.Popen(
                    [sys.executable, "-m", "app.worker"],
                    cwd=ROOT,
                    env={**os.environ, "LOGURU_LEVEL": "WARNING"},
                    stdout=logfile,
                    stderr=logfile,
                    start_new_session=True,
                )
                logfile.close()
                children.append(child)
            plan["worker_pids"] = [child.pid for child in children]
            plan["status"] = "running"
            save(plan)
            last = -1
            deadline = time.monotonic() + 4 * 3600
            while not STOP and time.monotonic() < deadline:
                rows, states = observations(plan["jobs"], plan["trials"])
                plan["completed_runs"] = len(rows)
                plan["job_states"] = states
                plan["runs"] = rows
                if len(rows) != last:
                    print(f"Completed {len(rows)}/{plan['planned_runs']} conversations", flush=True)
                    last = len(rows)
                if all(s["status"] in {"succeeded", "failed", "cancelled"} for s in states) and len(
                    states
                ) == len(schedule):
                    plan["errors"] = [
                        s
                        for s in states
                        if s["status"] != "succeeded" and not s["reused_complete_observation"]
                    ]
                    if plan["errors"]:
                        plan["status"] = "incomplete"
                    else:
                        plan["metrics"] = summarize_runs(rows, plan["task_ids"], plan["trials"])
                        plan["status"] = "completed"
                        plan["finished_at"] = timestamp()
                    save(plan)
                    print("Experiment " + plan["status"], flush=True)
                    return
                save(plan)
                time.sleep(15)
            plan["status"] = "interrupted" if STOP else "timeout"
            save(plan)
        finally:
            for child in children:
                child.terminate()
            for child in children:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    # Let an in-flight queue job finish and preserve its lease/trace.
                    pass


if __name__ == "__main__":
    main()
