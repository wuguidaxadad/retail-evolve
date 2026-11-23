"""Run independent native RSI and ACE training, then a frozen three-arm comparison.

This is a process for the current local experiment, not a scheduled automation.
The original RSI scripts and actual coding agent perform the optimization.
"""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from app.config import ROOT, RUNTIME_ADAPTER_VERSION, model_signature, settings
from app.dataset import splits
from app.db import Deployment, Job, ReleaseEvent, TaskRun, Version, transaction
from app.experiment_metrics import MODES, summarize_runs
from app.frozen_simulator import simulator_signature
from app.native_bridge import atomic_json
from app.native_harness import native_code_version

MANIFEST = ROOT / "docs/aligned-customer-experiment.json"
REPORT = ROOT / "docs/ALIGNED_CUSTOMER_EXPERIMENT.md"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(plan):
    plan["updated_at"] = stamp()
    atomic_json(MANIFEST, plan)
    lines = [
        "# 角色分离与独立测试：零售客服三组实验",
        "",
        f"状态：{plan['status']}；阶段：{plan['stage']}",
        "",
        "客服 Gemini-3.5-Flash-Lite；优化器与 ACE DeepSeek-V4.1-Flash；模拟用户和评判 Hy3，使用 A6。",
        "保留冻结 Hy3 v10 模拟器；整文件 Auto-harness + 失败聚类 + Airbnb 数据划分，为组合实验，非严格复现。",
        "公开原生 Agent 起点；RSI 与 ACE 各自从无经验的同一基线训练，互不组合。",
        "保留上游循环与五轮无提升停止规则；RSI 与 ACE 均只使用同一 20 条修复集。",
        "三组统一取消本地 600 秒会话上限，保留上游 200 步限制及单次调用超时；调用失败或截断使实验未完成，不计业务失败。",
        "最终使用独立 74 条 scorecard，三组每题 1 次，共 222 次会话，报告单次成功率。",
        "20 条修复、20 条门禁、74 条最终测试互不重叠；最终测试只在版本冻结后运行。历史实验已用过部分任务，独立性仅针对本次优化过程。",
        "",
    ]
    if plan.get("simulator_limitation"):
        lines += ["", plan["simulator_limitation"]]
    if plan.get("metrics"):
        lines += ["| 分组 | 单次成功率 | 相对基线变化 | 四次全对率 |", "| --- | --- | --- | --- |"]
        for mode, name in [("baseline", "基线"), ("harness", "仅 RSI"), ("skills", "仅 Skill")]:
            m = plan["metrics"][mode]
            delta = m.get("success_delta_percentage_points")
            lines.append(
                f"| {name} | {m['successes']}/{m['runs']}（{m['success_rate']:.1%}） | "
                + (f"{delta:+.1f} 个百分点" if delta is not None else "—")
                + " | — |"
            )
        lines += [
            "",
            "没有按结果好坏筛选轨迹；报告单次成功率。渠道费用未核实，不虚报。",
        ]
    if plan.get("error"):
        lines += ["", "本次运行未完成，不报告完整提升率。", plan["error"]]
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    if settings.experiment_profile != "aligned-v1":
        raise ValueError("Aligned experiment profile required")
    if (settings.agent_model, settings.user_model, settings.learning_model, settings.optimizer_model) != ("gemini-3.5-flash-lite", "hy3", "deepseek-v4.1-flash", "deepseek-v4.1-flash"):
        raise ValueError("Mixed model-role configuration")
    from app.preflight import require_preflight
    require_preflight()
    if simulator_signature()["protocol"] != "scenario-faithfulness-v10":
        raise ValueError("Formal Hy3 run requires the frozen v10 simulator")
    client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=30, trust_env=False)
    code = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    baseline_id = native_code_version(code)
    from ace import Skillbook

    with transaction() as session:
        if not session.get(Version, baseline_id):
            session.add(
                Version(
                    id=baseline_id,
                    name="Gemini-3.5-Flash-Lite 上游原始客服 Agent",
                    origin="native-baseline",
                    status="validated",
                    harness_code=code,
                    skillbook=Skillbook().to_dict(),
                    metadata_json={
                        "runtime_kind": "auto_harness_full",
                        "model_config": model_signature(),
                        "baseline": True,
                    },
                )
            )
            session.flush()
        deployed = session.get(Deployment, 1, with_for_update=True)
        if deployed.version_id != baseline_id:
            session.add(
                ReleaseEvent(
                    version_id=baseline_id,
                    previous_id=deployed.version_id,
                    action="model-migration",
                    details={"model_config": model_signature(), "requested_customer_model": settings.agent_model},
                )
            )
            deployed.version_id, deployed.revision = baseline_id, deployed.revision + 1
    plan = (
        json.loads(MANIFEST.read_text())
        if MANIFEST.exists()
        else {
            "created_at": stamp(),
            "status": "running",
            "stage": "完整 RSI 初始基线",
            "baseline_version_id": baseline_id,
            "model_config": model_signature(),
            "optimizer_model": settings.optimizer_model,
            "trials": 1,
            "task_ids": splits()["final"],
            "planned_runs": 222,
            "modes": list(MODES),
            "evaluation_jobs": [],
        }
    )
    if plan["model_config"] != model_signature() or plan["baseline_version_id"] != baseline_id:
        raise ValueError("Do not mix models with an existing experiment")

    plan["status"] = "running"
    plan.pop("error", None)

    def submit(kind, payload, key):
        response = client.post(
            "/api/jobs", json={"kind": kind, **payload}, headers={"Idempotency-Key": key}
        )
        response.raise_for_status()
        return response.json()["id"]

    def state(job_id):
        with transaction() as session:
            job = session.get(Job, job_id)
            if job.status in {"failed", "cancelled", "invalidated"}:
                raise RuntimeError(f"{job.kind} {job.id}: {job.error or job.status}")
            return job.status, job.result

    children = []
    inherited_workers = []
    for pid in json.loads(os.getenv("RETAIL_REUSE_WORKERS", "[]")):
        try:
            os.kill(pid, 0)
            inherited_workers.append(pid)
        except ProcessLookupError:
            pass
    try:
        if not plan.get("rsi_job_id"):
            plan["rsi_job_id"] = submit(
                "rsi",
                {"version_id": baseline_id, "optimizer_model": settings.optimizer_model},
                "aligned-roles-scorecard74-full-rsi-frozen-v10-adapter8",
            )
            save(plan)
        # One additional queue consumer lets ACE learn from the unchanged baseline
        # while the original RSI controller remains active on the primary worker.
        for index in range(max(0, 5 - len(inherited_workers))):
            log = (ROOT / f".runtime/logs/aligned-experiment-worker-{index}.log").open("ab")
            child = subprocess.Popen(
                [sys.executable, "-m", "app.worker"],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            log.close()
            children.append(child)
        plan["worker_pids"] = inherited_workers + [p.pid for p in children]
        save(plan)
        while not plan.get("skill_job_id"):
            state(plan["rsi_job_id"])
            with transaction() as session:
                jobs = session.scalars(
                    select(Job).where(Job.kind == "evaluate", Job.status == "succeeded")
                ).all()
                source = next(
                    (
                        j
                        for j in jobs
                        if j.payload.get("native_parent_job") == plan["rsi_job_id"]
                        and j.payload.get("version_id") == baseline_id
                        and j.payload.get("split") == "learn"
                        and set(j.payload.get("task_ids", [])) == set(splits()["learn"])
                    ),
                    None,
                )
                source_id = source.id if source else None
            if source_id:
                plan["source_training_job_id"] = source_id
                plan["skill_job_id"] = submit(
                    "learn",
                    {"version_id": baseline_id, "source_job_id": source_id},
                    "aligned-roles-scorecard74-ace-frozen-v10-adapter8",
                )
                plan["stage"] = "完整 RSI 和独立 ACE 学习"
                save(plan)
            else:
                time.sleep(15)
        while True:
            selection = plan.get("user_early_stop")
            if selection:
                with transaction() as session:
                    selected = session.get(Version, selection["version_id"])
                    if (not selected or selected.status != "validated"
                            or selected.metadata_json.get("user_early_stop") != selection):
                        raise ValueError("Unverified early-stop version")
                rsi_status, rsi_result = "succeeded", {"version_id": selected.id}
            else:
                rsi_status, rsi_result = state(plan["rsi_job_id"])
            skill_status, skill_result = state(plan["skill_job_id"])
            if rsi_status == skill_status == "succeeded":
                versions = {
                    "baseline": baseline_id,
                    "harness": rsi_result["version_id"],
                    "skills": skill_result["version_id"],
                }
                plan["versions"] = versions
                break
            time.sleep(15)
        plan["stage"] = "固定版本后三组各 74 题，每题 1 次"
        task_ids = splits()["final"]
        plan["task_ids"], plan["planned_runs"] = task_ids, len(task_ids) * 3
        schedule = [(task, mode) for task in task_ids for mode in MODES]
        random.Random(20261013).shuffle(schedule)
        known = {(j["task_id"], j["mode"]) for j in plan["evaluation_jobs"]}
        for task, mode in schedule:
            if (task, mode) in known:
                continue
            job_id = submit(
                "evaluate",
                {
                    "version_id": versions[mode],
                    "split": "final",
                    "mode": mode,
                    "task_ids": [task],
                    "limit": 1,
                    "trials": 1,
                },
                f"aligned-roles-scorecard74-{task}-{mode}-frozen-v10-adapter8",
            )
            plan["evaluation_jobs"].append(
                {"job_id": job_id, "task_id": task, "mode": mode, "version_id": versions[mode]}
            )
            save(plan)
        mapping = {j["job_id"]: j for j in plan["evaluation_jobs"]}
        while True:
            statuses = [state(j)[0] for j in mapping]
            with transaction() as session:
                rows = session.scalars(select(TaskRun).where(TaskRun.job_id.in_(mapping))).all()
                observed = []
                for row in rows:
                    # Historical repeat trials remain stored but are excluded by index.
                    if row.trial != 0:
                        continue
                    spec, data = mapping[row.job_id], row.data
                    if (
                        row.trial not in range(1)
                        or row.version_id != spec["version_id"]
                        or data.get("adapter_version") != RUNTIME_ADAPTER_VERSION
                        or data.get("simulator") != simulator_signature()
                        or data.get("mode") != spec["mode"]
                        or not data.get("native_run")
                    ):
                        raise ValueError("Frozen native comparison provenance mismatch")
                    if (
                        data.get("agent_model"),
                        data.get("user_model"),
                        data.get("judge_model"),
                    ) != (settings.agent_model, settings.user_model, settings.judge_model):
                        raise ValueError("Mixed model roles in comparison")
                    if spec["mode"] != "skills" and data.get("injected_skill_ids"):
                        raise ValueError("Skill isolation failed")
                    inputs = outputs = cached = 0
                    for message in data.get("messages", []):
                        usage = message.get("usage") or {}
                        inputs += usage.get("prompt_tokens", 0) or 0
                        outputs += usage.get("completion_tokens", 0) or 0
                        raw_usage = (message.get("raw_data") or {}).get("usage") or {}
                        cached += (raw_usage.get("prompt_tokens_details") or {}).get(
                            "cached_tokens", 0
                        ) or 0
                    observed.append(
                        {
                            "run_id": row.id,
                            "job_id": row.job_id,
                            "task_id": row.task_id,
                            "mode": spec["mode"],
                            "trial": row.trial,
                            "reward": row.reward,
                            "seconds": data["elapsed_seconds"],
                            "termination_reason": data.get("termination_reason"),
                            "evaluation_validity": data.get("evaluation_validity"),
                            "input_tokens": inputs,
                            "output_tokens": outputs,
                            "cached_input_tokens": cached,
                            "estimated_agent_user_usd": None,
                        }
                    )
            plan["completed_runs"], plan["runs"] = len(observed), observed
            if all(status == "succeeded" for status in statuses):
                plan["metrics"] = summarize_runs(observed, task_ids, trials=1)
                for metric in plan["metrics"].values():
                    metric["estimated_agent_user_usd"] = None
                plan["status"], plan["stage"] = "completed", "三组对照已完成"
                save(plan)
                return
            save(plan)
            time.sleep(15)
    except Exception as exc:
        plan["status"], plan["error"] = (
            "incomplete",
            str(exc).replace(settings.api_key, "[redacted]"),
        )
        save(plan)
        raise
    finally:
        for child in children:
            child.terminate()
        for pid in inherited_workers:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
