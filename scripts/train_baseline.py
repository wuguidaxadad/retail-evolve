"""Collect the full official train split with the untouched upstream agent."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import timedelta

from ace import Skillbook
from sqlalchemy import select

from app.config import ROOT, model_signature, settings
from app.dataset import splits
from app.db import Job, TaskRun, Version, now, transaction
from app.full_rsi import environment, hashes, pinned_hashes, run_initial_stage, setup
from app.native_bridge import atomic_json, sanitize
from app.native_harness import native_code_version


def main():
    if len({settings.agent_model, settings.user_model, settings.learning_model}) != 1:
        raise ValueError("Full training baseline requires the same configured model for every role")
    report_prefix = re.sub(r"[^a-zA-Z0-9_.-]", "_", settings.agent_model)
    if settings.agent_model == "deepseek-v4.1-flash":
        report_prefix = "deepseek"
    run_id, token = uuid.uuid4().hex, uuid.uuid4().hex
    code = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    version_id = native_code_version(code)
    job = {"id": run_id, "payload": {"version_id": version_id, "model_config": model_signature()}}
    run, repo, private, adapter = setup(job)
    expected = set(splits()["learn"])
    if len(expected) != 74:
        raise ValueError("Unexpected official training split")
    report = {
        "id": run_id,
        "pid": os.getpid(),
        "status": "running",
        "stage": "full_train_baseline",
        "model_config": model_signature(),
        "version_id": version_id,
        "source_hash": hashlib.sha256(code.encode()).hexdigest(),
        "task_ids": sorted(expected),
        "planned_tasks": 74,
        "trials": 1,
        "concurrency": 6,
        "scope": "Original agent, full official train only; no optimizer or test evaluation",
        "run_directory": str(run),
        "started_at": now().isoformat(),
    }
    report_path = ROOT / "docs" / f"{report_prefix}-train-{run_id}.json"

    def save():
        atomic_json(report_path, sanitize(report))
        atomic_json(ROOT / "docs" / f"{report_prefix}-train-latest.json", sanitize(report))

    with transaction() as session:
        if not session.get(Version, version_id):
            session.add(
                Version(
                    id=version_id,
                    name=settings.agent_model + " original upstream baseline",
                    origin="native-baseline",
                    status="validated",
                    harness_code=code,
                    skillbook=Skillbook().to_dict(),
                    metadata_json={
                        "runtime_kind": "auto_harness_full",
                        "model_config": model_signature(),
                    },
                )
            )
        session.add(
            Job(
                id=run_id,
                kind="baseline_scan",
                status="running",
                payload=job["payload"],
                lease_token=token,
                lease_until=now() + timedelta(hours=2),
                idempotency_key="full-training-" + run_id,
            )
        )
    save()
    print(json.dumps({"id": run_id, "tasks": 74, "concurrency": 6}), flush=True)
    try:
        env = environment(job, repo, private, adapter, token)
        run_initial_stage(
            "benchmark.py",
            "train.log",
            settings.agent_model + " 全量 74 条训练基线",
            repo,
            env,
            job,
            token,
            private,
        )
        phases = [json.loads(p.read_text()) for p in (private / "phases").glob("*.json")]
        phase = next(p for p in phases if p["complete"] and set(p["task_ids"]) == expected)
        if phase["split"] != "train" or phase["code_hash"] != report["source_hash"]:
            raise ValueError("Training phase source or split mismatch")
        if hashes(repo) != pinned_hashes(repo) or (repo / "agent/agent.py").read_text() != code:
            raise ValueError("Baseline source changed")
        with transaction() as session:
            rows = session.scalars(select(TaskRun).where(TaskRun.job_id == phase["id"])).all()
            if len(rows) != 74 or {r.task_id for r in rows} != expected:
                raise ValueError("Missing or duplicated task observations")
            if any(r.data.get("executed_source_hash") != report["source_hash"] for r in rows):
                raise ValueError("Observed runtime source differs from baseline")
            observations = [
                {
                    "task_id": r.task_id,
                    "reward": r.reward,
                    "termination_reason": r.data.get("termination_reason"),
                    "trace_id": r.id,
                }
                for r in rows
            ]
            report.update(
                status="succeeded",
                stage="complete",
                phase_id=phase["id"],
                completed_tasks=74,
                raw_successes=sum(r.reward >= 0.5 for r in rows),
                mean_reward=phase["mean_reward"],
                observations=observations,
                diagnostic_review="pending: raw failures are not yet attributed",
                completed_at=now().isoformat(),
            )
            owner = session.get(Job, run_id)
            owner.status, owner.progress, owner.result = "succeeded", 1, sanitize(report)
        save()
    except BaseException as exc:
        report.update(status="failed", stage="incomplete", error=sanitize(str(exc)))
        save()
        with transaction() as session:
            owner = session.get(Job, run_id)
            owner.status, owner.error = "failed", sanitize(str(exc))
        raise


if __name__ == "__main__":
    main()
