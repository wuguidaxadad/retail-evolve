"""Provider, persistence and trace isolation around the unchanged upstream runner.

No proposal, gate, benchmark scoring, agent control-flow or suite-promotion rules
are reimplemented here. The original scripts run with the official train/test.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from .config import RUNTIME_ADAPTER_VERSION, agent_model_route, model_signature, settings
from .db import Job, TaskRun, Version, transaction
from .evaluation_validity import EvaluationValidityError, audit_protocol
from .native_harness import bind_source_snapshot, native_code_version

_phase = None


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


def sanitize(value):
    if isinstance(value, dict):
        return {
            k: sanitize(v)
            for k, v in value.items()
            if k.lower() not in {"api_key", "authorization", "openai_api_key"}
        }
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and settings.api_key:
        return value.replace(settings.api_key, "[redacted]")
    return value


def install():
    """Called by sitecustomize only inside an isolated upstream experiment."""
    if not os.getenv("RETAIL_NATIVE_PARENT_JOB"):
        return
    import tau2.run as facade
    import tau2.runner.batch as batch
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator import evaluator_nl_assertions as nl
    from tau2.runner.helpers import get_tasks

    from .frozen_simulator import (
        is_simulator_error,
        register_simulator,
        simulator_scope,
        simulator_signature,
    )

    repo = Path(os.environ["RETAIL_NATIVE_REPO"])
    private = Path(os.environ["RETAIL_NATIVE_PRIVATE"])
    parent_job = os.environ["RETAIL_NATIVE_PARENT_JOB"]
    lease_token = os.environ["RETAIL_NATIVE_LEASE_TOKEN"]
    process_id = uuid.uuid4().hex
    phase_ids = []
    script = Path(sys.argv[0]).name
    if script in {"gating.py", "record.py"}:
        original_exit = sys.exit

        def snapshot_suite():
            path = repo / "workspace/suite.json"
            return (
                json.loads(path.read_text()) if path.exists() else {"tasks": [], "threshold": 0.8}
            )

        before_suite = snapshot_suite()
        changed_paths = set()
        for args in [
            ("diff-index", "--name-only", "HEAD"),
            ("ls-files", "--others", "--exclude-standard"),
        ]:
            changed_paths.update(
                subprocess.check_output(["git", *args], cwd=repo, text=True).splitlines()
            )
        file_guard_violations = sorted(changed_paths - {"agent/agent.py", "PROGRAM.md"})
        started_at = datetime.now(timezone.utc).isoformat()

        def recorded_exit(code=0):
            folder = private / "commands"
            folder.mkdir(exist_ok=True)
            atomic_json(
                folder / (process_id + ".json"),
                {
                    "id": process_id,
                    "script": script,
                    "argv": sys.argv[1:],
                    "exit_code": code,
                    "created_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "phase_ids": phase_ids,
                    "suite_before": before_suite,
                    "suite_after": snapshot_suite(),
                    "file_guard_violations": file_guard_violations,
                    "code_hash": hashlib.sha256((repo / "agent/agent.py").read_bytes()).hexdigest(),
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                    ).strip(),
                },
            )
            original_exit(code)

        sys.exit = recorded_exit
    original_config_init = TextRunConfig.__init__
    original_domain = facade.run_domain
    original_single = batch.run_single_task
    # The compatible endpoint/key are environment-only and never serialized in Info.
    os.environ["OPENAI_API_KEY"] = settings.api_key
    os.environ["OPENAI_API_BASE"] = settings.base_url
    os.environ["OPENAI_BASE_URL"] = settings.base_url
    nl.DEFAULT_LLM_NL_ASSERTIONS = "openai/" + settings.judge_model
    from .agent import llm_args
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
        **llm_args(settings.judge_model),
        "response_format": {"type": "json_object"},
    }

    def config_init(self, **kwargs):
        kwargs["user"] = register_simulator()
        kwargs["llm_agent"] = agent_model_route()
        kwargs.update(
            num_trials=1,
            log_level="ERROR",
            max_retries=2,
            retry_delay=2,
            timeout=None,
            verbose_logs=False,
        )
        kwargs["llm_args_agent"] = llm_args(settings.agent_model)
        kwargs["llm_args_user"] = llm_args(settings.user_model)
        kwargs["save_to"] = str(private / ("native-" + uuid.uuid4().hex))
        original_config_init(self, **kwargs)

    def persist_simulation(simulation, task, elapsed, resumed=False):
        phase = _phase
        if phase is None:
            return simulation
        reward = simulation.reward_info
        expected = task.evaluation_criteria.nl_assertions or [] if task.evaluation_criteria else []
        normal_stop = simulation.termination_reason.value in {"agent_stop", "user_stop"}
        if expected and normal_stop:
            actual = (
                [r.nl_assertion for r in reward.nl_assertions]
                if reward and reward.nl_assertions
                else []
            )
            if sorted(actual) != sorted(expected):
                raise ValueError("Native judge did not cover every declared NL assertion")
        data = sanitize(simulation.model_dump(mode="json"))
        data["evaluation_validity"] = audit_protocol(data)
        from .dataset import task_summary

        data["task_summary"] = task_summary(task.model_dump(mode="json"))
        data.update(
            adapter_version=RUNTIME_ADAPTER_VERSION,
            native_run=True,
            simulator=simulator_signature(),
            native_phase=phase["id"],
            native_split=phase["split"],
            executed_source_hash=phase["code_hash"],
            source_binding="immutable source snapshot factory",
            elapsed_seconds=round(elapsed, 2),
            resumed_from_checkpoint=resumed,
            mode="harness",
            injected_skill_ids=[],
            agent_model=settings.agent_model,
            user_model=settings.user_model,
            judge_model=settings.judge_model,
            evaluator="unmodified tau2 native run_domain ALL_WITH_NL_ASSERTIONS",
        )
        with transaction() as s:
            owner = s.get(Job, parent_job)
            if not owner or owner.status != "running" or owner.lease_token != lease_token:
                raise RuntimeError("Native experiment no longer running")
            old = s.scalar(
                select(TaskRun).where(
                    TaskRun.job_id == phase["id"], TaskRun.task_id == str(task.id)
                )
            )
            if old is None:
                s.add(
                    TaskRun(
                        job_id=phase["id"],
                        task_id=str(task.id),
                        trial=0,
                        label=(
                            "invalid_observation"
                            if data["evaluation_validity"]["status"] == "invalid"
                            else "evaluation"
                        ),
                        version_id=phase["version_id"],
                        reward=float(reward.reward) if reward else 0.0,
                        data=data,
                    )
                )
            child = s.get(Job, phase["id"], with_for_update=True)
            child.progress += 1 / len(phase["task_ids"])
            child.stage = f"原生 {phase['split']} · 完成任务 {task.id}"
        if phase["split"] == "train" and data["evaluation_validity"]["status"] != "invalid":
            latest = repo / "workspace/traces/latest"
            baseline = repo / "workspace/traces/baseline"
            latest.mkdir(parents=True, exist_ok=True)
            baseline.mkdir(parents=True, exist_ok=True)
            value = json.dumps(data, ensure_ascii=False, indent=2)
            (latest / (str(task.id) + ".json")).write_text(value)
            if not (baseline / (str(task.id) + ".json")).exists():
                (baseline / (str(task.id) + ".json")).write_text(value)
        return simulation

    def single(config, task, *args, **kwargs):
        started = time.monotonic()
        with simulator_scope(parent_job + "/" + str(task.id)):
            try:
                simulation = original_single(config, task, *args, **kwargs)
            except Exception as exc:
                if not is_simulator_error(exc):
                    raise
                # Return a quarantined observation rather than allowing upstream
                # generic exception retries to silently reroll this scenario.
                from tau2.data_model.simulation import SimulationRun
                simulation = SimulationRun(
                    id=uuid.uuid4().hex, task_id=str(task.id),
                    start_time=datetime.now(timezone.utc).isoformat(),
                    end_time=datetime.now(timezone.utc).isoformat(),
                    duration=time.monotonic() - started,
                    termination_reason="infrastructure_error", messages=[],
                    info={"error_type": "SimulatorProtocolError", "error": str(exc)},
                )
        return persist_simulation(simulation, task, time.monotonic() - started)

    def domain(config):
        global _phase
        if config.task_split_name not in {"train", "test"}:
            raise ValueError("Native RSI uses official train/test only")
        if settings.experiment_profile == "aligned-v1":
            from .dataset import splits
            allowed = splits()["learn" if config.task_split_name == "train" else "validation"]
            if config.task_ids is not None and not set(map(str, config.task_ids)) <= set(allowed):
                raise ValueError("Optimizer attempted to access a sealed scorecard task")
            config.task_ids = list(config.task_ids) if config.task_ids is not None else allowed
            # Upstream orchestration is retained; select our frozen task IDs from
            # the full pool rather than intersecting with the legacy train/test.
            selected = get_tasks(task_set_name=config.domain, task_split_name=None,
                                 task_ids=config.task_ids, num_tasks=None)
        else:
            selected = get_tasks(task_set_name=config.domain, task_split_name=config.task_split_name,
                                 task_ids=config.task_ids, num_tasks=None)
        code = (repo / "agent/agent.py").read_text()
        # Bind evaluation to exactly the source hashed below, bypassing Python's
        # stale agent.agent import cache when a process evaluates multiple edits.
        config.agent = bind_source_snapshot(code)
        version_id = native_code_version(code)
        phase_id = uuid.uuid4().hex
        phase = {
            "id": phase_id,
            "split": config.task_split_name,
            "task_ids": [str(t.id) for t in selected],
            "version_id": version_id,
            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": False,
            "process_id": process_id,
            "script": script,
        }
        ledger = repo / "workspace/iterations.jsonl"
        iteration = len(ledger.read_text().splitlines()) if ledger.exists() else 0
        run_config = config.model_dump(mode="json", exclude={"save_to", "auto_resume"})
        checkpoint_identity = {
            "script": script,
            "iteration": iteration,
            "call": len(phase_ids),
            "code_hash": phase["code_hash"],
            "model_config": model_signature(),
            "run_config": run_config,
            "task_ids": phase["task_ids"],
        }
        digest = hashlib.sha256(
            json.dumps(checkpoint_identity, sort_keys=True).encode()
        ).hexdigest()
        checkpoint_dir = private / "checkpoints" / digest
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(checkpoint_dir / "identity.json", checkpoint_identity)
        config.save_to = str(checkpoint_dir)
        config.auto_resume = True
        phase["checkpoint"] = str(checkpoint_dir)
        phase_ids.append(phase_id)
        with transaction() as s:
            owner = s.get(Job, parent_job, with_for_update=True)
            if not owner or owner.status != "running" or owner.lease_token != lease_token:
                raise RuntimeError("Native experiment lease expired")
            if not s.get(Version, version_id):
                s.add(
                    Version(
                        id=version_id,
                        name="完整原生策略 " + version_id[:6],
                        origin="auto-harness-native",
                        parent_id=os.environ["RETAIL_NATIVE_BASELINE_VERSION"],
                        harness_code=code,
                        skillbook={
                            "skills": {},
                            "sections": {},
                            "next_id": 0,
                            "schema_version": "2",
                        },
                        metadata_json={
                            "runtime_kind": "auto_harness_full",
                            "native_parent_job": parent_job,
                            "model_config": model_signature(),
                        },
                    )
                )
                s.flush()
            s.add(
                Job(
                    id=phase_id,
                    kind="evaluate",
                    status="running",
                    attempts=1,
                    idempotency_key="native-phase-" + phase_id,
                    payload={
                        "version_id": version_id,
                        "split": "learn" if phase["split"] == "train" else "upstream_test",
                        "mode": "harness",
                        "task_ids": phase["task_ids"],
                        "trials": 1,
                        "model_config": model_signature(),
                        "native_parent_job": parent_job,
                    },
                )
            )
        _phase = phase
        phase_dir = private / "phases"
        phase_dir.mkdir(parents=True, exist_ok=True)
        phase_path = phase_dir / (phase_id + ".json")
        atomic_json(phase_path, phase)
        try:
            logical_split = config.task_split_name
            if settings.experiment_profile == "aligned-v1":
                config.task_split_name = None
            try:
                result = original_domain(config)
            finally:
                config.task_split_name = logical_split
            # The original runner resumes already-completed simulations without
            # calling single(); mirror those real results into this phase's DB rows.
            with transaction() as session:
                persisted = set(
                    session.scalars(select(TaskRun.task_id).where(TaskRun.job_id == phase_id))
                )
            tasks_by_id = {str(task.id): task for task in selected}
            for simulation in result.simulations:
                task_id = str(simulation.task_id)
                if task_id not in persisted:
                    persist_simulation(simulation, tasks_by_id[task_id], 0, resumed=True)
            got = {str(s.task_id) for s in result.simulations}
            if got != set(phase["task_ids"]):
                raise ValueError("Original runner omitted tasks: cannot report a complete score")
            infrastructure_failures = [
                str(sim.task_id)
                for sim in result.simulations
                if sim.termination_reason.value in {"infrastructure_error", "timeout"}
                and audit_protocol(sim.model_dump(mode="json"))["status"] != "invalid"
            ]
            if infrastructure_failures:
                phase["infrastructure_failures"] = infrastructure_failures
                raise ConnectionError(
                    "Native infrastructure failures require checkpoint recovery: "
                    + ",".join(infrastructure_failures)
                )
            invalid_observations = {
                str(sim.task_id): audit_protocol(sim.model_dump(mode="json"))
                for sim in result.simulations
                if audit_protocol(sim.model_dump(mode="json"))["status"] == "invalid"
            }
            if invalid_observations:
                phase["invalid_observations"] = invalid_observations
                # Keep the upstream checkpoint and raw rewards for diagnosis.
                # No score is returned to gating.py or the optimizer.
                raise EvaluationValidityError(
                    "Invalid evaluation protocol: " + ",".join(invalid_observations)
                )
            scores = {
                str(s.task_id): float(s.reward_info.reward) if s.reward_info else 0.0
                for s in result.simulations
            }
            if (settings.experiment_profile == "aligned-v1" and phase["split"] == "train"
                    and set(phase["task_ids"]) == set(splits()["learn"])):
                from .failure_analysis import analyze_training
                evidence = [sanitize(sim.model_dump(mode="json")) for sim in result.simulations]
                analysis = analyze_training(evidence, phase["code_hash"], phase["task_ids"])
                atomic_json(repo / "workspace/failure_analysis.json", analysis)
                atomic_json(private / (phase_id + "-failure-analysis.json"), analysis)
                phase["failure_analysis"] = phase_id + "-failure-analysis.json"
            phase.update(
                complete=True, scores=scores, mean_reward=sum(scores.values()) / len(scores)
            )
            with transaction() as s:
                child = s.get(Job, phase_id, with_for_update=True)
                child.status, child.progress, child.stage = "succeeded", 1, "原生评测完成"
                child.result = {
                    "scores": scores,
                    "mean_reward": phase["mean_reward"],
                    "sample_count": len(scores),
                    "trials": 1,
                    "split": child.payload["split"],
                    "mode": "harness",
                    "metric": "official tau2 native runner",
                    "model_config": model_signature(),
                }
            return result
        except BaseException as exc:
            with transaction() as s:
                child = s.get(Job, phase_id)
                child.status, child.error = "failed", sanitize(str(exc))[:3000]
            raise
        finally:
            atomic_json(phase_path, phase)
            _phase = None
            # Test trajectories exist only in private controller storage / MySQL.
            # They are never copied into the coding agent's train trace directory.

    TextRunConfig.__init__ = config_init
    batch.run_single_task = single
    facade.run_domain = domain
