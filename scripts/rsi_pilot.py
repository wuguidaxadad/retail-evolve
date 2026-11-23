"""One real RSI loop on a fixed tiny subset; never a full-reproduction score."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import timedelta

import httpx
from ace import Skillbook

from app.config import ROOT, model_signature, settings
from app.dataset import splits
from app.db import Job, Version, now, transaction
from app.full_rsi import CORE, environment, git, hashes, pinned_hashes, setup, validate_gate_sources
from app.native_bridge import atomic_json, sanitize
from app.native_harness import native_code_version


def main():
    if len({settings.agent_model, settings.user_model, settings.learning_model}) != 1:
        raise ValueError("Pilot requires the same configured model for every role")
    run_id, token = uuid.uuid4().hex, uuid.uuid4().hex
    code = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    baseline = native_code_version(code)
    scope = {"train": splits()["learn"][:3], "test": splits()["validation"][:2]}
    job = {"id": run_id, "payload": {"version_id": baseline, "model_config": model_signature()}}
    run, repo, private, adapter = setup(job)
    report_path = ROOT / "docs" / f"rsi-pilot-{run_id}.json"
    report = {
        "id": run_id,
        "status": "running",
        "stage": "initializing",
        "scope": scope,
        "model_config": model_signature(),
        "purpose": "one-loop engineering pilot, not full reproduction",
        "run_directory": str(run),
        "report": str(report_path),
    }

    def update(stage, **extra):
        report.update(stage=stage, **extra)
        atomic_json(report_path, sanitize(report))
        atomic_json(ROOT / "docs/rsi-pilot-latest.json", sanitize(report))
        with transaction() as session:
            owner = session.get(Job, run_id)
            if owner:
                owner.stage = stage
                owner.lease_until = now() + timedelta(hours=1)
        print(stage, flush=True)

    with transaction() as session:
        if not session.get(Version, baseline):
            session.add(
                Version(
                    id=baseline,
                    name=settings.agent_model + " upstream baseline",
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
                kind="pilot",
                status="running",
                lease_token=token,
                lease_until=now() + timedelta(hours=1),
                payload=job["payload"],
                idempotency_key="pilot-" + run_id,
            )
        )
    gateway = None
    try:
        config = repo / "experiment_config.yaml"
        config.write_text(config.read_text().replace("max_concurrency: 6", "max_concurrency: 2"))
        git(repo, "add", "experiment_config.yaml")
        git(
            repo,
            "commit",
            "-qm",
            "Pilot only: concurrency two; fixed subset selected by controller",
        )
        # Install the production bridge, then constrain only the requested sample.
        # Original scripts, gate logic, scoring and source binding remain unchanged.
        (adapter / "sitecustomize.py").write_text(
            "from app.native_bridge import install\ninstall()\n"
            "import tau2.run as facade\noriginal = facade.run_domain\n"
            f"scope = {scope!r}\n"
            "def pilot_domain(config):\n"
            "    allowed = scope[config.task_split_name]\n"
            "    if config.task_ids is None: config.task_ids = allowed\n"
            "    if not set(map(str, config.task_ids)) <= set(allowed): raise ValueError('Outside pilot scope')\n"
            "    return original(config)\n"
            "facade.run_domain = pilot_domain\n"
        )
        env = environment(job, repo, private, adapter, token)

        def execute(args, label, allowed=(0,), timeout=1200):
            update(label)
            with (private / (label + ".log")).open("w") as log:
                process = subprocess.Popen(
                    args, cwd=repo, env=env, stdout=log, stderr=log, start_new_session=True
                )
                try:
                    status = process.wait(timeout=timeout)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                    raise
            if status not in allowed:
                raise RuntimeError(f"{label} exited {status}; inspect its isolated log")
            return status

        update(
            "baseline_validation", baseline_source_hash=hashlib.sha256(code.encode()).hexdigest()
        )
        execute([sys.executable, "prepare.py"], "baseline_validation")
        execute([sys.executable, "benchmark.py"], "baseline_training")
        # Dedicated gateway uses current source; no restart of the user's services.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (private / "gateway.log").open("w") as log:
            gateway = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.codex_gateway:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        for _ in range(50):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("Pilot gateway did not start")
        prompt = """This is ONE engineering pilot iteration, not the full experiment. Read PROGRAM.md.
The controller has already run the original prepare.py and benchmark.py on a fixed tiny subset.
Perform ONLY Step 2 and Step 3: analyze workspace/traces/latest train traces, append your hypothesis
and changes to workspace/learnings.md, and edit agent/agent.py with one generalizable improvement.
You may modify the complete agent including generate_next_message, state, messages or instructions.
Do not force a control-flow change without evidence. Never hardcode task/customer/order IDs.
Do not access paths outside this repo, secrets, test traces, or evaluation criteria. No new tools.
Do not change any other file. Do not run benchmark/gating/prepare/record or commit: the controller
will execute the original gate and retain/revert your change. Do not spawn other agents.
Finish only after saving the actual edit and explain exactly what changed.
"""
        command = [
            shutil.which("codex"),
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--disable",
            "apps",
            "--disable",
            "plugins",
            "--disable",
            "multi_agent",
            "-m",
            settings.learning_model,
            "-s",
            "workspace-write",
            "-C",
            str(repo),
            "--json",
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
            "-c",
            'model_reasoning_effort="none"',
            "-c",
            'model_provider="a6"',
            "-c",
            'model_providers.a6.name="A6"',
            "-c",
            f'model_providers.a6.base_url="http://127.0.0.1:{port}/v1"',
            "-c",
            'model_providers.a6.env_key="A6_RSI_API_KEY"',
            "-c",
            'model_providers.a6.wire_api="responses"',
            "-c",
            'shell_environment_policy.inherit="all"',
            prompt,
        ]
        execute(command, "optimizer", timeout=600)
        candidate = (repo / "agent/agent.py").read_text()
        if candidate == code:
            raise RuntimeError("Optimizer produced no saved candidate")
        compile(candidate, "agent/agent.py", "exec")

        def functions(source):
            return {
                n.name: ast.dump(n, include_attributes=False)
                for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef)
            }

        before, after = functions(code), functions(candidate)
        report["changed_functions"] = [
            k for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)
        ]
        report["candidate_source_hash"] = hashlib.sha256(candidate.encode()).hexdigest()
        (private / "candidate.py").write_text(candidate)
        (ROOT / "docs" / f"rsi-pilot-{run_id}.diff").write_text(
            git(repo, "diff", "--", "agent/agent.py") + "\n"
        )
        status = execute([sys.executable, "gating.py"], "candidate_gate", allowed=(0, 1))
        phases = [json.loads(p.read_text()) for p in (private / "phases").glob("*.json")]
        if any(not p["complete"] for p in phases):
            raise RuntimeError("Pilot has incomplete infrastructure observations")
        gates = [json.loads(p.read_text()) for p in (private / "commands").glob("*.json")]
        gate = next(c for c in gates if c["script"] == "gating.py")
        executed = [p for p in phases if p["id"] in gate["phase_ids"]]
        validate_gate_sources(gate, executed)
        validation = [p for p in executed if p["split"] == "test"]
        if len(validation) != 1:
            raise RuntimeError("Candidate did not complete pilot validation")
        if status == 0:
            git(repo, "add", "agent/agent.py")
            git(repo, "commit", "-qm", "Pilot candidate passed original gate on fixed subset")
            execute(
                [
                    sys.executable,
                    "record.py",
                    "--val-score",
                    str(validation[0]["mean_reward"]),
                    "--evals-passed",
                    "0",
                    "--evals-total",
                    "0",
                ],
                "record_candidate",
            )
        else:
            git(repo, "checkout", "HEAD", "--", "agent/agent.py")
        if hashes(repo) != pinned_hashes(repo):
            raise RuntimeError("Upstream core changed")
        expected = candidate if status == 0 else code
        if (repo / "agent/agent.py").read_text() != expected:
            raise RuntimeError("Final source mismatch")
        update(
            "complete",
            status="succeeded",
            kept=status == 0,
            baseline_validation=next(
                p["mean_reward"] for p in phases if p["script"] == "prepare.py"
            ),
            candidate_validation=validation[0]["mean_reward"],
            phases=phases,
            core_files_verified=list(CORE),
            final_source_hash=hashlib.sha256(expected.encode()).hexdigest(),
        )
        with transaction() as session:
            owner = session.get(Job, run_id)
            owner.status = "succeeded"
            owner.result = sanitize(report)
    except BaseException as exc:
        update("failed", status="failed", error=sanitize(str(exc)))
        with transaction() as session:
            owner = session.get(Job, run_id)
            owner.status = "failed"
            owner.error = sanitize(str(exc))
        raise
    finally:
        if gateway and gateway.poll() is None:
            gateway.terminate()
            gateway.wait(timeout=15)


if __name__ == "__main__":
    main()
