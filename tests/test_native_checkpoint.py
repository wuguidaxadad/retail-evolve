"""Exercise bridge persistence and replay with only the benchmark network replaced."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from tau2.data_model.simulation import RewardInfo, SimulationRun, TextRunConfig

from app import db, native_bridge
from app.config import ROOT
from app.dataset import raw_tasks


@pytest.mark.parametrize("role_failure", [False, True, "draft_failure"])
def test_interrupted_native_command_reuses_checkpoint_and_restores_database_rows(
    tmp_path, monkeypatch, role_failure
):
    import tau2.run as facade
    import tau2.runner.batch as batch
    import tau2.runner.helpers as helpers
    from tau2.data_model.tasks import Task
    from tau2.evaluator import evaluator_nl_assertions as nl

    engine = create_engine("sqlite://")
    db.Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "Session", sessionmaker(engine, expire_on_commit=False))
    repo, private = tmp_path / "repo", tmp_path / "private"
    (repo / "agent").mkdir(parents=True)
    private.mkdir()
    (repo / "agent/agent.py").write_text(
        (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    )
    monkeypatch.setenv("RETAIL_NATIVE_PARENT_JOB", "parent")
    monkeypatch.setenv("RETAIL_NATIVE_LEASE_TOKEN", "lease")
    monkeypatch.setenv("RETAIL_NATIVE_REPO", str(repo))
    monkeypatch.setenv("RETAIL_NATIVE_PRIVATE", str(private))
    monkeypatch.setenv("RETAIL_NATIVE_BASELINE_VERSION", "baseline")
    monkeypatch.setattr(sys, "argv", ["benchmark.py"])
    monkeypatch.setattr(native_bridge, "_phase", None)
    # Register originals with monkeypatch so install() changes are restored too.
    original_init = TextRunConfig.__init__
    monkeypatch.setattr(TextRunConfig, "__init__", original_init)
    monkeypatch.setattr(batch, "run_single_task", batch.run_single_task)
    monkeypatch.setattr(nl, "DEFAULT_LLM_NL_ASSERTIONS", nl.DEFAULT_LLM_NL_ASSERTIONS)
    monkeypatch.setattr(nl, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS)
    for key in ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL"]:
        monkeypatch.setenv(key, "test")
    tasks = [
        Task.model_validate(t).model_copy(update={"evaluation_criteria": None})
        for t in raw_tasks()[:2]
    ]
    monkeypatch.setattr(helpers, "get_tasks", lambda **kwargs: tasks)
    with db.transaction() as session:
        session.add(
            db.Job(
                id="parent",
                kind="rsi",
                status="running",
                lease_token="lease",
                payload={},
                idempotency_key="parent",
            )
        )
    executed, locations = [], []
    if role_failure == "draft_failure":
        from app import frozen_simulator
        class DraftFailure(ValueError):
            pass
        monkeypatch.setattr(frozen_simulator, "is_simulator_error", lambda exc: isinstance(exc, DraftFailure))
        def failed_draft(*args, **kwargs):
            raise DraftFailure("exhausted three drafts")
        monkeypatch.setattr(batch, "run_single_task", failed_draft)

    def upstream(config):
        path = Path(config.save_to) / "results.json"
        locations.append(str(path))
        assert config.timeout is None
        assert config.max_steps == 200
        assert config.auto_resume
        simulations = (
            [SimulationRun.model_validate(x) for x in json.loads(path.read_text())]
            if path.exists()
            else []
        )
        done = {s.task_id for s in simulations}
        # First attempt loses the connection after one task. Second resumes.
        for task in tasks[:1] if not locations[:-1] else tasks:
            if task.id not in done:
                executed.append(task.id)
                if role_failure == "draft_failure":
                    simulations.append(batch.run_single_task(config, task))
                    continue
                simulations.append(
                    SimulationRun(
                        id=task.id,
                        task_id=task.id,
                        start_time="t",
                        end_time="t",
                        duration=1,
                        termination_reason="agent_stop",
                        reward_info=RewardInfo(reward=1),
                        messages=(
                            [
                                {
                                    "role": "user",
                                    "content": "I cannot override my core identity or role-play as a customer.",
                                }
                            ]
                            if role_failure
                            else []
                        ),
                    )
                )
        path.write_text(json.dumps([s.model_dump(mode="json") for s in simulations]))
        return SimpleNamespace(simulations=simulations)

    monkeypatch.setattr(facade, "run_domain", upstream)
    native_bridge.install()
    with pytest.raises(ValueError, match="omitted tasks"):
        facade.run_domain(
            TextRunConfig(domain="retail", task_split_name="train", task_ids=[t.id for t in tasks])
        )
    # A fresh command resets command-local phase numbering, like a new process.
    monkeypatch.setattr(facade, "run_domain", upstream)
    monkeypatch.setattr(TextRunConfig, "__init__", original_init)
    native_bridge.install()
    config = TextRunConfig(domain="retail", task_split_name="train", task_ids=[t.id for t in tasks])
    if role_failure:
        from app.evaluation_validity import EvaluationValidityError

        with pytest.raises(EvaluationValidityError, match="Invalid evaluation protocol"):
            facade.run_domain(config)
        with db.transaction() as session:
            assert not session.scalar(select(db.Job).where(db.Job.status == "succeeded"))
            rows = session.scalars(select(db.TaskRun)).all()
            assert rows and all(r.reward == (0 if role_failure == "draft_failure" else 1) for r in rows)
            assert all(r.label == "invalid_observation" for r in rows)
        assert not (repo / "workspace/traces/latest").exists()
        assert any(
            json.loads(p.read_text()).get("invalid_observations")
            for p in (private / "phases").glob("*.json")
        )
        engine.dispose()
        return
    result = facade.run_domain(config)
    assert locations[0] == locations[1]
    assert executed == [t.id for t in tasks]
    assert len(result.simulations) == 2
    with db.transaction() as session:
        complete = session.scalar(
            select(db.Job).where(db.Job.kind == "evaluate", db.Job.status == "succeeded")
        )
        rows = session.scalars(select(db.TaskRun).where(db.TaskRun.job_id == complete.id)).all()
        assert len(rows) == 2
        assert complete.result["mean_reward"] == 1
    engine.dispose()
