from __future__ import annotations

import difflib
import json

from sqlalchemy import select
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment

from .config import ROOT, RUNTIME_ADAPTER_VERSION, model_signature
from .dataset import choose_tasks, splits
from .db import Conversation, Deployment, Job, TaskRun, Version, transaction
from .evaluation_validity import audit_protocol, require_scoreable
from .gate import run_gate
from .learning import learn_skills
from .native_harness import make_agent
from .queue import progress
from .simulation import simulate


def version_dict(row):
    return {
        "id": row.id,
        "parent_id": row.parent_id,
        "name": row.name,
        "origin": row.origin,
        "status": row.status,
        "harness_code": row.harness_code,
        "skillbook": row.skillbook,
        "metadata": row.metadata_json,
        "created_at": row.created_at.isoformat(),
    }


def get_version(version_id):
    with transaction() as session:
        row = session.get(Version, version_id)
        if not row:
            raise ValueError("版本不存在")
        if row.status in {"invalidated", "superseded"}:
            raise ValueError("该版本来自已作废的开发评测，请使用有效版本")
        return version_dict(row)


def source_runs(source_id, version_id):
    with transaction() as session:
        job = session.get(Job, source_id)
        if (
            not job
            or job.kind != "evaluate"
            or job.status != "succeeded"
            or job.payload.get("split") != "learn"
            or job.payload.get("version_id") != version_id
        ):
            raise ValueError("学习来源必须是父版本已完成的学习集评测")
        rows = session.scalars(
            select(TaskRun)
            .where(TaskRun.job_id == source_id, TaskRun.label == "evaluation")
            .order_by(TaskRun.id)
        ).all()
        if any(r.data.get("termination_reason") in {"timeout", "infrastructure_error"} for r in rows):
            raise ValueError("学习来源含截断或调用失败的轨迹，必须先恢复评测")
        for row in rows:
            require_scoreable(row.data)
        source_version = session.get(Version, version_id)
        if (
            source_version
            and source_version.metadata_json.get("runtime_kind") == "auto_harness_full"
        ):
            expected = model_signature()
            if job.payload.get("model_config") != expected or any(
                r.data.get("agent_model") != expected["agent"]
                or r.data.get("user_model") != expected["user"]
                or r.data.get("judge_model") != expected["judge"]
                or (expected["simulator"]["sha256"] is not None
                    and r.data.get("simulator") != expected["simulator"])
                for r in rows
            ):
                raise ValueError("学习轨迹与当前客服、模拟器或评判模型不一致")
        if not rows or any(
            r.task_id not in splits()["learn"]
            or r.data.get("adapter_version") not in {2, 3, 4, 5, RUNTIME_ADAPTER_VERSION}
            for r in rows
        ):
            raise ValueError("学习轨迹必须来自学习集，且通过当前场景适配版本校验")
        return [
            {
                "id": r.id,
                "job_id": r.job_id,
                "task_id": r.task_id,
                "reward": r.reward,
                "data": r.data,
            }
            for r in rows
        ]


def evaluate_set(job, token, version, mode, ids, label, trials=1, start=0, span=0.9):
    results = {}
    for i, task_id in enumerate(ids):
        scores = []
        for trial in range(trials):
            with transaction() as session:
                previous = session.scalar(
                    select(TaskRun).where(
                        TaskRun.job_id == job["id"],
                        TaskRun.task_id == task_id,
                        TaskRun.label == label,
                        TaskRun.trial == trial,
                    )
                )
                score = previous.reward if previous else None
                if previous:
                    require_scoreable(previous.data)
            if score is None:

                def on_step(step):
                    progress(
                        job["id"],
                        token,
                        start + span * (i / len(ids)),
                        f"{label} · 任务 {task_id} · 第 {step} 步",
                    )

                score, data = simulate(task_id, version, mode, trial, on_step)
                data["evaluation_validity"] = audit_protocol(data)
                with transaction() as session:
                    live = session.get(Job, job["id"], with_for_update=True)
                    if live.status != "running" or live.lease_token != token:
                        raise RuntimeError("执行租约失效")
                    session.add(
                        TaskRun(
                            job_id=job["id"],
                            task_id=task_id,
                            trial=trial,
                            label=label,
                            version_id=version["id"],
                            reward=score,
                            data=data,
                        )
                    )
                require_scoreable(data)
            scores.append(score)
        results[task_id] = sum(scores) / trials
        progress(
            job["id"],
            token,
            start + span * ((i + 1) / len(ids)),
            f"{label} · 完成 {i + 1}/{len(ids)}",
        )
    return results


def add_candidate(job, token, parent, code, book, metadata, origin):
    metadata = {
        **metadata,
        "model_config": model_signature(),
        "code_diff": "".join(
            difflib.unified_diff(
                parent["harness_code"].splitlines(keepends=True),
                code.splitlines(keepends=True),
                fromfile=parent["id"],
                tofile=job["id"],
            )
        ),
    }
    if parent.get("metadata", {}).get("runtime_kind") == "auto_harness_full":
        metadata["runtime_kind"] = "auto_harness_full"
    with transaction() as session:
        live = session.get(Job, job["id"], with_for_update=True)
        if live.status != "running" or live.lease_token != token:
            raise RuntimeError("执行租约失效")
        # Idempotent across worker retries, including a crash before job completion.
        existing = session.get(Version, job["id"])
        if existing:
            return {"version_id": existing.id}
        session.add(
            Version(
                id=job["id"],
                parent_id=parent["id"],
                name=("经验学习" if origin == "ace" else "策略进化") + " " + job["id"][:6],
                origin=origin,
                harness_code=code,
                skillbook=book,
                metadata_json=metadata,
            )
        )
    return {"version_id": job["id"]}


def run_chat(job, token):
    payload = job["payload"]
    with transaction() as session:
        conv = session.get(Conversation, payload["conversation_id"])
        if not conv or conv.active_job != job["id"]:
            raise ValueError("会话任务已失效")
        revision, db_data, history, version_id = (
            conv.revision,
            conv.state,
            conv.messages,
            conv.version_id,
        )
    env = get_environment(db=RetailDB.model_validate(db_data))
    agent = make_agent(env, get_version(version_id))
    messages = []
    for message in history:
        cls = {"user": UserMessage, "assistant": AssistantMessage, "tool": ToolMessage}[
            message["role"]
        ]
        messages.append(cls.model_validate(message))
    state = agent.get_init_state(messages)
    incoming = UserMessage(role="user", content=payload["message"])
    for step in range(16):
        progress(job["id"], token, min(0.9, step / 16), f"客服处理中 · 第 {step + 1} 步")
        response, state = agent.generate_next_message(incoming, state)
        if not response.tool_calls:
            break
        tool_messages = []
        for call in response.tool_calls:
            try:
                value = env.make_tool_call(call.name, **call.arguments)
                content = json.dumps(
                    value.model_dump(mode="json") if hasattr(value, "model_dump") else value,
                    ensure_ascii=False,
                )
                error = False
            except Exception as exc:
                content, error = str(exc), True
            tool_messages.append(
                ToolMessage(role="tool", id=call.id, name=call.name, content=content, error=error)
            )
        incoming = MultiToolMessage(role="tool", tool_messages=tool_messages)
    else:
        raise ValueError("客服执行步骤超出限制")
    result = {"reply": response.content or "", "conversation_id": payload["conversation_id"]}
    # Business mutations and completion share a single commit, fenced by lease+revision.
    with transaction() as session:
        live = session.get(Job, job["id"], with_for_update=True)
        conv = session.get(Conversation, payload["conversation_id"], with_for_update=True)
        if (
            live.status != "running"
            or live.lease_token != token
            or conv.active_job != job["id"]
            or conv.revision != revision
        ):
            raise RuntimeError("会话已更新或任务已取消")
        conv.messages = [m.model_dump(mode="json") for m in state.messages]
        conv.state = env.tools.db.model_dump(mode="json")
        conv.revision += 1
        conv.active_job = None
        live.status, live.result, live.progress, live.stage = "succeeded", result, 1, "完成"
        live.lease_token, live.lease_until = None, None
    return result


def dispatch(job, token):
    p = job["payload"]
    if p.get("model_config") and p["model_config"] != model_signature():
        raise ValueError("模型配置已变化，请用当前配置重新提交任务，避免混合实验结果")
    if job["kind"] == "chat":
        return run_chat(job, token)
    version = get_version(p["version_id"])
    if job["kind"] == "rsi":
        from .full_rsi import run_full_rsi

        return run_full_rsi(job, token)
    if job["kind"] == "evaluate":
        ids = choose_tasks(p["split"], p.get("task_ids"), p.get("limit", 1))
        scores = evaluate_set(
            job, token, version, p.get("mode", "combined"), ids, "evaluation", p.get("trials", 1)
        )
        return {
            "scores": scores,
            "mean_reward": sum(scores.values()) / len(scores),
            "split": p["split"],
            "mode": p.get("mode", "combined"),
            "sample_count": len(scores),
            "trials": p.get("trials", 1),
            "metric": "official tau2 ALL_WITH_NL_ASSERTIONS",
            "model_config": model_signature(),
        }
    if job["kind"] in {"learn", "evolve"}:
        runs = source_runs(p["source_job_id"], version["id"])
        with transaction() as session:
            if session.get(Version, job["id"]):
                return {"version_id": job["id"]}

        def tick(v, s):
            return progress(job["id"], token, v * 0.9, s)

        if job["kind"] == "learn":
            with transaction() as session:
                checkpoint = (session.get(Job, job["id"]).result or {}).get("ace_checkpoint")

            def save_checkpoint(value):
                with transaction() as session:
                    owner = session.get(Job, job["id"], with_for_update=True)
                    if owner.status != "running" or owner.lease_token != token:
                        raise RuntimeError("ACE worker no longer owns the job lease")
                    owner.result = {"ace_checkpoint": value}

            book, audit = learn_skills(
                version, runs, tick, checkpoint=checkpoint, save_checkpoint=save_checkpoint
            )
            code, explanation, origin = version["harness_code"], "", "ace"
        else:
            raise ValueError("旧受限优化接口已停用；请运行完整 RSI 循环")
        return add_candidate(
            job,
            token,
            version,
            code,
            book,
            {
                "source_job_id": p["source_job_id"],
                "training_task_ids": [r["task_id"] for r in runs],
                "audit": audit,
                "learning_pipeline": "ace-recursive-v1",
                "explanation": explanation,
            },
            origin,
        )
    if job["kind"] == "gate":
        if not version["parent_id"]:
            raise ValueError("原始版本不需要候选检查")
        with transaction() as session:
            live = session.get(Job, job["id"], with_for_update=True)
            if "incumbent_version_id" not in p:
                p = {**p, "incumbent_version_id": session.get(Deployment, 1).version_id}
                live.payload = p
        parent = get_version(p["incumbent_version_id"])
        train_ids = choose_tasks("learn", p.get("train_ids"), p.get("limit", 1))
        val_ids = choose_tasks("validation", p.get("validation_ids"), p.get("limit", 1))
        groups = [
            (parent, train_ids, "parent_train"),
            (version, train_ids, "candidate_train"),
            (parent, val_ids, "parent_validation"),
            (version, val_ids, "candidate_validation"),
        ]
        results = []
        for i, (v, ids, label) in enumerate(groups):
            results.append(
                evaluate_set(
                    job, token, v, "combined", ids, label, p.get("trials", 1), i * 0.23, 0.23
                )
            )
        report = run_gate(ROOT / "artifacts" / job["id"], *results)
        report.update(
            train_ids=train_ids,
            validation_ids=val_ids,
            trials=p.get("trials", 1),
            version_id=version["id"],
            incumbent_version_id=parent["id"],
            model_config=model_signature(),
        )
        with transaction() as session:
            live = session.get(Job, job["id"], with_for_update=True)
            candidate = session.get(Version, version["id"], with_for_update=True)
            if live.status != "running" or live.lease_token != token:
                raise RuntimeError("执行租约失效")
            if candidate.status == "validated":
                return candidate.metadata_json["gate"]
            candidate.status = "validated" if report["passed"] else "rejected"
            candidate.metadata_json = {**candidate.metadata_json, "gate": report}
        return report
    raise ValueError("未知任务类型")
