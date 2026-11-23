from __future__ import annotations

import json
import time

from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.evaluator import evaluator_nl_assertions as nl_evaluator
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator

from .agent import llm_args
from .config import RUNTIME_ADAPTER_VERSION, agent_model_route, settings
from .dataset import raw_tasks, task_summary
from .native_harness import make_agent


class ObservableOrchestrator(Orchestrator):
    def __init__(self, *args, on_step=None, **kwargs):
        self.on_step = on_step
        self.observed_steps = 0
        super().__init__(*args, **kwargs)

    def step(self):
        self.observed_steps += 1
        if self.on_step:
            self.on_step(self.observed_steps)
        return super().step()


def simulation_user(task):
    return UserSimulator(
        llm="openai/" + settings.user_model,
        llm_args=llm_args(settings.user_model),
        instructions=str(task.user_scenario),
    )


def simulate(task_id, version, mode, trial=0, on_step=None):
    tasks = {str(t.id): t for t in get_tasks(None)}
    task = tasks[str(task_id)]
    if version.get("metadata", {}).get("runtime_kind") == "auto_harness_full":
        return simulate_native(task, version, mode, trial)
    env = get_environment()
    agent = make_agent(env, version, mode)
    user = simulation_user(task)
    start = time.monotonic()
    orchestrator = ObservableOrchestrator(
        domain="retail",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=agent.program["MAX_STEPS"],
        max_errors=5,
        seed=300 + trial,
        validate_communication=True,
        timeout=None,
        on_step=on_step,
    )
    simulation = orchestrator.run()
    # Keep the official evaluator and task reward basis, including NL assertions.
    # Upstream's default judge is GPT-4.1; route it to the user's compatible service.
    nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS = "openai/" + settings.judge_model
    nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
        **llm_args(),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    reward = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
        solo_mode=False,
        domain="retail",
    )
    expected = task.evaluation_criteria.nl_assertions or []
    if expected and reward.nl_assertions is not None:
        judged = [r.nl_assertion for r in reward.nl_assertions]
        if sorted(judged) != sorted(expected):
            raise ValueError("评审模型未完整覆盖任务的自然语言断言，不能报告成功")
    summary = task_summary(next(t for t in raw_tasks() if str(t["id"]) == str(task_id)))
    data = simulation.model_dump(mode="json")
    data.update(
        reward_info=reward.model_dump(mode="json"),
        task_summary=summary,
        elapsed_seconds=round(time.monotonic() - start, 2),
        injected_skill_ids=agent.injected_skill_ids,
        mode=mode,
        agent_model=settings.agent_model,
        user_model=settings.user_model,
        evaluator="tau2 pinned ALL_WITH_NL_ASSERTIONS",
        judge_model=settings.judge_model,
        adapter_version=RUNTIME_ADAPTER_VERSION,
    )
    return float(reward.reward), data


def simulate_native(task, version, mode, trial):
    """Use the same native tau runner used by upstream auto-harness."""
    import uuid

    # Each evaluation binds its own immutable source/skill snapshot.
    from copy import deepcopy

    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.run import run_single_task

    from .frozen_simulator import register_simulator, simulator_scope, simulator_signature
    from .native_harness import make_native_agent

    version = deepcopy(version)
    name = "retail_native_" + uuid.uuid4().hex
    executed_agents = []

    def factory(tools, domain_policy, **kwargs):
        agent = make_native_agent(None, version, mode, tools=tools, domain_policy=domain_policy)
        executed_agents.append(agent)
        return agent

    registry.register_agent_factory(factory, name)
    nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS = "openai/" + settings.judge_model
    nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
        **llm_args(settings.judge_model),
        "response_format": {"type": "json_object"},
    }
    config = TextRunConfig(
        domain="retail",
        user=register_simulator(),
        agent=name,
        llm_agent=agent_model_route(),
        llm_user="openai/" + settings.user_model,
        llm_args_agent=llm_args(),
        llm_args_user=llm_args(settings.user_model),
        num_trials=1,
        seed=300 + trial,
        timeout=None,
        max_retries=2,
        log_level="ERROR",
    )
    start = time.monotonic()
    try:
        with simulator_scope(name + "/" + str(task.id)):
            result = run_single_task(
                config, task, seed=300 + trial, evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS
            )
    finally:
        registry._agent_factories.pop(name, None)
    if not executed_agents:
        raise RuntimeError("Native evaluator did not instantiate the requested agent")
    if result.termination_reason.value in {"timeout", "infrastructure_error"}:
        raise ConnectionError("Incomplete native observation: " + result.termination_reason.value)
    reward = result.reward_info
    data = result.model_dump(mode="json")
    data.update(
        task_summary=task_summary(next(t for t in raw_tasks() if str(t["id"]) == str(task.id))),
        elapsed_seconds=round(time.monotonic() - start, 2),
        native_run=True,
        simulator=simulator_signature(),
        injected_skill_ids=executed_agents[-1].injected_skill_ids,
        executed_source_hash=executed_agents[-1].executed_source_hash,
        mode=mode,
        agent_model=settings.agent_model,
        user_model=settings.user_model,
        judge_model=settings.judge_model,
        adapter_version=RUNTIME_ADAPTER_VERSION,
        evaluator="official tau2 native run_single_task ALL_WITH_NL_ASSERTIONS",
    )
    return float(reward.reward), data


def learning_trace(run):
    data = run["data"]
    from .evaluation_validity import require_scoreable

    require_scoreable(data)
    messages = data.get("messages", [])
    answers = [
        m.get("content", "") for m in messages if m.get("role") == "assistant" and m.get("content")
    ]
    return {
        "question": data["task_summary"],
        "reasoning": json.dumps(messages, ensure_ascii=False),
        "answer": answers[-1] if answers else "",
        "feedback": json.dumps(data["reward_info"], ensure_ascii=False),
        "trace_id": str(run["id"]),
        "source": {"task_id": run["task_id"], "job_id": run["job_id"], "split": "learn"},
    }
