"""Offline wiring proof, not a benchmark improvement experiment."""

import hashlib

import pytest
from ace import Skillbook
from tau2.data_model.message import UserMessage
from tau2.data_model.tasks import Task
from tau2.user.user_simulator import UserSimulator

from app.config import ROOT
from app.dataset import raw_tasks
from app.simulation import simulate_native


@pytest.mark.parametrize("pipeline", ["comparison", "rsi_snapshot"])
def test_executable_edit_changes_official_evaluation_with_identical_prompt(monkeypatch, pipeline):
    def user_reply(self, message, state):
        return UserMessage(
            role="user", content=("Check my order." if message.content != "Done." else "###STOP###")
        ), state

    monkeypatch.setattr(UserSimulator, "generate_next_message", user_reply)
    # No model calls are allowed: the candidate itself supplies deterministic actions.
    import litellm

    monkeypatch.setattr(litellm, "completion", lambda *a, **k: pytest.fail("network call"))
    task_data = raw_tasks()[0].copy()
    task_data["evaluation_criteria"] = {
        "actions": [
            {
                "action_id": "audit",
                "name": "get_order_details",
                "arguments": {"order_id": "#W2378156"},
            }
        ],
        "reward_basis": ["ACTION"],
        "nl_assertions": [],
        "communicate_info": [],
    }
    task = Task.model_validate(task_data)
    original = (ROOT / "vendor/auto-harness/agent/templates/tau_bench.py").read_text()
    suffix = """
from tau2.data_model.message import ToolCall
class HarnessAgent(HarnessAgent):
    def generate_next_message(self, message, state):
        if DO_ACTION and not getattr(self, "audit_called", False):
            self.audit_called = True
            state.messages.append(message)
            return AssistantMessage(role="assistant", tool_calls=[ToolCall(
                id="audit", name="get_order_details", arguments={"order_id": "#W2378156"}
            )]), state
        return AssistantMessage(role="assistant", content="Done."), state
"""
    observations = []
    for enabled in (False, True, False):
        code = original + "\nDO_ACTION = " + repr(enabled) + "\n" + suffix
        # Deliberately reuse the id: source edits must never hit a stale factory.
        version = {"id": "same-id", "harness_code": code, "skillbook": Skillbook().to_dict()}
        if pipeline == "comparison":
            reward, data = simulate_native(task, version, "harness", 0)
            assert data["executed_source_hash"] == hashlib.sha256(code.encode()).hexdigest()
        else:
            from tau2.data_model.simulation import TextRunConfig
            from tau2.evaluator.evaluator import EvaluationType
            from tau2.run import run_single_task

            from app.native_harness import bind_source_snapshot

            result = run_single_task(
                TextRunConfig(domain="retail", agent=bind_source_snapshot(code)),
                task,
                seed=300,
                evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
            )
            data = result.model_dump(mode="json")
            reward = result.reward_info.reward
        calls = [c for m in data["messages"] for c in (m.get("tool_calls") or [])]
        assert any(c["name"] == "get_order_details" for c in calls) == enabled
        assert data["reward_info"]["action_checks"][0]["action_match"] == enabled
        observations.append(reward)
    assert observations == [0.0, 1.0, 0.0]
